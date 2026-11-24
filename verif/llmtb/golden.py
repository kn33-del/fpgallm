"""Bit-exact Python reference models, transcribed from the RTL numerics.

Every function here is the *spec* the RTL is checked against. Where the RTL comments
reference a not-yet-written "golden model", these functions are derived line-by-line
from the RTL arithmetic (widths, rounding, saturation) so mismatches are real bugs
in control/datapath plumbing rather than format disagreements.
"""
import math

import numpy as np

from . import cfg

I8 = (-128, 127)
I16 = (-32768, 32767)


def wrap(x, bits):
    m = 1 << bits
    return ((np.asarray(x, dtype=object) + (m >> 1)) % m) - (m >> 1)


def sat(x, lo_hi):
    lo, hi = lo_hi
    return np.clip(x, lo, hi)


# =========================================================================================
# T1: C = requant(A x B)
# =========================================================================================
def requant(acc, scales, fmt, shift=16):
    acc = np.asarray(acc, dtype=np.int64)
    acc = ((acc + (1 << 31)) % (1 << 32)) - (1 << 31)          # ACC_W = 32 wrap
    if fmt == cfg.OUT_ACC32:
        return acc
    p = acc * np.asarray(scales, dtype=np.int64)               # PW = 64, fits int64
    mag = np.abs(p)
    if shift > 0:
        mag = (mag + (1 << (shift - 1))) >> shift              # round half away from zero
    r = np.where(p < 0, -mag, mag)
    return sat(r, I8 if fmt == cfg.OUT_I8 else I16)


def _acc(A, B, chunk=2048):
    """Exact integer A@B, chunked over N so huge int8 weights never become one int64 matrix.
    float64 is exact here (|acc| < 2^53) and uses BLAS."""
    A = np.asarray(A, dtype=np.float64)
    out = np.empty((A.shape[0], B.shape[1]), dtype=np.int64)
    for j in range(0, B.shape[1], chunk):
        out[:, j:j + chunk] = np.rint(A @ np.asarray(B[:, j:j + chunk], dtype=np.float64)).astype(np.int64)
    return out


def gemm(A, B, scales, fmt, shift=16):
    acc = _acc(A, B)
    return requant(acc, np.broadcast_to(np.asarray(scales, dtype=np.int64), acc.shape), fmt, shift)


def out_elem(fmt):
    return {cfg.OUT_I8: (1, np.int8), cfg.OUT_VEC: (2, np.int16), cfg.OUT_ACC32: (4, np.int32)}[fmt]


def mm_tile_geometry(c):
    kt, nt = cfg.ceil_div(c["k"], cfg.ARR_R), cfg.ceil_div(c["n"], cfg.ARR_C)
    eb, _ = out_elem(c["out_fmt"])
    tile_out_bytes = cfg.ceil_div(cfg.ARR_C * eb * 8, cfg.BEAT_W) * cfg.BEAT_BYTES
    return kt, nt, tile_out_bytes


def mm_load_operands(mem, c):
    """Read A, B, scales of an mm_cmd_t (dict) from a logical RegionMem (T1 tile layout)."""
    m, k, n = c["m"], c["k"], c["n"]
    kt, nt, _ = mm_tile_geometry(c)
    arg, aoff = cfg.addr_split(c["a_base"])
    A = np.zeros((m, kt * 32), dtype=np.int64)
    for r in range(m):
        A[r] = mem[arg].read_np(aoff + r * kt * 32, kt * 32, np.int8)
    # B addresses are per-PC bytes (llm_pkg contract). In the unit benches' logical memory a
    # tile is stored contiguously, so logical byte = per-PC byte * N_HBM.
    brg, boff = cfg.addr_split(c["b_base"])
    boff = (boff & cfg.PC_MASK) * cfg.N_HBM
    stride = (c["b_row_stride"] * cfg.N_HBM) or kt * cfg.TILE_BYTES
    B = np.zeros((kt * 32, nt * 32), dtype=np.int8)
    for j in range(nt):                       # one strip read per N tile: kt tiles are contiguous
        strip = mem[brg].read_np(boff + j * stride, kt * 1024, np.int8).reshape(kt, 32, 32)
        B[:, j * 32:(j + 1) * 32] = strip.reshape(kt * 32, 32)
    A[:, k:] = 0
    B[k:, :] = 0
    B[:, n:] = 0
    S = np.zeros(nt * 32, dtype=np.int64)
    if c["out_fmt"] != cfg.OUT_ACC32:
        srg, soff = cfg.addr_split(c["s_base"])
        for j in range(nt):
            S[j * 32:(j + 1) * 32] = mem[srg].read_np(soff + j * cfg.SCALE_TILE_BYTES, 32, np.int32)
    return A, B, S


def mm_execute(mem, c, shift=16, write=True):
    """Reference engine: compute and (optionally) write C exactly as T1 lays it out."""
    m = c["m"]
    kt, nt, tob = mm_tile_geometry(c)
    A, B, S = mm_load_operands(mem, c)
    C = gemm(A, B, S, c["out_fmt"], shift)
    if write:
        eb, dt = out_elem(c["out_fmt"])
        crg, coff = cfg.addr_split(c["c_base"])
        for r in range(m):
            for j in range(nt):
                blk = np.zeros(tob // eb, dtype=dt)
                blk[:32] = C[r, j * 32:(j + 1) * 32].astype(dt)
                mem[crg].write(coff + r * nt * tob + j * tob, blk.astype(np.dtype(dt).newbyteorder("<")).tobytes())
    return C


def pack_b_tiles(W, k_pad=None, n_pad=None):
    """Logical tile layout: tile (n_tile j, k_tile i) at (j*kt + i)*1024, row r = K index."""
    K, N = W.shape
    kt, nt = cfg.ceil_div(K, 32), cfg.ceil_div(N, 32)
    Wp = np.zeros((kt * 32, nt * 32), dtype=np.int8)
    Wp[:K, :N] = W
    out = bytearray()
    for j in range(nt):
        for i in range(kt):
            out += Wp[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32].astype(np.int8).tobytes()
    return bytes(out)


def pack_scales(S):
    nt = cfg.ceil_div(len(S), 32)
    Sp = np.zeros(nt * 32, dtype=np.int32)
    Sp[:len(S)] = S
    return Sp.astype("<i4").tobytes()


# =========================================================================================
# T2: RoPE + softmax (mirror rope_unit / softmax_unit)
# =========================================================================================
QKV_FRAC, SCORE_FRAC, COEFF_FRAC = 7, 8, 15


def round_i8_rope(v):
    shift = QKV_FRAC + COEFF_FRAC
    v = int(v)
    mag = abs(v)
    r = (mag + (1 << (shift - 1))) >> shift
    if v < 0:
        return -128 if r >= 128 else -r
    return min(r, 127)


def rope_head(x, cos, sin, pos, rotate):
    """x: int16[HEAD_DIM] (Q7), cos/sin: int16[HEAD_DIM/2] (Q15). Returns int8[HEAD_DIM]."""
    P = cfg.HEAD_DIM // 2
    out = np.zeros(cfg.HEAD_DIM, dtype=np.int64)
    for i in range(P):
        x0, x1 = int(x[i]), int(x[i + P])
        if pos == 0 or not rotate:
            y0, y1 = x0 << COEFF_FRAC, x1 << COEFF_FRAC
        else:
            c, s = int(cos[i]), int(sin[i])
            y0, y1 = x0 * c - x1 * s, x0 * s + x1 * c
        out[i], out[i + P] = round_i8_rope(y0), round_i8_rope(y1)
    return out


EXP_LUT = [65536, 51039, 39750, 30957, 24109, 18776, 14623, 11388,
           8869, 6907, 5380, 4190, 3263, 2541, 1979, 1541,
           1200, 935, 728, 567, 442, 344, 268, 209,
           162, 127, 99, 77, 60, 47, 36, 28,
           22, 17, 13, 10, 8, 6, 5, 4,
           3, 2, 2, 1, 1, 1, 1, 1] + [0] * 17


def exp_neg(x):
    step = SCORE_FRAC - 2
    if x >= (16 << SCORE_FRAC):
        return 0
    cell, frac = x >> step, x & ((1 << step) - 1)
    hi, lo = EXP_LUT[cell], EXP_LUT[cell + 1]
    corr = (hi - lo) * frac + (1 << (step - 1))
    return (hi - (corr >> step)) & ((1 << 17) - 1)


def softmax_i8(scores):
    s = [int(v) for v in scores]
    mx = max(s + [-32768])
    e = [exp_neg(mx - v) for v in s]
    tot = sum(e)
    return np.array([(127 * x + (tot >> 1)) // tot for x in e], dtype=np.int64)


# =========================================================================================
# T3: vector units (mirror rmsnorm / residual_add / silu / eltwise_mul / argmax)
# =========================================================================================
SIG_TABLE = [(32768, 8025), (40793, 7118), (47911, 5670), (53581, 4143), (57724, 2841),
             (60565, 1863), (62428, 1187), (63615, 742), (64357, 459), (64816, 281),
             (65097, 172), (65269, 105), (65374, 64), (65438, 38), (65476, 24), (65500, 14)]


def rmsnorm(x, gamma, qmult=1, qshift=0, eps=1, frac=8):
    x = [int(v) for v in x]
    n = len(x)
    sum_sq = sum(v * v for v in x)
    mean = sum_sq // n
    radicand = (mean + max(eps, 1)) << 16
    root = math.isqrt(radicand)
    gain = ((1 << (16 + frac + 8)) // root) & ((1 << (frac + 17)) - 1)
    out = []
    for v, g in zip(x, gamma):
        q = (v * int(g) * gain * qmult) >> (2 * frac + 16 + qshift)
        out.append(max(-128, min(127, q)))
    return np.array(out, dtype=np.int64)


def resadd(x, a):
    return sat(np.asarray(x, dtype=np.int64) + np.asarray(a, dtype=np.int64), I16)


def silu(x, frac=8):
    out = []
    for v in (int(t) for t in x):
        mag = abs(v)
        seg = mag >> (frac - 1)
        f = mag & ((1 << (frac - 1)) - 1)
        base, delta = (65536, 0) if seg >= 16 else SIG_TABLE[seg]
        pos = (base + ((delta * f) >> (frac - 1))) & 0x1FFFF
        sig = (65536 - pos) if v < 0 else pos
        y = (v * sig) >> 16
        y = ((y + 32768) % 65536) - 32768
        out.append(y)
    return np.array(out, dtype=np.int64)


def eltwise_mul(a, b, qmult=1, qshift=0, frac=8):
    q = (np.asarray(a, dtype=object) * np.asarray(b, dtype=object) * qmult) >> (2 * frac + qshift)
    return np.array([max(-128, min(127, int(v))) for v in q], dtype=np.int64)


def silu_mul_interleaved(gu, length, qmult=1, qshift=0):
    """gu: int16 gate/up array, interleaved per 32-wide n_tile ([g0..g31, u0..u31, g32..])."""
    idx = np.arange(length)
    g = np.asarray(gu)[(idx // 32) * 64 + idx % 32]
    u = np.asarray(gu)[(idx // 32) * 64 + 32 + idx % 32]
    return eltwise_mul(silu(g), u, qmult, qshift)


def argmax(x):
    return int(np.argmax(np.asarray(x, dtype=np.int64)))
