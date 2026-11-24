"""Mirror of llm_pkg: parameters + bit-exact packing of the packed structs.

SV packed structs are MSB-first: the first field listed occupies the top bits.
Set env LLM_TINY=1 to mirror the `ifdef LLM_TINY` dims (must match the build define).
"""
import os

TINY = os.environ.get("LLM_TINY", "0") == "1"
MODEL = "tiny" if TINY else os.environ.get("LLM_MODEL", "llama32_1b")

if TINY:
    D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM = 128, 2, 1, 64
    D_FF, N_LAYERS, VOCAB, MAX_SEQ = 256, 2, 256, 64
elif MODEL == "tinyllama":          # mirrors `ifdef LLM_TINYLLAMA in llm_pkg
    D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM = 2048, 32, 4, 64
    D_FF, N_LAYERS, VOCAB, MAX_SEQ = 5632, 22, 32000, 2048
else:
    D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM = 2048, 32, 8, 64
    D_FF, N_LAYERS, VOCAB, MAX_SEQ = 8192, 16, 128256, 2048

DATA_W, ACC_W, VEC_W, SCALE_W = 8, 32, 16, 32
ARR_R = ARR_C = 32
KERNEL_MHZ, PORT_CEIL_GBPS, PLAN_HBM_GBPS = 300, 307, 280
AXI_DW, AXI_AW, AXI_IDW, N_HBM = 256, 34, 6, 32
T1_SHIFT = 28 if MODEL == "tinyllama" else 16   # mirrors llm_pkg::T1_REQUANT_SHIFT
M_AXI_AW = 64                         # kernel m_axi address width (base[pc] + off[28:0])
BEAT_W = AXI_DW
WSTREAM_W = N_HBM * AXI_DW
BEAT_BYTES = BEAT_W // 8                 # 32
TILE_BYTES = ARR_R * ARR_C               # 1024
SCALE_BEATS = (ARR_C * SCALE_W + BEAT_W - 1) // BEAT_W   # 4
SCALE_TILE_BYTES = SCALE_BEATS * BEAT_BYTES              # 128

RGN_HBM, RGN_ACT, RGN_SCRATCH, RGN_RSVD = 0, 1, 2, 3
OUT_I8, OUT_VEC, OUT_ACC32 = 0, 1, 2
V_RMSNORM, V_RESADD, V_SILU_MUL, V_ARGMAX = 0, 1, 2, 3
(OP_NOP, OP_END, OP_EMBED, OP_RMSNORM, OP_ATTN_BLOCK, OP_RESADD,
 OP_MLP_BLOCK, OP_GEMM, OP_ARGMAX) = range(9)

KV_PC_BASE = 0x1000_0000
KV_BLK_BYTES = 4096


PC_MASK = (1 << 29) - 1
TILE_PC_BYTES = TILE_BYTES // N_HBM      # 32: one tile = 32 B in each pseudo-channel


def ceil_div(a, b):
    return (a + b - 1) // b


def kv_pc_off(layer, kvh, is_v):
    idx = ((layer * N_KV_HEADS + kvh) * 2) + is_v
    return (KV_PC_BASE + (idx << 12)) & ((1 << 29) - 1)


class Struct:
    """Packed struct. fields = [(name, width) | (name, Struct)], MSB first."""

    def __init__(self, name, fields):
        self.name = name
        self.fields = []
        for fname, w in fields:
            self.fields.append((fname, w))
        self.width = sum(f.width if isinstance(f, Struct) else f for _, f in self.fields)

    def pack(self, **kw):
        v = 0
        for fname, f in self.fields:
            w = f.width if isinstance(f, Struct) else f
            x = kw.get(fname, 0)
            if isinstance(f, Struct) and isinstance(x, dict):
                x = f.pack(**x)
            v = (v << w) | (int(x) & ((1 << w) - 1))
        return v

    def unpack(self, v):
        out = {}
        shift = self.width
        for fname, f in self.fields:
            w = f.width if isinstance(f, Struct) else f
            shift -= w
            out[fname] = (v >> shift) & ((1 << w) - 1)
        return out


ADDR = Struct("addr_t", [("rgn", 2), ("off", AXI_AW)])
ADDR_W = ADDR.width


def addr(rgn, off):
    return ADDR.pack(rgn=rgn, off=off)


def addr_split(a):
    d = ADDR.unpack(a)
    return d["rgn"], d["off"]


MEM_REQ = Struct("mem_req_t", [("base", ADDR_W), ("row_beats", 16), ("n_rows", 16), ("row_stride", 32)])
MM_CMD = Struct("mm_cmd_t", [("a_base", ADDR_W), ("b_base", ADDR_W), ("s_base", ADDR_W), ("c_base", ADDR_W),
                             ("m", 16), ("k", 16), ("n", 16), ("b_is_kv", 1), ("b_row_stride", 32),
                             ("out_fmt", 2), ("tag", 4)])
ATTN_CMD = Struct("attn_cmd_t", [("layer", 8), ("pos", 16), ("x_base", ADDR_W), ("w_base", ADDR_W),
                                 ("s_base", ADDR_W), ("out_base", ADDR_W)])
MLP_CMD = Struct("mlp_cmd_t", [("layer", 8), ("x_base", ADDR_W), ("w_base", ADDR_W), ("s_base", ADDR_W),
                               ("out_base", ADDR_W)])
VEC_CMD = Struct("vec_cmd_t", [("op", 3), ("src0", ADDR_W), ("src1", ADDR_W), ("dst", ADDR_W), ("len", 24)])
DMA_CMD = Struct("dma_cmd_t", [("src", ADDR_W), ("dst", ADDR_W), ("len_beats", 24)])
_INSTR_USED = 8 + 8 + 48 + 4 * ADDR_W
INSTR = Struct("instr_t", [("op", 8), ("layer", 8), ("m", 16), ("k", 16), ("n", 16), ("src0", ADDR_W),
                           ("src1", ADDR_W), ("src2", ADDR_W), ("dst", ADDR_W), ("rsvd", AXI_DW - _INSTR_USED)])
KV_TAG = Struct("kv_tag_t", [("layer", 8), ("kvh", 4), ("is_v", 1), ("pos", 16)])
PERF = Struct("perf_t", [("ws_beats", 48), ("ws_active", 48), ("ws_starve", 48), ("ws_stall", 48),
                         ("axi_err", 1)])

assert INSTR.width == AXI_DW, "instr_t must be exactly one beat"

# ---- T2 layout constants (mirror t2_attn_pkg) -------------------------------------------
HEAD_BYTES = HEAD_DIM
HEAD_BEATS = HEAD_BYTES // BEAT_BYTES
HEAD_VEC_BYTES = HEAD_DIM * 2
QKV_ELEMENTS = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM
QKV_VEC_BYTES = QKV_ELEMENTS * 2
QKV_BYTES = QKV_ELEMENTS
SCORE_BYTES = ceil_div(MAX_SEQ, ARR_C) * ARR_C * 2
PROB_BYTES = ceil_div(MAX_SEQ, ARR_R) * BEAT_BYTES
COEFF_BYTES = HEAD_DIM * 2
WO_OFFSET = ceil_div(D_MODEL, ARR_R) * ceil_div(QKV_ELEMENTS, ARR_C) * TILE_PC_BYTES   # per-PC
WO_SCALE_OFFSET = ceil_div(QKV_ELEMENTS, ARR_C) * SCALE_TILE_BYTES
V_N_TILE_STRIDE = ceil_div(MAX_SEQ, ARR_R) * TILE_PC_BYTES                              # per-PC
K_SCALE_BYTES = ceil_div(MAX_SEQ, ARR_C) * SCALE_TILE_BYTES
V_SCALE_BYTES = ceil_div(HEAD_DIM, ARR_C) * SCALE_TILE_BYTES
KV_SCALE_BYTES = K_SCALE_BYTES + V_SCALE_BYTES
T2_SCRATCH_BASE = 0
T2_KV_SCALE_BASE = 0x084000000
T2_ROPE_BASE = 0x085000000
T2_QKV_VEC = T2_SCRATCH_BASE
T2_QKV_I8 = T2_QKV_VEC + QKV_VEC_BYTES
T2_SCORES = T2_QKV_I8 + QKV_BYTES
T2_PROBS = T2_SCORES + SCORE_BYTES
T2_HEADS = T2_PROBS + PROB_BYTES

# ---- T3 layout constants (mirror mlp_seq) --------------------------------------------------
T3_GU_BYTES = 2 * D_FF * 2
T3_WD_OFFSET = D_MODEL * 2 * D_FF // N_HBM       # per-PC
T3_SD_OFFSET = 2 * D_FF * 4
T3_UP_OFFSET = ARR_C * 2
