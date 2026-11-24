"""Full pipeline: llm_top (LLM_TINY dims) driven like the Vitis host.

Host flow per token: write instr_base/n_instr/token_pos/token_in over AXI-Lite, ap_start,
wait for interrupt/ap_done, read result_token and perf counters. HBM is a 32-port AXI model.
Scoreboard: result token vs bit-exact golden; final residual x and logits (peeked from the
on-chip act_buf); KV cache contents in HBM (llm_pkg layout) after every token.
"""
import random

import numpy as np
import pyuvm

from llmtb import cfg
from llmtb.bfm import AxiHbm, AxiLiteMaster, Comp, rv
from llmtb.mem import beat_to_bytes
from llmtb.model import ACT_X, SCR_LOGITS, TinyModel
from llmtb.uvm import BaseEnv, BaseTest

assert cfg.TINY, "test_top must run with LLM_TINY=1 (run.py sets it)"


class IrqMon(Comp):
    def __init__(self, dut):
        super().__init__("irq")
        self.d = dut
        self.rises = 0
        self.prev = 0

    def sample(self):
        v = rv(self.d.interrupt)
        if v and not self.prev:
            self.rises += 1
        self.prev = v


class Watchdog(Comp):
    """Every `every` cycles logs the sequencer + unit FSM states (finds hangs fast)."""
    PROBES = [("ctrl.st", "u_ctrl.st"), ("ctrl.pc", "u_ctrl.pc"), ("ctrl.op", "u_ctrl.cur.op"),
              ("attn", "u_attn.u_seq.state"), ("rope", "u_attn.u_rope.state"), ("smax", "u_attn.u_sm.state"),
              ("mlp", "u_mlp.u_seq.state"), ("vec", "u_mlp.u_vu.state"), ("t1", "u_mm.u_ctrl.state"),
              ("arb_busy", "u_mm_arb.busy"), ("rt_r_own", "u_mem.u_rt.r_own_v"), ("rt_w_own", "u_mem.u_rt.w_own_v"),
              ("kv", "u_mem.u_kv.st"), ("dma_busy", "u_mem.u_dma.busy")]

    def __init__(self, dut, log, every=20000):
        super().__init__("watchdog")
        self.d, self.log, self.every = dut, log, every

    def snapshot(self):
        out = []
        for name, path in self.PROBES:
            h = self.d
            try:
                for part in path.split("."):
                    h = getattr(h, part)
                out.append(f"{name}={rv(h)}")
            except Exception:
                out.append(f"{name}=?")
        return " ".join(out)

    def sample(self):
        if self.domain.cycle % self.every == 0:
            self.log(f"[watchdog cyc {self.domain.cycle}] {self.snapshot()}")


class TopEnv(BaseEnv):
    clk_name, rst_name = "ap_clk", "ap_rst_n"      # Vitis kernel port names
    hbm_kw = dict(lat=(20, 40), p_r=1.0)

    def make_bfms(self):
        d = self.dut
        self.hbm = self.domain.add(AxiHbm(d, rng=random.Random(self.seed), **self.hbm_kw))
        self.axil = self.domain.add(AxiLiteMaster(d))
        self.irq = self.domain.add(IrqMon(d))
        self.wd = self.domain.add(Watchdog(d, self.logger.info))


class TopStressEnv(TopEnv):
    hbm_kw = dict(lat=(40, 120), p_r=0.6, p_ar=0.5, p_aw=0.5, p_w=0.6, b_lat=(10, 40))


def peek_act(dut, rgn, off, nbytes, dtype):
    """Read the on-chip act_buf through the simulator hierarchy (debug/scoreboard only)."""
    act = dut.u_mem.u_act
    base = (off >> 5) + (4096 if rgn == cfg.RGN_SCRATCH else 0)
    raw = b"".join(beat_to_bytes(rv(act.mem[base + i]), 32) for i in range(cfg.ceil_div(nbytes, 32)))
    return np.frombuffer(raw[:nbytes], dtype).astype(np.int64)


class TopDecode(BaseTest):
    env_cls = TopEnv
    n_tokens = 3
    first_token = 7
    token_timeout = 400_000

    def build_phase(self):
        super().build_phase()
        self.errs = []

    def extra_checks(self):
        return self.errs

    def check_kv(self, model, pos):
        hbm = self.env.hbm.mem
        for l in range(cfg.N_LAYERS):
            for kvh in range(cfg.N_KV_HEADS):
                kb, vb = cfg.kv_pc_off(l, kvh, 0), cfg.kv_pc_off(l, kvh, 1)
                for p in range(pos + 1):
                    for d in range(cfg.HEAD_DIM):
                        k_hw = np.frombuffer(hbm.read(((d % 32) << 29) | (kb + ((p // 32) * 2 + d // 32) * 32 + p % 32), 1), np.int8)[0]
                        v_hw = np.frombuffer(hbm.read(((p % 32) << 29) | (vb + ((d // 32) * (cfg.MAX_SEQ // 32) + p // 32) * 32 + d % 32), 1), np.int8)[0]
                        if k_hw != model.Kc[l, kvh, d, p] or v_hw != model.Vc[l, kvh, p, d]:
                            self.errs.append(f"KV cache l={l} kvh={kvh} pos={p} dim={d}: K {k_hw}/{model.Kc[l, kvh, d, p]} "
                                             f"V {v_hw}/{model.Vc[l, kvh, p, d]}")
                            return

    async def body(self):
        env, ax, dut = self.env, self.env.axil, self.env.dut
        model = TinyModel(seed=env.seed)
        A = model.pack(env.hbm.mem, range(self.n_tokens))
        self.logger.info(f"packed model: {model.n_instr} instructions @ {A['prog']:#x}")
        await ax.write(0x10, A["prog"] & 0xFFFF_FFFF)
        await ax.write(0x14, A["prog"] >> 32)
        await ax.write(0x18, model.n_instr)
        await ax.write(0x04, 1)                     # GIE
        await ax.write(0x08, 1)                     # IER[0] = done
        token = self.first_token
        for pos in range(self.n_tokens):
            exp = model.step(token, pos)
            await ax.write(0x1C, pos)
            await ax.write(0x24, token)
            irq0 = env.irq.rises
            t0 = env.domain.cycle
            await ax.write(0x00, 1)                 # ap_start
            try:
                await env.domain.wait_until(lambda: env.irq.rises > irq0, timeout=self.token_timeout,
                                            what=f"ap_done token {pos}")
            except TimeoutError:
                self.errs.append(f"token {pos}: HANG, no ap_done in {self.token_timeout} cycles; "
                                 f"state: {env.wd.snapshot()}")
                return
            cyc = env.domain.cycle - t0
            await ax.write(0x0C, 1)                 # ack ISR
            st = await ax.read(0x00)
            got = await ax.read(0x20)
            perf = {n: await ax.read(a) for n, a in (("kernel_cycles", 0x40), ("ws_beats", 0x48), ("ws_active", 0x50),
                                                      ("ws_starve", 0x58), ("ws_stall", 0x60), ("axi_err", 0x68))}
            x_hw = peek_act(dut, cfg.RGN_ACT, ACT_X, cfg.D_MODEL * 2, "<i2")
            lg_hw = peek_act(dut, cfg.RGN_SCRATCH, SCR_LOGITS, cfg.VOCAB * 2, "<i2")
            if not st & 0x2:
                self.errs.append(f"token {pos}: ap_done bit not set (ap_ctrl={st:#x})")
            nbx = int(np.sum(x_hw != exp["x"]))
            nbl = int(np.sum(lg_hw != exp["logits"]))
            if nbx:
                i = int(np.nonzero(x_hw != exp["x"])[0][0])
                self.errs.append(f"token {pos}: final residual x {nbx}/{cfg.D_MODEL} mismatches (x[{i}] {x_hw[i]} vs {exp['x'][i]})")
            if nbl:
                i = int(np.nonzero(lg_hw != exp["logits"])[0][0])
                self.errs.append(f"token {pos}: logits {nbl}/{cfg.VOCAB} mismatches (logit[{i}] {lg_hw[i]} vs {exp['logits'][i]})")
            if got != exp["token"]:
                self.errs.append(f"token {pos}: result_token {got} != golden {exp['token']}")
            if perf["axi_err"]:
                self.errs.append(f"token {pos}: axi_err set")
            self.check_kv(model, pos)
            wbytes = perf["ws_beats"] * cfg.TILE_BYTES
            self.profile.add(f"token{pos}", in_token=token, out_token=got, golden_token=exp["token"], cycles=cyc,
                             us_at_300MHz=round(cyc / cfg.KERNEL_MHZ, 2), weight_tiles=perf["ws_beats"],
                             weight_GBps=round(wbytes / max(perf["kernel_cycles"], 1) * cfg.KERNEL_MHZ / 1e3, 2),
                             ws_busy_frac=round(perf["ws_active"] / max(perf["kernel_cycles"], 1), 3),
                             ws_starve=perf["ws_starve"], ws_stall=perf["ws_stall"],
                             logit_max=int(exp["logits"].max()), x_absmax=int(np.abs(exp["x"]).max()))
            self.logger.info(f"token pos={pos}: in={token} out={got} golden={exp['token']} cycles={cyc}")
            token = got
        self.expect_hbm_clean()

    def expect_hbm_clean(self):
        for e in self.env.hbm.errors:
            self.errs.append(e)


@pyuvm.test()
class TopDecode3Tokens(TopDecode):
    """3 autoregressive tokens (pos 0..2), every layer/unit/memory path, nominal HBM."""


@pyuvm.test()
class TopDecodeStress(TopDecode):
    """Same flow with a slow, bursty HBM (latency 40-120, 50-60% ready/valid)."""
    env_cls = TopStressEnv
    n_tokens = 2
    first_token = 200


@pyuvm.test()
class TopDecodeCrossTile(TopDecode):
    """34 tokens: KV cache crosses the 32-position tile boundary (ragged score/value tiles)."""
    n_tokens = 34
    first_token = 3
