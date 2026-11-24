"""Sparse byte-addressable memories (one per region, plus a flat HBM for the AXI model)."""
import numpy as np

from . import cfg

PAGE = 1 << 16


class Sparse:
    def __init__(self, fill=0):
        self.pages = {}
        self.fill = fill

    def _page(self, p, create):
        pg = self.pages.get(p)
        if pg is None and create:
            pg = bytearray([self.fill]) * PAGE
            self.pages[p] = pg
        return pg

    def read(self, off, n):
        out = bytearray(n)
        i = 0
        while i < n:
            p, o = divmod(off + i, PAGE)
            k = min(n - i, PAGE - o)
            pg = self._page(p, False)
            if pg is not None:
                out[i:i + k] = pg[o:o + k]
            elif self.fill:
                out[i:i + k] = bytes([self.fill]) * k
            i += k
        return bytes(out)

    def write(self, off, data, strb=None):
        data = bytes(data)
        if strb is not None:
            for j, b in enumerate(data):
                if (strb >> j) & 1:
                    p, o = divmod(off + j, PAGE)
                    self._page(p, True)[o] = b
            return
        i, n = 0, len(data)
        while i < n:
            p, o = divmod(off + i, PAGE)
            k = min(n - i, PAGE - o)
            self._page(p, True)[o:o + k] = data[i:i + k]
            i += k

    # typed helpers
    def read_np(self, off, count, dtype):
        dt = np.dtype(dtype)
        return np.frombuffer(self.read(off, count * dt.itemsize), dtype=dt.newbyteorder("<")).astype(dt)

    def write_np(self, off, arr, dtype):
        self.write(off, np.ascontiguousarray(np.asarray(arr).astype(np.dtype(dtype).newbyteorder("<"))).tobytes())


class RegionMem:
    """Logical memory used by the unit tests: addr_t {rgn, off} -> Sparse per region."""

    def __init__(self):
        self.r = {k: Sparse() for k in (cfg.RGN_HBM, cfg.RGN_ACT, cfg.RGN_SCRATCH, cfg.RGN_RSVD)}

    def __getitem__(self, rgn):
        return self.r[rgn]

    def read_addr(self, a, n):
        rgn, off = cfg.addr_split(a)
        return self.r[rgn].read(off, n)

    def write_addr(self, a, data):
        rgn, off = cfg.addr_split(a)
        self.r[rgn].write(off, data)


def beat_to_bytes(v, nbytes):
    return int(v).to_bytes(nbytes, "little")


def bytes_to_beat(b):
    return int.from_bytes(b, "little")
