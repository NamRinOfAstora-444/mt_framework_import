"""
read_stream.py
Faithful port of EasyStuff/ReadStream.ms (Lukas Cone, 2017).

MaxScript exposes a flat set of read* functions that operate on an open file
handle (`fl`). In Python we wrap a file object in a BinaryReader so the same
call shapes (reader.read_long(), reader.read_be_float(), etc.) read directly
against an in-memory buffer.

All numeric semantics are matched to the original:
  - read8  / read8s   : byte normalised to 0..1 / -1..1
  - read16 / read16s  : ushort/short normalised to 0..1 / -1..1
  - half float decode  : original _rdhf bit twiddling
  - pixel normal       : (byte-127) * 1/128
The coordinate-system matrix (MTFCormat) lives in unimesh_loader, not here.
"""

import struct
import mathutils


class BinaryReader:
    """Wraps a bytes buffer and mirrors the MaxScript read* helper set."""

    __slots__ = ("data", "pos", "size")

    SEEK_SET = 0
    SEEK_CUR = 1
    SEEK_END = 2

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.size = len(data)

    # -- seeking (mirrors fseek fl off mode) --------------------------------
    def seek(self, offset, whence=SEEK_SET):
        if whence == self.SEEK_SET:
            self.pos = offset
        elif whence == self.SEEK_CUR:
            self.pos += offset
        else:  # SEEK_END
            self.pos = self.size + offset

    def tell(self):
        return self.pos

    def skip(self, n):
        self.pos += n

    def _take(self, n):
        b = self.data[self.pos:self.pos + n]
        if len(b) < n:
            raise EOFError(
                "Read past end of file at offset %d (wanted %d bytes)" % (self.pos, n))
        self.pos += n
        return b

    # -- raw integers (little endian) ---------------------------------------
    def read_byte(self, signed=False):
        b = self._take(1)
        return struct.unpack("<b" if signed else "<B", b)[0]

    def read_short(self, signed=False):
        return struct.unpack("<h" if signed else "<H", self._take(2))[0]

    def read_long(self, signed=False):
        return struct.unpack("<i" if signed else "<I", self._take(4))[0]

    def read_float(self):
        return struct.unpack("<f", self._take(4))[0]

    def read_int64(self, signed=False):
        return struct.unpack("<q" if signed else "<Q", self._take(8))[0]

    # -- raw integers (big endian) ------------------------------------------
    def read_be_short(self, signed=False):
        return struct.unpack(">h" if signed else ">H", self._take(2))[0]

    def read_be_long(self, signed=False):
        return struct.unpack(">i" if signed else ">I", self._take(4))[0]

    def read_be_float(self):
        return struct.unpack(">f", self._take(4))[0]

    def read_be_int64(self, signed=False):
        return struct.unpack(">q" if signed else ">Q", self._take(8))[0]

    # -- normalised reads (ReadStream.ms exact constants) -------------------
    def read16s(self):                       # signed short / 32767
        return self.read_short(signed=True) / 32767.0

    def read16(self):                        # ushort / 65535
        return self.read_short(signed=False) / 65535.0

    def read_be16s(self):                    # ReadBE16s
        v = self.read_be_short(signed=False)
        if v > 32767:
            v = -1 - (v ^ 0xFFFF)
        return v / 32767.0

    def read_be16(self):                     # ReadBE16
        return self.read_be_short(signed=False) / 65535.0

    def read8s(self):                        # signed byte * 0.0078125
        return self.read_byte(signed=True) * 0.0078125

    def read8(self):                         # unsigned byte / 255 (approx const in src)
        return self.read_byte(signed=False) * 0.00392156862

    # -- vectors -------------------------------------------------------------
    def read_vector3(self):
        return mathutils.Vector(
            (self.read_float(), self.read_float(), self.read_float()))

    def read_be_vector3(self):
        return mathutils.Vector(
            (self.read_be_float(), self.read_be_float(), self.read_be_float()))

    # -- pixel normal (3 bytes, -127..128 scaled) ---------------------------
    def read_pixel_normal(self):
        b = self._take(3)
        return mathutils.Vector((
            (b[0] - 127) * 0.0078125,
            (b[1] - 127) * 0.0078125,
            (b[2] - 127) * 0.0078125,
        ))

    def read_pixel_normal16(self):
        x = (self.read_short(signed=False) - 32767) / 32767.0
        y = (self.read_short(signed=False) - 32767) / 32767.0
        z = (self.read_short(signed=False) - 32767) / 32767.0
        self.read_short()  # 4th component consumed in source layout
        return mathutils.Vector((x, y, z))

    # -- half floats (port of _rdhf) ----------------------------------------
    @staticmethod
    def _rdhf(data):
        # Port of _rdhf. MaxScript bit.get is 1-based: bit 16 -> python bit 15.
        sign = 1 if ((data >> 15) & 1) else 0
        exponent = ((data & 0x7C00) >> 10) - 16
        fraction = data & 0x3FF
        exponent_f = exponent + 127
        result = ((fraction << 13) | (exponent_f << 23) | (sign << 31)) & 0xFFFFFFFF
        f = struct.unpack("<f", struct.pack("<I", result))[0]
        return f * 2.0

    def read_half_float(self):
        return self._rdhf(self.read_short(signed=False))

    def read_be_half_float(self):
        v = self.read_be_short(signed=False)
        return self._rdhf(v)

    # -- pointer reads (32 / 64 bit, ReadPointer / ReadBEPointer) -----------
    def read_pointer(self, x64):
        if x64:
            tmp = self.read_long(signed=False)
            self.read_long(signed=False)  # high dword discarded
            return tmp
        return self.read_long(signed=False)

    def read_be_pointer(self, x64):
        if x64:
            self.read_be_long(signed=False)
            return self.read_be_long(signed=False)
        return self.read_be_long(signed=False)

    # -- strings -------------------------------------------------------------
    def read_chars(self, num):
        """bReadChars: read up to `num` bytes, stop at first NUL."""
        out = []
        skip = False
        for _ in range(num):
            bb = self.read_byte(signed=False)
            if bb == 0 and not skip:
                skip = True
            if not skip:
                out.append(chr(bb))
        return "".join(out)


# ----------------------------------------------------------------------------
# Triangle / strip collectors (CollectTris / CollectStrips, LE + BE).
# Each returns a list of [a, b, c] index triplets, +modf added to each index.
# ----------------------------------------------------------------------------
def collect_tris(reader, num, modf=1):
    out = []
    for _ in range(num):
        out.append([
            reader.read_short(signed=False) + modf,
            reader.read_short(signed=False) + modf,
            reader.read_short(signed=False) + modf,
        ])
    return out


def collect_be_tris(reader, num, modf=1):
    out = []
    for _ in range(num):
        out.append([
            reader.read_be_short(signed=False) + modf,
            reader.read_be_short(signed=False) + modf,
            reader.read_be_short(signed=False) + modf,
        ])
    return out


def _collect_strips(reader, num, modf, read_idx):
    """Shared de-stripification. 0xFFFF is a restart sentinel."""
    out = []
    f1 = read_idx()
    f2 = read_idx()
    f = 2
    while f < num:
        cf = read_idx()
        if cf == 0xFFFF:
            f1 = read_idx()
            f2 = read_idx()
            f += 3
        else:
            if (f % 2) == 1:
                tri = [cf, f2, f1]
            else:
                tri = [f1, f2, cf]
            out.append([tri[0] + modf, tri[1] + modf, tri[2] + modf])
            f1 = f2
            f2 = cf
            f += 1
    return out


def collect_strips(reader, num, modf=1):
    return _collect_strips(reader, num, modf,
                           lambda: reader.read_short(signed=False))


def collect_be_strips(reader, num, modf=1):
    # NOTE: ReadBEShort here returns the *unsigned* big-endian short, matching
    # the MaxScript CollectBEStrips which calls ReadBEShort (unsigned form).
    return _collect_strips(reader, num, modf,
                           lambda: reader.read_be_short(signed=False))


def proxima(value1, value2=0.0, epsilon=0.0001):
    return (value1 <= value2 + epsilon) and (value1 >= value2 - epsilon)
