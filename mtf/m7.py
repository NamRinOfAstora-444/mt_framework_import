"""MT Framework MTF/7 vertex formats.

Ported from the original Lukas Cone MT Framework Tool (MTF/7.ms).
These formats are used by newer MT Framework models, including the MHW-era
v237 model layouts.
"""
import mathutils

from . import DecoderBase, register, alias, REGISTRY
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


def _pos(fl, header):
    rf = fl.read_be_float if header.bendian else fl.read_float
    return cormat(mathutils.Vector((rf(), rf(), rf())))


def _normal(fl):
    n = cormat(mathutils.Vector((fl.read8s(), fl.read8s(), fl.read8s())))
    fl.read_byte()
    fl.read_long()
    return n


def _half(fl, header):
    rf = fl.read_be_half_float if header.bendian else fl.read_half_float
    return rf()


def _packed4(fl, header):
    value = fl.read_be_long() if header.bendian else fl.read_long(signed=False)
    w = [
        (value & 0x3ff) * 0.000977517,
        ((value >> 10) & 0x3ff) * 0.000977517,
        ((value >> 20) & 0x3ff) * 0.000977517,
    ]
    w.append(max(0.0, 1.0 - w[0] - w[1] - w[2]))
    return w


def _packed8(fl, header):
    w = _packed4(fl, header)[:3]
    rf = fl.read_byte
    w.extend([rf() * 0.25, rf() * 0.25, rf() * 0.25, rf() * 0.25])
    w.append(max(0.0, 1.0 - sum(w)))
    return w


def _bones(fl, header, count):
    rb = fl.read_byte
    return [rb(signed=False) + 1 for _ in range(count)]


class MODVertexBuffer818904dc(DecoderBase):
    id = "Model17[U1N]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]
    def _decode(self, header, count):
        fl = header.fl
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            fl.read_long()


class MODVertexBufferf06033f(DecoderBase):
    id = "SkinnedCharacter14[U1NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]
    def _decode(self, header, count):
        fl = header.fl
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed4(fl, header)
            bones = _bones(fl, header, 4)
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBuffer81f58067(DecoderBase):
    id = "SkinnedCharacterFace04[U1NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]
    def _decode(self, header, count):
        fl = header.fl
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed8(fl, header)
            bones = _bones(fl, header, 8)
            es_skin_optimizer(self, 8, bones, weights)


class MODVertexBufferf471fe45(DecoderBase):
    id = "SkinnedCharacter15[U12NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=2, data=self.uv2array)]
    def _decode(self, header, count):
        fl = header.fl
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv2array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed4(fl, header)
            bones = _bones(fl, header, 4)
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBuffer3c730760(DecoderBase):
    id = "PaintedCharacter11[U-201NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=-2, data=self.alphaarray), ESUVArray(id=0, data=self.colorarray)]
    def _decode(self, header, count):
        fl = header.fl
        rb = fl.read_byte
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed4(fl, header)
            bones = _bones(fl, header, 4)
            self.colorarray.append([rb(signed=False), rb(signed=False), rb(signed=False)])
            a = rb(signed=False)
            self.alphaarray.append([a, a, a])
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBufferb2fc0083(DecoderBase):
    id = "SkinnedCharacter16[U12NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=2, data=self.uv2array)]
    def _decode(self, header, count):
        fl = header.fl
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv2array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed4(fl, header)
            bones = _bones(fl, header, 4)
            fl.read_long()
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBuffer366995a7(DecoderBase):
    id = "SkinnedCharacterFace05[U1NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=-2, data=self.alphaarray), ESUVArray(id=0, data=self.colorarray)]
    def _decode(self, header, count):
        fl = header.fl
        rb = fl.read_byte
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed8(fl, header)
            bones = _bones(fl, header, 8)
            fl.read_long()
            self.colorarray.append([rb(signed=False), rb(signed=False), rb(signed=False)])
            a = rb(signed=False)
            self.alphaarray.append([a, a, a])
            es_skin_optimizer(self, 8, bones, weights)


class MODVertexBufferc9690ab8(DecoderBase):
    id = "PaintedModel11[U-201234N]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=2, data=self.uv2array), ESUVArray(id=4, data=self.uv4array), ESUVArray(id=3, data=self.uv3array), ESUVArray(id=-2, data=self.alphaarray), ESUVArray(id=0, data=self.colorarray)]
    def _decode(self, header, count):
        fl = header.fl
        rb = fl.read_byte
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv2array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv3array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv4array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.colorarray.append([rb(signed=False), rb(signed=False), rb(signed=False)])
            a = rb(signed=False)
            self.alphaarray.append([a, a, a])


class MODVertexBufferb8e69244(DecoderBase):
    id = "PaintedCharacterFace01[U-2012NS]"
    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array), ESUVArray(id=2, data=self.uv2array), ESUVArray(id=-2, data=self.alphaarray), ESUVArray(id=0, data=self.colorarray)]
    def _decode(self, header, count):
        fl = header.fl
        rb = fl.read_byte
        for _ in range(count):
            self.vertarray.append(_pos(fl, header))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([_half(fl, header), 1 - _half(fl, header), 0])
            self.uv2array.append([_half(fl, header), 1 - _half(fl, header), 0])
            weights = _packed8(fl, header)
            bones = _bones(fl, header, 8)
            fl.read_long()
            self.colorarray.append([rb(signed=False), rb(signed=False), rb(signed=False)])
            a = rb(signed=False)
            self.alphaarray.append([a, a, a])
            es_skin_optimizer(self, 8, bones, weights)


register("818904dc", MODVertexBuffer818904dc)
register("f06033f", MODVertexBufferf06033f)
register("81f58067", MODVertexBuffer81f58067)
register("f471fe45", MODVertexBufferf471fe45)
register("3c730760", MODVertexBuffer3c730760)
register("b2fc0083", MODVertexBufferb2fc0083)
register("366995a7", MODVertexBuffer366995a7)
register("c9690ab8", MODVertexBufferc9690ab8)
register("b8e69244", MODVertexBufferb8e69244)

# MTF/7.ms aliases.
if "d829702c" in REGISTRY:
    alias("a756f2f9", "d829702c")
if "5e7f202d" in REGISTRY:
    alias("a5104ca0", "5e7f202d")
alias("f637401c", "f06033f")
