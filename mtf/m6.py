"""
mtf/m6.py  --  full port of MTF/6.ms

Float-position vertex formats (MH3U/MH4U/Mercenaries-3ds era and related).
These all use 32-bit float positions, a signed-byte normal + pad, float UVs,
and various skin layouts. Block hash 1b36016 (this test model's format) is the
alias of f606f017 at the bottom.

Decoders:
    f606f017 SkinnedCharacter08[U1NS]
    82917009 Model04[U1N]            (positions NOT scaled by vtbuffscale)
    638f1011 Model05[U12N]
    9dfd7019 SkinnedCharacter10[U1NS]
    43fb3015 SkinnedCharacter09[U1NS]
    9a9d201d SkinnedCharacter11[U1NS]
    3f78800c PaintedModel06[U-201N]
    59dc400b PaintedModel10[U-201N]
    6180300a Model12[U1N]
    7ba7401b PaintedCharacter09[U-201NS]
    866b6018 SkinnedCharacter12[U12NS]
    d6784014 SkinnedCharacter13[U1NS]
    ce5a801c PaintedCharacter10[U-201NS]
  aliases:
    ae252019 = 9dfd7019
    1b36016  = f606f017
"""

import mathutils

from . import DecoderBase, register, alias
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


def _scaled_pos(rf, scale, bbmin):
    raw = mathutils.Vector((rf(), rf(), rf()))
    return cormat(mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin)


def _normal(fl):
    n = cormat(mathutils.Vector((fl.read8s(), fl.read8s(), fl.read8s())))
    fl.read_byte()
    return n


class MODVertexBufferf606f017(DecoderBase):
    id = "SkinnedCharacter08[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            fl.read_long()
            cbn.append(fl.read_byte(signed=False) + 1)
            cbn.append(fl.read_byte(signed=False) + 1)
            cwt.append(fl.read8())
            cwt.append(fl.read8())
            es_skin_optimizer(self, 4, cbn, cwt)


class MODVertexBuffer82917009(DecoderBase):
    id = "Model04[U1N]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            # NOTE: this format does NOT apply vtbuffscale / bbmin.
            self.vertarray.append(cormat(mathutils.Vector((rf(), rf(), rf()))))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])


class MODVertexBuffer638f1011(DecoderBase):
    id = "Model05[U12N]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=2, data=self.uv2array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            fl.read_long()
            fl.read_long()
            self.uv2array.append([rf(), 1 - rf(), 0])


class MODVertexBuffer9dfd7019(DecoderBase):
    id = "SkinnedCharacter10[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            fl.read_long()
            for _b in range(3):
                cbn.append(fl.read_byte(signed=False) + 1)
                cbn.append(fl.read_byte(signed=False) + 1)
                cwt.append(fl.read8())
                cwt.append(fl.read8())
            es_skin_optimizer(self, 8, cbn, cwt)


class MODVertexBuffer43fb3015(DecoderBase):
    id = "SkinnedCharacter09[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            fl.read_long()
            es_skin_optimizer(self, 2, cbn, cwt)


class MODVertexBuffer9a9d201d(DecoderBase):
    id = "SkinnedCharacter11[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            fl.read_long()
            cbn.append(fl.read_byte(signed=False) + 1)
            cbn.append(fl.read_byte(signed=False) + 1)
            cwt.append(fl.read8())
            cwt.append(fl.read8())
            fl.skip(40)
            es_skin_optimizer(self, 4, cbn, cwt)


class MODVertexBuffer3f78800c(DecoderBase):
    id = "PaintedModel06[U-201N]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=-2, data=self.alphaarray),
                        ESUVArray(id=0, data=self.colorarray)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            fl.read_long()
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])


class MODVertexBuffer59dc400b(DecoderBase):
    id = "PaintedModel10[U-201N]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=-2, data=self.alphaarray),
                        ESUVArray(id=0, data=self.colorarray)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])


class MODVertexBuffer6180300a(DecoderBase):
    id = "Model12[U1N]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            fl.read_long()
            fl.read_long()


class MODVertexBuffer7ba7401b(DecoderBase):
    id = "PaintedCharacter09[U-201NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=-2, data=self.alphaarray),
                        ESUVArray(id=0, data=self.colorarray)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])
            es_skin_optimizer(self, 2, cbn, cwt)


class MODVertexBuffer866b6018(DecoderBase):
    id = "SkinnedCharacter12[U12NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=2, data=self.uv2array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            fl.read_long()
            cbn.append(fl.read_byte(signed=False) + 1)
            cbn.append(fl.read_byte(signed=False) + 1)
            cwt.append(fl.read8())
            cwt.append(fl.read8())
            self.uv2array.append([rf(), 1 - rf(), 0])
            es_skin_optimizer(self, 4, cbn, cwt)


class MODVertexBufferd6784014(DecoderBase):
    id = "SkinnedCharacter13[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            es_skin_optimizer(self, 2, cbn, cwt)


class MODVertexBufferce5a801c(DecoderBase):
    id = "PaintedCharacter10[U-201NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array),
                        ESUVArray(id=-2, data=self.alphaarray),
                        ESUVArray(id=0, data=self.colorarray)]

    def _decode(self, header, count):
        fl = header.fl; scale = header.vtbuffscale; bbmin = header.bbmin
        rf = fl.read_be_float if header.bendian else fl.read_float
        for _ in range(count):
            self.vertarray.append(_scaled_pos(rf, scale, bbmin))
            self.normalarray.append(_normal(fl))
            self.uv1array.append([rf(), 1 - rf(), 0])
            cbn = [fl.read_byte(signed=False) + 1, fl.read_byte(signed=False) + 1]
            cwt = [fl.read8(), fl.read8()]
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])
            fl.read_long()
            es_skin_optimizer(self, 2, cbn, cwt)


register("f606f017", MODVertexBufferf606f017)
register("82917009", MODVertexBuffer82917009)
register("638f1011", MODVertexBuffer638f1011)
register("9dfd7019", MODVertexBuffer9dfd7019)
register("43fb3015", MODVertexBuffer43fb3015)
register("9a9d201d", MODVertexBuffer9a9d201d)
register("3f78800c", MODVertexBuffer3f78800c)
register("59dc400b", MODVertexBuffer59dc400b)
register("6180300a", MODVertexBuffer6180300a)
register("7ba7401b", MODVertexBuffer7ba7401b)
register("866b6018", MODVertexBuffer866b6018)
register("d6784014", MODVertexBufferd6784014)
register("ce5a801c", MODVertexBufferce5a801c)
alias("ae252019", "9dfd7019")
alias("1b36016", "f606f017")
