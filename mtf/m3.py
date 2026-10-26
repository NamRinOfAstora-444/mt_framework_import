"""
mtf/m3.py  --  port of MTF/3.ms

Demonstrates the v2 decoder path (hex-string dispatch keys) plus vertex
colour / alpha channels and 2-weight / 8-weight skinning.

Decoders:
    MODVertexBuffera013501e  "PaintedCharacter04[U-201NS]"   key "a013501e"
    MODVertexBufferd84e3026  "PaintedCharacter05[U-201NS]"   key "d84e3026"

Cross-file aliases from the original (c66fa03a, b86de02a, 77d87022) reference
structs defined in other MTF files; they are registered in those modules once
ported, to keep the global-namespace dispatch intact.
"""

import mathutils

from . import DecoderBase, register
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


class MODVertexBuffera013501e(DecoderBase):
    id = "PaintedCharacter04[U-201NS]"

    def _build_uvarray(self):
        self.uvarray = [
            ESUVArray(id=1, data=self.uv1array),
            ESUVArray(id=-2, data=self.alphaarray),
            ESUVArray(id=0, data=self.colorarray),
        ]

    def _decode(self, header, count):
        fl = header.fl
        scale = header.vtbuffscale
        bbmin = header.bbmin
        be = header.bendian
        r16s = fl.read_be16s if be else fl.read16s
        rhf = fl.read_be_half_float if be else fl.read_half_float
        for _ in range(count):
            raw = mathutils.Vector((r16s(), r16s(), r16s()))
            pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
            self.vertarray.append(cormat(pos))
            cwt = [r16s()]
            cwt.append(1 - cwt[0])
            self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
            fl.read_long()
            self.uv1array.append([rhf(), 1 - rhf(), 0])
            cbn = [rhf() + 1, rhf() + 1]
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])
            es_skin_optimizer(self, 2, cbn, cwt)


class MODVertexBufferd84e3026(DecoderBase):
    id = "PaintedCharacter05[U-201NS]"

    def _build_uvarray(self):
        self.uvarray = [
            ESUVArray(id=1, data=self.uv1array),
            ESUVArray(id=-2, data=self.alphaarray),
            ESUVArray(id=0, data=self.colorarray),
        ]

    def _decode(self, header, count):
        fl = header.fl
        scale = header.vtbuffscale
        bbmin = header.bbmin
        be = header.bendian
        r16s = fl.read_be16s if be else fl.read16s
        rhf = fl.read_be_half_float if be else fl.read_half_float
        for _ in range(count):
            raw = mathutils.Vector((r16s(), r16s(), r16s()))
            # NOTE: this format subtracts bbmin (sign differs from a013501e).
            pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) - bbmin
            self.vertarray.append(cormat(pos))
            cwt = [r16s()]
            self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
            for _w in range(4):
                cwt.append(fl.read8())
            bns = [fl.read_byte(signed=False) + 1 for _ in range(8)]
            self.uv1array.append([rhf(), 1 - rhf(), 0])
            cwt.append(rhf())
            cwt.append(rhf())
            fl.read_long()
            self.colorarray.append([fl.read8(), fl.read8(), fl.read8()])
            ca = fl.read8()
            self.alphaarray.append([ca, ca, ca])
            w8 = 1.0
            for w in range(7):
                w8 -= cwt[w]
            cwt.append(w8)
            es_skin_optimizer(self, 8, bns, cwt)


register("a013501e", MODVertexBuffera013501e)
register("d84e3026", MODVertexBufferd84e3026)
