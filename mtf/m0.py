"""
mtf/m0.py  --  port of MTF/0.ms  (MTF v1 vertex formats)

Reference implementation for the decoder pattern. Every per-vertex read is a
line-for-line translation of the MaxScript `on create` loop, with both the
big-endian and little-endian branches preserved.

Decoders covered here:
    MODVertexBuffer0  -> key 0   "Model_ver1 [U12NS]"
    MODVertexBuffer4  = alias of 0 (non-tangent)
    MODVertexBuffer8  -> key 8   "DSModel_ver1 [U1NS]"
"""

import mathutils

from . import DecoderBase, register, alias
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


class MODVertexBuffer0(DecoderBase):
    id = "Model_ver1 [U12NS]"

    def _build_uvarray(self):
        self.uvarray = [
            ESUVArray(id=1, data=self.uv1array),
            ESUVArray(id=2, data=self.uv2array),
        ]

    def _bone_ids(self, header):
        fl = header.fl
        cbn = [fl.read_byte(signed=False) + 1 for _ in range(4)]
        cwt = [fl.read8() for _ in range(4)]
        es_skin_optimizer(self, 4, cbn, cwt)

    def _decode(self, header, count):
        fl = header.fl
        scale = header.vtbuffscale
        bbmin = header.bbmin
        if header.bendian:
            for _ in range(count):
                raw = mathutils.Vector((fl.read_be16s(), fl.read_be16s(), fl.read_be16s()))
                pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
                self.vertarray.append(cormat(pos))
                fl.read_short()
                self._bone_ids(header)
                self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
                fl.read_long()
                self.uv1array.append([fl.read_be_half_float(), 1 - fl.read_be_half_float(), 0])
                self.uv2array.append([fl.read_be_half_float(), 1 - fl.read_be_half_float(), 0])
        else:
            for _ in range(count):
                raw = mathutils.Vector((fl.read16s(), fl.read16s(), fl.read16s()))
                pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
                self.vertarray.append(cormat(pos))
                fl.read_short()
                self._bone_ids(header)
                self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
                fl.read_long()
                self.uv1array.append([fl.read_half_float(), 1 - fl.read_half_float(), 0])
                self.uv2array.append([fl.read_half_float(), 1 - fl.read_half_float(), 0])


class MODVertexBuffer8(DecoderBase):
    id = "DSModel_ver1 [U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _bone_ids(self, header):
        fl = header.fl
        cbn = [fl.read_byte(signed=False) + 1 for _ in range(8)]
        cwt = [fl.read8() for _ in range(8)]
        es_skin_optimizer(self, 8, cbn, cwt)

    def _decode(self, header, count):
        fl = header.fl
        scale = header.vtbuffscale
        bbmin = header.bbmin
        if header.bendian:
            for _ in range(count):
                raw = mathutils.Vector((fl.read_be16s(), fl.read_be16s(), fl.read_be16s()))
                pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
                self.vertarray.append(cormat(pos))
                fl.read_short()
                self._bone_ids(header)
                self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
                self.uv1array.append([fl.read_be_half_float(), 1 - fl.read_be_half_float(), 0])
        else:
            for _ in range(count):
                raw = mathutils.Vector((fl.read16s(), fl.read16s(), fl.read16s()))
                pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
                self.vertarray.append(cormat(pos))
                fl.read_short()
                self._bone_ids(header)
                self.normalarray.append(cormat(fl.read_pixel_normal())); fl.read_byte()
                self.uv1array.append([fl.read_half_float(), 1 - fl.read_half_float(), 0])


# v1 dispatch keys are the raw integer blocktype.
register(0, MODVertexBuffer0)
register(8, MODVertexBuffer8)
alias(4, 0)  # MODVertexBuffer4 = MODVertexBuffer0  (non tangent)
