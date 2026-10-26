"""MT Framework / MTF/4.ms -> Blender decoder

DMC4 SE / MTF version 4 test decoder set.

This file follows the uploaded MaxScript source MTF/4.ms directly. It keeps
all 11 concrete vertex-buffer layouts from that file and does not replace the
unsupported `= ok` hashes with guessed decoders.

Drop this file in:
    mt_framework_import/mtf/m4.py

It expects the existing Blender port's DecoderBase, register(), alias(),
ESUVArray, cormat(), and es_skin_optimizer().
"""

import mathutils

from . import DecoderBase, register
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


def _readers(header):
    fl = header.fl
    be = header.bendian
    return {
        "i16s": fl.read_be16s if be else fl.read16s,
        "f32": fl.read_be_float if be else fl.read_float,
        "half": fl.read_be_half_float if be else fl.read_half_float,
        "short_u": (lambda: fl.read_be_short(signed=False)) if be else (lambda: fl.read_short(signed=False)),
        "byte_u": lambda: fl.read_byte(signed=False),
        "u8": fl.read8,
        "i8s": fl.read8s,
    }


class _M4Base(DecoderBase):
    """Small helpers shared by the MTF/4 layouts."""

    def _build_uvarray(self, include_colors=False, uv2=False):
        arrays = [ESUVArray(id=1, data=self.uv1array)]
        if uv2:
            arrays.append(ESUVArray(id=2, data=self.uv2array))
        if include_colors:
            arrays.extend([
                ESUVArray(id=-2, data=self.alphaarray),
                ESUVArray(id=0, data=self.colorarray),
            ])
        self.uvarray = arrays

    def _pos_i16(self, r, header):
        raw = mathutils.Vector((r["i16s"](), r["i16s"](), r["i16s"]()))
        scale = header.vtbuffscale
        bbmin = header.bbmin
        pos = mathutils.Vector(tuple(raw[i] * scale[i] for i in range(3))) + bbmin
        self.vertarray.append(cormat(pos))

    def _pos_f32(self, r):
        self.vertarray.append(cormat(mathutils.Vector((r["f32"](), r["f32"](), r["f32"]()))))

    def _normal_i8(self, r, fl):
        n = mathutils.Vector((r["i8s"](), r["i8s"](), r["i8s"]()))
        self.normalarray.append(cormat(n))
        fl.read_byte()  # fourth normal component/pad

    def _uv1(self, r):
        self.uv1array.append([r["half"](), 1.0 - r["half"](), 0.0])

    def _color_alpha(self, r):
        self.colorarray.append([r["u8"](), r["u8"](), r["u8"]()])
        a = r["u8"]()
        self.alphaarray.append([a, a, a])


class MODVertexBuffer14d40021(_M4Base):
    id = "SkinnedCharacter011[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)

            # wts = [read16s/readbe16s, 0, 0, 0]
            wts = [r["i16s"](), 0.0, 0.0, 0.0]

            self._normal_i8(r, fl)
            fl.read_long()

            bns = [r["byte_u"]() + 1 for _ in range(4)]
            self._uv1(r)
            wts[1] = r["half"]()
            wts[2] = r["half"]()
            wts[3] = 1.0 - wts[0] - wts[1] - wts[2]
            es_skin_optimizer(self, 4, bns, wts)


class MODVertexBuffera7d7d037(_M4Base):
    id = "Model03[U1]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_f32(r)
            fl.read_long()
            self._uv1(r)


class MODVertexBuffer207d6038(_M4Base):
    id = "PaintedModel03[U-201]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_f32(r)
            fl.read_long()
            self._uv1(r)
            self._color_alpha(r)


class MODVertexBuffer77d87023(_M4Base):
    id = "PaintedCharacter011[U-201NS]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            wts = [r["i16s"](), 0.0, 0.0, 0.0]
            self._normal_i8(r, fl)
            fl.read_long()
            bns = [r["byte_u"]() + 1 for _ in range(4)]
            self._uv1(r)
            wts[1] = r["half"]()
            wts[2] = r["half"]()
            self._color_alpha(r)
            wts[3] = 1.0 - wts[0] - wts[1] - wts[2]
            es_skin_optimizer(self, 4, bns, wts)


class MODVertexBuffera013501f(_M4Base):
    id = "PaintedCharacter03[U-201NS]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            w0 = r["i16s"]()
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)
            bns = [r["half"]() + 1, r["half"]() + 1]
            self._color_alpha(r)
            es_skin_optimizer(self, 2, bns, [w0, 1.0 - w0])


class MODVertexBuffera8fab019(_M4Base):
    id = "SingleSkinnedCharacter011[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            bone = r["short_u"]() + 1
            self.weightarray.append([1.0])
            self.boneidarray.append([bone])
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)


class MODVertexBufferbb424025(_M4Base):
    id = "SkinnedCharacterFace011[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]
            self._normal_i8(r, fl)
            weights.extend(r["u8"]() for _ in range(4))
            bns = [r["byte_u"]() + 1 for _ in range(8)]
            self._uv1(r)
            weights.append(r["half"]())
            weights.append(r["half"]())
            fl.read_long()
            weights.append(1.0 - sum(weights[:7]))
            es_skin_optimizer(self, 8, bns, weights)


class MODVertexBufferc31f201d(_M4Base):
    id = "SkinnedCharacter021[U1NS]"

    def _build_uvarray(self):
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            w0 = r["i16s"]()
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)
            bns = [r["half"]() + 1, r["half"]() + 1]
            es_skin_optimizer(self, 2, bns, [w0, 1.0 - w0])


class MODVertexBufferd8297029(_M4Base):
    id = "PaintedModel04[U-201N]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_f32(r)
            self._normal_i8(r, fl)
            self._color_alpha(r)
            self._uv1(r)


class MODVertexBufferd84e3027(_M4Base):
    id = "PaintedCharacter02[U-201NS]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]
            self._normal_i8(r, fl)
            weights.extend(r["u8"]() for _ in range(4))
            bns = [r["byte_u"]() + 1 for _ in range(8)]
            self._uv1(r)
            weights.append(r["half"]())
            weights.append(r["half"]())
            fl.read_long()
            self._color_alpha(r)
            weights.append(1.0 - sum(weights[:7]))
            es_skin_optimizer(self, 8, bns, weights)


class MODVertexBuffer5e7f202d(_M4Base):
    id = "Model06[U12N]"

    def _build_uvarray(self):
        self._build_uvarray(uv2=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_f32(r)
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)
            self.uv2array.append([r["half"](), 1.0 - r["half"](), 0.0])


class MODVertexBuffer49b4f02a(_M4Base):
    id = "PaintedModel021[U-201N]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_f32(r)
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)
            self._color_alpha(r)


class MODVertexBuffercbf6c01b(_M4Base):
    id = "PaintedModel051[U-201N]"

    def _build_uvarray(self):
        self._build_uvarray(include_colors=True)

    def _decode(self, header, count):
        fl = header.fl
        r = _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            fl.read_short()  # source: readshort (discarded)
            self._normal_i8(r, fl)
            fl.read_long()
            self._uv1(r)
            self._color_alpha(r)


# Registration mirrors execute("MODVertexBuffer" + blocktype).
register("14d40021", MODVertexBuffer14d40021)
register("a7d7d037", MODVertexBuffera7d7d037)
register("207d6038", MODVertexBuffer207d6038)
register("77d87023", MODVertexBuffer77d87023)
register("a013501f", MODVertexBuffera013501f)
register("a8fab019", MODVertexBuffera8fab019)
register("bb424025", MODVertexBufferbb424025)
register("c31f201d", MODVertexBufferc31f201d)
register("d8297029", MODVertexBufferd8297029)
register("d84e3027", MODVertexBufferd84e3027)
register("5e7f202d", MODVertexBuffer5e7f202d)
register("49b4f02a", MODVertexBuffer49b4f02a)
register("cbf6c01b", MODVertexBuffercbf6c01b)
