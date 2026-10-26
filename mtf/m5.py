"""MT Framework MTF/5 decoder set (Monster Hunter XX).

Ported into Blender's DecoderBase interface. The layouts cover the vertex
buffer hashes defined by the supplied MTF/5 script. No original MaxScript is
executed at runtime.
"""
import mathutils

from . import DecoderBase, register, alias
from ..easystuff.unimesh_loader import ESUVArray, cormat, es_skin_optimizer


def _readers(header):
    fl = header.fl
    be = header.bendian
    return {
        "i16s": fl.read_be16s if be else fl.read16s,
        "f32": fl.read_be_float if be else fl.read_float,
        "half": fl.read_be_half_float if be else fl.read_half_float,
        "u16": (lambda: fl.read_be_short(signed=False)) if be else (lambda: fl.read_short(signed=False)),
        "i16": (lambda: fl.read_be_short(signed=True)) if be else (lambda: fl.read_short(signed=True)),
        "byte": lambda: fl.read_byte(signed=False),
        "u8": fl.read8,
        "i8s": fl.read8s,
    }


class _M5Base(DecoderBase):
    # Keep decoded normals for all versions. For MHGU v212, the mesh builder
    # uses them to validate triangle winding per face; XX v230 stays unchanged.

    def _set_uvs(self, uv2=False, colors=False):
        arr = [ESUVArray(id=1, data=self.uv1array)]
        if uv2:
            arr.append(ESUVArray(id=2, data=self.uv2array))
        if colors:
            arr.extend([ESUVArray(id=-2, data=self.alphaarray),
                        ESUVArray(id=0, data=self.colorarray)])
        self.uvarray = arr

    def _pos_float(self, r):
        self.vertarray.append(cormat(mathutils.Vector((r["f32"](), r["f32"](), r["f32"]()))))

    def _pos_i16(self, r, header):
        raw = mathutils.Vector((r["i16s"](), r["i16s"](), r["i16s"]()))
        p = mathutils.Vector(tuple(raw[i] * header.vtbuffscale[i] for i in range(3))) + header.bbmin
        self.vertarray.append(cormat(p))

    def _normal(self, r, fl):
        self.normalarray.append(cormat(mathutils.Vector((r["i8s"](), r["i8s"](), r["i8s"]()))))
        fl.read_byte()

    def _uv(self, target, r):
        target.append([r["half"](), 1.0 - r["half"](), 0.0])

    def _color_alpha(self, r):
        self.colorarray.append([r["u8"](), r["u8"](), r["u8"]()])
        a = r["u8"]()
        self.alphaarray.append([a, a, a])


class MODVertexBuffer9399c037(_M5Base):
    id = "PaintedModel071[U-2012N]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True, colors=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_float(r); self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r); self._uv(self.uv2array, r); self._color_alpha(r)


class MODVertexBufferafa63031(_M5Base):
    id = "Model091[U12N]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_float(r); self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r); self._uv(self.uv2array, r)


class MODVertexBufferc66fa03e(_M5Base):
    id = "Model111[U12]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_float(r); fl.read_long()
            self._uv(self.uv1array, r); self._uv(self.uv2array, r)


class MODVertexBuffercbcf7027(_M5Base):
    id = "SkinnedCharacterFace021[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]
            self._normal(r, fl)
            weights.extend(r["u8"]() for _ in range(4))
            bones = [r["byte"]() + 1 for _ in range(8)]
            self._uv(self.uv1array, r)
            weights.extend((r["half"](), r["half"]()))
            fl.read_long(); fl.read_long()
            self._uv(self.uv2array, r)
            fl.read_long()
            weights.append(1.0 - sum(weights))
            es_skin_optimizer(self, 8, bones, weights)


class MODVertexBufferd9e801e(_M5Base):
    id = "SkinnedCharacter061[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"](), 0.0]
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r)
            bones = [r["half"]() + 1, r["half"]() + 1]
            self._uv(self.uv2array, r)
            weights[1] = 1.0 - weights[0]
            es_skin_optimizer(self, 2, bones, weights)


class MODVertexBufferda55a023(_M5Base):
    id = "SkinnedCharacter051[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"](), 0.0, 0.0, 0.0]
            self._normal(r, fl); fl.read_long()
            bones = [r["byte"]() + 1 for _ in range(4)]
            self._uv(self.uv1array, r)
            weights[1], weights[2] = r["half"](), r["half"]()
            self._uv(self.uv2array, r)
            weights[3] = 1.0 - weights[0] - weights[1] - weights[2]
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBuffer1b9a2021(_M5Base):
    id = "PaintedCharacter08[U-2012NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True, colors=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]; weights.append(1.0 - weights[0])
            self._normal(r, fl); fl.read_long()
            bones = [r["half"]() + 1, r["half"]() + 1]
            self._uv(self.uv1array, r)
            fl.read_long(); fl.read_long()
            self._uv(self.uv2array, r); self._color_alpha(r)
            es_skin_optimizer(self, 2, bones, weights)


class MODVertexBuffer707fb01c(_M5Base):
    id = "PaintedCharacter06[U-2012NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True, colors=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            bone = r["u16"]() + 1
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r)
            fl.read_long(); fl.read_long()
            self._uv(self.uv2array, r); self._color_alpha(r)
            self.boneidarray.append([bone]); self.weightarray.append([1.0])


class MODVertexBuffercc510026(_M5Base):
    id = "PaintedCharacter07[U-2012NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True, colors=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]
            self._normal(r, fl); fl.read_long()
            bones = [r["byte"]() + 1 for _ in range(4)]
            weights.extend((r["half"](), r["half"]()))
            self._uv(self.uv1array, r)
            fl.read_long(); fl.read_long()
            self._uv(self.uv2array, r); self._color_alpha(r)
            weights.append(1.0 - sum(weights))
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBuffer64593025(_M5Base):
    id = "SkinnedCharacter031[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"](), 0.0, 0.0, 0.0]
            self._normal(r, fl); fl.read_long()
            bones = [r["byte"]() + 1 for _ in range(4)]
            self._uv(self.uv1array, r)
            weights[1], weights[2] = r["half"](), r["half"]()
            fl.read_long()
            self._uv(self.uv2array, r); fl.read_long()
            weights[3] = 1.0 - weights[0] - weights[1] - weights[2]
            es_skin_optimizer(self, 4, bones, weights)


class MODVertexBufferb3921020(_M5Base):
    id = "SkinnedCharacter041[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            weights = [r["i16s"]()]; weights.append(1.0 - weights[0])
            self._normal(r, fl); fl.read_long()
            bones = [r["half"]() + 1, r["half"]() + 1]
            self._uv(self.uv1array, r); self._uv(self.uv2array, r)
            fl.skip(8)
            es_skin_optimizer(self, 2, bones, weights)


class MODVertexBufferd829702c(_M5Base):
    id = "Model011[U1N]"
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_float(r); self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r)


class MODVertexBuffera8fab018_ow(_M5Base):
    id = "SingleSkinnedCharacter012[U1NS]"
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            bone = r["u16"]() + 1
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r)
            self.boneidarray.append([bone]); self.weightarray.append([1.0])


class MODVertexBuffercbf6c01a_ow(_M5Base):
    id = "PaintedModel052[U-201N]"
    def _build_uvarray(self):
        self._set_uvs(colors=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header); fl.read_short()
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r); self._color_alpha(r)


class MODVertexBufferd877801b_ow(_M5Base):
    id = "SingleSkinnedCharacter021[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            bone = r["u16"]() + 1
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r); fl.read_long()
            self._uv(self.uv2array, r); fl.read_long()
            self.boneidarray.append([bone]); self.weightarray.append([1.0])


class MODVertexBuffer667b1019_ow(_M5Base):
    id = "SingleSkinnedCharacter041[U12NS]"
    def _build_uvarray(self):
        self._set_uvs(uv2=True)
    def _decode(self, header, count):
        fl, r = header.fl, _readers(header)
        for _ in range(count):
            self._pos_i16(r, header)
            bone = r["u16"]() + 1
            self._normal(r, fl); fl.read_long()
            self._uv(self.uv1array, r); self._uv(self.uv2array, r)
            self.boneidarray.append([bone]); self.weightarray.append([1.0])


# Register the MTF/5 layouts.
for _key, _cls in (
    ("9399c037", MODVertexBuffer9399c037),
    ("afa63031", MODVertexBufferafa63031),
    ("c66fa03e", MODVertexBufferc66fa03e),
    ("cbcf7027", MODVertexBuffercbcf7027),
    ("d9e801e", MODVertexBufferd9e801e),
    ("da55a023", MODVertexBufferda55a023),
    ("1b9a2021", MODVertexBuffer1b9a2021),
    ("707fb01c", MODVertexBuffer707fb01c),
    ("cc510026", MODVertexBuffercc510026),
    ("64593025", MODVertexBuffer64593025),
    ("b3921020", MODVertexBufferb3921020),
    ("d829702c", MODVertexBufferd829702c),
    ("a8fab018", MODVertexBuffera8fab018_ow),
    ("cbf6c01a", MODVertexBuffercbf6c01a_ow),
    ("d877801b", MODVertexBufferd877801b_ow),
    ("667b1019", MODVertexBuffer667b1019_ow),
):
    register(_key, _cls)

# Aliases declared in MTF/5.ms; these classes are registered by m4.py.
alias("14d40022", "14d40021")
alias("207d603b", "207d6038")
alias("49b4f02d", "49b4f02a")
alias("5e7f2030", "5e7f202d")
alias("77d87024", "77d87023")
alias("a7d7d03a", "a7d7d037")
alias("bb424027", "bb424025")
