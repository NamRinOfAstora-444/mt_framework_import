"""
mtf/ decoder package.

The MaxScript stored ~113 `MODVertexBuffer<hash>` structs across MTF/0.ms ..
MTF/7.ms, and dispatched at load time with:

    execute ("MODVertexBuffer" + blocktype)

i.e. a global-namespace lookup by block hash. Here each decoder is a small
class and REGISTRY maps the hash key to it. Hash keys match what the mesh-part
loader produces:

    * v1 parts  -> raw integer blocktype  (e.g. 0, 4, 8)
    * v2 / v3   -> lowercase hex string   (e.g. "d8297021")

Adding a decoder = translate the MaxScript struct's `on create` read loop into
`_decode(self, header, count)` and call `register("<key>")` (or add an alias).

See m0.py for the fully-worked reference (the MTF v1 formats).
"""

REGISTRY = {}


def register(key, cls):
    REGISTRY[key] = cls


def alias(new_key, existing_key):
    """Mirror of `MODVertexBufferX = MODVertexBufferY` lines."""
    REGISTRY[new_key] = REGISTRY[existing_key]


def get(key):
    return REGISTRY.get(key)


class DecoderBase:
    """Common scaffold for every vertex-buffer decoder.

    Subclasses set `id` and implement `_decode(header, count)`, appending to the
    standard arrays. Skin weights go through es_skin_optimizer exactly as in the
    MaxScript (which keeps the same bone/weight collapse behaviour).
    """

    id = "unk[N]"

    def __init__(self, header, count):
        self.vertarray = []
        self.uv1array = []
        self.uv2array = []
        self.uv3array = []
        self.colorarray = []
        self.alphaarray = []
        self.normalarray = []
        self.weightarray = []
        self.boneidarray = []
        self.facearray = None
        self.uvarray = []
        self.count = count
        self.headerref = header
        self._build_uvarray()
        if header is not None and header.fl is not None:
            self._decode(header, count)

    def _build_uvarray(self):
        """Default UV channel set; decoders override before _decode if needed."""
        from ..easystuff.unimesh_loader import ESUVArray
        self.uvarray = [ESUVArray(id=1, data=self.uv1array)]

    def _decode(self, header, count):
        raise NotImplementedError


def load_all():
    """Import every decoder module so they self-register into REGISTRY."""
    from . import m0  # noqa: F401
    # Additional decoder modules (m1..m7) are imported here as they are ported.
    try:
        from . import m1  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m2  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m3  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m4  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m5  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m6  # noqa: F401
    except ImportError:
        pass
    try:
        from . import m7  # noqa: F401
    except ImportError:
        pass
