# Author: Jonas Rembser CERN 08/2026

################################################################################
# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

"""A small semantic type model shared by both transpiler backends.

The model deliberately does *not* try to reproduce the C++ type system.  The
generated code declares local variables as ``auto``, so the real types are
always the ones the C++ compiler deduces.  This model exists only to decide
which *rewrite* to apply -- for example whether ``x.sum()`` must become
``Sum(x)``, whether ``a | b`` is a logical or a bitwise or, or whether a value
needs Python's floor-division semantics.  Being approximate here can produce a
compile error, never a silently wrong number.
"""

import re

from .errors import PyDeclareError

# ---------------------------------------------------------------------------
# The type model
# ---------------------------------------------------------------------------

FUND = "fund"  # bool, int, double, ...
CONTAINER = "container"  # RVec<T>, std::vector<T>, std::array<T, N>
OPAQUE = "opaque"  # any other C++ class known to cling, e.g. PtEtaPhiMVector
UNKNOWN = "unknown"  # we could not infer it; emit auto and let C++ decide
VOID = "void"


class CppType:
    """A semantic type: a kind, plus a C++ spelling for the leaf types."""

    __slots__ = ("kind", "name", "elem", "container", "size")

    def __init__(self, kind, name=None, elem=None, container=None, size=None):
        self.kind = kind
        self.name = name
        self.elem = elem
        self.container = container
        self.size = size

    # -- predicates ---------------------------------------------------------

    @property
    def is_container(self):
        return self.kind == CONTAINER

    @property
    def is_fund(self):
        return self.kind == FUND

    @property
    def is_unknown(self):
        return self.kind == UNKNOWN

    @property
    def is_bool(self):
        """True for bool scalars and for containers of bool."""
        if self.kind == FUND:
            return self.name == "bool"
        if self.kind == CONTAINER:
            return self.elem is not None and self.elem.is_bool
        return False

    @property
    def is_integral(self):
        leaf = self.elem if self.kind == CONTAINER else self
        return leaf is not None and leaf.kind == FUND and leaf.name in INTEGRAL_TYPES

    @property
    def is_floating(self):
        leaf = self.elem if self.kind == CONTAINER else self
        return leaf is not None and leaf.kind == FUND and leaf.name in FLOATING_TYPES

    @property
    def is_signed(self):
        leaf = self.elem if self.kind == CONTAINER else self
        return leaf is not None and leaf.kind == FUND and leaf.name not in UNSIGNED_TYPES

    @property
    def is_numeric(self):
        return self.is_integral or self.is_floating

    def scalar(self):
        """The element type for containers, self otherwise."""
        if self.kind == CONTAINER:
            return self.elem if self.elem is not None else UNKNOWN_T
        return self

    # -- spelling -----------------------------------------------------------

    def cpp(self):
        """The canonical C++ spelling of this type."""
        if self.kind in (FUND, OPAQUE):
            return self.name
        if self.kind == VOID:
            return "void"
        if self.kind == CONTAINER:
            inner = self.elem.cpp() if self.elem is not None else "double"
            if self.container == "array":
                return "std::array<{}, {}>".format(inner, self.size)
            if self.container == "vector":
                return "std::vector<{}>".format(inner)
            return "ROOT::VecOps::RVec<{}>".format(inner)
        raise PyDeclareError("Cannot spell an unknown type in C++")

    def __repr__(self):
        if self.kind == UNKNOWN:
            return "<unknown>"
        try:
            return self.cpp()
        except PyDeclareError:
            return "<{}>".format(self.kind)

    def __eq__(self, other):
        if not isinstance(other, CppType):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.name == other.name
            and self.container == other.container
            and self.size == other.size
            and self.elem == other.elem
        )

    def __hash__(self):
        return hash((self.kind, self.name, self.container, self.size))


# ---------------------------------------------------------------------------
# Fundamental types
# ---------------------------------------------------------------------------

# Canonical spelling -> conversion rank.  Used only to pick a result type for
# mixed-type arithmetic; the C++ compiler has the final word.
INTEGRAL_RANKS = {
    "bool": 0,
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "short": 2,
    "unsigned short": 2,
    "int": 3,
    "unsigned int": 4,
    "long": 5,
    "unsigned long": 6,
    "long long": 7,
    "unsigned long long": 8,
}
FLOATING_RANKS = {"float": 20, "double": 21, "long double": 22}

INTEGRAL_TYPES = frozenset(INTEGRAL_RANKS)
FLOATING_TYPES = frozenset(FLOATING_RANKS)
UNSIGNED_TYPES = frozenset(
    ["bool", "unsigned char", "unsigned short", "unsigned int", "unsigned long", "unsigned long long"]
)

# Spellings the user may write, normalised to the canonical form above.
_FUND_ALIASES = {
    "bool": "bool",
    "_Bool": "bool",
    "char": "char",
    "signed char": "signed char",
    "unsigned char": "unsigned char",
    "short": "short",
    "short int": "short",
    "signed short": "short",
    "unsigned short": "unsigned short",
    "unsigned short int": "unsigned short",
    "int": "int",
    "signed": "int",
    "signed int": "int",
    "unsigned": "unsigned int",
    "unsigned int": "unsigned int",
    "long": "long",
    "long int": "long",
    "signed long": "long",
    "unsigned long": "unsigned long",
    "unsigned long int": "unsigned long",
    "long long": "long long",
    "long long int": "long long",
    "unsigned long long": "unsigned long long",
    "unsigned long long int": "unsigned long long",
    "float": "float",
    "double": "double",
    "long double": "long double",
    # ROOT typedefs that show up as RDataFrame column types
    "Bool_t": "bool",
    "Char_t": "char",
    "UChar_t": "unsigned char",
    "Short_t": "short",
    "UShort_t": "unsigned short",
    "Int_t": "int",
    "UInt_t": "unsigned int",
    "Long_t": "long",
    "ULong_t": "unsigned long",
    "Long64_t": "long long",
    "ULong64_t": "unsigned long long",
    "Float_t": "float",
    "Double_t": "double",
    "size_t": "unsigned long",
    "std::size_t": "unsigned long",
    "ROOT::RDF::RSampleInfo": None,  # not fundamental, falls through to opaque
}

BOOL_T = CppType(FUND, "bool")
INT_T = CppType(FUND, "int")
LONG_T = CppType(FUND, "long")
ULONG_T = CppType(FUND, "unsigned long")
LLONG_T = CppType(FUND, "long long")
ULLONG_T = CppType(FUND, "unsigned long long")
FLOAT_T = CppType(FUND, "float")
DOUBLE_T = CppType(FUND, "double")
UNKNOWN_T = CppType(UNKNOWN)
VOID_T = CppType(VOID)


def fund(name):
    return CppType(FUND, name)


def rvec_of(elem):
    """An RVec with the given element type."""
    return CppType(CONTAINER, elem=elem, container="rvec")


# Backwards-compatible short name used in a few places.
rvec = rvec_of


# ---------------------------------------------------------------------------
# Parsing C++ type spellings
# ---------------------------------------------------------------------------

_RVEC_ALIASES = {
    "RVecB": "bool",
    "RVecC": "char",
    "RVecD": "double",
    "RVecF": "float",
    "RVecI": "int",
    "RVecL": "long",
    "RVecLL": "long long",
    "RVecU": "unsigned int",
    "RVecUL": "unsigned long",
    "RVecULL": "unsigned long long",
}

_CONTAINER_HEADS = {
    "RVec": "rvec",
    "ROOT::RVec": "rvec",
    "VecOps::RVec": "rvec",
    "ROOT::VecOps::RVec": "rvec",
    "vector": "vector",
    "std::vector": "vector",
    "array": "array",
    "std::array": "array",
}


def _strip_cv_ref(s):
    s = s.strip()
    while s.endswith("&") or s.endswith("*"):
        if s.endswith("*"):
            raise PyDeclareError("Pointer types are not supported as arguments: '{}'".format(s))
        s = s[:-1].strip()
    # Remove leading/trailing const and volatile
    changed = True
    while changed:
        changed = False
        for kw in ("const", "volatile"):
            if s.startswith(kw + " "):
                s = s[len(kw) + 1 :].strip()
                changed = True
            if s.endswith(" " + kw):
                s = s[: -len(kw) - 1].strip()
                changed = True
    return s


def _split_template_args(s):
    """Split 'int, 3' or 'RVec<int>, 2' on top-level commas."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_type(spelling):
    """Parse a C++ type spelling into a CppType.

    Understands fundamental types (and the usual ROOT typedefs), RVec, its
    short aliases (RVecF, ...), std::vector and std::array, and treats anything
    else as an opaque C++ class.
    """
    if spelling is None:
        return UNKNOWN_T
    if isinstance(spelling, CppType):
        return spelling

    s = _strip_cv_ref(str(spelling))
    if not s:
        return UNKNOWN_T
    if s == "void":
        return VOID_T

    # Whitespace normalisation: collapse runs of spaces
    s = re.sub(r"\s+", " ", s)

    if s in _FUND_ALIASES and _FUND_ALIASES[s] is not None:
        return CppType(FUND, _FUND_ALIASES[s])

    # The short aliases are spelled both bare and namespace-qualified.
    alias = s.split("::")[-1] if s.startswith(("ROOT::", "ROOT::VecOps::")) else s
    if alias in _RVEC_ALIASES:
        return CppType(CONTAINER, elem=CppType(FUND, _RVEC_ALIASES[alias]), container="rvec")

    m = re.match(r"^([\w:]+)\s*<(.*)>$", s)
    if m:
        head, args = m.group(1), m.group(2)
        kind = _CONTAINER_HEADS.get(head)
        if kind is not None:
            parts = _split_template_args(args)
            if not parts:
                raise PyDeclareError("Cannot parse container type '{}'".format(spelling))
            elem = parse_type(parts[0])
            size = None
            if kind == "array":
                if len(parts) < 2:
                    raise PyDeclareError("std::array needs a size: '{}'".format(spelling))
                size = parts[1]
            return CppType(CONTAINER, elem=elem, container=kind, size=size)
        # Some other template: opaque
        return CppType(OPAQUE, s)

    return CppType(OPAQUE, s)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def _rank(t):
    if t.kind != FUND:
        return None
    if t.name in INTEGRAL_RANKS:
        return INTEGRAL_RANKS[t.name]
    return FLOATING_RANKS.get(t.name)


def int_op_type(a, b):
    """The scalar type an integer operation on *a* and *b* is carried out in.

    Mirrors ROOT::Internal::PyDeclare::IntOpType: Python has no unsigned
    integers, so a mix of a signed and an unsigned operand is worked out in a
    signed 64-bit type rather than converting the signed side.
    """
    sa, sb = a.scalar(), b.scalar()
    if not (sa.is_integral and sb.is_integral):
        return promote_scalar(sa, sb)
    if sa.is_signed or sb.is_signed:
        return LLONG_T
    return promote_scalar(sa, sb)


def acc_type(elem):
    """The scalar type numpy accumulates a sum or product of *elem* into."""
    if not elem.is_integral:
        return elem
    return LLONG_T if (elem.is_signed or elem.is_bool) else ULLONG_T


def promote_scalar(a, b):
    """Usual-arithmetic-conversions-ish result of combining two scalar types."""
    if a.kind == UNKNOWN or b.kind == UNKNOWN:
        return UNKNOWN_T
    if a.kind != FUND or b.kind != FUND:
        return UNKNOWN_T
    ra, rb = _rank(a), _rank(b)
    if ra is None or rb is None:
        return UNKNOWN_T
    # bool and char never survive arithmetic: Python/C++ both promote to int
    winner = a if ra >= rb else b
    if _rank(winner) < INTEGRAL_RANKS["int"]:
        return INT_T
    return winner


def promote(a, b):
    """Result type of an elementwise binary operation between a and b."""
    if a.kind == UNKNOWN or b.kind == UNKNOWN:
        return UNKNOWN_T
    a_cont, b_cont = a.is_container, b.is_container
    if not a_cont and not b_cont:
        return promote_scalar(a, b)
    elem = promote_scalar(a.scalar(), b.scalar())
    # The result of mixing containers is always an RVec: that is what all the
    # ROOT::VecOps operators return.
    return CppType(CONTAINER, elem=elem, container="rvec")


def with_scalar(t, scalar_type):
    """Same shape as t, but with the given scalar/element type."""
    if t.is_container:
        return CppType(CONTAINER, elem=scalar_type, container="rvec")
    return scalar_type


def bool_like(t):
    """bool for scalars, RVec<bool> for containers -- the type of a comparison."""
    return with_scalar(t, BOOL_T)


# ---------------------------------------------------------------------------
# numpy dtype <-> C++
# ---------------------------------------------------------------------------

NUMPY_TO_CPP = {
    "bool": "bool",
    "bool_": "bool",
    "int8": "char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long",
    "uint64": "unsigned long",
    "intp": "long",
    "float32": "float",
    "float64": "double",
    "float_": "double",
    "double": "double",
    "single": "float",
    "int_": "long",
    "intc": "int",
    "uintc": "unsigned int",
    "longlong": "long long",
}
