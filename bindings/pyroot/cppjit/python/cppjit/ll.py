"""Low-level utilities, to be used for "emergencies only"."""

import ctypes
import sys
import warnings

import cppjit

try:
    import __pypy__

    del __pypy__
    ispypy = True
except ImportError:
    ispypy = False

__all__ = [
    "argv",
    "argc",
    "cast",
    "static_cast",
    "reinterpret_cast",
    "dynamic_cast",
    "malloc",
    "free",
    "array_new",
    "array_delete",
    "bind_value",
    "as_std_any",
    "signals_as_exception",
    "set_signals_as_exception",
    "FatalError",
    "BusError",
    "SegmentationViolation",
    "IllegalInstruction",
    "AbortSignal",
]


# convenience functions to create C-style argv/argc
def argv():
    argc = len(sys.argv)  # noqa: F841
    cargsv = (ctypes.c_char_p * len(sys.argv))(*(x.encode() for x in sys.argv))
    return ctypes.POINTER(ctypes.c_char_p)(cargsv)


def argc():
    return len(sys.argv)


def bind_value(type_name, address, dims=None):
    """Read a value of the C++ type `type_name` from `address`.

    Hands back the object that cppjit's converters would produce reading the
    same memory themselves: a bound proxy for a class type, a Python value
    for a builtin. Meant for code that has a type name and an address in hand
    but no C++ entity to read them from, such as a framework exposing its own
    data description.

    `address` is an integer, as returned by `addressof`. If `dims` is given,
    it is a non-empty sequence of non-negative integers describing the shape
    of an array starting at `address`, and the type name must end in `[]` or
    `*` (scalar converters ignore `dims`); a `LowLevelView` of that shape is
    returned.

        v = ll.bind_value('double', addr)             # a float
        a = ll.bind_value('double[]', addr, (2, 3))   # a 2x3 view

    No lifetime or bounds checking is or can be done: the caller vouches for
    the address, the type and the shape.
    """
    return cppjit._backend.bind_value(type_name, address, dims)


def as_std_any(obj):
    """Wrap a reference to the Python object `obj` in a `std::any`.

    For C++ code that has to keep `obj` alive, such as a data source reading
    from the memory of a NumPy array: the `std::any`, and each copy that C++
    makes of it, holds a reference to `obj`, which is released when that copy
    is destroyed. The reference count is changed with the GIL taken, so C++
    may do this from any thread.

        keepalive = ll.as_std_any(array)
    """
    return cppjit._backend.as_std_any(obj)


# import low-level python converters
for _name in ["addressof", "as_cobject", "as_capsule", "as_ctypes", "as_memoryview"]:
    try:
        exec("%s = cppjit._backend.%s" % (_name, _name))
        __all__.append(_name)
    except AttributeError:
        pass
del _name

# create low-level helpers once
if not hasattr(cppjit.gbl, "__cppjit_internal") or not hasattr(
    cppjit.gbl.__cppjit_internal, "cppjit_cast"
):
    cppjit.cppdef("""namespace __cppjit_internal {
    // type casting
      template<typename T, typename U>
      T cppjit_cast(U val) { return (T)val; }

      template<typename T, typename U>
      T cppjit_static_cast(U val) { return static_cast<T>(val); }

      template<typename T, typename U>
      T cppjit_reinterpret_cast(U val) { return reinterpret_cast<T>(val); }

      template<typename T, typename S>
      T* cppjit_dynamic_cast(S* obj) { return dynamic_cast<T*>(obj); }

    // memory allocation/free-ing
      template<typename T>
      T* cppjit_malloc(size_t count=1) { return (T*)malloc(sizeof(T*)*count); }

      template<typename T>
      T* cppjit_array_new(size_t count) { return new T[count]; }

      template<typename T>
      void cppjit_array_delete(T* ptr) { delete[] ptr; }
    }""")


# helper for sizing arrays
class ArraySizer(object):
    def __init__(self, func):
        self.func = func

    def __getitem__(self, t):
        self.array_type = t
        return self

    def __call__(self, size, managed=False):
        res = self.func[self.array_type](size)
        try:
            res.reshape((size,) + res.shape[1:])
            if managed:
                res.__python_owns__ = True
        except AttributeError:
            res.__reshape__((size,))
            if managed:
                warnings.warn("managed low-level arrays of instances not supported")
        return res


class CArraySizer(ArraySizer):
    def __call__(self, size, managed=False):
        res = ArraySizer.__call__(self, size, managed)
        res.__cpp_array__ = False
        return res


# import casting helpers
cast = cppjit.gbl.__cppjit_internal.cppjit_cast
static_cast = cppjit.gbl.__cppjit_internal.cppjit_static_cast
reinterpret_cast = cppjit.gbl.__cppjit_internal.cppjit_reinterpret_cast
dynamic_cast = cppjit.gbl.__cppjit_internal.cppjit_dynamic_cast

# import memory allocation/free-ing helpers
malloc = CArraySizer(cppjit.gbl.__cppjit_internal.cppjit_malloc)
free = cppjit.gbl.free  # for symmetry
array_new = ArraySizer(cppjit.gbl.__cppjit_internal.cppjit_array_new)
array_delete = cppjit.gbl.__cppjit_internal.cppjit_array_delete

# signals as exceptions
if not ispypy:
    FatalError = cppjit._backend.FatalError
    BusError = cppjit._backend.BusError
    SegmentationViolation = cppjit._backend.SegmentationViolation
    IllegalInstruction = cppjit._backend.IllegalInstruction
    AbortSignal = cppjit._backend.AbortSignal

    class signals_as_exception:
        def __enter__(self):
            cppjit._backend.SetGlobalSignalPolicy(1)

        def __exit__(self, type, value, traceback):
            cppjit._backend.SetGlobalSignalPolicy(0)

    set_signals_as_exception = cppjit._backend.SetGlobalSignalPolicy

else:

    class FatalError(Exception):
        pass

    class BusError(FatalError):
        pass

    class SegmentationViolation(FatalError):
        pass

    class IllegalInstruction(FatalError):
        pass

    class AbortSignal(FatalError):
        pass

    class signals_as_exception:
        def __enter__(self):
            pass  # not yet implemented

        def __exit__(self, type, value, traceback):
            pass  # not yet implemented

    def set_signals_as_exception(seton):
        return False


del ispypy
