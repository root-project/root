#ifndef CPYRT_PYSTRINGS_H
#define CPYRT_PYSTRINGS_H

namespace cppjit::cpyrt {

// python strings kept for performance reasons

// X(variable, string): each entry declares PyStrings::variable, holding the
// interned python string for "string"
#define CPYRT_PYSTRINGS(X)                                                     \
  X(gAssign, "__assign__")                                                     \
  X(gBases, "__bases__")                                                       \
  X(gBase, "__base__")                                                         \
  X(gContains, "contains")                                                     \
  X(gCopy, "copy")                                                             \
  X(gCppBool, "__cpp_bool__")                                                  \
  X(gCppName, "__cpp_name__")                                                  \
  X(gAnnotations, "__annotations__")                                           \
  X(gCastCpp, "__cast_cpp__")                                                  \
  X(gCType, "__ctype__")                                                       \
  X(gDeref, "__deref__")                                                       \
  X(gPreInc, "__preinc__")                                                     \
  X(gPostInc, "__postinc__")                                                   \
  X(gDict, "__dict__")                                                         \
  X(gEmptyString, "")                                                          \
  X(gEq, "__eq__")                                                             \
  X(gFollow, "__follow__")                                                     \
  X(gHasValue, "has_value")                                                    \
  X(gGetItem, "__getitem__")                                                   \
  X(gGetNoCheck, "_getitem__unchecked")                                        \
  X(gSetItem, "__setitem__")                                                   \
  X(gInit, "__init__")                                                         \
  X(gIter, "__iter__")                                                         \
  X(gLen, "__len__")                                                           \
  X(gLifeLine, "__lifeline")                                                   \
  X(gModule, "__module__")                                                     \
  X(gMRO, "__mro__")                                                           \
  X(gName, "__name__")                                                         \
  X(gNe, "__ne__")                                                             \
  X(gRepr, "__repr__")                                                         \
  X(gCppRepr, "__cpp_repr")                                                    \
  X(gStr, "__str__")                                                           \
  X(gCppStr, "__cpp_str")                                                      \
  X(gTypeCode, "typecode")                                                     \
  X(gCTypesType, "_type_")                                                     \
                                                                               \
  X(gUnderlying, "__underlying")                                               \
  X(gRealInit, "__real_init")                                                  \
                                                                               \
  X(gAdd, "__add__")                                                           \
  X(gSub, "__sub__")                                                           \
  X(gMul, "__mul__")                                                           \
  X(gDiv, "__truediv__")                                                       \
                                                                               \
  X(gLShift, "__lshift__")                                                     \
  X(gLShiftC, "__lshiftc__")                                                   \
                                                                               \
  X(gAt, "at")                                                                 \
  X(gBegin, "begin")                                                           \
  X(gEnd, "end")                                                               \
  X(gFirst, "first")                                                           \
  X(gSecond, "second")                                                         \
  X(gSize, "size")                                                             \
  X(gTemplate, "Template")                                                     \
  X(gVectorAt, "_vector__at")                                                  \
  X(gInsert, "insert")                                                         \
  X(gValueType, "value_type")                                                  \
  X(gValueTypePtr, "_value_type")                                              \
  X(gValueSize, "value_size")                                                  \
                                                                               \
  X(gCppReal, "__cpp_real")                                                    \
  X(gCppImag, "__cpp_imag")                                                    \
                                                                               \
  X(gThisModule, "cppjit")                                                     \
                                                                               \
  X(gDispInit, "_init_dispatchptr")                                            \
  X(gDispGet, "_get_dispatch")                                                 \
                                                                               \
  X(gExPythonize, "__cppjit_explicit_pythonize__")                             \
  X(gPythonize, "__cppjit_pythonize__")                                        \
                                                                               \
  X(gArray, "__array__")                                                       \
  X(gDType, "dtype")                                                           \
  X(gFromBuffer, "frombuffer")

namespace PyStrings {

#define CPYRT_DECLARE_PYSTRING(var, str) extern PyObject* var;
CPYRT_PYSTRINGS(CPYRT_DECLARE_PYSTRING)
#undef CPYRT_DECLARE_PYSTRING

} // namespace PyStrings

bool CreatePyStrings();
PyObject* DestroyPyStrings();

} // namespace cppjit::cpyrt

#endif // !CPYRT_PYSTRINGS_H
