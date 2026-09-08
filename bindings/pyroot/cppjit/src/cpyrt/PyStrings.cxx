// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "PyStrings.h"

//- data _____________________________________________________________________
PyObject* cpyrt::PyStrings::gAssign = nullptr;
PyObject* cpyrt::PyStrings::gBases = nullptr;
PyObject* cpyrt::PyStrings::gBase = nullptr;
PyObject* cpyrt::PyStrings::gContains = nullptr;
PyObject* cpyrt::PyStrings::gCopy = nullptr;
PyObject* cpyrt::PyStrings::gCppBool = nullptr;
PyObject* cpyrt::PyStrings::gCppName = nullptr;
PyObject* cpyrt::PyStrings::gAnnotations = nullptr;
PyObject* cpyrt::PyStrings::gCastCpp = nullptr;
PyObject* cpyrt::PyStrings::gCType = nullptr;
PyObject* cpyrt::PyStrings::gDeref = nullptr;
PyObject* cpyrt::PyStrings::gPreInc = nullptr;
PyObject* cpyrt::PyStrings::gPostInc = nullptr;
PyObject* cpyrt::PyStrings::gDict = nullptr;
PyObject* cpyrt::PyStrings::gEmptyString = nullptr;
PyObject* cpyrt::PyStrings::gEq = nullptr;
PyObject* cpyrt::PyStrings::gFollow = nullptr;
PyObject* cpyrt::PyStrings::gGetItem = nullptr;
PyObject* cpyrt::PyStrings::gGetNoCheck = nullptr;
PyObject* cpyrt::PyStrings::gSetItem = nullptr;
PyObject* cpyrt::PyStrings::gInit = nullptr;
PyObject* cpyrt::PyStrings::gIter = nullptr;
PyObject* cpyrt::PyStrings::gLen = nullptr;
PyObject* cpyrt::PyStrings::gLifeLine = nullptr;
PyObject* cpyrt::PyStrings::gModule = nullptr;
PyObject* cpyrt::PyStrings::gMRO = nullptr;
PyObject* cpyrt::PyStrings::gName = nullptr;
PyObject* cpyrt::PyStrings::gNe = nullptr;
PyObject* cpyrt::PyStrings::gRepr = nullptr;
PyObject* cpyrt::PyStrings::gCppRepr = nullptr;
PyObject* cpyrt::PyStrings::gStr = nullptr;
PyObject* cpyrt::PyStrings::gCppStr = nullptr;
PyObject* cpyrt::PyStrings::gTypeCode = nullptr;
PyObject* cpyrt::PyStrings::gCTypesType = nullptr;

PyObject* cpyrt::PyStrings::gUnderlying = nullptr;
PyObject* cpyrt::PyStrings::gRealInit = nullptr;

PyObject* cpyrt::PyStrings::gAdd = nullptr;
PyObject* cpyrt::PyStrings::gSub = nullptr;
PyObject* cpyrt::PyStrings::gMul = nullptr;
PyObject* cpyrt::PyStrings::gDiv = nullptr;

PyObject* cpyrt::PyStrings::gLShift = nullptr;
PyObject* cpyrt::PyStrings::gLShiftC = nullptr;

PyObject* cpyrt::PyStrings::gAt = nullptr;
PyObject* cpyrt::PyStrings::gBegin = nullptr;
PyObject* cpyrt::PyStrings::gEnd = nullptr;
PyObject* cpyrt::PyStrings::gFirst = nullptr;
PyObject* cpyrt::PyStrings::gSecond = nullptr;
PyObject* cpyrt::PyStrings::gSize = nullptr;
PyObject* cpyrt::PyStrings::gTemplate = nullptr;
PyObject* cpyrt::PyStrings::gVectorAt = nullptr;
PyObject* cpyrt::PyStrings::gInsert = nullptr;
PyObject* cpyrt::PyStrings::gValueType = nullptr;
PyObject* cpyrt::PyStrings::gValueTypePtr = nullptr;
PyObject* cpyrt::PyStrings::gValueSize = nullptr;

PyObject* cpyrt::PyStrings::gCppReal = nullptr;
PyObject* cpyrt::PyStrings::gCppImag = nullptr;

PyObject* cpyrt::PyStrings::gThisModule = nullptr;

PyObject* cpyrt::PyStrings::gDispInit = nullptr;
PyObject* cpyrt::PyStrings::gDispGet = nullptr;

PyObject* cpyrt::PyStrings::gExPythonize = nullptr;
PyObject* cpyrt::PyStrings::gPythonize = nullptr;

PyObject* cpyrt::PyStrings::gArray = nullptr;
PyObject* cpyrt::PyStrings::gDType = nullptr;
PyObject* cpyrt::PyStrings::gFromBuffer = nullptr;

//-----------------------------------------------------------------------------
#define CPPJIT_INITIALIZE_STRING(var, str)                                     \
  if (!(PyStrings::var = cpyrt_PyText_InternFromString((char*)#str)))          \
  return false

bool cpyrt::CreatePyStrings() {
  // Build cache of commonly used python strings (the cache is python intern, so
  // all strings are shared python-wide, not just in cppjit).
  CPPJIT_INITIALIZE_STRING(gAssign, __assign__);
  CPPJIT_INITIALIZE_STRING(gBases, __bases__);
  CPPJIT_INITIALIZE_STRING(gBase, __base__);
  CPPJIT_INITIALIZE_STRING(gContains, contains);
  CPPJIT_INITIALIZE_STRING(gCopy, copy);
  CPPJIT_INITIALIZE_STRING(gCppBool, __cpp_bool__);
  CPPJIT_INITIALIZE_STRING(gCppName, __cpp_name__);
  CPPJIT_INITIALIZE_STRING(gAnnotations, __annotations__);
  CPPJIT_INITIALIZE_STRING(gCastCpp, __cast_cpp__);
  CPPJIT_INITIALIZE_STRING(gCType, __ctype__);
  CPPJIT_INITIALIZE_STRING(gDeref, __deref__);
  CPPJIT_INITIALIZE_STRING(gPreInc, __preinc__);
  CPPJIT_INITIALIZE_STRING(gPostInc, __postinc__);
  CPPJIT_INITIALIZE_STRING(gDict, __dict__);
  if (!(PyStrings::gEmptyString = cpyrt_PyText_FromString((char*)"")))
    return false;
  CPPJIT_INITIALIZE_STRING(gEq, __eq__);
  CPPJIT_INITIALIZE_STRING(gFollow, __follow__);
  CPPJIT_INITIALIZE_STRING(gGetItem, __getitem__);
  CPPJIT_INITIALIZE_STRING(gGetNoCheck, _getitem__unchecked);
  CPPJIT_INITIALIZE_STRING(gSetItem, __setitem__);
  CPPJIT_INITIALIZE_STRING(gInit, __init__);
  CPPJIT_INITIALIZE_STRING(gIter, __iter__);
  CPPJIT_INITIALIZE_STRING(gLen, __len__);
  CPPJIT_INITIALIZE_STRING(gLifeLine, __lifeline);
  CPPJIT_INITIALIZE_STRING(gModule, __module__);
  CPPJIT_INITIALIZE_STRING(gMRO, __mro__);
  CPPJIT_INITIALIZE_STRING(gName, __name__);
  CPPJIT_INITIALIZE_STRING(gNe, __ne__);
  CPPJIT_INITIALIZE_STRING(gRepr, __repr__);
  CPPJIT_INITIALIZE_STRING(gCppRepr, __cpp_repr);
  CPPJIT_INITIALIZE_STRING(gStr, __str__);
  CPPJIT_INITIALIZE_STRING(gCppStr, __cpp_str);
  CPPJIT_INITIALIZE_STRING(gTypeCode, typecode);
  CPPJIT_INITIALIZE_STRING(gCTypesType, _type_);

  CPPJIT_INITIALIZE_STRING(gUnderlying, __underlying);
  CPPJIT_INITIALIZE_STRING(gRealInit, __real_init);

  CPPJIT_INITIALIZE_STRING(gAdd, __add__);
  CPPJIT_INITIALIZE_STRING(gSub, __sub__);
  CPPJIT_INITIALIZE_STRING(gMul, __mul__);
  CPPJIT_INITIALIZE_STRING(gDiv, CPPJIT__div__);

  CPPJIT_INITIALIZE_STRING(gLShift, __lshift__);
  CPPJIT_INITIALIZE_STRING(gLShiftC, __lshiftc__);

  CPPJIT_INITIALIZE_STRING(gAt, at);
  CPPJIT_INITIALIZE_STRING(gBegin, begin);
  CPPJIT_INITIALIZE_STRING(gEnd, end);
  CPPJIT_INITIALIZE_STRING(gFirst, first);
  CPPJIT_INITIALIZE_STRING(gSecond, second);
  CPPJIT_INITIALIZE_STRING(gSize, size);
  CPPJIT_INITIALIZE_STRING(gTemplate, Template);
  CPPJIT_INITIALIZE_STRING(gVectorAt, _vector__at);
  CPPJIT_INITIALIZE_STRING(gInsert, insert);
  CPPJIT_INITIALIZE_STRING(gValueType, value_type);
  CPPJIT_INITIALIZE_STRING(gValueTypePtr, _value_type);
  CPPJIT_INITIALIZE_STRING(gValueSize, value_size);

  CPPJIT_INITIALIZE_STRING(gCppReal, __cpp_real);
  CPPJIT_INITIALIZE_STRING(gCppImag, __cpp_imag);

  CPPJIT_INITIALIZE_STRING(gThisModule, cppjit);

  CPPJIT_INITIALIZE_STRING(gDispInit, _init_dispatchptr);
  CPPJIT_INITIALIZE_STRING(gDispGet, _get_dispatch);

  CPPJIT_INITIALIZE_STRING(gExPythonize, __cppjit_explicit_pythonize__);
  CPPJIT_INITIALIZE_STRING(gPythonize, __cppjit_pythonize__);

  CPPJIT_INITIALIZE_STRING(gArray, __array__);
  CPPJIT_INITIALIZE_STRING(gDType, dtype);
  CPPJIT_INITIALIZE_STRING(gFromBuffer, frombuffer);

  return true;
}

//-----------------------------------------------------------------------------
PyObject* cpyrt::DestroyPyStrings() {
  // Remove all cached python strings.
  Py_DECREF(PyStrings::gBases);
  PyStrings::gBases = nullptr;
  Py_DECREF(PyStrings::gBase);
  PyStrings::gBase = nullptr;
  Py_DECREF(PyStrings::gContains);
  PyStrings::gContains = nullptr;
  Py_DECREF(PyStrings::gCopy);
  PyStrings::gCopy = nullptr;
  Py_DECREF(PyStrings::gCppBool);
  PyStrings::gCppBool = nullptr;
  Py_DECREF(PyStrings::gCppName);
  PyStrings::gCppName = nullptr;
  Py_DECREF(PyStrings::gAnnotations);
  PyStrings::gAnnotations = nullptr;
  Py_DECREF(PyStrings::gCType);
  PyStrings::gCType = nullptr;
  Py_DECREF(PyStrings::gDeref);
  PyStrings::gDeref = nullptr;
  Py_DECREF(PyStrings::gPreInc);
  PyStrings::gPreInc = nullptr;
  Py_DECREF(PyStrings::gPostInc);
  PyStrings::gPostInc = nullptr;
  Py_DECREF(PyStrings::gDict);
  PyStrings::gDict = nullptr;
  Py_DECREF(PyStrings::gEmptyString);
  PyStrings::gEmptyString = nullptr;
  Py_DECREF(PyStrings::gEq);
  PyStrings::gEq = nullptr;
  Py_DECREF(PyStrings::gFollow);
  PyStrings::gFollow = nullptr;
  Py_DECREF(PyStrings::gGetItem);
  PyStrings::gGetItem = nullptr;
  Py_DECREF(PyStrings::gGetNoCheck);
  PyStrings::gGetNoCheck = nullptr;
  Py_DECREF(PyStrings::gSetItem);
  PyStrings::gSetItem = nullptr;
  Py_DECREF(PyStrings::gInit);
  PyStrings::gInit = nullptr;
  Py_DECREF(PyStrings::gIter);
  PyStrings::gIter = nullptr;
  Py_DECREF(PyStrings::gLen);
  PyStrings::gLen = nullptr;
  Py_DECREF(PyStrings::gLifeLine);
  PyStrings::gLifeLine = nullptr;
  Py_DECREF(PyStrings::gModule);
  PyStrings::gModule = nullptr;
  Py_DECREF(PyStrings::gMRO);
  PyStrings::gMRO = nullptr;
  Py_DECREF(PyStrings::gName);
  PyStrings::gName = nullptr;
  Py_DECREF(PyStrings::gNe);
  PyStrings::gNe = nullptr;
  Py_DECREF(PyStrings::gTypeCode);
  PyStrings::gTypeCode = nullptr;
  Py_DECREF(PyStrings::gCTypesType);
  PyStrings::gCTypesType = nullptr;

  Py_DECREF(PyStrings::gUnderlying);
  PyStrings::gUnderlying = nullptr;
  Py_DECREF(PyStrings::gRealInit);
  PyStrings::gRealInit = nullptr;

  Py_DECREF(PyStrings::gAdd);
  PyStrings::gAdd = nullptr;
  Py_DECREF(PyStrings::gSub);
  PyStrings::gSub = nullptr;
  Py_DECREF(PyStrings::gMul);
  PyStrings::gMul = nullptr;
  Py_DECREF(PyStrings::gDiv);
  PyStrings::gDiv = nullptr;

  Py_DECREF(PyStrings::gLShift);
  PyStrings::gLShift = nullptr;
  Py_DECREF(PyStrings::gLShiftC);
  PyStrings::gLShiftC = nullptr;

  Py_DECREF(PyStrings::gAt);
  PyStrings::gAt = nullptr;
  Py_DECREF(PyStrings::gBegin);
  PyStrings::gBegin = nullptr;
  Py_DECREF(PyStrings::gEnd);
  PyStrings::gEnd = nullptr;
  Py_DECREF(PyStrings::gFirst);
  PyStrings::gFirst = nullptr;
  Py_DECREF(PyStrings::gSecond);
  PyStrings::gSecond = nullptr;
  Py_DECREF(PyStrings::gSize);
  PyStrings::gSize = nullptr;
  Py_DECREF(PyStrings::gTemplate);
  PyStrings::gTemplate = nullptr;
  Py_DECREF(PyStrings::gVectorAt);
  PyStrings::gVectorAt = nullptr;
  Py_DECREF(PyStrings::gInsert);
  PyStrings::gInsert = nullptr;
  Py_DECREF(PyStrings::gValueType);
  PyStrings::gValueType = nullptr;
  Py_DECREF(PyStrings::gValueTypePtr);
  PyStrings::gValueTypePtr = nullptr;
  Py_DECREF(PyStrings::gValueSize);
  PyStrings::gValueSize = nullptr;

  Py_DECREF(PyStrings::gCppReal);
  PyStrings::gCppReal = nullptr;
  Py_DECREF(PyStrings::gCppImag);
  PyStrings::gCppImag = nullptr;

  Py_DECREF(PyStrings::gThisModule);
  PyStrings::gThisModule = nullptr;

  Py_DECREF(PyStrings::gDispInit);
  PyStrings::gDispInit = nullptr;
  Py_DECREF(PyStrings::gDispGet);
  PyStrings::gDispGet = nullptr;

  Py_DECREF(PyStrings::gExPythonize);
  PyStrings::gExPythonize = nullptr;
  Py_DECREF(PyStrings::gPythonize);
  PyStrings::gPythonize = nullptr;

  Py_DECREF(PyStrings::gArray);
  PyStrings::gArray = nullptr;
  Py_DECREF(PyStrings::gDType);
  PyStrings::gDType = nullptr;
  Py_DECREF(PyStrings::gFromBuffer);
  PyStrings::gFromBuffer = nullptr;

  Py_RETURN_NONE;
}
