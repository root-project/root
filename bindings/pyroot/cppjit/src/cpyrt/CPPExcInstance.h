#ifndef CPYRT_CPPEXCINSTANCE_H
#define CPYRT_CPPEXCINSTANCE_H

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// cppjit::cpyrt::CPPExceptionInstance                                      //
//                                                                          //
// Python-side proxy, encapsulaties a C++ exception object.                 //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

namespace cppjit::cpyrt {

class CPPExcInstance {
public:
  PyBaseExceptionObject fBase;
  PyObject* fCppInstance;
  PyObject* fTopMessage;
};

//- object proxy type and type verification ----------------------------------
CPYRT_IMPORT PyTypeObject CPPExcInstance_Type;

template <typename T> inline bool CPPExcInstance_Check(T* object) {
  return object && PyObject_TypeCheck(object, &CPPExcInstance_Type);
}

template <typename T> inline bool CPPExcInstance_CheckExact(T* object) {
  return object && Py_TYPE(object) == &CPPExcInstance_Type;
}

} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPEXCINSTANCE_H
