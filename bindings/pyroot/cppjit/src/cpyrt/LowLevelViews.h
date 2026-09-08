#ifndef CPYRT_LOWLEVELVIEWS_H
#define CPYRT_LOWLEVELVIEWS_H

// Bindings
#include "Dimensions.h"

// Standard
#include <complex>
#include <cstddef>
#include <stddef.h>

namespace cppjit::cpyrt {

class Converter;

class LowLevelView {
public:
  enum EFlags {
    kDefault = 0x0000,
    kIsCppArray = 0x0001, // allocated with new[]
    kIsFixed = 0x0002,    // fixed size array (assumed flat)
    kIsOwner = 0x0004
  }; // Python owns

public:
  PyObject_HEAD Py_buffer fBufInfo;
  void** fBuf;
  Converter* fConverter;
  Converter* fElemCnv;

  typedef LowLevelView* (*Creator_t)(void*, cdims_t);
  Creator_t fCreator; // for slicing, which requires copying

public:
  void* get_buf() { return fBuf ? *fBuf : fBufInfo.buf; }
  void set_buf(void** buf) {
    fBuf = buf;
    fBufInfo.buf = get_buf();
  }

  bool resize(size_t sz);
};

#define CPPJIT_DECL_VIEW_CREATOR(type)                                         \
  PyObject* CreateLowLevelView(type*, cdims_t shape);                          \
  PyObject* CreateLowLevelView(type**, cdims_t shape)

CPPJIT_DECL_VIEW_CREATOR(bool);
CPPJIT_DECL_VIEW_CREATOR(char);
CPPJIT_DECL_VIEW_CREATOR(signed char);
CPPJIT_DECL_VIEW_CREATOR(unsigned char);
CPPJIT_DECL_VIEW_CREATOR(std::byte);
PyObject* CreateLowLevelView_i8(int8_t*, cdims_t shape);
PyObject* CreateLowLevelView_i8(int8_t**, cdims_t shape);
PyObject* CreateLowLevelView_i8(uint8_t*, cdims_t shape);
PyObject* CreateLowLevelView_i8(uint8_t**, cdims_t shape);
CPPJIT_DECL_VIEW_CREATOR(short);
CPPJIT_DECL_VIEW_CREATOR(unsigned short);
CPPJIT_DECL_VIEW_CREATOR(int);
CPPJIT_DECL_VIEW_CREATOR(unsigned int);
CPPJIT_DECL_VIEW_CREATOR(long);
CPPJIT_DECL_VIEW_CREATOR(unsigned long);
CPPJIT_DECL_VIEW_CREATOR(long long);
CPPJIT_DECL_VIEW_CREATOR(unsigned long long);
CPPJIT_DECL_VIEW_CREATOR(float);
CPPJIT_DECL_VIEW_CREATOR(double);
CPPJIT_DECL_VIEW_CREATOR(long double);
CPPJIT_DECL_VIEW_CREATOR(std::complex<float>);
CPPJIT_DECL_VIEW_CREATOR(std::complex<double>);
CPPJIT_DECL_VIEW_CREATOR(std::complex<int>);
CPPJIT_DECL_VIEW_CREATOR(std::complex<long>);

PyObject* CreateLowLevelViewString(char**, cdims_t shape);
PyObject* CreateLowLevelViewString(const char**, cdims_t shape);

inline PyObject* CreatePointerView(void* ptr, cdims_t shape = 0) {
  return CreateLowLevelView((uintptr_t*)ptr, shape);
}

//- low level view type and type verification --------------------------------
extern PyTypeObject LowLevelView_Type;

template <typename T> inline bool LowLevelView_Check(T* object) {
  return object && PyObject_TypeCheck(object, &LowLevelView_Type);
}

template <typename T> inline bool LowLevelView_CheckExact(T* object) {
  return object && Py_TYPE(object) == &LowLevelView_Type;
}

} // namespace cppjit::cpyrt

#endif // !CPYRT_LOWLEVELVIEWS_H
