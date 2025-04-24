#ifndef CPYCPPYY_LOWLEVELVIEWS_H
#define CPYCPPYY_LOWLEVELVIEWS_H

// Bindings
#include "Dimensions.h"

// Standard
#include <complex>
#include <stddef.h>
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
#include <cstddef>
#endif


namespace CPyCppyy {

class Converter;

class LowLevelView {
public:
    enum EFlags {
        kDefault     = 0x0000,
        kIsCppArray  = 0x0001,    // allocated with new[]
        kIsFixed     = 0x0002,    // fixed size array (assumed flat)
        kIsOwner     = 0x0004 };  // Python owns

public:
    PyObject_HEAD
    Py_buffer   fBufInfo;
    void**      fBuf;
    Converter*  fConverter;
    Converter*  fElemCnv;

    typedef LowLevelView* (*Creator_t)(void*, cdims_t);
    Creator_t   fCreator;    // for slicing, which requires copying

public:
    void* get_buf() { return fBuf ? *fBuf : fBufInfo.buf; }
    void  set_buf(void** buf) { fBuf = buf; fBufInfo.buf = get_buf(); }

    bool resize(size_t sz);
};

#define CPPYY_DECL_VIEW_CREATOR(type)                                        \
    PyObject* CreateLowLevelView(type*,  cdims_t shape);                     \
    PyObject* CreateLowLevelView(type**, cdims_t shape)

CPPYY_DECL_VIEW_CREATOR(bool);
CPPYY_DECL_VIEW_CREATOR(char);
CPPYY_DECL_VIEW_CREATOR(signed char);
CPPYY_DECL_VIEW_CREATOR(unsigned char);
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
CPPYY_DECL_VIEW_CREATOR(std::byte);
#endif
PyObject* CreateLowLevelView_i8(int8_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i8(int8_t**, cdims_t shape);
PyObject* CreateLowLevelView_i8(uint8_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i8(uint8_t**, cdims_t shape);
PyObject* CreateLowLevelView_i16(int16_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i16(int16_t**, cdims_t shape);
PyObject* CreateLowLevelView_i16(uint16_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i16(uint16_t**, cdims_t shape);
PyObject* CreateLowLevelView_i32(int32_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i32(int32_t**, cdims_t shape);
PyObject* CreateLowLevelView_i32(uint32_t*,  cdims_t shape);
PyObject* CreateLowLevelView_i32(uint32_t**, cdims_t shape);

CPPYY_DECL_VIEW_CREATOR(short);
CPPYY_DECL_VIEW_CREATOR(unsigned short);
CPPYY_DECL_VIEW_CREATOR(int);
CPPYY_DECL_VIEW_CREATOR(unsigned int);
CPPYY_DECL_VIEW_CREATOR(long);
CPPYY_DECL_VIEW_CREATOR(unsigned long);
CPPYY_DECL_VIEW_CREATOR(long long);
CPPYY_DECL_VIEW_CREATOR(unsigned long long);
CPPYY_DECL_VIEW_CREATOR(float);
CPPYY_DECL_VIEW_CREATOR(double);
CPPYY_DECL_VIEW_CREATOR(long double);
CPPYY_DECL_VIEW_CREATOR(std::complex<float>);
CPPYY_DECL_VIEW_CREATOR(std::complex<double>);
CPPYY_DECL_VIEW_CREATOR(std::complex<int>);
CPPYY_DECL_VIEW_CREATOR(std::complex<long>);

PyObject* CreateLowLevelViewString(char**, cdims_t shape);
PyObject* CreateLowLevelViewString(const char**, cdims_t shape);

inline PyObject* CreatePointerView(void* ptr, cdims_t shape = 0) {
    return CreateLowLevelView((uintptr_t*)ptr, shape);
}

//- low level view type and type verification --------------------------------
extern PyTypeObject LowLevelView_Type;

template<typename T>
inline bool LowLevelView_Check(T* object)
{
    return object && PyObject_TypeCheck(object, &LowLevelView_Type);
}

template<typename T>
inline bool LowLevelView_CheckExact(T* object)
{
    return object && Py_TYPE(object) == &LowLevelView_Type;
}

} // namespace CPyCppyy

#endif // !CPYCPPYY_LOWLEVELVIEWS_H
