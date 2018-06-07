// Bindings
#include "CPyCppyy.h"
#include "DeclareConverters.h"
#include "CallContext.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CustomPyTypes.h"
#include "LowLevelViews.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TupleOfInstances.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <limits.h>
#include <stddef.h>      // for ptrdiff_t
#include <string.h>
#include <utility>
#include <sstream>
#if __cplusplus > 201402L
#include <string_view>
#endif

// FIXME: Should refer to CPyCppyy::Parameter in the code.
#ifdef R__CXXMODULES
  #define Parameter CPyCppyy::Parameter
#endif


//- data _____________________________________________________________________
namespace CPyCppyy {

// factories
    typedef Converter* (*cf_t)(long size);
    typedef std::map<std::string, cf_t> ConvFactories_t;
    static ConvFactories_t gConvFactories;
    extern PyObject* gNullPtrObject;

}

//- pretend-ctypes helpers ---------------------------------------------------
#if PY_VERSION_HEX >= 0x02050000

struct CPyCppyy_tagCDataObject { // non-public (but so far very stable)
    PyObject_HEAD
    char* b_ptr;
};

static inline PyTypeObject* GetCTypesType(const char* name) {
    PyObject* ct = PyImport_ImportModule("ctypes");
    if (!ct) return nullptr;
    PyTypeObject* ct_t = (PyTypeObject*)PyObject_GetAttrString(ct, name);
    Py_DECREF(ct);
    return ct_t;
}

#endif

//- custom helpers to check ranges -------------------------------------------
static inline bool CPyCppyy_PyLong_AsBool(PyObject* pyobject)
{
// range-checking python integer to C++ bool conversion
    long l = PyLong_AsLong(pyobject);
// fail to pass float -> bool; the problem is rounding (0.1 -> 0 -> False)
    if (!(l == 0|| l == 1) || PyFloat_Check(pyobject)) {
        PyErr_SetString(PyExc_ValueError, "boolean value should be bool, or integer 1 or 0");
        return (bool)-1;
    }
    return (bool)l;
}

static inline char CPyCppyy_PyUnicode_AsChar(PyObject* pyobject) {
// python string to C++ char conversion
    return (char)CPyCppyy_PyUnicode_AsString(pyobject)[0];
}

static inline unsigned short CPyCppyy_PyLong_AsUShort(PyObject* pyobject)
{
// range-checking python integer to C++ unsigend short int conversion

// prevent p2.7 silent conversions and do a range check
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        PyErr_SetString(PyExc_TypeError, "unsigned short conversion expects an integer object");
        return (unsigned short)-1;
    }
    long l = PyLong_AsLong(pyobject);
    if (l < 0 || USHRT_MAX < l) {
        PyErr_Format(PyExc_ValueError, "integer %ld out of range for unsigned short", l);
        return (unsigned short)-1;

    }
    return (unsigned short)l;
}

static inline short CPyCppyy_PyLong_AsShort(PyObject* pyobject)
{
// range-checking python integer to C++ short int conversion
// prevent p2.7 silent conversions and do a range check
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        PyErr_SetString(PyExc_TypeError, "short int conversion expects an integer object");
        return (short)-1;
    }
    long l = PyLong_AsLong(pyobject);
    if (l < SHRT_MIN || SHRT_MAX < l) {
        PyErr_Format(PyExc_ValueError, "integer %ld out of range for short int", l);
        return (short)-1;

    }
    return (short)l;
}

static inline int CPyCppyy_PyLong_AsStrictInt(PyObject* pyobject)
{
// strict python integer to C++ integer conversion

// p2.7 and later silently converts floats to long, therefore require this
// check; earlier pythons may raise a SystemError which should be avoided as
// it is confusing
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        PyErr_SetString(PyExc_TypeError, "int/long conversion expects an integer object");
        return -1;
    }
    long l = PyLong_AsLong(pyobject);
    if (l < INT_MIN || INT_MAX < l) {
        PyErr_Format(PyExc_ValueError, "integer %ld out of range for int", l);
        return (int)-1;

    }
    return (int)l;
}

static inline long CPyCppyy_PyLong_AsStrictLong(PyObject* pyobject)
{
// strict python integer to C++ long integer conversion

// prevent float -> long (see CPyCppyy_PyLong_AsStrictInt)
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        PyErr_SetString(PyExc_TypeError, "int/long conversion expects an integer object");
        return (long)-1;
    }
    return (long)PyLong_AsLong(pyobject);
}


//- base converter implementation --------------------------------------------
PyObject* CPyCppyy::Converter::FromMemory(void*)
{
// could happen if no derived class override
    PyErr_SetString(PyExc_TypeError, "C++ type can not be converted from memory");
    return nullptr;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Converter::ToMemory(PyObject*, void*)
{
// could happen if no derived class override
    PyErr_SetString(PyExc_TypeError, "C++ type can not be converted to memory");
    return false;
}


//- helper macro's -----------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONVERTER(name, type, stype, F1, F2, tc)            \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
/* convert <pyobject> to C++ 'type', set arg for call */                     \
    type val = (type)F2(pyobject);                                           \
    if (val == (type)-1 && PyErr_Occurred())                                 \
        return false;                                                        \
    para.fValue.f##name = val;                                               \
    para.fTypeCode = tc;                                                     \
    return true;                                                             \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
    return F1((stype)*((type*)address));                                     \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(PyObject* value, void* address)     \
{                                                                            \
    type s = (type)F2(value);                                                \
    if (s == (type)-1 && PyErr_Occurred())                                   \
        return false;                                                        \
    *((type*)address) = (type)s;                                             \
    return true;                                                             \
}

//----------------------------------------------------------------------------
static inline Int_t ExtractChar(PyObject* pyobject, const char* tname, Int_t low, Int_t high)
{
    Int_t lchar = -1;
    if (CPyCppyy_PyUnicode_Check(pyobject)) {
        if (CPyCppyy_PyUnicode_GET_SIZE(pyobject) == 1)
            lchar = (Int_t)CPyCppyy_PyUnicode_AsChar(pyobject);
        else
            PyErr_Format(PyExc_ValueError, "%s expected, got string of size " PY_SSIZE_T_FORMAT,
                tname, CPyCppyy_PyUnicode_GET_SIZE(pyobject));
    } else if (!PyFloat_Check(pyobject)) {   // don't allow truncating conversion
        lchar = PyLong_AsLong(pyobject);
        if (lchar == -1 && PyErr_Occurred())
            ; // empty, as error already set
        else if (!(low <= lchar && lchar <= high)) {
            PyErr_Format(PyExc_ValueError,
                "integer to character: value %d not in range [%d,%d]", lchar, low, high);
            lchar = -1;
        }
    } else
        PyErr_SetString(PyExc_TypeError, "char or small int type expected");

    return lchar;
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(name, type, F1)                 \
bool CPyCppyy::Const##name##RefConverter::SetArg(                            \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
    type val = (type)F1(pyobject);                                           \
    if (val == (type)-1 && PyErr_Occurred())                                 \
       return false;                                                         \
    para.fValue.f##name = val;                                               \
    para.fRef = &para.fValue.f##name;                                        \
    para.fTypeCode = 'r';                                                    \
    return true;                                                             \
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONST_CHAR_REF_CONVERTER(name, type, low, high)     \
bool CPyCppyy::Const##name##RefConverter::SetArg(                            \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
/* convert <pyobject> to C++ <<type>>, set arg for call, allow int -> char */\
    type val = (type)ExtractChar(pyobject, #type, low, high);                \
    if (val == (type)-1 && PyErr_Occurred())                                 \
        return false;                                                        \
    para.fValue.fLong = val;                                                 \
    para.fTypeCode = 'l';                                                    \
    return true;                                                             \
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_BASIC_CHAR_CONVERTER(name, type, low, high)               \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
/* convert <pyobject> to C++ <<type>>, set arg for call, allow int -> char */\
    long val = ExtractChar(pyobject, #type, low, high);                      \
    if (val == -1 && PyErr_Occurred())                                       \
        return false;                                                        \
    para.fValue.fLong = val;                                                 \
    para.fTypeCode = 'l';                                                    \
    return true;                                                             \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
    return CPyCppyy_PyUnicode_FromFormat("%c", *((type*)address));           \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(PyObject* value, void* address)     \
{                                                                            \
    if (CPyCppyy_PyUnicode_Check(value)) {                                   \
        const char* buf = CPyCppyy_PyUnicode_AsString(value);                \
        if (PyErr_Occurred())                                                \
            return false;                                                    \
        int len = CPyCppyy_PyUnicode_GET_SIZE(value);                        \
        if (len != 1) {                                                      \
            PyErr_Format(PyExc_TypeError, #type" expected, got string of size %d", len);\
            return false;                                                    \
        }                                                                    \
        *((type*)address) = (type)buf[0];                                    \
    } else {                                                                 \
        long l = PyLong_AsLong(value);                                       \
        if (l == -1 && PyErr_Occurred())                                     \
            return false;                                                    \
        if (!(low <= l && l <= high)) {                                      \
            PyErr_Format(PyExc_ValueError,                                   \
                "integer to character: value %ld not in range [%d,%d]", l, low, high);\
            return false;                                                    \
        }                                                                    \
        *((type*)address) = (type)l;                                         \
    }                                                                        \
    return true;                                                             \
}


//- converters for built-ins -------------------------------------------------
CPPYY_IMPL_BASIC_CONVERTER(Long, long, long, PyLong_FromLong, CPyCppyy_PyLong_AsStrictLong, 'l')

//----------------------------------------------------------------------------
bool CPyCppyy::LongRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ long&, set arg for call
#if PY_VERSION_HEX < 0x03000000
    if (RefInt_CheckExact(pyobject)) {
        para.fValue.fVoidp = (void*)&((PyIntObject*)pyobject)->ob_ival;
        para.fTypeCode = 'V';
        return true;
    }
#endif

#if PY_VERSION_HEX < 0x02050000
    PyErr_SetString(PyExc_TypeError, "use cppyy.Long for pass-by-ref of longs");
    return false;
#endif

// TODO: this keeps a refcount to the type .. it should be okay to drop that
    static PyTypeObject* c_long_type = GetCTypesType("c_long");
    if (Py_TYPE(pyobject) == c_long_type) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }

    PyErr_SetString(PyExc_TypeError, "use ctypes.c_long for pass-by-ref of longs");
    return false;
}

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONST_CHAR_REF_CONVERTER(Char,  char,          CHAR_MIN,  CHAR_MAX)
CPPYY_IMPL_BASIC_CONST_CHAR_REF_CONVERTER(UChar, unsigned char,        0, UCHAR_MAX)

CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Bool,      bool,    CPyCppyy_PyLong_AsBool)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Short,     short,   CPyCppyy_PyLong_AsShort)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(UShort,    unsigned short, CPyCppyy_PyLong_AsUShort)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Int,       Int_t,     CPyCppyy_PyLong_AsStrictInt)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(UInt,      UInt_t,    PyLongOrInt_AsULong)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Long,      Long_t,    CPyCppyy_PyLong_AsStrictLong)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(ULong,     ULong_t,   PyLongOrInt_AsULong)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(LongLong,  Long64_t,  PyLong_AsLongLong)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(ULongLong, ULong64_t, PyLongOrInt_AsULong64)

//----------------------------------------------------------------------------
bool CPyCppyy::IntRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ (pseudo)int&, set arg for call
#if PY_VERSION_HEX < 0x03000000
    if (RefInt_CheckExact(pyobject)) {
        para.fValue.fVoidp = (void*)&((PyIntObject*)pyobject)->ob_ival;
        para.fTypeCode = 'V';
        return true;
    }
#endif

#if PY_VERSION_HEX >= 0x02050000
// TODO: this keeps a refcount to the type .. it should be okay to drop that
    static PyTypeObject* c_int_type = GetCTypesType("c_int");
    if (Py_TYPE(pyobject) == c_int_type) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }
#endif

// alternate, pass pointer from buffer
    int buflen = Utility::GetBuffer(pyobject, 'i', sizeof(int), para.fValue.fVoidp);
    if (para.fValue.fVoidp && buflen) {
        para.fTypeCode = 'V';
        return true;
    };

#if PY_VERSION_HEX < 0x02050000
    PyErr_SetString(PyExc_TypeError, "use cppyy.Long for pass-by-ref of ints");
#else
    PyErr_SetString(PyExc_TypeError, "use ctypes.c_int for pass-by-ref of ints");
#endif
    return false;
}

//----------------------------------------------------------------------------
// convert <pyobject> to C++ bool, allow int/long -> bool, set arg for call
CPPYY_IMPL_BASIC_CONVERTER(
    Bool, bool, long, PyInt_FromLong, CPyCppyy_PyLong_AsBool, 'l')

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CHAR_CONVERTER(Char,  char,          CHAR_MIN,  CHAR_MAX)
CPPYY_IMPL_BASIC_CHAR_CONVERTER(UChar, unsigned char,        0, UCHAR_MAX)

PyObject* CPyCppyy::UCharAsIntConverter::FromMemory(void* address)
{
// special case to be used with arrays: return a Python int instead of str
// (following the same convention as module array.array)
    return PyInt_FromLong((long)*((unsigned char*)address));
}


//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONVERTER(
    Short, short, long, PyInt_FromLong, CPyCppyy_PyLong_AsShort, 'l')
CPPYY_IMPL_BASIC_CONVERTER(
    UShort, unsigned short, long, PyInt_FromLong, CPyCppyy_PyLong_AsUShort, 'l')
CPPYY_IMPL_BASIC_CONVERTER(
    Int, Int_t, long, PyInt_FromLong, CPyCppyy_PyLong_AsStrictInt, 'l')

//----------------------------------------------------------------------------
bool CPyCppyy::ULongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ unsigned long, set arg for call
    para.fValue.fULong = PyLongOrInt_AsULong(pyobject);
    if (PyErr_Occurred())
        return false;
    para.fTypeCode = 'L';
    return true;
}

PyObject* CPyCppyy::ULongConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned long read at <address>
    return PyLong_FromUnsignedLong(*((unsigned long*)address));
}

bool CPyCppyy::ULongConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ unsigned long, write it at <address>
    unsigned long u = PyLongOrInt_AsULong(value);
    if (PyErr_Occurred())
        return false;
    *((unsigned long*)address) = u;
    return true;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::UIntConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned int read at <address>
    return PyLong_FromUnsignedLong(*((UInt_t*)address));
}

bool CPyCppyy::UIntConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ unsigned int, write it at <address>
    ULong_t u = PyLongOrInt_AsULong(value);
    if (PyErr_Occurred())
        return false;

    if (u > (ULong_t)UINT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "value too large for unsigned int");
        return false;
    }

    *((UInt_t*)address) = (UInt_t)u;
    return true;
}

//- floating point converters ------------------------------------------------
CPPYY_IMPL_BASIC_CONVERTER(
    Float,  float,  double, PyFloat_FromDouble, PyFloat_AsDouble, 'f')
CPPYY_IMPL_BASIC_CONVERTER(
    Double, double, double, PyFloat_FromDouble, PyFloat_AsDouble, 'd')

CPPYY_IMPL_BASIC_CONVERTER(
    LongDouble, LongDouble_t, LongDouble_t, PyFloat_FromDouble, PyFloat_AsDouble, 'g')

//----------------------------------------------------------------------------
bool CPyCppyy::DoubleRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ double&, set arg for call
    if (RefFloat_CheckExact(pyobject)) {
        para.fValue.fVoidp = (void*)&((PyFloatObject*)pyobject)->ob_fval;
        para.fTypeCode = 'V';
        return true;
    }

// alternate, pass pointer from buffer
    int buflen = Utility::GetBuffer(pyobject, 'd', sizeof(double), para.fValue.fVoidp);
    if (para.fValue.fVoidp && buflen) {
        para.fTypeCode = 'V';
        return true;
    }

    PyErr_SetString(PyExc_TypeError, "use cppyy.Double for pass-by-ref of doubles");
    return false;
}

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Float,      float,      PyFloat_AsDouble)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(Double,     double,     PyFloat_AsDouble)
CPPYY_IMPL_BASIC_CONST_REF_CONVERTER(LongDouble, LongDouble_t, PyFloat_AsDouble)

//----------------------------------------------------------------------------
bool CPyCppyy::VoidConverter::SetArg(PyObject*, Parameter&, CallContext*)
{
// can't happen (unless a type is mapped wrongly), but implemented for completeness
    PyErr_SetString(PyExc_SystemError, "void/unknown arguments can\'t be set");
    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::LongLongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ long long, set arg for call
    if (PyFloat_Check(pyobject)) {
    // special case: float implements nb_int, but allowing rounding conversions
    // interferes with overloading
        PyErr_SetString(PyExc_ValueError, "can not convert float to long long");
        return false;
    }

    para.fValue.fLongLong = PyLong_AsLongLong(pyobject);
    if (PyErr_Occurred())
        return false;
    para.fTypeCode = 'q';
    return true;
}

PyObject* CPyCppyy::LongLongConverter::FromMemory(void* address)
{
// construct python object from C++ long long read at <address>
    return PyLong_FromLongLong(*(Long64_t*)address);
}

bool CPyCppyy::LongLongConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ long long, write it at <address>
    Long64_t ll = PyLong_AsLongLong(value);
    if (ll == -1 && PyErr_Occurred())
        return false;
    *((Long64_t*)address) = ll;
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::ULongLongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ unsigned long long, set arg for call
    para.fValue.fULongLong = PyLongOrInt_AsULong64(pyobject);
    if (PyErr_Occurred())
        return false;
    para.fTypeCode = 'Q';
    return true;
}

PyObject* CPyCppyy::ULongLongConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned long long read at <address>
    return PyLong_FromUnsignedLongLong(*(ULong64_t*)address);
}

bool CPyCppyy::ULongLongConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ unsigned long long, write it at <address>
    Long64_t ull = PyLongOrInt_AsULong64(value);
    if (PyErr_Occurred())
        return false;
    *((ULong64_t*)address) = ull;
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CStringConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// construct a new string and copy it in new memory
    const char* s = CPyCppyy_PyUnicode_AsStringChecked(pyobject);
    if (PyErr_Occurred())
        return false;

    fBuffer = std::string(s, CPyCppyy_PyUnicode_GET_SIZE(pyobject));

// verify (too long string will cause truncation, no crash)
    if (fMaxSize != -1 && fMaxSize < (long)fBuffer.size())
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for char array (truncated)");
    else if (fMaxSize != -1)
        fBuffer.resize(fMaxSize, '\0');      // padd remainder of buffer as needed

// set the value and declare success
    para.fValue.fVoidp = (void*)fBuffer.c_str();
    para.fTypeCode = 'p';
    return true;
}

PyObject* CPyCppyy::CStringConverter::FromMemory(void* address)
{
// construct python object from C++ const char* read at <address>
    if (address && *(char**)address) {
        if (fMaxSize != -1) {      // need to prevent reading beyond boundary
            std::string buf(*(char**)address, fMaxSize);    // cut on fMaxSize
            return CPyCppyy_PyUnicode_FromString(buf.c_str());   // cut on \0
        }

        return CPyCppyy_PyUnicode_FromString(*(char**)address);
    }

// empty string in case there's no address
    Py_INCREF(PyStrings::gEmptyString);
    return PyStrings::gEmptyString;
}

bool CPyCppyy::CStringConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ const char*, write it at <address>
    const char* s = CPyCppyy_PyUnicode_AsStringChecked(value);
    if (PyErr_Occurred())
        return false;

// verify (too long string will cause truncation, no crash)
    if (fMaxSize < (UInt_t)CPyCppyy_PyUnicode_GET_SIZE(value))
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for char array (truncated)");

    if (fMaxSize != -1)
        strncpy(*(char**)address, s, fMaxSize);   // padds remainder
    else
    // coverity[secure_coding] - can't help it, it's intentional.
        strcpy(*(char**)address, s);

   return true;
}


//- pointer/array conversions ------------------------------------------------
namespace {

using namespace CPyCppyy;

inline bool CArraySetArg(PyObject* pyobject, Parameter& para, char tc, int size)
{
// general case of loading a C array pointer (void* + type code) as function argument
    if (pyobject == gNullPtrObject || (PyInt_Check(pyobject) && PyInt_AsLong(pyobject) == 0)) {
        para.fValue.fVoidp = nullptr;
    } else {
        int buflen = Utility::GetBuffer(pyobject, tc, size, para.fValue.fVoidp);
        if (!para.fValue.fVoidp || buflen == 0)
            return false;
    }
    para.fTypeCode = 'p';
    return true;
}

} // unnamed namespace


//----------------------------------------------------------------------------
bool CPyCppyy::NonConstCStringConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// attempt base class first (i.e. passing a string), but if that fails, try a buffer
    if (this->CStringConverter::SetArg(pyobject, para, ctxt))
        return true;

// apparently failed, try char buffer
    PyErr_Clear();
    return CArraySetArg(pyobject, para, 'c', sizeof(char));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::NonConstCStringConverter::FromMemory(void* address)
{
// assume this is a buffer access if the size is known; otherwise assume string
    if (fMaxSize != -1)
        return CPyCppyy_PyUnicode_FromStringAndSize(*(char**)address, fMaxSize);
    return this->CStringConverter::FromMemory(address);
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidArrayConverter::GetAddressSpecialCase(PyObject* pyobject, void*& address)
{
// (1): C++11 style "null pointer"
    if (pyobject == gNullPtrObject) {
        address = nullptr;
        return true;
    }

// (2): allow integer zero to act as a null pointer (C NULL), no deriveds
    if (PyInt_CheckExact(pyobject) || PyLong_CheckExact(pyobject)) {
        long val = (long)PyLong_AsLong(pyobject);
        if (val == 0l) {
            address = (void*)val;
            return true;
        }

        return false;
    }

// (3): opaque PyCapsule (CObject in older pythons) from somewhere
    if (CPyCppyy_PyCapsule_CheckExact(pyobject)) {
        address = (void*)CPyCppyy_PyCapsule_GetPointer(pyobject, nullptr);
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidArrayConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// just convert pointer if it is a C++ object
    if (CPPInstance_Check(pyobject)) {
    // depending on memory policy, some objects are no longer owned when passed to C++
        if (!fKeepControl && !UseStrictOwnership(ctxt))
            ((CPPInstance*)pyobject)->CppOwns();

   // set pointer (may be null) and declare success
        para.fValue.fVoidp = ((CPPInstance*)pyobject)->GetObject();
        para.fTypeCode = 'p';
        return true;
    }

// handle special cases
    if (GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
        para.fTypeCode = 'p';
        return true;
    }

// final try: attempt to get buffer
    int buflen = Utility::GetBuffer(pyobject, '*', 1, para.fValue.fVoidp, false);

// ok if buffer exists (can't perform any useful size checks)
    if (para.fValue.fVoidp && buflen != 0) {
        para.fTypeCode = 'p';
        return true;
    }

// give up
    return false;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::VoidArrayConverter::FromMemory(void* address)
{
// nothing sensible can be done, just return <address> as pylong
    if (!address || *(ptrdiff_t*)address == 0) {
        Py_INCREF(gNullPtrObject);
        return gNullPtrObject;
    }
    return CreatePointerView(*(ptrdiff_t**)address);
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidArrayConverter::ToMemory(PyObject* value, void* address)
{
// just convert pointer if it is a C++ object
    if (CPPInstance_Check(value)) {
    // depending on memory policy, some objects are no longer owned when passed to C++
        if (!fKeepControl && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            ((CPPInstance*)value)->CppOwns();

    // set pointer (may be null) and declare success
        *(void**)address = ((CPPInstance*)value)->GetObject();
        return true;
    }

// handle special cases
    void* ptr = nullptr;
    if (GetAddressSpecialCase(value, ptr)) {
        *(void**)address = ptr;
        return true;
    }

// final try: attempt to get buffer
    void* buf = nullptr;
    int buflen = Utility::GetBuffer(value, '*', 1, buf, false);
    if (!buf || buflen == 0)
        return false;

    *(void**)address = buf;
    return true;
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_ARRAY_CONVERTER(name, type, code)                         \
bool CPyCppyy::name##ArrayConverter::SetArg(                                 \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
    return CArraySetArg(pyobject, para, code, sizeof(type));                 \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##ArrayConverter::FromMemory(void* address)          \
{                                                                            \
    Py_ssize_t shape[] = {1, fSize};                                         \
    return CreateLowLevelView(*(type**)address, shape);                      \
}                                                                            \
                                                                             \
bool CPyCppyy::name##ArrayConverter::ToMemory(PyObject* value, void* address)\
{                                                                            \
    void* buf = nullptr;                                                     \
    int buflen = Utility::GetBuffer(value, code, sizeof(type), buf);         \
    if (!buf || buflen == 0)                                                 \
        return false;                                                        \
    if (0 <= fSize) {                                                        \
        if (fSize < buflen/(int)sizeof(type)) {                              \
            PyErr_SetString(PyExc_ValueError, "buffer too large for value"); \
            return false;                                                    \
        }                                                                    \
        memcpy(*(type**)address, buf, 0 < buflen ? ((size_t)buflen) : sizeof(type));\
    } else                                                                   \
        *(type**)address = (type*)buf;                                       \
    return true;                                                             \
}

#define CPPYY_IMPL_ARRAY_CONVERTER2(name, type, code)                        \
CPPYY_IMPL_ARRAY_CONVERTER(name, type, code)                                 \
                                                                             \
bool CPyCppyy::name##ArrayRefConverter::SetArg(                              \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    bool result = name##ArrayConverter::SetArg(pyobject, para, ctxt);        \
    para.fTypeCode = 'V';                                                    \
    return result;                                                           \
}


//----------------------------------------------------------------------------
CPPYY_IMPL_ARRAY_CONVERTER2(Bool,   bool,                  'b')  // signed char
CPPYY_IMPL_ARRAY_CONVERTER (UChar,  unsigned char,         'B')
CPPYY_IMPL_ARRAY_CONVERTER2(Short,  short,                 'h')
CPPYY_IMPL_ARRAY_CONVERTER2(UShort, unsigned short,        'H')
CPPYY_IMPL_ARRAY_CONVERTER (Int,    int,                   'i')
CPPYY_IMPL_ARRAY_CONVERTER2(UInt,   unsigned int,          'I')
CPPYY_IMPL_ARRAY_CONVERTER (Long,   long,                  'l')
CPPYY_IMPL_ARRAY_CONVERTER2(ULong,  unsigned long,         'L')
CPPYY_IMPL_ARRAY_CONVERTER2(LLong,  long long,             'q')
CPPYY_IMPL_ARRAY_CONVERTER2(ULLong, unsigned long long,    'Q')
CPPYY_IMPL_ARRAY_CONVERTER2(Float,  float,                 'f')
CPPYY_IMPL_ARRAY_CONVERTER (Double, double,                'd')


//- converters for special cases ---------------------------------------------
#define CPPYY_IMPL_STRING_AS_PRIMITIVE_CONVERTER(name, type, F1, F2)         \
CPyCppyy::name##Converter::name##Converter(bool keepControl) :               \
    CppObjectConverter(Cppyy::GetScope(#type), keepControl) {}               \
                                                                             \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    if (CPyCppyy_PyUnicode_Check(pyobject)) {                                \
        fBuffer = type(CPyCppyy_PyUnicode_AsString(pyobject),                \
                       CPyCppyy_PyUnicode_GET_SIZE(pyobject));               \
        para.fValue.fVoidp = &fBuffer;                                       \
        para.fTypeCode = 'V';                                                \
        return true;                                                         \
    }                                                                        \
                                                                             \
    if (!(PyInt_Check(pyobject) || PyLong_Check(pyobject))) {                \
        bool result = CppObjectConverter::SetArg(pyobject, para, ctxt);      \
        para.fTypeCode = 'V';                                                \
        return result;                                                       \
    }                                                                        \
    return false;                                                            \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
    if (address)                                                             \
        return CPyCppyy_PyUnicode_FromStringAndSize(((type*)address)->F1(), ((type*)address)->F2()); \
    Py_INCREF(PyStrings::gEmptyString);                                      \
    return PyStrings::gEmptyString;                                          \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(PyObject* value, void* address)     \
{                                                                            \
    if (CPyCppyy_PyUnicode_Check(value)) {                                   \
        *((type*)address) = CPyCppyy_PyUnicode_AsString(value);              \
        return true;                                                         \
    }                                                                        \
                                                                             \
    return CppObjectConverter::ToMemory(value, address);                     \
}

CPPYY_IMPL_STRING_AS_PRIMITIVE_CONVERTER(STLString, std::string, c_str, size)
#if __cplusplus > 201402L
CPPYY_IMPL_STRING_AS_PRIMITIVE_CONVERTER(STLStringView, std::string_view, data, size)
#endif

//----------------------------------------------------------------------------
bool CPyCppyy::CppObjectConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance*, set arg for call
    if (!CPPInstance_Check(pyobject)) {
        if (GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
            para.fTypeCode = 'p';      // allow special cases such as nullptr
            return true;
        }

   // not a cppyy object (TODO: handle SWIG etc.)
        return false;
    }

    CPPInstance* pyobj = (CPPInstance*)pyobject;
    if (pyobj->ObjectIsA() && Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && !UseStrictOwnership(ctxt))
            ((CPPInstance*)pyobject)->CppOwns();

    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (pyobj->ObjectIsA() != fClass) {
            para.fValue.fLong += Cppyy::GetBaseOffset(
                pyobj->ObjectIsA(), fClass, para.fValue.fVoidp, 1 /* up-cast */);
        }

   // set pointer (may be null) and declare success
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CppObjectConverter::FromMemory(void* address)
{
// construct python object from C++ instance read at <address>
    return BindCppObject(address, fClass);
}

//----------------------------------------------------------------------------
bool CPyCppyy::CppObjectConverter::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ instance, write it at <address>
    if (!CPPInstance_Check(value)) {
        void* ptr = nullptr;
        if (GetAddressSpecialCase(value, ptr)) {
            *(void**)address = ptr;          // allow special cases such as nullptr
            return true;
        }

    // not a cppyy object (TODO: handle SWIG etc.)
        return false;
    }

    if (Cppyy::IsSubtype(((CPPInstance*)value)->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            ((CPPInstance*)value)->CppOwns();

    // call assignment operator through a temporarily wrapped object proxy
        PyObject* pyobj = BindCppObjectNoCast(address, fClass);
        ((CPPInstance*)pyobj)->CppOwns();     // TODO: might be recycled (?)
        PyObject* result = PyObject_CallMethod(pyobj, (char*)"__assign__", (char*)"O", value);
        Py_DECREF(pyobj);
        if (result) {
            Py_DECREF(result);
            return true;
        }
    }

    return false;
}

// TODO: CONSOLIDATE ValueCpp, RefCpp, and CppObject ...

//----------------------------------------------------------------------------
bool CPyCppyy::ValueCppObjectConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ instance, set arg for call
    if (!CPPInstance_Check(pyobject))
        return false;

    CPPInstance* pyobj = (CPPInstance*)pyobject;
    if (pyobj->ObjectIsA() && Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (!para.fValue.fVoidp)
            return false;

        if (pyobj->ObjectIsA() != fClass) {
            para.fValue.fLong += Cppyy::GetBaseOffset(
                pyobj->ObjectIsA(), fClass, para.fValue.fVoidp, 1 /* up-cast */);
        }

        para.fTypeCode = 'V';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::RefCppObjectConverter::SetArg(
        PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ instance&, set arg for call
    if (!CPPInstance_Check(pyobject))
        return false;
    CPPInstance* pyobj = (CPPInstance*)pyobject;

// reject moves
    if (pyobj->fFlags & CPPInstance::kIsRValue)
        return false;

    if (pyobj->ObjectIsA() && Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (pyobj->ObjectIsA() != fClass) {
            para.fValue.fLong += Cppyy::GetBaseOffset(
                pyobj->ObjectIsA(), fClass, para.fValue.fVoidp, 1 /* up-cast */);
        }

       para.fTypeCode = 'V';
       return true;
    }

    return false;
}

//----------------------------------------------------------------------------
#if PY_VERSION_HEX < 0x03000000
const size_t refcount_cutoff = 1;
#else
// p3 has at least 2 ref-counts, as contrary to p2, it will create a descriptor
// copy for the method holding self in the case of __init__; but there can also
// be a reference held by the frame object, which is indistinguishable from a
// local variable reference, so the cut-off has to remain 2.
const size_t refcount_cutoff = 2;
#endif

bool CPyCppyy::MoveCppObjectConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance&&, set arg for call
    if (!CPPInstance_Check(pyobject))
        return false;
    CPPInstance* pyobj = (CPPInstance*)pyobject;

// moving is same as by-ref, but have to check that move is allowed
    int moveit_reason = 0;
    if (pyobj->fFlags & CPPInstance::kIsRValue) {
        pyobj->fFlags &= ~CPPInstance::kIsRValue;
        moveit_reason = 2;
    } else if (pyobject->ob_refcnt == refcount_cutoff) {
        moveit_reason = 1;
    }

    if (moveit_reason) {
        bool result = this->RefCppObjectConverter::SetArg(pyobject, para, ctxt);
        if (!result && moveit_reason == 2)       // restore the movability flag?
            ((CPPInstance*)pyobject)->fFlags |= CPPInstance::kIsRValue;
        return result;
    }

    PyErr_SetString(PyExc_ValueError, "object is not an rvalue");
    return false;      // not a temporary or movable object
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
bool CPyCppyy::CppObjectPtrConverter<ISREFERENCE>::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance**, set arg for call
    if (!CPPInstance_Check(pyobject))
        return false;              // not a cppyy object (TODO: handle SWIG etc.)

    CPPInstance* pyobj = (CPPInstance*)pyobject;
    if (Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && !UseStrictOwnership(ctxt))
            pyobj->CppOwns();

    // set pointer (may be null) and declare success
        if (pyobj->fFlags & CPPInstance::kIsReference)
            para.fValue.fVoidp = pyobj->fObject; // already a ptr to object
        else
            para.fValue.fVoidp = &pyobj->fObject;
        para.fTypeCode = ISREFERENCE ? 'V' : 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
PyObject* CPyCppyy::CppObjectPtrConverter<ISREFERENCE>::FromMemory(void* address)
{
// construct python object from C++ instance* read at <address>
    return BindCppObject(address, fClass, CPPInstance::kIsReference);
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
bool CPyCppyy::CppObjectPtrConverter<ISREFERENCE>::ToMemory(PyObject* value, void* address)
{
// convert <value> to C++ instance*, write it at <address>
    if (!CPPInstance_Check(value))
        return false;              // not a cppyy object (TODO: handle SWIG etc.)

    if (Cppyy::IsSubtype(((CPPInstance*)value)->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            ((CPPInstance*)value)->CppOwns();

    // register the value for potential recycling
        MemoryRegulator::RegisterPyObject((CPPInstance*)value, ((CPPInstance*)value)->GetObject());

    // set pointer (may be null) and declare success
        *(void**)address = ((CPPInstance*)value)->GetObject();
        return true;
    }

    return false;
}


namespace CPyCppyy {
// Instantiate the templates
    template class CPyCppyy::CppObjectPtrConverter<true>;
    template class CPyCppyy::CppObjectPtrConverter<false>;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CppObjectArrayConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* txt */)
{
// convert <pyobject> to C++ instance**, set arg for call
    if (!TupleOfInstances_CheckExact(pyobject))
        return false;              // no guarantee that the tuple is okay

// treat the first instance of the tuple as the start of the array, and pass it
// by pointer (TODO: store and check sizes)
    if (PyTuple_Size(pyobject) < 1)
        return false;

    PyObject* first = PyTuple_GetItem(pyobject, 0);
    if (!CPPInstance_Check(first))
        return false;              // should not happen

    if (Cppyy::IsSubtype(((CPPInstance*)first)->ObjectIsA(), fClass)) {
    // no memory policies supported; set pointer (may be null) and declare success
        para.fValue.fVoidp = ((CPPInstance*)first)->fObject;
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CppObjectArrayConverter::FromMemory(void* address)
{
// construct python tuple of instances from C++ array read at <address>
    if (m_size <= 0)     // if size unknown, just hand out the first object
        return BindCppObjectNoCast(address, fClass);

    return BindCppObjectArray(address, fClass, m_size);
}

//----------------------------------------------------------------------------
bool CPyCppyy::CppObjectArrayConverter::ToMemory(PyObject* /* value */, void* /* address */)
{
// convert <value> to C++ array of instances, write it at <address>

// TODO: need to have size both for the array and from the input
    PyErr_SetString(PyExc_NotImplementedError,
        "access to C-arrays of objects not yet implemented!");
    return false;
}

//___________________________________________________________________________
// CLING WORKAROUND -- classes for STL iterators are completely undefined in that
// they come in a bazillion different guises, so just do whatever
bool CPyCppyy::STLIteratorConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
    if (!CPPInstance_Check(pyobject))
        return false;

// just set the pointer value, no check
    CPPInstance* pyobj = (CPPInstance*)pyobject;
    para.fValue.fVoidp = pyobj->GetObject();
    para.fTypeCode = 'V';
    return true;
}
// -- END CLING WORKAROUND

//----------------------------------------------------------------------------
bool CPyCppyy::VoidPtrRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ void*&, set arg for call
    if (CPPInstance_Check(pyobject)) {
        para.fValue.fVoidp = &((CPPInstance*)pyobject)->fObject;
        para.fTypeCode = 'V';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidPtrPtrConverter::SetArg(
      PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ void**, set arg for call
    if (CPPInstance_Check(pyobject)) {
    // this is a C++ object, take and set its address
        para.fValue.fVoidp = &((CPPInstance*)pyobject)->fObject;
        para.fTypeCode = 'p';
        return true;
    }

// buffer objects are allowed under "user knows best"
    int buflen = Utility::GetBuffer(pyobject, '*', 1, para.fValue.fVoidp, false);

// ok if buffer exists (can't perform any useful size checks)
    if (para.fValue.fVoidp && buflen != 0) {
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::VoidPtrPtrConverter::FromMemory(void* address)
{
// read a void** from address; since this is unknown, long is used (user can cast)
    if (!address || *(ptrdiff_t*)address == 0) {
        Py_INCREF(gNullPtrObject);
        return gNullPtrObject;
    }
    return CreatePointerView(*(ptrdiff_t**)address);
}

//----------------------------------------------------------------------------
bool CPyCppyy::PyObjectConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// by definition: set and declare success
    para.fValue.fVoidp = pyobject;
    para.fTypeCode = 'p';
    return true;
}

PyObject* CPyCppyy::PyObjectConverter::FromMemory(void* address)
{
// construct python object from C++ PyObject* read at <address>
    PyObject* pyobject = *((PyObject**)address);

    if (!pyobject) {
        Py_RETURN_NONE;
    }

    Py_INCREF(pyobject);
    return pyobject;
}

bool CPyCppyy::PyObjectConverter::ToMemory(PyObject* value, void* address)
{
// no conversion needed, write <value> at <address>
    Py_INCREF(value);
    *((PyObject**)address) = value;
    return true;
}


//- function pointer converter -----------------------------------------------
bool CPyCppyy::FunctionPointerConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /*ctxt*/)
{
    if (!CPPOverload_Check(pyobject))
        return false;

    CPPOverload* ol = (CPPOverload*)pyobject;
    if (!ol->fMethodInfo || ol->fMethodInfo->fMethods.empty())
        return false;

// find the overload with matching signature
    for (auto& m : ol->fMethodInfo->fMethods) {
        PyObject* sig = m->GetSignature(false);
        bool found = fSignature == CPyCppyy_PyUnicode_AsString(sig);
        Py_DECREF(sig);
        if (found) {
            para.fValue.fVoidp = (void*)m->GetFunctionAddress();
            if (!para.fValue.fVoidp)
                return false;
            para.fTypeCode = 'p';
            return true;
        }
    }

    return false;
}


//- smart pointer converters -------------------------------------------------
bool CPyCppyy::SmartPtrCppObjectConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
    char typeCode = fIsRef ? 'p' : 'V';

    if (!CPPInstance_Check(pyobject)) {
        if (fIsRef && GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
            para.fTypeCode = typeCode;      // allow special cases such as nullptr
            return true;
        }

        return false;
    }

    CPPInstance* pyobj = (CPPInstance*)pyobject;

// for the case where we have a 'hidden' smart pointer:
    if ((pyobj->fFlags & CPPInstance::kIsSmartPtr) &&
            Cppyy::IsSubtype(pyobj->fSmartPtrType, fSmartPtrType)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (fKeepControl && !UseStrictOwnership(ctxt))
            ((CPPInstance*)pyobject)->CppOwns();

    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->fObject;
        if (pyobj->fSmartPtrType != fSmartPtrType) {
            para.fValue.fLong += Cppyy::GetBaseOffset(
                pyobj->fSmartPtrType, fSmartPtrType, para.fValue.fVoidp, 1 /* up-cast */);
        }

    // set pointer (may be null) and declare success
        para.fTypeCode = typeCode;
        return true;
    }

// for the case where we have an 'exposed' smart pointer:
    if (pyobj->ObjectIsA() && Cppyy::IsSubtype(pyobj->ObjectIsA(), fSmartPtrType)) {
    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (pyobj->ObjectIsA() != fSmartPtrType) {
            para.fValue.fLong += Cppyy::GetBaseOffset(
                pyobj->ObjectIsA(), fSmartPtrType, para.fValue.fVoidp, 1 /* up-cast */);
        }

    // set pointer (may be null) and declare success
        para.fTypeCode = typeCode;
        return true;
    }

    return false;
}

PyObject* CPyCppyy::SmartPtrCppObjectConverter::FromMemory(void* address)
{
    if (!address || !fSmartPtrType)
        return nullptr;

// TODO: note the mismatch between address, which is the smart pointer, and the
// declared type, which is the raw pointer
    CPPInstance* pyobj = (CPPInstance*)BindCppObjectNoCast(address, fRawPtrType);
    if (pyobj)
        pyobj->SetSmartPtr(fSmartPtrType, fDereferencer);

    return (PyObject*)pyobj;
}


//----------------------------------------------------------------------------
namespace {

// clang libcpp and gcc use the same structure (ptr, size)
#if defined (_LIBCPP_INITIALIZER_LIST) || defined(__GNUC__)
struct faux_initlist
{
    typedef size_t size_type;
    typedef void*  iterator;
    iterator  _M_array;
    size_type _M_len;
};
#else
#define NO_KNOWN_INITIALIZER_LIST 1
#endif

} // unnamed namespace

bool CPyCppyy::InitializerListConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /*ctxt*/)
{
#ifdef NO_KNOWN_INITIALIZER_LIST
    return false;
#else
// convert the given argument to an initializer list temporary
    if (!PySequence_Check(pyobject))
        return false;

// can only construct empty lists, so use a fake initializer list
    size_t len = (size_t)PySequence_Size(pyobject);
    faux_initlist* fake = (faux_initlist*)malloc(sizeof(faux_initlist)+fValueSize*len);
    fake->_M_len = (faux_initlist::size_type)len;
    fake->_M_array = (faux_initlist::iterator)((char*)fake+sizeof(faux_initlist));

    for (faux_initlist::size_type i = 0; i < fake->_M_len; ++i) {
        PyObject* item = PySequence_GetItem(pyobject, i);
        if (!fConverter->ToMemory(item, (char*)fake->_M_array + i*fValueSize)) {
            Py_DECREF(item);
            free((void*)fake);
            return false;
        }
        Py_DECREF(item);
    }

// TODO: this passes, but how to clean up?
    para.fValue.fVoidp = (void*)fake;
    para.fTypeCode = 'X';
    return true;
#endif
}

//----------------------------------------------------------------------------
bool CPyCppyy::NotImplementedConverter::SetArg(PyObject*, Parameter&, CallContext*)
{
// raise a NotImplemented exception to take a method out of overload resolution
    PyErr_SetString(PyExc_NotImplementedError, "this method can not (yet) be called");
    return false;
}


//- factories ----------------------------------------------------------------
CPyCppyy::Converter* CPyCppyy::CreateConverter(const std::string& fullType, long size)
{
// The matching of the fulltype to a converter factory goes through up to five levels:
//   1) full, exact match
//   2) match of decorated, unqualified type
//   3) accept const ref as by value
//   4) accept ref as pointer
//   5) generalized cases (covers basically all C++ classes)
//
// If all fails, void is used, which will generate a run-time warning when used.

// an exactly matching converter is best
    ConvFactories_t::iterator h = gConvFactories.find(fullType);
    if (h != gConvFactories.end())
        return (h->second)(size);

// resolve typedefs etc.
    const std::string& resolvedType = Cppyy::ResolveName(fullType);

// a full, qualified matching converter is preferred
    if (resolvedType != fullType) {
        h = gConvFactories.find(resolvedType);
        if (h != gConvFactories.end())
            return (h->second)(size);
    }

//-- nothing? ok, collect information about the type and possible qualifiers/decorators
    bool isConst = strncmp(resolvedType.c_str(), "const", 5) == 0;
    const std::string& cpd = Utility::Compound(resolvedType);
    std::string realType   = TypeManip::clean_type(resolvedType, false, true);

// accept unqualified type (as python does not know about qualifiers)
    h = gConvFactories.find(realType + cpd);
    if (h != gConvFactories.end())
        return (h->second)(size);

// drop const, as that is mostly meaningless to python (with the exception
// of c-strings, but those are specialized in the converter map)
    if (isConst) {
        realType = TypeManip::remove_const(realType);
        h = gConvFactories.find(realType + cpd);
        if (h != gConvFactories.end())
            return (h->second)(size);
    }

// CLING WORKAROUND -- if the type is a fixed-size array, it will have a funky
// resolved type like MyClass(&)[N], which TClass::GetClass() fails on. So, strip
// it down:
/* TODO: remove TClassEdit usage
   if (cpd == "[]")
      realType = TClassEdit::CleanType(realType.substr(0, realType.rfind("(")).c_str(), 1);
*/
// -- CLING WORKAROUND

//-- still nothing? try pointer instead of array (for builtins)
    if (cpd == "[]") {
        h = gConvFactories.find(realType + "*");
        if (h != gConvFactories.end())
            return (h->second)(size);
    }

//-- special case: initializer list
    auto pos = realType.find("initializer_list");
    if (pos == 0 /* no std:: */ || pos == 5 /* with std:: */) {
    // get the type of the list and create a converter (TODO: get hold of value_type?)
        auto pos2 = realType.find('<');
        std::string value_type = realType.substr(pos2+1, realType.size()-pos2-2);
        Converter* cnv = CreateConverter(value_type);
        if (cnv)
            return new InitializerListConverter(cnv, Cppyy::SizeOf(value_type));
    }

//-- still nothing? use a generalized converter
    bool control = cpd == "&" || isConst;

// converters for known C++ classes and default (void*)
    Converter* result = nullptr;
    if (Cppyy::TCppScope_t klass = Cppyy::GetScope(realType)) {
        Cppyy::TCppType_t raw; Cppyy::TCppMethod_t deref;
        if (Cppyy::GetSmartPtrInfo(realType, raw, deref)) {
            if (cpd == "") {
                result = new SmartPtrCppObjectConverter(klass, raw, deref, control);
            } else if (cpd == "&") {
                result = new SmartPtrCppObjectConverter(klass, raw, deref);
            } else if (cpd == "*" && size <= 0) {
                result = new SmartPtrCppObjectConverter(klass, raw, deref, control, true);
            }
        }

        if (!result) {
        // CLING WORKAROUND -- special case for STL iterators
            if (realType.find("__gnu_cxx::__normal_iterator", 0) /* vector */ == 0)
                result = new STLIteratorConverter();
            else
       // -- CLING WORKAROUND
            if (cpd == "**" || cpd == "*[]" || cpd == "&*")
                result = new CppObjectPtrConverter<false>(klass, control);
            else if (cpd == "*&")
                result = new CppObjectPtrConverter<true>(klass, control);
            else if (cpd == "*" && size <= 0)
                result = new CppObjectConverter(klass, control);
            else if (cpd == "&")
                result = new RefCppObjectConverter(klass);
            else if (cpd == "&&")
                result = new MoveCppObjectConverter(klass);
            else if (cpd == "[]" || size > 0)
                result = new CppObjectArrayConverter(klass, size, false);
            else if (cpd == "")             // by value
                result = new ValueCppObjectConverter(klass, true);
        }
    } else if (resolvedType.find("(*)") != std::string::npos ||
               (resolvedType.find("::*)") != std::string::npos)) {
    // this is a function function pointer
    // TODO: find better way of finding the type
    // TODO: a converter that generates wrappers as appropriate
        auto pos2 = resolvedType.find("*)");
        result = new FunctionPointerConverter(resolvedType.substr(pos2+2));
    }

    if (!result && cpd == "&&")                       // unhandled moves
        result = new NotImplementedConverter();

    if (!result && h != gConvFactories.end())
    // converter factory available, use it to create converter
        result = (h->second)(size);
    else if (!result) {
        if (cpd != "") {
            result = new VoidArrayConverter();        // "user knows best"
        } else {
            result = new VoidConverter();             // fails on use
        }
    }

    return result;
}

//----------------------------------------------------------------------------
namespace {

using namespace CPyCppyy;

static struct InitConvFactories_t {
public:
    InitConvFactories_t() {
    // load all converter factories in the global map 'gConvFactories'
        CPyCppyy::ConvFactories_t& gf = gConvFactories;

    // factories for built-ins
        gf["bool"] =                        (cf_t)+[](long) { return new BoolConverter{}; };
        gf["const bool&"] =                 (cf_t)+[](long) { return new ConstBoolRefConverter{}; };
        gf["char"] =                        (cf_t)+[](long) { return new CharConverter{}; };
        gf["const char&"] =                 (cf_t)+[](long) { return new ConstCharRefConverter{}; };
        gf["signed char"] =                 (cf_t)+[](long) { return new CharConverter{}; };
        gf["const signed char&"] =          (cf_t)+[](long) { return new ConstCharRefConverter{}; };
        gf["unsigned char"] =               (cf_t)+[](long) { return new UCharConverter{}; };
        gf["const unsigned char&"] =        (cf_t)+[](long) { return new ConstUCharRefConverter{}; };
        gf["UCharAsInt"] =                  (cf_t)+[](long) { return new UCharAsIntConverter{}; };
        gf["short"] =                       (cf_t)+[](long) { return new ShortConverter{}; };
        gf["const short&"] =                (cf_t)+[](long) { return new ConstShortRefConverter{}; };
        gf["unsigned short"] =              (cf_t)+[](long) { return new UShortConverter{}; };
        gf["const unsigned short&"] =       (cf_t)+[](long) { return new ConstUShortRefConverter{}; };
        gf["int"] =                         (cf_t)+[](long) { return new IntConverter{}; };
        gf["int&"] =                        (cf_t)+[](long) { return new IntRefConverter{}; };
        gf["const int&"] =                  (cf_t)+[](long) { return new ConstIntRefConverter{}; };
        gf["unsigned int"] =                (cf_t)+[](long) { return new UIntConverter{}; };
        gf["const unsigned int&"] =         (cf_t)+[](long) { return new ConstUIntRefConverter{}; };
        gf["internal_enum_type_t"] =        (cf_t)+[](long) { return new IntConverter{}; };
        gf["internal_enum_type_t&"] =       (cf_t)+[](long) { return new IntRefConverter{}; };
        gf["long"] =                        (cf_t)+[](long) { return new LongConverter{}; };
        gf["long&"] =                       (cf_t)+[](long) { return new LongRefConverter{}; };
        gf["const long&"] =                 (cf_t)+[](long) { return new ConstLongRefConverter{}; };
        gf["unsigned long"] =               (cf_t)+[](long) { return new ULongConverter{}; };
        gf["const unsigned long&"] =        (cf_t)+[](long) { return new ConstULongRefConverter{}; };
        gf["long long"] =                   (cf_t)+[](long) { return new LongLongConverter{}; };
        gf["const long long&"] =            (cf_t)+[](long) { return new ConstLongLongRefConverter{}; };
        gf["unsigned long long"] =          (cf_t)+[](long) { return new ULongLongConverter{}; };
        gf["const unsigned long long&"] =   (cf_t)+[](long) { return new ConstULongLongRefConverter{}; };

        gf["float"] =                       (cf_t)+[](long) { return new FloatConverter{}; };
        gf["const float&"] =                (cf_t)+[](long) { return new ConstFloatRefConverter{}; };
        gf["double"] =                      (cf_t)+[](long) { return new DoubleConverter{}; };
        gf["double&"] =                     (cf_t)+[](long) { return new DoubleRefConverter{}; };
        gf["const double&"] =               (cf_t)+[](long) { return new ConstDoubleRefConverter{}; };
        gf["long double"] =                 (cf_t)+[](long) { return new LongDoubleConverter{}; };
        gf["const long double&"] =          (cf_t)+[](long) { return new ConstLongDoubleRefConverter{}; };
        gf["void"] =                        (cf_t)+[](long) { return new VoidConverter{}; };

    // pointer/array factories
        gf["bool*"] =                       (cf_t)+[](long sz) { return new BoolArrayConverter{sz}; };
        gf["bool&"] =                       (cf_t)+[](long sz) { return new BoolArrayRefConverter{sz}; };
        gf["const unsigned char*"] =        (cf_t)+[](long sz) { return new UCharArrayConverter{sz}; };
        gf["unsigned char*"] =              (cf_t)+[](long sz) { return new UCharArrayConverter{sz}; };
        gf["short*"] =                      (cf_t)+[](long sz) { return new ShortArrayConverter{sz}; };
        gf["short&"] =                      (cf_t)+[](long sz) { return new ShortArrayRefConverter{sz}; };
        gf["unsigned short*"] =             (cf_t)+[](long sz) { return new UShortArrayConverter{sz}; };
        gf["unsigned short&"] =             (cf_t)+[](long sz) { return new UShortArrayRefConverter{sz}; };
        gf["int*"] =                        (cf_t)+[](long sz) { return new IntArrayConverter{sz}; };
        gf["unsigned int*"] =               (cf_t)+[](long sz) { return new UIntArrayConverter{sz}; };
        gf["unsigned int&"] =               (cf_t)+[](long sz) { return new UIntArrayRefConverter{sz}; };
        gf["long*"] =                       (cf_t)+[](long sz) { return new LongArrayConverter{sz}; };
        gf["unsigned long*"] =              (cf_t)+[](long sz) { return new ULongArrayConverter{sz}; };
        gf["unsigned long&"] =              (cf_t)+[](long sz) { return new ULongArrayRefConverter{sz}; };
        gf["float*"] =                      (cf_t)+[](long sz) { return new FloatArrayConverter{sz}; };
        gf["float&"] =                      (cf_t)+[](long sz) { return new FloatArrayRefConverter{sz}; };
        gf["double*"] =                     (cf_t)+[](long sz) { return new DoubleArrayConverter{sz}; };
        gf["long long*"] =                  (cf_t)+[](long sz) { return new LLongArrayConverter{sz}; };
        gf["long long&"] =                  (cf_t)+[](long sz) { return new LLongArrayRefConverter{sz}; };
        gf["unsigned long long*"] =         (cf_t)+[](long sz) { return new ULLongArrayConverter{sz}; };
        gf["unsigned long long&"] =         (cf_t)+[](long sz) { return new ULLongArrayRefConverter{sz}; };
        gf["void*"] =                       (cf_t)+[](long sz) { return new VoidArrayConverter{static_cast<bool>(sz)}; };

    // aliases
        gf["Long64_t"] =                    gf["long long"];
        gf["Long64_t*"] =                   gf["long long*"];
        gf["Long64_t&"] =                   gf["long long&"];
        gf["const Long64_t&"] =             gf["const long long&"];
        gf["ULong64_t"] =                   gf["unsigned long long"];
        gf["ULong64_t*"] =                  gf["unsigned long long*"];
        gf["ULong64_t&"] =                  gf["unsigned long long&"];
        gf["const ULong64_t&"] =            gf["const unsigned long long&"];

    // factories for special cases
        gf["const char*"] =                 (cf_t)+[](long) { return new CStringConverter{}; };
        gf["const char[]"] =                (cf_t)+[](long) { return new CStringConverter{}; };
        gf["char*"] =                       (cf_t)+[](long) { return new NonConstCStringConverter{}; };
        gf["std::string"] =                 (cf_t)+[](long) { return new STLStringConverter{}; };
        gf["string"] =                      (cf_t)+[](long) { return new STLStringConverter{}; };
        gf["const std::string&"] =          (cf_t)+[](long) { return new STLStringConverter{}; };
        gf["const string&"] =               (cf_t)+[](long) { return new STLStringConverter{}; };
#if __cplusplus > 201402L
        gf["std::string_view"] =            (cf_t)+[](long) { return new STLStringViewConverter{}; };
        gf["string_view"] =                 (cf_t)+[](long) { return new STLStringViewConverter{}; };
        gf["experimental::basic_string_view<char,char_traits<char> >"] = (cf_t)+[](long) { return new STLStringViewConverter{}; };
#endif
        gf["void*&"] =                      (cf_t)+[](long) { return new VoidPtrRefConverter{}; };
        gf["void**"] =                      (cf_t)+[](long) { return new VoidPtrPtrConverter{}; };
        gf["void*[]"] =                     (cf_t)+[](long) { return new VoidPtrPtrConverter{}; };
        gf["PyObject*"] =                   (cf_t)+[](long) { return new PyObjectConverter{}; };
        gf["_object*"] =                    (cf_t)+[](long) { return new PyObjectConverter{}; };
        gf["FILE*"] =                       (cf_t)+[](long) { return new VoidArrayConverter{}; };
        gf["Float16_t"] =                   (cf_t)+[](long) { return new FloatConverter{}; };
        gf["const Float16_t&"] =            (cf_t)+[](long) { return new ConstFloatRefConverter{}; };
        gf["Double32_t"] =                  (cf_t)+[](long) { return new DoubleConverter{}; };
        gf["Double32_t&"] =                 (cf_t)+[](long) { return new DoubleRefConverter{}; };
        gf["const Double32_t&"] =           (cf_t)+[](long) { return new ConstDoubleRefConverter{}; };
    }
} initConvFactories_;

} // unnamed namespace
