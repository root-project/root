// Bindings
#include "CPyCppyy.h"
#include "DeclareExecutors.h"
#include "CPPInstance.h"
#include "LowLevelViews.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <cstring>
#include <map>
#include <new>
#include <sstream>
#include <utility>
#include <sys/types.h>


//- data _____________________________________________________________________
namespace CPyCppyy {

    typedef Executor* (*ef_t) ();
    typedef std::map<std::string, ef_t> ExecFactories_t;
    static ExecFactories_t gExecFactories;

    extern PyObject* gNullPtrObject;
}


//- helpers ------------------------------------------------------------------
namespace {

#ifdef WITH_THREAD
    class GILControl {
    public:
        GILControl() : fSave(PyEval_SaveThread()) { }
        ~GILControl() {
            PyEval_RestoreThread(fSave);
        }
    private:
        PyThreadState* fSave;
    };
#endif

} // unnamed namespace

#ifdef WITH_THREAD
#define CPPYY_IMPL_GILCALL(rtype, tcode)                                     \
static inline rtype GILCall##tcode(                                          \
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CPyCppyy::CallContext* ctxt)\
{                                                                            \
    if (!ReleasesGIL(ctxt))                                                  \
        return Cppyy::Call##tcode(method, self, ctxt->GetSize(), ctxt->GetArgs());\
    GILControl gc{};                                                         \
    return Cppyy::Call##tcode(method, self, ctxt->GetSize(), ctxt->GetArgs());\
}
#else
#define CPPYY_IMPL_GILCALL(rtype, tcode)                                     \
static inline rtype GILCall##tcode(                                          \
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CPyCppyy::CallContext* ctxt)\
{                                                                            \
    return Cppyy::Call##tcode(method, self, ctxt->GetSize(), ctxt->GetArgs());\
}
#endif

CPPYY_IMPL_GILCALL(void,          V)
CPPYY_IMPL_GILCALL(unsigned char, B)
CPPYY_IMPL_GILCALL(char,          C)
CPPYY_IMPL_GILCALL(short,         H)
CPPYY_IMPL_GILCALL(Int_t,         I)
CPPYY_IMPL_GILCALL(Long_t,        L)
CPPYY_IMPL_GILCALL(Long64_t,      LL)
CPPYY_IMPL_GILCALL(float,         F)
CPPYY_IMPL_GILCALL(double,        D)
CPPYY_IMPL_GILCALL(LongDouble_t,  LD)
CPPYY_IMPL_GILCALL(void*,         R)

static inline Cppyy::TCppObject_t GILCallO(Cppyy::TCppMethod_t method,
    Cppyy::TCppObject_t self, CPyCppyy::CallContext* ctxt, Cppyy::TCppType_t klass)
{
#ifdef WITH_THREAD
    if (!ReleasesGIL(ctxt))
#endif
        return Cppyy::CallO(method, self, ctxt->GetSize(), ctxt->GetArgs(), klass);
#ifdef WITH_THREAD
    GILControl gc{};
    return Cppyy::CallO(method, self, ctxt->GetSize(), ctxt->GetArgs(), klass);
#endif
}

static inline Cppyy::TCppObject_t GILCallConstructor(
    Cppyy::TCppMethod_t method, Cppyy::TCppType_t klass, CPyCppyy::CallContext* ctxt)
{
#ifdef WITH_THREAD
    if (!ReleasesGIL(ctxt))
#endif
        return Cppyy::CallConstructor(method, klass, ctxt->GetSize(), ctxt->GetArgs());
#ifdef WITH_THREAD
    GILControl gc{};
    return Cppyy::CallConstructor(method, klass, ctxt->GetSize(), ctxt->GetArgs());
#endif
}

static inline PyObject* CPyCppyy_PyUnicode_FromLong(long cl)
{
// python chars are range(256)
    if (cl < -256 || cl > 255) {
        PyErr_SetString(PyExc_ValueError, "char conversion out of range");
        return nullptr;
    }
    int c = (int)cl;
    if (c < 0) return CPyCppyy_PyUnicode_FromFormat("%c", 256 - std::abs(c));
    return CPyCppyy_PyUnicode_FromFormat("%c", c);
}

static inline PyObject* CPyCppyy_PyUnicode_FromULong(unsigned long uc)
{
// TODO: range check here?
    if (255 < uc) {
        PyErr_SetString(PyExc_ValueError, "char conversion out of range");
        return nullptr;
    }
    int c = (int)uc;
    return CPyCppyy_PyUnicode_FromFormat("%c", c);
}

static inline PyObject* CPyCppyy_PyBool_FromLong(long b)
{
    PyObject* result = (bool)b ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}


//- executors for built-ins --------------------------------------------------
PyObject* CPyCppyy::BoolExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python bool return value
    bool retval = GILCallB(method, self, ctxt);
    PyObject* result = retval ? Py_True : Py_False;
    Py_INCREF(result);
    return result;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::BoolConstRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python bool return value
    return CPyCppyy_PyBool_FromLong(*((bool*)GILCallR(method, self, ctxt)));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CharExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method with argument <self, ctxt>, construct python string return value
// with the single char
    return CPyCppyy_PyUnicode_FromLong((int)GILCallC(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CharConstRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python string return value
// with the single char
    return CPyCppyy_PyUnicode_FromLong(*((char*)GILCallR(method, self, ctxt)));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::UCharExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, args>, construct python string return value
// with the single char
    return CPyCppyy_PyUnicode_FromLong((unsigned char)GILCallB(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::UCharConstRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python string return value
// with the single char from the pointer return
    return CPyCppyy_PyUnicode_FromLong(*((unsigned char*)GILCallR(method, self, ctxt)));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::WCharExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, args>, construct python string return value
// with the single char
    wchar_t res = (wchar_t)GILCallL(method, self, ctxt);
    return PyUnicode_FromWideChar(&res, 1);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::IntExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python int return value
    return PyInt_FromLong((int)GILCallI(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::Int8Executor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python int return value
    return PyInt_FromLong((int8_t)GILCallC(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::UInt8Executor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python int return value
    return PyInt_FromLong((uint8_t)GILCallB(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::ShortExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python int return value
    return PyInt_FromLong((short)GILCallH(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::LongExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python long return value
    return PyLong_FromLong((Long_t)GILCallL(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::ULongExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python unsigned long return value
    return PyLong_FromUnsignedLong((ULong_t)GILCallLL(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::LongLongExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python long long return value
    Long64_t result = GILCallLL(method, self, ctxt);
    return PyLong_FromLongLong(result);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::ULongLongExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python unsigned long long return value
    ULong64_t result = (ULong64_t)GILCallLL(method, self, ctxt);
    return PyLong_FromUnsignedLongLong(result);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::FloatExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python float return value
    return PyFloat_FromDouble((double)GILCallF(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::DoubleExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python float return value
    return PyFloat_FromDouble((double)GILCallD(method, self, ctxt));
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::LongDoubleExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python float return value
    return PyFloat_FromDouble((double)GILCallLD(method, self, ctxt));
}

//----------------------------------------------------------------------------
bool CPyCppyy::RefExecutor::SetAssignable(PyObject* pyobject)
{
// prepare "buffer" for by-ref returns, used with __setitem__
    if (pyobject) {
        Py_INCREF(pyobject);
        fAssignable = pyobject;
        return true;
    }

    fAssignable = nullptr;
    return false;
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_REFEXEC(name, type, stype, F1, F2)                        \
PyObject* CPyCppyy::name##RefExecutor::Execute(                              \
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt) \
{                                                                            \
    type* ref = (type*)GILCallR(method, self, ctxt);                         \
    if (!ref) { /* can happen if wrapper compilation fails */                \
        PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");\
        return nullptr;                                                      \
    }                                                                        \
    if (!fAssignable)                                                        \
        return F1((stype)*ref);                                              \
    else {                                                                   \
        *ref = (type)F2(fAssignable);                                        \
        Py_DECREF(fAssignable);                                              \
        fAssignable = nullptr;                                               \
        if (*ref == (type)-1 && PyErr_Occurred())                            \
            return nullptr;                                                  \
        Py_INCREF(Py_None);                                                  \
        return Py_None;                                                      \
    }                                                                        \
}

CPPYY_IMPL_REFEXEC(Bool,   bool,   Long_t,   CPyCppyy_PyBool_FromLong,    PyLong_AsLong)
CPPYY_IMPL_REFEXEC(Char,   char,   Long_t,   CPyCppyy_PyUnicode_FromLong, PyLong_AsLong)
CPPYY_IMPL_REFEXEC(UChar,  unsigned char,  ULong_t,  CPyCppyy_PyUnicode_FromULong, PyLongOrInt_AsULong)
CPPYY_IMPL_REFEXEC(Int8,   int8_t,  Long_t,  PyInt_FromLong, PyLong_AsLong)
CPPYY_IMPL_REFEXEC(UInt8,  uint8_t, ULong_t, PyInt_FromLong, PyLongOrInt_AsULong)
CPPYY_IMPL_REFEXEC(Short,  short,          Long_t,   PyInt_FromLong,     PyLong_AsLong)
CPPYY_IMPL_REFEXEC(UShort, unsigned short, ULong_t,  PyInt_FromLong,     PyLongOrInt_AsULong)
CPPYY_IMPL_REFEXEC(Int,    Int_t,    Long_t,   PyInt_FromLong,     PyLong_AsLong)
CPPYY_IMPL_REFEXEC(UInt,   UInt_t,   ULong_t,  PyLong_FromUnsignedLong, PyLongOrInt_AsULong)
CPPYY_IMPL_REFEXEC(Long,   Long_t,   Long_t,   PyLong_FromLong,    PyLong_AsLong)
CPPYY_IMPL_REFEXEC(ULong,  ULong_t,  ULong_t,  PyLong_FromUnsignedLong, PyLongOrInt_AsULong)
CPPYY_IMPL_REFEXEC(LongLong,  Long64_t,  Long64_t,   PyLong_FromLongLong,         PyLong_AsLongLong)
CPPYY_IMPL_REFEXEC(ULongLong, ULong64_t, ULong64_t,  PyLong_FromUnsignedLongLong, PyLongOrInt_AsULong64)
CPPYY_IMPL_REFEXEC(Float,  float,  double, PyFloat_FromDouble, PyFloat_AsDouble)
CPPYY_IMPL_REFEXEC(Double, double, double, PyFloat_FromDouble, PyFloat_AsDouble)
CPPYY_IMPL_REFEXEC(LongDouble, LongDouble_t, LongDouble_t, PyFloat_FromDouble, PyFloat_AsDouble)

template<typename T>
static inline PyObject* PyComplex_FromComplex(const std::complex<T>& c) {
    return PyComplex_FromDoubles(c.real(), c.imag());
}

template<typename T>
static inline std::complex<T> PyComplex_AsComplex(PyObject* pycplx) {
    Py_complex cplx = PyComplex_AsCComplex(pycplx);
    return std::complex<T>(cplx.real, cplx.imag);
}

CPPYY_IMPL_REFEXEC(ComplexD, std::complex<double>,
    std::complex<double>, PyComplex_FromComplex<double>, PyComplex_AsComplex<double>)


//----------------------------------------------------------------------------
PyObject* CPyCppyy::STLStringRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, return python string return value
    std::string* result = (std::string*)GILCallR(method, self, ctxt);
    if (!fAssignable)
        return CPyCppyy_PyUnicode_FromStringAndSize(result->c_str(), result->size());

    *result = std::string(
        CPyCppyy_PyUnicode_AsString(fAssignable), CPyCppyy_PyUnicode_GET_SIZE(fAssignable));

    Py_DECREF(fAssignable);
    fAssignable = nullptr;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::VoidExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, return None
    GILCallV(method, self, ctxt);
    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CStringExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python string return value
    char* result = (char*)GILCallR(method, self, ctxt);
    if (!result) {
        Py_INCREF(PyStrings::gEmptyString);
        return PyStrings::gEmptyString;
    }

    return CPyCppyy_PyUnicode_FromString(result);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::WCStringExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python unicode return value
    wchar_t* result = (wchar_t*)GILCallR(method, self, ctxt);
    if (!result) {
        wchar_t w = L'\0';
        return PyUnicode_FromWideChar(&w, 0);
    }

    return PyUnicode_FromWideChar(result, wcslen(result));
}


//- pointer/array executors --------------------------------------------------
PyObject* CPyCppyy::VoidArrayExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python long return value
    Long_t* result = (Long_t*)GILCallR(method, self, ctxt);
    if (!result) {
        Py_INCREF(gNullPtrObject);
        return gNullPtrObject;
    }
    return CreatePointerView(result);
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_ARRAY_EXEC(name, type)                                    \
PyObject* CPyCppyy::name##ArrayExecutor::Execute(                            \
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt) \
{                                                                            \
    return CreateLowLevelView((type*)GILCallR(method, self, ctxt));          \
}

CPPYY_IMPL_ARRAY_EXEC(Bool,     bool)
CPPYY_IMPL_ARRAY_EXEC(UChar,    unsigned char)
CPPYY_IMPL_ARRAY_EXEC(Short,    short)
CPPYY_IMPL_ARRAY_EXEC(UShort,   unsigned short)
CPPYY_IMPL_ARRAY_EXEC(Int,      int)
CPPYY_IMPL_ARRAY_EXEC(UInt,     unsigned int)
CPPYY_IMPL_ARRAY_EXEC(Long,     long)
CPPYY_IMPL_ARRAY_EXEC(ULong,    unsigned long)
CPPYY_IMPL_ARRAY_EXEC(LLong,    long long)
CPPYY_IMPL_ARRAY_EXEC(ULLong,   unsigned long long)
CPPYY_IMPL_ARRAY_EXEC(Float,    float)
CPPYY_IMPL_ARRAY_EXEC(Double,   double)
CPPYY_IMPL_ARRAY_EXEC(ComplexF, std::complex<float>)
CPPYY_IMPL_ARRAY_EXEC(ComplexD, std::complex<double>)
CPPYY_IMPL_ARRAY_EXEC(ComplexI, std::complex<int>)
CPPYY_IMPL_ARRAY_EXEC(ComplexL, std::complex<long>)


//- special cases ------------------------------------------------------------
#define CPPYY_COMPLEX_EXEC(code, type)                                       \
PyObject* CPyCppyy::Complex##code##Executor::Execute(                        \
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt) \
{                                                                            \
    static Cppyy::TCppScope_t scopeid = Cppyy::GetScope("std::complex<"#type">");\
    std::complex<type>* result =                                             \
        (std::complex<type>*)GILCallO(method, self, ctxt, scopeid);          \
    if (!result) {                                                           \
        PyErr_SetString(PyExc_ValueError, "NULL result where temporary expected");\
        return nullptr;                                                      \
    }                                                                        \
                                                                             \
    PyObject* pyres = PyComplex_FromDoubles(result->real(), result->imag()); \
    ::operator delete(result); /* Cppyy::CallO calls ::operator new */       \
    return pyres;                                                            \
}

CPPYY_COMPLEX_EXEC(D, double)

//----------------------------------------------------------------------------
PyObject* CPyCppyy::STLStringExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python string return value

// TODO: make use of GILLCallS (?!)
    static Cppyy::TCppScope_t sSTLStringScope = Cppyy::GetScope("std::string");
    std::string* result = (std::string*)GILCallO(method, self, ctxt, sSTLStringScope);
    if (!result) {
        Py_INCREF(PyStrings::gEmptyString);
        return PyStrings::gEmptyString;
    }

    PyObject* pyresult =
        CPyCppyy_PyUnicode_FromStringAndSize(result->c_str(), result->size());
    ::operator delete(result); // calls Cppyy::CallO which calls ::operator new

    return pyresult;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::STLWStringExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python string return value
    static Cppyy::TCppScope_t sSTLWStringScope = Cppyy::GetScope("std::wstring");
    std::wstring* result = (std::wstring*)GILCallO(method, self, ctxt, sSTLWStringScope);
    if (!result) {
        wchar_t w = L'\0';
        return PyUnicode_FromWideChar(&w, 0);
    }

    PyObject* pyresult = PyUnicode_FromWideChar(result->c_str(), result->size());
    ::operator delete(result); // calls Cppyy::CallO which calls ::operator new

    return pyresult;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstancePtrExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python proxy object return value
    return BindCppObject((void*)GILCallR(method, self, ctxt), fClass);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execution will bring a temporary in existence
    Cppyy::TCppObject_t value = GILCallO(method, self, ctxt, fClass);

    if (!value) {
        if (!PyErr_Occurred())         // callee may have set a python error itself
            PyErr_SetString(PyExc_ValueError, "nullptr result where temporary expected");
        return nullptr;
    }

// the result can then be bound
    PyObject* pyobj = BindCppObjectNoCast(value, fClass, CPPInstance::kIsValue);
    if (!pyobj)
        return nullptr;

// python ref counting will now control this object's life span
    ((CPPInstance*)pyobj)->PythonOwns();
    return pyobj;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// executor binds the result to the left-hand side, overwriting if an old object
    PyObject* result = BindCppObject((void*)GILCallR(method, self, ctxt), fClass);
    if (!result || !fAssignable)
        return result;
    else {
    // this generic code is quite slow compared to its C++ equivalent ...
        PyObject* assign = PyObject_GetAttr(result, PyStrings::gAssign);
        if (!assign) {
            PyErr_Clear();
            PyObject* descr = PyObject_Str(result);
            if (descr && CPyCppyy_PyUnicode_CheckExact(descr)) {
                PyErr_Format(PyExc_TypeError, "cannot assign to return object (%s)",
                             CPyCppyy_PyUnicode_AsString(descr));
            } else {
                PyErr_SetString(PyExc_TypeError, "cannot assign to result");
            }
            Py_XDECREF(descr);
            Py_DECREF(result);
            Py_DECREF(fAssignable); fAssignable = nullptr;
            return nullptr;
        }

        PyObject* res2 = PyObject_CallFunction(assign, const_cast<char*>("O"), fAssignable);

        Py_DECREF(assign);
        Py_DECREF(result);
        Py_DECREF(fAssignable); fAssignable = nullptr;

        if (res2) {
            Py_DECREF(res2);            // typically, *this from operator=()
            Py_RETURN_NONE;
        }

        return nullptr;
    }
}

//----------------------------------------------------------------------------
static inline PyObject* SetInstanceCheckError(PyObject* pyobj) {
    PyObject* pystr = PyObject_Str(pyobj);
    if (pystr) {
        PyErr_Format(PyExc_TypeError,
           "C++ object expected, got %s", CPyCppyy_PyUnicode_AsString(pystr));
        Py_DECREF(pystr);
    } else
        PyErr_SetString(PyExc_TypeError, "C++ object expected");
    return nullptr;
}

PyObject* CPyCppyy::InstancePtrPtrExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python C++ proxy object
// return ptr value
    if (fAssignable && !CPPInstance_Check(fAssignable))
        return SetInstanceCheckError(fAssignable);

    void** result = (void**)GILCallR(method, self, ctxt);
    if (!fAssignable)
        return BindCppObject((void*)result, fClass,
                             CPPInstance::kIsPtrPtr | CPPInstance::kIsReference);

    CPPInstance* cppinst = (CPPInstance*)fAssignable;
    *result = cppinst->fObject;

    Py_DECREF(fAssignable);
    fAssignable = nullptr;

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstancePtrRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct python C++ proxy object
// ignoring ref) return ptr value
    if (fAssignable && !CPPInstance_Check(fAssignable))
        return SetInstanceCheckError(fAssignable);

    void** result = (void**)GILCallR(method, self, ctxt);
    if (!fAssignable)
        return BindCppObject(*result, fClass);

    CPPInstance* cppinst = (CPPInstance*)fAssignable;
    *result = cppinst->fObject;

    Py_DECREF(fAssignable);
    fAssignable = nullptr;

    Py_RETURN_NONE;
}


//- smart pointers -----------------------------------------------------------
PyObject* CPyCppyy::SmartPtrExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// smart pointer executor
    Cppyy::TCppObject_t value = GILCallO(method, self, ctxt, fSmartPtrType);

    if (!value) {
        if (!PyErr_Occurred())          // callee may have set a python error itself
            PyErr_SetString(PyExc_ValueError, "NULL result where temporary expected");
        return nullptr;
    }

// fixme? - why doesn't this do the same as `self.__smartptr__().get()'
    CPPInstance* pyobj = (CPPInstance*)BindCppObjectNoCast(value, fRawPtrType);

    if (pyobj) {
        pyobj->SetSmartPtr(fSmartPtrType, fDereferencer);
        pyobj->PythonOwns();  // life-time control by python ref-counting
    }

    return (PyObject*)pyobj;
}

PyObject* CPyCppyy::SmartPtrPtrExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
    Cppyy::TCppObject_t value = GILCallR(method, self, ctxt);
    if (!value)
        return nullptr;

// todo: why doesn't this do the same as `self.__smartptr__().get()'
    CPPInstance* pyobj = (CPPInstance*)BindCppObjectNoCast(value, fRawPtrType);

    if (pyobj)
        pyobj->SetSmartPtr(fSmartPtrType, fDereferencer);

    return (PyObject*)pyobj;
}

PyObject* CPyCppyy::SmartPtrRefExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
    Cppyy::TCppObject_t value = GILCallR(method, self, ctxt);
    if (!value)
        return nullptr;

    //if (!fAssignable) {

// fixme? - why doesn't this do the same as `self.__smartptr__().get()'
    CPPInstance* pyobj = (CPPInstance*)BindCppObjectNoCast(value, fRawPtrType);

    if (pyobj)
        pyobj->SetSmartPtr(fSmartPtrType, fDereferencer);

    return (PyObject*)pyobj;

   // todo: assignment not done yet
   //
  /*} else {

       PyObject* result = BindCppObject((void*)value, fClass);

   // this generic code is quite slow compared to its C++ equivalent ...
       PyObject* assign = PyObject_GetAttrString(result, const_cast<char*>("__assign__"));
       if (!assign) {
           PyErr_Clear();
           PyObject* descr = PyObject_Str(result);
           if (descr && PyBytes_CheckExact(descr)) {
               PyErr_Format(PyExc_TypeError, "cannot assign to return object (%s)",
                   PyBytes_AS_STRING(descr));
           } else {
               PyErr_SetString(PyExc_TypeError, "cannot assign to result");
           }
           Py_XDECREF(descr);
           Py_DECREF(result);
           Py_DECREF(fAssignable); fAssignable = nullptr;
           return nullptr;
       }

       PyObject* res2 = PyObject_CallFunction(
           assign, const_cast<char*>("O"), fAssignable);


       Py_DECREF(assign);
       Py_DECREF(result);
       Py_DECREF(fAssignable); fAssignable = nullptr;

       if (res2) {
           Py_DECREF(res2);             // typically, *this from operator=()
           Py_RETURN_NONE;
       }

       return nullptr;
   }
   */
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceArrayExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, construct TupleOfInstances from
// return value
    long dims[] = {1, (long)fArraySize};
    return BindCppObjectArray((void*)GILCallR(method, self, ctxt), fClass, dims);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::ConstructorExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t klass, CallContext* ctxt)
{
// package return address in PyObject* for caller to handle appropriately (see
// CPPConstructor for the actual build of the PyObject)
    return (PyObject*)GILCallConstructor(method, (Cppyy::TCppType_t)klass, ctxt);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::PyObjectExecutor::Execute(
    Cppyy::TCppMethod_t method, Cppyy::TCppObject_t self, CallContext* ctxt)
{
// execute <method> with argument <self, ctxt>, return python object
    return (PyObject*)GILCallR(method, self, ctxt);
}


//- factories ----------------------------------------------------------------
CPyCppyy::Executor* CPyCppyy::CreateExecutor(const std::string& fullType)
{
// The matching of the fulltype to an executor factory goes through up to 4 levels:
//   1) full, qualified match
//   2) drop '&' as by ref/full type is often pretty much the same python-wise
//   3) C++ classes, either by ref/ptr or by value
//   4) additional special case for enums
//
// If all fails, void is used, which will cause the return type to be ignored on use

// an exactly matching executor is best
    ExecFactories_t::iterator h = gExecFactories.find(fullType);
    if (h != gExecFactories.end())
        return (h->second)();

// resolve typedefs etc.
    const std::string& resolvedType = Cppyy::ResolveName(fullType);

// a full, qualified matching executor is preferred
    if (resolvedType != fullType) {
         h = gExecFactories.find(resolvedType);
         if (h != gExecFactories.end())
              return (h->second)();
    }

//-- nothing? ok, collect information about the type and possible qualifiers/decorators
    bool isConst = strncmp(resolvedType.c_str(), "const", 5)  == 0;
    const std::string& cpd = Utility::Compound(resolvedType);
    std::string realType = TypeManip::clean_type(resolvedType, false);

// accept unqualified type (as python does not know about qualifiers)
    h = gExecFactories.find(realType + cpd);
    if (h != gExecFactories.end())
        return (h->second)();

// drop const, as that is mostly meaningless to python (with the exception
// of c-strings, but those are specialized in the converter map)
    if (isConst) {
        realType = TypeManip::remove_const(realType);
        h = gExecFactories.find(realType + cpd);
        if (h != gExecFactories.end())
            return (h->second)();
    }

//-- still nothing? try pointer instead of array (for builtins)
    if (cpd == "[]") {
        h = gExecFactories.find(realType + "*");
        if (h != gExecFactories.end())
            return (h->second)();           // TODO: use array size
    }

// C++ classes and special cases
    Executor* result = 0;
    if (Cppyy::TCppType_t klass = Cppyy::GetScope(realType)) {
        Cppyy::TCppType_t raw; Cppyy::TCppMethod_t deref;
        if (Cppyy::GetSmartPtrInfo(realType, raw, deref)) {
            if (cpd == "") {
                result = new SmartPtrExecutor(klass, raw, deref);
            } else if (cpd == "*") {
                result = new SmartPtrPtrExecutor(klass, raw, deref);
            } else if (cpd == "&") {
                result = new SmartPtrRefExecutor(klass, raw, deref);
            }
        }

        if (!result) {
            if (cpd == "")
                result = new InstanceExecutor(klass);
            else if (cpd == "&")
                result = new InstanceRefExecutor(klass);
            else if (cpd == "**" || cpd == "*[]" || cpd == "&*")
                result = new InstancePtrPtrExecutor(klass);
            else if (cpd == "*&")
                result = new InstancePtrRefExecutor(klass);
            else if (cpd == "[]") {
                Py_ssize_t asize = Utility::ArraySize(resolvedType);
                if (0 < asize)
                    result = new InstanceArrayExecutor(klass, asize);
                else
                    result = new InstancePtrRefExecutor(klass);
            } else
                result = new InstancePtrExecutor(klass);
        }
    } else {
    // unknown: void* may work ("user knows best"), void will fail on use of return value
        h = (cpd == "") ? gExecFactories.find("void") : gExecFactories.find("void*");
    }

    if (!result && h != gExecFactories.end())
    // executor factory available, use it to create executor
        result = (h->second)();

   return result;                  // may still be null
}


//----------------------------------------------------------------------------
namespace {

using namespace CPyCppyy;

#define WSTRING "basic_string<wchar_t,char_traits<wchar_t>,allocator<wchar_t> >"

struct InitExecFactories_t {
public:
    InitExecFactories_t() {
    // load all executor factories in the global map 'gExecFactories'
        CPyCppyy::ExecFactories_t& gf = gExecFactories;

    // factories for built-ins
        gf["bool"] =                        (ef_t)+[]() { return new BoolExecutor{}; };
        gf["bool&"] =                       (ef_t)+[]() { return new BoolRefExecutor{}; };
        gf["const bool&"] =                 (ef_t)+[]() { return new BoolConstRefExecutor{}; };
        gf["char"] =                        (ef_t)+[]() { return new CharExecutor{}; };
        gf["signed char"] =                 (ef_t)+[]() { return new CharExecutor{}; };
        gf["unsigned char"] =               (ef_t)+[]() { return new UCharExecutor{}; };
        gf["char&"] =                       (ef_t)+[]() { return new CharRefExecutor{}; };
        gf["signed char&"] =                (ef_t)+[]() { return new CharRefExecutor{}; };
        gf["unsigned char&"] =              (ef_t)+[]() { return new UCharRefExecutor{}; };
        gf["const char&"] =                 (ef_t)+[]() { return new CharConstRefExecutor{}; };
        gf["const signed char&"] =          (ef_t)+[]() { return new CharConstRefExecutor{}; };
        gf["const unsigned char&"] =        (ef_t)+[]() { return new UCharConstRefExecutor{}; };
        gf["wchar_t"] =                     (ef_t)+[]() { return new WCharExecutor{}; };
        gf["int8_t"] =                      (ef_t)+[]() { return new Int8Executor{}; };
        gf["int8_t&"] =                     (ef_t)+[]() { return new Int8RefExecutor{}; };
        gf["const int8_t&"] =               (ef_t)+[]() { return new Int8RefExecutor{}; };
        gf["uint8_t"] =                     (ef_t)+[]() { return new UInt8Executor{}; };
        gf["uint8_t&"] =                    (ef_t)+[]() { return new UInt8RefExecutor{}; };
        gf["const uint8_t&"] =              (ef_t)+[]() { return new UInt8RefExecutor{}; };
        gf["short"] =                       (ef_t)+[]() { return new ShortExecutor{}; };
        gf["short&"] =                      (ef_t)+[]() { return new ShortRefExecutor{}; };
        gf["unsigned short"] =              (ef_t)+[]() { return new IntExecutor{}; };
        gf["unsigned short&"] =             (ef_t)+[]() { return new UShortRefExecutor{}; };
        gf["int"] =                         (ef_t)+[]() { return new IntExecutor{}; };
        gf["int&"] =                        (ef_t)+[]() { return new IntRefExecutor{}; };
        gf["unsigned int"] =                (ef_t)+[]() { return new ULongExecutor{}; };
        gf["unsigned int&"] =               (ef_t)+[]() { return new UIntRefExecutor{}; };
        gf["internal_enum_type_t"] =        (ef_t)+[]() { return new IntExecutor{}; };
        gf["internal_enum_type_t&"] =       (ef_t)+[]() { return new IntRefExecutor{}; };
        gf["long"] =                        (ef_t)+[]() { return new LongExecutor{}; };
        gf["long&"] =                       (ef_t)+[]() { return new LongRefExecutor{}; };
        gf["unsigned long"] =               (ef_t)+[]() { return new ULongExecutor{}; };
        gf["unsigned long&"] =              (ef_t)+[]() { return new ULongRefExecutor{}; };
        gf["long long"] =                   (ef_t)+[]() { return new LongLongExecutor{}; };
        gf["Long64_t"] =                    (ef_t)+[]() { return new LongLongExecutor{}; };
        gf["long long&"] =                  (ef_t)+[]() { return new LongLongRefExecutor{}; };
        gf["Long64_t&"] =                   (ef_t)+[]() { return new LongLongRefExecutor{}; };
        gf["unsigned long long"] =          (ef_t)+[]() { return new ULongLongExecutor{}; };
        gf["ULong64_t"] =                   (ef_t)+[]() { return new ULongLongExecutor{}; };
        gf["unsigned long long&"] =         (ef_t)+[]() { return new ULongLongRefExecutor{}; };
        gf["ULong64_t&"] =                  (ef_t)+[]() { return new ULongLongRefExecutor{}; };

        gf["float"] =                       (ef_t)+[]() { return new FloatExecutor{}; };
        gf["float&"] =                      (ef_t)+[]() { return new FloatRefExecutor{}; };
        gf["Float16_t"] =                   (ef_t)+[]() { return new FloatExecutor{}; };
        gf["Float16_t&"] =                  (ef_t)+[]() { return new FloatRefExecutor{}; };
        gf["double"] =                      (ef_t)+[]() { return new DoubleExecutor{}; };
        gf["double&"] =                     (ef_t)+[]() { return new DoubleRefExecutor{}; };
        gf["Double32_t"] =                  (ef_t)+[]() { return new DoubleExecutor{}; };
        gf["Double32_t&"] =                 (ef_t)+[]() { return new DoubleRefExecutor{}; };
        gf["long double"] =                 (ef_t)+[]() { return new LongDoubleExecutor{}; }; // TODO: lost precision
        gf["long double&"] =                (ef_t)+[]() { return new LongDoubleRefExecutor{}; };
        gf["void"] =                        (ef_t)+[]() { return new VoidExecutor{}; };

    // pointer/array factories
        gf["void*"] =                       (ef_t)+[]() { return new VoidArrayExecutor{}; };
        gf["bool*"] =                       (ef_t)+[]() { return new BoolArrayExecutor{}; };
        gf["const unsigned char*"] =        (ef_t)+[]() { return new UCharArrayExecutor{}; };
        gf["unsigned char*"] =              (ef_t)+[]() { return new UCharArrayExecutor{}; };
        gf["short*"] =                      (ef_t)+[]() { return new ShortArrayExecutor{}; };
        gf["unsigned short*"] =             (ef_t)+[]() { return new UShortArrayExecutor{}; };
        gf["int*"] =                        (ef_t)+[]() { return new IntArrayExecutor{}; };
        gf["unsigned int*"] =               (ef_t)+[]() { return new UIntArrayExecutor{}; };
        gf["internal_enum_type_t*"] =       (ef_t)+[]() { return new UIntArrayExecutor{}; };
        gf["long*"] =                       (ef_t)+[]() { return new LongArrayExecutor{}; };
        gf["unsigned long*"] =              (ef_t)+[]() { return new ULongArrayExecutor{}; };
        gf["long long*"] =                  (ef_t)+[]() { return new LLongArrayExecutor{}; };
        gf["Long64_t*"] =                   (ef_t)+[]() { return new LLongArrayExecutor{}; };
        gf["unsigned long long*"] =         (ef_t)+[]() { return new ULLongArrayExecutor{}; };
        gf["ULong64_t*"] =                  (ef_t)+[]() { return new ULLongArrayExecutor{}; };
        gf["float*"] =                      (ef_t)+[]() { return new FloatArrayExecutor{}; };
        gf["double*"] =                     (ef_t)+[]() { return new DoubleArrayExecutor{}; };
        gf["complex<float>*"] =             (ef_t)+[]() { return new ComplexFArrayExecutor{}; };
        gf["complex<double>*"] =            (ef_t)+[]() { return new ComplexDArrayExecutor{}; };
        gf["complex<int>*"] =               (ef_t)+[]() { return new ComplexIArrayExecutor{}; };
        gf["complex<long>*"] =              (ef_t)+[]() { return new ComplexLArrayExecutor{}; };

    // factories for special cases
        gf["const char*"] =                 (ef_t)+[]() { return new CStringExecutor{}; };
        gf["char*"] =                       (ef_t)+[]() { return new CStringExecutor{}; };
        gf["const signed char*"] =          (ef_t)+[]() { return new CStringExecutor{}; };
        gf["signed char*"] =                (ef_t)+[]() { return new CStringExecutor{}; };
        gf["wchar_t*"] =                    (ef_t)+[]() { return new WCStringExecutor{}; };
        gf["std::string"] =                 (ef_t)+[]() { return new STLStringExecutor{}; };
        gf["string"] =                      (ef_t)+[]() { return new STLStringExecutor{}; };
        gf["std::string&"] =                (ef_t)+[]() { return new STLStringRefExecutor{}; };
        gf["string&"] =                     (ef_t)+[]() { return new STLStringRefExecutor{}; };
        gf["std::wstring"] =                (ef_t)+[]() { return new STLWStringExecutor{}; };
        gf["std::" WSTRING] =               (ef_t)+[]() { return new STLWStringExecutor{}; };
        gf[WSTRING] =                       (ef_t)+[]() { return new STLWStringExecutor{}; };
        gf["complex<double>"] =             (ef_t)+[]() { return new ComplexDExecutor{}; };
        gf["complex<double>&"] =            (ef_t)+[]() { return new ComplexDRefExecutor{}; };
        gf["__init__"] =                    (ef_t)+[]() { return new ConstructorExecutor{}; };
        gf["PyObject*"] =                   (ef_t)+[]() { return new PyObjectExecutor{}; };
        gf["_object*"] =                    (ef_t)+[]() { return new PyObjectExecutor{}; };
        gf["FILE*"] =                       (ef_t)+[]() { return new VoidArrayExecutor{}; };
    }
} initExecvFactories_;

} // unnamed namespace
