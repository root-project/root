// Bindings
#include "CPyCppyy.h"
#include "DeclareConverters.h"
#include "CallContext.h"
#include "CPPExcInstance.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CustomPyTypes.h"
#include "LowLevelViews.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TemplateProxy.h"
#include "TupleOfInstances.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <complex>
#include <limits.h>
#include <stddef.h>      // for ptrdiff_t
#include <string.h>
#include <algorithm>
#include <array>
#include <locale>        // for wstring_convert
#include <regex>
#include <utility>
#include <sstream>
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
#include <cstddef>
#include <string_view>
#endif
// codecvt does not exist for gcc4.8.5 and is in principle deprecated; it is
// only used in py2 for char -> wchar_t conversion for std::wstring; if not
// available, the conversion is done through Python (requires an extra copy)
#if PY_VERSION_HEX < 0x03000000
#if defined(__GNUC__) && !defined(__APPLE__)
# if __GNUC__ > 4 && __has_include("codecvt")
# include <codecvt>
# define HAS_CODECVT 1
# endif
#else
#include <codecvt>
#define HAS_CODECVT 1
#endif
#endif // py2


//- data _____________________________________________________________________
namespace CPyCppyy {

// factories
    typedef std::map<std::string, cf_t> ConvFactories_t;
    static ConvFactories_t gConvFactories;

// special objects
    extern PyObject* gNullPtrObject;
    extern PyObject* gDefaultObject;

// regular expression for matching function pointer
    static std::regex s_fnptr("\\((\\w*:*)*\\*&*\\)");
}

// Define our own PyUnstable_Object_IsUniqueReferencedTemporary function if the
// Python version is lower than 3.14, the version where that function got introduced.
#if PY_VERSION_HEX < 0x030e0000
#if PY_VERSION_HEX < 0x03000000
const Py_ssize_t MOVE_REFCOUNT_CUTOFF = 1;
#elif PY_VERSION_HEX < 0x03080000
// p3 has at least 2 ref-counts, as contrary to p2, it will create a descriptor
// copy for the method holding self in the case of __init__; but there can also
// be a reference held by the frame object, which is indistinguishable from a
// local variable reference, so the cut-off has to remain 2.
const Py_ssize_t MOVE_REFCOUNT_CUTOFF = 2;
#else
// since py3.8, vector calls behave again as expected
const Py_ssize_t MOVE_REFCOUNT_CUTOFF = 1;
#endif
inline bool PyUnstable_Object_IsUniqueReferencedTemporary(PyObject *pyobject) {
    return Py_REFCNT(pyobject) <= MOVE_REFCOUNT_CUTOFF;
}
#endif

//- pretend-ctypes helpers ---------------------------------------------------
struct CPyCppyy_tagCDataObject {       // non-public (but stable)
    PyObject_HEAD
    char* b_ptr;
    int   b_needsfree;
};

struct CPyCppyy_tagPyCArgObject {      // not public (but stable; note that older
    PyObject_HEAD                      // Pythons protect 'D' with HAVE_LONG_LONG)
    void* pffi_type;
    char tag;
    union {                            // for convenience, kept only relevant vals
        long long q;
        long double D;
        void *p;
    } value;
    PyObject* obj;
};

// indices of ctypes types into the array caches (note that c_complex and c_fcomplex
// do not exist as types in ctypes)
#define ct_c_bool        0
#define ct_c_char        1
#define ct_c_shar        1
#define ct_c_wchar       2
#define ct_c_byte        3
#define ct_c_int8        3
#define ct_c_ubyte       4
#define ct_c_uchar       4
#define ct_c_uint8       4
#define ct_c_short       5
#define ct_c_ushort      6
#define ct_c_uint16      7
#define ct_c_int         8
#define ct_c_uint        9
#define ct_c_uint32     10
#define ct_c_long       11
#define ct_c_ulong      12
#define ct_c_longlong   13
#define ct_c_ulonglong  14
#define ct_c_float      15
#define ct_c_double     16
#define ct_c_longdouble 17
#define ct_c_char_p     18
#define ct_c_wchar_p    19
#define ct_c_void_p     20
#define ct_c_fcomplex   21
#define ct_c_complex    22
#define ct_c_pointer    23
#define ct_c_funcptr    24
#define ct_c_int16      25
#define ct_c_int32      26
#define NTYPES          27

static std::array<const char*, NTYPES> gCTypesNames = {
    "c_bool", "c_char", "c_wchar", "c_byte", "c_ubyte", "c_short", "c_ushort", "c_uint16",
    "c_int", "c_uint", "c_uint32", "c_long", "c_ulong", "c_longlong", "c_ulonglong",
    "c_float", "c_double", "c_longdouble",
    "c_char_p", "c_wchar_p", "c_void_p", "c_fcomplex", "c_complex",
    "_Pointer", "_CFuncPtr" };
static std::array<PyTypeObject*, NTYPES> gCTypesTypes;
static std::array<PyTypeObject*, NTYPES> gCTypesPtrTypes;

// Both GetCTypesType and GetCTypesPtrType, rely on the ctypes module itself
// caching the types (thus also making them unique), so no ref-count is needed.
// Further, by keeping a ref-count on the module, it won't be off-loaded until
// the 2nd cleanup cycle.
static PyTypeObject* GetCTypesType(int nidx)
{
    static PyObject* ctmod = PyImport_ImportModule("ctypes");   // ref-count kept
    if (!ctmod) {
        PyErr_Clear();
        return nullptr;
    }
    PyTypeObject* ct_t = gCTypesTypes[nidx];
    if (!ct_t) {
        ct_t = (PyTypeObject*)PyObject_GetAttrString(ctmod, gCTypesNames[nidx]);
        if (!ct_t) PyErr_Clear();
        else {
            gCTypesTypes[nidx] = ct_t;
            Py_DECREF(ct_t);
        }
    }
    return ct_t;
}

static PyTypeObject* GetCTypesPtrType(int nidx)
{
    static PyObject* ctmod = PyImport_ImportModule("ctypes");   // ref-count kept
    if (!ctmod) {
        PyErr_Clear();
        return nullptr;
    }
    PyTypeObject* cpt_t = gCTypesPtrTypes[nidx];
    if (!cpt_t) {
        if (strcmp(gCTypesNames[nidx], "c_char") == 0)  {
            cpt_t = (PyTypeObject*)PyObject_GetAttrString(ctmod, "c_char_p");
        } else {
            PyObject* ct_t = (PyObject*)GetCTypesType(nidx);
            if (ct_t) {
                PyObject* ptrcreat = PyObject_GetAttrString(ctmod, "POINTER");
                cpt_t = (PyTypeObject*)PyObject_CallFunctionObjArgs(ptrcreat, ct_t, NULL);
                Py_DECREF(ptrcreat);
            }
        }
        if (cpt_t) {
            gCTypesPtrTypes[nidx] = cpt_t;
            Py_DECREF(cpt_t);
        }
    }
    return cpt_t;
}

static bool IsPyCArgObject(PyObject* pyobject)
{
    static PyTypeObject* pycarg_type = nullptr;
    if (!pycarg_type) {
        PyObject* ctmod = PyImport_ImportModule("ctypes");
        if (!ctmod) PyErr_Clear();
        else {
             PyTypeObject* ct_t = (PyTypeObject*)PyObject_GetAttrString(ctmod, "c_int");
             PyObject* cobj = ct_t->tp_new(ct_t, nullptr, nullptr);
             PyObject* byref = PyObject_GetAttrString(ctmod, "byref");
             PyObject* pyptr = PyObject_CallFunctionObjArgs(byref, cobj, NULL);
             Py_DECREF(byref); Py_DECREF(cobj); Py_DECREF(ct_t);
             pycarg_type = Py_TYPE(pyptr);  // static, no ref-count needed
             Py_DECREF(pyptr);
             Py_DECREF(ctmod);
        }
    }
    return Py_TYPE(pyobject) == pycarg_type;
}

#if PY_VERSION_HEX < 0x30d0000
static bool IsCTypesArrayOrPointer(PyObject* pyobject)
{
    static PyTypeObject* cstgdict_type = nullptr;
    if (!cstgdict_type) {
    // get any pointer type to initialize the extended dictionary type
        PyTypeObject* ct_int = GetCTypesType(ct_c_int);
        if (ct_int && ct_int->tp_dict) {
            cstgdict_type = Py_TYPE(ct_int->tp_dict);
        }
    }

    PyTypeObject* pytype = Py_TYPE(pyobject);
    if (pytype->tp_dict && Py_TYPE(pytype->tp_dict) == cstgdict_type)
        return true;
    return false;
}
#else
// the internals of ctypes have been redone, requiring a more complex checking
namespace {

typedef struct {
    PyTypeObject *DictRemover_Type;
    PyTypeObject *PyCArg_Type;
    PyTypeObject *PyCField_Type;
    PyTypeObject *PyCThunk_Type;
    PyTypeObject *StructParam_Type;
    PyTypeObject *PyCType_Type;
    PyTypeObject *PyCStructType_Type;
    PyTypeObject *UnionType_Type;
    PyTypeObject *PyCPointerType_Type;
// ... unused fields omitted ...
} _cppyy_ctypes_state;

} // unnamed namespace

static bool IsCTypesArrayOrPointer(PyObject* pyobject)
{
    static _cppyy_ctypes_state* state = nullptr;
    if (!state) {
        PyObject* ctmod = PyImport_AddModule("_ctypes");   // the extension module, not the Python one
        if (ctmod)
            state = (_cppyy_ctypes_state*)PyModule_GetState(ctmod);
    }

    // verify for object types that have a C payload
    if (state && (PyObject_IsInstance((PyObject*)Py_TYPE(pyobject), (PyObject*)state->PyCType_Type) ||
                  PyObject_IsInstance((PyObject*)Py_TYPE(pyobject), (PyObject*)state->PyCPointerType_Type))) {
        return true;
    }

    return false;
}
#endif


//- helper to establish life lines -------------------------------------------
static inline bool SetLifeLine(PyObject* holder, PyObject* target, intptr_t ref)
{
// set a lifeline from on the holder to the target, using the ref as label
    if (!holder) return false;

// 'ref' is expected to be the converter address or data memory location, so
// that the combination of holder and ref is unique, but also identifiable for
// reuse when the C++ side is being overwritten
    std::ostringstream attr_name;
    attr_name << "__" << ref;
    auto res = PyObject_SetAttrString(holder, (char*)attr_name.str().c_str(), target);
    return res != -1;
}

static bool HasLifeLine(PyObject* holder, intptr_t ref)
{
// determine if a lifeline was previously set for the ref on the holder
   if (!holder) return false;

    std::ostringstream attr_name;
    attr_name << "__" << ref;
    PyObject* res = PyObject_GetAttrString(holder, (char*)attr_name.str().c_str());

    if (res) {
        Py_DECREF(res);
        return true;
    }

    PyErr_Clear();
    return false;
}


//- helper to work with both CPPInstance and CPPExcInstance ------------------
static inline CPyCppyy::CPPInstance* GetCppInstance(
    PyObject* pyobject, Cppyy::TCppType_t klass = (Cppyy::TCppType_t)0, bool accept_rvalue = false)
{
    using namespace CPyCppyy;
    if (CPPInstance_Check(pyobject))
        return (CPPInstance*)pyobject;
    if (CPPExcInstance_Check(pyobject))
        return (CPPInstance*)((CPPExcInstance*)pyobject)->fCppInstance;

// this is not a C++ proxy; allow custom cast to C++
    PyObject* castobj = PyObject_CallMethodNoArgs(pyobject, PyStrings::gCastCpp);
    if (castobj) {
        if (CPPInstance_Check(castobj))
            return (CPPInstance*)castobj;
        else if (klass && PyTuple_CheckExact(castobj)) {
        // allow implicit conversion from a tuple of arguments
            PyObject* pyclass = GetScopeProxy(klass);
            if (pyclass) {
                CPPInstance* pytmp = (CPPInstance*)PyObject_Call(pyclass, castobj, NULL);
                Py_DECREF(pyclass);
                if (CPPInstance_Check(pytmp)) {
                    if (accept_rvalue)
                        pytmp->fFlags |= CPPInstance::kIsRValue;
                    Py_DECREF(castobj);
                    return pytmp;
                }
                Py_XDECREF(pytmp);
            }
        }

        Py_DECREF(castobj);
        return nullptr;
    }

    PyErr_Clear();
    return nullptr;
}


//- custom helpers to check ranges -------------------------------------------
static inline bool ImplicitBool(PyObject* pyobject, CPyCppyy::CallContext* ctxt)
{
    using namespace CPyCppyy;
    if (!AllowImplicit(ctxt) && PyBool_Check(pyobject)) {
        if (!NoImplicit(ctxt)) ctxt->fFlags |= CallContext::kHaveImplicit;
        return false;
    }
    return true;
}

static inline bool StrictBool(PyObject* pyobject, CPyCppyy::CallContext* ctxt)
{
    using namespace CPyCppyy;
    if (!AllowImplicit(ctxt) && !PyBool_Check(pyobject)) {
        if (!NoImplicit(ctxt)) ctxt->fFlags |= CallContext::kHaveImplicit;
        return false;
    }
    return true;
}

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


// range-checking python integer to C++ integer conversion (prevents p2.7 silent conversions)
#define CPPYY_PYLONG_AS_TYPE(name, type, limit_low, limit_high)              \
static inline type CPyCppyy_PyLong_As##name(PyObject* pyobject)              \
{                                                                            \
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {                \
        if (pyobject == CPyCppyy::gDefaultObject)                            \
            return (type)0;                                                  \
        PyErr_SetString(PyExc_TypeError, #type" conversion expects an integer object");\
        return (type)-1;                                                     \
    }                                                                        \
    long l = PyLong_AsLong(pyobject);                                        \
    if (l < limit_low || limit_high < l) {                                   \
        PyErr_Format(PyExc_ValueError, "integer %ld out of range for "#type, l);\
        return (type)-1;                                                     \
    }                                                                        \
    return (type)l;                                                          \
}

CPPYY_PYLONG_AS_TYPE(UInt8,     uint8_t,        0,         UCHAR_MAX)
CPPYY_PYLONG_AS_TYPE(Int8,      int8_t,         SCHAR_MIN, SCHAR_MAX)
CPPYY_PYLONG_AS_TYPE(UInt16,    uint16_t,       0,         UINT16_MAX)
CPPYY_PYLONG_AS_TYPE(Int16,     int16_t,        INT16_MIN, INT16_MAX)
CPPYY_PYLONG_AS_TYPE(UInt32,    uint32_t,       0,         UINT32_MAX)
CPPYY_PYLONG_AS_TYPE(Int32,     int32_t,        INT32_MIN, INT32_MAX)
CPPYY_PYLONG_AS_TYPE(UShort,    unsigned short, 0,         USHRT_MAX)
CPPYY_PYLONG_AS_TYPE(Short,     short,          SHRT_MIN,  SHRT_MAX)
CPPYY_PYLONG_AS_TYPE(StrictInt, int,            INT_MIN,   INT_MAX)

static inline long CPyCppyy_PyLong_AsStrictLong(PyObject* pyobject)
{
// strict python integer to C++ long integer conversion

// prevent float -> long (see CPyCppyy_PyLong_AsStrictInt)
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        if (pyobject == CPyCppyy::gDefaultObject)
            return (long)0;
        PyErr_SetString(PyExc_TypeError, "int/long conversion expects an integer object");
        return (long)-1;
    }

    return (long)PyLong_AsLong(pyobject);   // already does long range check
}

static inline PY_LONG_LONG CPyCppyy_PyLong_AsStrictLongLong(PyObject* pyobject)
{
// strict python integer to C++ long long integer conversion

// prevent float -> long (see CPyCppyy_PyLong_AsStrictInt)
    if (!(PyLong_Check(pyobject) || PyInt_Check(pyobject))) {
        if (pyobject == CPyCppyy::gDefaultObject)
            return (PY_LONG_LONG)0;
        PyErr_SetString(PyExc_TypeError, "int/long conversion expects an integer object");
        return (PY_LONG_LONG)-1;
    }

    return PyLong_AsLongLong(pyobject);     // already does long range check
}


//- helper for pointer/array/reference conversions ---------------------------
static inline bool CArraySetArg(
    PyObject* pyobject, CPyCppyy::Parameter& para, char tc, int size, bool check=true)
{
// general case of loading a C array pointer (void* + type code) as function argument
    if (pyobject == CPyCppyy::gNullPtrObject || pyobject == CPyCppyy::gDefaultObject)
        para.fValue.fVoidp = nullptr;
    else {
        Py_ssize_t buflen = CPyCppyy::Utility::GetBuffer(pyobject, tc, size, para.fValue.fVoidp, check);
        if (!buflen) {
        // stuck here as it's the least common
            if (CPyCppyy_PyLong_AsStrictInt(pyobject) == 0)
                para.fValue.fVoidp = nullptr;
            else {
                PyErr_Format(PyExc_TypeError,     // ValueError?
                    "could not convert argument to buffer or nullptr");
                return false;
            }
        }
    }
    para.fTypeCode = 'p';
    return true;
}


//- helper for implicit conversions ------------------------------------------
static inline CPyCppyy::CPPInstance* ConvertImplicit(Cppyy::TCppType_t klass,
    PyObject* pyobject, CPyCppyy::Parameter& para, CPyCppyy::CallContext* ctxt, bool manage=true)
{
    using namespace CPyCppyy;

// filter out copy and move constructors
    if (IsConstructor(ctxt->fFlags) && klass == ctxt->fCurScope && ctxt->GetSize() == 1)
        return nullptr;

// only proceed if implicit conversions are allowed (in "round 2") or if the
// argument is exactly a tuple or list, as these are the equivalent of
// initializer lists and thus "syntax" not a conversion
    if (!AllowImplicit(ctxt)) {
        PyTypeObject* pytype = (PyTypeObject*)Py_TYPE(pyobject);
        if (!(pytype == &PyList_Type || pytype == &PyTuple_Type)) {// || !CPPInstance_Check(pyobject))) {
            if (!NoImplicit(ctxt)) ctxt->fFlags |= CallContext::kHaveImplicit;
            return nullptr;
        }
    }

// exercise implicit conversion
    PyObject* pyscope = CreateScopeProxy(klass);
    if (!CPPScope_Check(pyscope)) {
        Py_XDECREF(pyscope);
        return nullptr;
    }

// call constructor of argument type to attempt implicit conversion (disallow any
// implicit conversions by the scope's constructor itself)
    PyObject* args = PyTuple_New(1);
    Py_INCREF(pyobject); PyTuple_SET_ITEM(args, 0, pyobject);

    ((CPPScope*)pyscope)->fFlags |= CPPScope::kNoImplicit;
    CPPInstance* pytmp = (CPPInstance*)PyObject_Call(pyscope, args, NULL);
    if (!pytmp && PyTuple_CheckExact(pyobject)) {
    // special case: allow implicit conversion from given set of arguments in tuple
        PyErr_Clear();
        pytmp = (CPPInstance*)PyObject_Call(pyscope, pyobject, NULL);
    }
    ((CPPScope*)pyscope)->fFlags &= ~CPPScope::kNoImplicit;

    Py_DECREF(args);
    Py_DECREF(pyscope);

    if (pytmp) {
    // implicit conversion succeeded!
        if (manage) ctxt->AddTemporary((PyObject*)pytmp);
        para.fValue.fVoidp = pytmp->GetObjectRaw();
        para.fTypeCode = 'V';
        return pytmp;
    }

    PyErr_Clear();
    return nullptr;
}


//- base converter implementation --------------------------------------------
CPyCppyy::Converter::~Converter()
{
    /* empty */
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::Converter::FromMemory(void*)
{
// could happen if no derived class override
    PyErr_SetString(PyExc_TypeError, "C++ type cannot be converted from memory");
    return nullptr;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Converter::ToMemory(PyObject*, void*, PyObject* /* ctxt */)
{
// could happen if no derived class override
    PyErr_SetString(PyExc_TypeError, "C++ type cannot be converted to memory");
    return false;
}


//- helper macro's -----------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONVERTER_BODY(name, type, stype, ctype, F1, F2, tc)\
/* convert <pyobject> to C++ 'type', set arg for call */                     \
    type val = (type)F2(pyobject);                                           \
    if (val == (type)-1 && PyErr_Occurred()) {                               \
        static PyTypeObject* ctypes_type = nullptr;                          \
        if (!ctypes_type) {                                                  \
            auto error = CPyCppyy::Utility::FetchPyError();                  \
            ctypes_type = GetCTypesType(ct_##ctype);                         \
            CPyCppyy::Utility::RestorePyError(error);                        \
        }                                                                    \
        if (Py_TYPE(pyobject) == ctypes_type) {                              \
            PyErr_Clear();                                                   \
            val = *((type*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr);     \
        } else if (pyobject == CPyCppyy::gDefaultObject) {                   \
            PyErr_Clear();                                                   \
            val = (type)0;                                                   \
        } else                                                               \
            return false;                                                    \
    }                                                                        \
    para.fValue.f##name = val;                                               \
    para.fTypeCode = tc;                                                     \
    return true;

#define CPPYY_IMPL_BASIC_CONVERTER_METHODS(name, type, stype, ctype, F1, F2) \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
    return F1((stype)*((type*)address));                                     \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(                                    \
    PyObject* value, void* address, PyObject* /* ctxt */)                    \
{                                                                            \
    type s = (type)F2(value);                                                \
    if (s == (type)-1 && PyErr_Occurred()) {                                 \
        if (value == CPyCppyy::gDefaultObject) {                             \
            PyErr_Clear();                                                   \
            s = (type)0;                                                     \
        } else                                                               \
            return false;                                                    \
    }                                                                        \
    *((type*)address) = (type)s;                                             \
    return true;                                                             \
}

#define CPPYY_IMPL_BASIC_CONVERTER_NI(name, type, stype, ctype, F1, F2, tc)  \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    if (!StrictBool(pyobject, ctxt))                                         \
        return false;                                                        \
    CPPYY_IMPL_BASIC_CONVERTER_BODY(name, type, stype, ctype, F1, F2, tc)    \
}                                                                            \
CPPYY_IMPL_BASIC_CONVERTER_METHODS(name, type, stype, ctype, F1, F2)

#define CPPYY_IMPL_BASIC_CONVERTER_IB(name, type, stype, ctype, F1, F2, tc)  \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    if (!ImplicitBool(pyobject, ctxt))                                       \
        return false;                                                        \
    CPPYY_IMPL_BASIC_CONVERTER_BODY(name, type, stype, ctype, F1, F2, tc)    \
}                                                                            \
CPPYY_IMPL_BASIC_CONVERTER_METHODS(name, type, stype, ctype, F1, F2)

#define CPPYY_IMPL_BASIC_CONVERTER_NB(name, type, stype, ctype, F1, F2, tc)  \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* /*ctxt*/)              \
{                                                                            \
    if (PyBool_Check(pyobject))                                              \
        return false;                                                        \
    CPPYY_IMPL_BASIC_CONVERTER_BODY(name, type, stype, ctype, F1, F2, tc)    \
}                                                                            \
CPPYY_IMPL_BASIC_CONVERTER_METHODS(name, type, stype, ctype, F1, F2)

//----------------------------------------------------------------------------
static inline int ExtractChar(PyObject* pyobject, const char* tname, int low, int high)
{
    int lchar = -1;
    if (PyBytes_Check(pyobject)) {
        if (PyBytes_GET_SIZE(pyobject) == 1)
            lchar = (int)(PyBytes_AsString(pyobject)[0]);
        else
            PyErr_Format(PyExc_ValueError, "%s expected, got bytes of size " PY_SSIZE_T_FORMAT,
                tname, PyBytes_GET_SIZE(pyobject));
    } else if (CPyCppyy_PyText_Check(pyobject)) {
        if (CPyCppyy_PyText_GET_SIZE(pyobject) == 1)
            lchar = (int)(CPyCppyy_PyText_AsString(pyobject)[0]);
        else
            PyErr_Format(PyExc_ValueError, "%s expected, got str of size " PY_SSIZE_T_FORMAT,
                tname, CPyCppyy_PyText_GET_SIZE(pyobject));
    } else if (pyobject == CPyCppyy::gDefaultObject) {
        lchar = (int)'\0';
    } else if (!PyFloat_Check(pyobject)) {   // don't allow truncating conversion
        lchar = (int)PyLong_AsLong(pyobject);
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
#define CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(name, ctype)                     \
PyObject* CPyCppyy::name##RefConverter::FromMemory(void* ptr)                \
{                                                                            \
/* convert a reference to int to Python through ctypes pointer object */     \
    PyTypeObject* ctypes_type = GetCTypesType(ct_##ctype);                   \
    if (!ctypes_type) {                                                      \
        PyErr_SetString(PyExc_RuntimeError, "no ctypes available");          \
        return nullptr;                                                      \
    }                                                                        \
    PyObject* ref = ctypes_type->tp_new(ctypes_type, nullptr, nullptr);      \
    ((CPyCppyy_tagCDataObject*)ref)->b_ptr = (char*)ptr;                     \
    ((CPyCppyy_tagCDataObject*)ref)->b_needsfree = 0;                        \
    return ref;                                                              \
}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONST_REFCONVERTER(name, type, ctype, F1)           \
bool CPyCppyy::Const##name##RefConverter::SetArg(                            \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
    type val = (type)F1(pyobject);                                           \
    if (val == (type)-1 && PyErr_Occurred()) {                               \
        if (pyobject == CPyCppyy::gDefaultObject) {                          \
            PyErr_Clear();                                                   \
            val = (type)0;                                                   \
        } else                                                               \
            return false;                                                    \
    }                                                                        \
    para.fValue.f##name = val;                                               \
    para.fRef = &para.fValue.f##name;                                        \
    para.fTypeCode = 'r';                                                    \
    return true;                                                             \
}                                                                            \
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(Const##name, ctype)

//----------------------------------------------------------------------------
#define CPPYY_IMPL_BASIC_CONST_CHAR_REFCONVERTER(name, type, ctype, low, high)\
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
}                                                                            \
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(Const##name, ctype)


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
    /* return char in "native" str type as that's more natural in use */     \
    return CPyCppyy_PyText_FromFormat("%c", *((type*)address));              \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(                                    \
    PyObject* value, void* address, PyObject* /* ctxt */)                    \
{                                                                            \
    Py_ssize_t len;                                                          \
    const char* cstr = nullptr;                                              \
    if (PyBytes_Check(value))                                                \
        PyBytes_AsStringAndSize(value, (char**)&cstr, &len);                 \
    else                                                                     \
        cstr = CPyCppyy_PyText_AsStringAndSize(value, &len);                 \
    if (cstr) {                                                              \
        if (len != 1) {                                                      \
            PyErr_Format(PyExc_TypeError, #type" expected, got string of size %zd", len);\
            return false;                                                    \
        }                                                                    \
        *((type*)address) = (type)cstr[0];                                   \
    } else {                                                                 \
        PyErr_Clear();                                                       \
        long l = PyLong_AsLong(value);                                       \
        if (l == -1 && PyErr_Occurred()) {                                   \
            if (value == CPyCppyy::gDefaultObject) {                         \
                PyErr_Clear();                                               \
                l = (long)0;                                                 \
            } else                                                           \
                return false;                                                \
        }                                                                    \
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
CPPYY_IMPL_BASIC_CONVERTER_IB(Long, long, long, c_long, PyLong_FromLong, CPyCppyy_PyLong_AsStrictLong, 'l')

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

    if (Py_TYPE(pyobject) == GetCTypesType(ct_c_long)) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }

    if (CArraySetArg(pyobject, para, 'l', sizeof(long))) {
        para.fTypeCode = 'V';
        return true;
    }

    PyErr_SetString(PyExc_TypeError, "use ctypes.c_long for pass-by-ref of longs");
    return false;
}

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONST_CHAR_REFCONVERTER(Char,  char,          c_char,  CHAR_MIN,  CHAR_MAX)
CPPYY_IMPL_BASIC_CONST_CHAR_REFCONVERTER(UChar, unsigned char, c_uchar,        0, UCHAR_MAX)

CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Bool,   bool,           c_bool,      CPyCppyy_PyLong_AsBool)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Int8,   int8_t,         c_int8,      CPyCppyy_PyLong_AsInt8)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(UInt8,  uint8_t,        c_uint8,     CPyCppyy_PyLong_AsUInt8)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Int16,  int16_t,        c_int16,     CPyCppyy_PyLong_AsInt16)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(UInt16, uint16_t,       c_uint16,    CPyCppyy_PyLong_AsUInt16)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Int32,  int32_t,        c_int32,     CPyCppyy_PyLong_AsInt32)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(UInt32, uint32_t,       c_uint32,    CPyCppyy_PyLong_AsUInt32)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Short,  short,          c_short,     CPyCppyy_PyLong_AsShort)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(UShort, unsigned short, c_ushort,    CPyCppyy_PyLong_AsUShort)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Int,    int,            c_int,       CPyCppyy_PyLong_AsStrictInt)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(UInt,   unsigned int,   c_uint,      PyLongOrInt_AsULong)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Long,   long,           c_long,      CPyCppyy_PyLong_AsStrictLong)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(ULong,  unsigned long,  c_ulong,     PyLongOrInt_AsULong)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(LLong,  PY_LONG_LONG,   c_longlong,  CPyCppyy_PyLong_AsStrictLongLong)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(ULLong, PY_ULONG_LONG,  c_ulonglong, PyLongOrInt_AsULong64)

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
    if (Py_TYPE(pyobject) == GetCTypesType(ct_c_int)) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }
#endif

// alternate, pass pointer from buffer
    Py_ssize_t buflen = Utility::GetBuffer(pyobject, 'i', sizeof(int), para.fValue.fVoidp);
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
#define CPPYY_IMPL_REFCONVERTER(name, ctype, type, code)                     \
bool CPyCppyy::name##RefConverter::SetArg(                                   \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
/* convert a reference to int to Python through ctypes pointer object */     \
    if (Py_TYPE(pyobject) == GetCTypesType(ct_##ctype)) {                    \
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;\
        para.fTypeCode = 'V';                                                \
        return true;                                                         \
    }                                                                        \
    bool res = CArraySetArg(pyobject, para, code, sizeof(type));             \
    if (!res) {                                                              \
        PyErr_SetString(PyExc_TypeError, "use ctypes."#ctype" for pass-by-ref of "#type);\
        return false;                                                        \
    }                                                                        \
    para.fTypeCode = 'V';                                                    \
    return res;                                                              \
}                                                                            \
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(name, ctype)

CPPYY_IMPL_REFCONVERTER(Bool,    c_bool,       bool,               '?');
CPPYY_IMPL_REFCONVERTER(Char,    c_char,       char,               'b');
CPPYY_IMPL_REFCONVERTER(WChar,   c_wchar,      wchar_t,            'u');
CPPYY_IMPL_REFCONVERTER(Char16,  c_uint16,     char16_t,           'H');
CPPYY_IMPL_REFCONVERTER(Char32,  c_uint32,     char32_t,           'I');
CPPYY_IMPL_REFCONVERTER(SChar,   c_byte,       signed char,        'b');
CPPYY_IMPL_REFCONVERTER(UChar,   c_ubyte,      unsigned char,      'B');
CPPYY_IMPL_REFCONVERTER(Int8,    c_int8,       int8_t,             'b');
CPPYY_IMPL_REFCONVERTER(UInt8,   c_uint8,      uint8_t,            'B');
CPPYY_IMPL_REFCONVERTER(Int16,   c_int16,      int16_t,            'h');
CPPYY_IMPL_REFCONVERTER(UInt16,  c_uint16,     uint16_t,           'H');
CPPYY_IMPL_REFCONVERTER(Int32,   c_int32,      int32_t,            'i');
CPPYY_IMPL_REFCONVERTER(UInt32,  c_uint32,     uint32_t,           'I');
CPPYY_IMPL_REFCONVERTER(Short,   c_short,      short,              'h');
CPPYY_IMPL_REFCONVERTER(UShort,  c_ushort,     unsigned short,     'H');
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(Int, c_int);
CPPYY_IMPL_REFCONVERTER(UInt,    c_uint,       unsigned int,       'I');
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(Long, c_long);
CPPYY_IMPL_REFCONVERTER(ULong,   c_ulong,      unsigned long,      'L');
CPPYY_IMPL_REFCONVERTER(LLong,   c_longlong,   long long,          'q');
CPPYY_IMPL_REFCONVERTER(ULLong,  c_ulonglong,  unsigned long long, 'Q');
CPPYY_IMPL_REFCONVERTER(Float,   c_float,      float,              'f');
CPPYY_IMPL_REFCONVERTER_FROM_MEMORY(Double, c_double);
CPPYY_IMPL_REFCONVERTER(LDouble, c_longdouble, PY_LONG_DOUBLE,     'g');


//----------------------------------------------------------------------------
// convert <pyobject> to C++ bool, allow int/long -> bool, set arg for call
CPPYY_IMPL_BASIC_CONVERTER_NI(
    Bool, bool, long, c_bool, PyBool_FromLong, CPyCppyy_PyLong_AsBool, 'l')

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CHAR_CONVERTER(Char,  char,          CHAR_MIN,  CHAR_MAX)
CPPYY_IMPL_BASIC_CHAR_CONVERTER(UChar, unsigned char,        0, UCHAR_MAX)

PyObject* CPyCppyy::SCharAsIntConverter::FromMemory(void* address)
{
// special case to be used with arrays: return a Python int instead of str
// (following the same convention as module array.array)
    return PyInt_FromLong((long)*((signed char*)address));
}

PyObject* CPyCppyy::UCharAsIntConverter::FromMemory(void* address)
{
// special case to be used with arrays: return a Python int instead of str
// (following the same convention as module array.array)
    return PyInt_FromLong((long)*((unsigned char*)address));
}

//----------------------------------------------------------------------------
bool CPyCppyy::WCharConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ <wchar_t>, set arg for call
    if (!PyUnicode_Check(pyobject) || CPyCppyy_PyUnicode_GET_SIZE(pyobject) != 1) {
        PyErr_SetString(PyExc_ValueError, "single wchar_t character expected");
        return false;
    }
    wchar_t val;
    Py_ssize_t res = CPyCppyy_PyUnicode_AsWideChar(pyobject, &val, 1);
    if (res == -1)
        return false;
    para.fValue.fLong = (long)val;
    para.fTypeCode = 'U';
    return true;
}

PyObject* CPyCppyy::WCharConverter::FromMemory(void* address)
{
    return PyUnicode_FromWideChar((const wchar_t*)address, 1);
}

bool CPyCppyy::WCharConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
    if (!PyUnicode_Check(value) || CPyCppyy_PyUnicode_GET_SIZE(value) != 1) {
        PyErr_SetString(PyExc_ValueError, "single wchar_t character expected");
        return false;
    }
    wchar_t val;
    Py_ssize_t res = CPyCppyy_PyUnicode_AsWideChar(value, &val, 1);
    if (res == -1)
        return false;
    *((wchar_t*)address) = val;
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Char16Converter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ <char16_t>, set arg for call
    if (!PyUnicode_Check(pyobject) || CPyCppyy_PyUnicode_GET_SIZE(pyobject) != 1) {
        PyErr_SetString(PyExc_ValueError, "single char16_t character expected");
        return false;
    }

    PyObject* bstr = PyUnicode_AsUTF16String(pyobject);
    if (!bstr) return false;

    char16_t val = *(char16_t*)(PyBytes_AS_STRING(bstr) + sizeof(char16_t) /*BOM*/);
    Py_DECREF(bstr);
    para.fValue.fLong = (long)val;
    para.fTypeCode = 'U';
    return true;
}

PyObject* CPyCppyy::Char16Converter::FromMemory(void* address)
{
    return PyUnicode_DecodeUTF16((const char*)address, sizeof(char16_t), nullptr, nullptr);
}

bool CPyCppyy::Char16Converter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
    if (!PyUnicode_Check(value) || CPyCppyy_PyUnicode_GET_SIZE(value) != 1) {
        PyErr_SetString(PyExc_ValueError, "single char16_t character expected");
        return false;
    }

    PyObject* bstr = PyUnicode_AsUTF16String(value);
    if (!bstr) return false;

    *((char16_t*)address) = *(char16_t*)(PyBytes_AS_STRING(bstr) + sizeof(char16_t) /*BOM*/);
    Py_DECREF(bstr);
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::Char32Converter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ <char32_t>, set arg for call
    if (!PyUnicode_Check(pyobject) || 2 < CPyCppyy_PyUnicode_GET_SIZE(pyobject)) {
        PyErr_SetString(PyExc_ValueError, "single char32_t character expected");
        return false;
    }

    PyObject* bstr = PyUnicode_AsUTF32String(pyobject);
    if (!bstr) return false;

    char32_t val = *(char32_t*)(PyBytes_AS_STRING(bstr) + sizeof(char32_t) /*BOM*/);
    Py_DECREF(bstr);
    para.fValue.fLong = (long)val;
    para.fTypeCode = 'U';
    return true;
}

PyObject* CPyCppyy::Char32Converter::FromMemory(void* address)
{
    return PyUnicode_DecodeUTF32((const char*)address, sizeof(char32_t), nullptr, nullptr);
}

bool CPyCppyy::Char32Converter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
    if (!PyUnicode_Check(value) || 2 < CPyCppyy_PyUnicode_GET_SIZE(value)) {
        PyErr_SetString(PyExc_ValueError, "single char32_t character expected");
        return false;
    }

    PyObject* bstr = PyUnicode_AsUTF32String(value);
    if (!bstr) return false;

    *((char32_t*)address) = *(char32_t*)(PyBytes_AS_STRING(bstr) + sizeof(char32_t) /*BOM*/);
    Py_DECREF(bstr);
    return true;
}

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONVERTER_IB(
    Int8,  int8_t,  long, c_int8, PyInt_FromLong, CPyCppyy_PyLong_AsInt8,  'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    UInt8, uint8_t, long, c_uint8, PyInt_FromLong, CPyCppyy_PyLong_AsUInt8, 'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    Int16,  int16_t,  long, c_int16, PyInt_FromLong, CPyCppyy_PyLong_AsInt16,  'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    UInt16, uint16_t, long, c_uint16, PyInt_FromLong, CPyCppyy_PyLong_AsUInt16, 'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    Int32,  int32_t,  long, c_int32, PyInt_FromLong, CPyCppyy_PyLong_AsInt32,  'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    UInt32, uint32_t, long, c_uint32, PyInt_FromLong, CPyCppyy_PyLong_AsUInt32, 'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    Short, short, long, c_short, PyInt_FromLong, CPyCppyy_PyLong_AsShort, 'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    UShort, unsigned short, long, c_ushort, PyInt_FromLong, CPyCppyy_PyLong_AsUShort, 'l')
CPPYY_IMPL_BASIC_CONVERTER_IB(
    Int, int, long, c_uint, PyInt_FromLong, CPyCppyy_PyLong_AsStrictInt, 'l')

//----------------------------------------------------------------------------
bool CPyCppyy::ULongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ unsigned long, set arg for call
    if (!ImplicitBool(pyobject, ctxt))
        return false;

    para.fValue.fULong = PyLongOrInt_AsULong(pyobject);
    if (para.fValue.fULong == (unsigned long)-1 && PyErr_Occurred())
        return false;
    para.fTypeCode = 'L';
    return true;
}

PyObject* CPyCppyy::ULongConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned long read at <address>
    return PyLong_FromUnsignedLong(*((unsigned long*)address));
}

bool CPyCppyy::ULongConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ unsigned long, write it at <address>
    unsigned long u = PyLongOrInt_AsULong(value);
    if (u == (unsigned long)-1 && PyErr_Occurred()) {
        if (value == CPyCppyy::gDefaultObject) {
            PyErr_Clear();
            u = (unsigned long)0;
        } else
            return false;
    }
    *((unsigned long*)address) = u;
    return true;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::UIntConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned int read at <address>
    return PyLong_FromUnsignedLong(*((unsigned int*)address));
}

bool CPyCppyy::UIntConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ unsigned int, write it at <address>
    unsigned long u = PyLongOrInt_AsULong(value);
    if (u == (unsigned long)-1 && PyErr_Occurred())
        return false;

    if (u > (unsigned long)UINT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "value too large for unsigned int");
        return false;
    }

    *((unsigned int*)address) = (unsigned int)u;
    return true;
}

//- floating point converters ------------------------------------------------
CPPYY_IMPL_BASIC_CONVERTER_NB(
    Float,  float,  double, c_float,  PyFloat_FromDouble, PyFloat_AsDouble, 'f')
CPPYY_IMPL_BASIC_CONVERTER_NB(
    Double, double, double, c_double, PyFloat_FromDouble, PyFloat_AsDouble, 'd')

CPPYY_IMPL_BASIC_CONVERTER_NB(
    LDouble, PY_LONG_DOUBLE, PY_LONG_DOUBLE, c_longdouble, PyFloat_FromDouble, PyFloat_AsDouble, 'g')

CPyCppyy::ComplexDConverter::ComplexDConverter(bool keepControl) :
    InstanceConverter(Cppyy::GetScope("std::complex<double>"), keepControl) {}

// special case for std::complex<double>, maps it to/from Python's complex
bool CPyCppyy::ComplexDConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
    const Py_complex& pc = PyComplex_AsCComplex(pyobject);
    if (pc.real != -1.0 || !PyErr_Occurred()) {
        fBuffer.real(pc.real);
        fBuffer.imag(pc.imag);
        para.fValue.fVoidp = &fBuffer;
        para.fTypeCode = 'V';
        return true;
    }

    return this->InstanceConverter::SetArg(pyobject, para, ctxt);
}

PyObject* CPyCppyy::ComplexDConverter::FromMemory(void* address)
{
    std::complex<double>* dc = (std::complex<double>*)address;
    return PyComplex_FromDoubles(dc->real(), dc->imag());
}

bool CPyCppyy::ComplexDConverter::ToMemory(PyObject* value, void* address, PyObject* ctxt)
{
    const Py_complex& pc = PyComplex_AsCComplex(value);
    if (pc.real != -1.0 || !PyErr_Occurred()) {
         std::complex<double>* dc = (std::complex<double>*)address;
         dc->real(pc.real);
         dc->imag(pc.imag);
         return true;
    }
    return this->InstanceConverter::ToMemory(value, address, ctxt);
}

//----------------------------------------------------------------------------
bool CPyCppyy::DoubleRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ double&, set arg for call
#if PY_VERSION_HEX < 0x03000000
    if (RefFloat_CheckExact(pyobject)) {
        para.fValue.fVoidp = (void*)&((PyFloatObject*)pyobject)->ob_fval;
        para.fTypeCode = 'V';
        return true;
    }
#endif

#if PY_VERSION_HEX >= 0x02050000
    if (Py_TYPE(pyobject) == GetCTypesType(ct_c_double)) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }
#endif

// alternate, pass pointer from buffer
    Py_ssize_t buflen = Utility::GetBuffer(pyobject, 'd', sizeof(double), para.fValue.fVoidp);
    if (para.fValue.fVoidp && buflen) {
        para.fTypeCode = 'V';
        return true;
    }

#if PY_VERSION_HEX < 0x02050000
    PyErr_SetString(PyExc_TypeError, "use cppyy.Double for pass-by-ref of doubles");
#else
    PyErr_SetString(PyExc_TypeError, "use ctypes.c_double for pass-by-ref of doubles");
#endif
    return false;
}

//----------------------------------------------------------------------------
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Float,   float,          c_float,      PyFloat_AsDouble)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(Double,  double,         c_double,     PyFloat_AsDouble)
CPPYY_IMPL_BASIC_CONST_REFCONVERTER(LDouble, PY_LONG_DOUBLE, c_longdouble, PyFloat_AsDouble)

//----------------------------------------------------------------------------
bool CPyCppyy::VoidConverter::SetArg(PyObject*, Parameter&, CallContext*)
{
// can't happen (unless a type is mapped wrongly), but implemented for completeness
    PyErr_SetString(PyExc_SystemError, "void/unknown arguments can\'t be set");
    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::LLongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ long long, set arg for call
    if (!ImplicitBool(pyobject, ctxt))
        return false;

    para.fValue.fLLong = CPyCppyy_PyLong_AsStrictLongLong(pyobject);
    if (PyErr_Occurred())
        return false;
    para.fTypeCode = 'q';
    return true;
}

PyObject* CPyCppyy::LLongConverter::FromMemory(void* address)
{
// construct python object from C++ long long read at <address>
    return PyLong_FromLongLong(*(PY_LONG_LONG*)address);
}

bool CPyCppyy::LLongConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ long long, write it at <address>
    PY_LONG_LONG ll = PyLong_AsLongLong(value);
    if (ll == -1 && PyErr_Occurred()) {
        if (value == CPyCppyy::gDefaultObject) {
            PyErr_Clear();
            ll = (PY_LONG_LONG)0;
        } else
            return false;
    }
    *((PY_LONG_LONG*)address) = ll;
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::ULLongConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ unsigned long long, set arg for call
    if (!ImplicitBool(pyobject, ctxt))
        return false;

    para.fValue.fULLong = PyLongOrInt_AsULong64(pyobject);
    if (PyErr_Occurred())
        return false;
    para.fTypeCode = 'Q';
    return true;
}

PyObject* CPyCppyy::ULLongConverter::FromMemory(void* address)
{
// construct python object from C++ unsigned long long read at <address>
    return PyLong_FromUnsignedLongLong(*(PY_ULONG_LONG*)address);
}

bool CPyCppyy::ULLongConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ unsigned long long, write it at <address>
    PY_ULONG_LONG ull = PyLongOrInt_AsULong64(value);
    if (PyErr_Occurred()) {
        if (value == CPyCppyy::gDefaultObject) {
            PyErr_Clear();
            ull = (PY_ULONG_LONG)0;
        } else
            return false;
    }
    *((PY_ULONG_LONG*)address) = ull;
    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CStringConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// construct a new string and copy it in new memory
    Py_ssize_t len;
    const char* cstr = CPyCppyy_PyText_AsStringAndSize(pyobject, &len);
    if (!cstr) {
    // special case: allow ctypes c_char_p
        auto error = CPyCppyy::Utility::FetchPyError();
        if (Py_TYPE(pyobject) == GetCTypesType(ct_c_char_p)) {
            SetLifeLine(ctxt->fPyContext, pyobject, (intptr_t)this);
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
            para.fTypeCode = 'V';
            return true;
        }
        CPyCppyy::Utility::RestorePyError(error);
        return false;
    }

// verify (too long string will cause truncation, no crash)
    if (fMaxSize != std::string::npos && fMaxSize < fBuffer.size())
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for char array (truncated)");

    if (!ctxt->fPyContext) {
    // use internal buffer as workaround
        fBuffer = std::string(cstr, len);
        if (fMaxSize != std::string::npos)
            fBuffer.resize(fMaxSize, '\0');      // pad remainder of buffer as needed
        cstr = fBuffer.c_str();
    } else
        SetLifeLine(ctxt->fPyContext, pyobject, (intptr_t)this);

// set the value and declare success
    para.fValue.fVoidp = (void*)cstr;
    para.fTypeCode = 'p';
    return true;
}

PyObject* CPyCppyy::CStringConverter::FromMemory(void* address)
{
// construct python object from C++ const char* read at <address>
    if (address && *(void**)address) {
        if (fMaxSize != std::string::npos)       // need to prevent reading beyond boundary
            return CPyCppyy_PyText_FromStringAndSize(*(char**)address, (Py_ssize_t)fMaxSize);

        if (*(void**)address == (void*)fBuffer.data())     // if we're buffering, we know the size
            return CPyCppyy_PyText_FromStringAndSize((char*)fBuffer.data(), fBuffer.size());

    // no idea about lentgth: cut on \0
        return CPyCppyy_PyText_FromString(*(char**)address);
    }

// empty string in case there's no address
    Py_INCREF(PyStrings::gEmptyString);
    return PyStrings::gEmptyString;
}

bool CPyCppyy::CStringConverter::ToMemory(PyObject* value, void* address, PyObject* ctxt)
{
// convert <value> to C++ const char*, write it at <address>
    Py_ssize_t len;
    const char* cstr = CPyCppyy_PyText_AsStringAndSize(value, &len);
    if (!cstr) return false;

// verify (too long string will cause truncation, no crash)
    if (fMaxSize != std::string::npos && fMaxSize < (std::string::size_type)len)
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for char array (truncated)");

// if address is available, and it wasn't set by this converter, assume a byte-wise copy;
// otherwise assume a pointer copy (this relies on the converter to be used for properties,
// or for argument passing, but not both at the same time; this is currently the case)
    void* ptrval = *(void**)address;
    if (ptrval == (void*)fBuffer.data()) {
        fBuffer = std::string(cstr, len);
        *(void**)address = (void*)fBuffer.data();
        return true;
    } else if (ptrval && HasLifeLine(ctxt, (intptr_t)ptrval)) {
        ptrval = nullptr;
    // fall through; ptrval is nullptr means we're managing it
    }

// the string is (going to be) managed by us: assume pointer copy
    if (!ptrval) {
        SetLifeLine(ctxt, value, (intptr_t)address);
        *(void**)address = (void*)cstr;
        return true;
    }

// the pointer value is non-zero and not ours: assume byte copy
    if (fMaxSize != std::string::npos)
        strncpy(*(char**)address, cstr, fMaxSize);    // pads remainder
    else
    // coverity[secure_coding] - can't help it, it's intentional.
        strcpy(*(char**)address, cstr);

   return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::WCStringConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// construct a new string and copy it in new memory
    Py_ssize_t len = PyUnicode_GetSize(pyobject);
    if (len == (Py_ssize_t)-1 && PyErr_Occurred())
        return false;

    fBuffer = (wchar_t*)realloc(fBuffer, sizeof(wchar_t)*(len+1));
    Py_ssize_t res = CPyCppyy_PyUnicode_AsWideChar(pyobject, fBuffer, len);
    if (res == -1)
        return false;   // could free the buffer here

// set the value and declare success
    fBuffer[len] = L'\0';
    para.fValue.fVoidp = (void*)fBuffer;
    para.fTypeCode = 'p';
    return true;
}

PyObject* CPyCppyy::WCStringConverter::FromMemory(void* address)
{
// construct python object from C++ wchar_t* read at <address>
    if (address && *(wchar_t**)address) {
        if (fMaxSize != std::wstring::npos)      // need to prevent reading beyond boundary
            return PyUnicode_FromWideChar(*(wchar_t**)address, (Py_ssize_t)fMaxSize);
    // with unknown size
        return PyUnicode_FromWideChar(*(wchar_t**)address, wcslen(*(wchar_t**)address));
    }

// empty string in case there's no valid address
    wchar_t w = L'\0';
    return PyUnicode_FromWideChar(&w, 0);
}

bool CPyCppyy::WCStringConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ wchar_t*, write it at <address>
    Py_ssize_t len = PyUnicode_GetSize(value);
    if (len == (Py_ssize_t)-1 && PyErr_Occurred())
        return false;

// verify (too long string will cause truncation, no crash)
    if (fMaxSize != std::wstring::npos && fMaxSize < (std::wstring::size_type)len)
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for wchar_t array (truncated)");

    Py_ssize_t res = -1;
    if (fMaxSize != std::wstring::npos)
        res = CPyCppyy_PyUnicode_AsWideChar(value, *(wchar_t**)address, (Py_ssize_t)fMaxSize);
    else
    // coverity[secure_coding] - can't help it, it's intentional.
        res = CPyCppyy_PyUnicode_AsWideChar(value, *(wchar_t**)address, len);

    if (res == -1) return false;
    return true;
}

//----------------------------------------------------------------------------
#define CPYCPPYY_WIDESTRING_CONVERTER(name, type, encode, decode, snull)     \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)            \
{                                                                            \
/* change string encoding and copy into local buffer */                      \
    PyObject* bstr = encode(pyobject);                                       \
    if (!bstr) return false;                                                 \
                                                                             \
    Py_ssize_t len = PyBytes_GET_SIZE(bstr) - sizeof(type) /*BOM*/;          \
    fBuffer = (type*)realloc(fBuffer, len + sizeof(type));                   \
    memcpy(fBuffer, PyBytes_AS_STRING(bstr) + sizeof(type) /*BOM*/, len);    \
    Py_DECREF(bstr);                                                         \
                                                                             \
    fBuffer[len/sizeof(type)] = snull;                                       \
    para.fValue.fVoidp = (void*)fBuffer;                                     \
    para.fTypeCode = 'p';                                                    \
    return true;                                                             \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
/* construct python object from C++ <type>* read at <address> */             \
    if (address && *(type**)address) {                                       \
        if (fMaxSize != std::wstring::npos)                                  \
            return decode(*(const char**)address, (Py_ssize_t)fMaxSize*sizeof(type), nullptr, nullptr);\
        return decode(*(const char**)address,                                \
            std::char_traits<type>::length(*(type**)address)*sizeof(type), nullptr, nullptr);\
    }                                                                        \
                                                                             \
/* empty string in case there's no valid address */                          \
    type w = snull;                                                          \
    return decode((const char*)&w, 0, nullptr, nullptr);                     \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)\
{                                                                            \
/* convert <value> to C++ <type>*, write it at <address> */                  \
    PyObject* bstr = encode(value);                                          \
    if (!bstr) return false;                                                 \
                                                                             \
    Py_ssize_t len = PyBytes_GET_SIZE(bstr) - sizeof(type) /*BOM*/;          \
    Py_ssize_t maxbytes = (Py_ssize_t)fMaxSize*sizeof(type);                 \
                                                                             \
/* verify (too long string will cause truncation, no crash) */               \
    if (fMaxSize != std::wstring::npos && maxbytes < len) {                  \
        PyErr_Warn(PyExc_RuntimeWarning, (char*)"string too long for "#type" array (truncated)");\
        len = maxbytes;                                                      \
    }                                                                        \
                                                                             \
    memcpy(*((void**)address), PyBytes_AS_STRING(bstr) + sizeof(type) /*BOM*/, len);\
    Py_DECREF(bstr);                                                         \
/* debatable, but probably more convenient in most cases to null-terminate if enough space */\
    if (len/sizeof(type) < fMaxSize) (*(type**)address)[len/sizeof(type)] = snull;\
    return true;                                                             \
}

CPYCPPYY_WIDESTRING_CONVERTER(CString16, char16_t, PyUnicode_AsUTF16String, PyUnicode_DecodeUTF16, u'\0')
CPYCPPYY_WIDESTRING_CONVERTER(CString32, char32_t, PyUnicode_AsUTF32String, PyUnicode_DecodeUTF32, U'\0')

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
    if (fMaxSize != std::string::npos)
        return CPyCppyy_PyText_FromStringAndSize(*(char**)address, (Py_ssize_t)fMaxSize);
    return this->CStringConverter::FromMemory(address);
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidArrayConverter::GetAddressSpecialCase(PyObject* pyobject, void*& address)
{
// (1): C++11 style "null pointer"
    if (pyobject == gNullPtrObject || pyobject == gDefaultObject) {
        address = nullptr;
        return true;
    }

// (2): allow integer zero to act as a null pointer (C NULL), no deriveds
    if (PyInt_CheckExact(pyobject) || PyLong_CheckExact(pyobject)) {
        intptr_t val = (intptr_t)PyLong_AsLongLong(pyobject);
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
    CPPInstance* pyobj = GetCppInstance(pyobject);
    if (pyobj) {
    // depending on memory policy, some objects are no longer owned when passed to C++
        if (!fKeepControl && !UseStrictOwnership(ctxt))
            pyobj->CppOwns();

   // set pointer (may be null) and declare success
        para.fValue.fVoidp = pyobj->GetObject();
        para.fTypeCode = 'p';
        return true;
    }

// handle special cases
    if (GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
        para.fTypeCode = 'p';
        return true;
    }

// allow ctypes voidp (which if got as a buffer will return void**, not void*); use
// isintance instead of an exact check, b/c c_void_p is the type mapper for typedefs
// of void* (typically opaque handles)
    if (PyObject_IsInstance(pyobject, (PyObject*)GetCTypesType(ct_c_void_p))) {
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;
    }

// allow any other ctypes pointer type
    if (IsCTypesArrayOrPointer(pyobject)) {
        void** payload = (void**)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        if (payload) {
            para.fValue.fVoidp = *payload;
            para.fTypeCode = 'p';
            return true;
        }
    }

// final try: attempt to get buffer
    Py_ssize_t buflen = Utility::GetBuffer(pyobject, '*', 1, para.fValue.fVoidp, false);

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
    if (!address || *(uintptr_t*)address == 0) {
        Py_INCREF(gNullPtrObject);
        return gNullPtrObject;
    }
    return CreatePointerView(*(uintptr_t**)address);
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidArrayConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// just convert pointer if it is a C++ object
    CPPInstance* pyobj = GetCppInstance(value);
    if (pyobj) {
    // depending on memory policy, some objects are no longer owned when passed to C++
        if (!fKeepControl && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            pyobj->CppOwns();

    // set pointer (may be null) and declare success
        *(void**)address = pyobj->GetObject();
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
    Py_ssize_t buflen = Utility::GetBuffer(value, '*', 1, buf, false);
    if (!buf || buflen == 0)
        return false;

    *(void**)address = buf;
    return true;
}

namespace {

// Copy a buffer to memory address with an array converter.
template<class type>
bool ToArrayFromBuffer(PyObject* owner, void* address, PyObject* ctxt,
                       const void * buf, Py_ssize_t buflen,
                       CPyCppyy::dims_t& shape, bool isFixed)
{
    if (buflen == 0)
        return false;

    Py_ssize_t oldsz = 1;
    for (Py_ssize_t idim = 0; idim < shape.ndim(); ++idim) {
        if (shape[idim] == CPyCppyy::UNKNOWN_SIZE) {
            oldsz = -1;
            break;
        }
        oldsz *= shape[idim];
    }
    if (shape.ndim() != CPyCppyy::UNKNOWN_SIZE && 0 < oldsz && oldsz < buflen) {
        PyErr_SetString(PyExc_ValueError, "buffer too large for value");
        return false;
    }

    if (isFixed)
        memcpy(*(type**)address, buf, (0 < buflen ? buflen : 1)*sizeof(type));
    else {
        *(type**)address = (type*)buf;
        shape.ndim(1);
        shape[0] = buflen;
        SetLifeLine(ctxt, owner, (intptr_t)address);
    }
    return true;
}

}

//----------------------------------------------------------------------------
#define CPPYY_IMPL_ARRAY_CONVERTER(name, ctype, type, code, suffix)          \
CPyCppyy::name##ArrayConverter::name##ArrayConverter(cdims_t dims) :         \
        fShape(dims) {                                                       \
    fIsFixed = dims ? fShape[0] != UNKNOWN_SIZE : false;                     \
}                                                                            \
                                                                             \
bool CPyCppyy::name##ArrayConverter::SetArg(                                 \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    /* filter ctypes first b/c their buffer conversion will be wrong */      \
    bool convOk = false;                                                     \
                                                                             \
    /* 2-dim case: ptr-ptr types */                                          \
    if (fShape.ndim() == 2) {                                                \
        if (Py_TYPE(pyobject) == GetCTypesPtrType(ct_##ctype)) {             \
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;\
            para.fTypeCode = 'p';                                            \
            convOk = true;                                                   \
        } else if (Py_TYPE(pyobject) == GetCTypesType(ct_c_void_p)) {        \
        /* special case: pass address of c_void_p buffer to return the address */\
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;\
            para.fTypeCode = 'p';                                            \
            convOk = true;                                                   \
        } else if (LowLevelView_Check(pyobject) &&                           \
                ((LowLevelView*)pyobject)->fBufInfo.ndim == 2 &&             \
                strchr(((LowLevelView*)pyobject)->fBufInfo.format, code)) {  \
            para.fValue.fVoidp = ((LowLevelView*)pyobject)->get_buf();       \
            para.fTypeCode = 'p';                                            \
            convOk = true;                                                   \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* 1-dim (accept pointer), or unknown (accept pointer as cast) */        \
    if (!convOk) {                                                           \
        PyTypeObject* ctypes_type = GetCTypesType(ct_##ctype);               \
        if (Py_TYPE(pyobject) == ctypes_type) {                              \
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;\
            para.fTypeCode = 'p';                                            \
            convOk = true;                                                   \
        } else if (Py_TYPE(pyobject) == GetCTypesPtrType(ct_##ctype)) {      \
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;\
            para.fTypeCode = 'V';                                            \
            convOk = true;                                                   \
        } else if (IsPyCArgObject(pyobject)) {                               \
            CPyCppyy_tagPyCArgObject* carg = (CPyCppyy_tagPyCArgObject*)pyobject;\
            if (carg->obj && Py_TYPE(carg->obj) == ctypes_type) {            \
                para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)carg->obj)->b_ptr;\
                para.fTypeCode = 'p';                                        \
                convOk = true;                                               \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* cast pointer type */                                                  \
    if (!convOk) {                                                           \
        bool ismulti = fShape.ndim() > 1;                                   \
        convOk = CArraySetArg(pyobject, para, code, ismulti ? sizeof(void*) : sizeof(type), true);\
    }                                                                        \
                                                                             \
    /* memory management and offsetting */                                   \
    if (convOk) SetLifeLine(ctxt->fPyContext, pyobject, (intptr_t)this);     \
                                                                             \
    return convOk;                                                           \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##ArrayConverter::FromMemory(void* address)          \
{                                                                            \
    if (!fIsFixed)                                                           \
        return CreateLowLevelView##suffix((type**)address, fShape);          \
    return CreateLowLevelView##suffix(*(type**)address, fShape);             \
}                                                                            \
                                                                             \
bool CPyCppyy::name##ArrayConverter::ToMemory(                               \
    PyObject* value, void* address, PyObject* ctxt)                          \
{                                                                            \
    if (fShape.ndim() <= 1 || fIsFixed) {                                    \
        void* buf = nullptr;                                                 \
        Py_ssize_t buflen = Utility::GetBuffer(value, code, sizeof(type), buf);\
        return ToArrayFromBuffer<type>(value, address, ctxt, buf, buflen, fShape, fIsFixed);\
    } else { /* multi-dim, non-flat array; assume structure matches */       \
        void* buf = nullptr; /* TODO: GetBuffer() assumes flat? */           \
        Py_ssize_t buflen = Utility::GetBuffer(value, code, sizeof(void*), buf);\
        if (buflen == 0) return false;                                       \
        *(type**)address = (type*)buf;                                       \
        SetLifeLine(ctxt, value, (intptr_t)address);                         \
    }                                                                        \
    return true;                                                             \
}


//----------------------------------------------------------------------------
CPPYY_IMPL_ARRAY_CONVERTER(Bool,     c_bool,       bool,                 '?', )
CPPYY_IMPL_ARRAY_CONVERTER(SChar,    c_char,       signed char,          'b', )
CPPYY_IMPL_ARRAY_CONVERTER(UChar,    c_ubyte,      unsigned char,        'B', )
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
CPPYY_IMPL_ARRAY_CONVERTER(Byte,     c_ubyte,      std::byte,            'B', )
#endif
CPPYY_IMPL_ARRAY_CONVERTER(Int8,     c_byte,       int8_t,               'b', _i8)
CPPYY_IMPL_ARRAY_CONVERTER(Int16,    c_int16,      int16_t,              'h', _i16)
CPPYY_IMPL_ARRAY_CONVERTER(Int32,    c_int32,      int32_t,              'i', _i32)
CPPYY_IMPL_ARRAY_CONVERTER(UInt8,    c_ubyte,      uint8_t,              'B', _i8)
CPPYY_IMPL_ARRAY_CONVERTER(UInt16,   c_uint16,     uint16_t,             'H', _i16)
CPPYY_IMPL_ARRAY_CONVERTER(UInt32,   c_uint32,     uint32_t,             'I', _i32)
CPPYY_IMPL_ARRAY_CONVERTER(Short,    c_short,      short,                'h', )
CPPYY_IMPL_ARRAY_CONVERTER(UShort,   c_ushort,     unsigned short,       'H', )
CPPYY_IMPL_ARRAY_CONVERTER(Int,      c_int,        int,                  'i', )
CPPYY_IMPL_ARRAY_CONVERTER(UInt,     c_uint,       unsigned int,         'I', )
CPPYY_IMPL_ARRAY_CONVERTER(Long,     c_long,       long,                 'l', )
CPPYY_IMPL_ARRAY_CONVERTER(ULong,    c_ulong,      unsigned long,        'L', )
CPPYY_IMPL_ARRAY_CONVERTER(LLong,    c_longlong,   long long,            'q', )
CPPYY_IMPL_ARRAY_CONVERTER(ULLong,   c_ulonglong,  unsigned long long,   'Q', )
CPPYY_IMPL_ARRAY_CONVERTER(Float,    c_float,      float,                'f', )
CPPYY_IMPL_ARRAY_CONVERTER(Double,   c_double,     double,               'd', )
CPPYY_IMPL_ARRAY_CONVERTER(LDouble,  c_longdouble, long double,          'g', )
CPPYY_IMPL_ARRAY_CONVERTER(ComplexF, c_fcomplex,   std::complex<float>,  'z', )
CPPYY_IMPL_ARRAY_CONVERTER(ComplexD, c_complex,    std::complex<double>, 'Z', )


//----------------------------------------------------------------------------
bool CPyCppyy::CStringArrayConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
    if (Py_TYPE(pyobject) == GetCTypesPtrType(ct_c_char_p) || \
            (1 < fShape.ndim() && PyObject_IsInstance(pyobject, (PyObject*)GetCTypesType(ct_c_pointer)))) {
    // 2nd predicate is ebatable: it's a catch-all for ctypes-styled multi-dimensional objects,
    // which at this point does not check further dimensionality
        para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        para.fTypeCode = 'V';
        return true;

    } else if (PySequence_Check(pyobject) && !CPyCppyy_PyText_Check(pyobject)
#if PY_VERSION_HEX >= 0x03000000
        && !PyBytes_Check(pyobject)
#endif
    ) {
        //for (auto& p : fBuffer) free(p);
        fBuffer.clear();

        size_t len = (size_t)PySequence_Size(pyobject);
        if (len == (size_t)-1) {
            PyErr_SetString(PyExc_ValueError, "can not convert sequence object of unknown length");
            return false;
        }

        fBuffer.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            PyObject* item = PySequence_GetItem(pyobject, i);
            if (item) {
                Py_ssize_t sz;
                const char* p = CPyCppyy_PyText_AsStringAndSize(item, &sz);
                Py_DECREF(item);

                if (p) fBuffer.push_back(p);
                else {
                    PyErr_Format(PyExc_TypeError, "could not convert item %d to string", (int)i);
                    return false;
                }

            } else
                return false;
        }

        para.fValue.fVoidp = (void*)fBuffer.data();
        para.fTypeCode = 'p';
        return true;
    }

    return SCharArrayConverter::SetArg(pyobject, para, ctxt);
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::CStringArrayConverter::FromMemory(void* address)
{
    if (fIsFixed)
        return CreateLowLevelView(*(char**)address, fShape);
    else if (fShape[0] == UNKNOWN_SIZE)
        return CreateLowLevelViewString((const char**)address, fShape);
    return CreateLowLevelViewString(*(const char***)address, fShape);
}

//----------------------------------------------------------------------------
bool CPyCppyy::CStringArrayConverter::ToMemory(PyObject* value, void* address, PyObject* ctxt)
{
// As a special array converter, the CStringArrayConverter one can also copy strings in the array,
// and not only buffers.
    Py_ssize_t len;
    if (const char* cstr = CPyCppyy_PyText_AsStringAndSize(value, &len)) {
        return ToArrayFromBuffer<char>(value, address, ctxt, cstr, len, fShape, fIsFixed);
    }
    return SCharArrayConverter::ToMemory(value, address, ctxt);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::NonConstCStringArrayConverter::FromMemory(void* address)
{
    if (fIsFixed)
        return CreateLowLevelView(*(char**)address, fShape);
    else if (fShape[0] == UNKNOWN_SIZE)
        return CreateLowLevelViewString((char**)address, fShape);
    return CreateLowLevelViewString(*(char***)address, fShape);
}

//- converters for special cases ---------------------------------------------
bool CPyCppyy::NullptrConverter::SetArg(PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// Only allow C++11 style nullptr to pass
    if (pyobject == gNullPtrObject || pyobject == gDefaultObject) {
        para.fValue.fVoidp = nullptr;
        para.fTypeCode = 'p';
        return true;
    }
    return false;
}


//----------------------------------------------------------------------------
template<typename T>
static inline bool CPyCppyy_PyUnicodeAsBytes2Buffer(PyObject* pyobject, T& buffer) {
    PyObject* pybytes = nullptr;
    if (PyBytes_Check(pyobject)) {
        Py_INCREF(pyobject);
        pybytes = pyobject;
    } else if (PyUnicode_Check(pyobject)) {
#if PY_VERSION_HEX < 0x03030000
        pybytes = PyUnicode_EncodeUTF8(
            PyUnicode_AS_UNICODE(pyobject), CPyCppyy_PyUnicode_GET_SIZE(pyobject), nullptr);
#else
        pybytes = PyUnicode_AsUTF8String(pyobject);
#endif
    }

    if (pybytes) {
        Py_ssize_t len;
        const char* cstr = nullptr;
        PyBytes_AsStringAndSize(pybytes, (char**)&cstr, &len);
        if (cstr) buffer = T{cstr, (typename T::size_type)len};
        Py_DECREF(pybytes);
        return (bool)cstr;
    }

    return false;
}

#define CPPYY_IMPL_STRING_AS_PRIMITIVE_CONVERTER(name, type, F1, F2)         \
CPyCppyy::name##Converter::name##Converter(bool keepControl) :               \
    InstanceConverter(Cppyy::GetScope(#type), keepControl) {}                \
                                                                             \
bool CPyCppyy::name##Converter::SetArg(                                      \
    PyObject* pyobject, Parameter& para, CallContext* ctxt)                  \
{                                                                            \
    if (CPyCppyy_PyUnicodeAsBytes2Buffer(pyobject, fBuffer)) {               \
        para.fValue.fVoidp = &fBuffer;                                       \
        para.fTypeCode = 'V';                                                \
        return true;                                                         \
    }                                                                        \
                                                                             \
    PyErr_Clear();                                                           \
    if (!(PyInt_Check(pyobject) || PyLong_Check(pyobject))) {                \
        bool result = InstanceConverter::SetArg(pyobject, para, ctxt);       \
        para.fTypeCode = 'V';                                                \
        return result;                                                       \
    }                                                                        \
                                                                             \
    return false;                                                            \
}                                                                            \
                                                                             \
PyObject* CPyCppyy::name##Converter::FromMemory(void* address)               \
{                                                                            \
    if (address)                                                             \
        return InstanceConverter::FromMemory(address);                       \
    auto* empty = new type();                                                \
    return BindCppObjectNoCast(empty, fClass, CPPInstance::kIsOwner);        \
}                                                                            \
                                                                             \
bool CPyCppyy::name##Converter::ToMemory(                                    \
    PyObject* value, void* address, PyObject* ctxt)                          \
{                                                                            \
    if (CPyCppyy_PyUnicodeAsBytes2Buffer(value, *((type*)address)))          \
        return true;                                                         \
    return InstanceConverter::ToMemory(value, address, ctxt);                \
}

CPPYY_IMPL_STRING_AS_PRIMITIVE_CONVERTER(STLString, std::string, c_str, size)


CPyCppyy::STLWStringConverter::STLWStringConverter(bool keepControl) :
    InstanceConverter(Cppyy::GetScope("std::wstring"), keepControl) {}

bool CPyCppyy::STLWStringConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
    if (PyUnicode_Check(pyobject)) {
        Py_ssize_t len = CPyCppyy_PyUnicode_GET_SIZE(pyobject);
        fBuffer.resize(len);
        CPyCppyy_PyUnicode_AsWideChar(pyobject, &fBuffer[0], len);
        para.fValue.fVoidp = &fBuffer;
        para.fTypeCode = 'V';
        return true;
    }
#if PY_VERSION_HEX < 0x03000000
    else if (PyString_Check(pyobject)) {
#ifdef HAS_CODECVT
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> cnv;
        fBuffer = cnv.from_bytes(PyString_AS_STRING(pyobject));
#else
        PyObject* pyu = PyUnicode_FromString(PyString_AS_STRING(pyobject));
        if (!pyu) return false;
        Py_ssize_t len = CPyCppyy_PyUnicode_GET_SIZE(pyu);
        fBuffer.resize(len);
        CPyCppyy_PyUnicode_AsWideChar(pyu, &fBuffer[0], len);
#endif
        para.fValue.fVoidp = &fBuffer;
        para.fTypeCode = 'V';
        return true;
    }
#endif

    if (!(PyInt_Check(pyobject) || PyLong_Check(pyobject))) {
        bool result = InstancePtrConverter<false>::SetArg(pyobject, para, ctxt);
        para.fTypeCode = 'V';
        return result;
    }

    return false;
}

PyObject* CPyCppyy::STLWStringConverter::FromMemory(void* address)
{
    if (address)
        return PyUnicode_FromWideChar(((std::wstring*)address)->c_str(), ((std::wstring*)address)->size());
    wchar_t w = L'\0';
    return PyUnicode_FromWideChar(&w, 0);
}

bool CPyCppyy::STLWStringConverter::ToMemory(PyObject* value, void* address, PyObject* ctxt)
{
    if (PyUnicode_Check(value)) {
        Py_ssize_t len = CPyCppyy_PyUnicode_GET_SIZE(value);
        wchar_t* buf = new wchar_t[len+1];
        CPyCppyy_PyUnicode_AsWideChar(value, buf, len);
        *((std::wstring*)address) = std::wstring(buf, len);
        delete[] buf;
        return true;
    }
    return InstanceConverter::ToMemory(value, address, ctxt);
}


#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
CPyCppyy::STLStringViewConverter::STLStringViewConverter(bool keepControl) :
    InstanceConverter(Cppyy::GetScope("std::string_view"), keepControl) {}

bool CPyCppyy::STLStringViewConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// normal instance convertion (eg. string_view object passed)
    if (!PyInt_Check(pyobject) && !PyLong_Check(pyobject)) {
        CallContextRAII<CallContext::kNoImplicit> noimp(ctxt);
        if (InstanceConverter::SetArg(pyobject, para, ctxt)) {
            para.fTypeCode = 'V';
            return true;
        } else
            PyErr_Clear();
    }

// passing of a Python string; buffering done Python-side b/c str is immutable
    Py_ssize_t len;
    const char* cstr = CPyCppyy_PyText_AsStringAndSize(pyobject, &len);
    if (cstr) {
        SetLifeLine(ctxt->fPyContext, pyobject, (intptr_t)this);
        fBuffer = std::string_view(cstr, (std::string_view::size_type)len);
        para.fValue.fVoidp = &fBuffer;
        para.fTypeCode = 'V';
        return true;
    }

    if (!CPPInstance_Check(pyobject))
        return false;

// special case of a C++ std::string object; life-time management is left to
// the caller to ensure any external changes propagate correctly
    if (CPPInstance_Check(pyobject)) {
        static Cppyy::TCppScope_t sStringID = Cppyy::GetScope("std::string");
        CPPInstance* pyobj = (CPPInstance*)pyobject;
        if (pyobj->ObjectIsA() == sStringID) {
            void* ptr = pyobj->GetObject();
            if (!ptr)
                return false;     // leaves prior conversion error for report

            PyErr_Clear();

            fBuffer = *((std::string*)ptr);
            para.fValue.fVoidp = &fBuffer;
            para.fTypeCode = 'V';
            return true;
        }
    }

    return false;
}

PyObject* CPyCppyy::STLStringViewConverter::FromMemory(void* address)
{
    if (address)
        return InstanceConverter::FromMemory(address);
    auto* empty = new std::string_view();
    return BindCppObjectNoCast(empty, fClass, CPPInstance::kIsOwner);
}

bool CPyCppyy::STLStringViewConverter::ToMemory(
    PyObject* value, void* address, PyObject* ctxt)
{
// common case of simple object assignment
    if (InstanceConverter::ToMemory(value, address, ctxt))
        return true;

// assignment of a Python string; buffering done Python-side b/c str is immutable
    Py_ssize_t len;
    const char* cstr = CPyCppyy_PyText_AsStringAndSize(value, &len);
    if (cstr) {
        SetLifeLine(ctxt, value, (intptr_t)this);
        *reinterpret_cast<std::string_view*>(address) = \
            std::string_view(cstr, (std::string_view::size_type)len);
        return true;
    }

    return false;
}
#endif


bool CPyCppyy::STLStringMoveConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ std::string&&, set arg for call
    int moveit_reason = 3;   // move on temporary fBuffer
    if (CPPInstance_Check(pyobject)) {
        CPPInstance* pyobj = (CPPInstance*)pyobject;
        if (pyobj->fFlags & CPPInstance::kIsRValue) {
            pyobj->fFlags &= ~CPPInstance::kIsRValue;
            moveit_reason = 2;
        } else if (PyUnstable_Object_IsUniqueReferencedTemporary(pyobject)) {
            moveit_reason = 1;
        } else
            moveit_reason = 0;
    }

    if (moveit_reason) {
        bool result = this->STLStringConverter::SetArg(pyobject, para, ctxt);
        if (!result && moveit_reason == 2)       // restore the movability flag?
            ((CPPInstance*)pyobject)->fFlags |= CPPInstance::kIsRValue;
        return result;
    }

    PyErr_SetString(PyExc_ValueError, "object is not an rvalue");
    return false;      // not a temporary or movable object
}


//----------------------------------------------------------------------------
template <bool ISCONST>
bool CPyCppyy::InstancePtrConverter<ISCONST>::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance*, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject, ISCONST ? fClass : (Cppyy::TCppType_t)0);
    if (!pyobj) {
        if (GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
            para.fTypeCode = 'p';      // allow special cases such as nullptr
            return true;
        }

   // not a cppyy object (TODO: handle SWIG etc.)
        return false;
    }

    // smart pointers should only extract the pointer if this is NOT an implicit
    // conversion to another smart pointer
    if (pyobj->IsSmart() && IsConstructor(ctxt->fFlags) && Cppyy::IsSmartPtr(ctxt->fCurScope))
        return false;

    Cppyy::TCppType_t oisa = pyobj->ObjectIsA();
    if (oisa && (oisa == fClass || Cppyy::IsSubtype(oisa, fClass))) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && !UseStrictOwnership(ctxt))
            pyobj->CppOwns();

    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (oisa != fClass) {
            para.fValue.fIntPtr += Cppyy::GetBaseOffset(
                oisa, fClass, para.fValue.fVoidp, 1 /* up-cast */);
        }

    // set pointer (may be null) and declare success
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
template <bool ISCONST>
PyObject* CPyCppyy::InstancePtrConverter<ISCONST>::FromMemory(void* address)
{
// construct python object from C++ instance read at <address>
    if (ISCONST)
        return BindCppObject(*(void**)address, fClass);                   // by pointer value
    return BindCppObject(address, fClass, CPPInstance::kIsReference);     // modifiable
}

//----------------------------------------------------------------------------
template <bool ISCONST>
bool CPyCppyy::InstancePtrConverter<ISCONST>::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ instance, write it at <address>
    CPPInstance* pyobj = GetCppInstance(value, ISCONST ? fClass : (Cppyy::TCppType_t)0);
    if (!pyobj) {
        void* ptr = nullptr;
        if (GetAddressSpecialCase(value, ptr)) {
            *(void**)address = ptr;          // allow special cases such as nullptr
            return true;
        }

    // not a cppyy object (TODO: handle SWIG etc.)
        return false;
    }

    if (Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            ((CPPInstance*)value)->CppOwns();

        *(void**)address = pyobj->GetObject();
        return true;
    }

    return false;
}

// TODO: CONSOLIDATE Instance, InstanceRef, InstancePtr ...

//----------------------------------------------------------------------------
bool CPyCppyy::InstanceConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject, fClass);
    if (pyobj) {
        auto oisa = pyobj->ObjectIsA();
        if (oisa && (oisa == fClass || Cppyy::IsSubtype(oisa, fClass))) {
        // calculate offset between formal and actual arguments
            para.fValue.fVoidp = pyobj->GetObject();
            if (!para.fValue.fVoidp)
                return false;

            if (oisa != fClass) {
                para.fValue.fIntPtr += Cppyy::GetBaseOffset(
                    pyobj->ObjectIsA(), fClass, para.fValue.fVoidp, 1 /* up-cast */);
            }

            para.fTypeCode = 'V';
            return true;
        }
    }

    return (bool)ConvertImplicit(fClass, pyobject, para, ctxt);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceConverter::FromMemory(void* address)
{
// This should not need a cast (ie. BindCppObjectNoCast), but performing the cast
// here means callbacks receive down-casted object when passed by-ptr, which is
// needed for object identity. The latter case is assumed to be more common than
// conversion of (global) objects.
    return BindCppObject((Cppyy::TCppObject_t)address, fClass);
}

//----------------------------------------------------------------------------
bool CPyCppyy::InstanceConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// assign value to C++ instance living at <address> through assignment operator
    PyObject* pyobj = BindCppObjectNoCast(address, fClass);
#if PY_VERSION_HEX >= 0x03080000
    PyObject* result = PyObject_CallMethodOneArg(pyobj, PyStrings::gAssign, value);
#else
    PyObject* result = PyObject_CallMethod(pyobj, (char*)"__assign__", (char*)"O", value);
#endif
    Py_DECREF(pyobj);

    if (result) {
        Py_DECREF(result);
        return true;
    }
    return false;
}


//----------------------------------------------------------------------------
bool CPyCppyy::InstanceRefConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance&, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject, fIsConst ? fClass : (Cppyy::TCppType_t)0);
    if (pyobj) {

    // reject moves
        if (pyobj->fFlags & CPPInstance::kIsRValue)
            return false;

    // smart pointers can end up here in case of a move, so preferentially match
    // the smart type directly
        bool argset = false;
        Cppyy::TCppType_t cls = 0;
        if (pyobj->IsSmart()) {
            cls = pyobj->ObjectIsA(false);
            if (cls && Cppyy::IsSubtype(cls, fClass)) {
                para.fValue.fVoidp = pyobj->GetObjectRaw();
                argset = true;
            }
        }

        if (!argset) {
            cls = pyobj->ObjectIsA();
            if (cls && Cppyy::IsSubtype(cls, fClass)) {
                para.fValue.fVoidp = pyobj->GetObject();
                argset = true;
            }
        }

        if (argset) {
        // do not allow null pointers through references
            if (!para.fValue.fVoidp) {
                PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
                return false;
            }

        // calculate offset between formal and actual arguments
            if (cls != fClass) {
                para.fValue.fIntPtr += Cppyy::GetBaseOffset(
                    cls, fClass, para.fValue.fVoidp, 1 /* up-cast */);
            }

            para.fTypeCode = 'V';
            return true;
        }
    }

    if (!fIsConst)      // no implicit conversion possible
        return false;

    return (bool)ConvertImplicit(fClass, pyobject, para, ctxt);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceRefConverter::FromMemory(void* address)
{
    return BindCppObjectNoCast((Cppyy::TCppObject_t)address, fClass, CPPInstance::kIsReference);
}

//----------------------------------------------------------------------------
bool CPyCppyy::InstanceMoveConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance&&, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject, fClass, true /* accept_rvalue */);
    if (!pyobj || (pyobj->fFlags & CPPInstance::kIsLValue)) {
    // implicit conversion is fine as the temporary by definition is moveable
        return (bool)ConvertImplicit(fClass, pyobject, para, ctxt);
    }

// moving is same as by-ref, but have to check that move is allowed
    int moveit_reason = 0;
    if (pyobj->fFlags & CPPInstance::kIsRValue) {
        pyobj->fFlags &= ~CPPInstance::kIsRValue;
        moveit_reason = 2;
    } else if (PyUnstable_Object_IsUniqueReferencedTemporary(pyobject)) {
        moveit_reason = 1;
    }

    if (moveit_reason) {
        bool result = this->InstanceRefConverter::SetArg(pyobject, para, ctxt);
        if (!result && moveit_reason == 2)       // restore the movability flag?
            ((CPPInstance*)pyobject)->fFlags |= CPPInstance::kIsRValue;
        return result;
    }

    PyErr_SetString(PyExc_ValueError, "object is not an rvalue");
    return false;      // not a temporary or movable object
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
bool CPyCppyy::InstancePtrPtrConverter<ISREFERENCE>::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// convert <pyobject> to C++ instance**, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject);
    if (!pyobj) {
        if (!ISREFERENCE && (pyobject == gNullPtrObject || pyobject == gDefaultObject)) {
        // allow nullptr as a special case
            para.fValue.fVoidp = nullptr;
            para.fTypeCode = 'p';
            return true;
        }
        return false;              // not a cppyy object (TODO: handle SWIG etc.)
    }

    if (Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && !UseStrictOwnership(ctxt))
            pyobj->CppOwns();

    // set pointer (may be null) and declare success
        if (pyobj->fFlags & CPPInstance::kIsReference) // already a ptr to object?
            para.fValue.fVoidp = pyobj->GetObjectRaw();
        else
            para.fValue.fVoidp = &pyobj->GetObjectRaw();
        para.fTypeCode = ISREFERENCE ? 'V' : 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
PyObject* CPyCppyy::InstancePtrPtrConverter<ISREFERENCE>::FromMemory(void* address)
{
// construct python object from C++ instance* read at <address>
    return BindCppObject(*(void**)address, fClass, CPPInstance::kIsReference | CPPInstance::kIsPtrPtr);
}

//----------------------------------------------------------------------------
template <bool ISREFERENCE>
bool CPyCppyy::InstancePtrPtrConverter<ISREFERENCE>::ToMemory(
    PyObject* value, void* address, PyObject* /* ctxt */)
{
// convert <value> to C++ instance*, write it at <address>
    CPPInstance* pyobj = GetCppInstance(value);
    if (!pyobj) {
        if (value == gNullPtrObject || value == gDefaultObject) {
        // allow nullptr as a special case
            *(void**)address = nullptr;
            return true;
        }
        return false;              // not a cppyy object (TODO: handle SWIG etc.)
    }

    if (Cppyy::IsSubtype(pyobj->ObjectIsA(), fClass)) {
    // depending on memory policy, some objects need releasing when passed into functions
        if (!KeepControl() && CallContext::sMemoryPolicy != CallContext::kUseStrict)
            pyobj->CppOwns();

    // register the value for potential recycling
        MemoryRegulator::RegisterPyObject(pyobj, pyobj->GetObject());

    // set pointer (may be null) and declare success
        *(void**)address = pyobj->GetObject();
        return true;
    }

    return false;
}


namespace CPyCppyy {
// Instantiate the templates
    template class CPyCppyy::InstancePtrConverter<true>;
    template class CPyCppyy::InstancePtrConverter<false>;
    template class CPyCppyy::InstancePtrPtrConverter<true>;
    template class CPyCppyy::InstancePtrPtrConverter<false>;
}

//----------------------------------------------------------------------------
bool CPyCppyy::InstanceArrayConverter::SetArg(
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
        para.fValue.fVoidp = ((CPPInstance*)first)->GetObject();
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::InstanceArrayConverter::FromMemory(void* address)
{
// construct python tuple of instances from C++ array read at <address>
    return BindCppObjectArray(*(char**)address, fClass, fShape);
}

//----------------------------------------------------------------------------
bool CPyCppyy::InstanceArrayConverter::ToMemory(
    PyObject* /* value */, void* /* address */, PyObject* /* ctxt */)
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
    CPPInstance* pyobj = GetCppInstance(pyobject);
    if (pyobj) {
        para.fValue.fVoidp = &pyobj->GetObjectRaw();
        para.fTypeCode = 'V';
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------
CPyCppyy::VoidPtrPtrConverter::VoidPtrPtrConverter(cdims_t dims) :
        fShape(dims) {
    fIsFixed = dims ? fShape[0] != UNKNOWN_SIZE : false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::VoidPtrPtrConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* /* ctxt */)
{
// convert <pyobject> to C++ void**, set arg for call
    CPPInstance* pyobj = GetCppInstance(pyobject);
    if (pyobj) {
    // this is a C++ object, take and set its address
        para.fValue.fVoidp = &pyobj->GetObjectRaw();
        para.fTypeCode = 'p';
        return true;
    } else if (IsPyCArgObject(pyobject)) {
        CPyCppyy_tagPyCArgObject* carg = (CPyCppyy_tagPyCArgObject*)pyobject;
        if (carg->obj) {
            para.fValue.fVoidp = (void*)((CPyCppyy_tagCDataObject*)carg->obj)->b_ptr;
            para.fTypeCode = 'p';
            return true;
        }
    }

// buffer objects are allowed under "user knows best" (this includes the buffer
// interface to ctypes.c_void_p, which results in a void**)
    Py_ssize_t buflen = Utility::GetBuffer(pyobject, '*', 1, para.fValue.fVoidp, false);

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
// read a void** from address; since this is unknown, uintptr_t is used (user can cast)
    if (!address || *(ptrdiff_t*)address == 0) {
        Py_INCREF(gNullPtrObject);
        return gNullPtrObject;
    }
    if (!fIsFixed)
        return CreatePointerView((uintptr_t**)address, fShape);
    return CreatePointerView(*(uintptr_t**)address, fShape);
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

bool CPyCppyy::PyObjectConverter::ToMemory(PyObject* value, void* address, PyObject* /* ctxt */)
{
// no conversion needed, write <value> at <address>
    Py_INCREF(value);
    Py_XDECREF(*((PyObject**)address));
    *((PyObject**)address) = value;
    return true;
}


//- function pointer converter -----------------------------------------------
static unsigned int sWrapperCounter = 0;
// cache mapping signature/return type to python callable and corresponding wrapper
typedef std::string RetSigKey_t;
static std::map<RetSigKey_t, std::vector<void*>> sWrapperFree;
static std::map<RetSigKey_t, std::map<PyObject*, void*>> sWrapperLookup;
static std::map<PyObject*, std::pair<void*, RetSigKey_t>> sWrapperWeakRefs;
static std::map<void*, PyObject**> sWrapperReference;

static PyObject* WrapperCacheEraser(PyObject*, PyObject* pyref)
{
    auto ipos = sWrapperWeakRefs.find(pyref);
    if (ipos != sWrapperWeakRefs.end()) {
        auto key = ipos->second.second;

    // disable this callback and store on free list for possible re-use
        void* wpraddress = ipos->second.first;
        PyObject** oldref = sWrapperReference[wpraddress];
        const auto& lookup = sWrapperLookup.find(key);
        if (lookup != sWrapperLookup.end()) lookup->second.erase(*oldref);
        *oldref = nullptr;        // to detect deletions
        sWrapperFree[ipos->second.second].push_back(wpraddress);

    // clean up and remove weak reference from admin
        Py_DECREF(ipos->first);
        sWrapperWeakRefs.erase(ipos);
    }

    Py_RETURN_NONE;
}
static PyMethodDef gWrapperCacheEraserMethodDef = {
    const_cast<char*>("internal_WrapperCacheEraser"),
    (PyCFunction)WrapperCacheEraser,
    METH_O, nullptr
};

static void* PyFunction_AsCPointer(PyObject* pyobject,
    const std::string& rettype, const std::string& signature)
{
// Convert a bound C++ function pointer or callable python object to a C-style
// function pointer. The former is direct, the latter involves a JIT-ed wrapper.
    static PyObject* sWrapperCacheEraser = PyCFunction_New(&gWrapperCacheEraserMethodDef, nullptr);

    using namespace CPyCppyy;

    if (CPPOverload_Check(pyobject)) {
        CPPOverload* ol = (CPPOverload*)pyobject;
        if (!ol->fMethodInfo || ol->fMethodInfo->fMethods.empty())
            return nullptr;

    // find the overload with matching signature
        for (auto& m : ol->fMethodInfo->fMethods) {
            PyObject* sig = m->GetSignature(false);
            bool found = signature == CPyCppyy_PyText_AsString(sig);
            Py_DECREF(sig);
            if (found) {
                void* fptr = (void*)m->GetFunctionAddress();
                if (fptr) return fptr;
                break;  // fall-through, with calling through Python
            }
        }
    }

    if (TemplateProxy_Check(pyobject)) {
    // get the actual underlying template matching the signature
        TemplateProxy* pytmpl = (TemplateProxy*)pyobject;
        std::string fullname = pytmpl->fTI->fCppName;
        if (pytmpl->fTemplateArgs)
            fullname += CPyCppyy_PyText_AsString(pytmpl->fTemplateArgs);
        Cppyy::TCppScope_t scope = ((CPPClass*)pytmpl->fTI->fPyClass)->fCppType;
        Cppyy::TCppMethod_t cppmeth = Cppyy::GetMethodTemplate(scope, fullname, signature);
        if (cppmeth) {
            void* fptr = (void*)Cppyy::GetFunctionAddress(cppmeth, false);
            if (fptr) return fptr;
        }
        // fall-through, with calling through Python
    }

    if (PyObject_IsInstance(pyobject, (PyObject*)GetCTypesType(ct_c_funcptr))) {
    // ctypes function pointer
        void* fptr = *(void**)((CPyCppyy_tagCDataObject*)pyobject)->b_ptr;
        return fptr;
    }

    if (PyCallable_Check(pyobject)) {
    // generic python callable: create a C++ wrapper function
        void* wpraddress = nullptr;

    // re-use existing wrapper if possible
        auto key = rettype+signature;
        const auto& lookup = sWrapperLookup.find(key);
        if (lookup != sWrapperLookup.end()) {
            const auto& existing = lookup->second.find(pyobject);
            if (existing != lookup->second.end() && *sWrapperReference[existing->second] == pyobject)
                wpraddress = existing->second;
        }

     // check for a pre-existing, unused, wrapper if not found
        if (!wpraddress) {
           const auto& freewrap = sWrapperFree.find(key);
           if (freewrap != sWrapperFree.end() && !freewrap->second.empty()) {
               wpraddress = freewrap->second.back();
               freewrap->second.pop_back();
               *(sWrapperReference[wpraddress]) = pyobject;
               sWrapperLookup[key][pyobject] = wpraddress;
               PyObject* wref = PyWeakref_NewRef(pyobject, sWrapperCacheEraser);
               if (wref) sWrapperWeakRefs[wref] = std::make_pair(wpraddress, key);
               else PyErr_Clear();     // happens for builtins which don't need this
           }
        }

     // create wrapper if no re-use possible
        if (!wpraddress) {
            if (!Utility::IncludePython())
                return nullptr;

        // extract argument types
            const std::vector<std::string>& argtypes = TypeManip::extract_arg_types(signature);
            int nArgs = (int)argtypes.size();

        // wrapper name
            std::ostringstream wname;
            wname << "fptr_wrap" << ++sWrapperCounter;

       // build wrapper function code
            std::ostringstream code;
            code << "namespace __cppyy_internal {\n  "
                 << rettype << " " << wname.str() << "(";
            for (int i = 0; i < nArgs; ++i) {
                code << argtypes[i] << " arg" << i;
                if (i != nArgs-1) code << ", ";
            }
            code << ") {\n";

        // start function body
            Utility::ConstructCallbackPreamble(rettype, argtypes, code);

        // create a referenceable pointer
            PyObject** ref = new PyObject*{pyobject};

        // function call itself and cleanup
            code << "    PyObject** ref = (PyObject**)" << (intptr_t)ref << ";\n"
                    "    PyObject* pyresult = nullptr;\n"
                    "    if (*ref) pyresult = PyObject_CallFunctionObjArgs(*ref";
            for (int i = 0; i < nArgs; ++i)
                code << ", pyargs[" << i << "]";
            code << ", NULL);\n"
                    "    else PyErr_SetString(PyExc_TypeError, \"callable was deleted\");\n";

        // close
            Utility::ConstructCallbackReturn(rettype, nArgs, code);

        // end of namespace
            code << "}";

        // finally, compile the code
            if (!Cppyy::Compile(code.str()))
                return nullptr;

        // TODO: is there no easier way?
            static Cppyy::TCppScope_t scope = Cppyy::GetScope("__cppyy_internal");
            const auto& idx = Cppyy::GetMethodIndicesFromName(scope, wname.str());
            wpraddress = Cppyy::GetFunctionAddress(Cppyy::GetMethod(scope, idx[0]), false);
            sWrapperReference[wpraddress] = ref;

        // cache the new wrapper
            sWrapperLookup[key][pyobject] = wpraddress;
            PyObject* wref = PyWeakref_NewRef(pyobject, sWrapperCacheEraser);
            if (wref) sWrapperWeakRefs[wref] = std::make_pair(wpraddress, key);
            else PyErr_Clear();     // happens for builtins which don't need this
        }

    // now pass the pointer to the wrapper function (may be null)
        return wpraddress;
    }

    return nullptr;
}

bool CPyCppyy::FunctionPointerConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// special case: allow nullptr singleton:
    if (pyobject == gNullPtrObject || pyobject == gDefaultObject) {
        para.fValue.fVoidp = nullptr;
        para.fTypeCode = 'p';
        return true;
    }

// normal case, get a function pointer
    void* fptr = PyFunction_AsCPointer(pyobject, fRetType, fSignature);
    if (fptr) {
        SetLifeLine(ctxt->fPyContext, pyobject, (intptr_t)this);
        para.fValue.fVoidp = fptr;
        para.fTypeCode = 'p';
        return true;
    }

    return false;
}

PyObject* CPyCppyy::FunctionPointerConverter::FromMemory(void* address)
{
// A function pointer in clang is represented by a Type, not a FunctionDecl and it's
// not possible to get the latter from the former: the backend will need to support
// both. Since that is far in the future, we'll use a std::function instead.
    if (address)
        return Utility::FuncPtr2StdFunction(fRetType, fSignature, *(void**)address);
    PyErr_SetString(PyExc_TypeError, "can not convert null function pointer");
    return nullptr;
}

bool CPyCppyy::FunctionPointerConverter::ToMemory(
    PyObject* pyobject, void* address, PyObject* ctxt)
{
// special case: allow nullptr singleton:
    if (pyobject == gNullPtrObject || pyobject == gDefaultObject) {
        *((void**)address) = nullptr;
        return true;
    }

// normal case, get a function pointer
    void* fptr = PyFunction_AsCPointer(pyobject, fRetType, fSignature);
    if (fptr) {
        SetLifeLine(ctxt, pyobject, (intptr_t)address);
        *((void**)address) = fptr;
        return true;
    }

    return false;
}


//- std::function converter --------------------------------------------------
bool CPyCppyy::StdFunctionConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
// prefer normal "object" conversion
    CallContextRAII<CallContext::kNoImplicit> noimp(ctxt);
    if (fConverter->SetArg(pyobject, para, ctxt))
        return true;

    PyErr_Clear();

// else create a wrapper function
    if (this->FunctionPointerConverter::SetArg(pyobject, para, ctxt)) {
    // retrieve the wrapper pointer and capture it in a temporary std::function,
    // then try normal conversion a second time
        PyObject* func = this->FunctionPointerConverter::FromMemory(&para.fValue.fVoidp);
        if (func) {
            SetLifeLine(ctxt->fPyContext, func, (intptr_t)this);
            bool result = fConverter->SetArg(func, para, ctxt);
            if (result) ctxt->AddTemporary(func);
            else Py_DECREF(func);
            return result;
        }
    }

    return false;
}

PyObject* CPyCppyy::StdFunctionConverter::FromMemory(void* address)
{
    return fConverter->FromMemory(address);
}

bool CPyCppyy::StdFunctionConverter::ToMemory(PyObject* value, void* address, PyObject* ctxt)
{
// if the value is not an std::function<> but a generic Python callable, the
// conversion is done through the assignment, which may involve a temporary
    if (address) SetLifeLine(ctxt, value, (intptr_t)address);
    return fConverter->ToMemory(value, address, ctxt);
}


//- smart pointer converters -------------------------------------------------
bool CPyCppyy::SmartPtrConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
    char typeCode = fIsRef ? 'p' : 'V';

    if (!CPPInstance_Check(pyobject)) {
        // TODO: not sure how this is correct for pass-by-ref nor does it help with
        // implicit conversions for pass-by-value
        if (fIsRef && GetAddressSpecialCase(pyobject, para.fValue.fVoidp)) {
            para.fTypeCode = typeCode;      // allow special cases such as nullptr
            return true;
        }

        return false;
    }

    CPPInstance* pyobj = (CPPInstance*)pyobject;
    Cppyy::TCppType_t oisa = pyobj->ObjectIsA();

// for the case where we have a 'hidden' smart pointer:
    if (Cppyy::TCppType_t tsmart = pyobj->GetSmartIsA()) {
        if (Cppyy::IsSubtype(tsmart, fSmartPtrType)) {
        // depending on memory policy, some objects need releasing when passed into functions
            if (!fKeepControl && !UseStrictOwnership(ctxt))
                ((CPPInstance*)pyobject)->CppOwns();

        // calculate offset between formal and actual arguments
            para.fValue.fVoidp = pyobj->GetSmartObject();
            if (tsmart != fSmartPtrType) {
                para.fValue.fIntPtr += Cppyy::GetBaseOffset(
                    tsmart, fSmartPtrType, para.fValue.fVoidp, 1 /* up-cast */);
            }

        // set pointer (may be null) and declare success
            para.fTypeCode = typeCode;
            return true;
        }
    }

// for the case where we have an 'exposed' smart pointer:
    if (!pyobj->IsSmart() && Cppyy::IsSubtype(oisa, fSmartPtrType)) {
    // calculate offset between formal and actual arguments
        para.fValue.fVoidp = pyobj->GetObject();
        if (oisa != fSmartPtrType) {
            para.fValue.fIntPtr += Cppyy::GetBaseOffset(
                oisa, fSmartPtrType, para.fValue.fVoidp, 1 /* up-cast */);
        }

    // set pointer (may be null) and declare success
        para.fTypeCode = typeCode;
        return true;
    }

// The automatic conversion of ordinary obejcts to smart pointers is disabled
// for PyROOT because it can cause trouble with overload resolution. If a
// function has overloads for both ordinary objects and smart pointers, then
// the implicit conversion to smart pointers can result in the smart pointer
// overload being hit, even though there would be an overload for the regular
// object. Since PyROOT didn't have this feature before 6.32 anyway, disabling
// it was the safest option.
#if 0
// for the case where we have an ordinary object to convert
    if (!pyobj->IsSmart() && Cppyy::IsSubtype(oisa, fUnderlyingType)) {
    // create the relevant smart pointer and make the pyobject "smart"
        CPPInstance* pysmart = (CPPInstance*)ConvertImplicit(fSmartPtrType, pyobject, para, ctxt, false);
        if (!CPPInstance_Check(pysmart)) {
            Py_XDECREF(pysmart);
            return false;
        }

    // copy internals from the fresh smart object to the original, making it smart
        pyobj->GetObjectRaw() = pysmart->GetSmartObject();
        pyobj->SetSmart(CreateScopeProxy(fSmartPtrType)); //(PyObject*)Py_TYPE(pysmart));
        pyobj->PythonOwns();
        pysmart->CppOwns();
        Py_DECREF(pysmart);

        return true;
    }
#endif

// final option, try mapping pointer types held (TODO: do not allow for non-const ref)
    if (pyobj->IsSmart() && Cppyy::IsSubtype(oisa, fUnderlyingType)) {
        para.fValue.fVoidp = ((CPPInstance*)pyobject)->GetSmartObject();
        para.fTypeCode = 'V';
        return true;
    }

    return false;
}

PyObject* CPyCppyy::SmartPtrConverter::FromMemory(void* address)
{
    if (!address || !fSmartPtrType)
        return nullptr;

    return BindCppObjectNoCast(address, fSmartPtrType);
}

bool CPyCppyy::SmartPtrConverter::ToMemory(PyObject* value, void* address, PyObject*)
{
// assign value to C++ instance living at <address> through assignment operator (this
// is similar to InstanceConverter::ToMemory, but prevents wrapping the smart ptr)
    PyObject* pyobj = BindCppObjectNoCast(address, fSmartPtrType, CPPInstance::kNoWrapConv);
#if PY_VERSION_HEX >= 0x03080000
    PyObject* result = PyObject_CallMethodOneArg(pyobj, PyStrings::gAssign, value);
#else
    PyObject* result = PyObject_CallMethod(pyobj, (char*)"__assign__", (char*)"O", value);
#endif
    Py_DECREF(pyobj);

    if (result) {
        Py_DECREF(result);
        return true;
    }
    return false;
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
#elif defined (_MSC_VER)
struct faux_initlist
{
     typedef char* iterator;
     iterator _M_array; // ie. _First;
     iterator _Last;
};
#else
#define NO_KNOWN_INITIALIZER_LIST 1
#endif

} // unnamed namespace

CPyCppyy::InitializerListConverter::InitializerListConverter(Cppyy::TCppType_t klass, std::string const &value_type)

    : InstanceConverter{klass},
      fValueTypeName{value_type},
      fValueType{Cppyy::GetScope(value_type)},
      fValueSize{Cppyy::SizeOf(value_type)}
{
}

CPyCppyy::InitializerListConverter::~InitializerListConverter()
{
    for (Converter *converter : fConverters) {
       if (converter && converter->HasState()) delete converter;
    }
    if (fBuffer) Clear();
}

void CPyCppyy::InitializerListConverter::Clear() {
    if (fValueType) {
        faux_initlist* fake = (faux_initlist*)fBuffer;
#if defined (_LIBCPP_INITIALIZER_LIST) || defined(__GNUC__)
        for (faux_initlist::size_type i = 0; i < fake->_M_len; ++i) {
#elif defined (_MSC_VER)
        for (size_t i = 0; (fake->_M_array+i*fValueSize) != fake->_Last; ++i) {
#endif
            void* memloc = (char*)fake->_M_array + i*fValueSize;
            Cppyy::CallDestructor(fValueType, (Cppyy::TCppObject_t)memloc);
        }
    }

    free(fBuffer);
    fBuffer = nullptr;
}

bool CPyCppyy::InitializerListConverter::SetArg(
    PyObject* pyobject, Parameter& para, CallContext* ctxt)
{
#ifdef NO_KNOWN_INITIALIZER_LIST
    return false;
#else
    if (fBuffer) Clear();

// convert the given argument to an initializer list temporary; this is purely meant
// to be a syntactic thing, so only _python_ sequences are allowed; bound C++ proxies
// (likely explicitly created std::initializer_list, go through an instance converter
    if (!PySequence_Check(pyobject) || CPyCppyy_PyText_Check(pyobject)
#if PY_VERSION_HEX >= 0x03000000
        || PyBytes_Check(pyobject)
#else
        || PyUnicode_Check(pyobject)
#endif
        )
        return false;

    if (CPPInstance_Check(pyobject))
        return this->InstanceConverter::SetArg(pyobject, para, ctxt);

    void* buf = nullptr;
    Py_ssize_t buflen = Utility::GetBuffer(pyobject, '*', (int)fValueSize, buf, true);
    faux_initlist* fake = nullptr;
    size_t entries = 0;
    if (buf && buflen) {
    // dealing with an array here, pass on whole-sale
        fake = (faux_initlist*)malloc(sizeof(faux_initlist));
        fBuffer = (void*)fake;
        fake->_M_array = (faux_initlist::iterator)buf;
#if defined (_LIBCPP_INITIALIZER_LIST) || defined(__GNUC__)
        fake->_M_len = (faux_initlist::size_type)buflen;
#elif defined (_MSC_VER)
        fake->_Last = fake->_M_array+buflen*fValueSize;
#endif
    } else if (fValueSize) {
    // Remove any errors set by GetBuffer(); note that if the argument was an array
    // that failed to extract because of a type mismatch, the following will perform
    // a (rather inefficient) copy. No warning is issued b/c e.g. numpy doesn't do
    // so either.
        PyErr_Clear();

    // can only construct empty lists, so use a fake initializer list
        size_t len = (size_t)PySequence_Size(pyobject);
        fake = (faux_initlist*)malloc(sizeof(faux_initlist)+fValueSize*len);
        fBuffer = (void*)fake;
        fake->_M_array = (faux_initlist::iterator)((char*)fake+sizeof(faux_initlist));
#if defined (_LIBCPP_INITIALIZER_LIST) || defined(__GNUC__)
        fake->_M_len = (faux_initlist::size_type)len;
        for (faux_initlist::size_type i = 0; i < fake->_M_len; ++i) {
#elif defined (_MSC_VER)
        fake->_Last = fake->_M_array+len*fValueSize;
        for (size_t i = 0; (fake->_M_array+i*fValueSize) != fake->_Last; ++i) {
#endif
            PyObject* item = PySequence_GetItem(pyobject, i);
            bool convert_ok = false;
            if (item) {
                if (fConverters.empty())
                    fConverters.emplace_back(CreateConverter(fValueTypeName));
                if (!fConverters.back()) {
                    if (CPPInstance_Check(item)) {
                    // by convention, use byte copy
                        memcpy((char*)fake->_M_array + i*fValueSize,
                               ((CPPInstance*)item)->GetObject(), fValueSize);
                        convert_ok = true;
                    }
                } else {
                    void* memloc = (char*)fake->_M_array + i*fValueSize;
                    if (fValueType) {
                    // we need to construct a default object for the constructor to assign into; this is
                    // clunky, but the use of a copy constructor isn't much better as the Python object
                    // need not be a C++ object
                        memloc = (void*)Cppyy::Construct(fValueType, memloc);
                        if (memloc) entries += 1;
                        else {
                           PyErr_SetString(PyExc_TypeError,
                              "default ctor needed for initializer list of objects");
                        }
                    }
                    if (memloc) {
                        if (i >= fConverters.size()) {
                            fConverters.emplace_back(CreateConverter(fValueTypeName));
                        }
                        convert_ok = fConverters[i]->ToMemory(item, memloc);
                    }
                }


                Py_DECREF(item);
            } else
                PyErr_Format(PyExc_TypeError, "failed to get item %d from sequence", (int)i);

            if (!convert_ok) {
#if defined (_LIBCPP_INITIALIZER_LIST) || defined(__GNUC__)
                fake->_M_len = (faux_initlist::size_type)entries;
#elif defined (_MSC_VER)
                fake->_Last = fake->_M_array+entries*fValueSize;
#endif
                Clear();
                return false;
            }
        }
    }

    if (!fake)     // no buffer and value size indeterminate
        return false;

    para.fValue.fVoidp = (void*)fake;
    para.fTypeCode = 'V';     // means ptr that backend has to free after call
    return true;
#endif
}

//----------------------------------------------------------------------------
bool CPyCppyy::NotImplementedConverter::SetArg(PyObject*, Parameter&, CallContext*)
{
// raise a NotImplemented exception to take a method out of overload resolution
    PyErr_SetString(PyExc_NotImplementedError, "this method cannot (yet) be called");
    return false;
}


//- helper to refactor some code from CreateConverter ------------------------
static inline CPyCppyy::Converter* selectInstanceCnv(Cppyy::TCppScope_t klass,
        const std::string& cpd, CPyCppyy::cdims_t dims, bool isConst, bool control)
{
    using namespace CPyCppyy;
    Converter* result = nullptr;

    if (cpd == "**" || cpd == "*[]" || cpd == "&*")
        result = new InstancePtrPtrConverter<false>(klass, control);
    else if (cpd == "*&")
        result = new InstancePtrPtrConverter<true>(klass, control);
    else if (cpd == "*" && dims.ndim() == UNKNOWN_SIZE) {
        if (isConst) result = new InstancePtrConverter<true>(klass, control);
        else result = new InstancePtrConverter<false>(klass, control);
    }
    else if (cpd == "&")
        result = new InstanceRefConverter(klass, isConst);
    else if (cpd == "&&")
        result = new InstanceMoveConverter(klass);
    else if (cpd == "[]" || dims)
        result = new InstanceArrayConverter(klass, dims, false);
    else if (cpd == "")             // by value
        result = new InstanceConverter(klass, true);

    return result;
}

//- factories ----------------------------------------------------------------
CPYCPPYY_EXPORT
CPyCppyy::Converter* CPyCppyy::CreateConverter(const std::string& fullType, cdims_t dims)
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
    if (h != gConvFactories.end()) {
        return (h->second)(dims);
    }

// resolve typedefs etc.
    const std::string& resolvedType = Cppyy::ResolveName(fullType);

// a full, qualified matching converter is preferred
    if (resolvedType != fullType) {
        h = gConvFactories.find(resolvedType);
        if (h != gConvFactories.end())
            return (h->second)(dims);
    }

//-- nothing? ok, collect information about the type and possible qualifiers/decorators
    bool isConst = strncmp(resolvedType.c_str(), "const", 5) == 0;
    const std::string& cpd = TypeManip::compound(resolvedType);
    std::string realType   = TypeManip::clean_type(resolvedType, false, true);

// accept unqualified type (as python does not know about qualifiers)
    h = gConvFactories.find((isConst ? "const " : "") + realType + cpd);
    if (h != gConvFactories.end())
        return (h->second)(dims);

// drop const, as that is mostly meaningless to python (with the exception
// of c-strings, but those are specialized in the converter map)
    if (isConst) {
        h = gConvFactories.find(realType + cpd);
        if (h != gConvFactories.end())
            return (h->second)(dims);
    }

//-- still nothing? try pointer instead of array (for builtins)
    if (cpd.compare(0, 3, "*[]") == 0) {
    // special case, array of pointers
        h = gConvFactories.find(realType + " ptr");
        if (h != gConvFactories.end()) {
        // upstream treats the pointer type as the array element type, but that pointer is
        // treated as a low-level view as well, unless it's a void*/char* so adjust the dims
            if (realType != "void" && realType != "char") {
                dim_t newdim = dims.ndim() == UNKNOWN_SIZE ? 2 : dims.ndim()+1;
                dims_t newdims = dims_t(newdim);
            // TODO: sometimes the array size is known and can thus be verified; however,
            // currently the meta layer does not provide this information
                newdims[0] = dims ? dims[0] : UNKNOWN_SIZE;     // the array
                newdims[1] = UNKNOWN_SIZE;                      // the pointer
                if (2 < newdim) {
                    for (int i = 2; i < (newdim-1); ++i)
                        newdims[i] = dims[i-1];
                }

                return (h->second)(newdims);
            }
            return (h->second)(dims);
        }

    } else if (!cpd.empty() && (std::string::size_type)std::count(cpd.begin(), cpd.end(), '*') == cpd.size()) {
    // simple array; set or resize as necessary
        h = gConvFactories.find(realType + " ptr");
        if (h != gConvFactories.end())
            return (h->second)((!dims && 1 < cpd.size()) ? dims_t(cpd.size()) : dims);

    }  else if (2 <= cpd.size() && (std::string::size_type)std::count(cpd.begin(), cpd.end(), '[') == cpd.size() / 2) {
    // fixed array, dims will have size if available
        h = gConvFactories.find(realType + " ptr");
        if (h != gConvFactories.end())
            return (h->second)(dims);
    }

//-- special case: initializer list
    if (realType.compare(0, 16, "initializer_list") == 0) {
    // get the type of the list and create a converter (TODO: get hold of value_type?)
        auto pos = realType.find('<');
        std::string value_type = realType.substr(pos+1, realType.size()-pos-2);
        return new InitializerListConverter(Cppyy::GetScope(realType), value_type);
    }

//-- still nothing? use a generalized converter
    bool control = cpd == "&" || isConst;

//-- special case: std::function
    auto pos = resolvedType.find("function<");
    if (pos == 0 /* no std:: */ || pos == 5 /* with std:: */ ||
        pos == 6 /* const no std:: */ || pos == 11 /* const with std:: */ ) {

    // get actual converter for normal passing
        Converter* cnv = selectInstanceCnv(
            Cppyy::GetScope(realType), cpd, dims, isConst, control);

        if (cnv) {
        // get the type of the underlying (TODO: use target_type?)
            auto pos1 = resolvedType.find("(", pos+9);
            auto pos2 = resolvedType.rfind(")");
            if (pos1 != std::string::npos && pos2 != std::string::npos) {
                auto sz1 = pos1-pos-9;
                if (resolvedType[pos+9+sz1-1] == ' ') sz1 -= 1;

                return new StdFunctionConverter(cnv,
                    resolvedType.substr(pos+9, sz1), resolvedType.substr(pos1, pos2-pos1+1));
            } else if (cnv->HasState())
                delete cnv;
        }
    }

// converters for known C++ classes and default (void*)
    Converter* result = nullptr;
    if (Cppyy::TCppScope_t klass = Cppyy::GetScope(realType)) {
        Cppyy::TCppType_t raw{0};
        if (Cppyy::GetSmartPtrInfo(realType, &raw, nullptr)) {
            if (cpd == "") {
                result = new SmartPtrConverter(klass, raw, control);
            } else if (cpd == "&") {
                result = new SmartPtrConverter(klass, raw);
            } else if (cpd == "*" && dims.ndim() == UNKNOWN_SIZE) {
                result = new SmartPtrConverter(klass, raw, control, true);
            }
        }

        if (!result) {
        // CLING WORKAROUND -- special case for STL iterators
            if (Utility::IsSTLIterator(realType)) {
                static STLIteratorConverter c;
                result = &c;
            } else
       // -- CLING WORKAROUND
                result = selectInstanceCnv(klass, cpd, dims, isConst, control);
        }
    } else {
        std::smatch sm;
        if (std::regex_search(resolvedType, sm, s_fnptr)) {
        // this is a function pointer
            auto pos1 = sm.position(0);
            auto pos2 = resolvedType.rfind(')');
            result = new FunctionPointerConverter(
                resolvedType.substr(0, pos1), resolvedType.substr(pos1+sm.length(), pos2-1));
        }
    }

    if (!result && cpd == "&&") {
    // for builtin, can use const-ref for r-ref
        h = gConvFactories.find("const " + realType + "&");
        if (h != gConvFactories.end())
            return (h->second)(dims);
    // else, unhandled moves
        result = new NotImplementedConverter();
    }

    if (!result && h != gConvFactories.end())
    // converter factory available, use it to create converter
        result = (h->second)(dims);
    else if (!result) {
    // default to something reasonable, assuming "user knows best"
        if (cpd.size() == 2 && cpd != "&&") // "**", "*[]", "*&"
            result = new VoidPtrPtrConverter(dims.ndim());
        else if (!cpd.empty())
            result = new VoidArrayConverter();        // "user knows best"
        else
            result = new NotImplementedConverter();   // fails on use
    }

    return result;
}

//----------------------------------------------------------------------------
CPYCPPYY_EXPORT
void CPyCppyy::DestroyConverter(Converter* p)
{
    if (p && p->HasState())
        delete p;  // state-less converters are always shared
}

//----------------------------------------------------------------------------
CPYCPPYY_EXPORT
bool CPyCppyy::RegisterConverter(const std::string& name, cf_t fac)
{
// register a custom converter
    auto f = gConvFactories.find(name);
    if (f != gConvFactories.end())
        return false;

    gConvFactories[name] = fac;
    return true;
}

//----------------------------------------------------------------------------
CPYCPPYY_EXPORT
bool CPyCppyy::RegisterConverterAlias(const std::string& name, const std::string& target)
{
// register a custom converter that is a reference to an existing converter
    auto f = gConvFactories.find(name);
    if (f != gConvFactories.end())
        return false;

    auto t = gConvFactories.find(target);
    if (t == gConvFactories.end())
        return false;

    gConvFactories[name] = t->second;
    return true;
}

//----------------------------------------------------------------------------
CPYCPPYY_EXPORT
bool CPyCppyy::UnregisterConverter(const std::string& name)
{
// remove a custom converter
    auto f = gConvFactories.find(name);
    if (f != gConvFactories.end()) {
        gConvFactories.erase(f);
        return true;
    }
    return false;
}


//----------------------------------------------------------------------------
namespace {

using namespace CPyCppyy;

inline static
std::string::size_type dims2stringsz(cdims_t d) {
    return (d && d.ndim() != UNKNOWN_SIZE) ? d[0] : std::string::npos;
}

#define STRINGVIEW "basic_string_view<char,char_traits<char> >"
#define WSTRING1 "std::basic_string<wchar_t>"
#define WSTRING2 "std::basic_string<wchar_t,std::char_traits<wchar_t>,std::allocator<wchar_t>>"

//-- aliasing special case: C complex (is binary compatible with C++ std::complex)
#ifndef _WIN32
#define CCOMPLEX_D "_Complex double"
#define CCOMPLEX_F "_Complex float"
#else
#define CCOMPLEX_D "_C_double_complex"
#define CCOMPLEX_F "_C_float_complex"
#endif

static struct InitConvFactories_t {
public:
    InitConvFactories_t() {
    // load all converter factories in the global map 'gConvFactories'
        CPyCppyy::ConvFactories_t& gf = gConvFactories;

    // factories for built-ins
        gf["bool"] =                        (cf_t)+[](cdims_t) { static BoolConverter c{};           return &c; };
        gf["const bool&"] =                 (cf_t)+[](cdims_t) { static ConstBoolRefConverter c{};   return &c; };
        gf["bool&"] =                       (cf_t)+[](cdims_t) { static BoolRefConverter c{};        return &c; };
        gf["char"] =                        (cf_t)+[](cdims_t) { static CharConverter c{};           return &c; };
        gf["const char&"] =                 (cf_t)+[](cdims_t) { static ConstCharRefConverter c{};   return &c; };
        gf["char&"] =                       (cf_t)+[](cdims_t) { static CharRefConverter c{};        return &c; };
        gf["signed char&"] =                (cf_t)+[](cdims_t) { static SCharRefConverter c{};       return &c; };
        gf["unsigned char"] =               (cf_t)+[](cdims_t) { static UCharConverter c{};          return &c; };
        gf["const unsigned char&"] =        (cf_t)+[](cdims_t) { static ConstUCharRefConverter c{};  return &c; };
        gf["unsigned char&"] =              (cf_t)+[](cdims_t) { static UCharRefConverter c{};       return &c; };
        gf["SCharAsInt"] =                  (cf_t)+[](cdims_t) { static SCharAsIntConverter c{};     return &c; };
        gf["UCharAsInt"] =                  (cf_t)+[](cdims_t) { static UCharAsIntConverter c{};     return &c; };
        gf["wchar_t"] =                     (cf_t)+[](cdims_t) { static WCharConverter c{};          return &c; };
        gf["char16_t"] =                    (cf_t)+[](cdims_t) { static Char16Converter c{};         return &c; };
        gf["char32_t"] =                    (cf_t)+[](cdims_t) { static Char32Converter c{};         return &c; };
        gf["wchar_t&"] =                    (cf_t)+[](cdims_t) { static WCharRefConverter c{};       return &c; };
        gf["char16_t&"] =                   (cf_t)+[](cdims_t) { static Char16RefConverter c{};      return &c; };
        gf["char32_t&"] =                   (cf_t)+[](cdims_t) { static Char32RefConverter c{};      return &c; };
        gf["int8_t"] =                      (cf_t)+[](cdims_t) { static Int8Converter c{};           return &c; };
        gf["const int8_t&"] =               (cf_t)+[](cdims_t) { static ConstInt8RefConverter c{};   return &c; };
        gf["int8_t&"] =                     (cf_t)+[](cdims_t) { static Int8RefConverter c{};        return &c; };
        gf["int16_t"] =                     (cf_t)+[](cdims_t) { static Int16Converter c{};          return &c; };
        gf["const int16_t&"] =              (cf_t)+[](cdims_t) { static ConstInt16RefConverter c{};  return &c; };
        gf["int16_t&"] =                    (cf_t)+[](cdims_t) { static Int16RefConverter c{};       return &c; };
        gf["int32_t"] =                     (cf_t)+[](cdims_t) { static Int32Converter c{};          return &c; };
        gf["const int32_t&"] =              (cf_t)+[](cdims_t) { static ConstInt32RefConverter c{};  return &c; };
        gf["int32_t&"] =                    (cf_t)+[](cdims_t) { static Int32RefConverter c{};       return &c; };
        gf["uint8_t"] =                     (cf_t)+[](cdims_t) { static UInt8Converter c{};          return &c; };
        gf["const uint8_t&"] =              (cf_t)+[](cdims_t) { static ConstUInt8RefConverter c{};  return &c; };
        gf["uint8_t&"] =                    (cf_t)+[](cdims_t) { static UInt8RefConverter c{};       return &c; };
        gf["uint16_t"] =                    (cf_t)+[](cdims_t) { static UInt16Converter c{};         return &c; };
        gf["const uint16_t&"] =             (cf_t)+[](cdims_t) { static ConstUInt16RefConverter c{}; return &c; };
        gf["uint16_t&"] =                   (cf_t)+[](cdims_t) { static UInt16RefConverter c{};      return &c; };
        gf["uint32_t"] =                    (cf_t)+[](cdims_t) { static UInt32Converter c{};         return &c; };
        gf["const uint32_t&"] =             (cf_t)+[](cdims_t) { static ConstUInt32RefConverter c{}; return &c; };
        gf["uint32_t&"] =                   (cf_t)+[](cdims_t) { static UInt32RefConverter c{};      return &c; };
        gf["short"] =                       (cf_t)+[](cdims_t) { static ShortConverter c{};          return &c; };
        gf["const short&"] =                (cf_t)+[](cdims_t) { static ConstShortRefConverter c{};  return &c; };
        gf["short&"] =                      (cf_t)+[](cdims_t) { static ShortRefConverter c{};       return &c; };
        gf["unsigned short"] =              (cf_t)+[](cdims_t) { static UShortConverter c{};         return &c; };
        gf["const unsigned short&"] =       (cf_t)+[](cdims_t) { static ConstUShortRefConverter c{}; return &c; };
        gf["unsigned short&"] =             (cf_t)+[](cdims_t) { static UShortRefConverter c{};      return &c; };
        gf["int"] =                         (cf_t)+[](cdims_t) { static IntConverter c{};            return &c; };
        gf["int&"] =                        (cf_t)+[](cdims_t) { static IntRefConverter c{};         return &c; };
        gf["const int&"] =                  (cf_t)+[](cdims_t) { static ConstIntRefConverter c{};    return &c; };
        gf["unsigned int"] =                (cf_t)+[](cdims_t) { static UIntConverter c{};           return &c; };
        gf["const unsigned int&"] =         (cf_t)+[](cdims_t) { static ConstUIntRefConverter c{};   return &c; };
        gf["unsigned int&"] =               (cf_t)+[](cdims_t) { static UIntRefConverter c{};        return &c; };
        gf["long"] =                        (cf_t)+[](cdims_t) { static LongConverter c{};           return &c; };
        gf["long&"] =                       (cf_t)+[](cdims_t) { static LongRefConverter c{};        return &c; };
        gf["const long&"] =                 (cf_t)+[](cdims_t) { static ConstLongRefConverter c{};   return &c; };
        gf["unsigned long"] =               (cf_t)+[](cdims_t) { static ULongConverter c{};          return &c; };
        gf["const unsigned long&"] =        (cf_t)+[](cdims_t) { static ConstULongRefConverter c{};  return &c; };
        gf["unsigned long&"] =              (cf_t)+[](cdims_t) { static ULongRefConverter c{};       return &c; };
        gf["long long"] =                   (cf_t)+[](cdims_t) { static LLongConverter c{};          return &c; };
        gf["const long long&"] =            (cf_t)+[](cdims_t) { static ConstLLongRefConverter c{};  return &c; };
        gf["long long&"] =                  (cf_t)+[](cdims_t) { static LLongRefConverter c{};       return &c; };
        gf["unsigned long long"] =          (cf_t)+[](cdims_t) { static ULLongConverter c{};         return &c; };
        gf["const unsigned long long&"] =   (cf_t)+[](cdims_t) { static ConstULLongRefConverter c{}; return &c; };
        gf["unsigned long long&"] =         (cf_t)+[](cdims_t) { static ULLongRefConverter c{};      return &c; };

        gf["float"] =                       (cf_t)+[](cdims_t) { static FloatConverter c{};           return &c; };
        gf["const float&"] =                (cf_t)+[](cdims_t) { static ConstFloatRefConverter c{};   return &c; };
        gf["float&"] =                      (cf_t)+[](cdims_t) { static FloatRefConverter c{};        return &c; };
        gf["double"] =                      (cf_t)+[](cdims_t) { static DoubleConverter c{};          return &c; };
        gf["double&"] =                     (cf_t)+[](cdims_t) { static DoubleRefConverter c{};       return &c; };
        gf["const double&"] =               (cf_t)+[](cdims_t) { static ConstDoubleRefConverter c{};  return &c; };
        gf["long double"] =                 (cf_t)+[](cdims_t) { static LDoubleConverter c{};         return &c; };
        gf["const long double&"] =          (cf_t)+[](cdims_t) { static ConstLDoubleRefConverter c{}; return &c; };
        gf["long double&"] =                (cf_t)+[](cdims_t) { static LDoubleRefConverter c{};      return &c; };
        gf["std::complex<double>"] =        (cf_t)+[](cdims_t) { return new ComplexDConverter{}; };
        gf["const std::complex<double>&"] = (cf_t)+[](cdims_t) { return new ComplexDConverter{}; };
        gf["void"] =                        (cf_t)+[](cdims_t) { static VoidConverter c{};            return &c; };

    // pointer/array factories
        gf["bool ptr"] =                    (cf_t)+[](cdims_t d) { return new BoolArrayConverter{d}; };
        gf["signed char ptr"] =             (cf_t)+[](cdims_t d) { return new SCharArrayConverter{d}; };
        gf["signed char**"] =               (cf_t)+[](cdims_t)   { return new SCharArrayConverter{{UNKNOWN_SIZE, UNKNOWN_SIZE}}; };
        gf["const unsigned char*"] =        (cf_t)+[](cdims_t d) { return new UCharArrayConverter{d}; };
        gf["unsigned char ptr"] =           (cf_t)+[](cdims_t d) { return new UCharArrayConverter{d}; };
        gf["SCharAsInt*"] =                 gf["signed char ptr"];
        gf["SCharAsInt[]"] =                gf["signed char ptr"];
        gf["UCharAsInt*"] =                 gf["unsigned char ptr"];
        gf["UCharAsInt[]"] =                gf["unsigned char ptr"];
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
        gf["std::byte ptr"] =               (cf_t)+[](cdims_t d) { return new ByteArrayConverter{d}; };
#endif
        gf["int8_t ptr"] =                  (cf_t)+[](cdims_t d) { return new Int8ArrayConverter{d}; };
        gf["int16_t ptr"] =                 (cf_t)+[](cdims_t d) { return new Int16ArrayConverter{d}; };
        gf["int32_t ptr"] =                 (cf_t)+[](cdims_t d) { return new Int32ArrayConverter{d}; };
        gf["uint8_t ptr"] =                 (cf_t)+[](cdims_t d) { return new UInt8ArrayConverter{d}; };
        gf["uint16_t ptr"] =                (cf_t)+[](cdims_t d) { return new UInt16ArrayConverter{d}; };
        gf["uint32_t ptr"] =                (cf_t)+[](cdims_t d) { return new UInt32ArrayConverter{d}; };
        gf["short ptr"] =                   (cf_t)+[](cdims_t d) { return new ShortArrayConverter{d}; };
        gf["unsigned short ptr"] =          (cf_t)+[](cdims_t d) { return new UShortArrayConverter{d}; };
        gf["int ptr"] =                     (cf_t)+[](cdims_t d) { return new IntArrayConverter{d}; };
        gf["unsigned int ptr"] =            (cf_t)+[](cdims_t d) { return new UIntArrayConverter{d}; };
        gf["long ptr"] =                    (cf_t)+[](cdims_t d) { return new LongArrayConverter{d}; };
        gf["unsigned long ptr"] =           (cf_t)+[](cdims_t d) { return new ULongArrayConverter{d}; };
        gf["long long ptr"] =               (cf_t)+[](cdims_t d) { return new LLongArrayConverter{d}; };
        gf["unsigned long long ptr"] =      (cf_t)+[](cdims_t d) { return new ULLongArrayConverter{d}; };
        gf["float ptr"] =                   (cf_t)+[](cdims_t d) { return new FloatArrayConverter{d}; };
        gf["double ptr"] =                  (cf_t)+[](cdims_t d) { return new DoubleArrayConverter{d}; };
        gf["long double ptr"] =             (cf_t)+[](cdims_t d) { return new LDoubleArrayConverter{d}; };
        gf["std::complex<float> ptr"] =     (cf_t)+[](cdims_t d) { return new ComplexFArrayConverter{d}; };
        gf["std::complex<double> ptr"] =    (cf_t)+[](cdims_t d) { return new ComplexDArrayConverter{d}; };
        gf["void*"] =                       (cf_t)+[](cdims_t d) { return new VoidArrayConverter{(bool)d}; };

    // aliases
        gf["signed char"] =                 gf["char"];
        gf["const signed char&"] =          gf["const char&"];
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
        gf["std::byte"] =                   gf["uint8_t"];
        gf["byte"] =                        gf["uint8_t"];
        gf["const std::byte&"] =            gf["const uint8_t&"];
        gf["const byte&"] =                 gf["const uint8_t&"];
        gf["std::byte&"] =                  gf["uint8_t&"];
        gf["byte&"] =                       gf["uint8_t&"];
#endif
        gf["std::int8_t"] =                 gf["int8_t"];
        gf["const std::int8_t&"] =          gf["const int8_t&"];
        gf["std::int8_t&"] =                gf["int8_t&"];
        gf["std::uint8_t"] =                gf["uint8_t"];
        gf["const std::uint8_t&"] =         gf["const uint8_t&"];
        gf["std::uint8_t&"] =               gf["uint8_t&"];
        gf["internal_enum_type_t"] =        gf["int"];
        gf["internal_enum_type_t&"] =       gf["int&"];
        gf["const internal_enum_type_t&"] = gf["const int&"];
        gf["internal_enum_type_t ptr"] =    gf["int ptr"];
#ifdef _WIN32
        gf["__int64"] =                     gf["long long"];
        gf["const __int64&"] =              gf["const long long&"];
        gf["__int64&"] =                    gf["long long&"];
        gf["__int64 ptr"] =                 gf["long long ptr"];
        gf["unsigned __int64"] =            gf["unsigned long long"];
        gf["const unsigned __int64&"] =     gf["const unsigned long long&"];
        gf["unsigned __int64&"] =           gf["unsigned long long&"];
        gf["unsigned __int64 ptr"] =        gf["unsigned long long ptr"];
#endif
        gf[CCOMPLEX_D] =                    gf["std::complex<double>"];
        gf["const " CCOMPLEX_D "&"] =       gf["const std::complex<double>&"];
        gf[CCOMPLEX_F " ptr"] =             gf["std::complex<float> ptr"];
        gf[CCOMPLEX_D " ptr"] =             gf["std::complex<double> ptr"];

    // factories for special cases
        gf["nullptr_t"] =                   (cf_t)+[](cdims_t) { static NullptrConverter c{};        return &c;};
        gf["const char*"] =                 (cf_t)+[](cdims_t) { return new CStringConverter{}; };
        gf["const signed char*"] =          gf["const char*"];
        gf["const char*&&"] =               gf["const char*"];
        gf["const char[]"] =                (cf_t)+[](cdims_t) { return new CStringConverter{}; };
        gf["char*"] =                       (cf_t)+[](cdims_t d) { return new NonConstCStringConverter{dims2stringsz(d)}; };
        gf["char[]"] =                      (cf_t)+[](cdims_t d) { return new NonConstCStringArrayConverter{d, true}; };
        gf["signed char*"] =                gf["char*"];
        gf["wchar_t*"] =                    (cf_t)+[](cdims_t) { return new WCStringConverter{}; };
        gf["char16_t*"] =                   (cf_t)+[](cdims_t) { return new CString16Converter{}; };
        gf["char16_t[]"] =                  (cf_t)+[](cdims_t d) { return new CString16Converter{dims2stringsz(d)}; };
        gf["char32_t*"] =                   (cf_t)+[](cdims_t) { return new CString32Converter{}; };
        gf["char32_t[]"] =                  (cf_t)+[](cdims_t d) { return new CString32Converter{dims2stringsz(d)}; };
    // TODO: the following are handled incorrectly upstream (char16_t** where char16_t* intended)?!
        gf["char16_t**"] =                  gf["char16_t*"];
        gf["char32_t**"] =                  gf["char32_t*"];
        gf["const char**"] =                (cf_t)+[](cdims_t) { return new CStringArrayConverter{{UNKNOWN_SIZE, UNKNOWN_SIZE}, false}; };
        gf["char**"] =                      gf["const char**"];
        gf["const char*[]"] =               (cf_t)+[](cdims_t d) { return new CStringArrayConverter{d, false}; };
        gf["char*[]"] =                     (cf_t)+[](cdims_t d) { return new NonConstCStringArrayConverter{d, false}; };
        gf["char ptr"] =                    gf["char*[]"];
        gf["std::string"] =                 (cf_t)+[](cdims_t) { return new STLStringConverter{}; };
        gf["const std::string&"] =          gf["std::string"];
        gf["string"] =                      gf["std::string"];
        gf["const string&"] =               gf["std::string"];
        gf["std::string&&"] =               (cf_t)+[](cdims_t) { return new STLStringMoveConverter{}; };
        gf["string&&"] =                    gf["std::string&&"];
#if (__cplusplus > 201402L) || (defined(_MSC_VER) && _MSVC_LANG > 201402L)
        gf["std::string_view"] =            (cf_t)+[](cdims_t) { return new STLStringViewConverter{}; };
        gf[STRINGVIEW] =                    gf["std::string_view"];
        gf["std::string_view&"] =           gf["std::string_view"];
        gf["const std::string_view&"] =     gf["std::string_view"];
        gf["const " STRINGVIEW "&"] =       gf["std::string_view"];
#endif
        gf["std::wstring"] =                (cf_t)+[](cdims_t) { return new STLWStringConverter{}; };
        gf[WSTRING1] =                      gf["std::wstring"];
        gf[WSTRING2] =                      gf["std::wstring"];
        gf["const std::wstring&"] =         gf["std::wstring"];
        gf["const " WSTRING1 "&"] =         gf["std::wstring"];
        gf["const " WSTRING2 "&"] =         gf["std::wstring"];
        gf["void*&"] =                      (cf_t)+[](cdims_t) { static VoidPtrRefConverter c{};     return &c; };
        gf["void**"] =                      (cf_t)+[](cdims_t d) { return new VoidPtrPtrConverter{d}; };
        gf["void ptr"] =                    gf["void**"];
        gf["PyObject*"] =                   (cf_t)+[](cdims_t) { static PyObjectConverter c{};       return &c; };
        gf["_object*"] =                    gf["PyObject*"];
        gf["FILE*"] =                       (cf_t)+[](cdims_t) { return new VoidArrayConverter{}; };
    }
} initConvFactories_;

} // unnamed namespace
