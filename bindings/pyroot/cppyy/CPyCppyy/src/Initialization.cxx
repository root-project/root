// Bindings
#include "CPyCppyy.h"
#include "CallContext.h"
#include "Converters.h"
#include "CPPDataMember.h"
#include "CPPExcInstance.h"
#include "CPPInstance.h"
#include "CPPOverload.h"
#include "CPPScope.h"
#include "CustomPyTypes.h"
#include "LowLevelViews.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TemplateProxy.h"
#include "TupleOfInstances.h"
#include "Utility.h"

#define CPYCPPYY_INTERNAL 1
#include "CPyCppyy/DispatchPtr.h"
namespace CPyCppyy {
void* Instance_AsVoidPtr(PyObject* pyobject);
PyObject* Instance_FromVoidPtr(
    void* addr, const std::string& classname, bool python_owns = false);
} // namespace CPyCppyy
#undef CPYCPPYY_INTERNAL

// Standard
#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>


// Note: as of py3.11, dictionary objects no longer carry a function pointer for
// the lookup, so it can no longer be shimmed and "from cppyy.interactive import *"
// thus no longer works.
#if PY_VERSION_HEX < 0x030b0000

//- from Python's dictobject.c -------------------------------------------------
#if PY_VERSION_HEX >= 0x03030000
    typedef struct PyDictKeyEntry {
    /* Cached hash code of me_key. */
        Py_hash_t me_hash;
        PyObject *me_key;
        PyObject *me_value; /* This field is only meaningful for combined tables */
    } PyDictEntry;

    typedef struct _dictkeysobject {
        Py_ssize_t dk_refcnt;
        Py_ssize_t dk_size;
        dict_lookup_func dk_lookup;
        Py_ssize_t dk_usable;
#if PY_VERSION_HEX >= 0x03060000
        Py_ssize_t dk_nentries;
        union {
            int8_t as_1[8];
            int16_t as_2[4];
            int32_t as_4[2];
#if SIZEOF_VOID_P > 4
            int64_t as_8[1];
#endif
        } dk_indices;
#else
        PyDictKeyEntry dk_entries[1];
#endif
    } PyDictKeysObject;

#define CPYCPPYY_GET_DICT_LOOKUP(mp)                                          \
    ((dict_lookup_func&)mp->ma_keys->dk_lookup)

#else

#define CPYCPPYY_GET_DICT_LOOKUP(mp)                                          \
    ((dict_lookup_func&)mp->ma_lookup)

#endif

#endif // PY_VERSION_HEX < 0x030b0000

//- data -----------------------------------------------------------------------
static PyObject* nullptr_repr(PyObject*)
{
    return CPyCppyy_PyText_FromString("nullptr");
}

static void nullptr_dealloc(PyObject*)
{
    Py_FatalError("deallocating nullptr");
}

static int nullptr_nonzero(PyObject*)
{
    return 0;
}

static PyNumberMethods nullptr_as_number = {
    0, 0, 0,
#if PY_VERSION_HEX < 0x03000000
    0,
#endif
    0, 0, 0, 0, 0, 0,
    (inquiry)nullptr_nonzero,           // tp_nonzero (nb_bool in p3)
    0, 0, 0, 0, 0, 0,
#if PY_VERSION_HEX < 0x03000000
    0,                                  // nb_coerce
#endif
    0, 0, 0,
#if PY_VERSION_HEX < 0x03000000
    0, 0,
#endif
    0, 0, 0,
#if PY_VERSION_HEX < 0x03000000
    0,                                  // nb_inplace_divide
#endif
    0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02020000
    , 0                                 // nb_floor_divide
#if PY_VERSION_HEX < 0x03000000
    , 0                                 // nb_true_divide
#else
    , 0                                 // nb_true_divide
#endif
    , 0, 0
#endif
#if PY_VERSION_HEX >= 0x02050000
    , 0                                 // nb_index
#endif
#if PY_VERSION_HEX >= 0x03050000
    , 0                                 // nb_matrix_multiply
    , 0                                 // nb_inplace_matrix_multiply
#endif
};

static PyTypeObject PyNullPtr_t_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "nullptr_t",         // tp_name
    sizeof(PyObject),    // tp_basicsize
    0,                   // tp_itemsize
    nullptr_dealloc,     // tp_dealloc (never called)
    0, 0, 0, 0,
    nullptr_repr,        // tp_repr
    &nullptr_as_number,  // tp_as_number
    0, 0,
#if PY_VERSION_HEX >= 0x030d0000
    (hashfunc)Py_HashPointer, // tp_hash
#else
    (hashfunc)_Py_HashPointer, // tp_hash
#endif
    0, 0, 0, 0, 0, Py_TPFLAGS_DEFAULT, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                  // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                  // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                  // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                  // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                  // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                  // tp_versions_used
#endif
};


static PyObject* default_repr(PyObject*)
{
    return CPyCppyy_PyText_FromString("type default");
}

static void default_dealloc(PyObject*)
{
    Py_FatalError("deallocating default");
}

static PyTypeObject PyDefault_t_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    "default_t",         // tp_name
    sizeof(PyObject),    // tp_basicsize
    0,                   // tp_itemsize
    default_dealloc,     // tp_dealloc (never called)
    0, 0, 0, 0,
    default_repr,        // tp_repr
    0, 0, 0,
#if PY_VERSION_HEX >= 0x030d0000
    (hashfunc)Py_HashPointer, // tp_hash
#else
    (hashfunc)_Py_HashPointer, // tp_hash
#endif
    0, 0, 0, 0, 0, Py_TPFLAGS_DEFAULT, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#if PY_VERSION_HEX >= 0x02030000
    , 0                  // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                  // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                  // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                 // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                 // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                 // tp_versions_used
#endif
};

namespace {

struct {
   PyObject_HEAD
} _CPyCppyy_NullPtrStruct{PyObject_HEAD_INIT(&PyNullPtr_t_Type)};

struct {
   PyObject_HEAD
} _CPyCppyy_DefaultStruct{PyObject_HEAD_INIT(&PyDefault_t_Type)};

// TODO: refactor with Converters.cxx
struct CPyCppyy_tagCDataObject {       // non-public (but stable)
    PyObject_HEAD
    char* b_ptr;
    int   b_needsfree;
};

} // unnamed namespace

namespace CPyCppyy {
    PyObject* gThisModule    = nullptr;
    PyObject* gPyTypeMap     = nullptr;
    PyObject* gNullPtrObject = nullptr;
    PyObject* gDefaultObject = nullptr;
    PyObject* gBusException  = nullptr;
    PyObject* gSegvException = nullptr;
    PyObject* gIllException  = nullptr;
    PyObject* gAbrtException = nullptr;
    std::set<Cppyy::TCppType_t> gPinnedTypes;
    std::ostringstream gCapturedError;
    std::streambuf* gOldErrorBuffer = nullptr;

    std::map<std::string, std::vector<PyObject*>> &pythonizations()
    {
       static std::map<std::string, std::vector<PyObject*>> pyzMap;
       return pyzMap;
    }
}


//- private helpers ------------------------------------------------------------
namespace {

using namespace CPyCppyy;


//----------------------------------------------------------------------------
#if PY_VERSION_HEX < 0x030b0000
namespace {

class GblGetter {
public:
    GblGetter() {
        PyObject* cppyy = PyImport_AddModule((char*)"cppyy");
        fGbl = PyObject_GetAttrString(cppyy, (char*)"gbl");
    }
    ~GblGetter() { Py_DECREF(fGbl); }

    PyObject* operator*() { return fGbl; }

private:
    PyObject* fGbl;
};

} // unnamed namespace

#if PY_VERSION_HEX >= 0x03060000
inline Py_ssize_t OrgDictLookup(PyDictObject* mp, PyObject* key,
    Py_hash_t hash, PyObject*** value_addr, Py_ssize_t* hashpos)
{
    return (*gDictLookupOrg)(mp, key, hash, value_addr, hashpos);
}
#define CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos)           \
    OrgDictLookup(mp, key, hash, value_addr, hashpos)

Py_ssize_t CPyCppyyLookDictString(PyDictObject* mp, PyObject* key,
    Py_hash_t hash, PyObject*** value_addr, Py_ssize_t* hashpos)

#elif PY_VERSION_HEX >= 0x03030000
inline PyDictKeyEntry* OrgDictLookup(
    PyDictObject* mp, PyObject* key, Py_hash_t hash, PyObject*** value_addr)
{
    return (*gDictLookupOrg)(mp, key, hash, value_addr);
}

#define CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos)           \
    OrgDictLookup(mp, key, hash, value_addr)

PyDictKeyEntry* CPyCppyyLookDictString(
    PyDictObject* mp, PyObject* key, Py_hash_t hash, PyObject*** value_addr)

#else /* < 3.3 */

inline PyDictEntry* OrgDictLookup(PyDictObject* mp, PyObject* key, long hash)
{
    return (*gDictLookupOrg)(mp, key, hash);
}

#define CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos)           \
    OrgDictLookup(mp, key, hash)

PyDictEntry* CPyCppyyLookDictString(PyDictObject* mp, PyObject* key, long hash)
#endif
{
    static GblGetter gbl;
#if PY_VERSION_HEX >= 0x03060000
    Py_ssize_t ep;
#else
    PyDictEntry* ep;
#endif

// first search dictionary itself
    ep = CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos);
    if (gDictLookupActive)
        return ep;

#if PY_VERSION_HEX >= 0x03060000
    if (ep >= 0)
#else
    if (!ep || (ep->me_key && ep->me_value))
#endif
        return ep;

// filter for builtins
    if (PyDict_GetItem(PyEval_GetBuiltins(), key) != 0)
        return ep;

// normal lookup failed, attempt to get C++ enum/global/class from top-level
    gDictLookupActive = true;

// attempt to get C++ enum/global/class from top-level
    PyObject* val = PyObject_GetAttr(*gbl, key);

    if (val) {
    // success ...

        if (CPPDataMember_CheckExact(val)) {
        // don't want to add to dictionary (the proper place would be the
        // dictionary of the (meta)class), but modifying ep will be noticed no
        // matter what; just return the actual value and live with the copy in
        // the dictionary (mostly, this is correct)
            PyObject* actual_val = Py_TYPE(val)->tp_descr_get(val, nullptr, nullptr);
            Py_DECREF(val);
            val = actual_val;
        }

    // add reference to C++ entity in the given dictionary
        CPYCPPYY_GET_DICT_LOOKUP(mp) = gDictLookupOrg;      // prevent recursion
        if (PyDict_SetItem((PyObject*)mp, key, val) == 0) {
            ep = CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos);
        } else {
#if PY_VERSION_HEX >= 0x03060000
            ep = -1;
#else
            ep->me_key   = nullptr;
            ep->me_value = nullptr;
#endif
        }
        CPYCPPYY_GET_DICT_LOOKUP(mp) = CPyCppyyLookDictString;   // restore

    // done with val
        Py_DECREF(val);
    } else
        PyErr_Clear();

#if PY_VERSION_HEX >= 0x03030000
    if (mp->ma_keys->dk_usable <= 0) {
    // big risk that this lookup will result in a resize, so force it here
    // to be able to reset the lookup function; of course, this is nowhere
    // near fool-proof, but should cover interactive usage ...
        CPYCPPYY_GET_DICT_LOOKUP(mp) = gDictLookupOrg;
        const int maxinsert = 5;
        PyObject* buf[maxinsert];
        for (int varmax = 1; varmax <= maxinsert; ++varmax) {
            for (int ivar = 0; ivar < varmax; ++ivar) {
                buf[ivar] = CPyCppyy_PyText_FromFormat("__CPYCPPYY_FORCE_RESIZE_%d", ivar);
                PyDict_SetItem((PyObject*)mp, buf[ivar], Py_None);
            }
            for (int ivar = 0; ivar < varmax; ++ivar) {
                PyDict_DelItem((PyObject*)mp, buf[ivar]);
                Py_DECREF(buf[ivar]);
            }
            if (0 < mp->ma_keys->dk_usable)
                break;
        }

    // make sure the entry pointer is still valid by re-doing the lookup
        ep = CPYCPPYY_ORGDICT_LOOKUP(mp, key, hash, value_addr, hashpos);

    // full reset of all lookup functions
        gDictLookupOrg = CPYCPPYY_GET_DICT_LOOKUP(mp);
        CPYCPPYY_GET_DICT_LOOKUP(mp) = CPyCppyyLookDictString;   // restore
    }
#endif

// stopped calling into the reflection system
    gDictLookupActive = false;
    return ep;
}

#endif // PY_VERSION_HEX < 0x030b0000

//----------------------------------------------------------------------------
static PyObject* SetCppLazyLookup(PyObject*, PyObject* args)
{
#if PY_VERSION_HEX < 0x030b0000
// Modify the given dictionary to install the lookup function that also
// tries the global C++ namespace before failing. Called on a module's dictionary,
// this allows for lazy lookups. This works fine for p3.2 and earlier, but should
// not be used beyond interactive code for p3.3 and later b/c resizing causes the
// lookup function to revert to the default (lookdict_unicode_nodummy).
    PyDictObject* dict = nullptr;
    if (!PyArg_ParseTuple(args, const_cast<char*>("O!"), &PyDict_Type, &dict))
        return nullptr;

    CPYCPPYY_GET_DICT_LOOKUP(dict) = CPyCppyyLookDictString;
#else
// As of py3.11, there is no longer a lookup function pointer in the dict object
// to replace. Since this feature is not widely advertised, it's simply dropped
    PyErr_Warn(PyExc_RuntimeWarning, (char*)"lazy lookup is no longer supported");
    (void)args; // avoid warning about unused parameter
#endif

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* MakeCppTemplateClass(PyObject*, PyObject* args)
{
// Create a binding for a templated class instantiation.

// args is class name + template arguments; build full instantiation
    Py_ssize_t nArgs = PyTuple_GET_SIZE(args);
    if (nArgs < 2) {
        PyErr_Format(PyExc_TypeError, "too few arguments for template instantiation");
        return nullptr;
    }

// build "< type, type, ... >" part of class name (modifies pyname)
    const std::string& tmpl_name =
        Utility::ConstructTemplateArgs(PyTuple_GET_ITEM(args, 0), args, nullptr, Utility::kNone, 1);
    if (!tmpl_name.size())
        return nullptr;

    return CreateScopeProxy(tmpl_name);
}

//----------------------------------------------------------------------------
static char* GCIA_kwlist[] = {(char*)"instance", (char*)"field", (char*)"byref", NULL};
static void* GetCPPInstanceAddress(const char* fname, PyObject* args, PyObject* kwds)
{
// Helper to get the address (address-of-address) of various object proxy types.
    CPPInstance* pyobj = 0; PyObject* pyname = 0; int byref = 0;
    if (PyArg_ParseTupleAndKeywords(args, kwds, const_cast<char*>("O|O!b"), GCIA_kwlist,
            &pyobj, &CPyCppyy_PyText_Type, &pyname, &byref)) {

        if (CPPInstance_Check(pyobj)) {
            if (pyname != 0) {
            // locate property proxy for offset info
                CPPDataMember* pyprop = nullptr;

                PyObject* pyclass = (PyObject*)Py_TYPE((PyObject*)pyobj);
                PyObject* dict = PyObject_GetAttr(pyclass, PyStrings::gDict);
                pyprop = (CPPDataMember*)PyObject_GetItem(dict, pyname);
                Py_DECREF(dict);

                if (CPPDataMember_Check(pyprop)) {
                // this is an address of a value (i.e. &myobj->prop)
                    void* addr = (void*)pyprop->GetAddress(pyobj);
                    Py_DECREF(pyprop);
                    return addr;
                }

                Py_XDECREF(pyprop);

                PyErr_Format(PyExc_TypeError,
                    "%s is not a valid data member", CPyCppyy_PyText_AsString(pyname));
                return nullptr;
            }

        // this is an address of an address (i.e. &myobj, with myobj of type MyObj*)
        // note that the return result may be null
            if (!byref) return ((CPPInstance*)pyobj)->GetObject();
            return &((CPPInstance*)pyobj)->GetObjectRaw();

        } else if (CPyCppyy_PyText_Check(pyobj)) {
        // special cases for access to the CPyCppyy API
            std::string req = CPyCppyy_PyText_AsString((PyObject*)pyobj);
            if (req == "Instance_AsVoidPtr")
                return (void*)&Instance_AsVoidPtr;
            else if (req == "Instance_FromVoidPtr")
                return (void*)&Instance_FromVoidPtr;
        }
    }

    if (!PyErr_Occurred())
        PyErr_Format(PyExc_ValueError, "invalid argument for %s", fname);
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* addressof(PyObject* /* dummy */, PyObject* args, PyObject* kwds)
{
// Return object proxy address as a value (cppyy-style), or the same for an array.
    void* addr = GetCPPInstanceAddress("addressof", args, kwds);
    if (addr)
        return PyLong_FromLongLong((intptr_t)addr);
    else if (!PyErr_Occurred()) {
        return PyLong_FromLong(0);
    } else if (PyTuple_CheckExact(args) && PyTuple_GET_SIZE(args) == 1) {
        PyErr_Clear();
        PyObject* arg0 = PyTuple_GET_ITEM(args, 0);

    // nullptr special case
        if (arg0 == gNullPtrObject || (PyInt_Check(arg0) && PyInt_AsLong(arg0) == 0))
            return PyLong_FromLong(0);

    // overload if unambiguous
        if (CPPOverload_CheckExact(arg0)) {
            const auto& methods = ((CPPOverload*)arg0)->fMethodInfo->fMethods;
            if (methods.size() != 1) {
                PyErr_SetString(PyExc_TypeError, "overload is not unambiguous");
                return nullptr;
            }

            Cppyy::TCppFuncAddr_t caddr = methods[0]->GetFunctionAddress();
            return PyLong_FromLongLong((intptr_t)caddr);
        }

    // C functions (incl. ourselves)
        if (PyCFunction_Check(arg0)) {
            void* caddr = (void*)PyCFunction_GetFunction(arg0);
            return PyLong_FromLongLong((intptr_t)caddr);
        }

    // final attempt: any type of buffer
        Utility::GetBuffer(arg0, '*', 1, addr, false);
        if (addr) return PyLong_FromLongLong((intptr_t)addr);
    }

// error message if not already set
    if (!PyErr_Occurred()) {
        if (PyTuple_CheckExact(args) && PyTuple_GET_SIZE(args)) {
            PyObject* str = PyObject_Str(PyTuple_GET_ITEM(args, 0));
            if (str && CPyCppyy_PyText_Check(str))
                PyErr_Format(PyExc_TypeError, "unknown object %s", CPyCppyy_PyText_AsString(str));
            else
                PyErr_Format(PyExc_TypeError, "unknown object at %p", (void*)PyTuple_GET_ITEM(args, 0));
            Py_XDECREF(str);
        }
    }
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* AsCObject(PyObject* /* unused */, PyObject* args, PyObject* kwds)
{
// Return object proxy as an opaque CObject.
    void* addr = GetCPPInstanceAddress("as_cobject", args, kwds);
    if (addr)
        return CPyCppyy_PyCapsule_New((void*)addr, nullptr, nullptr);
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* AsCapsule(PyObject* /* unused */, PyObject* args, PyObject* kwds)
{
// Return object proxy as an opaque PyCapsule.
    void* addr = GetCPPInstanceAddress("as_capsule", args, kwds);
    if (addr)
#if PY_VERSION_HEX < 0x02060000
        return PyCObject_FromVoidPtr(addr, nullptr);
#else
        return PyCapsule_New(addr, nullptr, nullptr);
#endif
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* AsCTypes(PyObject* /* unused */, PyObject* args, PyObject* kwds)
{
// Return object proxy as a ctypes c_void_p
    void* addr = GetCPPInstanceAddress("as_ctypes", args, kwds);
    if (!addr)
        return nullptr;

// TODO: refactor code below with converters code
    static PyTypeObject* ct_cvoidp = nullptr;
    if (!ct_cvoidp) {
        PyObject* ctmod = PyImport_ImportModule("ctypes");   // ref-count kept
        if (!ctmod) return nullptr;

        ct_cvoidp = (PyTypeObject*)PyObject_GetAttrString(ctmod, "c_void_p");
        Py_DECREF(ctmod);
        if (!ct_cvoidp) return nullptr;
        Py_DECREF(ct_cvoidp);     // module keeps a reference
    }

    PyObject* ref = ct_cvoidp->tp_new(ct_cvoidp, nullptr, nullptr);
    *(void**)((CPyCppyy_tagCDataObject*)ref)->b_ptr = addr;
    ((CPyCppyy_tagCDataObject*)ref)->b_needsfree = 0;
    return ref;
}

//----------------------------------------------------------------------------
static PyObject* AsMemoryView(PyObject* /* unused */, PyObject* pyobject)
{
// Return a raw memory view on arrays of PODs.
    if (!CPPInstance_Check(pyobject)) {
        PyErr_SetString(PyExc_TypeError, "C++ object proxy expected");
        return nullptr;
    }

    CPPInstance* pyobj = (CPPInstance*)pyobject;
    Cppyy::TCppType_t klass = ((CPPClass*)Py_TYPE(pyobject))->fCppType;

    Py_ssize_t array_len = pyobj->ArrayLength();

    if (array_len < 0 || !Cppyy::IsAggregate(klass)) {
        PyErr_SetString(
            PyExc_TypeError, "object is not a proxy to an array of PODs of known size");
        return nullptr;
    }

    Py_buffer view;

    view.obj            = pyobject;
    view.buf            = pyobj->GetObject();
    view.itemsize       = Cppyy::SizeOf(klass);
    view.len            = view.itemsize * array_len;
    view.readonly       = 0;
    view.format         = NULL;   // i.e. "B" assumed
    view.ndim           = 1;
    view.shape          = NULL;
    view.strides        = NULL;
    view.suboffsets     = NULL;
    view.internal       = NULL;

    return PyMemoryView_FromBuffer(&view);
}

//----------------------------------------------------------------------------
static PyObject* BindObject(PyObject*, PyObject* args, PyObject* kwds)
{
// From a long representing an address or a PyCapsule/CObject, bind to a class.
    Py_ssize_t argc = PyTuple_GET_SIZE(args);
    if (argc != 2) {
        PyErr_Format(PyExc_TypeError,
            "bind_object takes 2 positional arguments but (" PY_SSIZE_T_FORMAT " were given)", argc);
        return nullptr;
    }

// convert 2nd argument first (used for both pointer value and instance cases)
    Cppyy::TCppType_t cast_type = 0;
    PyObject* arg1 = PyTuple_GET_ITEM(args, 1);
    if (!CPyCppyy_PyText_Check(arg1)) {          // not string, then class
        if (CPPScope_Check(arg1))
            cast_type = ((CPPClass*)arg1)->fCppType;
        else
            arg1 = PyObject_GetAttr(arg1, PyStrings::gName);
    } else
        Py_INCREF(arg1);

    if (!cast_type && arg1) {
        cast_type = (Cppyy::TCppType_t)Cppyy::GetScope(CPyCppyy_PyText_AsString(arg1));
        Py_DECREF(arg1);
    }

    if (!cast_type) {
        PyErr_SetString(PyExc_TypeError,
            "bind_object expects a valid class or class name as an argument");
        return nullptr;
    }

// next, convert the first argument, some pointer value or a pre-existing instance
    PyObject* arg0 = PyTuple_GET_ITEM(args, 0);

    if (CPPInstance_Check(arg0)) {
    // if this instance's class has a relation to the requested one, calculate the
    // offset, erase if from any caches, and update the pointer and type
        CPPInstance* arg0_pyobj = (CPPInstance*)arg0;
        Cppyy::TCppType_t cur_type = arg0_pyobj->ObjectIsA(false /* check_smart */);

        bool isPython = CPPScope_Check(arg1) && \
            (((CPPClass*)arg1)->fFlags & CPPScope::kIsPython);

        if (cur_type == cast_type && !isPython) {
            Py_INCREF(arg0);      // nothing to do
            return arg0;
        }

        int direction = 0;
        Cppyy::TCppType_t base = 0, derived = 0;
        if (Cppyy::IsSubtype(cast_type, cur_type)) {
            derived = cast_type;
            base    = cur_type;
            direction = -1;      // down-cast
        } else if (Cppyy::IsSubtype(cur_type, cast_type)) {
            base    = cast_type;
            derived = cur_type;
            direction =  1;      // up-cast
        } else {
            PyErr_SetString(PyExc_TypeError,
                "provided instance and provided target type are unrelated");
            return nullptr;
        }

        Cppyy::TCppObject_t address = (Cppyy::TCppObject_t)arg0_pyobj->GetObject();
        ptrdiff_t offset = Cppyy::GetBaseOffset(derived, base, address, direction);

    // it's debatable whether a new proxy should be created rather than updating
    // the old, but changing the old object would be changing the behavior of all
    // code that has a reference to it, which may not be the intention if the cast
    // is on a C++ data member; this probably is the "least surprise" option

    // ownership is taken over as needed, again following the principle of "least
    // surprise" as most likely only the cast object will be retained
        bool owns = arg0_pyobj->fFlags & CPPInstance::kIsOwner;

        if (!isPython) {
        // ordinary C++ class
            PyObject* pyobj = BindCppObjectNoCast(
                (void*)((intptr_t)address + offset), cast_type, owns ? CPPInstance::kIsOwner : 0);
            if (owns && pyobj) arg0_pyobj->CppOwns();
            return pyobj;

        } else {
        // rebinding to a Python-side class, create a fresh instance first to be able to
        // perform a lookup of the original dispatch object and if found, return original
            void* cast_address = (void*)((intptr_t)address + offset);
            PyObject* pyobj = ((PyTypeObject*)arg1)->tp_new((PyTypeObject*)arg1, nullptr, nullptr);
            ((CPPInstance*)pyobj)->GetObjectRaw() = cast_address;

            PyObject* dispproxy = CPyCppyy::GetScopeProxy(cast_type);
            PyObject* res = PyObject_CallMethodOneArg(dispproxy, PyStrings::gDispGet, pyobj);
            /* Note: the resultant object is borrowed */
            if (CPPInstance_Check(res) && ((CPPInstance*)res)->GetObject() == cast_address) {
                ((CPPInstance*)pyobj)->CppOwns();     // make sure C++ object isn't deleted
                Py_DECREF(pyobj);                     //  on DECREF (is default, but still)
                pyobj = res;
            } else {
                if (res) Py_DECREF(res);              // most likely Py_None
                else PyErr_Clear();                   // should not happen
            }
            Py_DECREF(dispproxy);

            if (pyobj && owns) {
                arg0_pyobj->CppOwns();
                ((CPPInstance*)pyobj)->PythonOwns();
            }

            return pyobj;
        }
    }

// not a pre-existing object; get the address and bind
    void* addr = nullptr;
    if (arg0 != gNullPtrObject) {
        addr = CPyCppyy_PyCapsule_GetPointer(arg0, nullptr);
        if (PyErr_Occurred()) {
            PyErr_Clear();

            addr = PyLong_AsVoidPtr(arg0);
            if (PyErr_Occurred()) {
                PyErr_Clear();

            // last chance, perhaps it's a buffer/array (return from void*)
                Py_ssize_t buflen = Utility::GetBuffer(PyTuple_GetItem(args, 0), '*', 1, addr, false);
                if (!addr || !buflen) {
                    PyErr_SetString(PyExc_TypeError,
                        "bind_object requires a CObject/Capsule, long integer, buffer, or instance as first argument");
                    return nullptr;
                }
            }
        }
    }

    bool do_cast = false;
    if (kwds) {
        PyObject* cast = PyDict_GetItemString(kwds, "cast");
        do_cast = cast && PyObject_IsTrue(cast);
    }

    if (do_cast)
        return BindCppObject(addr, cast_type);

    return BindCppObjectNoCast(addr, cast_type);
}

//----------------------------------------------------------------------------
static PyObject* Move(PyObject*, PyObject* pyobject)
{
// Prepare the given C++ object for moving.
    if (!CPPInstance_Check(pyobject)) {
        PyErr_SetString(PyExc_TypeError, "C++ object expected");
        return nullptr;
    }

    ((CPPInstance*)pyobject)->fFlags |= CPPInstance::kIsRValue;
    Py_INCREF(pyobject);
    return pyobject;
}


//----------------------------------------------------------------------------
static PyObject* AddPythonization(PyObject*, PyObject* args)
{
// Remove a previously registered pythonizor from the given scope.
    PyObject* pythonizor = nullptr; const char* scope;
    if (!PyArg_ParseTuple(args, const_cast<char*>("Os"), &pythonizor, &scope))
        return nullptr;

    if (!PyCallable_Check(pythonizor)) {
        PyObject* pystr = PyObject_Str(pythonizor);
        PyErr_Format(PyExc_TypeError,
            "given \'%s\' object is not callable", CPyCppyy_PyText_AsString(pystr));
        Py_DECREF(pystr);
        return nullptr;
    }

    Py_INCREF(pythonizor);
    pythonizations()[scope].push_back(pythonizor);

    Py_RETURN_NONE;
}


//----------------------------------------------------------------------------
static PyObject* RemovePythonization(PyObject*, PyObject* args)
{
// Remove a previously registered pythonizor from the given scope.
    PyObject* pythonizor = nullptr; const char* scope;
    if (!PyArg_ParseTuple(args, const_cast<char*>("Os"), &pythonizor, &scope))
        return nullptr;

    auto &pyzMap = pythonizations();
    auto p1 = pyzMap.find(scope);
    if (p1 != pyzMap.end()) {
        auto p2 = std::find(p1->second.begin(), p1->second.end(), pythonizor);
        if (p2 != p1->second.end()) {
            p1->second.erase(p2);
            Py_RETURN_TRUE;
        }
    }

    Py_RETURN_FALSE;
}

//----------------------------------------------------------------------------
static PyObject* PinType(PyObject*, PyObject* pyclass)
{
// Add a pinning so that objects of type `derived' are interpreted as
// objects of type `base'.
    if (!CPPScope_Check(pyclass)) {
        PyErr_SetString(PyExc_TypeError, "C++ class expected");
        return nullptr;
    }

    gPinnedTypes.insert(((CPPClass*)pyclass)->fCppType);

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* AddTypeReducer(PyObject*, PyObject* args)
{
// Add a type reducer to map type2 to type2 on function returns.
    const char *reducable, *reduced;
    if (!PyArg_ParseTuple(args, const_cast<char*>("ss"), &reducable, &reduced))
        return nullptr;

    Cppyy::AddTypeReducer(reducable, reduced);

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* SetMemoryPolicy(PyObject*, PyObject* args)
{
// Set the global memory policy, which affects object ownership when objects
// are passed as function arguments.
    PyObject* policy = nullptr;
    if (!PyArg_ParseTuple(args, const_cast<char*>("O!"), &PyInt_Type, &policy))
        return nullptr;

    long old = (long)CallContext::sMemoryPolicy;

    long l = PyInt_AS_LONG(policy);
    if (CallContext::SetMemoryPolicy((CallContext::ECallFlags)l)) {
        return PyInt_FromLong(old);
    }

    PyErr_Format(PyExc_ValueError, "Unknown policy %ld", l);
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* SetGlobalSignalPolicy(PyObject*, PyObject* args)
{
// Set the global signal policy, which determines whether a jmp address
// should be saved to return to after a C++ segfault.
    PyObject* setProtected = 0;
    if (!PyArg_ParseTuple(args, const_cast<char*>("O"), &setProtected))
        return nullptr;

    if (CallContext::SetGlobalSignalPolicy(PyObject_IsTrue(setProtected))) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

//----------------------------------------------------------------------------
static PyObject* SetOwnership(PyObject*, PyObject* args)
{
// Set the ownership (True is python-owns) for the given object.
    CPPInstance* pyobj = nullptr; PyObject* pykeep = nullptr;
    if (!PyArg_ParseTuple(args, const_cast<char*>("O!O!"),
            &CPPInstance_Type, (void*)&pyobj, &PyInt_Type, &pykeep))
        return nullptr;

    (bool)PyLong_AsLong(pykeep) ? pyobj->PythonOwns() : pyobj->CppOwns();

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* AddSmartPtrType(PyObject*, PyObject* args)
{
// Add a smart pointer to the list of known smart pointer types.
    const char* type_name;
    if (!PyArg_ParseTuple(args, const_cast<char*>("s"), &type_name))
        return nullptr;

    Cppyy::AddSmartPtrType(type_name);

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* BeginCaptureStderr(PyObject*, PyObject*)
{
    gOldErrorBuffer = std::cerr.rdbuf();
    std::cerr.rdbuf(gCapturedError.rdbuf());

    Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* EndCaptureStderr(PyObject*, PyObject*)
{
// restore old rdbuf and return captured result
    std::cerr.rdbuf(gOldErrorBuffer);
    gOldErrorBuffer = nullptr;

    std::string capturedError = std::move(gCapturedError).str();

    gCapturedError.str("");
    gCapturedError.clear();

    return Py_BuildValue("s", capturedError.c_str());
}
} // unnamed namespace


//- data -----------------------------------------------------------------------
static PyMethodDef gCPyCppyyMethods[] = {
    {(char*) "CreateScopeProxy", (PyCFunction)CPyCppyy::CreateScopeProxy,
      METH_VARARGS, (char*)"cppyy internal function"},
    {(char*) "MakeCppTemplateClass", (PyCFunction)MakeCppTemplateClass,
      METH_VARARGS, (char*)"cppyy internal function"},
    {(char*) "_set_cpp_lazy_lookup", (PyCFunction)SetCppLazyLookup,
      METH_VARARGS, (char*)"cppyy internal function"},
    {(char*) "_DestroyPyStrings", (PyCFunction)CPyCppyy::DestroyPyStrings,
      METH_NOARGS, (char*)"cppyy internal function"},
    {(char*) "addressof", (PyCFunction)addressof,
      METH_VARARGS | METH_KEYWORDS, (char*)"Retrieve address of proxied object or field as a value."},
    {(char*) "as_cobject", (PyCFunction)AsCObject,
      METH_VARARGS | METH_KEYWORDS, (char*)"Retrieve address of proxied object or field in a CObject."},
    {(char*) "as_capsule", (PyCFunction)AsCapsule,
      METH_VARARGS | METH_KEYWORDS, (char*)"Retrieve address of proxied object or field in a PyCapsule."},
    {(char*) "as_ctypes", (PyCFunction)AsCTypes,
      METH_VARARGS | METH_KEYWORDS, (char*)"Retrieve address of proxied object or field in a ctypes c_void_p."},
    {(char*) "as_memoryview", (PyCFunction)AsMemoryView,
      METH_O, (char*)"Represent an array of objects as raw memory."},
    {(char*)"bind_object", (PyCFunction)BindObject,
      METH_VARARGS | METH_KEYWORDS, (char*) "Create an object of given type, from given address."},
    {(char*) "move", (PyCFunction)Move,
      METH_O, (char*)"Cast the C++ object to become movable."},
    {(char*) "add_pythonization", (PyCFunction)AddPythonization,
      METH_VARARGS, (char*)"Add a pythonizor."},
    {(char*) "remove_pythonization", (PyCFunction)RemovePythonization,
      METH_VARARGS, (char*)"Remove a pythonizor."},
    {(char*) "_pin_type", (PyCFunction)PinType,
      METH_O, (char*)"Install a type pinning."},
    {(char*) "_add_type_reducer", (PyCFunction)AddTypeReducer,
      METH_VARARGS, (char*)"Add a type reducer."},
    {(char*) "SetMemoryPolicy", (PyCFunction)SetMemoryPolicy,
      METH_VARARGS, (char*)"Determines object ownership model."},
    {(char*) "SetGlobalSignalPolicy", (PyCFunction)SetGlobalSignalPolicy,
      METH_VARARGS, (char*)"Trap signals in safe mode to prevent interpreter abort."},
    {(char*) "SetOwnership", (PyCFunction)SetOwnership,
      METH_VARARGS, (char*)"Modify held C++ object ownership."},
    {(char*) "AddSmartPtrType", (PyCFunction)AddSmartPtrType,
      METH_VARARGS, (char*) "Add a smart pointer to the list of known smart pointer types."},
    {(char*) "_begin_capture_stderr", (PyCFunction)BeginCaptureStderr,
      METH_NOARGS, (char*) "Begin capturing stderr to a in memory buffer."},
    {(char*) "_end_capture_stderr", (PyCFunction)EndCaptureStderr,
      METH_NOARGS, (char*) "End capturing stderr and returns the captured buffer."},
    {nullptr, nullptr, 0, nullptr}
};


#if PY_VERSION_HEX >= 0x03000000
struct module_state {
    PyObject *error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

static int cpycppyymodule_traverse(PyObject* m, visitproc visit, void* arg)
{
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int cpycppyymodule_clear(PyObject* m)
{
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "libcppyy",
    nullptr,
    sizeof(struct module_state),
    gCPyCppyyMethods,
    nullptr,
    cpycppyymodule_traverse,
    cpycppyymodule_clear,
    nullptr
};
#endif

namespace CPyCppyy {

//----------------------------------------------------------------------------
PyObject* Init()
{
// Initialization of extension module libcppyy.

// load commonly used python strings
    if (!CPyCppyy::CreatePyStrings())
        return nullptr;

// setup interpreter
#if PY_VERSION_HEX < 0x03090000
    PyEval_InitThreads();
#endif

#if PY_VERSION_HEX < 0x030b0000
// prepare for laziness (the insert is needed to capture the most generic lookup
// function, just in case ...)
    PyObject* dict = PyDict_New();
    PyObject* notstring = PyInt_FromLong(5);
    PyDict_SetItem(dict, notstring, notstring);
    Py_DECREF(notstring);
#if PY_VERSION_HEX >= 0x03030000
    gDictLookupOrg = (dict_lookup_func)((PyDictObject*)dict)->ma_keys->dk_lookup;
#else
    gDictLookupOrg = (dict_lookup_func)((PyDictObject*)dict)->ma_lookup;
#endif
    Py_DECREF(dict);
#endif // PY_VERSION_HEX < 0x030b0000

// setup this module
#if PY_VERSION_HEX >= 0x03000000
    gThisModule = PyModule_Create(&moduledef);
#else
    gThisModule = Py_InitModule(const_cast<char*>("libcppyy"), gCPyCppyyMethods);
#endif
    if (!gThisModule)
        return nullptr;

// keep gThisModule, but do not increase its reference count even as it is borrowed,
// or a self-referencing cycle would be created

// external types
    gPyTypeMap = PyDict_New();
    PyModule_AddObject(gThisModule, "type_map", gPyTypeMap);    // steals reference

// Pythonizations ...
    PyModule_AddObject(gThisModule, "UserExceptions",     PyDict_New());

// inject meta type
    if (!Utility::InitProxy(gThisModule, &CPPScope_Type, "CPPScope"))
        return nullptr;

// inject object proxy type
    if (!Utility::InitProxy(gThisModule, &CPPInstance_Type, "CPPInstance"))
        return nullptr;

// inject exception object proxy type
    if (!Utility::InitProxy(gThisModule, &CPPExcInstance_Type, "CPPExcInstance"))
        return nullptr;

// inject method proxy type
    if (!Utility::InitProxy(gThisModule, &CPPOverload_Type, "CPPOverload"))
        return nullptr;

// inject template proxy type
    if (!Utility::InitProxy(gThisModule, &TemplateProxy_Type, "TemplateProxy"))
        return nullptr;

// inject property proxy type
    if (!Utility::InitProxy(gThisModule, &CPPDataMember_Type, "CPPDataMember"))
        return nullptr;

// inject custom data types
#if PY_VERSION_HEX < 0x03000000
    if (!Utility::InitProxy(gThisModule, &RefFloat_Type, "Double"))
        return nullptr;

    if (!Utility::InitProxy(gThisModule, &RefInt_Type, "Long"))
        return nullptr;
#endif

    if (!Utility::InitProxy(gThisModule, &CustomInstanceMethod_Type, "InstanceMethod"))
        return nullptr;

    if (!Utility::InitProxy(gThisModule, &TupleOfInstances_Type, "InstanceArray"))
       return nullptr;

    if (!Utility::InitProxy(gThisModule, &LowLevelView_Type, "LowLevelView"))
        return nullptr;

    if (!Utility::InitProxy(gThisModule, &PyNullPtr_t_Type, "nullptr_t"))
        return nullptr;

// custom iterators
    if (PyType_Ready(&InstanceArrayIter_Type) < 0)
        return nullptr;

    if (PyType_Ready(&IndexIter_Type) < 0)
        return nullptr;

    if (PyType_Ready(&VectorIter_Type) < 0)
        return nullptr;

// inject identifiable nullptr and default
    gNullPtrObject = (PyObject*)&_CPyCppyy_NullPtrStruct;
    Py_INCREF(gNullPtrObject);
    PyModule_AddObject(gThisModule, (char*)"nullptr", gNullPtrObject);

    gDefaultObject = (PyObject*)&_CPyCppyy_DefaultStruct;
    Py_INCREF(gDefaultObject);
    PyModule_AddObject(gThisModule, (char*)"default", gDefaultObject);

// C++-specific exceptions
    PyObject* cppfatal = PyErr_NewException((char*)"cppyy.ll.FatalError", nullptr, nullptr);
    PyModule_AddObject(gThisModule, (char*)"FatalError", cppfatal);

    gBusException  = PyErr_NewException((char*)"cppyy.ll.BusError", cppfatal, nullptr);
    PyModule_AddObject(gThisModule, (char*)"BusError", gBusException);
    gSegvException = PyErr_NewException((char*)"cppyy.ll.SegmentationViolation", cppfatal, nullptr);
    PyModule_AddObject(gThisModule, (char*)"SegmentationViolation", gSegvException);
    gIllException  = PyErr_NewException((char*)"cppyy.ll.IllegalInstruction", cppfatal, nullptr);
    PyModule_AddObject(gThisModule, (char*)"IllegalInstruction", gIllException);
    gAbrtException = PyErr_NewException((char*)"cppyy.ll.AbortSignal", cppfatal, nullptr);
    PyModule_AddObject(gThisModule, (char*)"AbortSignal", gAbrtException);

// policy labels
    PyModule_AddObject(gThisModule, (char*)"kMemoryHeuristics",
        PyInt_FromLong((int)CallContext::kUseHeuristics));
    PyModule_AddObject(gThisModule, (char*)"kMemoryStrict",
        PyInt_FromLong((int)CallContext::kUseStrict));

// gbl namespace is injected in cppyy.py

// create the memory regulator
    static MemoryRegulator s_memory_regulator;

#if PY_VERSION_HEX >= 0x03000000
    Py_INCREF(gThisModule);
#endif
    return gThisModule;
}

} // namespace CPyCppyy
