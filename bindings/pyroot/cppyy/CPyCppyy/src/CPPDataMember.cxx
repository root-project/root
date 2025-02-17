// Bindings
#include "CPyCppyy.h"
#include "CPyCppyy/Reflex.h"
#include "PyStrings.h"
#include "CPPDataMember.h"
#include "CPPInstance.h"
#include "Dimensions.h"
#include "LowLevelViews.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <algorithm>
#include <vector>
#include <limits.h>
#include <structmember.h>

namespace CPyCppyy {

enum ETypeDetails {
    kNone          = 0x0000,
    kIsStaticData  = 0x0001,
    kIsConstData   = 0x0002,
    kIsArrayType   = 0x0004,
    kIsEnumPrep    = 0x0008,
    kIsEnumType    = 0x0010,
    kIsCachable    = 0x0020
};

//= CPyCppyy data member as Python property behavior =========================
static PyObject* dm_get(CPPDataMember* dm, CPPInstance* pyobj, PyObject* /* kls */)
{
// cache lookup for low level views
    if (pyobj && dm->fFlags & kIsCachable) {
        CPyCppyy::CI_DatamemberCache_t& cache = pyobj->GetDatamemberCache();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == dm->fOffset) {
                if (it->second) {
                    Py_INCREF(it->second);
                    return it->second;
                } else
                    cache.erase(it);
                break;
            }
        }
    }

// non-initialized or public data accesses through class (e.g. by help())
    void* address = dm->GetAddress(pyobj);
    if (!address || (intptr_t)address == -1 /* Cling error */)
        return nullptr;

    if (dm->fFlags & (kIsEnumPrep | kIsEnumType)) {
        if (dm->fFlags & kIsEnumPrep) {
        // still need to do lookup; only ever try this once, then fallback on converter
            dm->fFlags &= ~kIsEnumPrep;

        // fDescription contains the full name of the actual enum value object
            const std::string& lookup = CPyCppyy_PyText_AsString(dm->fDescription);
            const std::string& enum_type  = TypeManip::extract_namespace(lookup);
            const std::string& enum_scope = TypeManip::extract_namespace(enum_type);

            PyObject* pyscope = nullptr;
            if (enum_scope.empty()) pyscope = GetScopeProxy(Cppyy::gGlobalScope);
            else pyscope = CreateScopeProxy(enum_scope);
            if (pyscope) {
                PyObject* pyEnumType = PyObject_GetAttrString(pyscope,
                    enum_type.substr(enum_scope.size() ? enum_scope.size()+2 : 0, std::string::npos).c_str());
                if (pyEnumType) {
                    PyObject* pyval = PyObject_GetAttrString(pyEnumType,
                        lookup.substr(enum_type.size()+2, std::string::npos).c_str());
                    Py_DECREF(pyEnumType);
                    if (pyval) {
                        Py_DECREF(dm->fDescription);
                        dm->fDescription = pyval;
                        dm->fFlags |= kIsEnumType;
                    }
                }
                Py_DECREF(pyscope);
            }
            if (!(dm->fFlags & kIsEnumType))
                PyErr_Clear();
        }

        if (dm->fFlags & kIsEnumType) {
            Py_INCREF(dm->fDescription);
            return dm->fDescription;
        }
    }

    if (dm->fConverter != 0) {
        PyObject* result = dm->fConverter->FromMemory((dm->fFlags & kIsArrayType) ? &address : address);
        if (!result)
            return result;

    // low level views are expensive to create, so cache them on the object instead
        bool isLLView = LowLevelView_CheckExact(result);
        if (isLLView && CPPInstance_Check(pyobj)) {
            Py_INCREF(result);
            pyobj->GetDatamemberCache().push_back(std::make_pair(dm->fOffset, result));
            dm->fFlags |= kIsCachable;
        }

    // ensure that the encapsulating class does not go away for the duration
    // of the data member's lifetime, if it is a bound type (it doesn't matter
    // for builtin types, b/c those are copied over into python types and thus
    // end up being "stand-alone")
    // TODO: should be done for LLViews as well
        else if (pyobj && !(dm->fFlags & kIsStaticData) && CPPInstance_Check(result)) {
            if (PyObject_SetAttr(result, PyStrings::gLifeLine, (PyObject*)pyobj) == -1)
                PyErr_Clear();     // ignored
        }

        return result;
    }

    PyErr_Format(PyExc_NotImplementedError,
        "no converter available for \"%s\"", dm->GetName().c_str());
    return nullptr;
}

//-----------------------------------------------------------------------------
static int dm_set(CPPDataMember* dm, CPPInstance* pyobj, PyObject* value)
{
// Set the value of the C++ datum held.
    const int errret = -1;

    if (!value) {
    // we're being deleted; fine for namespaces (redo lookup next time), but makes
    // no sense for classes/structs
        if (!Cppyy::IsNamespace(dm->fEnclosingScope)) {
            PyErr_SetString(PyExc_TypeError, "data member deletion is not supported");
            return errret;
        }

    // deletion removes the proxy, with the idea that a fresh lookup can be made,
    // to support Cling's shadowing of declarations (TODO: the use case here is
    // redeclared variables, for which fDescription is indeed th ename; it might
    // fail for enums).
        return PyObject_DelAttr((PyObject*)Py_TYPE(pyobj), dm->fDescription);
    }

// filter const objects to prevent changing their values
    if (dm->fFlags & kIsConstData) {
        PyErr_SetString(PyExc_TypeError, "assignment to const data not allowed");
        return errret;
    }

// remove cached low level view, if any (will be restored upon reaeding)
    if (dm->fFlags & kIsCachable) {
        CPyCppyy::CI_DatamemberCache_t& cache = pyobj->GetDatamemberCache();
        for (auto it = cache.begin(); it != cache.end(); ++it) {
            if (it->first == dm->fOffset) {
                Py_XDECREF(it->second);
                cache.erase(it);
                break;
            }
        }
    }

    intptr_t address = (intptr_t)dm->GetAddress(pyobj);
    if (!address || address == -1 /* Cling error */)
        return errret;

// for fixed size arrays
    void* ptr = (void*)address;
    if (dm->fFlags & kIsArrayType)
        ptr = &address;

// actual conversion; return on success
    if (dm->fConverter && dm->fConverter->ToMemory(value, ptr, (PyObject*)pyobj))
        return 0;

// set a python error, if not already done
    if (!PyErr_Occurred())
        PyErr_SetString(PyExc_RuntimeError, "property type mismatch or assignment not allowed");

// failure ...
    return errret;
}

//= CPyCppyy data member construction/destruction ===========================
static CPPDataMember* dm_new(PyTypeObject* pytype, PyObject*, PyObject*)
{
// Create and initialize a new property descriptor.
    CPPDataMember* dm = (CPPDataMember*)pytype->tp_alloc(pytype, 0);

    dm->fOffset         = 0;
    dm->fFlags          = 0;
    dm->fConverter      = nullptr;
    dm->fEnclosingScope = 0;
    dm->fDescription    = nullptr;
    dm->fDoc            = nullptr;

    new (&dm->fFullType) std::string{};

    return dm;
}

//----------------------------------------------------------------------------
static void dm_dealloc(CPPDataMember* dm)
{
// Deallocate memory held by this descriptor.
    using namespace std;
    if (dm->fConverter && dm->fConverter->HasState()) delete dm->fConverter;
    Py_XDECREF(dm->fDescription);  // never exposed so no GC necessary
    Py_XDECREF(dm->fDoc);

    dm->fFullType.~string();

    Py_TYPE(dm)->tp_free((PyObject*)dm);
}

static PyMemberDef dm_members[] = {
        {(char*)"__doc__", T_OBJECT, offsetof(CPPDataMember, fDoc), 0,
                (char*)"writable documentation"},
        {NULL, 0, 0, 0, nullptr}  /* Sentinel */
};

//= CPyCppyy datamember proxy access to internals ============================
static PyObject* dm_reflex(CPPDataMember* dm, PyObject* args)
{
// Provide the requested reflection information.
    Cppyy::Reflex::RequestId_t request = -1;
    Cppyy::Reflex::FormatId_t  format  = Cppyy::Reflex::OPTIMAL;
    if (!PyArg_ParseTuple(args, const_cast<char*>("i|i:__cpp_reflex__"), &request, &format))
        return nullptr;

    if (request == Cppyy::Reflex::TYPE) {
        if (format == Cppyy::Reflex::OPTIMAL || format == Cppyy::Reflex::AS_STRING)
            return CPyCppyy_PyText_FromString(dm->fFullType.c_str());
    } else if (request == Cppyy::Reflex::OFFSET) {
        if (format == Cppyy::Reflex::OPTIMAL)
            return PyLong_FromLong(dm->fOffset);
    }

    PyErr_Format(PyExc_ValueError, "unsupported reflex request %d or format %d", request, format);
    return nullptr;
}

//----------------------------------------------------------------------------
static PyMethodDef dm_methods[] = {
    {(char*)"__cpp_reflex__", (PyCFunction)dm_reflex, METH_VARARGS,
      (char*)"C++ datamember reflection information" },
    {(char*)nullptr, nullptr, 0, nullptr }
};


//= CPyCppyy data member type ================================================
PyTypeObject CPPDataMember_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.CPPDataMember",  // tp_name
    sizeof(CPPDataMember),         // tp_basicsize
    0,                             // tp_itemsize
    (destructor)dm_dealloc,        // tp_dealloc
    0,                             // tp_vectorcall_offset / tp_print
    0,                             // tp_getattr
    0,                             // tp_setattr
    0,                             // tp_as_async / tp_compare
    0,                             // tp_repr
    0,                             // tp_as_number
    0,                             // tp_as_sequence
    0,                             // tp_as_mapping
    0,                             // tp_hash
    0,                             // tp_call
    0,                             // tp_str
    0,                             // tp_getattro
    0,                             // tp_setattro
    0,                             // tp_as_buffer
    Py_TPFLAGS_DEFAULT,            // tp_flags
    (char*)"cppyy data member (internal)",       // tp_doc
    0,                             // tp_traverse
    0,                             // tp_clear
    0,                             // tp_richcompare
    0,                             // tp_weaklistoffset
    0,                             // tp_iter
    0,                             // tp_iternext
    dm_methods,                    // tp_methods
    dm_members,                    // tp_members
    0,                             // tp_getset
    0,                             // tp_base
    0,                             // tp_dict
    (descrgetfunc)dm_get,          // tp_descr_get
    (descrsetfunc)dm_set,          // tp_descr_set
    0,                             // tp_dictoffset
    0,                             // tp_init
    0,                             // tp_alloc
    (newfunc)dm_new,               // tp_new
    0,                             // tp_free
    0,                             // tp_is_gc
    0,                             // tp_bases
    0,                             // tp_mro
    0,                             // tp_cache
    0,                             // tp_subclasses
    0                              // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
    , 0                            // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                            // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                            // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                           // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                           // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                           // tp_versions_used
#endif
};

} // namespace CPyCppyy


//- public members -----------------------------------------------------------
void CPyCppyy::CPPDataMember::Set(Cppyy::TCppScope_t scope, Cppyy::TCppIndex_t idata)
{
    fEnclosingScope = scope;
    fOffset         = Cppyy::GetDatamemberOffset(scope, idata); // TODO: make lazy
    fFlags          = Cppyy::IsStaticData(scope, idata) ? kIsStaticData : 0;

    std::vector<dim_t> dims;
    int ndim = 0; Py_ssize_t size = 0;
    while (0 < (size = Cppyy::GetDimensionSize(scope, idata, ndim))) {
         ndim += 1;
         if (size == INT_MAX)      // meaning: incomplete array type
             size = UNKNOWN_SIZE;
         if (ndim == 1) dims.reserve(4);
         dims.push_back((dim_t)size);
    }
    if (!dims.empty())
        fFlags |= kIsArrayType;

    const std::string name = Cppyy::GetDatamemberName(scope, idata);
    fFullType = Cppyy::GetDatamemberType(scope, idata);
    if (Cppyy::IsEnumData(scope, idata)) {
        if (fFullType.find("(anonymous)") == std::string::npos &&
            fFullType.find("(unnamed)")   == std::string::npos) {
        // repurpose fDescription for lazy lookup of the enum later
            fDescription = CPyCppyy_PyText_FromString((fFullType + "::" + name).c_str());
            fFlags |= kIsEnumPrep;
        }
        fFullType = Cppyy::ResolveEnum(fFullType);
        fFlags |= kIsConstData;
    } else if (Cppyy::IsConstData(scope, idata)) {
        fFlags |= kIsConstData;
    }

// if this data member is an array, the conversion needs to be pointer to object for instances,
// to prevent the need for copying in the conversion; furthermore, fixed arrays' full type for
// builtins are not declared as such if more than 1-dim (TODO: fix in clingwrapper)
    if (!dims.empty() && fFullType.back() != '*') {
        if (Cppyy::GetScope(fFullType)) fFullType += '*';
        else if (fFullType.back() != ']') {
            for (auto d: dims) fFullType += d == UNKNOWN_SIZE ? "*" : "[]";
        }
    }

    if (dims.empty())
        fConverter = CreateConverter(fFullType);
    else
        fConverter = CreateConverter(fFullType, {(dim_t)dims.size(), dims.data()});

    if (!(fFlags & kIsEnumPrep))
        fDescription = CPyCppyy_PyText_FromString(name.c_str());
}

//-----------------------------------------------------------------------------
void CPyCppyy::CPPDataMember::Set(Cppyy::TCppScope_t scope, const std::string& name, void* address)
{
    fEnclosingScope = scope;
    fDescription    = CPyCppyy_PyText_FromString(name.c_str());
    fOffset         = (intptr_t)address;
    fFlags          = kIsStaticData | kIsConstData;
    fConverter      = CreateConverter("internal_enum_type_t");
    fFullType       = "unsigned int";
}


//-----------------------------------------------------------------------------
void* CPyCppyy::CPPDataMember::GetAddress(CPPInstance* pyobj)
{
// class attributes, global properties
    if (fFlags & kIsStaticData)
        return (void*)fOffset;

// special case: non-static lookup through class
    if (!pyobj) {
        PyErr_SetString(PyExc_AttributeError, "attribute access requires an instance");
        return nullptr;
    }

// instance attributes; requires valid object for full address
    if (!CPPInstance_Check(pyobj)) {
        PyErr_Format(PyExc_TypeError,
            "object instance required for access to property \"%s\"", GetName().c_str());
        return nullptr;
    }

    void* obj = pyobj->GetObject();
    if (!obj) {
        PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
        return nullptr;
   }

// the proxy's internal offset is calculated from the enclosing class
    ptrdiff_t offset = 0;
    Cppyy::TCppType_t oisa = pyobj->ObjectIsA();
    if (oisa != fEnclosingScope)
        offset = Cppyy::GetBaseOffset(oisa, fEnclosingScope, obj, 1 /* up-cast */);

    return (void*)((intptr_t)obj + offset + fOffset);
}


//-----------------------------------------------------------------------------
std::string CPyCppyy::CPPDataMember::GetName()
{
    if (fFlags & kIsEnumType) {
        PyObject* repr = PyObject_Repr(fDescription);
        if (repr) {
            std::string res = CPyCppyy_PyText_AsString(repr);
            Py_DECREF(repr);
            return res;
        }
        PyErr_Clear();
        return "<unknown>";
    } else if (fFlags & kIsEnumPrep) {
        std::string fullName = CPyCppyy_PyText_AsString(fDescription);
        return fullName.substr(fullName.rfind("::")+2, std::string::npos);
    }

    return CPyCppyy_PyText_AsString(fDescription);
}
