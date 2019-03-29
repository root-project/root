// Bindings
#include "CPyCppyy.h"
#include "CPPScope.h"
#include "CPPDataMember.h"
#include "CPPFunction.h"
#include "CPPOverload.h"
#include "Dispatcher.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TemplateProxy.h"
#include "TypeManip.h"
#include "Utility.h"

// Standard
#include <string.h>
#include <algorithm>         // for for_each
#include <set>
#include <string>
#include <vector>


namespace CPyCppyy {

extern PyTypeObject CPPInstance_Type;

//= CPyCppyy type proxy construction/destruction =============================
static PyObject* meta_alloc(PyTypeObject* meta, Py_ssize_t nitems)
{
// pure memory allocation; object initialization is in pt_new
    return PyType_Type.tp_alloc(meta, nitems);
}

//----------------------------------------------------------------------------
static void meta_dealloc(CPPScope* scope)
{
    if (scope->fFlags & CPPScope::kIsNamespace) {
        if (scope->fImp.fUsing) {
            for (auto pyobj : *scope->fImp.fUsing) Py_DECREF(pyobj);
            delete scope->fImp.fUsing; scope->fImp.fUsing = nullptr;
        }
    } else {
        for (auto& pp : *scope->fImp.fCppObjects) Py_DECREF(pp.second);
        delete scope->fImp.fCppObjects; scope->fImp.fCppObjects = nullptr;
    }
    free(scope->fModuleName);
    return PyType_Type.tp_dealloc((PyObject*)scope);
}

//-----------------------------------------------------------------------------
static PyObject* meta_getcppname(CPPScope* scope, void*)
{
    if ((void*)scope == (void*)&CPPInstance_Type)
        return CPyCppyy_PyUnicode_FromString("CPPInstance_Type");
    return CPyCppyy_PyUnicode_FromString(Cppyy::GetScopedFinalName(scope->fCppType).c_str());
}

//-----------------------------------------------------------------------------
static PyObject* meta_getmodule(CPPScope* scope, void*)
{
    if ((void*)scope == (void*)&CPPInstance_Type)
        return CPyCppyy_PyUnicode_FromString("cppyy.gbl");

    if (scope->fModuleName)
        return CPyCppyy_PyUnicode_FromString(scope->fModuleName);

// get C++ representation of outer scope
    std::string modname =
        TypeManip::extract_namespace(Cppyy::GetScopedFinalName(scope->fCppType));
    if (modname.empty())
        return CPyCppyy_PyUnicode_FromString(const_cast<char*>("cppyy.gbl"));

// now peel scopes one by one, pulling in the python naming (which will
// simply recurse if not overridden in python)
    PyObject* pymodule = nullptr;
    PyObject* pyscope = CPyCppyy::GetScopeProxy(Cppyy::GetScope(modname));
    if (pyscope) {
    // get the module of our module
        pymodule = PyObject_GetAttr(pyscope, PyStrings::gModule);
        if (pymodule) {
        // append name of our module
            PyObject* pymodname = PyObject_GetAttr(pyscope, PyStrings::gName);
            if (pymodname) {
                CPyCppyy_PyUnicode_AppendAndDel(
                    &pymodule, CPyCppyy_PyUnicode_FromString("."));
                CPyCppyy_PyUnicode_AppendAndDel(&pymodule, pymodname);
            }
        }
        Py_DECREF(pyscope);
    }

    if (pymodule)
        return pymodule;
    PyErr_Clear();

// lookup through python failed, so simply cook up a '::' -> '.' replacement
    TypeManip::cppscope_to_pyscope(modname);
    return CPyCppyy_PyUnicode_FromString(("cppyy.gbl."+modname).c_str());
}

//-----------------------------------------------------------------------------
static int meta_setmodule(CPPScope* scope, PyObject* value, void*)
{
    if ((void*)scope == (void*)&CPPInstance_Type) {
        PyErr_SetString(PyExc_AttributeError,
            "attribute \'__module__\' of 'cppyy.CPPScope\' objects is not writable");
        return -1;
    }

    const char* newname = CPyCppyy_PyUnicode_AsStringChecked(value);
    if (!value)
        return -1;

    free(scope->fModuleName);
    Py_ssize_t sz = CPyCppyy_PyUnicode_GET_SIZE(value);
    scope->fModuleName = (char*)malloc(sz+1);
    memcpy(scope->fModuleName, newname, sz+1);

    return 0;
}

//----------------------------------------------------------------------------
static PyObject* meta_repr(CPPScope* scope)
{
// Specialized b/c type_repr expects __module__ to live in the dictionary,
// whereas it is a property (to save memory).
    if ((void*)scope == (void*)&CPPInstance_Type)
        return CPyCppyy_PyUnicode_FromFormat(
            const_cast<char*>("<class cppyy.CPPInstance at %p>"), scope);

    if (scope->fFlags & (CPPScope::kIsMeta | CPPScope::kIsPython)) {
    // either meta type or Python-side derived class: use default type printing
        return PyType_Type.tp_repr((PyObject*)scope);
    }

// printing of C++ classes
    PyObject* modname = meta_getmodule(scope, nullptr);
    std::string clName = Cppyy::GetFinalName(scope->fCppType);
    const char* kind = (scope->fFlags & CPPScope::kIsNamespace) ? "namespace" : "class";

    PyObject* repr = CPyCppyy_PyUnicode_FromFormat("<%s %s.%s at %p>",
        kind, CPyCppyy_PyUnicode_AsString(modname), clName.c_str(), scope);

    Py_DECREF(modname);
    return repr;
}


//= CPyCppyy type metaclass behavior =========================================
static PyObject* pt_new(PyTypeObject* subtype, PyObject* args, PyObject* kwds)
{
// Called when CPPScope acts as a metaclass; since type_new always resets
// tp_alloc, and since it does not call tp_init on types, the metaclass is
// being fixed up here, and the class is initialized here as well.

// fixup of metaclass (left permanent, and in principle only called once b/c
// cppyy caches python classes)
    subtype->tp_alloc   = (allocfunc)meta_alloc;
    subtype->tp_dealloc = (destructor)meta_dealloc;

// creation of the python-side class
    CPPScope* result = (CPPScope*)PyType_Type.tp_new(subtype, args, kwds);
    if (!result)
        return nullptr;

    result->fFlags      = CPPScope::kNone;
    result->fModuleName = nullptr;

// initialization of class (based on metatype)
    const char* mp = strstr(subtype->tp_name, "_meta");
    if (!mp || !CPPScope_CheckExact(subtype)) {
    // there has been a user meta class override in a derived class, so do
    // the consistent thing, thus allowing user control over naming
        result->fCppType = Cppyy::GetScope(
            CPyCppyy_PyUnicode_AsString(PyTuple_GET_ITEM(args, 0)));
    } else {
    // coming here from cppyy or from sub-classing in python; take the
    // C++ type from the meta class to make sure that the latter category
    // has fCppType properly set (it inherits the meta class, but has an
    // otherwise unknown (or wrong) C++ type)
        result->fCppType = ((CPPScope*)subtype)->fCppType;

    // the following is not robust, but by design, C++ classes get their
    // dictionaries filled after creation (chicken & egg problem as they
    // can return themselves in methods), whereas a derived Python class
    // with method overrides will have a non-empty dictionary (even if it
    // has no methods, it will at least have a module name)
        if (3 <= PyTuple_GET_SIZE(args)) {
            PyObject* dct = PyTuple_GET_ITEM(args, 2);
            Py_ssize_t sz = PyDict_Size(dct);
            if (0 < sz && !Cppyy::IsNamespace(result->fCppType)) {
                result->fFlags |= CPPScope::kIsPython;
                if (!InsertDispatcher(result, dct)) {
                    PyErr_Warn(PyExc_RuntimeWarning,
                        (char*)"no python-side overrides supported");
                }
            } else if (sz == (Py_ssize_t)-1)
                PyErr_Clear();
        }
    }

    if (!Cppyy::IsNamespace(result->fCppType))
        result->fImp.fCppObjects = new CppToPyMap_t;
    else {
        result->fImp.fUsing = nullptr;
        result->fFlags |= CPPScope::kIsNamespace;
    }

    return (PyObject*)result;
}

//----------------------------------------------------------------------------
static PyObject* meta_getattro(PyObject* pyclass, PyObject* pyname)
{
// normal type-based lookup
    PyObject* attr = PyType_Type.tp_getattro(pyclass, pyname);
    if (attr || pyclass == (PyObject*)&CPPInstance_Type)
        return attr;

    if (!CPyCppyy_PyUnicode_CheckExact(pyname) || !CPPScope_Check(pyclass))
        return nullptr;

// filter for python specials
    std::string name = CPyCppyy_PyUnicode_AsString(pyname);
    if (name.size() >= 2 && name.compare(0, 2, "__") == 0 &&
            name.compare(name.size()-2, name.size(), "__") == 0)
        return nullptr;

// more elaborate search in case of failure (eg. for inner classes on demand)
    std::vector<Utility::PyError_t> errors;
    Utility::FetchError(errors);
    attr = CreateScopeProxy(name, pyclass);

    CPPScope* klass = ((CPPScope*)pyclass);
    if (!attr) {
        Utility::FetchError(errors);
        Cppyy::TCppScope_t scope = klass->fCppType;

    // namespaces may have seen updates in their list of global functions, which
    // are available as "methods" even though they're not really that
        if (klass->fFlags & CPPScope::kIsNamespace) {
        // tickle lazy lookup of functions
            const std::vector<Cppyy::TCppIndex_t> methods =
                Cppyy::GetMethodIndicesFromName(scope, name);
            if (!methods.empty()) {
            // function exists, now collect overloads
                std::vector<PyCallable*> overloads;
                for (auto idx : methods) {
                    overloads.push_back(
                        new CPPFunction(scope, Cppyy::GetMethod(scope, idx)));
                }

            // Note: can't re-use Utility::AddClass here, as there's the risk of
            // a recursive call. Simply add method directly, as we're guaranteed
            // that it doesn't exist yet.
                attr = (PyObject*)CPPOverload_New(name, overloads);

            // If both templated and not, the templated one needs to be user-facing
            // in order to expose the instantiation mechanims.
                if (Cppyy::ExistsMethodTemplate(scope, name)) {
                    TemplateProxy* pytmpl = TemplateProxy_New(name, name, pyclass);
                    pytmpl->AddOverload((CPPOverload*)attr);
                    attr = (PyObject*)pytmpl;
                }
            }

        // tickle lazy lookup of data members
            if (!attr) {
                Cppyy::TCppIndex_t dmi = Cppyy::GetDatamemberIndex(scope, name);
                if (dmi != (Cppyy::TCppIndex_t)-1) attr = (PyObject*)CPPDataMember_New(scope, dmi);
            }
        }

    // function templates that have not been instantiated
        if (!attr && Cppyy::ExistsMethodTemplate(scope, name)) {
            attr = (PyObject*)TemplateProxy_New(name, name, pyclass);
        } else {
        // for completeness in error reporting
            PyErr_Format(PyExc_TypeError, "\'%s\' is not a known C++ template", name.c_str());
            Utility::FetchError(errors);
        }

    // enums types requested as type (rather than the constants)
    // TODO: IsEnum should deal with the scope, using klass->GetListOfEnums()->FindObject()
        if (!attr && Cppyy::IsEnum(scope == Cppyy::gGlobalScope ? name : Cppyy::GetScopedFinalName(scope)+"::"+name)) {
        // special case; enum types; for now, pretend int
        // TODO: although fine for C++98, this isn't correct in C++11
            Py_INCREF(&PyInt_Type);
            attr = (PyObject*)&PyInt_Type;
        } else {
        // for completeness in error reporting
            PyErr_Format(PyExc_TypeError, "\'%s\' is not a known C++ enum", name.c_str());
            Utility::FetchError(errors);
        }

        if (attr) {
        // cache the result
            if (CPPDataMember_Check(attr)) {
                PyObject_SetAttr((PyObject*)Py_TYPE(pyclass), pyname, attr);
                Py_DECREF(attr);
                attr = PyType_Type.tp_getattro(pyclass, pyname);
            } else
                PyObject_SetAttr(pyclass, pyname, attr);

        } else {
            Utility::FetchError(errors);
        }
    }

    if (!attr && (klass->fFlags & CPPScope::kIsNamespace)) {
    // refresh using list as necessary
        const std::vector<Cppyy::TCppScope_t>& uv = Cppyy::GetUsingNamespaces(klass->fCppType);
        if (!klass->fImp.fUsing || uv.size() != klass->fImp.fUsing->size()) {
            if (klass->fImp.fUsing) {
                for (auto pyref : *klass->fImp.fUsing) Py_DECREF(pyref);
                klass->fImp.fUsing->clear();
            } else
                klass->fImp.fUsing = new std::vector<PyObject*>;

        // reload and reset weak refs
            for (auto uid : uv) {
                PyObject* pyuscope = CreateScopeProxy(Cppyy::GetScopedFinalName(uid));
                if (pyuscope) {
                    klass->fImp.fUsing->push_back(PyWeakref_NewRef(pyuscope, nullptr));
                // the namespace may not otherwise be held, so tie the lifetimes
                    PyObject_SetAttr(pyclass, pyname, pyuscope);
                    Py_DECREF(pyuscope);
                }
            }
        }

    // try all outstanding using namespaces in turn to find the attribute (will cache
    // locally later; TODO: doing so may cause pathological cases)
        for (auto pyref : *klass->fImp.fUsing) {
            PyObject* pyuscope = PyWeakref_GetObject(pyref);
            if (pyuscope) {
                attr = PyObject_GetAttr(pyuscope, pyname);
                if (attr) break;
            }
        }
    }

    if (attr)
        std::for_each(errors.begin(), errors.end(), Utility::PyError_t::Clear);
    else {
    // not found: prepare a full error report
        PyObject* topmsg = nullptr;
        PyObject* sklass = PyObject_Str(pyclass);
        if (sklass) {
            topmsg = CPyCppyy_PyUnicode_FromFormat("%s has no attribute \'%s\'. Full details:",
                CPyCppyy_PyUnicode_AsString(sklass), CPyCppyy_PyUnicode_AsString(pyname));
            Py_DECREF(sklass);
        } else {
            topmsg = CPyCppyy_PyUnicode_FromFormat("no such attribute \'%s\'. Full details:",
                CPyCppyy_PyUnicode_AsString(pyname));
        }
        SetDetailedException(errors, topmsg /* steals */, PyExc_AttributeError /* default error */);
    }

    return attr;
}


//----------------------------------------------------------------------------
// p2.7 does not have a __dir__ in object, and object.__dir__ in p3 does not
// quite what I'd expected of it, so the following pulls in the internal code
#include "PyObjectDir27.inc"

static PyObject* meta_dir(CPPScope* klass)
{
// Collect a list of everything (currently) available in the namespace.
// The backend can filter by returning empty strings. Special care is
// taken for functions, which need not be unique (overloading).
    using namespace Cppyy;

    if ((void*)klass == (void*)&CPPInstance_Type)
        return PyList_New(0);

    if (!CPyCppyy::CPPScope_Check((PyObject*)klass)) {
        PyErr_SetString(PyExc_TypeError, "C++ proxy scope expected");
        return nullptr;
    }

    PyObject* dirlist = _generic_dir((PyObject*)klass);
    if (!(klass->fFlags & CPPScope::kIsNamespace))
        return dirlist;

    std::set<std::string> cppnames;
    Cppyy::GetAllCppNames(klass->fCppType, cppnames);

// get rid of duplicates
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(dirlist); ++i)
        cppnames.insert(CPyCppyy_PyUnicode_AsString(PyList_GET_ITEM(dirlist, i)));

    Py_DECREF(dirlist);
    dirlist = PyList_New(cppnames.size());

// copy total onto python list
    Py_ssize_t i = 0;
    for (const auto& name : cppnames) {
        PyList_SET_ITEM(dirlist, i++, CPyCppyy_PyUnicode_FromString(name.c_str()));
    }
    return dirlist;
}

//-----------------------------------------------------------------------------
static PyMethodDef meta_methods[] = {
    {(char*)"__dir__", (PyCFunction)meta_dir, METH_NOARGS, nullptr},
    {(char*)nullptr, nullptr, 0, nullptr}
};


//-----------------------------------------------------------------------------
static PyGetSetDef meta_getset[] = {
    {(char*)"__cppname__", (getter)meta_getcppname, nullptr, nullptr, nullptr},
    {(char*)"__module__",  (getter)meta_getmodule,  (setter)meta_setmodule, nullptr, nullptr},
    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}
};


//= CPyCppyy object proxy type type ==========================================
PyTypeObject CPPScope_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.CPPScope",       // tp_name
    sizeof(CPyCppyy::CPPScope),    // tp_basicsize
    0,                             // tp_itemsize
    0,                             // tp_dealloc
    0,                             // tp_print
    0,                             // tp_getattr
    0,                             // tp_setattr
    0,                             // tp_compare
    (reprfunc)meta_repr,           // tp_repr
    0,                             // tp_as_number
    0,                             // tp_as_sequence
    0,                             // tp_as_mapping
    0,                             // tp_hash
    0,                             // tp_call
    0,                             // tp_str
    (getattrofunc)meta_getattro,   // tp_getattro
    0,                             // tp_setattro
    0,                             // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     // tp_flags
    (char*)"CPyCppyy metatype (internal)",        // tp_doc
    0,                             // tp_traverse
    0,                             // tp_clear
    0,                             // tp_richcompare
    0,                             // tp_weaklistoffset
    0,                             // tp_iter
    0,                             // tp_iternext
    meta_methods,                  // tp_methods
    0,                             // tp_members
    meta_getset,                   // tp_getset
    &PyType_Type,                  // tp_base
    0,                             // tp_dict
    0,                             // tp_descr_get
    0,                             // tp_descr_set
    0,                             // tp_dictoffset
    0,                             // tp_init
    0,                             // tp_alloc
    (newfunc)pt_new,               // tp_new
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
};

} // namespace CPyCppyy
