// Bindings
#include "CPyCppyy.h"
#include "TemplateProxy.h"
#include "CPPClassMethod.h"
#include "CPPConstructor.h"
#include "CPPFunction.h"
#include "CPPMethod.h"
#include "CPPOverload.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "Utility.h"


namespace CPyCppyy {

//----------------------------------------------------------------------------
void TemplateProxy::Set(const std::string& cppname, const std::string& pyname, PyObject* pyclass)
{
// Initialize the proxy for the given 'pyclass.'
    fCppName      = CPyCppyy_PyUnicode_FromString(const_cast<char*>(cppname.c_str()));
    fPyName       = CPyCppyy_PyUnicode_FromString(const_cast<char*>(pyname.c_str()));
    fTemplateArgs = nullptr;
    Py_XINCREF(pyclass);
    fPyClass      = pyclass;
    fSelf         = nullptr;
    std::vector<PyCallable*> dummy;
    fNonTemplated = CPPOverload_New(pyname, dummy);
    fTemplated    = CPPOverload_New(pyname, dummy);
    new (&fDispatchMap) TP_DispatchMap_t{};
}

//----------------------------------------------------------------------------
void TemplateProxy::AddOverload(CPPOverload* mp) {
// Store overloads of this templated method.
    fNonTemplated->AddMethod(mp);
}

void TemplateProxy::AddOverload(PyCallable* pc) {
// Store overload of this templated method.
    fNonTemplated->AddMethod(pc);
}

void TemplateProxy::AddTemplate(PyCallable* pc)
{
// Store know template methods.
    fTemplated->AddMethod(pc);
}

//----------------------------------------------------------------------------
PyCallable* TemplateProxy::Instantiate(const std::string& fullname, PyObject* args, Utility::ArgPreference pref)
{
// Instantiate (and cache) templated methods, return method if any
    std::string proto = "";

    Py_ssize_t nArgs = PyTuple_GET_SIZE(args);
    if (nArgs != 0) {
        PyObject* tpArgs = PyTuple_New(nArgs);
        for (int i = 0; i < nArgs; ++i) {
            PyObject* itemi = PyTuple_GET_ITEM(args, i);

        // special case for arrays
            PyObject* pytc = PyObject_GetAttr(itemi, PyStrings::gTypeCode);
            if (!(pytc && CPyCppyy_PyUnicode_Check(pytc))) {
            // normal case (not an array)
                PyErr_Clear();
                PyObject* tp = (PyObject*)Py_TYPE(itemi);
                Py_INCREF(tp);
                PyTuple_SET_ITEM(tpArgs, i, tp);
            } else {
            // array, build up a pointer type
                char tc = ((char*)CPyCppyy_PyUnicode_AsString(pytc))[0];
                const char* ptrname = 0;
                switch (tc) {
                    case 'b': ptrname = "char*";           break;
                    case 'h': ptrname = "short*";          break;
                    case 'H': ptrname = "unsigned short*"; break;
                    case 'i': ptrname = "int*";            break;
                    case 'I': ptrname = "unsigned int*";   break;
                    case 'l': ptrname = "long*";           break;
                    case 'L': ptrname = "unsigned long*";  break;
                    case 'f': ptrname = "float*";          break;
                    case 'd': ptrname = "double*";         break;
                    default:  ptrname = "void*";  // TODO: verify if this is right
                }
                if (ptrname) {
                    PyObject* pyptrname = CPyCppyy_PyUnicode_FromString(ptrname);
                    PyTuple_SET_ITEM(tpArgs, i, pyptrname);
                // string added, but not counted towards nStrings
                } else {
                // this should cleanly fail instantiation
                    Py_INCREF(Py_None);
                    PyTuple_SET_ITEM(tpArgs, i, Py_None);
                }
            }
            Py_XDECREF(pytc);
        }

        const std::string& name_v1 = Utility::ConstructTemplateArgs(nullptr, tpArgs, args, pref);
        Py_DECREF(tpArgs);
        if (name_v1.size())
            proto = name_v1.substr(1, name_v1.size()-2);
    }

// the following causes instantiation as necessary
    Cppyy::TCppScope_t scope = ((CPPClass*)fPyClass)->fCppType;
    Cppyy::TCppMethod_t cppmeth = Cppyy::GetMethodTemplate(scope, fullname, proto);
    if (cppmeth) {    // overload stops here
        PyCallable* meth = nullptr;
        if (Cppyy::IsNamespace(scope))
            meth = new CPPFunction(scope, cppmeth);
        else if (Cppyy::IsStaticMethod(cppmeth))
            meth = new CPPClassMethod(scope, cppmeth);
        else if (Cppyy::IsConstructor(cppmeth))
            meth = new CPPConstructor(scope, cppmeth);
        else
            meth = new CPPMethod(scope, cppmeth);

    // add to overload of instantiated templates
        AddTemplate(meth);

        return meth;
    }

    return nullptr;
}


//= CPyCppyy template proxy construction/destruction =========================
static TemplateProxy* tpp_new(PyTypeObject*, PyObject*, PyObject*)
{
// Create a new empty template method proxy.
    TemplateProxy* pytmpl = PyObject_GC_New(TemplateProxy, &TemplateProxy_Type);
    pytmpl->fCppName      = nullptr;
    pytmpl->fPyName       = nullptr;
    pytmpl->fTemplateArgs = nullptr;
    pytmpl->fPyClass      = nullptr;
    pytmpl->fSelf         = nullptr;
    pytmpl->fNonTemplated = nullptr;
    pytmpl->fTemplated    = nullptr;
    pytmpl->fWeakrefList  = nullptr;

    PyObject_GC_Track(pytmpl);
    return pytmpl;
}

//----------------------------------------------------------------------------
static int tpp_clear(TemplateProxy* pytmpl)
{
// Garbage collector clear of held python member objects.
    Py_CLEAR(pytmpl->fCppName);
    Py_CLEAR(pytmpl->fPyName);
    Py_CLEAR(pytmpl->fTemplateArgs);
    Py_CLEAR(pytmpl->fPyClass);
    Py_CLEAR(pytmpl->fSelf);
    Py_CLEAR(pytmpl->fNonTemplated);
    Py_CLEAR(pytmpl->fTemplated);

    return 0;
}

//----------------------------------------------------------------------------
static void tpp_dealloc(TemplateProxy* pytmpl)
{
// Destroy the given template method proxy.
    if (pytmpl->fWeakrefList)
        PyObject_ClearWeakRefs((PyObject*)pytmpl);
    PyObject_GC_UnTrack(pytmpl);
    tpp_clear(pytmpl);
    for (const auto& p : pytmpl->fDispatchMap)
        Py_DECREF(p.second);
    pytmpl->fDispatchMap.~TP_DispatchMap_t();
    PyObject_GC_Del(pytmpl);
}

//----------------------------------------------------------------------------
static PyObject* tpp_doc(TemplateProxy* pytmpl, void*)
{
// Forward to method proxies to doc all overloads
    PyObject* doc = nullptr;
    if (pytmpl->fNonTemplated)
        doc = PyObject_GetAttrString((PyObject*)pytmpl->fNonTemplated, "__doc__");
    if (pytmpl->fTemplated) {
        PyObject* doc2 = PyObject_GetAttrString((PyObject*)pytmpl->fTemplated, "__doc__");
        if (doc && doc2) {
            CPyCppyy_PyUnicode_AppendAndDel(&doc, CPyCppyy_PyUnicode_FromString("\n"));
            CPyCppyy_PyUnicode_AppendAndDel(&doc, doc2);
        } else if (!doc && doc2) {
            doc = doc2;
        }
    }

    if (doc)
        return doc;

    return CPyCppyy_PyUnicode_FromString(TemplateProxy_Type.tp_doc);
}

//----------------------------------------------------------------------------
static int tpp_traverse(TemplateProxy* pytmpl, visitproc visit, void* arg)
{
// Garbage collector traverse of held python member objects.
    Py_VISIT(pytmpl->fCppName);
    Py_VISIT(pytmpl->fPyName);
    Py_VISIT(pytmpl->fTemplateArgs);
    Py_VISIT(pytmpl->fPyClass);
    Py_VISIT(pytmpl->fSelf);
    Py_VISIT(pytmpl->fNonTemplated);
    Py_VISIT(pytmpl->fTemplated);

    return 0;
}

//= CPyCppyy template proxy callable behavior ================================
static PyObject* tpp_call(TemplateProxy* pytmpl, PyObject* args, PyObject* kwds)
{
// Dispatcher to the actual member method, several uses possible; in order:
//
// case 1: explicit template previously selected through subscript
//
// case 2: select known non-template overload
//
//    obj.method(a0, a1, ...)
//       => obj->method(a0, a1, ...)        // non-template
//
// case 3: select known template overload
//
//    obj.method(a0, a1, ...)
//       => obj->method(a0, a1, ...)        // all known templates
//
// case 4: auto-instantiation from types of arguments
//
//    obj.method(a0, a1, ...)
//       => obj->method<type(a0), type(a1), ...>(a0, a1, ...)
//
// Note: explicit instantiation needs to use [] syntax:
//
//    obj.method[type<a0>, type<a1>, ...](a0, a1, ...)
//

// TODO: should previously instantiated templates be considered first?

// short-cut through memoization map
    uint64_t sighash = HashSignature(args);

// look for known signatures ...
    CPPOverload* ol = nullptr;
    for (const auto& p : pytmpl->fDispatchMap) {
        if (p.first == sighash) {
            ol = p.second;
            break;
        }
    }

    if (ol != nullptr) {
        PyObject* result = nullptr;
        if (!pytmpl->fSelf) {
            result = CPPOverload_Type.tp_call((PyObject*)ol, args, kwds);
        } else {
            PyObject* pymeth = CPPOverload_Type.tp_descr_get(
                (PyObject*)ol, pytmpl->fSelf, (PyObject*)&CPPOverload_Type);
            result = CPPOverload_Type.tp_call(pymeth, args, kwds);
            Py_DECREF(pymeth);
        }
        if (result)
            return result;
        PyErr_Clear();
    }

// do not mix template instantiations with implicit conversions
    if (!kwds) {
        kwds = PyDict_New();
    } else {
        Py_INCREF(kwds);
    }

// case 1: explicit template previously selected through subscript

    if (pytmpl->fTemplateArgs) {
    // instantiate explicitly
        PyObject* pyfullname = CPyCppyy_PyUnicode_FromString(
            CPyCppyy_PyUnicode_AsString(pytmpl->fCppName));
        CPyCppyy_PyUnicode_Append(&pyfullname, pytmpl->fTemplateArgs);
        PyCallable* meth = pytmpl->Instantiate(
            CPyCppyy_PyUnicode_AsString(pyfullname), args, Utility::kNone);
        if (meth) {
        // store overload
            PyObject* dct = PyObject_GetAttr(pytmpl->fPyClass, PyStrings::gDict);
            if (dct) {
                PyObject* attr = PyObject_GetItem(dct, pyfullname);
                Py_DECREF(dct);
                if (CPPOverload_Check(attr)) {
                    ((CPPOverload*)attr)->AddMethod(meth);
                    meth = nullptr;
                } else
                    PyErr_Clear();
                Py_XDECREF(attr);
            }

            if (meth) { // meaning, wasn't stored
                PyObject* pymeth = (PyObject*)CPPOverload_New(CPyCppyy_PyUnicode_AsString(pyfullname), meth);
                PyObject_SetAttr(pytmpl->fPyClass, pyfullname, pymeth);
                Py_DECREF(pymeth);
            }

        // retrieve fresh (for boundedness) and call
            bool isNS = (((CPPScope*)pytmpl->fPyClass)->fFlags & CPPScope::kIsNamespace);
            PyObject* pymeth = PyObject_GetAttr((pytmpl->fSelf && !isNS) ? pytmpl->fSelf : pytmpl->fPyClass, pyfullname);
            Py_DECREF(pyfullname);

            PyObject* result;
            PyDict_SetItem(kwds, PyStrings::gNoImplicit, Py_True);
            if (isNS && pytmpl->fSelf) {
            // this is a global method added a posteriori to the class
                 Py_ssize_t sz = PyTuple_GET_SIZE(args);
                 PyObject* newArgs = PyTuple_New(sz+1);
                 for (int i = 0; i < sz; ++i) {
                     PyObject* item = PyTuple_GET_ITEM(args, i);
                     Py_INCREF(item);
                     PyTuple_SET_ITEM(newArgs, i+1, item);
                 }
                 PyTuple_SET_ITEM(newArgs, 0, (PyObject*)pytmpl->fSelf);
                 result = CPPOverload_Type.tp_call(pymeth, newArgs, kwds);
                 Py_DECREF(newArgs);
            } else
                 result = CPPOverload_Type.tp_call(pymeth, args, kwds);
            if (result) {
                pytmpl->fDispatchMap.push_back(std::make_pair(sighash, (CPPOverload*)pymeth));
            } else { Py_DECREF(pymeth); }
            Py_DECREF(kwds);
            return result;
        } else {
            Py_DECREF(pyfullname);
        // debatable ... should this drop through?
        }
    }

// case 2: select known non-template overload

// simply forward the call: all non-templated methods are defined on class definition
// and thus already available
    PyObject* pymeth = CPPOverload_Type.tp_descr_get(
        (PyObject*)pytmpl->fNonTemplated, pytmpl->fSelf, (PyObject*)&CPPOverload_Type);
// now call the method with the arguments (loops internally)
    PyDict_SetItem(kwds, PyStrings::gNoImplicit, Py_True);
    PyObject* result = CPPOverload_Type.tp_call(pymeth, args, kwds);
    Py_DECREF(pymeth); pymeth = nullptr;
    if (result) {
        Py_INCREF(pytmpl->fNonTemplated);
        pytmpl->fDispatchMap.push_back(std::make_pair(sighash, pytmpl->fNonTemplated));
        Py_DECREF(kwds);
        return result;
    }
// TODO: collect error here, as the failure may be either an overload
// failure after which we should continue; or a real failure, which should
// be reported.
    PyErr_Clear();

// case 3: select known template overload
    pymeth = CPPOverload_Type.tp_descr_get(
        (PyObject*)pytmpl->fTemplated, pytmpl->fSelf, (PyObject*)&CPPOverload_Type);
// now call the method with the arguments (loops internally)
    PyDict_SetItem(kwds, PyStrings::gNoImplicit, Py_True);
    result = CPPOverload_Type.tp_call(pymeth, args, kwds);
    Py_DECREF(pymeth); pymeth = nullptr;
    if (result) {
        Py_INCREF(pytmpl->fTemplated);
        pytmpl->fDispatchMap.push_back(std::make_pair(sighash, pytmpl->fTemplated));
        Py_DECREF(kwds);
        return result;
    }
// TODO: collect error here, as the failure may be either an overload
// failure after which we should continue; or a real failure, which should
// be reported.
    PyErr_Clear();

// case 4: auto-instantiation from types of arguments
    for (auto pref : {Utility::kReference, Utility::kPointer, Utility::kValue}) {
        // TODO: no need to loop if there are no non-instance arguments; also, should any
        // failed lookup se removed?
        PyCallable* meth = pytmpl->Instantiate(
            CPyCppyy_PyUnicode_AsString(pytmpl->fCppName), args, pref);
        if (meth) {
        // re-retrieve the cached method to bind it, then call it
            PyObject* pymeth = CPPOverload_Type.tp_descr_get(
                (PyObject*)pytmpl->fTemplated, pytmpl->fSelf, (PyObject*)&CPPOverload_Type);
            PyDict_SetItem(kwds, PyStrings::gNoImplicit, Py_True);
            result = CPPOverload_Type.tp_call(pymeth, args, kwds);
            Py_DECREF(pymeth);
            if (result) {
                Py_INCREF(pytmpl->fTemplated);
                pytmpl->fDispatchMap.push_back(std::make_pair(sighash, pytmpl->fTemplated));
                Py_DECREF(kwds);
                return result;
            } else
                PyErr_Clear();
        }
    }

// moderately generic error message, but should be clear enough
    PyErr_Format(PyExc_TypeError, "cannot resolve method template call for \'%s\'",
        CPyCppyy_PyUnicode_AsString(pytmpl->fPyName));
    Py_DECREF(kwds);
    return nullptr;
}

//----------------------------------------------------------------------------
static TemplateProxy* tpp_descrget(TemplateProxy* pytmpl, PyObject* pyobj, PyObject*)
{
// create and use a new template proxy (language requirement)
    TemplateProxy* newPyTmpl = (TemplateProxy*)TemplateProxy_Type.tp_alloc(&TemplateProxy_Type, 0);

// new method is to be bound to current object (may be nullptr)
    Py_XINCREF(pyobj);
    newPyTmpl->fSelf = pyobj;

// copy name and class pointers
    Py_INCREF(pytmpl->fCppName);
    newPyTmpl->fCppName = pytmpl->fCppName;

    Py_INCREF(pytmpl->fPyName);
    newPyTmpl->fPyName = pytmpl->fPyName;

    Py_XINCREF(pytmpl->fTemplateArgs);
    newPyTmpl->fTemplateArgs = pytmpl->fTemplateArgs;

    Py_XINCREF(pytmpl->fPyClass);
    newPyTmpl->fPyClass = pytmpl->fPyClass;

// copy non-templated method proxy pointer
    Py_INCREF(pytmpl->fNonTemplated);
    newPyTmpl->fNonTemplated = pytmpl->fNonTemplated;

// copy templated method proxy pointer
    Py_INCREF(pytmpl->fTemplated);
    newPyTmpl->fTemplated = pytmpl->fTemplated;

    return newPyTmpl;
}

//----------------------------------------------------------------------------
static PyObject* tpp_subscript(TemplateProxy* pytmpl, PyObject* args)
{
// Explicit template member lookup/instantiation.
    PyObject* newArgs;
    if (!PyTuple_Check(args)) {
        newArgs = PyTuple_New(1);
        Py_INCREF(args);
        PyTuple_SET_ITEM(newArgs, 0, args);
    } else {
        Py_INCREF(args);
        newArgs = args;
    }

    PyObject* pymeth = nullptr;

// construct full, explicit name of function
    PyObject* pyfullname = CPyCppyy_PyUnicode_FromString(CPyCppyy_PyUnicode_AsString(pytmpl->fCppName));
    PyObject* tmpl_args = CPyCppyy_PyUnicode_FromString(Utility::ConstructTemplateArgs(nullptr, newArgs).c_str());
    Py_DECREF(newArgs);
    CPyCppyy_PyUnicode_Append(&pyfullname, tmpl_args);

// find template cached in dictionary, if any
    PyObject* dct = PyObject_GetAttr(pytmpl->fPyClass, PyStrings::gDict);
    bool hasTmpl = dct ? false : (bool)PyDict_GetItem(dct, pyfullname);
    Py_XDECREF(dct);
    if (hasTmpl) {    // overloads stop here, as there is an explicit match
         bool use_self = pytmpl->fSelf && !(((CPPScope*)pytmpl->fPyClass)->fFlags & CPPScope::kIsNamespace);
         pymeth = PyObject_GetAttr(use_self ? pytmpl->fSelf : pytmpl->fPyClass, pyfullname);
    }
    Py_DECREF(pyfullname);

// if found, return the overload, otherwise return fresh
    if (pymeth) {
        Py_DECREF(tmpl_args);
        return pymeth;
    }

// nothing found, return fresh template trampoline with constructed types
    TemplateProxy* typeBoundMethod = tpp_descrget(pytmpl, pytmpl->fSelf, nullptr);
    Py_XDECREF(typeBoundMethod->fTemplateArgs);
    typeBoundMethod->fTemplateArgs = tmpl_args;
    return (PyObject*)typeBoundMethod;
}

//-----------------------------------------------------------------------------
static PyObject* tpp_getuseffi(CPPOverload*, void*)
{   
    return PyInt_FromLong(0); // dummy (__useffi__ unused)
}   
    
//-----------------------------------------------------------------------------
static int tpp_setuseffi(CPPOverload*, PyObject*, void*)
{   
    return 0;                 // dummy (__useffi__ unused)
}   


//----------------------------------------------------------------------------
static PyMappingMethods tpp_as_mapping = {
    nullptr, (binaryfunc)tpp_subscript, nullptr
};

static PyGetSetDef tpp_getset[] = {
    {(char*)"__doc__", (getter)tpp_doc, nullptr, nullptr, nullptr},
    {(char*)"__useffi__", (getter)tpp_getuseffi, (setter)tpp_setuseffi,
      (char*)"unused", nullptr},
    {(char*)nullptr,   nullptr,         nullptr, nullptr, nullptr}
};


//= CPyCppyy template proxy type =============================================
PyTypeObject TemplateProxy_Type = {
   PyVarObject_HEAD_INIT(&PyType_Type, 0)
   (char*)"cppyy.TemplateProxy", // tp_name
   sizeof(TemplateProxy),     // tp_basicsize
   0,                         // tp_itemsize
   (destructor)tpp_dealloc,   // tp_dealloc
   0,                         // tp_print
   0,                         // tp_getattr
   0,                         // tp_setattr
   0,                         // tp_compare
   0,                         // tp_repr
   0,                         // tp_as_number
   0,                         // tp_as_sequence
   &tpp_as_mapping,           // tp_as_mapping
   0,                         // tp_hash
   (ternaryfunc)tpp_call,     // tp_call
   0,                         // tp_str
   0,                         // tp_getattro
   0,                         // tp_setattro
   0,                         // tp_as_buffer
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,      // tp_flags
   (char*)"cppyy template proxy (internal)",     // tp_doc
   (traverseproc)tpp_traverse,// tp_traverse
   (inquiry)tpp_clear,        // tp_clear
   0,                         // tp_richcompare
   offsetof(TemplateProxy, fWeakrefList),        // tp_weaklistoffset
   0,                         // tp_iter
   0,                         // tp_iternext
   0,                         // tp_methods
   0,                         // tp_members
   tpp_getset,                // tp_getset
   0,                         // tp_base
   0,                         // tp_dict
   (descrgetfunc)tpp_descrget,// tp_descr_get
   0,                         // tp_descr_set
   0,                         // tp_dictoffset
   0,                         // tp_init
   0,                         // tp_alloc
   (newfunc)tpp_new,          // tp_new
   0,                         // tp_free
   0,                         // tp_is_gc
   0,                         // tp_bases
   0,                         // tp_mro
   0,                         // tp_cache
   0,                         // tp_subclasses
   0                          // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
   , 0                        // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
   , 0                        // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
   , 0                        // tp_finalize
#endif
};

} // namespace CPyCppyy
