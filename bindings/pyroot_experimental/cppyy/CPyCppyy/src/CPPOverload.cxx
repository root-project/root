// Bindings
#include "CPyCppyy.h"
#include "structmember.h"    // from Python
#if PY_VERSION_HEX >= 0x02050000
#include "code.h"            // from Python
#else
#include "compile.h"         // from Python
#endif
#ifndef CO_NOFREE
// python2.2 does not have CO_NOFREE defined
#define CO_NOFREE       0x0040
#endif
#include "CPPOverload.h"
#include "CPPInstance.h"
#include "CallContext.h"
#include "PyStrings.h"
#include "Utility.h"

// Standard
#include <algorithm>
#include <sstream>
#include <vector>


namespace CPyCppyy {

namespace {

// from CPython's instancemethod: Free list for method objects to safe malloc/free overhead
// The im_self element is used to chain the elements.
static CPPOverload* free_list;
static int numfree = 0;
#ifndef CPPOverload_MAXFREELIST
#define CPPOverload_MAXFREELIST 32
#endif


// TODO: only used here, but may be better off integrated with Pythonize.cxx callbacks
class TPythonCallback : public PyCallable {
public:
    PyObject* fCallable;

    TPythonCallback(PyObject* callable) : fCallable(nullptr)
    {
        if (!PyCallable_Check(callable)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return;
        }
        Py_INCREF(callable);
        fCallable = callable;
    }

    virtual ~TPythonCallback() {
        Py_DECREF(fCallable);
        fCallable = nullptr;
    }

    virtual PyObject* GetSignature(bool /*show_formalargs*/ = true) {
        return CPyCppyy_PyUnicode_FromString("*args, **kwargs");
    }
    virtual PyObject* GetPrototype(bool /*show_formalargs*/ = true) {
        return CPyCppyy_PyUnicode_FromString("<callback>");
    }
    virtual PyObject* GetDocString() {
        if (PyObject_HasAttrString(fCallable, "__doc__")) {
            return PyObject_GetAttrString(fCallable, "__doc__");
        } else {
            return GetPrototype();
        }
    }

    virtual int GetPriority() { return 100; };
    virtual bool IsGreedy() { return false; };

    virtual int GetMaxArgs() { return 100; };
    virtual PyObject* GetCoVarNames() { // TODO: pick these up from the callable
        Py_RETURN_NONE;
    }
    virtual PyObject* GetArgDefault(int /* iarg */) { // TODO: pick these up from the callable
        Py_RETURN_NONE;
    }

    virtual PyObject* GetScopeProxy() { // should this be the module ??
        Py_RETURN_NONE;
    }

    virtual Cppyy::TCppFuncAddr_t GetFunctionAddress() {
        return (Cppyy::TCppFuncAddr_t)nullptr;
    }

    virtual PyCallable* Clone() { return new TPythonCallback(*this); }

    virtual PyObject* Call(
            CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* /* ctxt = 0 */) {

        PyObject* newArgs = nullptr;
        if (self) {
            Py_ssize_t nargs = PyTuple_Size(args);
            newArgs = PyTuple_New(nargs+1);
            Py_INCREF(self);
            PyTuple_SET_ITEM(newArgs, 0, (PyObject*)self);
            for (Py_ssize_t iarg = 0; iarg < nargs; ++iarg) {
                PyObject* pyarg = PyTuple_GET_ITEM(args, iarg);
                Py_INCREF(pyarg);
                PyTuple_SET_ITEM(newArgs, iarg+1, pyarg);
            }
        } else {
            Py_INCREF(args);
            newArgs = args;
        }
        return PyObject_Call(fCallable, newArgs, kwds);
    }
};

// helper to test whether a method is used in a pseudo-function modus
static inline bool IsPseudoFunc(CPPOverload* pymeth)
{
    return pymeth->fMethodInfo->fFlags & CallContext::kIsPseudoFunc;
}

// helper to sort on method priority
static int PriorityCmp(PyCallable* left, PyCallable* right)
{
    return left->GetPriority() > right->GetPriority();
}

// return helper
static inline void ResetCallState(CPPInstance*& selfnew, CPPInstance* selfold, bool clear)
{
    if (selfnew != selfold) {
        Py_XDECREF(selfnew);
        selfnew = selfold;
    }

    if (clear)
        PyErr_Clear();
}

// helper to factor out return logic of mp_call
static inline PyObject* HandleReturn(
    CPPOverload* pymeth, CPPInstance* oldSelf, PyObject* result)
{
// special case for python exceptions, propagated through C++ layer
    if (result) {

    // if this method creates new objects, always take ownership
        if (IsCreator(pymeth->fMethodInfo->fFlags)) {

        // either be a constructor with a fresh object proxy self ...
            if (IsConstructor(pymeth->fMethodInfo->fFlags)) {
                if (pymeth->fSelf)
                    pymeth->fSelf->PythonOwns();
            }

        // ... or be a method with an object proxy return value
            else if (CPPInstance_Check(result))
                ((CPPInstance*)result)->PythonOwns();
        }

    // if this new object falls inside self, make sure its lifetime is proper
        if (CPPInstance_Check(pymeth->fSelf) && CPPInstance_Check(result)) {
            ptrdiff_t offset = (ptrdiff_t)(
                (CPPInstance*)result)->GetObject() - (ptrdiff_t)pymeth->fSelf->GetObject();
            if (0 <= offset && offset < (ptrdiff_t)Cppyy::SizeOf(pymeth->fSelf->ObjectIsA())) {
                if (PyObject_SetAttr(result, PyStrings::gLifeLine, (PyObject*)pymeth->fSelf) == -1)
                    PyErr_Clear();     // ignored
            }
        }
    }

// reset self as necessary to allow re-use of the CPPOverload
    ResetCallState(pymeth->fSelf, oldSelf, false);

    return result;
}


//= CPyCppyy method proxy object behaviour ===================================
static PyObject* mp_name(CPPOverload* pymeth, void*)
{
    return CPyCppyy_PyUnicode_FromString(pymeth->GetName().c_str());
}

//-----------------------------------------------------------------------------
static PyObject* mp_module(CPPOverload* /* pymeth */, void*)
{
    Py_INCREF(PyStrings::gThisModule);
    return PyStrings::gThisModule;
}

//-----------------------------------------------------------------------------
static PyObject* mp_doc(CPPOverload* pymeth, void*)
{
// Build python document string ('__doc__') from all C++-side overloads.
    CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

// collect doc strings
    CPPOverload::Methods_t::size_type nMethods = methods.size();
    if (nMethods == 0)       // from template proxy with no instantiations
        return nullptr;
    PyObject* doc = methods[0]->GetDocString();

// simple case
    if (nMethods == 1)
        return doc;

// overloaded method
    PyObject* separator = CPyCppyy_PyUnicode_FromString("\n");
    for (CPPOverload::Methods_t::size_type i = 1; i < nMethods; ++i) {
        CPyCppyy_PyUnicode_Append(&doc, separator);
        CPyCppyy_PyUnicode_AppendAndDel(&doc, methods[i]->GetDocString());
    }
    Py_DECREF(separator);

    return doc;
}

//-----------------------------------------------------------------------------
static PyObject* mp_meth_func(CPPOverload* pymeth, void*)
{
// Create a new method proxy to be returned.
    CPPOverload* newPyMeth = (CPPOverload*)CPPOverload_Type.tp_alloc(&CPPOverload_Type, 0);

// method info is shared, as it contains the collected overload knowledge
    *pymeth->fMethodInfo->fRefCount += 1;
    newPyMeth->fMethodInfo = pymeth->fMethodInfo;

// new method is unbound, track whether this proxy is used in the capacity of a
// method or a function (which normally is a CPPFunction)
    newPyMeth->fMethodInfo->fFlags |= CallContext::kIsPseudoFunc;

    return (PyObject*)newPyMeth;
}

//-----------------------------------------------------------------------------
static PyObject* mp_meth_self(CPPOverload* pymeth, void*)
{
// Return the bound self, if any; in case of pseudo-function role, pretend
// that the data member im_self does not exist.
    if (IsPseudoFunc(pymeth)) {
        PyErr_Format(PyExc_AttributeError,
            "function %s has no attribute \'im_self\'", pymeth->fMethodInfo->fName.c_str());
        return nullptr;
    } else if (pymeth->fSelf != 0) {
        Py_INCREF((PyObject*)pymeth->fSelf);
        return (PyObject*)pymeth->fSelf;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject* mp_meth_class(CPPOverload* pymeth, void*)
{
// Return scoping class; in case of pseudo-function role, pretend that there
// is no encompassing class (i.e. global scope).
    if (!IsPseudoFunc(pymeth) && pymeth->fMethodInfo->fMethods.size()) {
        PyObject* pyclass = pymeth->fMethodInfo->fMethods[0]->GetScopeProxy();
        if (!pyclass)
            PyErr_Format(PyExc_AttributeError,
                "function %s has no attribute \'im_class\'", pymeth->fMethodInfo->fName.c_str());
        return pyclass;
    }

    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject* mp_func_closure(CPPOverload* /* pymeth */, void*)
{
// Stub only, to fill out the python function interface.
    Py_RETURN_NONE;
}

//-----------------------------------------------------------------------------
static PyObject* mp_func_code(CPPOverload* pymeth, void*)
{
// Code details are used in module inspect to fill out interactive help()
#if PY_VERSION_HEX < 0x03000000
    CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

// collect arguments only if there is just 1 overload, otherwise put in a
// fake *args (see below for co_varnames)
    PyObject* co_varnames = methods.size() == 1 ? methods[0]->GetCoVarNames() : nullptr;
    if (!co_varnames) {
    // TODO: static methods need no 'self' (but is harmless otherwise)
        co_varnames = PyTuple_New(1 /* self */ + 1 /* fake */);
        PyTuple_SET_ITEM(co_varnames, 0, CPyCppyy_PyUnicode_FromString("self"));
        PyTuple_SET_ITEM(co_varnames, 1, CPyCppyy_PyUnicode_FromString("*args"));
    }

    int co_argcount = (int)PyTuple_Size(co_varnames);

// for now, code object representing the statement 'pass'
    PyObject* co_code = PyString_FromStringAndSize("d\x00\x00S", 4);

// tuples with all the const literals used in the function
    PyObject* co_consts = PyTuple_New(0);
    PyObject* co_names = PyTuple_New(0);

// names, freevars, and cellvars go unused
    PyObject* co_unused = PyTuple_New(0);

// filename is made-up
    PyObject* co_filename = PyString_FromString("cppyy.py");

// name is the function name, also through __name__ on the function itself
    PyObject* co_name = PyString_FromString(pymeth->GetName().c_str());

// firstlineno is the line number of first function code in the containing scope

// lnotab is a packed table that maps instruction count and line number
    PyObject* co_lnotab = PyString_FromString("\x00\x01\x0c\x01");

    PyObject* code = (PyObject*)PyCode_New(
        co_argcount,                             // argcount
        co_argcount+1,                           // nlocals
        2,                                       // stacksize
        CO_OPTIMIZED | CO_NEWLOCALS | CO_NOFREE, // flags
        co_code,                                 // code
        co_consts,                               // consts
        co_names,                                // names
        co_varnames,                             // varnames
        co_unused,                               // freevars
        co_unused,                               // cellvars
        co_filename,                             // filename
        co_name,                                 // name
        1,                                       // firstlineno
        co_lnotab);                              // lnotab

    Py_DECREF(co_lnotab);
    Py_DECREF(co_name);
    Py_DECREF(co_unused);
    Py_DECREF(co_filename);
    Py_DECREF(co_varnames);
    Py_DECREF(co_names);
    Py_DECREF(co_consts);
    Py_DECREF(co_code);

    return code;
#else
// not important for functioning of most code, so not implemented for p3 for now (TODO)
    pymeth = 0;
    Py_RETURN_NONE;
#endif
}

//-----------------------------------------------------------------------------
static PyObject* mp_func_defaults(CPPOverload* pymeth, void*)
{
// Create a tuple of default values, if there is only one method (otherwise
// leave undefined: this is only used by inspect for interactive help())
    CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

    if (methods.size() != 1)
        return PyTuple_New(0);

    int maxarg = methods[0]->GetMaxArgs();

    PyObject* defaults = PyTuple_New(maxarg);

    int itup = 0;
    for (int iarg = 0; iarg < maxarg; ++iarg) {
        PyObject* defvalue = methods[0]->GetArgDefault(iarg);
        if (defvalue)
            PyTuple_SET_ITEM(defaults, itup++, defvalue);
    }
    _PyTuple_Resize(&defaults, itup);

    return defaults;
}

//-----------------------------------------------------------------------------
static PyObject* mp_func_globals(CPPOverload* /* pymeth */, void*)
{
// Return this function's global dict (hard-wired to be the cppyy module); used
// for lookup of names from co_code indexing into co_names.
    PyObject* pyglobal = PyModule_GetDict(PyImport_AddModule((char*)"cppyy"));
    Py_XINCREF(pyglobal);
    return pyglobal;
}

//-----------------------------------------------------------------------------
PyObject* mp_getcreates(CPPOverload* pymeth, void*)
{
// Get '__creates__' boolean, which determines ownership of return values.
    return PyInt_FromLong((long)IsCreator(pymeth->fMethodInfo->fFlags));
}

//-----------------------------------------------------------------------------
static int mp_setcreates(CPPOverload* pymeth, PyObject* value, void*)
{
// Set '__creates__' boolean, which determines ownership of return values.
    if (!value) {        // means that __creates__ is being deleted
        pymeth->fMethodInfo->fFlags &= ~CallContext::kIsCreator;
        return 0;
    }

    long iscreator = PyLong_AsLong(value);
    if (iscreator == -1 && PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "a boolean 1 or 0 is required for __creates__");
        return -1;
    }

    if (iscreator)
        pymeth->fMethodInfo->fFlags |= CallContext::kIsCreator;
    else
        pymeth->fMethodInfo->fFlags &= ~CallContext::kIsCreator;

    return 0;
}

//-----------------------------------------------------------------------------
static PyObject* mp_getmempolicy(CPPOverload* pymeth, void*)
{
// Get '_mempolicy' enum, which determines ownership of call arguments.
    if (pymeth->fMethodInfo->fFlags & CallContext::kUseHeuristics)
        return PyInt_FromLong(CallContext::kUseHeuristics);

    if (pymeth->fMethodInfo->fFlags & CallContext::kUseStrict)
        return PyInt_FromLong(CallContext::kUseStrict);

    return PyInt_FromLong(-1);
}

//-----------------------------------------------------------------------------
static int mp_setmempolicy(CPPOverload* pymeth, PyObject* value, void*)
{
// Set '_mempolicy' enum, which determines ownership of call arguments.
    long mempolicy = PyLong_AsLong(value);
    if (mempolicy == CallContext::kUseHeuristics) {
        pymeth->fMethodInfo->fFlags |= CallContext::kUseHeuristics;
        pymeth->fMethodInfo->fFlags &= ~CallContext::kUseStrict;
    } else if (mempolicy == CallContext::kUseStrict) {
        pymeth->fMethodInfo->fFlags |= CallContext::kUseStrict;
        pymeth->fMethodInfo->fFlags &= ~CallContext::kUseHeuristics;
    } else {
        PyErr_SetString(PyExc_ValueError,
            "expected kMemoryStrict or kMemoryHeuristics as value for __mempolicy__");
        return -1;
    }

    return 0;
}

//-----------------------------------------------------------------------------
static PyObject* mp_getthreaded(CPPOverload* pymeth, void*)
{
// Get '_threaded' boolean, which determines whether the GIL will be released.
    return PyInt_FromLong(
        (long)(pymeth->fMethodInfo->fFlags & CallContext::kReleaseGIL));
}

//-----------------------------------------------------------------------------
static int mp_setthreaded(CPPOverload* pymeth, PyObject* value, void*)
{
// Set '_threaded' boolean, which determines whether the GIL will be released.
    long isthreaded = PyLong_AsLong(value);
    if (isthreaded == -1 && PyErr_Occurred()) {
        PyErr_SetString(PyExc_ValueError, "a boolean 1 or 0 is required for __release_gil__");
        return -1;
    }

    if (isthreaded)
        pymeth->fMethodInfo->fFlags |= CallContext::kReleaseGIL;
    else
        pymeth->fMethodInfo->fFlags &= ~CallContext::kReleaseGIL;

    return 0;
}

//-----------------------------------------------------------------------------
static PyObject* mp_getuseffi(CPPOverload*, void*)
{
    return PyInt_FromLong(0); // dummy (__useffi__ unused)
}

//-----------------------------------------------------------------------------
static int mp_setuseffi(CPPOverload*, PyObject*, void*)
{
    return 0;                 // dummy (__useffi__ unused)
}


//-----------------------------------------------------------------------------
static PyGetSetDef mp_getset[] = {
    {(char*)"__name__",   (getter)mp_name,   nullptr, nullptr, nullptr},
    {(char*)"__module__", (getter)mp_module, nullptr, nullptr, nullptr},
    {(char*)"__doc__",    (getter)mp_doc,    nullptr, nullptr, nullptr},

// to be more python-like, where these are duplicated as well; to actually
// derive from the python method or function type is too memory-expensive,
// given that most of the members of those types would not be used
    {(char*)"im_func",  (getter)mp_meth_func,  nullptr, nullptr, nullptr},
    {(char*)"im_self",  (getter)mp_meth_self,  nullptr, nullptr, nullptr},
    {(char*)"im_class", (getter)mp_meth_class, nullptr, nullptr, nullptr},

    {(char*)"func_closure",  (getter)mp_func_closure,  nullptr, nullptr, nullptr},
    {(char*)"func_code",     (getter)mp_func_code,     nullptr, nullptr, nullptr},
    {(char*)"func_defaults", (getter)mp_func_defaults, nullptr, nullptr, nullptr},
    {(char*)"func_globals",  (getter)mp_func_globals,  nullptr, nullptr, nullptr},
    {(char*)"func_doc",      (getter)mp_doc,           nullptr, nullptr, nullptr},
    {(char*)"func_name",     (getter)mp_name,          nullptr, nullptr, nullptr},

    {(char*)"__creates__",         (getter)mp_getcreates, (setter)mp_setcreates,
      (char*)"For ownership rules of result: if true, objects are python-owned", nullptr},
    {(char*)"__mempolicy__",       (getter)mp_getmempolicy, (setter)mp_setmempolicy,
      (char*)"For argument ownership rules: like global, either heuristic or strict", nullptr},
    {(char*)"__release_gil__",     (getter)mp_getthreaded, (setter)mp_setthreaded,
      (char*)"If true, releases GIL on call into C++", nullptr},
    {(char*)"__useffi__",          (getter)mp_getuseffi, (setter)mp_setuseffi,
      (char*)"not implemented", nullptr},
    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}
};

//= CPyCppyy method proxy function behavior ==================================
static PyObject* mp_call(CPPOverload* pymeth, PyObject* args, PyObject* kwds)
{
// Call the appropriate overload of this method.

    CPPInstance* oldSelf = pymeth->fSelf;

// get local handles to proxy internals
    auto& methods     = pymeth->fMethodInfo->fMethods;

    CPPOverload::Methods_t::size_type nMethods = methods.size();

    CallContext ctxt{};
    const auto mflags = pymeth->fMethodInfo->fFlags;
    const auto mempolicy = (mflags & (CallContext::kUseHeuristics | CallContext::kUseStrict));
    ctxt.fFlags |= mempolicy ? mempolicy : (uint64_t)CallContext::sMemoryPolicy;
    ctxt.fFlags |= (mflags & CallContext::kReleaseGIL);

// magic variable to prevent recursion passed by keyword?
    if (kwds && PyDict_CheckExact(kwds) && PyDict_Size(kwds) != 0) {
        if (PyDict_DelItem(kwds, PyStrings::gNoImplicit) == 0) {
            ctxt.fFlags |= CallContext::kNoImplicit;
            if (!PyDict_Size(kwds)) kwds = nullptr;
            else {
                PyErr_SetString(PyExc_TypeError, "keyword arguments are not yet supported");
                return nullptr;
            }
        } else
           PyErr_Clear();
    }

// simple case
    if (nMethods == 1) {
        ctxt.fFlags |= CallContext::kAllowImplicit;   // no two rounds needed
        PyObject* result = methods[0]->Call(pymeth->fSelf, args, kwds, &ctxt);
        return HandleReturn(pymeth, oldSelf, result);
    }

// otherwise, handle overloading
    uint64_t sighash = HashSignature(args);

// look for known signatures ...
    auto& dispatchMap = pymeth->fMethodInfo->fDispatchMap;
    PyCallable* pc = nullptr;
    for (const auto& p : dispatchMap) {
        if (p.first == sighash) {
            pc = p.second;
            break;
        }
    }
    if (pc != nullptr) {
        PyObject* result = pc->Call(pymeth->fSelf, args, kwds, &ctxt);
        result = HandleReturn(pymeth, oldSelf, result);

        if (result != 0)
            return result;

    // fall through: python is dynamic, and so, the hashing isn't infallible
        PyErr_Clear();
    }
    
// ... otherwise loop over all methods and find the one that does not fail
    if (!IsSorted(mflags)) {
        std::stable_sort(methods.begin(), methods.end(), PriorityCmp);
        pymeth->fMethodInfo->fFlags |= CallContext::kIsSorted;
    }

    std::vector<Utility::PyError_t> errors;
    for (int stage = 0; stage < 2; ++stage) {
        for (CPPOverload::Methods_t::size_type i = 0; i < nMethods; ++i) {
            PyObject* result = methods[i]->Call(pymeth->fSelf, args, kwds, &ctxt);

            if (result != 0) {
            // success: update the dispatch map for subsequent calls
                dispatchMap.push_back(std::make_pair(sighash, methods[i]));
                if (!errors.empty())
                    std::for_each(errors.begin(), errors.end(), Utility::PyError_t::Clear);
                return HandleReturn(pymeth, oldSelf, result);
            }

        // else failure ..
            if (stage != 0) {
                PyErr_Clear();    // first stage errors should be more informative
                continue;
            }

        // collect error message/trace (automatically clears exception, too)
            if (!PyErr_Occurred()) {
            // this should not happen; set an error to prevent core dump and report
                PyObject* sig = methods[i]->GetPrototype();
                PyErr_Format(PyExc_SystemError, "%s =>\n    %s",
                    CPyCppyy_PyUnicode_AsString(sig), (char*)"nullptr result without error in mp_call");
                Py_DECREF(sig);
            }
            Utility::FetchError(errors);
            ResetCallState(pymeth->fSelf, oldSelf, false);
        }

    // only move forward if implicit conversions are available
        if (!HaveImplicit(&ctxt))
            break;

        ctxt.fFlags |= CallContext::kAllowImplicit;
    }

// first summarize, then add details
    PyObject* topmsg = CPyCppyy_PyUnicode_FromFormat(
        "none of the %d overloaded methods succeeded. Full details:", (int)nMethods);
    SetDetailedException(errors, topmsg /* steals */, PyExc_TypeError /* default error */);

// report failure
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* mp_str(CPPOverload* cppinst)
{
// Print a description that includes the C++ name
     std::ostringstream s;
     s << "<C++ overload \"" << cppinst->fMethodInfo->fName << "\" at " << (void*)cppinst << ">";
     return CPyCppyy_PyUnicode_FromString(s.str().c_str());
}

//-----------------------------------------------------------------------------
static CPPOverload* mp_descrget(CPPOverload* pymeth, CPPInstance* pyobj, PyObject*)
{
// Descriptor; create and return a new bound method proxy (language requirement) if self
    if (!pyobj) {
        Py_INCREF(pymeth);
        return pymeth;       // unbound, e.g. free functions
    }

// else: bound
    CPPOverload* newPyMeth = free_list;
    if (newPyMeth != NULL) {
        free_list = (CPPOverload*)(newPyMeth->fSelf);
        (void)PyObject_INIT(newPyMeth, &CPPOverload_Type);
        numfree--;
    }
    else {
        newPyMeth = PyObject_GC_New(CPPOverload, &CPPOverload_Type);
        if (!newPyMeth)
            return nullptr;
    }

// method info is shared, as it contains the collected overload knowledge
    *pymeth->fMethodInfo->fRefCount += 1;
    newPyMeth->fMethodInfo = pymeth->fMethodInfo;

// new method is to be bound to current object
    Py_INCREF((PyObject*)pyobj);
    newPyMeth->fSelf = pyobj;

    PyObject_GC_Track(newPyMeth);
    return newPyMeth;
}


//= CPyCppyy method proxy construction/destruction ===========================
static CPPOverload* mp_new(PyTypeObject*, PyObject*, PyObject*)
{
// Create a new method proxy object.
    CPPOverload* pymeth = PyObject_GC_New(CPPOverload, &CPPOverload_Type);
    pymeth->fSelf = nullptr;
    pymeth->fMethodInfo = new CPPOverload::MethodInfo_t;

    PyObject_GC_Track(pymeth);
    return pymeth;
}

//-----------------------------------------------------------------------------
static void mp_dealloc(CPPOverload* pymeth)
{
// Deallocate memory held by method proxy object.
    PyObject_GC_UnTrack(pymeth);

    Py_CLEAR(pymeth->fSelf);

    if (--(*pymeth->fMethodInfo->fRefCount) <= 0) {
        delete pymeth->fMethodInfo;
    }

    if (numfree < CPPOverload_MAXFREELIST) {
        pymeth->fSelf = (CPyCppyy::CPPInstance*)free_list;
        free_list = pymeth;
        numfree++;
    }
    else {
        PyObject_GC_Del(pymeth);
    }
}

//-----------------------------------------------------------------------------
static Py_ssize_t mp_hash(CPPOverload* pymeth)
{
// Hash of method proxy object for insertion into dictionaries; with actual
// method (fMethodInfo) shared, its address is best suited.
    return _Py_HashPointer(pymeth->fMethodInfo);
}

//-----------------------------------------------------------------------------
static int mp_traverse(CPPOverload* pymeth, visitproc visit, void* args)
{
// Garbage collector traverse of held python member objects.
    if (pymeth->fSelf)
        return visit((PyObject*)pymeth->fSelf, args);

    return 0;
}

//-----------------------------------------------------------------------------
static int mp_clear(CPPOverload* pymeth)
{
// Garbage collector clear of held python member objects.
    Py_CLEAR(pymeth->fSelf);

    return 0;
}

//-----------------------------------------------------------------------------
static PyObject* mp_richcompare(CPPOverload* self, CPPOverload* other, int op)
{
// Rich set of comparison objects; only equals is defined.
    if (op != Py_EQ)
        return PyType_Type.tp_richcompare((PyObject*)self, (PyObject*)other, op);

// defined by type + (shared) MethodInfo + bound self, with special case for
// fSelf (i.e. pseudo-function)
    if ((Py_TYPE(self) == Py_TYPE(other) && self->fMethodInfo == other->fMethodInfo) && \
         ((IsPseudoFunc(self) && IsPseudoFunc(other)) || self->fSelf == other->fSelf)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}


//= CPyCppyy method proxy access to internals ================================
static PyObject* mp_overload(CPPOverload* pymeth, PyObject* sigarg)
{
// Select and call a specific C++ overload, based on its signature.
    if (!CPyCppyy_PyUnicode_Check(sigarg)) {
        PyErr_Format(PyExc_TypeError, "__overload__() argument 1 must be string, not %.50s",
            sigarg == Py_None ? "None" : Py_TYPE(sigarg)->tp_name);
        return nullptr;
    }

    PyObject* sig1 = CPyCppyy_PyUnicode_FromFormat("(%s)", CPyCppyy_PyUnicode_AsString(sigarg));

    CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;
    for (auto& meth : methods) {

        PyObject* sig2 = meth->GetSignature(false);
        if (PyObject_RichCompareBool(sig1, sig2, Py_EQ)) {
            Py_DECREF(sig2);

            CPPOverload* newmeth = mp_new(nullptr, nullptr, nullptr);
            CPPOverload::Methods_t vec; vec.push_back(meth->Clone());
            newmeth->Set(pymeth->fMethodInfo->fName, vec);

            if (pymeth->fSelf) {
                Py_INCREF(pymeth->fSelf);
                newmeth->fSelf = pymeth->fSelf;
            }
            newmeth->fMethodInfo->fFlags = pymeth->fMethodInfo->fFlags;

            Py_DECREF(sig1);
            return (PyObject*)newmeth;
        }

        Py_DECREF(sig2);
    }

    Py_DECREF(sig1);
    PyErr_Format(PyExc_LookupError,
        "signature \"%s\" not found", CPyCppyy_PyUnicode_AsString(sigarg));
    return nullptr;
}

//= CPyCppyy method proxy access to internals ================================
static PyObject* mp_add_overload(CPPOverload* pymeth, PyObject* new_overload)
{
    TPythonCallback* cb = new TPythonCallback(new_overload);
    pymeth->AddMethod(cb);
    Py_RETURN_NONE;
}

static PyMethodDef mp_methods[] = {
    {(char*)"__overload__",     (PyCFunction)mp_overload, METH_O,
      (char*)"select overload for dispatch" },
    {(char*)"__add_overload__", (PyCFunction)mp_add_overload, METH_O,
      (char*)"add a new overload" },
    {(char*)nullptr, nullptr, 0, nullptr }
};

} // unnamed namespace


//= CPyCppyy method proxy type ===============================================
PyTypeObject CPPOverload_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.CPPOverload",    // tp_name
    sizeof(CPPOverload),           // tp_basicsize
    0,                             // tp_itemsize
    (destructor)mp_dealloc,        // tp_dealloc
    0,                             // tp_print
    0,                             // tp_getattr
    0,                             // tp_setattr
    0,                             // tp_compare
    0,                             // tp_repr
    0,                             // tp_as_number
    0,                             // tp_as_sequence
    0,                             // tp_as_mapping
    (hashfunc)mp_hash,             // tp_hash
    (ternaryfunc)mp_call,          // tp_call
    (reprfunc)mp_str,              // tp_str
    0,                             // tp_getattro
    0,                             // tp_setattro
    0,                             // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,      // tp_flags
    (char*)"cppyy method proxy (internal)",       // tp_doc
    (traverseproc)mp_traverse,     // tp_traverse
    (inquiry)mp_clear,             // tp_clear
    (richcmpfunc)mp_richcompare,   // tp_richcompare
    0,                             // tp_weaklistoffset
    0,                             // tp_iter
    0,                             // tp_iternext
    mp_methods,                    // tp_methods
    0,                             // tp_members
    mp_getset,                     // tp_getset
    0,                             // tp_base
    0,                             // tp_dict
    (descrgetfunc)mp_descrget,     // tp_descr_get
    0,                             // tp_descr_set
    0,                             // tp_dictoffset
    0,                             // tp_init
    0,                             // tp_alloc
    (newfunc)mp_new,               // tp_new
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


//- public members -----------------------------------------------------------
void CPyCppyy::CPPOverload::Set(const std::string& name, std::vector<PyCallable*>& methods)
{
// Fill in the data of a freshly created method proxy.
    fMethodInfo->fName = name;
    fMethodInfo->fMethods.swap(methods);
    fMethodInfo->fFlags &= ~CallContext::kIsSorted;

// special case: all constructors are considered creators by default
    if (name == "__init__")
        fMethodInfo->fFlags |= (CallContext::kIsCreator | CallContext::kIsConstructor);

// special case, in heuristics mode also tag *Clone* methods as creators
    if (CallContext::sMemoryPolicy == CallContext::kUseHeuristics && \
            name.find("Clone") != std::string::npos)
        fMethodInfo->fFlags |= CallContext::kIsCreator;
}

//-----------------------------------------------------------------------------
void CPyCppyy::CPPOverload::AddMethod(PyCallable* pc)
{
// Fill in the data of a freshly created method proxy.
    fMethodInfo->fMethods.push_back(pc);
    fMethodInfo->fFlags &= ~CallContext::kIsSorted;
}

//-----------------------------------------------------------------------------
void CPyCppyy::CPPOverload::AddMethod(CPPOverload* meth)
{
    fMethodInfo->fMethods.insert(fMethodInfo->fMethods.end(),
        meth->fMethodInfo->fMethods.begin(), meth->fMethodInfo->fMethods.end());
    fMethodInfo->fFlags &= ~CallContext::kIsSorted;
    meth->fMethodInfo->fMethods.clear();
}

//-----------------------------------------------------------------------------
CPyCppyy::CPPOverload::MethodInfo_t::~MethodInfo_t()
{
// Destructor (this object is reference counted).
    for (Methods_t::iterator it = fMethods.begin(); it != fMethods.end(); ++it) {
        delete *it;
    }
    fMethods.clear();
    delete fRefCount;
}

// TODO: something like PyMethod_Fini to clear up the free_list
