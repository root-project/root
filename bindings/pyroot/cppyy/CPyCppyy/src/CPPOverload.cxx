// Bindings
#include "CPyCppyy.h"
#include "CPyCppyy/Reflex.h"
#include "structmember.h"    // from Python
#if PY_VERSION_HEX >= 0x02050000
#if PY_VERSION_HEX <  0x030b0000
#include "code.h"            // from Python
#endif
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
// The fSelf field is used to chain the elements.
static CPPOverload* free_list;
static int numfree = 0;
#ifndef CPPOverload_MAXFREELIST
#define CPPOverload_MAXFREELIST 32
#endif


// TODO: only used in pythonizations to add Python-side overloads to existing
// C++ overloads, but may be better off integrated with Pythonize.cxx callbacks
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

    ~TPythonCallback() override {
        Py_DECREF(fCallable);
        fCallable = nullptr;
    }

    PyObject* GetSignature(bool /*show_formalargs*/ = true) override {
        return CPyCppyy_PyText_FromString("*args, **kwargs");
    }
    PyObject* GetSignatureNames() override {
        return PyTuple_New(0);
    }
    PyObject* GetSignatureTypes() override {
        return PyTuple_New(0);
    }
    PyObject* GetPrototype(bool /*show_formalargs*/ = true) override {
        return CPyCppyy_PyText_FromString("<callback>");
    }
    PyObject* GetDocString() override {
        if (PyObject_HasAttrString(fCallable, "__doc__")) {
            return PyObject_GetAttrString(fCallable, "__doc__");
        } else {
            return GetPrototype();
        }
    }

    int GetPriority() override { return 100; };
    bool IsGreedy() override { return false; };

    int GetMaxArgs() override { return 100; };
    PyObject* GetCoVarNames() override { // TODO: pick these up from the callable
        Py_RETURN_NONE;
    }
    PyObject* GetArgDefault(int /* iarg */, bool /* silent */ =true) override {
        Py_RETURN_NONE;      // TODO: pick these up from the callable
    }

    PyObject* GetScopeProxy() override { // should this be the module ??
        Py_RETURN_NONE;
    }

    Cppyy::TCppFuncAddr_t GetFunctionAddress() override {
        return (Cppyy::TCppFuncAddr_t)nullptr;
    }

    PyCallable* Clone() override { return new TPythonCallback(*this); }

    PyObject* Call(CPPInstance*& self,
            CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* /* ctxt = 0 */) override {

#if PY_VERSION_HEX >= 0x03080000
        if (self) {
            if (nargsf & PY_VECTORCALL_ARGUMENTS_OFFSET) {      // mutation allowed?
                std::swap(((PyObject**)args-1)[0], (PyObject*&)self);
                nargsf &= ~PY_VECTORCALL_ARGUMENTS_OFFSET;
                args = args-1;
            } else {
                Py_ssize_t nkwargs = kwds ? PyTuple_GET_SIZE(kwds) : 0;
                Py_ssize_t totalargs = PyVectorcall_NARGS(nargsf)+nkwargs;
                PyObject** newArgs = (PyObject**)PyMem_Malloc((totalargs+1) * sizeof(PyObject*));
                if (!newArgs)
                    return nullptr;

                newArgs[0] = (PyObject*)self;
                if (0 < totalargs)
                    memcpy((void*)&newArgs[1], args, totalargs * sizeof(PyObject*));
                args = newArgs;
            }
            nargsf += 1;
        }

        PyObject* result = CPyCppyy_PyObject_Call(fCallable, args, nargsf, kwds);
        if (self) {
            if (nargsf & PY_VECTORCALL_ARGUMENTS_OFFSET)
                std::swap(((PyObject**)args-1)[0], (PyObject*&)self);
            else PyMem_Free((void*)args);
        }
#else
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
        PyObject* result = PyObject_Call(fCallable, newArgs, kwds);
        Py_DECREF(newArgs);
#endif
        return result;
    }
};

// helper to test whether a method is used in a pseudo-function modus
static inline bool IsPseudoFunc(CPPOverload* pymeth)
{
    return pymeth->fMethodInfo->fFlags & CallContext::kIsPseudoFunc;
}

// helper to sort on method priority
static int PriorityCmp(const std::pair<int, PyCallable*>& left, const std::pair<int, PyCallable*>& right)
{
    return left.first > right.first;
}

// return helper
static inline void ResetCallState(CPPInstance* descr_self, CPPInstance*& im_self)
{
// reset self if needed, allowing simple re-use
    if (descr_self != im_self) {
        Py_XDECREF(im_self);
        im_self = descr_self;
    }
}

// helper to factor out return logic of mp_call / mp_vectorcall
static inline PyObject* HandleReturn(
    CPPOverload* pymeth, CPPInstance* im_self, PyObject* result)
{
// special case for python exceptions, propagated through C++ layer
    if (result) {
        CPPInstance* cppres = (CPPInstance*)(CPPInstance_Check(result) ? result : nullptr);

    // if this method creates new objects, always take ownership
        if (IsCreator(pymeth->fMethodInfo->fFlags)) {

        // either be a constructor with a fresh object proxy self ...
            if (IsConstructor(pymeth->fMethodInfo->fFlags)) {
                if (im_self)
                    im_self->PythonOwns();
            }

        // ... or be a regular method with an object proxy return value
            else if (cppres)
                cppres->PythonOwns();
        }

    // if this new object falls inside self, make sure its lifetime is proper
        if (!(pymeth->fMethodInfo->fFlags & CallContext::kNeverLifeLine)) {
            int ll_action = 0;
            if ((PyObject*)im_self != result) {
                if (pymeth->fMethodInfo->fFlags & CallContext::kSetLifeLine)
                    ll_action = 1;
                else if (cppres && CPPInstance_Check(im_self)) {
                // if self was a by-value return and result is not, pro-actively protect result;
                // else if the return value falls within the memory of 'this', force a lifeline
                    if (!(cppres->fFlags & CPPInstance::kIsValue)) {     // no need if the result is temporary
                        if (im_self->fFlags & CPPInstance::kIsValue)
                            ll_action = 2;
                        else if (im_self->fFlags & CPPInstance::kHasLifeLine)
                            ll_action = 3;
                        else {
                            ptrdiff_t offset = (ptrdiff_t)cppres->GetObject() - (ptrdiff_t)im_self->GetObject();
                            if (0 <= offset && offset < (ptrdiff_t)Cppyy::SizeOf(im_self->ObjectIsA()))
                                 ll_action = 4;
                        }
                    }
                }
            }

            if (!ll_action)
                pymeth->fMethodInfo->fFlags |= CallContext::kNeverLifeLine;       // assume invariant semantics
            else {
                if (PyObject_SetAttr(result, PyStrings::gLifeLine, (PyObject*)im_self) == -1)
                    PyErr_Clear();         // ignored
                if (cppres) cppres->fFlags |= CPPInstance::kHasLifeLine;          // for chaining
                pymeth->fMethodInfo->fFlags |= CallContext::kSetLifeLine;         // for next time
            }
        }
    }

// reset self as necessary to allow re-use of the CPPOverload
    ResetCallState(pymeth->fSelf, im_self);

    return result;
}


//= CPyCppyy method proxy object behaviour ===================================
static PyObject* mp_name(CPPOverload* pymeth, void*)
{
    return CPyCppyy_PyText_FromString(pymeth->GetName().c_str());
}

//----------------------------------------------------------------------------
static PyObject* mp_module(CPPOverload* /* pymeth */, void*)
{
    Py_INCREF(PyStrings::gThisModule);
    return PyStrings::gThisModule;
}

//----------------------------------------------------------------------------
static PyObject* mp_doc(CPPOverload* pymeth, void*)
{
    if (pymeth->fMethodInfo->fDoc) {
        Py_INCREF(pymeth->fMethodInfo->fDoc);
        return pymeth->fMethodInfo->fDoc;
    }

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
    PyObject* separator = CPyCppyy_PyText_FromString("\n");
    for (CPPOverload::Methods_t::size_type i = 1; i < nMethods; ++i) {
        CPyCppyy_PyText_Append(&doc, separator);
        CPyCppyy_PyText_AppendAndDel(&doc, methods[i]->GetDocString());
    }
    Py_DECREF(separator);

    return doc;
}

static int mp_doc_set(CPPOverload* pymeth, PyObject *val, void *)
{
    Py_XDECREF(pymeth->fMethodInfo->fDoc);
    Py_INCREF(val);
    pymeth->fMethodInfo->fDoc = val;
    return 0;
}

/**
 * @brief Returns a dictionary with the input parameter names for all overloads.
 *
 * This dictionary may look like:
 *
 * {'double ::foo(int a, float b, double c)': ('a', 'b', 'c'),
 *  'float ::foo(float b)': ('b',),
 *  'int ::foo(int a)': ('a',),
 *  'int ::foo(int a, float b)': ('a', 'b')}
 */
static PyObject *mp_func_overloads_names(CPPOverload *pymeth)
{

   const CPPOverload::Methods_t &methods = pymeth->fMethodInfo->fMethods;

   PyObject *overloads_names_dict = PyDict_New();

   for (PyCallable *method : methods) {
      PyDict_SetItem(overloads_names_dict, method->GetPrototype(), method->GetSignatureNames());
   }

   return overloads_names_dict;
}

/**
 * @brief Returns a dictionary with the types of all overloads.
 *
 * This dictionary may look like:
 *
 * {'double ::foo(int a, float b, double c)': {'input_types': ('int', 'float', 'double'), 'return_type': 'double'},
 *  'float ::foo(float b)': {'input_types': ('float',), 'return_type': 'float'},
 *  'int ::foo(int a)': {'input_types': ('int',), 'return_type': 'int'},
 *  'int ::foo(int a, float b)': {'input_types': ('int', 'float'), 'return_type': 'int'}}
 */
static PyObject *mp_func_overloads_types(CPPOverload *pymeth)
{

   const CPPOverload::Methods_t &methods = pymeth->fMethodInfo->fMethods;

   PyObject *overloads_types_dict = PyDict_New();

   for (PyCallable *method : methods) {
      PyDict_SetItem(overloads_types_dict, method->GetPrototype(), method->GetSignatureTypes());
   }

   return overloads_types_dict;
}

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
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

//----------------------------------------------------------------------------
static PyObject* mp_func_closure(CPPOverload* /* pymeth */, void*)
{
// Stub only, to fill out the python function interface.
    Py_RETURN_NONE;
}

// To declare a variable as unused only when compiling for Python 3.
#if PY_VERSION_HEX < 0x03000000
#define CPyCppyy_Py3_UNUSED(name) name
#else
#define CPyCppyy_Py3_UNUSED(name)
#endif

//----------------------------------------------------------------------------
static PyObject* mp_func_code(CPPOverload* CPyCppyy_Py3_UNUSED(pymeth), void*)
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
        PyTuple_SET_ITEM(co_varnames, 0, CPyCppyy_PyText_FromString("self"));
        PyTuple_SET_ITEM(co_varnames, 1, CPyCppyy_PyText_FromString("*args"));
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
    Py_RETURN_NONE;
#endif
}

//----------------------------------------------------------------------------
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
        else
           PyErr_Clear();
    }
    _PyTuple_Resize(&defaults, itup);

    return defaults;
}

//----------------------------------------------------------------------------
static PyObject* mp_func_globals(CPPOverload* /* pymeth */, void*)
{
// Return this function's global dict (hard-wired to be the cppyy module); used
// for lookup of names from co_code indexing into co_names.
    PyObject* pyglobal = PyModule_GetDict(PyImport_AddModule((char*)"cppyy"));
    Py_XINCREF(pyglobal);
    return pyglobal;
}

//----------------------------------------------------------------------------
static inline int set_flag(CPPOverload* pymeth, PyObject* value, CallContext::ECallFlags flag, const char* name)
{
// Generic setter of a (boolean) flag.
    if (!value) {        // accept as false (delete)
        pymeth->fMethodInfo->fFlags &= ~flag;
        return 0;
    }

    long istrue = PyLong_AsLong(value);
    if (istrue == -1 && PyErr_Occurred()) {
        PyErr_Format(PyExc_ValueError, "a boolean 1 or 0 is required for %s", name);
        return -1;
    }

    if (istrue)
        pymeth->fMethodInfo->fFlags |= flag;
    else
        pymeth->fMethodInfo->fFlags &= ~flag;

    return 0;
}

//----------------------------------------------------------------------------
static PyObject* mp_getcreates(CPPOverload* pymeth, void*)
{
// Get '__creates__' boolean, which determines ownership of return values.
    return PyInt_FromLong((long)IsCreator(pymeth->fMethodInfo->fFlags));
}

//----------------------------------------------------------------------------
static int mp_setcreates(CPPOverload* pymeth, PyObject* value, void*)
{
// Set '__creates__' boolean, which determines ownership of return values.
    return set_flag(pymeth, value, CallContext::kIsCreator, "__creates__");
}

//----------------------------------------------------------------------------
static PyObject* mp_getmempolicy(CPPOverload* pymeth, void*)
{
// Get '_mempolicy' enum, which determines ownership of call arguments.
    if (pymeth->fMethodInfo->fFlags & CallContext::kUseHeuristics)
        return PyInt_FromLong(CallContext::kUseHeuristics);

    if (pymeth->fMethodInfo->fFlags & CallContext::kUseStrict)
        return PyInt_FromLong(CallContext::kUseStrict);

    return PyInt_FromLong(-1);
}

//----------------------------------------------------------------------------
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


//----------------------------------------------------------------------------
#define CPPYY_BOOLEAN_PROPERTY(name, flag, label)                            \
static PyObject* mp_get##name(CPPOverload* pymeth, void*) {                  \
    if (pymeth->fMethodInfo->fFlags & flag) {                                \
        Py_RETURN_TRUE;                                                      \
    }                                                                        \
    Py_RETURN_FALSE;                                                         \
}                                                                            \
                                                                             \
static int mp_set##name(CPPOverload* pymeth, PyObject* value, void*) {       \
    return set_flag(pymeth, value, flag, label);                             \
}

CPPYY_BOOLEAN_PROPERTY(lifeline, CallContext::kSetLifeLine, "__set_lifeline__")
CPPYY_BOOLEAN_PROPERTY(threaded, CallContext::kReleaseGIL,  "__release_gil__")
CPPYY_BOOLEAN_PROPERTY(useffi,   CallContext::kUseFFI,      "__useffi__")
CPPYY_BOOLEAN_PROPERTY(sig2exc,  CallContext::kProtected,   "__sig2exc__")

static PyObject* mp_getcppname(CPPOverload* pymeth, void*)
{
    if ((void*)pymeth == (void*)&CPPOverload_Type)
        return CPyCppyy_PyText_FromString("CPPOverload_Type");

    auto& methods = pymeth->fMethodInfo->fMethods;
    if (methods.empty())
        return CPyCppyy_PyText_FromString("void (*)()");   // debatable

    if (methods.size() == 1)
        return methods[0]->GetTypeName();

    return CPyCppyy_PyText_FromString("void* (*)(...)");   // id.
}


//----------------------------------------------------------------------------
static PyGetSetDef mp_getset[] = {
    {(char*)"__name__",   (getter)mp_name,   nullptr, nullptr, nullptr},
    {(char*)"__module__", (getter)mp_module, nullptr, nullptr, nullptr},
    {(char*)"__doc__",    (getter)mp_doc,    (setter)mp_doc_set, nullptr, nullptr},

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
    {(char*)"func_doc",      (getter)mp_doc,           (setter)mp_doc_set, nullptr, nullptr},
    {(char*)"func_name",     (getter)mp_name,          nullptr, nullptr, nullptr},
    {(char*)"func_overloads_types",    (getter)mp_func_overloads_types,    nullptr, nullptr, nullptr},
    {(char*)"func_overloads_names",    (getter)mp_func_overloads_names,    nullptr, nullptr, nullptr},


// flags to control behavior
    {(char*)"__creates__",         (getter)mp_getcreates, (setter)mp_setcreates,
      (char*)"For ownership rules of result: if true, objects are python-owned", nullptr},
    {(char*)"__mempolicy__",       (getter)mp_getmempolicy, (setter)mp_setmempolicy,
      (char*)"For argument ownership rules: like global, either heuristic or strict", nullptr},
    {(char*)"__set_lifeline__",    (getter)mp_getlifeline, (setter)mp_setlifeline,
      (char*)"If true, set a lifeline from the return value onto self", nullptr},
    {(char*)"__release_gil__",     (getter)mp_getthreaded, (setter)mp_setthreaded,
      (char*)"If true, releases GIL on call into C++", nullptr},
    {(char*)"__useffi__",          (getter)mp_getuseffi, (setter)mp_setuseffi,
      (char*)"not implemented", nullptr},
    {(char*)"__sig2exc__",         (getter)mp_getsig2exc, (setter)mp_setsig2exc,
      (char*)"If true, turn signals into Python exceptions", nullptr},

// basic reflection information
    {(char*)"__cpp_name__",        (getter)mp_getcppname, nullptr, nullptr, nullptr},

    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}
};

//= CPyCppyy method proxy function behavior ==================================
#if PY_VERSION_HEX >= 0x03080000
static PyObject* mp_vectorcall(
    CPPOverload* pymeth, PyObject* const *args, size_t nargsf, PyObject* kwds)
#else
static PyObject* mp_call(CPPOverload* pymeth, PyObject* args, PyObject* kwds)
#endif
{
#if PY_VERSION_HEX < 0x03080000
    size_t nargsf = PyTuple_GET_SIZE(args);
#endif

// Call the appropriate overload of this method.

// If called from a descriptor, then this could be a bound function with
// non-zero self; otherwise pymeth->fSelf is expected to always be nullptr.

    CPPInstance* im_self = pymeth->fSelf;

// get local handles to proxy internals
    auto& methods = pymeth->fMethodInfo->fMethods;

    CPPOverload::Methods_t::size_type nMethods = methods.size();

    CallContext ctxt{};
    const auto mflags = pymeth->fMethodInfo->fFlags;
    const auto mempolicy = (mflags & (CallContext::kUseHeuristics | CallContext::kUseStrict));
    ctxt.fFlags |= mempolicy ? mempolicy : (uint64_t)CallContext::sMemoryPolicy;
    ctxt.fFlags |= (mflags & CallContext::kReleaseGIL);
    ctxt.fFlags |= (mflags & CallContext::kProtected);
    if (IsConstructor(pymeth->fMethodInfo->fFlags)) ctxt.fFlags |= CallContext::kIsConstructor;
    ctxt.fFlags |= (pymeth->fFlags & (CallContext::kCallDirect | CallContext::kFromDescr));
    ctxt.fPyContext = (PyObject*)im_self;  // no Py_INCREF as no ownership

// check implicit conversions status (may be disallowed to prevent recursion)
    ctxt.fFlags |= (pymeth->fFlags & CallContext::kNoImplicit);

// simple case
    if (nMethods == 1) {
        if (!NoImplicit(&ctxt)) ctxt.fFlags |= CallContext::kAllowImplicit;    // no two rounds needed
        PyObject* result = methods[0]->Call(im_self, args, nargsf, kwds, &ctxt);
        return HandleReturn(pymeth, im_self, result);
    }

// otherwise, handle overloading
    uint64_t sighash = HashSignature(args, nargsf);

// look for known signatures ...
    auto& dispatchMap = pymeth->fMethodInfo->fDispatchMap;
    PyCallable* memoized_pc = nullptr;
    for (const auto& p : dispatchMap) {
        if (p.first == sighash) {
            memoized_pc = p.second;
            break;
        }
    }
    if (memoized_pc) {
    // it is necessary to enable implicit conversions as the memoized call may be from
    // such a conversion case; if the call fails, the implicit flag is reset below
        if (!NoImplicit(&ctxt)) ctxt.fFlags |= CallContext::kAllowImplicit;
        PyObject* result = memoized_pc->Call(im_self, args, nargsf, kwds, &ctxt);
        if (result)
            return HandleReturn(pymeth, im_self, result);

    // fall through: python is dynamic, and so, the hashing isn't infallible
        ctxt.fFlags &= ~CallContext::kAllowImplicit;
        PyErr_Clear();
        ResetCallState(pymeth->fSelf, im_self);
    }

// ... otherwise loop over all methods and find the one that does not fail
    if (!IsSorted(mflags)) {
    // sorting is based on priority, which is not stored on the method as it is used
    // only once, so copy the vector of methods into one where the priority can be
    // stored during sorting
        std::vector<std::pair<int, PyCallable*>> pm; pm.reserve(methods.size());
        for (auto ptr : methods)
            pm.emplace_back(ptr->GetPriority(), ptr);
        std::stable_sort(pm.begin(), pm.end(), PriorityCmp);
        for (CPPOverload::Methods_t::size_type i = 0; i < methods.size(); ++i)
            methods[i] = pm[i].second;
        pymeth->fMethodInfo->fFlags |= CallContext::kIsSorted;
    }

    std::vector<Utility::PyError_t> errors;
    std::vector<bool> implicit_possible(methods.size());
    for (int stage = 0; stage < 2; ++stage) {
        bool bHaveImplicit = false;
        for (CPPOverload::Methods_t::size_type i = 0; i < nMethods; ++i) {
            if (stage && !implicit_possible[i])
                continue;    // did not set implicit conversion, so don't try again

            PyObject* result = methods[i]->Call(im_self, args, nargsf, kwds, &ctxt);
            if (result) {
            // success: update the dispatch map for subsequent calls
                if (!memoized_pc)
                    dispatchMap.push_back(std::make_pair(sighash, methods[i]));
                else {
                // debatable: apparently there are two methods that map onto the same sighash
                // and preferring the latest may result in "ping pong."
                    for (auto& p : dispatchMap) {
                        if (p.first == sighash) {
                            p.second = methods[i];
                            break;
                        }
                    }
                }

                return HandleReturn(pymeth, im_self, result);
            }

        // else failure ..
            if (stage != 0) {
                PyErr_Clear();    // first stage errors should be the more informative
                ResetCallState(pymeth->fSelf, im_self);
                continue;
            }

        // collect error message/trace (automatically clears exception, too)
            if (!PyErr_Occurred()) {
            // this should not happen; set an error to prevent core dump and report
                PyObject* sig = methods[i]->GetPrototype();
                PyErr_Format(PyExc_SystemError, "%s =>\n    %s",
                    CPyCppyy_PyText_AsString(sig), (char*)"nullptr result without error in overload call");
                Py_DECREF(sig);
            }

        // retrieve, store, and clear errors
            bool callee_error = ctxt.fFlags & (CallContext::kPyException | CallContext::kCppException);
            ctxt.fFlags &= ~(CallContext::kPyException | CallContext::kCppException);
            Utility::FetchError(errors, callee_error);

            if (HaveImplicit(&ctxt)) {
                bHaveImplicit = true;
                implicit_possible[i] = true;
                ctxt.fFlags &= ~CallContext::kHaveImplicit;
            } else
                implicit_possible[i] = false;
            ResetCallState(pymeth->fSelf, im_self);
        }

    // only move forward if implicit conversions are available
        if (!bHaveImplicit)
            break;

        ctxt.fFlags |= CallContext::kAllowImplicit;
    }

// first summarize, then add details
    PyObject* topmsg = CPyCppyy_PyText_FromFormat(
        "none of the %d overloaded methods succeeded. Full details:", (int)nMethods);
    SetDetailedException(std::move(errors), topmsg /* steals */, PyExc_TypeError /* default error */);

// report failure
    return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* mp_str(CPPOverload* cppinst)
{
// Print a description that includes the C++ name
     std::ostringstream s;
     s << "<C++ overload \"" << cppinst->fMethodInfo->fName << "\" at " << (void*)cppinst << ">";
     return CPyCppyy_PyText_FromString(s.str().c_str());
}

//----------------------------------------------------------------------------
static CPPOverload* mp_descr_get(CPPOverload* pymeth, CPPInstance* pyobj, PyObject*)
{
// Descriptor; create and return a new, possibly bound, method proxy. This method
// has evolved with versions of python as follows:
//
//   Python version   |       Action
//        <- py2.7    | bound methods need to be first-class objects, so create a new
//                    |  method object if self is not nullptr or Py_None
//     py3.0-py3.7    | bound methods are no longer a language requirement, but
//                    |  still supported: for convenience, retain old behavior
//     py3.8 <=       | vector calls no longer call the descriptor, so when it is
//                    |  called, the method is likely stored, so should be new object

#if PY_VERSION_HEX < 0x03080000
    if (!pyobj || (PyObject*)pyobj == Py_None /* from unbound TemplateProxy */) {
        Py_XDECREF(pymeth->fSelf); pymeth->fSelf = nullptr;
        pymeth->fFlags |= CallContext::kCallDirect | CallContext::kFromDescr;
        Py_INCREF(pymeth);
        return pymeth;       // unbound, e.g. free functions
    }
#endif

// create a new method object
    bool gc_track = false;
    CPPOverload* newPyMeth = free_list;
    if (newPyMeth != NULL) {
        free_list = (CPPOverload*)(newPyMeth->fSelf);
        (void)PyObject_INIT(newPyMeth, &CPPOverload_Type);
        numfree--;
    } else {
        newPyMeth = PyObject_GC_New(CPPOverload, &CPPOverload_Type);
        if (!newPyMeth)
            return nullptr;
        gc_track = true;
    }

// method info is shared, as it contains the collected overload knowledge
    *pymeth->fMethodInfo->fRefCount += 1;
    newPyMeth->fMethodInfo = pymeth->fMethodInfo;

#if PY_VERSION_HEX >= 0x03080000
    newPyMeth->fVectorCall = pymeth->fVectorCall;

    if (pyobj && (PyObject*)pyobj != Py_None) {
        Py_INCREF((PyObject*)pyobj);
        newPyMeth->fSelf = pyobj;
        newPyMeth->fFlags = CallContext::kNone;
    } else {
        newPyMeth->fSelf = nullptr;
        newPyMeth->fFlags = CallContext::kCallDirect;
    }

// vector calls don't get here, unless a method is looked up on an instance, for
// e.g. class methods (C++ static); notify downstream to expect a 'self'
    newPyMeth->fFlags |= CallContext::kFromDescr;

#else
// new method is to be bound to current object
    Py_INCREF((PyObject*)pyobj);
    newPyMeth->fSelf = pyobj;

// reset flags of the new method, as there is a self (which may or may not have
// come in through direct call syntax, but that's now impossible to know, so this
// is the safer choice)
    newPyMeth->fFlags = CallContext::kNone;
#endif

    if (gc_track)
        PyObject_GC_Track(newPyMeth);

    return newPyMeth;
}


//= CPyCppyy method proxy construction/destruction ===========================
static CPPOverload* mp_new(PyTypeObject*, PyObject*, PyObject*)
{
// Create a new method proxy object.
    CPPOverload* pymeth = PyObject_GC_New(CPPOverload, &CPPOverload_Type);
    pymeth->fSelf = nullptr;
    pymeth->fFlags = CallContext::kNone;
    pymeth->fMethodInfo = new CPPOverload::MethodInfo_t;

    PyObject_GC_Track(pymeth);
    return pymeth;
}

//----------------------------------------------------------------------------
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
    } else {
        PyObject_GC_Del(pymeth);
    }
}

//----------------------------------------------------------------------------
static Py_ssize_t mp_hash(CPPOverload* pymeth)
{
// Hash of method proxy object for insertion into dictionaries; with actual
// method (fMethodInfo) shared, its address is best suited.
#if PY_VERSION_HEX >= 0x030d0000
    return Py_HashPointer(pymeth->fMethodInfo);
#else
    return _Py_HashPointer(pymeth->fMethodInfo);
#endif
}

//----------------------------------------------------------------------------
static int mp_traverse(CPPOverload* pymeth, visitproc visit, void* args)
{
// Garbage collector traverse of held python member objects.
    if (pymeth->fSelf)
        return visit((PyObject*)pymeth->fSelf, args);

    return 0;
}

//----------------------------------------------------------------------------
static int mp_clear(CPPOverload* pymeth)
{
// Garbage collector clear of held python member objects.
    Py_CLEAR(pymeth->fSelf);

    return 0;
}

//----------------------------------------------------------------------------
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
static PyObject* mp_overload(CPPOverload* pymeth, PyObject* args)
{
// Select and call a specific C++ overload, based on its signature.
    const char* sigarg = nullptr;
    PyObject* sigarg_tuple = nullptr;
    int want_const = -1;
    Py_ssize_t args_size = PyTuple_GET_SIZE(args);
    if (args_size &&
        PyArg_ParseTuple(args, const_cast<char*>("s|i:__overload__"), &sigarg, &want_const)) {
        want_const = args_size == 1 ? -1 : want_const;
        return pymeth->FindOverload(sigarg ? sigarg : "", want_const);
    } else if (args_size &&
               PyArg_ParseTuple(args, const_cast<char*>("O|i:__overload__"), &sigarg_tuple, &want_const)) {
        PyErr_Clear();
        want_const = args_size == 1 ? -1 : want_const;
        return pymeth->FindOverload(sigarg_tuple, want_const);
    } else {
        PyErr_Format(PyExc_TypeError, "Unexpected arguments to __overload__");
        return nullptr;
    }
}

static PyObject* mp_add_overload(CPPOverload* pymeth, PyObject* new_overload)
{
    TPythonCallback* cb = new TPythonCallback(new_overload);
    pymeth->AdoptMethod(cb);
    Py_RETURN_NONE;
}

static PyObject* mp_reflex(CPPOverload* pymeth, PyObject* args)
{
// Provide the requested reflection information.
    Cppyy::Reflex::RequestId_t request = -1;
    Cppyy::Reflex::FormatId_t  format  = Cppyy::Reflex::OPTIMAL;
    if (!PyArg_ParseTuple(args, const_cast<char*>("i|i:__cpp_reflex__"), &request, &format))
        return nullptr;

    return pymeth->fMethodInfo->fMethods[0]->Reflex(request, format);
}

//----------------------------------------------------------------------------
static PyMethodDef mp_methods[] = {
    {(char*)"__overload__",     (PyCFunction)mp_overload, METH_VARARGS,
      (char*)"select overload for dispatch" },
    {(char*)"__add_overload__", (PyCFunction)mp_add_overload, METH_O,
      (char*)"add a new overload" },
    {(char*)"__cpp_reflex__",   (PyCFunction)mp_reflex, METH_VARARGS,
      (char*)"C++ overload reflection information" },
    {(char*)nullptr, nullptr, 0, nullptr }
};

} // unnamed namespace


//= CPyCppyy method proxy type ===============================================
PyTypeObject CPPOverload_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type, 0)
    (char*)"cppyy.CPPOverload",        // tp_name
    sizeof(CPPOverload),               // tp_basicsize
    0,                                 // tp_itemsize
    (destructor)mp_dealloc,            // tp_dealloc
#if PY_VERSION_HEX >= 0x03080000
    offsetof(CPPOverload, fVectorCall),
#else
    0,                                 // tp_vectorcall_offset / tp_print
#endif
    0,                                 // tp_getattr
    0,                                 // tp_setattr
    0,                                 // tp_as_async / tp_compare
    0,                                 // tp_repr
    0,                                 // tp_as_number
    0,                                 // tp_as_sequence
    0,                                 // tp_as_mapping
    (hashfunc)mp_hash,                 // tp_hash
#if PY_VERSION_HEX >= 0x03080000
    (ternaryfunc)PyVectorcall_Call,    // tp_call
#else
    (ternaryfunc)mp_call,              // tp_call
#endif
    (reprfunc)mp_str,                  // tp_str
    0,                                 // tp_getattro
    0,                                 // tp_setattro
    0,                                 // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC
#if PY_VERSION_HEX >= 0x03080000
        | Py_TPFLAGS_HAVE_VECTORCALL | Py_TPFLAGS_METHOD_DESCRIPTOR
#endif
    ,                                  // tp_flags
    (char*)"cppyy method proxy (internal)", // tp_doc
    (traverseproc)mp_traverse,         // tp_traverse
    (inquiry)mp_clear,                 // tp_clear
    (richcmpfunc)mp_richcompare,       // tp_richcompare
    0,                                 // tp_weaklistoffset
    0,                                 // tp_iter
    0,                                 // tp_iternext
    mp_methods,                        // tp_methods
    0,                                 // tp_members
    mp_getset,                         // tp_getset
    0,                                 // tp_base
    0,                                 // tp_dict
    (descrgetfunc)mp_descr_get,        // tp_descr_get
    0,                                 // tp_descr_set
    0,                                 // tp_dictoffset
    0,                                 // tp_init
    0,                                 // tp_alloc
    (newfunc)mp_new,                   // tp_new
    0,                                 // tp_free
    0,                                 // tp_is_gc
    0,                                 // tp_bases
    0,                                 // tp_mro
    0,                                 // tp_cache
    0,                                 // tp_subclasses
    0                                  // tp_weaklist
#if PY_VERSION_HEX >= 0x02030000
    , 0                                // tp_del
#endif
#if PY_VERSION_HEX >= 0x02060000
    , 0                                // tp_version_tag
#endif
#if PY_VERSION_HEX >= 0x03040000
    , 0                                // tp_finalize
#endif
#if PY_VERSION_HEX >= 0x03080000
    , 0                                // tp_vectorcall
#endif
#if PY_VERSION_HEX >= 0x030c0000
    , 0                                // tp_watched
#endif
#if PY_VERSION_HEX >= 0x030d0000
    , 0                                // tp_versions_used
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

// special case, in heuristics mode also tag *Clone* methods as creators. Only
// check that Clone is present in the method name, not in the template argument
// list.
    if (CallContext::sMemoryPolicy == CallContext::kUseHeuristics) {
        std::string_view name_maybe_template = name;
        auto begin_template = name_maybe_template.find_first_of('<');
        if (begin_template <= name_maybe_template.size()) {
            name_maybe_template = name_maybe_template.substr(0, begin_template);
        }
        if (name_maybe_template.find("Clone") != std::string_view::npos) {
            fMethodInfo->fFlags |= CallContext::kIsCreator;
        }
    }

#if PY_VERSION_HEX >= 0x03080000
    fVectorCall = (vectorcallfunc)mp_vectorcall;
#endif
}

//----------------------------------------------------------------------------
void CPyCppyy::CPPOverload::AdoptMethod(PyCallable* pc)
{
// Fill in the data of a freshly created method proxy.
    fMethodInfo->fMethods.push_back(pc);
    fMethodInfo->fFlags &= ~CallContext::kIsSorted;
}

//----------------------------------------------------------------------------
void CPyCppyy::CPPOverload::MergeOverload(CPPOverload* meth)
{
    if (!HasMethods()) // if fresh method being filled: also copy flags
        fMethodInfo->fFlags = meth->fMethodInfo->fFlags;
    fMethodInfo->fMethods.insert(fMethodInfo->fMethods.end(),
        meth->fMethodInfo->fMethods.begin(), meth->fMethodInfo->fMethods.end());
    fMethodInfo->fFlags &= ~CallContext::kIsSorted;
    meth->fMethodInfo->fDispatchMap.clear();
    meth->fMethodInfo->fMethods.clear();
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPOverload::FindOverload(const std::string& signature, int want_const)
{
    bool accept_any = signature == ":any:";
    CPPOverload* newmeth = nullptr;

    std::string sig1{"("};
    if (!accept_any) {
        sig1.append(signature); sig1.append(")");
        sig1.erase(std::remove(sig1.begin(), sig1.end(), ' '), std::end(sig1));
    }

    CPPOverload::Methods_t& methods = fMethodInfo->fMethods;
    for (auto& meth : methods) {
        bool found = accept_any;
        if (!found) {
            PyObject* pysig2 = meth->GetSignature(false);
            std::string sig2(CPyCppyy_PyText_AsString(pysig2));
            sig2.erase(std::remove(sig2.begin(), sig2.end(), ' '), std::end(sig2));
            Py_DECREF(pysig2);
            if (sig1 == sig2) found = true;

            if (!found) {
                pysig2 = meth->GetSignature(true);
                std::string sig3(CPyCppyy_PyText_AsString(pysig2));
                sig3.erase(std::remove(sig3.begin(), sig3.end(), ' '), std::end(sig3));
                Py_DECREF(pysig2);
                if (sig1 == sig3) found = true;
            }
        }

        if (found && 0 <= want_const) {
            bool isconst = meth->IsConst();
            if (!((want_const && isconst) || (!want_const && !isconst)))
                found = false;
        }

        if (found) {
            if (!newmeth) {
                newmeth = mp_new(nullptr, nullptr, nullptr);
                CPPOverload::Methods_t vec; vec.push_back(meth->Clone());
                newmeth->Set(fMethodInfo->fName, vec);

                if (fSelf) {
                    Py_INCREF(fSelf);
                    newmeth->fSelf = fSelf;
                }
                newmeth->fMethodInfo->fFlags = fMethodInfo->fFlags;
            } else
                newmeth->AdoptMethod(meth->Clone());

            if (!accept_any)
                return (PyObject*)newmeth;
        }
    }

    if (!newmeth)
        PyErr_Format(PyExc_LookupError, "signature \"%s\" not found", signature.c_str());

    return (PyObject*)newmeth;
}

PyObject* CPyCppyy::CPPOverload::FindOverload(PyObject *args_tuple, int want_const)
{
    Py_ssize_t n = PyTuple_Size(args_tuple);

    CPPOverload::Methods_t& methods = fMethodInfo->fMethods;

    // This value is set based on the maximum penalty in Cppyy::CompareMethodArgType
    Py_ssize_t min_score = INT_MAX;
    bool found = false;
    size_t best_method = 0, method_index = 0;

    for (auto& meth : methods) {
        if (0 <= want_const) {
            bool isconst = meth->IsConst();
            if (!((want_const && isconst) || (!want_const && !isconst)))
                continue;
        }

        int score = meth->GetArgMatchScore(args_tuple);

        if (score < min_score) {
            found = true;
            min_score = score;
            best_method = method_index;
        }

        method_index++;
    }

    if (!found) {
        std::string sigargs("(");

        for (int i = 0; i < n; i++) {
            PyObject *pItem = PyTuple_GetItem(args_tuple, i);
            if(!CPyCppyy_PyText_Check(pItem)) {
                PyErr_Format(PyExc_LookupError, "argument types should be in string format");
                return (PyObject*) nullptr;
            }
            std::string arg_type(CPyCppyy_PyText_AsString(pItem));
            sigargs += arg_type + ", ";
        }
        sigargs += ")";

        PyErr_Format(PyExc_LookupError, "signature with arguments \"%s\" not found", sigargs.c_str());
        return (PyObject*) nullptr;
    }

    CPPOverload* newmeth = mp_new(nullptr, nullptr, nullptr);
    CPPOverload::Methods_t vec;
    vec.push_back(methods[best_method]->Clone());
    newmeth->Set(fMethodInfo->fName, vec);

    if (fSelf) {
        Py_INCREF(fSelf);
        newmeth->fSelf = fSelf;
    }
    newmeth->fMethodInfo->fFlags = fMethodInfo->fFlags;

    return (PyObject*) newmeth;
}

//----------------------------------------------------------------------------
CPyCppyy::CPPOverload::MethodInfo_t::~MethodInfo_t()
{
// Destructor (this object is reference counted).
    for (Methods_t::iterator it = fMethods.begin(); it != fMethods.end(); ++it) {
        delete *it;
    }
    fMethods.clear();
    delete fRefCount;
    Py_XDECREF(fDoc);
}

// TODO: something like PyMethod_Fini to clear up the free_list