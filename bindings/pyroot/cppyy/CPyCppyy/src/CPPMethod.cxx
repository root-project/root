// Bindings
#include "CPyCppyy.h"
#include "CPPMethod.h"
#include "CPPExcInstance.h"
#include "CPPInstance.h"
#include "Converters.h"
#include "Executors.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "SignalTryCatch.h"
#include "Utility.h"

#include "CPyCppyy/PyException.h"

// Standard
#include <algorithm>
#include <assert.h>
#include <string.h>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <typeinfo>
#include <memory>


//- data and local helpers ---------------------------------------------------
namespace CPyCppyy {
    extern PyObject* gThisModule;
    extern PyObject* gBusException;
    extern PyObject* gSegvException;
    extern PyObject* gIllException;
    extern PyObject* gAbrtException;
}


//- public helper ------------------------------------------------------------
CPyCppyy::PyCallArgs::~PyCallArgs() {
    if (fFlags & kSelfSwap)            // if self swap, fArgs has been offset by -1
        std::swap((PyObject*&)fSelf, ((PyObject**)fArgs)[0]);

#if PY_VERSION_HEX >= 0x03080000
    if (fFlags & kIsOffset) fArgs -= 1;

    if (fFlags & kDoItemDecref) {
        for (Py_ssize_t iarg = 0; iarg < CPyCppyy_PyArgs_GET_SIZE(fArgs, fNArgsf); ++iarg)
            Py_DECREF(fArgs[iarg]);
    }

    if (fFlags & kDoFree)
        PyMem_Free((void*)fArgs);
    else if (fFlags & kArgsSwap) {
    // if self swap, fArgs has been offset by -1
        int offset = (fFlags & kSelfSwap) ? 1 : 0;
        std::swap(((PyObject**)fArgs+offset)[0], ((PyObject**)fArgs+offset)[1]);
    }
#else
    if (fFlags & kDoDecref)
        Py_DECREF((PyObject*)fArgs);
    else if (fFlags & kArgsSwap)
        std::swap(PyTuple_GET_ITEM(fArgs, 0), PyTuple_GET_ITEM(fArgs, 1));
#endif
}


//- private helpers ----------------------------------------------------------
inline bool CPyCppyy::CPPMethod::VerifyArgCount_(Py_ssize_t actual)
{
// actual number of arguments must be between required and max args
    Py_ssize_t maxargs = (Py_ssize_t)fConverters.size();

    if (maxargs != actual) {
        if (actual < (Py_ssize_t)fArgsRequired) {
            SetPyError_(CPyCppyy_PyText_FromFormat(
                "takes at least %d arguments (%zd given)", fArgsRequired, actual));
            return false;
        } else if (maxargs < actual) {
            SetPyError_(CPyCppyy_PyText_FromFormat(
                "takes at most %zd arguments (%zd given)", maxargs, actual));
            return false;
        }
    }
    return true;
}

//----------------------------------------------------------------------------
inline void CPyCppyy::CPPMethod::Copy_(const CPPMethod& /* other */)
{
// fScope and fMethod handled separately

// do not copy caches
    fExecutor     = nullptr;
    fArgIndices   = nullptr;
    fArgsRequired = -1;
}

//----------------------------------------------------------------------------
inline void CPyCppyy::CPPMethod::Destroy_()
{
// destroy executor and argument converters
    if (fExecutor && fExecutor->HasState()) delete fExecutor;
    fExecutor = nullptr;

    for (auto p : fConverters) {
        if (p && p->HasState()) delete p;
    }
    fConverters.clear();

    delete fArgIndices; fArgIndices = nullptr;
    fArgsRequired = -1;
}

//----------------------------------------------------------------------------
inline PyObject* CPyCppyy::CPPMethod::ExecuteFast(
    void* self, ptrdiff_t offset, CallContext* ctxt)
{
// call into C++ through fExecutor; abstracted out from Execute() to prevent some
// code duplication with ProtectedCall()
    PyObject* result = nullptr;

    try {       // C++ try block
        result = fExecutor->Execute(fMethod, (Cppyy::TCppObject_t)((intptr_t)self+offset), ctxt);
    } catch (PyException&) {
        ctxt->fFlags |= CallContext::kPyException;
        result = nullptr;           // error already set
    } catch (std::exception& e) {
    // attempt to set the exception to the actual type, to allow catching with the Python C++ type
        static Cppyy::TCppType_t exc_type = (Cppyy::TCppType_t)Cppyy::GetScope("std::exception");

        ctxt->fFlags |= CallContext::kCppException;

        PyObject* pyexc_type = nullptr;
        PyObject* pyexc_obj  = nullptr;

    // TODO: factor this code with the same in ProxyWrappers (and cache it there to be able to
    // look up based on TCppType_t):
        Cppyy::TCppType_t actual = Cppyy::GetActualClass(exc_type, &e);
        const std::string& finalname = Cppyy::GetScopedFinalName(actual);
        const std::string& parentname = TypeManip::extract_namespace(finalname);
        PyObject* parent = CreateScopeProxy(parentname);
        if (parent) {
            pyexc_type = PyObject_GetAttrString(parent,
                parentname.empty() ? finalname.c_str() : finalname.substr(parentname.size()+2, std::string::npos).c_str());
            Py_DECREF(parent);
        }

        if (pyexc_type) {
        // create a copy of the exception (TODO: factor this code with the same in ProxyWrappers)
            PyObject* pyclass = CPyCppyy::GetScopeProxy(actual);
            PyObject* source = BindCppObjectNoCast(&e, actual);
            PyObject* pyexc_copy = PyObject_CallFunctionObjArgs(pyclass, source, nullptr);
            Py_DECREF(source);
            Py_DECREF(pyclass);
            if (pyexc_copy) {
                pyexc_obj = CPPExcInstance_Type.tp_new((PyTypeObject*)pyexc_type, nullptr, nullptr);
                ((CPPExcInstance*)pyexc_obj)->fCppInstance = (PyObject*)pyexc_copy;
            } else
                PyErr_Clear();
        } else
            PyErr_Clear();

        if (pyexc_type && pyexc_obj) {
            PyErr_SetObject(pyexc_type, pyexc_obj);
            Py_DECREF(pyexc_obj);
            Py_DECREF(pyexc_type);
        } else {
            PyErr_Format(PyExc_Exception, "%s (C++ exception)", e.what());
            Py_XDECREF(pyexc_obj);
            Py_XDECREF(pyexc_type);
        }

        result = nullptr;
    } catch (...) {
    // don't set the kCppException flag here, as there is basically no useful
    // extra information to be had and caller has to catch Exception either way
        PyErr_SetString(PyExc_Exception, "unhandled, unknown C++ exception");
        result = nullptr;
    }

// TODO: covers the PyException throw case, which does not seem to work on Windows, so
// instead leaves the error be
#ifdef _WIN32
    if (PyErr_Occurred()) {
        Py_XDECREF(result);
        result = nullptr;
    }
#endif

    return result;
}

//----------------------------------------------------------------------------
inline PyObject* CPyCppyy::CPPMethod::ExecuteProtected(
    void* self, ptrdiff_t offset, CallContext* ctxt)
{
// helper code to prevent some code duplication; this code embeds a "try/catch"
// block that saves the call environment for restoration in case of an otherwise
// fatal signal
    PyObject* result = 0;

    CLING_EXCEPTION_TRY {    // copy call environment to be able to jump back on signal
        result = ExecuteFast(self, offset, ctxt);
    } CLING_EXCEPTION_CATCH(excode) {
    // report any outstanding Python exceptions first
        if (PyErr_Occurred()) {
            std::cerr << "Python exception outstanding during C++ longjmp:" << std::endl;
            PyErr_Print();
            std::cerr << std::endl;
        }

    // unfortunately, the excodes are not the ones from signal.h, but enums from TSysEvtHandler.h
        if (excode == 0)
            PyErr_SetString(gBusException, "bus error in C++; program state was reset");
        else if (excode == 1)
            PyErr_SetString(gSegvException, "segfault in C++; program state was reset");
        else if (excode == 4)
            PyErr_SetString(gIllException, "illegal instruction in C++; program state was reset");
        else if (excode == 5)
            PyErr_SetString(gAbrtException, "abort from C++; program state was reset");
        else if (excode == 12)
            PyErr_SetString(PyExc_FloatingPointError, "floating point exception in C++; program state was reset");
        else
            PyErr_SetString(PyExc_SystemError, "problem in C++; program state was reset");
        result = 0;
    } CLING_EXCEPTION_ENDTRY;

    return result;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::InitConverters_()
{
// build buffers for argument dispatching
    const size_t nArgs = Cppyy::GetMethodNumArgs(fMethod);
    fConverters.resize(nArgs);

// setup the dispatch cache
    for (int iarg = 0; iarg < (int)nArgs; ++iarg) {
        const std::string& fullType = Cppyy::GetMethodArgType(fMethod, iarg);
        Converter* conv = CreateConverter(fullType);
        if (!conv) {
            PyErr_Format(PyExc_TypeError, "argument type %s not handled", fullType.c_str());
            return false;
        }

        fConverters[iarg] = conv;
    }

    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::InitExecutor_(Executor*& executor, CallContext* /* ctxt */)
{
// install executor conform to the return type
    executor = CreateExecutor(
        (bool)fMethod == true ? Cppyy::GetMethodResultType(fMethod) \
                              : Cppyy::GetScopedFinalName(fScope));

    if (!executor)
        return false;

    return true;
}

//----------------------------------------------------------------------------
std::string CPyCppyy::CPPMethod::GetSignatureString(bool fa)
{
// built a signature representation (used for doc strings)
    return Cppyy::GetMethodSignature(fMethod, fa);
}

//----------------------------------------------------------------------------
void CPyCppyy::CPPMethod::SetPyError_(PyObject* msg)
{
// Helper to report errors in a consistent format (derefs msg).
//
// Handles three cases:
//   1. No Python error occured yet:
//      Set a new TypeError with the message "msg" and the docstring of this
//      C++ method to give some context.
//   2. A C++ exception has occured:
//      Augment the exception message with the docstring of this method
//   3. A Python exception has occured:
//      Do nothing, Python exceptions are already informative enough

#if PY_VERSION_HEX >= 0x030c0000
    PyObject *evalue = PyErr_Occurred() ? PyErr_GetRaisedException() : nullptr;
    PyObject *etype = evalue ? (PyObject *)Py_TYPE(evalue) : nullptr;
#else
    PyObject *etype = nullptr;
    PyObject *evalue = nullptr;
    PyObject *etrace = nullptr;

    if (PyErr_Occurred()) {
        PyErr_Fetch(&etype, &evalue, &etrace);
    }
#endif

    const bool isCppExc = evalue && PyType_IsSubtype((PyTypeObject*)etype, &CPPExcInstance_Type);
 // If the error is not a CPPExcInstance, the error from Python itself is
 // already complete and messing with it would only make it less informative.
 // Just restore and return.
     if (evalue && !isCppExc) {
#if PY_VERSION_HEX >= 0x030c0000
        PyErr_SetRaisedException(evalue);
#else
        PyErr_Restore(etype, evalue, etrace);
#endif
        return;
     }

    PyObject* doc = GetDocString();
    const char* cdoc = CPyCppyy_PyText_AsString(doc);
    const char* cmsg = msg ? CPyCppyy_PyText_AsString(msg) : nullptr;
    PyObject* errtype = etype ? etype : PyExc_TypeError;
    PyObject* pyname = PyObject_GetAttr(errtype, PyStrings::gName);
    const char* cname = pyname ? CPyCppyy_PyText_AsString(pyname) : "Exception";

    if (!isCppExc) {
    // this is the case where no Python error has occured yet, and we set a new
    // error with context info
        PyErr_Format(errtype, "%s =>\n    %s: %s", cdoc, cname, cmsg ? cmsg : "");
    } else {
    // augment the top message with context information
        PyObject *&topMessage = ((CPPExcInstance*)evalue)->fTopMessage;
        Py_XDECREF(topMessage);
        if (msg) {
            topMessage = CPyCppyy_PyText_FromFormat("%s =>\n    %s: %s | ", cdoc, cname, cmsg);
        } else {
            topMessage = CPyCppyy_PyText_FromFormat("%s =>\n    %s: ", cdoc, cname);
        }
        // restore the updated error
#if PY_VERSION_HEX >= 0x030c0000
        PyErr_SetRaisedException(evalue);
#else
        PyErr_Restore(etype, evalue, etrace);
#endif
    }

    Py_XDECREF(pyname);
    Py_DECREF(doc);
    Py_XDECREF(msg);
}

//- constructors and destructor ----------------------------------------------
CPyCppyy::CPPMethod::CPPMethod(
        Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method) :
    fMethod(method), fScope(scope), fExecutor(nullptr), fArgIndices(nullptr),
    fArgsRequired(-1)
{
   // empty
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMethod::CPPMethod(const CPPMethod& other) :
    PyCallable(other), fMethod(other.fMethod), fScope(other.fScope)
{
    Copy_(other);
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMethod& CPyCppyy::CPPMethod::operator=(const CPPMethod& other)
{
    if (this != &other) {
        Destroy_();
        Copy_(other);
        fScope  = other.fScope;
        fMethod = other.fMethod;
    }

    return *this;
}

//----------------------------------------------------------------------------
CPyCppyy::CPPMethod::~CPPMethod()
{
    Destroy_();
}


//- public members -----------------------------------------------------------
/**
 * @brief Construct a Python string from the method's prototype
 * 
 * @param fa Show formal arguments of the method
 * @return PyObject* A Python string with the full method prototype, namespaces included.
 * 
 * For example, given:
 * 
 * int foo(int x);
 * 
 * namespace a {
 * namespace b {
 * namespace c {
 * int foo(int x);
 * }}}
 * 
 * This function returns:
 * 
 * 'int foo(int x)'
 * 'int a::b::c::foo(int x)'
 */
PyObject* CPyCppyy::CPPMethod::GetPrototype(bool fa)
{
    // Gather the fully qualified final scope of the method. This includes
    // all namespaces up to the one where the method is declared, for example:
    // namespace a { namespace b { void foo(); }}
    // gives
    // a::b
    std::string finalscope = Cppyy::GetScopedFinalName(fScope);
    return CPyCppyy_PyText_FromFormat("%s%s %s%s%s%s",
        (Cppyy::IsStaticMethod(fMethod) ? "static " : ""),
        Cppyy::GetMethodResultType(fMethod).c_str(),
        finalscope.c_str(),
        (finalscope.empty() ? "" : "::"), // Add final set of '::' if the method is scoped in namespace(s)
        Cppyy::GetMethodName(fMethod).c_str(),
        GetSignatureString(fa).c_str());
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::GetTypeName()
{
    PyObject* cppname = CPyCppyy_PyText_FromString(
        (GetReturnTypeName() + \
        " (" + (fScope ? Cppyy::GetScopedFinalName(fScope) + "::*)" : "*)")).c_str());
    CPyCppyy_PyText_AppendAndDel(&cppname, GetSignature(false /* show_formalargs */));
    return cppname;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::Reflex(Cppyy::Reflex::RequestId_t request, Cppyy::Reflex::FormatId_t format)
{
// C++ reflection tooling for methods.

    if (request == Cppyy::Reflex::RETURN_TYPE) {
        std::string rtn = GetReturnTypeName();
        Cppyy::TCppScope_t scope = 0;
        if (format == Cppyy::Reflex::OPTIMAL || format == Cppyy::Reflex::AS_TYPE)
            scope = Cppyy::GetScope(rtn);

        if (format == Cppyy::Reflex::AS_STRING || (format == Cppyy::Reflex::OPTIMAL && !scope))
            return CPyCppyy_PyText_FromString(rtn.c_str());
        else if (format == Cppyy::Reflex::AS_TYPE || format == Cppyy::Reflex::OPTIMAL) {
            if (scope) return CreateScopeProxy(scope);
            /* TODO: builtins as type */
        }
    }

    return PyCallable::Reflex(request, format);
}

//----------------------------------------------------------------------------
int CPyCppyy::CPPMethod::GetPriority()
{
// To help with overload selection, methods are given a priority based on the
// affinity of Python and C++ types. Priority only matters for methods that have
// an equal number of arguments and types that are possible substitutes (the
// normal selection mechanisms would simply distinguish them otherwise).

// The following types are ordered, in favor (variants implicit):
//
//  bool >> long >> int >> short
//  double >> long double >> float
//  const char* >> char
//
// Further, all integer types are preferred over floating point b/c int to float
// is allowed implicitly, float to int is not.
//
// Special cases that are disliked include void* and unknown/incomplete types.
// Also, moves are preferred over references. std::initializer_list is not a nice
// conversion candidate either, but needs to be higher priority to mix well with
// implicit conversions.
// TODO: extend this to favour classes that are not bases.
// TODO: profile this method (it's expensive, but should be called too often)

    int priority = 0;

    const size_t nArgs = Cppyy::GetMethodNumArgs(fMethod);
    for (int iarg = 0; iarg < (int)nArgs; ++iarg) {
        const std::string aname = Cppyy::GetMethodArgType(fMethod, iarg);

        if (Cppyy::IsBuiltin(aname)) {
        // complex type (note: double penalty: for complex and the template type)
            if (strstr(aname.c_str(), "std::complex"))
                priority -=   10;      // prefer double, float, etc. over conversion

        // integer types
            if (strstr(aname.c_str(), "bool"))
                priority +=    1;      // bool over int (does accept 1 and 0)
            else if (strstr(aname.c_str(), "long long"))
                priority +=   -5;      // will very likely fit
            else if (strstr(aname.c_str(), "long"))
                priority +=  -10;      // most affine integer type
            // no need to compare with int; leave at zero
            else if (strstr(aname.c_str(), "short"))
                priority +=  -50;      // not really relevant as a type

        // floating point types (note all numbers lower than integer types)
            else if (strstr(aname.c_str(), "float"))
                priority += -100;      // not really relevant as a type
            else if (strstr(aname.c_str(), "long double"))
                priority +=  -90;      // fits double with least loss of precision
            else if (strstr(aname.c_str(), "double"))
                priority +=  -80;      // most affine floating point type

        // string/char types
            else if (strstr(aname.c_str(), "char") && aname[aname.size()-1] != '*')
                priority += -60;       // prefer (const) char* over char

        // oddball
            else if (strstr(aname.c_str(), "void*"))
                priority -= 1000;      // void*/void** shouldn't be too greedy

        } else {
        // This is a user-defined type (class, struct, enum, etc.).

        // There's a bit of hysteresis here for templates: once GetScope() is called, their
        // IsComplete() succeeds, the other way around it does not. Since GetPriority() is
        // likely called several times in a sort, the GetScope() _must_ come first, or
        // different GetPriority() calls may return different results (since the 2nd time,
        // GetScope() will have been called from the first), killing the stable_sort.

        // prefer more derived classes
            const std::string& clean_name = TypeManip::clean_type(aname, false);
            Cppyy::TCppScope_t scope = Cppyy::GetScope(clean_name);
            if (scope)
                priority += static_cast<int>(Cppyy::GetNumBasesLongestBranch(scope));

            if (Cppyy::IsEnum(clean_name))
                priority -= 100;

        // a couple of special cases as explained above
            if (aname.find("initializer_list") != std::string::npos) {
                priority +=   150;     // needed for proper implicit conversion rules
            } else if (aname.rfind("&&", aname.size()-2) != std::string::npos) {
                priority +=   100;     // prefer moves over other ref/ptr
            } else if (scope && !Cppyy::IsComplete(clean_name)) {
            // class is known, but no dictionary available, 2 more cases: * and &
                if (aname[aname.size() - 1] == '&')
                    priority += -5000;
                else
                    priority += -2000; // prefer pointer passing over reference
            }
        }
    }

// prefer methods w/o optional arguments b/c ones with optional arguments are easier to
// select by providing the optional arguments explicitly
    priority += ((int)Cppyy::GetMethodReqArgs(fMethod) - (int)nArgs);

// add a small penalty to prefer non-const methods over const ones for get/setitem
    if (Cppyy::IsConstMethod(fMethod) && Cppyy::GetMethodName(fMethod) == "operator[]")
        priority += -10;

    return priority;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::IsGreedy()
{
// Methods will all void*-like arguments should be sorted after template
// instanstations, so that they don't greedily take over pointers to object.
// GetPriority() is too heavy-handed, as it will pull in all the argument
// types, so use this cheaper check.
    const size_t nArgs = Cppyy::GetMethodReqArgs(fMethod);
    if (!nArgs) return false;

    for (int iarg = 0; iarg < (int)nArgs; ++iarg) {
        const std::string aname = Cppyy::GetMethodArgType(fMethod, iarg);
        if (aname.find("void*") != 0)
            return false;
    }
    return true;
}


//----------------------------------------------------------------------------
int CPyCppyy::CPPMethod::GetMaxArgs()
{
    return (int)Cppyy::GetMethodNumArgs(fMethod);
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::GetCoVarNames()
{
// Build a tuple of the argument types/names.
    int co_argcount = (int)GetMaxArgs() /* +1 for self */;

// TODO: static methods need no 'self' (but is harmless otherwise)

    PyObject* co_varnames = PyTuple_New(co_argcount+1 /* self */);
    PyTuple_SET_ITEM(co_varnames, 0, CPyCppyy_PyText_FromString("self"));
    for (int iarg = 0; iarg < co_argcount; ++iarg) {
        std::string argrep = Cppyy::GetMethodArgType(fMethod, iarg);
        const std::string& parname = Cppyy::GetMethodArgName(fMethod, iarg);
        if (!parname.empty()) {
            argrep += " ";
            argrep += parname;
        }

        PyObject* pyspec = CPyCppyy_PyText_FromString(argrep.c_str());
        PyTuple_SET_ITEM(co_varnames, iarg+1, pyspec);
    }

    return co_varnames;
}

PyObject* CPyCppyy::CPPMethod::GetArgDefault(int iarg, bool silent)
{
// get and evaluate the default value (if any) of argument iarg of this method
    if (iarg >= (int)GetMaxArgs())
        return nullptr;

// borrowed reference to cppyy.gbl module to use its dictionary to eval in
    static PyObject* gbl = PyDict_GetItemString(PySys_GetObject((char*)"modules"), "cppyy.gbl");

    std::string defvalue = Cppyy::GetMethodArgDefault(fMethod, iarg);
    if (!defvalue.empty()) {
        PyObject** dctptr = _PyObject_GetDictPtr(gbl);
        if (!(dctptr && *dctptr))
            return nullptr;

        PyObject* gdct = *dctptr;
        PyObject* scope = nullptr;

        if (defvalue.rfind('(') != std::string::npos) {    // constructor-style call
        // try to tickle scope creation, just in case, first look in the scope where
        // the function lives, then in the global scope
            std::string possible_scope = defvalue.substr(0, defvalue.rfind('('));
            if (!Cppyy::IsBuiltin(possible_scope)) {
                std::string cand_scope = Cppyy::GetScopedFinalName(fScope)+"::"+possible_scope;
                scope = CreateScopeProxy(cand_scope);
                if (!scope) {
                    PyErr_Clear();
                // search within the global scope instead
                    scope = CreateScopeProxy(possible_scope);
                    if (!scope) PyErr_Clear();
                } else {
                // re-scope the scope; alternatively, the expression could be
                // compiled in the dictionary of the function's namespace, but
                // that would affect arguments passed to the constructor, too
                    defvalue = cand_scope + defvalue.substr(defvalue.rfind('('), std::string::npos);
                }
            }
        }

    // replace '::' -> '.'
        TypeManip::cppscope_to_pyscope(defvalue);

        if (!scope) {
        // a couple of common cases that python doesn't like (technically, 'L' is okay with older
        // pythons, but C long will always fit in Python int, so no need to bother)
            char c = defvalue.back();
            if (c == 'F' || c == 'D' || c == 'L') {
                int offset = 1;
                if (2 < defvalue.size() && defvalue[defvalue.size()-2] == 'U')
                    offset = 2;
                defvalue = defvalue.substr(0, defvalue.size()-offset);
            } else if (defvalue == "true") {
                defvalue = "True";
            } else if (defvalue == "false") {
                defvalue = "False";
            }
        }

    // attempt to evaluate the string representation (compilation is first to code to allow
    // the error message to indicate where it's coming from)
        PyObject* pyval = nullptr;

        PyObject* pycode = Py_CompileString((char*)defvalue.c_str(), "cppyy_default_compiler", Py_eval_input);
        if (pycode) {
            pyval = PyEval_EvalCode(
#if PY_VERSION_HEX < 0x03000000
                (PyCodeObject*)
#endif
                pycode, gdct, gdct);
            Py_DECREF(pycode);
        }

        if (!pyval && PyErr_Occurred() && silent) {
            PyErr_Clear();
            pyval = CPyCppyy_PyText_FromString(defvalue.c_str());    // allows continuation, but is likely to fail
        }

        Py_XDECREF(scope);
        return pyval;        // may be nullptr
    }

    PyErr_Format(PyExc_TypeError, "Could not construct default value for: %s", Cppyy::GetMethodArgName(fMethod, iarg).c_str());
    return nullptr;
}


bool CPyCppyy::CPPMethod::IsConst() {
    return Cppyy::IsConstMethod(GetMethod());
}


//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::GetScopeProxy()
{
// Get or build the scope of this method.
    return CreateScopeProxy(fScope);
}


//----------------------------------------------------------------------------
Cppyy::TCppFuncAddr_t CPyCppyy::CPPMethod::GetFunctionAddress()
{
// Return the C++ pointer of this function
    return Cppyy::GetFunctionAddress(fMethod, false /* don't check fast path envar */);
}

//----------------------------------------------------------------------------
int CPyCppyy::CPPMethod::GetArgMatchScore(PyObject* args_tuple)
{
    Py_ssize_t n = PyTuple_Size(args_tuple);

    int req_args = Cppyy::GetMethodReqArgs(fMethod);
    
    // Not enough arguments supplied: no match
    if (req_args > n)
        return INT_MAX;
    
    size_t score = 0;
    for (int i = 0; i < n; i++) {
        PyObject *pItem = PyTuple_GetItem(args_tuple, i);
        if(!CPyCppyy_PyText_Check(pItem)) {
            PyErr_SetString(PyExc_TypeError, "argument types should be in string format");
            return INT_MAX;
        }
        std::string req_type(CPyCppyy_PyText_AsString(pItem));

        size_t arg_score = Cppyy::CompareMethodArgType(fMethod, i, req_type);

        // Method is not compatible if even one argument does not match
        if (arg_score >= 10) {
            score = INT_MAX;
            break;
        }

        score += arg_score;
    }

    return score;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::Initialize(CallContext* ctxt)
{
// done if cache is already setup
    if (fArgsRequired != -1)
        return true;

    if (!InitConverters_())
        return false;

    if (!InitExecutor_(fExecutor, ctxt))
        return false;

// minimum number of arguments when calling
    fArgsRequired = (int)((bool)fMethod == true ? Cppyy::GetMethodReqArgs(fMethod) : 0);

    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::ProcessKwds(PyObject* self_in, PyCallArgs& cargs)
{
#if PY_VERSION_HEX >= 0x03080000
    if (!PyTuple_CheckExact(cargs.fKwds)) {
        SetPyError_(CPyCppyy_PyText_FromString("received unknown keyword names object"));
        return false;
    }
    Py_ssize_t nKeys = PyTuple_GET_SIZE(cargs.fKwds);
#else
    if (!PyDict_CheckExact(cargs.fKwds)) {
        SetPyError_(CPyCppyy_PyText_FromString("received unknown keyword arguments object"));
        return false;
    }
    Py_ssize_t nKeys = PyDict_Size(cargs.fKwds);
#endif

    if (nKeys == 0 && !self_in)
        return true;

    if (!fArgIndices) {
        fArgIndices = new std::map<std::string, int>{};
        for (int iarg = 0; iarg < (int)Cppyy::GetMethodNumArgs(fMethod); ++iarg)
            (*fArgIndices)[Cppyy::GetMethodArgName(fMethod, iarg)] = iarg;
    }

    Py_ssize_t nArgs = CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf) + (self_in ? 1 : 0);
    if (!VerifyArgCount_(nArgs+nKeys))
        return false;

    std::vector<PyObject*> vArgs{fConverters.size()};

// next, insert the keyword values
    PyObject *key, *value;
    Py_ssize_t maxpos = -1;

#if PY_VERSION_HEX >= 0x03080000
    Py_ssize_t npos_args = CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf);
    for (Py_ssize_t ikey = 0; ikey < nKeys; ++ikey) {
        key = PyTuple_GET_ITEM(cargs.fKwds, ikey);
        value = cargs.fArgs[npos_args+ikey];
#else
    Py_ssize_t pos = 0;
    while (PyDict_Next(cargs.fKwds, &pos, &key, &value)) {
#endif
        const char* ckey = CPyCppyy_PyText_AsStringChecked(key);
        if (!ckey)
            return false;

        auto p = fArgIndices->find(ckey);
        if (p == fArgIndices->end()) {
            SetPyError_(CPyCppyy_PyText_FromFormat("%s::%s got an unexpected keyword argument \'%s\'",
                Cppyy::GetFinalName(fScope).c_str(), Cppyy::GetMethodName(fMethod).c_str(), ckey));
            return false;
        }

        maxpos = p->second > maxpos ? p->second : maxpos;
        vArgs[p->second] = value;      // no INCREF yet for simple cleanup in case of error
    }

// if maxpos < nArgs, it will be detected & reported as a duplicate below
    Py_ssize_t maxargs = maxpos + 1;
    CPyCppyy_PyArgs_t newArgs = CPyCppyy_PyArgs_New(maxargs);

// set all values to zero to be able to check them later (this also guarantees normal
// cleanup by the tuple deallocation)
    for (Py_ssize_t i = 0; i < maxargs; ++i)
        CPyCppyy_PyArgs_SET_ITEM(newArgs, i, nullptr);

// fill out the positional arguments
    Py_ssize_t start = 0;
    if (self_in) {
        Py_INCREF(self_in);
        CPyCppyy_PyArgs_SET_ITEM(newArgs, 0, self_in);
        start = 1;
    }

    for (Py_ssize_t i = start; i < nArgs; ++i) {
        if (vArgs[i]) {
            SetPyError_(CPyCppyy_PyText_FromFormat("%s::%s got multiple values for argument %d",
                Cppyy::GetFinalName(fScope).c_str(), Cppyy::GetMethodName(fMethod).c_str(), (int)i+1));
            CPyCppyy_PyArgs_DEL(newArgs);
            return false;
        }

        PyObject* item = CPyCppyy_PyArgs_GET_ITEM(cargs.fArgs, i);
        Py_INCREF(item);
        CPyCppyy_PyArgs_SET_ITEM(newArgs, i, item);
    }

// fill out the keyword arguments
    for (Py_ssize_t i = nArgs; i < maxargs; ++i) {
        PyObject* item = vArgs[i];
        if (item) {
            Py_INCREF(item);
            CPyCppyy_PyArgs_SET_ITEM(newArgs, i, item);
        } else {
        // try retrieving the default
            item = GetArgDefault((int)i, false /* i.e. not silent */);
            if (!item) {
                CPyCppyy_PyArgs_DEL(newArgs);
                return false;
            }
            CPyCppyy_PyArgs_SET_ITEM(newArgs, i, item);
        }
    }

#if PY_VERSION_HEX >= 0x03080000
    if (cargs.fFlags & PyCallArgs::kDoFree) {
        if (cargs.fFlags & PyCallArgs::kIsOffset)
            cargs.fArgs -= 1;
#else
    if (cargs.fFlags & PyCallArgs::kDoDecref) {
#endif
       CPyCppyy_PyArgs_DEL(cargs.fArgs);
    }

    cargs.fArgs = newArgs;
    cargs.fNArgsf = maxargs;
#if PY_VERSION_HEX >= 0x03080000
    cargs.fFlags = PyCallArgs::kDoFree | PyCallArgs::kDoItemDecref;
#else
    cargs.fFlags = PyCallArgs::kDoDecref;
#endif

    return true;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::ProcessArgs(PyCallArgs& cargs)
{
// verify existence of self, return if ok
    if (cargs.fSelf) {
        if (cargs.fKwds) { return ProcessKwds(nullptr, cargs); }
        return true;
    }

// otherwise, check for a suitable 'self' in args and update accordingly
    if (CPyCppyy_PyArgs_GET_SIZE(cargs.fArgs, cargs.fNArgsf) != 0) {
        CPPInstance* pyobj = (CPPInstance*)CPyCppyy_PyArgs_GET_ITEM(cargs.fArgs, 0);

    // demand CPyCppyy object, and an argument that may match down the road
        if (CPPInstance_Check(pyobj)) {
            Cppyy::TCppType_t oisa = pyobj->ObjectIsA();
            if (fScope == Cppyy::gGlobalScope ||                // free global
                oisa == 0 ||                                    // null pointer or ctor call
                oisa == fScope ||                               // matching types
                Cppyy::IsSubtype(oisa, fScope)) {               // id.

            // reset self
                Py_INCREF(pyobj);      // corresponding Py_DECREF is in CPPOverload
                cargs.fSelf = pyobj;

            // offset args by 1
#if PY_VERSION_HEX >= 0x03080000
                cargs.fArgs += 1;
                cargs.fFlags |= PyCallArgs::kIsOffset;
#else
                if (cargs.fFlags & PyCallArgs::kDoDecref)
                    Py_DECREF((PyObject*)cargs.fArgs);
                cargs.fArgs = PyTuple_GetSlice(cargs.fArgs, 1, PyTuple_GET_SIZE(cargs.fArgs));
                cargs.fFlags |= PyCallArgs::kDoDecref;
#endif
                cargs.fNArgsf -= 1;

            // put the keywords, if any, in their places in the arguments array
                if (cargs.fKwds)
                    return ProcessKwds(nullptr, cargs);
                return true;
            }
        }
    }

// no self, set error and lament
    SetPyError_(CPyCppyy_PyText_FromFormat(
        "unbound method %s::%s must be called with a %s instance as first argument",
        Cppyy::GetFinalName(fScope).c_str(), Cppyy::GetMethodName(fMethod).c_str(),
        Cppyy::GetFinalName(fScope).c_str()));
    return false;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::ConvertAndSetArgs(CPyCppyy_PyArgs_t args, size_t nargsf, CallContext* ctxt)
{
    Py_ssize_t argc = CPyCppyy_PyArgs_GET_SIZE(args, nargsf);
    if (!VerifyArgCount_(argc))
        return false;

// pass current scope for which the call is made
    ctxt->fCurScope = fScope;

    if (argc == 0)
        return true;

// convert the arguments to the method call array
    bool isOK = true;
    Parameter* cppArgs = ctxt->GetArgs(argc);
    for (int i = 0; i < (int)argc; ++i) {
        if (!fConverters[i]->SetArg(CPyCppyy_PyArgs_GET_ITEM(args, i), cppArgs[i], ctxt)) {
            SetPyError_(CPyCppyy_PyText_FromFormat("could not convert argument %d", i+1));
            isOK = false;
            break;
        }
    }

    return isOK;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::Execute(void* self, ptrdiff_t offset, CallContext* ctxt)
{
// call the interface method
    PyObject* result = 0;

    if (CallContext::sSignalPolicy != CallContext::kProtected && \
        !(ctxt->fFlags & CallContext::kProtected)) {
    // bypasses try block (i.e. segfaults will abort)
        result = ExecuteFast(self, offset, ctxt);
    } else {
    // at the cost of ~10% performance, don't abort the interpreter on any signal
        result = ExecuteProtected(self, offset, ctxt);
    }

    if (!result && PyErr_Occurred())
        SetPyError_(0);

    return result;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::Call(CPPInstance*& self,
    CPyCppyy_PyArgs_t args, size_t nargsf, PyObject* kwds, CallContext* ctxt)
{
// setup as necessary
    if (fArgsRequired == -1 && !Initialize(ctxt))
        return nullptr;

// fetch self, verify, and put the arguments in usable order
    PyCallArgs cargs{self, args, nargsf, kwds};
    if (!ProcessArgs(cargs))
        return nullptr;

// self provides the python context for lifelines
    if (!ctxt->fPyContext)
        ctxt->fPyContext = (PyObject*)cargs.fSelf;    // no Py_INCREF as no ownership

// translate the arguments
    if (fArgsRequired || CPyCppyy_PyArgs_GET_SIZE(args, cargs.fNArgsf)) {
        if (!ConvertAndSetArgs(cargs.fArgs, cargs.fNArgsf, ctxt))
            return nullptr;
    }

// get the C++ object that this object proxy is a handle for
    void* object = self->GetObject();

// validity check that should not fail
    if (!object) {
        PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
        return nullptr;
    }

// get its class
    Cppyy::TCppType_t derived = self->ObjectIsA();

// calculate offset (the method expects 'this' to be an object of fScope)
    ptrdiff_t offset = 0;
    if (derived && derived != fScope)
        offset = Cppyy::GetBaseOffset(derived, fScope, object, 1 /* up-cast */);

// actual call; recycle self instead of returning new object for same address objects
    CPPInstance* pyobj = (CPPInstance*)Execute(object, offset, ctxt);
    if (CPPInstance_Check(pyobj) &&
            derived && pyobj->ObjectIsA() == derived &&
            pyobj->GetObject() == object) {
        Py_INCREF((PyObject*)self);
        Py_DECREF(pyobj);
        return (PyObject*)self;
    }

    return (PyObject*)pyobj;
}

//- protected members --------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::GetSignature(bool fa)
{
// construct python string from the method's signature
    return CPyCppyy_PyText_FromString(GetSignatureString(fa).c_str());
}

//----------------------------------------------------------------------------
std::string CPyCppyy::CPPMethod::GetReturnTypeName()
{
    return Cppyy::GetMethodResultType(fMethod);
}
