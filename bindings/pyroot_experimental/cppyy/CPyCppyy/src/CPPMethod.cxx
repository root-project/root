// Bindings
#include "CPyCppyy.h"
#include "CPPMethod.h"
#include "CPPInstance.h"
#include "Converters.h"
#include "Executors.h"
#include "ProxyWrappers.h"
#include "PyStrings.h"
#include "TypeManip.h"
#include "Utility.h"

#include "CPyCppyy/TPyException.h"

// Standard
#include <assert.h>
#include <string.h>
#include <exception>
#include <sstream>
#include <string>
#include <typeinfo>
#include <memory>


//- data and local helpers ---------------------------------------------------
namespace CPyCppyy {
    extern PyObject* gThisModule;
}


//- private helpers ----------------------------------------------------------
inline void CPyCppyy::CPPMethod::Copy_(const CPPMethod& /* other */)
{
// fScope and fMethod handled separately

// do not copy caches
    fExecutor     = nullptr;
    fArgsRequired = -1;

// being uninitialized will trigger setting up caches as appropriate
    fIsInitialized = false;
}

//----------------------------------------------------------------------------
inline void CPyCppyy::CPPMethod::Destroy_() const
{
// destroy executor and argument converters
    delete fExecutor;

    for (int i = 0; i < (int)fConverters.size(); ++i)
        delete fConverters[i];
}

//----------------------------------------------------------------------------
inline PyObject* CPyCppyy::CPPMethod::CallFast(
    void* self, ptrdiff_t offset, CallContext* ctxt)
{
// Helper code to prevent some duplication; this is called from CallSafe() as well
// as directly from CPPMethod::Execute in fast mode.
    PyObject* result = nullptr;

    try {       // C++ try block
        result = fExecutor->Execute(fMethod, (Cppyy::TCppObject_t)((intptr_t)self+offset), ctxt);
    } catch (TPyException&) {
        result = nullptr;           // error already set
    } catch (std::exception& e) {
    /* TODO: figure out what this is about ... ?
        if (gInterpreter->DiagnoseIfInterpreterException(e)) {
           return result;
        }

        // TODO: write w/o use of TClass

    // map user exceptions .. this needs to move to Cppyy.cxx
        TClass* cl = TClass::GetClass(typeid(e));

        PyObject* pyUserExcepts = PyObject_GetAttrString(gThisModule, "UserExceptions");
        std::string exception_type;
        if (cl) exception_type = cl->GetName();
        else {
            int errorCode;
            std::unique_ptr<char[]> demangled(TClassEdit::DemangleTypeIdName(typeid(e),errorCode));
            if (errorCode) exception_type = typeid(e).name();
            else exception_type = demangled.get();
        }
        PyObject* pyexc = PyDict_GetItemString(pyUserExcepts, exception_type.c_str());
        if (!pyexc) {
            PyErr_Clear();
            pyexc = PyDict_GetItemString(pyUserExcepts, ("std::"+exception_type).c_str());
        }
        Py_DECREF(pyUserExcepts);

        if (pyexc) {
            PyErr_Format(pyexc, "%s", e.what());
        } else {
            PyErr_Format(PyExc_Exception, "%s (C++ exception of type %s)", e.what(), exception_type.c_str());
        }
    */

        PyErr_Format(PyExc_Exception, "%s (C++ exception)", e.what());
        result = nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_Exception, "unhandled, unknown C++ exception");
        result = nullptr;
    }

// TODO: covers the TPyException throw case, which does not seem to work on Windows, so
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
inline PyObject* CPyCppyy::CPPMethod::CallSafe(
    void* self, ptrdiff_t offset, CallContext* ctxt)
{
// Helper code to prevent some code duplication; this code embeds a "try/catch"
// block that saves the stack for restoration in case of an otherwise fatal signal.
    PyObject* result = 0;

//   TRY {       // ROOT "try block"
    result = CallFast(self, offset, ctxt);
   //   } CATCH(excode) {
   //      PyErr_SetString(PyExc_SystemError, "problem in C++; program state has been reset");
   //      result = 0;
   //      Throw(excode);
   //   } ENDTRY;

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
    std::stringstream sig; sig << "(";
    int count = 0;
    const size_t nArgs = Cppyy::GetMethodNumArgs(fMethod);
    for (int iarg = 0; iarg < (int)nArgs; ++iarg) {
        if (count) sig << (fa ? ", " : ",");

        sig << Cppyy::GetMethodArgType(fMethod, iarg);

        if (fa) {
            const std::string& parname = Cppyy::GetMethodArgName(fMethod, iarg);
            if (!parname.empty())
                sig << " " << parname;

            const std::string& defvalue = Cppyy::GetMethodArgDefault(fMethod, iarg);
            if (!defvalue.empty())
                sig << " = " << defvalue;
        }
        count++;
    }
    sig << ")";
    return sig.str();
}

//----------------------------------------------------------------------------
void CPyCppyy::CPPMethod::SetPyError_(PyObject* msg)
{
// helper to report errors in a consistent format (derefs msg)
    PyObject *etype, *evalue, *etrace;
    PyErr_Fetch(&etype, &evalue, &etrace);

    std::string details = "";
    if (evalue) {
        PyObject* descr = PyObject_Str(evalue);
        if (descr) {
            details = CPyCppyy_PyUnicode_AsString(descr);
            Py_DECREF(descr);
        }
    }

    Py_XDECREF(evalue); Py_XDECREF(etrace);

    PyObject* doc = GetDocString();
    PyObject* errtype = etype;
    if (!errtype) {
        Py_INCREF(PyExc_TypeError);
        errtype = PyExc_TypeError;
    }
    PyObject* pyname = PyObject_GetAttr(errtype, PyStrings::gName);
    const char* cname = pyname ? CPyCppyy_PyUnicode_AsString(pyname) : "Exception";

    if (details.empty()) {
        PyErr_Format(errtype, "%s =>\n    %s: %s", CPyCppyy_PyUnicode_AsString(doc),
            cname, msg ? CPyCppyy_PyUnicode_AsString(msg) : "");
    } else if (msg) {
        PyErr_Format(errtype, "%s =>\n    %s: %s (%s)",
            CPyCppyy_PyUnicode_AsString(doc), cname, CPyCppyy_PyUnicode_AsString(msg),
            details.c_str());
    } else {
        PyErr_Format(errtype, "%s =>\n    %s: %s",
            CPyCppyy_PyUnicode_AsString(doc), cname, details.c_str());
    }

    Py_XDECREF(pyname);
    Py_XDECREF(etype);
    Py_DECREF(doc);
    Py_XDECREF(msg);
}

//- constructors and destructor ----------------------------------------------
CPyCppyy::CPPMethod::CPPMethod(
        Cppyy::TCppScope_t scope, Cppyy::TCppMethod_t method) :
    fMethod(method), fScope(scope), fExecutor(nullptr), fArgsRequired(-1),
    fIsInitialized(false)
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
PyObject* CPyCppyy::CPPMethod::GetPrototype(bool fa)
{
// construct python string from the method's prototype
    return CPyCppyy_PyUnicode_FromFormat("%s%s %s::%s%s",
        (Cppyy::IsStaticMethod(fMethod) ? "static " : ""),
        Cppyy::GetMethodResultType(fMethod).c_str(),
        Cppyy::GetScopedFinalName(fScope).c_str(), Cppyy::GetMethodName(fMethod).c_str(),
        GetSignatureString(fa).c_str());
}

//----------------------------------------------------------------------------
int CPyCppyy::CPPMethod::GetPriority()
{
// Method priorities exist (in lieu of true overloading) there to prevent
// void* or <unknown>* from usurping otherwise valid calls. TODO: extend this
// to favour classes that are not bases.
    int priority = 0;

    const size_t nArgs = Cppyy::GetMethodNumArgs(fMethod);
    for (int iarg = 0; iarg < (int)nArgs; ++iarg) {
        const std::string aname = Cppyy::GetMethodArgType(fMethod, iarg);

    // the following numbers are made up and may cause problems in specific
    // situations: use <obj>.<meth>.disp() for choice of exact dispatch
        if (Cppyy::IsBuiltin(aname)) {
        // happens for builtin types (and namespaces, but those can never be an
        // argument), NOT for unknown classes as that concept no longer exists
            if (strstr(aname.c_str(), "void*"))
            // TODO: figure out in general all void* converters
                priority -= 10000;     // void*/void** shouldn't be too greedy
            else if (strstr(aname.c_str(), "float"))
                priority -= 1000;      // double preferred (no float in python)
            else if (strstr(aname.c_str(), "long double"))
                priority -= 100;       // id, but better than float
            else if (strstr(aname.c_str(), "double"))
                priority -= 10;        // char, int, long can't convert float,
                                       // but vv. works, so prefer the int types
            else if (strstr(aname.c_str(), "bool"))
                priority += 1;         // bool over int (does accept 1 and 0)

        } else if (aname.find("initializer_list") != std::string::npos) {
        // difficult conversion, push it way down
            priority -= 2000;
        } else if (aname.rfind("&&", aname.size()-2) != std::string::npos) {
            priority += 100;
        } else if (!aname.empty() && !Cppyy::IsComplete(aname)) {
        // class is known, but no dictionary available, 2 more cases: * and &
            if (aname[ aname.size() - 1 ] == '&')
                priority -= 1000000;
            else
                priority -= 100000; // prefer pointer passing over reference
        }

    // prefer more derived classes
        Cppyy::TCppScope_t scope = Cppyy::GetScope(TypeManip::clean_type(aname));
        if (scope)
            priority += (int)Cppyy::GetNumBases(scope);
    }

// add a small penalty to prefer non-const methods over const ones for
// getitem/setitem
    if (Cppyy::IsConstMethod(fMethod) && Cppyy::GetMethodName(fMethod) == "operator[]")
        priority -= 1;

    return priority;
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
    PyTuple_SET_ITEM(co_varnames, 0, CPyCppyy_PyUnicode_FromString("self"));
    for (int iarg = 0; iarg < co_argcount; ++iarg) {
        std::string argrep = Cppyy::GetMethodArgType(fMethod, iarg);
        const std::string& parname = Cppyy::GetMethodArgName(fMethod, iarg);
        if (!parname.empty()) {
            argrep += " ";
            argrep += parname;
        }

        PyObject* pyspec = CPyCppyy_PyUnicode_FromString(argrep.c_str());
        PyTuple_SET_ITEM(co_varnames, iarg+1, pyspec);
    }

    return co_varnames;
}

PyObject* CPyCppyy::CPPMethod::GetArgDefault(int iarg)
{
// get the default value (if any) of argument iarg of this method
    if (iarg >= (int)GetMaxArgs())
        return nullptr;

    const std::string& defvalue = Cppyy::GetMethodArgDefault(fMethod, iarg);
    if (!defvalue.empty()) {

    // attempt to evaluate the string representation (will work for all builtin types)
        PyObject* pyval = (PyObject*)PyRun_String(
            (char*)defvalue.c_str(), Py_eval_input, gThisModule, gThisModule);
        if (!pyval && PyErr_Occurred()) {
            PyErr_Clear();
            return CPyCppyy_PyUnicode_FromString(defvalue.c_str());
        }

        return pyval;
    }

    return nullptr;
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
    return Cppyy::GetFunctionAddress(fMethod);
}


//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::Initialize(CallContext* ctxt)
{
// done if cache is already setup
    if (fIsInitialized == true)
        return true;

    if (!InitConverters_())
        return false;

    if (!InitExecutor_(fExecutor, ctxt))
        return false;

// minimum number of arguments when calling
    fArgsRequired = (bool)fMethod == true ? Cppyy::GetMethodReqArgs(fMethod) : 0;

// init done
    fIsInitialized = true;

    return true;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::PreProcessArgs(
        CPPInstance*& self, PyObject* args, PyObject*)
{
// verify existence of self, return if ok
    if (self) {
        Py_INCREF(args);
        return args;
    }

// otherwise, check for a suitable 'self' in args and update accordingly
    if (PyTuple_GET_SIZE(args) != 0) {
        CPPInstance* pyobj = (CPPInstance*)PyTuple_GET_ITEM(args, 0);

    // demand CPyCppyy object, and an argument that may match down the road
        if (CPPInstance_Check(pyobj) &&
             (fScope == Cppyy::gGlobalScope ||                  // free global
             (pyobj->ObjectIsA() == 0)     ||                   // null pointer or ctor call
             (Cppyy::IsSubtype(pyobj->ObjectIsA(), fScope)))) { // matching types

        // reset self
            Py_INCREF(pyobj);      // corresponding Py_DECREF is in CPPOverload
            self = pyobj;

        // offset args by 1 (new ref)
            return PyTuple_GetSlice(args, 1, PyTuple_GET_SIZE(args));
        }
    }

// no self, set error and lament
    SetPyError_(CPyCppyy_PyUnicode_FromFormat(
        "unbound method %s::%s must be called with a %s instance as first argument",
        Cppyy::GetFinalName(fScope).c_str(), Cppyy::GetMethodName(fMethod).c_str(),
        Cppyy::GetFinalName(fScope).c_str()));
    return nullptr;
}

//----------------------------------------------------------------------------
bool CPyCppyy::CPPMethod::ConvertAndSetArgs(PyObject* args, CallContext* ctxt)
{
    Py_ssize_t argc = PyTuple_GET_SIZE(args);
    Py_ssize_t argMax = (Py_ssize_t)fConverters.size();

// argc must be between min and max number of arguments
    if (argc < fArgsRequired) {
        SetPyError_(CPyCppyy_PyUnicode_FromFormat(
            "takes at least %ld arguments (%ld given)", fArgsRequired, argc));
        return false;
    } else if (argMax < argc) {
        SetPyError_(CPyCppyy_PyUnicode_FromFormat(
            "takes at most %ld arguments (%ld given)", argMax, argc));
        return false;
    }

// convert the arguments to the method call array
    bool isOK = true;
    Parameter* cppArgs = ctxt->GetArgs(argc);
    for (int i = 0; i < (int)argc; ++i) {
        if (!fConverters[i]->SetArg(PyTuple_GET_ITEM(args, i), cppArgs[i], ctxt)) {
            SetPyError_(CPyCppyy_PyUnicode_FromFormat("could not convert argument %d", i+1));
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

    if (CallContext::sSignalPolicy == CallContext::kFast) {
    // bypasses try block (i.e. segfaults will abort)
        result = CallFast(self, offset, ctxt);
    } else {
    // at the cost of ~10% performance, don't abort the interpreter on any signal
        result = CallSafe(self, offset, ctxt);
    }

// TODO: the following is dreadfully slow and dead-locks on Apache: revisit
// raising exceptions through callbacks by using magic returns
//    if (result && Utility::PyErr_Occurred_WithGIL()) {
//    // can happen in the case of a CINT error: trigger exception processing
//        Py_DECREF(result);
//        result = 0;
//    } else if (!result && PyErr_Occurred())
    if (!result && PyErr_Occurred())
        SetPyError_(0);

    return result;
}

//----------------------------------------------------------------------------
PyObject* CPyCppyy::CPPMethod::Call(
        CPPInstance*& self, PyObject* args, PyObject* kwds, CallContext* ctxt)
{
// preliminary check in case keywords are accidently used (they are ignored otherwise)
    if (kwds && PyDict_Size(kwds)) {
        PyErr_SetString(PyExc_TypeError, "keyword arguments are not yet supported");
        return nullptr;
    }

// setup as necessary
    if (!fIsInitialized && !Initialize(ctxt))
        return nullptr;

// fetch self, verify, and put the arguments in usable order
    if (!(args = PreProcessArgs(self, args, kwds)))
        return nullptr;

// translate the arguments
    if (fArgsRequired || PyTuple_GET_SIZE(args)) {
        if (!ConvertAndSetArgs(args, ctxt)) {
            Py_DECREF(args);
            return nullptr;
        }
    }

// get the C++ object that this object proxy is a handle for
    void* object = self->GetObject();

// validity check that should not fail
    if (!object) {
        PyErr_SetString(PyExc_ReferenceError, "attempt to access a null-pointer");
        Py_DECREF(args);
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
    Py_DECREF(args);

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
    return CPyCppyy_PyUnicode_FromString(GetSignatureString(fa).c_str());
}

//----------------------------------------------------------------------------
std::string CPyCppyy::CPPMethod::GetReturnTypeName()
{
    return Cppyy::GetMethodResultType(fMethod);
}
