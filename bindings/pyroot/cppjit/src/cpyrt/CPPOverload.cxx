// Bindings
#include "cpyrt.h"

using namespace cppjit;
#include "CPPInstance.h"
#include "CPPMethod.h"
#include "CPPOverload.h"
#include "CallContext.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"
#include "PyCallable.h"
#include "PyStrings.h"
#include "Utility.h"
#include "cppjit_interop.h"
#include "structmember.h" // from Python
#include "cpyrt/Reflex.h"

// Standard
#include <algorithm>
#include <sstream>
#include <vector>

namespace cppjit::cpyrt {

namespace {

// from CPython's instancemethod: Free list for method objects to safe
// malloc/free overhead The fSelf field is used to chain the elements.
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

  TPythonCallback(PyObject* callable) : fCallable(nullptr) {
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
    return PyUnicode_FromString("*args, **kwargs");
  }
  PyObject* GetSignatureNames() override { return PyTuple_New(0); }
  PyObject* GetSignatureTypes() override { return PyTuple_New(0); }
  PyObject* GetPrototype(bool /*show_formalargs*/ = true) override {
    return PyUnicode_FromString("<callback>");
  }
  PyObject* GetDocString() override {
    if (PyObject_HasAttrString(fCallable, "__doc__")) {
      return PyObject_GetAttrString(fCallable, "__doc__");
    } else {
      return GetPrototype();
    }
  }

  int GetMaxArgs() override { return 100; };
  PyObject* GetCoVarNames() override { // TODO: pick these up from the callable
    Py_RETURN_NONE;
  }
  PyObject* GetArgDefault(int /* iarg */, bool /* silent */ = true) override {
    Py_RETURN_NONE; // TODO: pick these up from the callable
  }

  PyObject* GetScopeProxy() override { // should this be the module ??
    Py_RETURN_NONE;
  }

  interop::TCppFuncAddr_t GetFunctionAddress() override {
    return (interop::TCppFuncAddr_t) nullptr;
  }

  PyCallable* Clone() override { return new TPythonCallback(*this); }

  PyObject* Call(CPPInstance*& self, cpyrt_PyArgs_t args, size_t nargsf,
                 PyObject* kwds, CallContext* /* ctxt = 0 */) override {

    if (self) {
      if (nargsf & PY_VECTORCALL_ARGUMENTS_OFFSET) { // mutation allowed?
        std::swap(((PyObject**)args - 1)[0], (PyObject*&)self);
        nargsf &= ~PY_VECTORCALL_ARGUMENTS_OFFSET;
        args = args - 1;
      } else {
        Py_ssize_t nkwargs = kwds ? PyTuple_GET_SIZE(kwds) : 0;
        Py_ssize_t totalargs = PyVectorcall_NARGS(nargsf) + nkwargs;
        PyObject** newArgs =
            (PyObject**)PyMem_Malloc((totalargs + 1) * sizeof(PyObject*));
        if (!newArgs)
          return nullptr;

        newArgs[0] = (PyObject*)self;
        if (0 < totalargs)
          memcpy((void*)&newArgs[1], args, totalargs * sizeof(PyObject*));
        args = newArgs;
      }
      nargsf += 1;
    }

    PyObject* result = cpyrt_PyObject_Call(fCallable, args, nargsf, kwds);
    if (self) {
      if (nargsf & PY_VECTORCALL_ARGUMENTS_OFFSET)
        std::swap(((PyObject**)args - 1)[0], (PyObject*&)self);
      else
        PyMem_Free((void*)args);
    }
    return result;
  }
};

// helper to test whether a method is used in a pseudo-function modus
static inline bool IsPseudoFunc(CPPOverload* pymeth) {
  return pymeth->fMethodInfo->fFlags & CallContext::kIsPseudoFunc;
}

// return helper
static inline void ResetCallState(CPPInstance* descr_self,
                                  CPPInstance*& im_self) {
  // reset self if needed, allowing simple re-use
  if (descr_self != im_self) {
    Py_XDECREF(im_self);
    im_self = descr_self;
  }
}

// helper to factor out return logic of mp_call / mp_vectorcall
static inline PyObject* HandleReturn(CPPOverload* pymeth, CPPInstance* im_self,
                                     PyObject* result) {
  // special case for python exceptions, propagated through C++ layer
  if (result) {
    CPPInstance* cppres =
        (CPPInstance*)(CPPInstance_Check(result) ? result : nullptr);

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
          // if self was a by-value return and result is not, pro-actively
          // protect result; else if the return value falls within the memory of
          // 'this', force a lifeline
          if (!(cppres->fFlags &
                CPPInstance::kIsValue)) { // no need if the result is temporary
            if (im_self->fFlags & CPPInstance::kIsValue)
              ll_action = 2;
            else if (im_self->fFlags & CPPInstance::kHasLifeLine)
              ll_action = 3;
            else {
              ptrdiff_t offset = (ptrdiff_t)cppres->GetObject() -
                                 (ptrdiff_t)im_self->GetObject();
              if (0 <= offset &&
                  offset < (ptrdiff_t)interop::SizeOf(im_self->ObjectIsA()))
                ll_action = 4;
            }
          }
        }
      }

      if (!ll_action)
        pymeth->fMethodInfo->fFlags |=
            CallContext::kNeverLifeLine; // assume invariant semantics
      else {
        if (PyObject_SetAttr(result, PyStrings::gLifeLine,
                             (PyObject*)im_self) == -1)
          PyErr_Clear(); // ignored
        if (cppres)
          cppres->fFlags |= CPPInstance::kHasLifeLine; // for chaining
        pymeth->fMethodInfo->fFlags |=
            CallContext::kSetLifeLine; // for next time
      }
    }
  }

  // reset self as necessary to allow re-use of the CPPOverload
  ResetCallState(pymeth->fSelf, im_self);

  return result;
}

//= cpyrt method proxy object behaviour ===================================
static PyObject* mp_name(CPPOverload* pymeth, void*) {
  return PyUnicode_FromString(pymeth->GetName().c_str());
}

//----------------------------------------------------------------------------
static PyObject* mp_module(CPPOverload* /* pymeth */, void*) {
  Py_INCREF(PyStrings::gThisModule);
  return PyStrings::gThisModule;
}

//----------------------------------------------------------------------------
static PyObject* mp_doc(CPPOverload* pymeth, void*) {
  if (pymeth->fMethodInfo->fDoc) {
    Py_INCREF(pymeth->fMethodInfo->fDoc);
    return pymeth->fMethodInfo->fDoc;
  }

  // Build python document string ('__doc__') from all C++-side overloads.
  CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

  // collect doc strings
  CPPOverload::Methods_t::size_type nMethods = methods.size();
  if (nMethods == 0) // from template proxy with no instantiations
    return nullptr;
  PyObject* doc = methods[0]->GetDocString();

  // simple case
  if (nMethods == 1)
    return doc;

  // overloaded method
  PyObject* separator = PyUnicode_FromString("\n");
  for (CPPOverload::Methods_t::size_type i = 1; i < nMethods; ++i) {
    PyUnicode_Append(&doc, separator);
    PyUnicode_AppendAndDel(&doc, methods[i]->GetDocString());
  }
  Py_DECREF(separator);

  return doc;
}

static int mp_doc_set(CPPOverload* pymeth, PyObject* val, void*) {
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
static PyObject* mp_func_overloads_names(CPPOverload* pymeth) {

  const CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

  PyObject* overloads_names_dict = PyDict_New();

  for (PyCallable* method : methods) {
    PyDict_SetItem(overloads_names_dict, method->GetPrototype(),
                   method->GetSignatureNames());
  }

  return overloads_names_dict;
}

/**
 * @brief Returns a dictionary with the types of all overloads.
 *
 * This dictionary may look like:
 *
 * {'double ::foo(int a, float b, double c)': {'input_types': ('int', 'float',
 * 'double'), 'return_type': 'double'}, 'float ::foo(float b)': {'input_types':
 * ('float',), 'return_type': 'float'}, 'int ::foo(int a)': {'input_types':
 * ('int',), 'return_type': 'int'}, 'int ::foo(int a, float b)': {'input_types':
 * ('int', 'float'), 'return_type': 'int'}}
 *
 * Each overload's value dict also carries 'is_const': the method's own
 * const-qualification (always False for free functions and static methods).
 */
static PyObject* mp_func_overloads_types(CPPOverload* pymeth) {

  const CPPOverload::Methods_t& methods = pymeth->fMethodInfo->fMethods;

  PyObject* overloads_types_dict = PyDict_New();

  for (PyCallable* method : methods) {
    PyDict_SetItem(overloads_types_dict, method->GetPrototype(),
                   method->GetSignatureTypes());
  }

  return overloads_types_dict;
}

//----------------------------------------------------------------------------
static PyObject* mp_meth_func(CPPOverload* pymeth, void*) {
  // Create a new method proxy to be returned.
  CPPOverload* newPyMeth =
      (CPPOverload*)CPPOverload_Type.tp_alloc(&CPPOverload_Type, 0);

  // method info is shared, as it contains the collected overload knowledge
  *pymeth->fMethodInfo->fRefCount += 1;
  newPyMeth->fMethodInfo = pymeth->fMethodInfo;

  // new method is unbound, track whether this proxy is used in the capacity of
  // a method or a function (which normally is a CPPFunction)
  newPyMeth->fMethodInfo->fFlags |= CallContext::kIsPseudoFunc;

  return (PyObject*)newPyMeth;
}

//----------------------------------------------------------------------------
static PyObject* mp_meth_self(CPPOverload* pymeth, void*) {
  // Return the bound self, if any; in case of pseudo-function role, pretend
  // that the data member im_self does not exist.
  if (IsPseudoFunc(pymeth)) {
    PyErr_Format(PyExc_AttributeError,
                 "function %s has no attribute \'im_self\'",
                 pymeth->fMethodInfo->fName.c_str());
    return nullptr;
  } else if (pymeth->fSelf != 0) {
    Py_INCREF((PyObject*)pymeth->fSelf);
    return (PyObject*)pymeth->fSelf;
  }

  Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* mp_meth_class(CPPOverload* pymeth, void*) {
  // Return scoping class; in case of pseudo-function role, pretend that there
  // is no encompassing class (i.e. global scope).
  if (!IsPseudoFunc(pymeth) && pymeth->fMethodInfo->fMethods.size()) {
    PyObject* pyclass = pymeth->fMethodInfo->fMethods[0]->GetScopeProxy();
    if (!pyclass)
      PyErr_Format(PyExc_AttributeError,
                   "function %s has no attribute \'im_class\'",
                   pymeth->fMethodInfo->fName.c_str());
    return pyclass;
  }

  Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* mp_func_closure(CPPOverload* /* pymeth */, void*) {
  // Stub only, to fill out the python function interface.
  Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* mp_func_code(CPPOverload*, void*) {
  // Code details are used in module inspect to fill out interactive help()
  // not important for functioning of most code, so not implemented for p3 for
  // now (TODO)
  Py_RETURN_NONE;
}

//----------------------------------------------------------------------------
static PyObject* mp_func_defaults(CPPOverload* pymeth, void*) {
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
static PyObject* mp_func_globals(CPPOverload* /* pymeth */, void*) {
  // Return this function's global dict (hard-wired to be the cppjit module);
  // used for lookup of names from co_code indexing into co_names.
  PyObject* pyglobal = PyModule_GetDict(PyImport_AddModule((char*)"cppjit"));
  Py_XINCREF(pyglobal);
  return pyglobal;
}

//----------------------------------------------------------------------------
static inline int set_flag(CPPOverload* pymeth, PyObject* value,
                           CallContext::ECallFlags flag, const char* name) {
  // Generic setter of a (boolean) flag.
  if (!value) { // accept as false (delete)
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
static PyObject* mp_getcreates(CPPOverload* pymeth, void*) {
  // Get '__creates__' boolean, which determines ownership of return values.
  return PyLong_FromLong((long)IsCreator(pymeth->fMethodInfo->fFlags));
}

//----------------------------------------------------------------------------
static int mp_setcreates(CPPOverload* pymeth, PyObject* value, void*) {
  // Set '__creates__' boolean, which determines ownership of return values.
  return set_flag(pymeth, value, CallContext::kIsCreator, "__creates__");
}

constexpr const char* mempolicy_error_message =
    "The __mempolicy__ attribute can't be used, because in the past it was "
    "reserved to manage the local memory policy. "
    "If you want to do that now, please implement a pythonization for your "
    "class that uses SetOwnership() to manage the "
    "ownership of arguments according to your needs.";

//----------------------------------------------------------------------------
static PyObject* mp_getmempolicy(CPPOverload*, void*) {
  PyErr_SetString(PyExc_RuntimeError, mempolicy_error_message);
  return nullptr;
}

//----------------------------------------------------------------------------
static int mp_setmempolicy(CPPOverload*, PyObject*, void*) {
  PyErr_SetString(PyExc_RuntimeError, mempolicy_error_message);
  return -1;
}

//----------------------------------------------------------------------------
#define CPPJIT_BOOLEAN_PROPERTY(name, flag, label)                             \
  static PyObject* mp_get##name(CPPOverload* pymeth, void*) {                  \
    if (pymeth->fMethodInfo->fFlags & flag) {                                  \
      Py_RETURN_TRUE;                                                          \
    }                                                                          \
    Py_RETURN_FALSE;                                                           \
  }                                                                            \
                                                                               \
  static int mp_set##name(CPPOverload* pymeth, PyObject* value, void*) {       \
    return set_flag(pymeth, value, flag, label);                               \
  }

// clang-format off
CPPJIT_BOOLEAN_PROPERTY(lifeline, CallContext::kSetLifeLine, "__set_lifeline__")
CPPJIT_BOOLEAN_PROPERTY(threaded, CallContext::kReleaseGIL,  "__release_gil__")
CPPJIT_BOOLEAN_PROPERTY(useffi,   CallContext::kUseFFI,      "__useffi__")
CPPJIT_BOOLEAN_PROPERTY(sig2exc,  CallContext::kProtected,   "__sig2exc__")
// clang-format on

static PyObject* mp_getcppname(CPPOverload* pymeth, void*) {
  if ((void*)pymeth == (void*)&CPPOverload_Type)
    return PyUnicode_FromString("CPPOverload_Type");

  auto& methods = pymeth->fMethodInfo->fMethods;
  if (methods.empty())
    return PyUnicode_FromString("void (*)()"); // debatable

  if (methods.size() == 1)
    return methods[0]->GetTypeName();

  return PyUnicode_FromString("void* (*)(...)"); // id.
}

//----------------------------------------------------------------------------
static PyGetSetDef mp_getset[] = {
    {(char*)"__name__", (getter)mp_name, nullptr, nullptr, nullptr},
    {(char*)"__module__", (getter)mp_module, nullptr, nullptr, nullptr},
    {(char*)"__doc__", (getter)mp_doc, (setter)mp_doc_set, nullptr, nullptr},

    // to be more python-like, where these are duplicated as well; to actually
    // derive from the python method or function type is too memory-expensive,
    // given that most of the members of those types would not be used
    {(char*)"im_func", (getter)mp_meth_func, nullptr, nullptr, nullptr},
    {(char*)"im_self", (getter)mp_meth_self, nullptr, nullptr, nullptr},
    {(char*)"im_class", (getter)mp_meth_class, nullptr, nullptr, nullptr},

    {(char*)"func_closure", (getter)mp_func_closure, nullptr, nullptr, nullptr},
    {(char*)"func_code", (getter)mp_func_code, nullptr, nullptr, nullptr},
    {(char*)"func_defaults", (getter)mp_func_defaults, nullptr, nullptr,
     nullptr},
    {(char*)"func_globals", (getter)mp_func_globals, nullptr, nullptr, nullptr},
    {(char*)"func_doc", (getter)mp_doc, (setter)mp_doc_set, nullptr, nullptr},
    {(char*)"func_name", (getter)mp_name, nullptr, nullptr, nullptr},
    {(char*)"func_overloads_types", (getter)mp_func_overloads_types, nullptr,
     nullptr, nullptr},
    {(char*)"func_overloads_names", (getter)mp_func_overloads_names, nullptr,
     nullptr, nullptr},

    // flags to control behavior
    {(char*)"__creates__", (getter)mp_getcreates, (setter)mp_setcreates,
     (char*)"For ownership rules of result: if true, objects are python-owned",
     nullptr},
    {(char*)"__mempolicy__", (getter)mp_getmempolicy, (setter)mp_setmempolicy,
     (char*)"Unused", nullptr},
    {(char*)"__set_lifeline__", (getter)mp_getlifeline, (setter)mp_setlifeline,
     (char*)"If true, set a lifeline from the return value onto self", nullptr},
    {(char*)"__release_gil__", (getter)mp_getthreaded, (setter)mp_setthreaded,
     (char*)"If true, releases GIL on call into C++", nullptr},
    {(char*)"__useffi__", (getter)mp_getuseffi, (setter)mp_setuseffi,
     (char*)"not implemented", nullptr},
    {(char*)"__sig2exc__", (getter)mp_getsig2exc, (setter)mp_setsig2exc,
     (char*)"If true, turn signals into Python exceptions", nullptr},

    // basic reflection information
    {(char*)"__cpp_name__", (getter)mp_getcppname, nullptr, nullptr, nullptr},

    {(char*)nullptr, nullptr, nullptr, nullptr, nullptr}};

//= cpyrt method proxy function behavior ==================================
static PyObject* mp_vectorcall(CPPOverload* pymeth, PyObject* const* args,
                               size_t nargsf, PyObject* kwds) {
  // Call the appropriate overload of this method.

  // If called from a descriptor, then this could be a bound function with
  // non-zero self; otherwise pymeth->fSelf is expected to always be nullptr.

  CPPInstance* im_self = pymeth->fSelf;

  // get local handles to proxy internals
  auto& methods = pymeth->fMethodInfo->fMethods;

  CPPOverload::Methods_t::size_type nMethods = methods.size();

  CallContext ctxt{};
  const auto mflags = pymeth->fMethodInfo->fFlags;
  ctxt.fFlags |= (mflags & CallContext::kReleaseGIL);
  ctxt.fFlags |= (mflags & CallContext::kProtected);
  if (IsConstructor(pymeth->fMethodInfo->fFlags))
    ctxt.fFlags |= CallContext::kIsConstructor;
  ctxt.fFlags |=
      (pymeth->fFlags & (CallContext::kCallDirect | CallContext::kFromDescr));
  ctxt.fPyContext = (PyObject*)im_self; // no Py_INCREF as no ownership

  // check implicit conversions status (may be disallowed to prevent recursion)
  ctxt.fFlags |= (pymeth->fFlags & CallContext::kNoImplicit);

  // simple case
  if (nMethods == 1) {
    if (!NoImplicit(&ctxt))
      ctxt.fFlags |= CallContext::kAllowImplicit; // no two rounds needed
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
    // it is necessary to enable implicit conversions as the memoized call may
    // be from such a conversion case; if the call fails, the implicit flag is
    // reset below
    if (!NoImplicit(&ctxt))
      ctxt.fFlags |= CallContext::kAllowImplicit;
    PyObject* result = memoized_pc->Call(im_self, args, nargsf, kwds, &ctxt);
    if (result)
      return HandleReturn(pymeth, im_self, result);

    // fall through: python is dynamic, and so, the hashing isn't infallible
    ctxt.fFlags &= ~(CallContext::kAllowImplicit | CallContext::kPyException |
                     CallContext::kCppException);
    PyErr_Clear();
    ResetCallState(pymeth->fSelf, im_self);
  }

  std::vector<interop::TCppMethod_t> overloads;
  bool is_operator = false;
  bool is_conversion_operator = false;
  bool is_static = false;
  for (auto i : pymeth->fMethodInfo->fMethods) {
    if (is_operator || dynamic_cast<cpyrt::CPPMethod*>(i)->IsOperator())
      is_operator = true;
    if (is_conversion_operator ||
        dynamic_cast<cpyrt::CPPMethod*>(i)->IsConversionOperator())
      is_conversion_operator = true;
    if (is_static || dynamic_cast<cpyrt::CPPMethod*>(i)->IsStaticMethod())
      is_static = true;
    overloads.push_back(i->GetMethod());
  }

  std::string proto = "";
  if (im_self && !(ctxt.fFlags & CallContext::kIsConstructor)) {
    PyObject* self = (PyObject*)im_self;
    [[maybe_unused]] auto res =
        AddTypeName(proto, (PyObject*)Py_TYPE(self), self, Utility::kNone);
    assert(res);
  }

  ctxt.fFlags |= CallContext::kAllowImplicit;
  {
    size_t i = ctxt.fFlags & CallContext::kIsConstructor && !im_self ? 1 : 0;
    CPPInstance* self = nullptr;
    if (i) {
      self = (CPPInstance*)cpyrt_PyArgs_GET_ITEM(args, 0);
    } else {
      self = im_self;
    }
    size_t nArgs = PyVectorcall_NARGS(nargsf);
    for (; i < nArgs; i++) {
      PyObject* obj = cpyrt_PyArgs_GET_ITEM(args, i);
      PyObject* typ = (PyObject*)Py_TYPE(obj);
      // TODO: this can be a flag within CPPOverload
      if ((pymeth->fMethodInfo->fName == "__getitem__") ||
          (pymeth->fMethodInfo->fName == "__setitem__")) {
        // unpack the tuple for Python getter and setter...
        // TODO: propogate this logic to TemplateProxy too
        if ((typ == (PyObject*)&PyTuple_Type) && (i == (im_self ? 0 : 1))) {
          for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(obj); j++) {
            if (!proto.empty())
              proto += ", ";
            PyObject* unpacking_obj = PyTuple_GET_ITEM(obj, j);
            PyObject* unpacking_typ = (PyObject*)Py_TYPE(unpacking_obj);
            if (!AddTypeName(proto, unpacking_typ, unpacking_obj,
                             Utility::kNone)) {
              proto += "__cppjit_internal::UnknownType";
              // break; // ???: do we need to raise error here
            }
          }
          continue;
        } else if (i == (im_self ? 1 : 2))
          continue;
      }
      if (self && IsConstructor(ctxt.fFlags) &&
          self->ObjectIsA(/*check_smart=*/false) !=
              pymeth->fMethodInfo->fScope) {
        // this is a constructor of cross-inherited class
        interop::TCppScope_t disp = self->ObjectIsA(/*check_smart=*/false);

        // happens for Python derived types (which have a dispatcher inserted
        // that is not otherwise user-visible: call it instead) and C++ derived
        // classes without public constructors

        // get the dispatcher class and verify
        PyObject* dispproxy = cpyrt::GetScopeProxy(disp);
        if (!dispproxy) {
          PyErr_SetString(PyExc_TypeError,
                          "dispatcher proxy was never created");
          return nullptr;
        }

        if (!(((CPPClass*)dispproxy)->fFlags & CPPScope::kIsPython)) {
          PyErr_SetString(PyExc_TypeError,
                          const_cast<char*>(("constructor for " +
                                             interop::GetScopedFinalName(disp) +
                                             " is not a dispatcher")
                                                .c_str()));
          return nullptr;
        }

        PyObject* pyobj =
            cpyrt_PyObject_Call(dispproxy, args + (self == im_self ? 0 : 1),
                                nargsf - (self == im_self ? 0 : 1), kwds);
        if (!pyobj)
          return nullptr;

        // retrieve the actual pointer, take over control, and set set
        // _internal_self
        intptr_t address = (intptr_t)((CPPInstance*)pyobj)->GetObject();
        if (address) {
          ((CPPInstance*)pyobj)
              ->CppOwns(); // b/c self will control the object on address
          PyObject* res = PyObject_CallMethodObjArgs(
              dispproxy, PyStrings::gDispInit, pyobj, (PyObject*)self, nullptr);
          Py_XDECREF(res);
        }
        Py_DECREF(dispproxy);

        // return object if successful, lament if not
        if (address) {
          Py_INCREF(self);

          // note: constructors are no longer set to take ownership by default;
          // instead that is decided by the method proxy (which carries a
          // creator flag) upon return
          self->Set((void*)address);
          if (IsCreator(pymeth->fMethodInfo->fFlags))
            self->PythonOwns();

          // mark as actual to prevent needless auto-casting and register on its
          // class
          self->fFlags |= CPPInstance::kIsActual;
          if (!(((CPPClass*)Py_TYPE(self))->fFlags & CPPScope::kIsSmart))
            MemoryRegulator::RegisterPyObject(
                self, interop::TCppObject_t((void*)address));

          // handling smart types this way is deeply fugly, but if CPPInstance
          // sets the proper types in op_new first, then the wrong init is
          // called
          if (((CPPClass*)Py_TYPE(self))->fFlags & CPPScope::kIsSmart) {
            PyObject* pyclass = CreateScopeProxy(
                ((CPPSmartClass*)Py_TYPE(self))->fUnderlyingType);
            if (pyclass) {
              self->SetSmart((PyObject*)Py_TYPE(self));
              Py_DECREF((PyObject*)Py_TYPE(self));
              Py_SET_TYPE(self, (PyTypeObject*)pyclass);
            }
          }

          // done with self
          Py_DECREF(self);

          Py_RETURN_NONE; // by definition
        }
      }
      if (IsConstructor(ctxt.fFlags) &&
          interop::GetScopedFinalName(pymeth->fMethodInfo->fScope)
                  .find("__cppjit_internal::Dispatcher") == 0) {
        if (typ == (PyObject*)&PyTuple_Type) {
          if (i > 1)
            proto += ", __cppjit_internal::Sep*";
          for (Py_ssize_t j = 0; j < PyTuple_GET_SIZE(obj); j++) {
            if (!proto.empty())
              proto += ", ";
            PyObject* unpacking_obj = PyTuple_GET_ITEM(obj, j);
            PyObject* unpacking_typ = (PyObject*)Py_TYPE(unpacking_obj);
            if (!AddTypeName(proto, unpacking_typ, unpacking_obj,
                             Utility::kNone)) {
              proto += "__cppjit_internal::UnknownType";
              // break; // ???: do we need to raise error here
            }
          }
          continue;
        }
      }
      if (!proto.empty())
        proto += ", ";
      if (!AddTypeName(proto, typ, obj, Utility::kNone)) {
        proto += "__cppjit_internal::UnknownType";
        // break; // ???: do we need to raise error here
      }
    }
    if (kwds) {
      for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwds); i++) {
        if (!proto.empty())
          proto += ", ";
        PyObject* obj = cpyrt_PyArgs_GET_ITEM(args, i + nArgs);
        PyObject* typ = (PyObject*)Py_TYPE(obj);
        if (!AddTypeName(proto, typ, obj, Utility::kNone)) {
          proto += "__cppjit_internal::UnknownType";
          break; // ???: do we need to raise error here
        }
      }
    }
  }

  std::vector<interop::TCppMethod_t> ambiguous_candidates;
  interop::TCppMethod_t meth = interop::BestOverloadFunctionMatch(
      overloads, proto, ambiguous_candidates, /*TODO:*/ nullptr, is_operator);
  if (!meth && (ctxt.fFlags & CallContext::kIsConstructor)) {
    // this might be a POD class/struct
    // and might need the self* for the overload resolution
    // to pick up the custom generated constructor defined in
    // __cppjit_internal namespace
    // FIXME: for the self parameter changes in CppInterOp
    std::string self_type_name = "";
    if (PyVectorcall_NARGS(nargsf) > 0) {
      PyObject* obj = cpyrt_PyArgs_GET_ITEM(args, 0);
      PyObject* typ = (PyObject*)Py_TYPE(obj);
      AddTypeName(self_type_name, typ, nullptr, Utility::kNone);
      proto = self_type_name + "**, " + proto;
      meth = interop::BestOverloadFunctionMatch(overloads, proto,
                                                ambiguous_candidates,
                                                /*TODO:*/ nullptr, is_operator);
    }
  }
  if (meth) {
    for (auto i : pymeth->fMethodInfo->fMethods) {
      if (i->GetMethod() == meth) {
        PyObject* result = i->Call(im_self, args, nargsf, kwds, &ctxt);
        if (result) {
          // success: update the dispatch map for subsequent calls
          if (!memoized_pc)
            dispatchMap.push_back(std::make_pair(sighash, i));
          else {
            // debatable: apparently there are two methods that map onto the
            // same sighash and preferring the latest may result in "ping
            // pong."
            for (auto& p : dispatchMap) {
              if (p.first == sighash) {
                p.second = i;
                break;
              }
            }
          }
          return HandleReturn(pymeth, im_self, result);
        }
        return nullptr;
      }
    }
  }
  std::ostringstream overload_signatures;
  if (ambiguous_candidates.empty()) {
    for (const auto i : overloads) {
      overload_signatures
          << "  "
          << interop::GetMethodReturnTypeAsString(
                 i) // TODO: mention if the function is static from the list
          << " " << interop::GetScopedFinalName(i.data)
          << interop::GetMethodSignature(i, true) << "\n";
    }
    PyErr_Format(
        gOverloadResolutionException,
        "Overload Resolution Failed.\nOverload set:\n%sDeduced Argument Types: "
        "(%s)\n%s",
        overload_signatures.str().c_str(), proto.c_str(),
        is_static ? "Found static methods in the overload set. If you want to "
                    "invoke the static overload, try calling the function "
                    "using the class type instead of the instance.\n"
                  : "");
  } else {
    for (const auto i : ambiguous_candidates) {
      overload_signatures << "  " << interop::GetMethodReturnTypeAsString(i)
                          << " " << interop::GetScopedFinalName(i.data)
                          << interop::GetMethodSignature(i, true) << "\n";
    }
    PyErr_Format(
        gOverloadAmbiguityException,
        "Overload Resolution Failed. Call to \"%s\" is ambiguous.\nAmbigious "
        "set:\n%sDeduced Argument Types: (%s)\n%s",
        pymeth->fMethodInfo->fName.c_str(), overload_signatures.str().c_str(),
        proto.c_str(),
        is_static ? "Found static methods in the overload set. If you want to "
                    "invoke the static overload, try calling the function "
                    "using the class type instead of the instance.\n"
                  : "");
  }
  return nullptr;
}

//----------------------------------------------------------------------------
static PyObject* mp_str(CPPOverload* cppinst) {
  // Print a description that includes the C++ name
  std::ostringstream s;
  s << "<C++ overload \"" << cppinst->fMethodInfo->fName << "\" at "
    << (void*)cppinst << ">";
  return PyUnicode_FromString(s.str().c_str());
}

//----------------------------------------------------------------------------
static CPPOverload* mp_descr_get(CPPOverload* pymeth, CPPInstance* pyobj,
                                 PyObject*) {
  // Descriptor; create and return a new, possibly bound, method proxy. This
  // method has evolved with versions of python as follows:
  //
  //   Python version   |       Action
  //        <- py2.7    | bound methods need to be first-class objects, so
  //        create a new
  //                    |  method object if self is not nullptr or Py_None
  //     py3.0-py3.7    | bound methods are no longer a language requirement,
  //     but
  //                    |  still supported: for convenience, retain old behavior
  //     py3.8 <=       | vector calls no longer call the descriptor, so when it
  //     is
  //                    |  called, the method is likely stored, so should be new
  //                    object

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

  newPyMeth->fVectorCall = pymeth->fVectorCall;

  if (pyobj && (PyObject*)pyobj != Py_None) {
    Py_INCREF((PyObject*)pyobj);
    newPyMeth->fSelf = pyobj;
    newPyMeth->fFlags = CallContext::kNone;
  } else {
    newPyMeth->fSelf = nullptr;
    newPyMeth->fFlags = CallContext::kCallDirect;
  }

  // vector calls don't get here, unless a method is looked up on an instance,
  // for e.g. class methods (C++ static); notify downstream to expect a 'self'
  newPyMeth->fFlags |= CallContext::kFromDescr;

  if (gc_track)
    PyObject_GC_Track(newPyMeth);

  return newPyMeth;
}

//= cpyrt method proxy construction/destruction ===========================
static CPPOverload* mp_new(PyTypeObject*, PyObject*, PyObject*) {
  // Create a new method proxy object.
  CPPOverload* pymeth = PyObject_GC_New(CPPOverload, &CPPOverload_Type);
  pymeth->fSelf = nullptr;
  pymeth->fFlags = CallContext::kNone;
  pymeth->fMethodInfo = new CPPOverload::MethodInfo_t;

  PyObject_GC_Track(pymeth);
  return pymeth;
}

//----------------------------------------------------------------------------
static void mp_dealloc(CPPOverload* pymeth) {
  // Deallocate memory held by method proxy object.
  PyObject_GC_UnTrack(pymeth);

  Py_CLEAR(pymeth->fSelf);

  if (--(*pymeth->fMethodInfo->fRefCount) <= 0) {
    delete pymeth->fMethodInfo;
  }

  if (numfree < CPPOverload_MAXFREELIST) {
    pymeth->fSelf = (cpyrt::CPPInstance*)free_list;
    free_list = pymeth;
    numfree++;
  } else {
    PyObject_GC_Del(pymeth);
  }
}

//----------------------------------------------------------------------------
static Py_ssize_t mp_hash(CPPOverload* pymeth) {
// Hash of method proxy object for insertion into dictionaries; with actual
// method (fMethodInfo) shared, its address is best suited.
#if PY_VERSION_HEX >= 0x030d0000
  return Py_HashPointer(pymeth->fMethodInfo);
#else
  return _Py_HashPointer(pymeth->fMethodInfo);
#endif
}

//----------------------------------------------------------------------------
static int mp_traverse(CPPOverload* pymeth, visitproc visit, void* args) {
  // Garbage collector traverse of held python member objects.
  if (pymeth->fSelf)
    return visit((PyObject*)pymeth->fSelf, args);

  return 0;
}

//----------------------------------------------------------------------------
static int mp_clear(CPPOverload* pymeth) {
  // Garbage collector clear of held python member objects.
  Py_CLEAR(pymeth->fSelf);

  return 0;
}

//----------------------------------------------------------------------------
static PyObject* mp_richcompare(CPPOverload* self, CPPOverload* other, int op) {
  // Rich set of comparison objects; only equals is defined.
  if (op != Py_EQ)
    return PyType_Type.tp_richcompare((PyObject*)self, (PyObject*)other, op);

  // defined by type + (shared) MethodInfo + bound self, with special case for
  // fSelf (i.e. pseudo-function)
  if ((Py_TYPE(self) == Py_TYPE(other) &&
       self->fMethodInfo == other->fMethodInfo) &&
      ((IsPseudoFunc(self) && IsPseudoFunc(other)) ||
       self->fSelf == other->fSelf)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

//= cpyrt method proxy access to internals ================================
static PyObject* mp_overload(CPPOverload* pymeth, PyObject* args) {
  // Select and call a specific C++ overload, based on its signature.
  const char* sigarg = nullptr;
  PyObject* sigarg_tuple = nullptr;
  int want_const = -1;
  Py_ssize_t args_size = PyTuple_GET_SIZE(args);
  if (args_size && PyArg_ParseTuple(args, const_cast<char*>("s|i:__overload__"),
                                    &sigarg, &want_const)) {
    want_const = args_size == 1 ? -1 : want_const;
    return pymeth->FindOverload(sigarg ? sigarg : "", want_const);
  } else if (args_size &&
             PyArg_ParseTuple(args, const_cast<char*>("O|i:__overload__"),
                              &sigarg_tuple, &want_const)) {
    PyErr_Clear();
    want_const = args_size == 1 ? -1 : want_const;
    return pymeth->FindOverload(sigarg_tuple, want_const);
  } else {
    PyErr_Format(PyExc_TypeError, "Unexpected arguments to __overload__");
    return nullptr;
  }
}

static PyObject* mp_add_overload(CPPOverload* pymeth, PyObject* new_overload) {
  TPythonCallback* cb = new TPythonCallback(new_overload);
  pymeth->AdoptMethod(cb);
  Py_RETURN_NONE;
}

static PyObject* mp_reflex(CPPOverload* pymeth, PyObject* args) {
  // Provide the requested reflection information.
  interop::Reflex::RequestId_t request = -1;
  interop::Reflex::FormatId_t format = interop::Reflex::OPTIMAL;
  if (!PyArg_ParseTuple(args, const_cast<char*>("i|i:__cpp_reflex__"), &request,
                        &format))
    return nullptr;

  return pymeth->fMethodInfo->fMethods[0]->Reflex(request, format);
}

//----------------------------------------------------------------------------
static PyMethodDef mp_methods[] = {
    {(char*)"__overload__", (PyCFunction)mp_overload, METH_VARARGS,
     (char*)"select overload for dispatch"},
    {(char*)"__add_overload__", (PyCFunction)mp_add_overload, METH_O,
     (char*)"add a new overload"},
    {(char*)"__cpp_reflex__", (PyCFunction)mp_reflex, METH_VARARGS,
     (char*)"C++ overload reflection information"},
    {(char*)nullptr, nullptr, 0, nullptr}};

} // unnamed namespace

//= cpyrt method proxy type ===============================================
PyTypeObject CPPOverload_Type = {
    PyVarObject_HEAD_INIT(&PyType_Type,
                          0)(char*) "cppjit.CPPOverload", // tp_name
    sizeof(CPPOverload),                                  // tp_basicsize
    0,                                                    // tp_itemsize
    (destructor)mp_dealloc,                               // tp_dealloc
    offsetof(CPPOverload, fVectorCall),
    0,                              // tp_getattr
    0,                              // tp_setattr
    0,                              // tp_as_async / tp_compare
    0,                              // tp_repr
    0,                              // tp_as_number
    0,                              // tp_as_sequence
    0,                              // tp_as_mapping
    (hashfunc)mp_hash,              // tp_hash
    (ternaryfunc)PyVectorcall_Call, // tp_call
    (reprfunc)mp_str,               // tp_str
    0,                              // tp_getattro
    0,                              // tp_setattro
    0,                              // tp_as_buffer
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_HAVE_VECTORCALL |
        Py_TPFLAGS_METHOD_DESCRIPTOR,        // tp_flags
    (char*)"cppjit method proxy (internal)", // tp_doc
    (traverseproc)mp_traverse,               // tp_traverse
    (inquiry)mp_clear,                       // tp_clear
    (richcmpfunc)mp_richcompare,             // tp_richcompare
    0,                                       // tp_weaklistoffset
    0,                                       // tp_iter
    0,                                       // tp_iternext
    mp_methods,                              // tp_methods
    0,                                       // tp_members
    mp_getset,                               // tp_getset
    0,                                       // tp_base
    0,                                       // tp_dict
    (descrgetfunc)mp_descr_get,              // tp_descr_get
    0,                                       // tp_descr_set
    0,                                       // tp_dictoffset
    0,                                       // tp_init
    0,                                       // tp_alloc
    (newfunc)mp_new,                         // tp_new
    0,                                       // tp_free
    0,                                       // tp_is_gc
    0,                                       // tp_bases
    0,                                       // tp_mro
    0,                                       // tp_cache
    0,                                       // tp_subclasses
    0,                                       // tp_weaklist
    0,                                       // tp_del
    0,                                       // tp_version_tag
    0,                                       // tp_finalize
    0                                        // tp_vectorcall
    CPYRT_PYTYPE_TAIL};

} // namespace cppjit::cpyrt

//- public members -----------------------------------------------------------
void cpyrt::CPPOverload::Set(const std::string& name,
                             interop::TCppScope_t scope,
                             std::vector<PyCallable*>& methods) {
  // Fill in the data of a freshly created method proxy.
  fMethodInfo->fName = name;
  fMethodInfo->fScope = scope;
  fMethodInfo->fMethods.swap(methods);

  // special case: all constructors are considered creators by default
  if (name == "__init__")
    fMethodInfo->fFlags |=
        (CallContext::kIsCreator | CallContext::kIsConstructor);

  // special case, in heuristics mode also tag *Clone* methods as creators. Only
  // check that Clone is present in the method name, not in the template
  // argument list.
  if (CallContext::GlobalPolicyFlags() & CallContext::kUseHeuristics) {
    std::string_view name_maybe_template = name;
    auto begin_template = name_maybe_template.find_first_of('<');
    if (begin_template <= name_maybe_template.size()) {
      name_maybe_template = name_maybe_template.substr(0, begin_template);
    }
    if (name_maybe_template.find("Clone") != std::string_view::npos) {
      fMethodInfo->fFlags |= CallContext::kIsCreator;
    }
  }

  fVectorCall = (vectorcallfunc)mp_vectorcall;
}

//----------------------------------------------------------------------------
void cpyrt::CPPOverload::AdoptMethod(PyCallable* pc) {
  // Fill in the data of a freshly created method proxy.
  fMethodInfo->fMethods.push_back(pc);
}

//----------------------------------------------------------------------------
void cpyrt::CPPOverload::MergeOverload(CPPOverload* meth) {
  if (!HasMethods()) // if fresh method being filled: also copy flags
    fMethodInfo->fFlags = meth->fMethodInfo->fFlags;
  fMethodInfo->fMethods.insert(fMethodInfo->fMethods.end(),
                               meth->fMethodInfo->fMethods.begin(),
                               meth->fMethodInfo->fMethods.end());
  meth->fMethodInfo->fDispatchMap.clear();
  meth->fMethodInfo->fMethods.clear();
}

//----------------------------------------------------------------------------
PyObject* cpyrt::CPPOverload::FindOverload(const std::string& signature,
                                           int want_const) {
  bool accept_any = signature == ":any:";
  CPPOverload* newmeth = nullptr;

  std::string sig1{"("};
  if (!accept_any) {
    sig1.append(signature);
    sig1.append(")");
    sig1.erase(std::remove(sig1.begin(), sig1.end(), ' '), std::end(sig1));
  }

  CPPOverload::Methods_t& methods = fMethodInfo->fMethods;
  for (auto& meth : methods) {
    bool found = accept_any;
    if (!found) {
      PyObject* pysig2 = meth->GetSignature(false);
      std::string sig2(PyUnicode_AsUTF8(pysig2));
      sig2.erase(std::remove(sig2.begin(), sig2.end(), ' '), std::end(sig2));
      Py_DECREF(pysig2);
      if (sig1 == sig2)
        found = true;

      if (!found) {
        pysig2 = meth->GetSignature(true);
        std::string sig3(PyUnicode_AsUTF8(pysig2));
        sig3.erase(std::remove(sig3.begin(), sig3.end(), ' '), std::end(sig3));
        Py_DECREF(pysig2);
        if (sig1 == sig3)
          found = true;
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
        CPPOverload::Methods_t vec;
        vec.push_back(meth->Clone());
        newmeth->Set(fMethodInfo->fName, nullptr, vec);

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
    PyErr_Format(PyExc_LookupError, "signature \"%s\" not found",
                 signature.c_str());

  return (PyObject*)newmeth;
}

PyObject* cpyrt::CPPOverload::FindOverload(PyObject* args_tuple,
                                           int want_const) {
  Py_ssize_t n = PyTuple_Size(args_tuple);

  CPPOverload::Methods_t& methods = fMethodInfo->fMethods;

  // This value is set based on the maximum penalty in
  // interop::CompareMethodArgType
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
      PyObject* pItem = PyTuple_GetItem(args_tuple, i);
      if (!PyUnicode_Check(pItem)) {
        PyErr_Format(PyExc_LookupError,
                     "argument types should be in string format");
        return (PyObject*)nullptr;
      }
      std::string arg_type(PyUnicode_AsUTF8(pItem));
      sigargs += arg_type + ", ";
    }
    sigargs += ")";

    PyErr_Format(PyExc_LookupError, "signature with arguments \"%s\" not found",
                 sigargs.c_str());
    return (PyObject*)nullptr;
  }

  CPPOverload* newmeth = mp_new(nullptr, nullptr, nullptr);
  CPPOverload::Methods_t vec;
  vec.push_back(methods[best_method]->Clone());
  newmeth->Set(fMethodInfo->fName, nullptr, vec);

  if (fSelf) {
    Py_INCREF(fSelf);
    newmeth->fSelf = fSelf;
  }
  newmeth->fMethodInfo->fFlags = fMethodInfo->fFlags;

  return (PyObject*)newmeth;
}

//----------------------------------------------------------------------------
cpyrt::CPPOverload::MethodInfo_t::~MethodInfo_t() {
  // Destructor (this object is reference counted).
  for (Methods_t::iterator it = fMethods.begin(); it != fMethods.end(); ++it) {
    delete *it;
  }
  fMethods.clear();
  delete fRefCount;
  Py_XDECREF(fDoc);
}

// TODO: something like PyMethod_Fini to clear up the free_list
