// Bindings
#include "cppjit_interop.h"
#include "cpyrt.h"

using namespace cppjit;
#include "CPPInstance.h"
#include "MemoryRegulator.h"
#include "ProxyWrappers.h"

// Standard
#include <assert.h>
#include <iostream>
#include <string.h>

//= pseudo-None type for masking out objects on the python side ===============
static PyTypeObject cpyrt_NoneType;

//-----------------------------------------------------------------------------
static Py_ssize_t AlwaysNullLength(PyObject*) { return 0; }

//-----------------------------------------------------------------------------
static PyMappingMethods cpyrt_NoneType_mapping = {
    AlwaysNullLength, (binaryfunc)0, (objobjargproc)0};

// silence warning about some cast operations
#if defined(__GNUC__) &&                                                       \
    (__GNUC__ >= 5 ||                                                          \
     (__GNUC__ >= 4 && ((__GNUC_MINOR__ == 2 && __GNUC_PATCHLEVEL__ >= 1) ||   \
                        (__GNUC_MINOR__ >= 3)))) &&                            \
    !__INTEL_COMPILER
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

//-----------------------------------------------------------------------------
namespace {

struct Initcpyrt_NoneType_t {
  Initcpyrt_NoneType_t() {
    // create a cpyrt NoneType (for references that went dodo) from NoneType
    memset(&cpyrt_NoneType, 0, sizeof(cpyrt_NoneType));

    ((PyObject&)cpyrt_NoneType).ob_type = &PyType_Type;
    Py_SET_REFCNT((PyObject*)&cpyrt_NoneType, 1);
    ((PyVarObject&)cpyrt_NoneType).ob_size = 0;

    cpyrt_NoneType.tp_name = const_cast<char*>("cpyrt_NoneType");
    cpyrt_NoneType.tp_flags = Py_TPFLAGS_HAVE_RICHCOMPARE;

    cpyrt_NoneType.tp_traverse = (traverseproc)0;
    cpyrt_NoneType.tp_clear = (inquiry)0;
    cpyrt_NoneType.tp_dealloc = (destructor)&Initcpyrt_NoneType_t::DeAlloc;
    cpyrt_NoneType.tp_repr = Py_TYPE(Py_None)->tp_repr;
    cpyrt_NoneType.tp_richcompare =
        (richcmpfunc)&Initcpyrt_NoneType_t::RichCompare;
    cpyrt_NoneType.tp_hash = (hashfunc)&Initcpyrt_NoneType_t::PtrHash;

    cpyrt_NoneType.tp_as_mapping = &cpyrt_NoneType_mapping;

    PyType_Ready(&cpyrt_NoneType);
  }

  static void DeAlloc(PyObject* pyobj) { Py_TYPE(pyobj)->tp_free(pyobj); }
  static int PtrHash(PyObject* pyobj) { return (int)ptrdiff_t(pyobj); }

  static PyObject* RichCompare(PyObject*, PyObject* other, int opid) {
    return PyObject_RichCompare(other, Py_None, opid);
  }

  static int Compare(PyObject*, PyObject* other) {
    // TODO the following isn't correct as it doesn't order, but will do for now
    // ...
    return !PyObject_RichCompareBool(other, Py_None, Py_EQ);
  }
};

} // unnamed namespace

// Memory regulation hooks
cpyrt::MemHook_t cpyrt::MemoryRegulator::registerHook = nullptr;
cpyrt::MemHook_t cpyrt::MemoryRegulator::unregisterHook = nullptr;

//- ctor/dtor ----------------------------------------------------------------
cpyrt::MemoryRegulator::MemoryRegulator() {
  // setup NoneType for referencing and create weakref cache
  static Initcpyrt_NoneType_t initcpyrt_NoneType;
}

//- public members -----------------------------------------------------------
bool cpyrt::MemoryRegulator::RecursiveRemove(interop::TCppObject_t cppobj,
                                             interop::TCppScope_t klass) {
  // if registered by the framework, called whenever a cppobj gets destroyed
  if (!cppobj)
    return false;

  PyObject* pyscope = GetScopeProxy(klass);
  if (!CPPScope_Check(pyscope)) {
    Py_XDECREF(pyscope);
    return false;
  }

  CppToPyMap_t* cppobjs = ((CPPClass*)pyscope)->fImp.fCppObjects;
  if (!cppobjs) { // table may have been deleted on shutdown
    Py_DECREF(pyscope);
    return false;
  }

  // see whether we're tracking this object
  CppToPyMap_t::iterator ppo = cppobjs->find(cppobj);

  if (ppo != cppobjs->end()) {
    // get the tracked object
    CPPInstance* pyobj = (CPPInstance*)ppo->second;

    // erase the object from tracking
    pyobj->fFlags &= ~CPPInstance::kIsRegulated;
    cppobjs->erase(ppo);

    // nullify the object
    if (!cpyrt_NoneType.tp_traverse) {
      // take a reference as we're copying its function pointers
      Py_INCREF(Py_TYPE(pyobj));

      // all object that arrive here are expected to be of the same type
      // ("instance")
      cpyrt_NoneType.tp_traverse = Py_TYPE(pyobj)->tp_traverse;
      cpyrt_NoneType.tp_clear = Py_TYPE(pyobj)->tp_clear;
      cpyrt_NoneType.tp_free = Py_TYPE(pyobj)->tp_free;
      cpyrt_NoneType.tp_flags |= Py_TYPE(pyobj)->tp_flags;
    } else if (cpyrt_NoneType.tp_traverse != Py_TYPE(pyobj)->tp_traverse) {
      // TODO: SystemError?
      std::cerr << "in cpyrt::MemoryRegulater, unexpected object of type: "
                << Py_TYPE(pyobj)->tp_name << std::endl;

      // drop object and leave before too much damage is done
      Py_DECREF(pyscope);
      return false;
    }

    // notify any other weak referents by playing dead
    Py_ssize_t refcnt = Py_REFCNT((PyObject*)pyobj);
    Py_SET_REFCNT((PyObject*)pyobj, 0);
    PyObject_ClearWeakRefs((PyObject*)pyobj);
    Py_SET_REFCNT((PyObject*)pyobj, refcnt);

    // cleanup object internals
    pyobj->CppOwns();         // held object is out of scope now anyway
    op_dealloc_nofree(pyobj); // normal object cleanup, while keeping memory

    // reset type object
    Py_INCREF((PyObject*)(void*)&cpyrt_NoneType);
    Py_DECREF(Py_TYPE(pyobj));
    ((PyObject*)pyobj)->ob_type = &cpyrt_NoneType;

    Py_DECREF(pyscope);
    return true;
  }

  // unregulated cppobj
  Py_DECREF(pyscope);
  return false;
}

//-----------------------------------------------------------------------------
bool cpyrt::MemoryRegulator::RegisterPyObject(CPPInstance* pyobj,
                                              interop::TCppObject_t cppobj) {
  // start tracking <cppobj> proxied by <pyobj>
  if (!(pyobj && cppobj))
    return false;

  if (registerHook) {
    auto res = registerHook(cppobj, pyobj->ObjectIsA(false));
    if (!res.second)
      return res.first;
  }

  CppToPyMap_t* cppobjs = ((CPPClass*)Py_TYPE(pyobj))->fImp.fCppObjects;
  if (!cppobjs)
    return false;

  // if an address was already associated with a different object, then stop
  // following the old and force insert the new proxy for following
  const auto& res = cppobjs->insert(std::make_pair(cppobj, (PyObject*)pyobj));
  if (!res.second) {
    ((CPPInstance*)res.first->second)->fFlags &= ~CPPInstance::kIsRegulated;
    (*cppobjs)[cppobj] = (PyObject*)pyobj;
  }

  pyobj->fFlags |= CPPInstance::kIsRegulated;
  return true;
}

//-----------------------------------------------------------------------------
bool cpyrt::MemoryRegulator::UnregisterPyObject(CPPInstance* pyobj,
                                                PyObject* pyclass) {
  // called when the proxy is garbage collected or is about delete the C++
  // instance
  if (!(pyobj && pyclass))
    return false;

  interop::TCppObject_t cppobj =
      pyobj->IsSmart() ? pyobj->GetObjectRaw() : pyobj->GetObject();
  if (!cppobj)
    return false;

  if (unregisterHook) {
    auto res = unregisterHook(cppobj, ((CPPClass*)pyclass)->fCppType);
    if (!res.second)
      return res.first;
  }

  CppToPyMap_t* cppobjs = ((CPPClass*)pyclass)->fImp.fCppObjects;
  if (!cppobjs)
    return false;

  // erase if tracked
  if (cppobjs->erase(cppobj)) {
    pyobj->fFlags &= ~CPPInstance::kIsRegulated;
    return true;
  }

  return false;
}

//-----------------------------------------------------------------------------
PyObject* cpyrt::MemoryRegulator::RetrievePyObject(interop::TCppObject_t cppobj,
                                                   PyObject* pyclass) {
  // lookup to see if a C++ address is already known, return old proxy if
  // tracked
  if (!(cppobj && pyclass))
    return nullptr;

  CppToPyMap_t* cppobjs = ((CPPClass*)pyclass)->fImp.fCppObjects;
  if (!cppobjs)
    return nullptr;

  CppToPyMap_t::iterator ppo = cppobjs->find(cppobj);
  if (ppo != cppobjs->end()) {
    Py_INCREF(ppo->second);
    return ppo->second;
  }

  return nullptr;
}

//-----------------------------------------------------------------------------
void cpyrt::MemoryRegulator::SetRegisterHook(MemHook_t h) {
  // Set custom register hook; called when a regulated object is to be tracked
  registerHook = h;
}

//-----------------------------------------------------------------------------
void cpyrt::MemoryRegulator::SetUnregisterHook(MemHook_t h) {
  // Set custom unregister hook; called when a regulated object is to be
  // untracked
  unregisterHook = h;
}
