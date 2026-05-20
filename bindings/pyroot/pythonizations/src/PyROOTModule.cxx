// Author: Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Bindings
#include "PyROOTPythonize.h"
#include "RPyROOTApplication.h"

// Cppyy
#include "CPyCppyy/API.h"

// ROOT
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"
#include "RConfigure.h"

// Standard
#include <any>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "IOHandler.cxx"

namespace PyROOT {

PyObject *gRootModule = nullptr;

PyObject *RegisterConverterAlias(PyObject * /*self*/, PyObject *args)
{
   PyObject *name = nullptr;
   PyObject *target = nullptr;

   if (!PyArg_ParseTuple(args, "UU:RegisterConverterAlias", &name, &target)) {
      return nullptr;
   }

   const char *nameStr = PyUnicode_AsUTF8AndSize(name, nullptr);
   if (!nameStr) {
      return nullptr;
   }

   const char *targetStr = PyUnicode_AsUTF8AndSize(target, nullptr);
   if (!targetStr) {
      return nullptr;
   }

   CPyCppyy::RegisterConverterAlias(nameStr, targetStr);

   Py_RETURN_NONE;
}

PyObject *RegisterExecutorAlias(PyObject * /*self*/, PyObject *args)
{
   PyObject *name = nullptr;
   PyObject *target = nullptr;

   if (!PyArg_ParseTuple(args, "UU:RegisterExecutorAlias", &name, &target)) {
      return nullptr;
   }

   const char *nameStr = PyUnicode_AsUTF8AndSize(name, nullptr);
   if (!nameStr) {
      return nullptr;
   }

   const char *targetStr = PyUnicode_AsUTF8AndSize(target, nullptr);
   if (!targetStr) {
      return nullptr;
   }

   CPyCppyy::RegisterExecutorAlias(nameStr, targetStr);

   Py_RETURN_NONE;
}

/// \brief A PyObject wrapper to track reference counting of external objects
///
/// This wrapper can be useful in shared ownership scenarios when a C++ object
/// is created on the Python side and there is no easy way to track its ownership
/// on the C++ side. If multiple instances of this class are given the same
/// Python proxy, they will increase/decrease its reference counting following
/// Python rules and ensure proper destruction of the underlying C++ object when
/// no other Python objects are referencing it.
class PyObjRefCounter final {
   PyObject *fObject{nullptr};

   void Reset(PyObject *object)
   {
      if (fObject) {
         Py_DecRef(fObject);
         fObject = nullptr;
      }
      if (object) {
         Py_INCREF(object);
         fObject = object;
      }
   }

public:
   PyObjRefCounter(PyObject *object) { Reset(object); }

   ~PyObjRefCounter() { Reset(nullptr); }

   PyObjRefCounter(const PyObjRefCounter &other) { Reset(other.fObject); }

   PyObjRefCounter &operator=(const PyObjRefCounter &other)
   {
      Reset(other.fObject);
      return *this;
   }

   PyObjRefCounter(PyObjRefCounter &&other)
   {
      fObject = other.fObject;
      other.fObject = nullptr;
   }

   PyObjRefCounter &operator=(PyObjRefCounter &&other)
   {
      fObject = other.fObject;
      other.fObject = nullptr;
      return *this;
   }
};

PyObject *PyObjRefCounterAsStdAny(PyObject * /*self*/, PyObject *args)
{
   PyObject *object = nullptr;

   PyArg_ParseTuple(args, "O:PyObjRefCounterAsStdAny", &object);

   // The std::any is managed by Python
   return CPyCppyy::Instance_FromVoidPtr(new std::any{std::in_place_type<PyObjRefCounter>, object}, "std::any",
                                         /*python_owns=*/true);
}

// Helper function to get the pointer to a buffer (heavily simplified copy of
// CPyCppyy::Utility::GetBuffer).
void GetBuffer(PyObject *pyobject, void *&buf)
{
   buf = nullptr;

   // Exclude text-like objects (policy decision)
   if (PyBytes_Check(pyobject) || PyUnicode_Check(pyobject))
      return;

   // Fast path: bytearray
   if (PyByteArray_CheckExact(pyobject)) {
      buf = PyByteArray_AsString(pyobject);
      return;
   }

   // Buffer protocol
   if (PyObject_CheckBuffer(pyobject)) {
      // Avoid potential issues with empty sequences
      if (PySequence_Check(pyobject) && PySequence_Size(pyobject) == 0)
         return;

      Py_buffer view;
      if (PyObject_GetBuffer(pyobject, &view, PyBUF_SIMPLE) == 0) {
         if (view.buf && view.len > 0)
            buf = view.buf;
         PyBuffer_Release(&view);
      } else {
         PyErr_Clear();
      }
   }
}

} // namespace PyROOT

// Methods offered by the interface
static PyMethodDef gPyROOTMethods[] = {
   {"AddCPPInstancePickling", (PyCFunction)PyROOT::AddCPPInstancePickling, METH_NOARGS,
    "Add a custom pickling mechanism for Cppyy Python proxy objects"},
   {"GetBranchAttr", (PyCFunction)PyROOT::GetBranchAttr, METH_VARARGS,
    "Allow to access branches as tree attributes"},
   {"AddTClassDynamicCastPyz", (PyCFunction)PyROOT::AddTClassDynamicCastPyz, METH_VARARGS,
    "Cast the void* returned by TClass::DynamicCast to the right type"},
   {"BranchPyz", (PyCFunction)PyROOT::BranchPyz, METH_VARARGS,
    "Fully enable the use of TTree::Branch from Python"},
   {"AddPrettyPrintingPyz", (PyCFunction)PyROOT::AddPrettyPrintingPyz, METH_VARARGS,
    "Add pretty printing pythonization"},
   {"InitApplication", (PyCFunction)PyROOT::RPyROOTApplication::InitApplication, METH_VARARGS,
    "Initialize interactive ROOT use from Python"},
   {"InstallGUIEventInputHook", (PyCFunction)PyROOT::RPyROOTApplication::InstallGUIEventInputHook, METH_NOARGS,
    "Install an input hook to process GUI events"},
   {"_CPPInstance__expand__", (PyCFunction)PyROOT::CPPInstanceExpand, METH_VARARGS,
    "Deserialize a pickled object"},
   {"JupyROOTExecutor", (PyCFunction)JupyROOTExecutor, METH_VARARGS, "Create JupyROOTExecutor"},
   {"JupyROOTDeclarer", (PyCFunction)JupyROOTDeclarer, METH_VARARGS, "Create JupyROOTDeclarer"},
   {"JupyROOTExecutorHandler_Clear", (PyCFunction)JupyROOTExecutorHandler_Clear, METH_NOARGS,
    "Clear JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_Ctor", (PyCFunction)JupyROOTExecutorHandler_Ctor, METH_NOARGS,
    "Create JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_Poll", (PyCFunction)JupyROOTExecutorHandler_Poll, METH_NOARGS,
    "Poll JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_EndCapture", (PyCFunction)JupyROOTExecutorHandler_EndCapture, METH_NOARGS,
    "End capture JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_InitCapture", (PyCFunction)JupyROOTExecutorHandler_InitCapture, METH_NOARGS,
    "Init capture JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_GetStdout", (PyCFunction)JupyROOTExecutorHandler_GetStdout, METH_NOARGS,
    "Get stdout JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_GetStderr", (PyCFunction)JupyROOTExecutorHandler_GetStderr, METH_NOARGS,
    "Get stderr JupyROOTExecutorHandler"},
   {"JupyROOTExecutorHandler_Dtor", (PyCFunction)JupyROOTExecutorHandler_Dtor, METH_NOARGS,
    "Destruct JupyROOTExecutorHandler"},
   {"CPyCppyyRegisterConverterAlias", (PyCFunction)PyROOT::RegisterConverterAlias, METH_VARARGS,
    "Register a custom converter that is a reference to an existing converter"},
   {"CPyCppyyRegisterExecutorAlias", (PyCFunction)PyROOT::RegisterExecutorAlias, METH_VARARGS,
    "Register a custom executor that is a reference to an existing executor"},
   {"PyObjRefCounterAsStdAny", (PyCFunction)PyROOT::PyObjRefCounterAsStdAny, METH_VARARGS,
    "Wrap a reference count to any Python object in a std::any for resource management in C++"},
   {NULL, NULL, 0, NULL}};

struct module_state {
   PyObject *error;
};

#define GETSTATE(m) ((struct module_state *)PyModule_GetState(m))

static int rootmodule_traverse(PyObject *m, visitproc visit, void *arg)
{
   Py_VISIT(GETSTATE(m)->error);
   return 0;
}

static int rootmodule_clear(PyObject *m)
{
   Py_CLEAR(GETSTATE(m)->error);
   return 0;
}

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT,       "libROOTPythonizations", NULL,
                                       sizeof(struct module_state), gPyROOTMethods,          NULL,
                                       rootmodule_traverse,         rootmodule_clear,        NULL};

/// Initialization of extension module libROOTPythonizations

extern "C" PyObject *PyInit_libROOTPythonizations()
{
   using namespace PyROOT;

   // setup PyROOT
   gRootModule = PyModule_Create(&moduledef);
   if (!gRootModule)
      return nullptr;

   // keep gRootModule, but do not increase its reference count even as it is borrowed,
   // or a self-referencing cycle would be created

   return gRootModule;
}
