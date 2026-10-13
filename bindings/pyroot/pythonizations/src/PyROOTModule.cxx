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

// Cppyy
#include "cpyrt/API.h"

// ROOT
#include "TError.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TVirtualPad.h"
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
   return cppjit::cpyrt::Instance_FromVoidPtr(new std::any{std::in_place_type<PyObjRefCounter>, object}, "std::any",
                                         /*python_owns=*/true);
}

} // namespace PyROOT

namespace {

/// Turns a ROOT warning into a Python warning, see ROOT::Internal::SetWarningHandler().
bool WarningHandler(const char *location, const char *msg)
{
   PyGILState_STATE state = PyGILState_Ensure();
   // Python code can't run while an exception is pending, e.g. a previous
   // warning turned into an error: decline, and ROOT prints the warning.
   bool handled = !PyErr_Occurred();
   if (handled) {
      // Prints or, per the warnings filters, raises: the exception stays
      // pending until the call into C++ returns to Python.
      PyErr_WarnExplicit(nullptr, msg, location, 0, "ROOT", nullptr);
   }
   PyGILState_Release(state);
   return handled;
}

int (*gPreviousInputHook)() = nullptr;

/// Processes GUI events while the Python prompt waits for input. Runs no
/// Python code, so a Ctrl-C arriving meanwhile stays pending for the prompt.
int GUIEventInputHook()
{
   PyGILState_STATE state = PyGILState_Ensure();
   if (gPad && gPad->IsWeb())
      gPad->UpdateAsync();
   gSystem->ProcessEvents();
   PyGILState_Release(state);

   return gPreviousInputHook ? gPreviousInputHook() : 0;
}

/// Turn ROOT warnings into Python warnings.
PyObject *InstallWarningHandler(PyObject * /*self*/, PyObject * /*args*/)
{
   ROOT::Internal::SetWarningHandler(&WarningHandler);
   Py_RETURN_NONE;
}

/// Install the PyOS_InputHook that processes GUI events, chaining to the
/// previous hook.
PyObject *InstallGUIEventInputHook(PyObject * /*self*/, PyObject * /*args*/)
{
   if (PyOS_InputHook != &GUIEventInputHook) {
      gPreviousInputHook = PyOS_InputHook;
      PyOS_InputHook = &GUIEventInputHook;
   }
   Py_RETURN_NONE;
}

} // namespace

// Methods offered by the interface
static PyMethodDef gPyROOTMethods[] = {
   {"AddCPPInstancePickling", (PyCFunction)PyROOT::AddCPPInstancePickling, METH_NOARGS,
    "Add a custom pickling mechanism for Cppyy Python proxy objects"},
   {"AddPrettyPrintingPyz", (PyCFunction)PyROOT::AddPrettyPrintingPyz, METH_VARARGS,
    "Add pretty printing pythonization"},
   {"InstallWarningHandler", (PyCFunction)InstallWarningHandler, METH_NOARGS,
    "Turn ROOT warnings into Python warnings"},
   {"InstallGUIEventInputHook", (PyCFunction)InstallGUIEventInputHook, METH_NOARGS,
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
