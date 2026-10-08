// Author: Enric Tejedor CERN  06/2018
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace PyROOT {

PyObject *gRootModule = nullptr;

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
   {"InstallWarningHandler", (PyCFunction)InstallWarningHandler, METH_NOARGS,
    "Turn ROOT warnings into Python warnings"},
   {"InstallGUIEventInputHook", (PyCFunction)InstallGUIEventInputHook, METH_NOARGS,
    "Install an input hook to process GUI events"},
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
