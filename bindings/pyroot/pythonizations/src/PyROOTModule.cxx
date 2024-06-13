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
#include "PyROOTWrapper.h"
#include "RPyROOTApplication.h"

// Cppyy
#include "CPyCppyy/API.h"
#include "../../cppyy/CPyCppyy/src/CallContext.h"
#include "../../cppyy/CPyCppyy/src/ProxyWrappers.h"

// ROOT
#include "TROOT.h"
#include "TSystem.h"
#include "RConfigure.h"

// Standard
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

   PyArg_ParseTuple(args, "UU:RegisterConverterAlias", &name, &target);

   CPyCppyy::RegisterConverterAlias(PyUnicode_AsUTF8(name), PyUnicode_AsUTF8(target));

   Py_RETURN_NONE;
}

PyObject *RegisterExecutorAlias(PyObject * /*self*/, PyObject *args)
{
   PyObject *name = nullptr;
   PyObject *target = nullptr;

   PyArg_ParseTuple(args, "UU:RegisterExecutorAlias", &name, &target);

   CPyCppyy::RegisterExecutorAlias(PyUnicode_AsUTF8(name), PyUnicode_AsUTF8(target));

   Py_RETURN_NONE;
}

} // namespace PyROOT

// Methods offered by the interface
static PyMethodDef gPyROOTMethods[] = {
   {(char *)"AddCPPInstancePickling", (PyCFunction)PyROOT::AddCPPInstancePickling, METH_VARARGS,
    (char *)"Add a custom pickling mechanism for Cppyy Python proxy objects"},
   {(char *)"GetBranchAttr", (PyCFunction)PyROOT::GetBranchAttr, METH_VARARGS,
    (char *)"Allow to access branches as tree attributes"},
   {(char *)"AddTClassDynamicCastPyz", (PyCFunction)PyROOT::AddTClassDynamicCastPyz, METH_VARARGS,
    (char *)"Cast the void* returned by TClass::DynamicCast to the right type"},
   {(char *)"AddTObjectEqNePyz", (PyCFunction)PyROOT::AddTObjectEqNePyz, METH_VARARGS,
    (char *)"Add equality and inequality comparison operators to TObject"},
   {(char *)"BranchPyz", (PyCFunction)PyROOT::BranchPyz, METH_VARARGS,
    (char *)"Fully enable the use of TTree::Branch from Python"},
   {(char *)"AddPrettyPrintingPyz", (PyCFunction)PyROOT::AddPrettyPrintingPyz, METH_VARARGS,
    (char *)"Add pretty printing pythonization"},
   {(char *)"InitApplication", (PyCFunction)PyROOT::RPyROOTApplication::InitApplication, METH_VARARGS,
    (char *)"Initialize interactive ROOT use from Python"},
   {(char *)"InstallGUIEventInputHook", (PyCFunction)PyROOT::RPyROOTApplication::InstallGUIEventInputHook, METH_NOARGS,
    (char *)"Install an input hook to process GUI events"},
   {(char *)"_CPPInstance__expand__", (PyCFunction)PyROOT::CPPInstanceExpand, METH_VARARGS,
    (char *)"Deserialize a pickled object"},
   {(char *)"ClearProxiedObjects", (PyCFunction)PyROOT::ClearProxiedObjects, METH_NOARGS,
    (char *)"Clear proxied objects regulated by PyROOT"},
   {(char *)"JupyROOTExecutor", (PyCFunction)JupyROOTExecutor, METH_VARARGS, (char *)"Create JupyROOTExecutor"},
   {(char *)"JupyROOTDeclarer", (PyCFunction)JupyROOTDeclarer, METH_VARARGS, (char *)"Create JupyROOTDeclarer"},
   {(char *)"JupyROOTExecutorHandler_Clear", (PyCFunction)JupyROOTExecutorHandler_Clear, METH_NOARGS,
    (char *)"Clear JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_Ctor", (PyCFunction)JupyROOTExecutorHandler_Ctor, METH_NOARGS,
    (char *)"Create JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_Poll", (PyCFunction)JupyROOTExecutorHandler_Poll, METH_NOARGS,
    (char *)"Poll JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_EndCapture", (PyCFunction)JupyROOTExecutorHandler_EndCapture, METH_NOARGS,
    (char *)"End capture JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_InitCapture", (PyCFunction)JupyROOTExecutorHandler_InitCapture, METH_NOARGS,
    (char *)"Init capture JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_GetStdout", (PyCFunction)JupyROOTExecutorHandler_GetStdout, METH_NOARGS,
    (char *)"Get stdout JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_GetStderr", (PyCFunction)JupyROOTExecutorHandler_GetStderr, METH_NOARGS,
    (char *)"Get stderr JupyROOTExecutorHandler"},
   {(char *)"JupyROOTExecutorHandler_Dtor", (PyCFunction)JupyROOTExecutorHandler_Dtor, METH_NOARGS,
    (char *)"Destruct JupyROOTExecutorHandler"},
   {(char *)"CPyCppyyRegisterConverterAlias", (PyCFunction)PyROOT::RegisterConverterAlias, METH_VARARGS,
    (char *)"Register a custom converter that is a reference to an existing converter"},
   {(char *)"CPyCppyyRegisterExecutorAlias", (PyCFunction)PyROOT::RegisterExecutorAlias, METH_VARARGS,
    (char *)"Register a custom executor that is a reference to an existing executor"},
   {NULL, NULL, 0, NULL}};

struct module_state {
   PyObject *error;
};

using namespace CPyCppyy;

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

   // setup PyROOT
   PyROOT::Init();

   // signal policy: don't abort interpreter in interactive mode
   CallContext::SetGlobalSignalPolicy(!gROOT->IsBatch());

   // inject ROOT namespace for convenience
   PyModule_AddObject(gRootModule, (char *)"ROOT", CreateScopeProxy("ROOT"));

   Py_INCREF(gRootModule);
   return gRootModule;
}
