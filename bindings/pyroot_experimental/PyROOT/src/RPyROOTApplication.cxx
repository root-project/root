// Author: Enric Tejedor CERN  04/2019
// Original PyROOT code by Wim Lavrijsen, LBL

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Bindings
#include "Python.h"
#include "RPyROOTApplication.h"

// ROOT
#include "TInterpreter.h"
#include "TSystem.h"
#include "TBenchmark.h"
#include "TStyle.h"
#include "TError.h"
#include "Getline.h"
#include "TVirtualMutex.h"


////////////////////////////////////////////////////////////////////////////
/// \brief Create an RPyROOTApplication.
/// \return false if gApplication is not null, true otherwise.
bool PyROOT::RPyROOTApplication::CreateApplication()
{
   if (!gApplication) {
      int argc = 1;
      char **argv = new char *[argc];

      // TODO: Consider parsing arguments for the RPyROOTApplication here

#if PY_VERSION_HEX < 0x03000000
      if (Py_GetProgramName() && strlen(Py_GetProgramName()) != 0)
         argv[0] = Py_GetProgramName();
      else
         argv[0] = (char *)"python";
#else
      argv[0] = (char *)"python";
#endif

      gApplication = new RPyROOTApplication("PyROOT", &argc, argv);
      delete[] argv; // TApplication ctor has copied argv, so done with it

      return true;
   }

   return false;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Setup the basic ROOT globals gBenchmark, gStyle and gProgname,
/// if not already set.
void PyROOT::RPyROOTApplication::InitROOTGlobals()
{
   if (!gBenchmark)
      gBenchmark = new TBenchmark();
   if (!gStyle)
      gStyle = new TStyle();

   if (!gProgName) // should have been set by TApplication
#if PY_VERSION_HEX < 0x03000000
      gSystem->SetProgname(Py_GetProgramName());
#else
      gSystem->SetProgname("python");
#endif
}

////////////////////////////////////////////////////////////////////////////
/// \brief Translate ROOT error/warning to Python.
static void ErrMsgHandler(int level, Bool_t abort, const char *location, const char *msg)
{
   // Initialization from gEnv (the default handler will return w/o msg b/c level too low)
   if (gErrorIgnoreLevel == kUnset)
      ::DefaultErrorHandler(kUnset - 1, kFALSE, "", "");

   if (level < gErrorIgnoreLevel)
      return;

   // Turn warnings into Python warnings
   if (level >= kError) {
      ::DefaultErrorHandler(level, abort, location, msg);
   } else if (level >= kWarning) {
      static const char *emptyString = "";
      if (!location)
         location = emptyString;
      // This warning might be triggered while holding the ROOT lock, while
      // some other thread is holding the GIL and waiting for the ROOT lock.
      // That will trigger a deadlock.
      // So if ROOT is in MT mode, use ROOT's error handler that doesn't take
      // the GIL.
      if (!gGlobalMutex) {
         // Either printout or raise exception, depending on user settings
         PyErr_WarnExplicit(NULL, (char *)msg, (char *)location, 0, (char *)"ROOT", NULL);
      } else {
         ::DefaultErrorHandler(level, abort, location, msg);
      }
   } else {
      ::DefaultErrorHandler(level, abort, location, msg);
   }
}

////////////////////////////////////////////////////////////////////////////
/// \brief Install the ROOT message handler which will turn ROOT error
/// messages into Python exceptions.
void PyROOT::RPyROOTApplication::InitROOTMessageCallback()
{
   SetErrorHandler((ErrorHandlerFunc_t)&ErrMsgHandler);
}

////////////////////////////////////////////////////////////////////////////
/// \brief Initialize an RPyROOTApplication.
PyObject *PyROOT::RPyROOTApplication::InitApplication()
{
   if (CreateApplication()) {
      InitROOTGlobals();
      InitROOTMessageCallback();
   }

   Py_RETURN_NONE;
}

////////////////////////////////////////////////////////////////////////////
/// \brief Construct a TApplication for PyROOT.
/// \param[in] acn Application class name.
/// \param[in] argc Number of arguments.
/// \param[in] argv Arguments.
PyROOT::RPyROOTApplication::RPyROOTApplication(const char *acn, int *argc, char **argv) : TApplication(acn, argc, argv)
{
   // Save current interpreter context
   gInterpreter->SaveContext();
   gInterpreter->SaveGlobalsContext();

   // Prevent crashes on accessing history
   Gl_histinit((char *)"-");

   // Prevent ROOT from exiting python
   SetReturnFromRun(true);
}

namespace {
static int (*sOldInputHook)() = nullptr;
static PyThreadState *sInputHookEventThreadState = nullptr;

static int EventInputHook()
{
   // This method is supposed to be called from CPython's command line and
   // drives the GUI
   PyEval_RestoreThread(sInputHookEventThreadState);
   gSystem->ProcessEvents();
   PyEval_SaveThread();

   if (sOldInputHook)
      return sOldInputHook();

   return 0;
}

} // unnamed namespace

////////////////////////////////////////////////////////////////////////////
/// \brief Install a method hook for sending events to the GUI.
PyObject *PyROOT::RPyROOTApplication::InstallGUIEventInputHook()
{
   if (PyOS_InputHook && PyOS_InputHook != &EventInputHook)
      sOldInputHook = PyOS_InputHook;

   sInputHookEventThreadState = PyThreadState_Get();

   PyOS_InputHook = (int (*)()) & EventInputHook;

   Py_RETURN_NONE;
}
