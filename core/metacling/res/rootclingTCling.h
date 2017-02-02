// @(#)root/utils:$Id$
// Author: Axel Naumann, 2014-04-07

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Provides bindings to TCling (compiled with rtti) from rootcling (compiled
// without rtti).

namespace cling {
   class Interpreter;
}

#ifndef R__DLLEXPORT
// I.e. we are imported.
# if _WIN32
#  define R__DLLEXPORT __declspec(dllimport)
# else
#  define R__DLLEXPORT __attribute__ ((visibility ("default")))
# endif
#endif

extern "C" {
   R__DLLEXPORT const char ** *TROOT__GetExtraInterpreterArgs();
   R__DLLEXPORT const char *TROOT__GetIncludeDir();
   R__DLLEXPORT const char *TROOT__GetEtcDir();
   R__DLLEXPORT cling::Interpreter *TCling__GetInterpreter();
}
