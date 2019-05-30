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

#ifndef R__DLLIMPORT
// I.e. we are imported.
# if _WIN32
#  define R__DLLIMPORT __declspec(dllimport)
# else
#  define R__DLLIMPORT __attribute__ ((visibility ("default")))
# endif
#endif

extern "C" {
   R__DLLIMPORT const char ** *TROOT__GetExtraInterpreterArgs();
   R__DLLIMPORT const char *TROOT__GetIncludeDir();
   R__DLLIMPORT const char *TROOT__GetEtcDir();
   R__DLLIMPORT cling::Interpreter *TCling__GetInterpreter();
}
