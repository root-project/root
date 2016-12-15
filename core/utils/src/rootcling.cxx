// Authors: Axel Naumann, Philippe Canal, Danilo Piparo

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_STAGE1_BUILD
#include "rootclingTCling.h"
#include "rootclingIO.h"
#endif

#include "rootcling_impl.h"
#include "RConfigure.h"
#include "RConfig.h"

#include <iostream>

#ifdef _WIN32
# ifdef system
#  undef system
# endif
# include <windows.h>
# include <Tlhelp32.h> // for MAX_MODULE_NAME32
# include <process.h>
# define PATH_MAX _MAX_PATH
# ifdef interface
// prevent error coming from clang/AST/Attrs.inc
#  undef interface
# endif
#else // _WIN32
# include <limits.h>
# include <unistd.h>
# include <dlfcn.h>
#endif

#ifdef __APPLE__
#include <libgen.h> // Needed for basename
#include <mach-o/dyld.h>
#endif

extern "C" {
R__DLLEXPORT void usedToIdentifyRootClingByDlSym() {}
}

////////////////////////////////////////////////////////////////////////////////

#ifdef __ICC
#pragma warning disable 69
#endif


int main(int argc, char **argv)
{
   // Force the emission of the symbol - the compiler cannot know that argv
   // is always set.
   if (!argv) {
      auto dummyVal =  (int)(long)&usedToIdentifyRootClingByDlSym;
      return dummyVal;
   }

   ROOT::Internal::RootCling::DriverConfig config{};
#ifdef R__HAVE_LLVMRESOURCEDIR
  config.fLLVMResourceDir= "@R__LLVMRESOURCEDIR@";
#endif

#ifdef ROOT_STAGE1_BUILD
   config.fBuildingROOTStage1 = true;
#else
   config.fBuildingROOTStage1 = false;
   config.fTROOT__GetExtraInterpreterArgs = &TROOT__GetExtraInterpreterArgs;
   config.fTCling__GetInterpreter = &TCling__GetInterpreter;
   config.fInitializeStreamerInfoROOTFile = &InitializeStreamerInfoROOTFile;
   config.fAddStreamerInfoToROOTFile = &AddStreamerInfoToROOTFile;
   config.fAddTypedefToROOTFile = &AddTypedefToROOTFile;
   config.fAddEnumToROOTFile = &AddEnumToROOTFile;
   config.fAddAncestorPCMROOTFile = &AddAncestorPCMROOTFile;
   config.fCloseStreamerInfoROOTFile = &CloseStreamerInfoROOTFile;
#endif
   return rootcling_driver(argc, argv, config);
}
