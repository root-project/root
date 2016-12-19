// Authors: Axel Naumann, Philippe Canal, Danilo Piparo

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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


ROOT::Internal::RootCling::TROOTSYSSetter gROOTSYSSetter;

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
   // This is ignored (in rootcling_impl.cxx) if R__EXTERN_LLVMDIR is defined.
   // This is not configured (i.e. R__HAVE_LLVMRESOURCEDIR is undefined) for
   // configure / make; the resource directory is instead determined by
   // TMetaUtils::GetLLVMResourceDir() in rootcling_impl.cxx.
   config.fLLVMResourceDir= "@R__LLVMRESOURCEDIR@";
#endif

   config.fBuildingROOTStage1 = true;

   return ROOT_rootcling_Driver(argc, argv, config);
}
