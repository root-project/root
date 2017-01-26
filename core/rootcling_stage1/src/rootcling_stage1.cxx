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

extern "C" {
   R__DLLEXPORT void usedToIdentifyRootClingByDlSym() {}
}


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
