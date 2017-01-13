// Authors: Axel Naumann, Philippe Canal, Danilo Piparo

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "rootclingTCling.h"
#include "rootclingIO.h"
#include "rootcling_impl.h"
#include "RConfigure.h"
#include "RConfig.h"
#include "TSystem.h"

extern "C" {
   R__DLLEXPORT void usedToIdentifyRootClingByDlSym() {}
}

int main(int argc, char **argv)
{
   // Force the emission of the symbol - the compiler cannot know that argv
   // is always set.
   if (!argv) {
      auto dummyVal =  (int)(long)&usedToIdentifyRootClingByDlSym;
      return dummyVal;
   }

   ROOT::Internal::RootCling::DriverConfig config{};

   config.fBuildingROOTStage1 = false;
   config.fPRootDir = &gRootDir;
   config.fTROOT__GetExtraInterpreterArgs = &TROOT__GetExtraInterpreterArgs;
   config.fTROOT__GetIncludeDir = &TROOT__GetIncludeDir;
   config.fTROOT__GetEtcDir = &TROOT__GetEtcDir;
   config.fTCling__GetInterpreter = &TCling__GetInterpreter;
   config.fInitializeStreamerInfoROOTFile = &InitializeStreamerInfoROOTFile;
   config.fAddStreamerInfoToROOTFile = &AddStreamerInfoToROOTFile;
   config.fAddTypedefToROOTFile = &AddTypedefToROOTFile;
   config.fAddEnumToROOTFile = &AddEnumToROOTFile;
   config.fAddAncestorPCMROOTFile = &AddAncestorPCMROOTFile;
   config.fCloseStreamerInfoROOTFile = &CloseStreamerInfoROOTFile;

   return ROOT_rootcling_Driver(argc, argv, config);
}
