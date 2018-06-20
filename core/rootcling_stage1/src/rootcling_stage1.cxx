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
#include <ROOT/RConfig.h>
#include <stdlib.h>

extern "C" {
   R__DLLEXPORT void usedToIdentifyRootClingByDlSym() {}
}

// force compiler to emit symbol for function above
static void (*dlsymaddr)() = &usedToIdentifyRootClingByDlSym;

static const char *GetIncludeDir() {
   ROOT::Internal::RootCling::assertROOTSYS();
   static std::string incdir = std::string(getenv("ROOTSYS")) + "/include";
   return incdir.c_str();
}

static const char *GetEtcDir() {
   ROOT::Internal::RootCling::assertROOTSYS();
   static std::string etcdir = std::string(getenv("ROOTSYS")) + "/etc";
   return etcdir.c_str();
}

int main(int argc, char **argv)
{
   (void) dlsymaddr; // avoid unused variable warning

   ROOT::Internal::RootCling::DriverConfig config{};

   config.fBuildingROOTStage1 = true;
   config.fTROOT__GetIncludeDir = &GetIncludeDir;
   config.fTROOT__GetEtcDir = &GetEtcDir;

   return ROOT_rootcling_Driver(argc, argv, config);
}
