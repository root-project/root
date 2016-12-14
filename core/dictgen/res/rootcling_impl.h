// Authors: Axel Naumann, Philippe Canal, Danilo Piparo

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

namespace cling {
   class Interpreter;
}

namespace ROOT {
namespace Internal {
namespace RootCling {
   struct DriverConfig {
      bool fBuildingROOTStage1 = false;
      std::string fLLVMResourceDir;

      // Function that might (rootcling) or might not (rootcling_tmp) be there.
      const char ** * (*fTROOT__GetExtraInterpreterArgs)() = nullptr;
      cling::Interpreter *(*fTCling__GetInterpreter)() = nullptr;
      void (*fInitializeStreamerInfoROOTFile)(const char *filename) = nullptr;
      void (*fAddStreamerInfoToROOTFile)(const char *normName) = nullptr;
      void (*fAddTypedefToROOTFile)(const char *tdname) = nullptr;
      void (*fAddEnumToROOTFile)(const char *tdname) = nullptr;
      void (*fAddAncestorPCMROOTFile)(const char *pcmName) = nullptr;
      bool (*fCloseStreamerInfoROOTFile)(bool writeEmptyRootPCM) = nullptr;
   };

   int rootcling_driver(int argc, char **argv, const DriverConfig& config);
} // namespace RootCling
} // namespace Internal
} // namespace ROOT
