/// \file RNTupleTTreeCheckerCLI.hxx
/// \ingroup NTuple ROOT7
/// \author Ida Caspary <ida.friederike.caspary@cern.ch>
/// \date 2024-10-14
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleTTreeCheckerCLI
#define ROOT_RNTupleTTreeCheckerCLI

#include "ROOT/RNTupleTTreeChecker.hxx"
#include <vector>
#include <string>

namespace ROOT {
namespace Experimental {
namespace RNTupleTTreeCheckerCLI {

struct CheckerConfig {
   std::string fTTreeFile;
   std::string fRNTupleFile;
   std::string fTTreeName;
   std::string fRNTupleName;
   bool fShouldRun = false;
};

CheckerConfig ParseArgs(const std::vector<std::string> &args);
CheckerConfig ParseArgs(int argc, char **argv);
void RunChecker(const CheckerConfig &config);

} // namespace RNTupleTTreeCheckerCLI
} // namespace Experimental
} // namespace ROOT

#endif // ROOT_RNTupleTTreeCheckerCLI
