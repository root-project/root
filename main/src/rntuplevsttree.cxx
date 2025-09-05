/// \file rntuplevsttree.cxx
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

#include "ROOT/RNTupleTTreeCheckerCLI.hxx"

using namespace ROOT::Experimental::RNTupleTTreeCheckerCLI;

int main(int argc, char **argv) {
    auto config = ParseArgs(argc, argv);

    if (!config.fShouldRun) {
        return 1;
    }

    RunChecker(config);

    return 0;
}
