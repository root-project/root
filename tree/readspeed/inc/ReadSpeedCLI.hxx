// Author: Enrico Guiraud, David Poulton 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/* This header contains helper functions for the root-readspeed program
   for CLI related actions, such as argument parsing and output printing. */

#ifndef ROOTREADSPEEDCLI
#define ROOTREADSPEEDCLI

#include "ReadSpeed.hxx"

#include <vector>

namespace ReadSpeed {

void PrintThroughput(const Result &r);

struct Args {
   Data fData;
   unsigned int fNThreads = 0;
   bool fAllBranches = false;
   bool fShouldRun = false;
};

Args ParseArgs(const std::vector<std::string> &args);
Args ParseArgs(int argc, char **argv);

} // namespace ReadSpeed   

#endif // ROOTREADSPEEDCLI
