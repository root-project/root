/// \file ttree2rntuple.cxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2023-10-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RNTupleImporterCLI.hxx"

using namespace ROOT::Experimental::RNTupleImporterCLI;

int main(int argc, char **argv)
{
   auto config = ParseArgs(argc, argv);

   if (!config.fShouldRun)
      return 1; // ParseArgs has printed the --help or has encountered an issue and logged about it

   RunImporter(config);

   return 0;
}
