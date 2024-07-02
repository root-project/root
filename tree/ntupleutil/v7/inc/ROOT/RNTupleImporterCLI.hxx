/// \file ROOT/RNTupleImporterCLI.hxx
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

/* This header contains helper functions for the ttree2rntuple program
   for CLI related actions, such as argument parsing and output printing. */

#ifndef ROOT7_RNTupleImporterCLI
#define ROOT7_RNTupleImporterCLI

#include "ROOT/RNTupleOptions.hxx"

#include <vector>

namespace ROOT {
namespace Experimental {
namespace RNTupleImporterCLI {

struct ImporterConfig {
   /// By default, compress RNTuple with zstd, level 5
   static constexpr int kDefaultCompressionSettings = 505;
   /// The name of the TTree to convert from.
   std::string fTreeName;
   /// The path to the ROOT file containing the TTree to convert from.
   std::string fTreePath;
   /// The name of the RNTuple to convert to.
   std::string fNTupleName;
   /// The path to the ROOT file that will contain the newly converted RNTuple.
   std::string fNTuplePath;
   /// The RNTuple write options to use when converting.
   RNTupleWriteOptions fNTupleOpts;
   /// Whether dots present in branch names should be converted to underscores.
   bool fConvertDots = false;
   /// Whether per-branch progress should be printed.
   bool fVerbose = false;
   /// Whether the importing should actually happen (or, for example, only a help text should be printed).
   bool fShouldRun = false;
};

ImporterConfig ParseArgs(const std::vector<std::string> &args);
ImporterConfig ParseArgs(int argc, char **argv);

void RunImporter(const ImporterConfig &config);
} // namespace RNTupleImporterCLI
} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_RNTupleImporterCLI
