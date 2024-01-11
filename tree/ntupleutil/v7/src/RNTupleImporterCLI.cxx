/// \file RNTupleImporterCLI.cxx
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

#include "TROOT.h"

using namespace ROOT::Experimental::RNTupleImporterCLI;

namespace {
const auto usageText = "Usage:\n"
                       " ttree2rntuple (--ttree|-t) name\n"
                       "               (--infile|-i) path\n"
                       "               (--outfile|-o) path\n"
                       "               [(--rntuple|-r) name]\n"
                       "               [(--compression|-c) compression]\n"
                       "               [--unzipped-page-size size]\n"
                       "               [--zipped-cluster-size size]\n"
                       "               [--max-unzipped-cluster-size size]\n"
                       "               [--convert-dots]\n"
                       "               [(--verbose|-v)]\n"
                       " ttree2rntuple [--help|-h]\n\n";

const auto argUsageText =
   "ttree2rntuple: a utility CLI for converting ROOT TTrees to ROOT RNTuples.\n"
   "\n"
   "Required arguments:\n"
   "  --ttree, -t name\n"
   "    The name of the source TTree to convert to RNTuple.\n"
   "  --infile, -i path\n"
   "    The path to the ROOT file that contains the source TTree.\n"
   "  --outfile, -o path\n"
   "    The path to the ROOT file to write the target RNTuple to. This file does not have to exist yet.\n"
   "    This may be the same file as the input file. Note that in this case the name of the target RNTuple should\n"
   "    be set to something different from the source TTree name using `--rntuple`, to avoid naming conflicts.\n"
   "\n"
   "Optional arguments:\n"
   "  --rntuple, -r name\n"
   "    The name of the target RNTuple.\n"
   "    This argument is optional. When not provided, the name of the source TTree will be used.\n"
   "  --compression, -c compression\n"
   "    The compression settings of the target RNTuple, provided as an integer.\n"
   "    This argument should be provided following ROOT's compression setting scheme (algorithm * 100 + level),\n"
   "    where `algorithm` uses the following mapping: {1: ZLIB, 2: LZMA, 4: LZMA, 5: ZSTD} \n"
   "    and `level` is a number from 0 to 9 (inclusive, 0 indicating no compression).\n"
   "    If not specified, the default zstd (505) compression is used.\n"
   "  --unzipped-page-size size\n"
   "    The desired approximate unzipped (in-memory) page size in bytes, provided as an integer.\n"
   "    If not specified, the default size of 64 * 1024 bytes is used.\n"
   "  --zipped-cluster-size size\n"
   "    The desired approximate zipped cluster size in bytes, provided as an integer.\n"
   "    If not specified, the default size of 50 * 1000 * 1000 bytes is used.\n"
   "  --max-unzipped-cluster-size size\n"
   "    The desired maximum unzipped (in-memory) cluster size in bytes, provided as an integer.\n"
   "    If not specified, the default size of 512 * 1024 * 1024 bytes is used.\n"
   "  --convert-dots\n"
   "    Whether to convert dots in branch names (if present) to underscores in field names.\n"
   "    RNTuple does not allow for dots in field names, so this option will convert them to underscores.\n"
   "    If not specified, no conversion happens and an error is thrown when branch names with dots are encountered.\n"
   "  --verbose, -v\n"
   "    Whether to print schema information and progress.\n"
   "    If not specified, nothing will be printed except for a brief report about the source TTree and target "
   "RNTuple.\n";
} // namespace

ImporterConfig ROOT::Experimental::RNTupleImporterCLI::ParseArgs(const std::vector<std::string> &args)
{
   // Print help message and exit if "--help"
   const auto argsProvided = args.size() >= 2;
   const auto helpUsed = argsProvided && (args[1] == "--help" || args[1] == "-h");

   if (!argsProvided || helpUsed) {
      std::cout << usageText;
      if (!argsProvided)
         std::cout << " Use --help or -h for usage help.";
      if (helpUsed)
         std::cout << argUsageText;
      std::cout << std::endl;

      return {};
   }

   enum class EArgState {
      kNone,
      kTreeName,
      kTreePath,
      kNTuplePath,
      kNTupleName,
      kCompression,
      kUnzippedPageSize,
      kUnzippedClusterSize,
      kZippedClusterSize
   } argState = EArgState::kNone;
   ImporterConfig config;
   // The config will overwrite the write options set by `RNTupleImporter`, so we need to set the default zstd
   // compression here as well.
   config.fNTupleOpts.SetCompression(config.kDefaultCompressionSettings);

   for (size_t i = 1; i < args.size(); ++i) {
      const auto &arg = args[i];

      if (arg == "--ttree" || arg == "-t") {
         argState = EArgState::kTreeName;
      } else if (arg == "--infile" || arg == "-i") {
         argState = EArgState::kTreePath;
      } else if (arg == "--outfile" || arg == "-o") {
         argState = EArgState::kNTuplePath;
      } else if (arg == "--rntuple" || arg == "-r") {
         argState = EArgState::kNTupleName;
      } else if (arg == "--compression" || arg == "-c") {
         argState = EArgState::kCompression;
      } else if (arg == "--unzipped-page-size") {
         argState = EArgState::kUnzippedPageSize;
      } else if (arg == "--zipped-cluster-size") {
         argState = EArgState::kZippedClusterSize;
      } else if (arg == "--max-unzipped-cluster-size") {
         argState = EArgState::kUnzippedClusterSize;
      } else if (arg == "--convert-dots") {
         config.fConvertDots = true;
      } else if (arg == "--verbose" || arg == "-v") {
         config.fVerbose = true;
      } else if (arg[0] == '-') {
         std::cerr << "Unrecognized option '" << arg << "'\n";
         return {};
      } else {
         switch (argState) {
         case EArgState::kTreeName: config.fTreeName = arg; break;
         case EArgState::kTreePath: config.fTreePath = arg; break;
         case EArgState::kNTuplePath: config.fNTuplePath = arg; break;
         case EArgState::kNTupleName: config.fNTupleName = arg; break;
         case EArgState::kCompression: config.fNTupleOpts.SetCompression(std::stoi(arg)); break;
         case EArgState::kUnzippedPageSize: config.fNTupleOpts.SetApproxUnzippedPageSize(std::stol(arg)); break;
         case EArgState::kZippedClusterSize: config.fNTupleOpts.SetApproxZippedClusterSize(std::stol(arg)); break;
         case EArgState::kUnzippedClusterSize: config.fNTupleOpts.SetMaxUnzippedClusterSize(std::stol(arg)); break;
         default: std::cerr << "Unrecognized option '" << arg << "'\n"; return {};
         }
      }
   }

   if (config.fTreeName == "") {
      std::cerr << "Please provide the name of the TTree to convert.\n\n" << usageText << "\n";
      return {};
   } else if (config.fTreePath == "") {
      std::cerr << "Please provide the name of the ROOT file containing the TTree to convert.\n\n" << usageText << "\n";
      return {};
   } else if (config.fNTuplePath == "") {
      std::cerr << "Please provide the name of the ROOT file to write the converted RNTuple to.\n\n"
                << usageText << "\n";
      return {};
   }

   config.fShouldRun = true;
   return config;
}

ImporterConfig ROOT::Experimental::RNTupleImporterCLI::ParseArgs(int argc, char **argv)
{
   std::vector<std::string> args;
   args.reserve(argc);

   for (int i = 0; i < argc; ++i) {
      args.emplace_back(argv[i]);
   }

   return ParseArgs(args);
}

void ROOT::Experimental::RNTupleImporterCLI::RunImporter(const ImporterConfig &config)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif

   auto importer = RNTupleImporter::Create(config.fTreePath, config.fTreeName, config.fNTuplePath);

   importer->SetWriteOptions(config.fNTupleOpts);
   importer->SetIsQuiet(!config.fVerbose);
   importer->SetConvertDotsInBranchNames(config.fConvertDots);

   if (!config.fNTupleName.empty())
      importer->SetNTupleName(config.fNTupleName);

   std::cout << "Converting TTree '" << config.fTreeName << "' in '" << config.fTreePath << "' to RNTuple '"
             << config.fTreeName << "' in '" << config.fNTuplePath << "'..." << std::endl;

   importer->Import();
}
