// Author: Enrico Guiraud, David Poulton 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ReadSpeedCLI.hxx"

#include <ROOT/TTreeProcessorMT.hxx> // for TTreeProcessorMT::SetTasksPerWorkerHint

#include <iostream>
#include <cstring>

using namespace ReadSpeed;

void ReadSpeed::PrintThroughput(const Result &r)
{
   std::cout << "Thread pool size:\t\t" << r.fThreadPoolSize << '\n';

   if (r.fMTSetupRealTime > 0.) {
      std::cout << "Real time to setup MT run:\t" << r.fMTSetupRealTime << " s\n";
      std::cout << "CPU time to setup MT run:\t" << r.fMTSetupCpuTime << " s\n";
   }

   std::cout << "Real time:\t\t\t" << r.fRealTime << " s\n";
   std::cout << "CPU time:\t\t\t" << r.fCpuTime << " s\n";

   std::cout << "Uncompressed data read:\t\t" << r.fUncompressedBytesRead << " bytes\n";
   std::cout << "Compressed data read:\t\t" << r.fCompressedBytesRead << " bytes\n";

   const uint effectiveThreads = std::max(r.fThreadPoolSize, 1u);

   std::cout << "Uncompressed throughput:\t" << r.fUncompressedBytesRead / r.fRealTime / 1024 / 1024 << " MB/s\n";
   std::cout << "\t\t\t\t" << r.fUncompressedBytesRead / r.fRealTime / 1024 / 1024 / effectiveThreads
             << " MB/s/thread for " << effectiveThreads << " threads\n";
   std::cout << "Compressed throughput:\t\t" << r.fCompressedBytesRead / r.fRealTime / 1024 / 1024 << " MB/s\n";
   std::cout << "\t\t\t\t" << r.fCompressedBytesRead / r.fRealTime / 1024 / 1024 / effectiveThreads
             << " MB/s/thread for " << effectiveThreads << " threads\n";
}

Args ReadSpeed::ParseArgs(const std::vector<std::string> &args)
{
   // Print help message and exit if "--help"
   if (args.size() < 2 || (args.size() == 2 && (args[1] == "--help" || args[1] == "-h"))) {
      std::cout << "Usage:\n"
                << "  root-readspeed --trees tname1 [tname2 ...]\n"
                << "                 --files fname1 [fname2 ...]\n"
                << "                 (--all-branches | --branches bname1 [bname2 ...] | --branches-regex bregex1 "
                   "[bregex2 ...])\n"
                << "                 [--threads nthreads]\n"
                << "  root-readspeed (--help|-h)\n";
      return {};
   }

   Data d;
   unsigned int nThreads = 0;

   enum class EArgState { kNone, kTrees, kFiles, kBranches, kThreads, kTasksPerWorkerHint } argState = EArgState::kNone;
   enum class EBranchState { kNone, kRegular, kRegex, kAll } branchState = EBranchState::kNone;
   const auto branchOptionsErrMsg =
      "Options --all-branches, --branches, and --branches-regex are mutually exclusive. You can use only one.\n";

   for (size_t i = 1; i < args.size(); ++i) {
      const auto &arg = args[i];

      if (arg == "--trees") {
         argState = EArgState::kTrees;
      } else if (arg == "--files") {
         argState = EArgState::kFiles;
      } else if (arg == "--all-branches") {
         argState = EArgState::kNone;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kAll) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kAll;
         d.fUseRegex = true;
         d.fBranchNames = {".*"};
      } else if (arg == "--branches") {
         argState = EArgState::kBranches;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kRegular) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kRegular;
      } else if (arg == "--branches-regex") {
         argState = EArgState::kBranches;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kRegex) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kRegex;
         d.fUseRegex = true;
      } else if (arg == "--threads") {
         argState = EArgState::kThreads;
      } else if (arg == "--tasks-per-worker") {
         argState = EArgState::kTasksPerWorkerHint;
      } else if (arg[0] == '-') {
         std::cerr << "Unrecognized option '" << arg << "'\n";
         return {};
      } else {
         switch (argState) {
         case EArgState::kTrees: d.fTreeNames.emplace_back(arg); break;
         case EArgState::kFiles: d.fFileNames.emplace_back(arg); break;
         case EArgState::kBranches: d.fBranchNames.emplace_back(arg); break;
         case EArgState::kThreads:
            nThreads = std::stoi(arg);
            argState = EArgState::kNone;
            break;
         case EArgState::kTasksPerWorkerHint:
            ROOT::TTreeProcessorMT::SetTasksPerWorkerHint(std::stoi(arg));
            argState = EArgState::kNone;
            break;
         default: std::cerr << "Unrecognized option '" << arg << "'\n"; return {};
         }
      }
   }

   return Args{std::move(d), nThreads, branchState == EBranchState::kAll, /*fShouldRun=*/true};
}

Args ReadSpeed::ParseArgs(int argc, char **argv)
{
   std::vector<std::string> args;
   args.reserve(argc);

   for (int i = 0; i < argc; ++i) {
      args.emplace_back(argv[i]);
   }

   return ParseArgs(args);
}