/* Copyright (C) 2020 Enrico Guiraud
   See the LICENSE file in the top directory for more information. */

#include "ReadSpeedCLI.hxx"

#include <ROOT/TTreeProcessorMT.hxx> // for TTreeProcessorMT::SetTasksPerWorkerHint

#include <iostream>
#include <cstring>

using namespace ReadSpeed;

void ReadSpeed::PrintThroughput(const Result &r)
{
   const uint effectiveThreads = std::max(r.fThreadPoolSize, 1u);

   std::cout << "Thread pool size:\t\t" << r.fThreadPoolSize << '\n';

   if (r.fMTSetupRealTime > 0.) {
      std::cout << "Real time to setup MT run:\t" << r.fMTSetupRealTime << " s\n";
      std::cout << "CPU time to setup MT run:\t" << r.fMTSetupCpuTime << " s\n";
   }

   std::cout << "Real time:\t\t\t" << r.fRealTime << " s\n";
   std::cout << "CPU time:\t\t\t" << r.fCpuTime << " s\n";

   std::cout << "Uncompressed data read:\t\t" << r.fUncompressedBytesRead << " bytes\n";
   std::cout << "Compressed data read:\t\t" << r.fCompressedBytesRead << " bytes\n";

   std::cout << "Uncompressed throughput:\t" << r.fUncompressedBytesRead / r.fRealTime / 1024 / 1024 << " MB/s\n";
   std::cout << "\t\t\t\t" << r.fUncompressedBytesRead / r.fRealTime / 1024 / 1024 / effectiveThreads
             << " MB/s/thread for " << effectiveThreads << " threads\n";
   std::cout << "Compressed throughput:\t\t" << r.fCompressedBytesRead / r.fRealTime / 1024 / 1024 << " MB/s\n";
   std::cout << "\t\t\t\t" << r.fCompressedBytesRead / r.fRealTime / 1024 / 1024 / effectiveThreads
             << " MB/s/thread for " << effectiveThreads << " threads\n";
}

Args ReadSpeed::ParseArgs(std::vector<std::string> args)
{
   // Print help message and exit if "--help"
   if (args.size() < 2 || (args.size() == 2 && (args[1].compare("--help") == 0 || args[1].compare("-h") == 0))) {
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
      auto arg = args[i];

      if (arg.compare("--trees") == 0) {
         argState = EArgState::kTrees;
      } else if (arg.compare("--files") == 0) {
         argState = EArgState::kFiles;
      } else if (arg.compare("--all-branches") == 0) {
         argState = EArgState::kNone;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kAll) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kAll;
         d.fUseRegex = true;
         d.fBranchNames = {".*"};
      } else if (arg.compare("--branches") == 0) {
         argState = EArgState::kBranches;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kRegular) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kRegular;
      } else if (arg.compare("--branches-regex") == 0) {
         argState = EArgState::kBranches;
         if (branchState != EBranchState::kNone && branchState != EBranchState::kRegex) {
            std::cerr << branchOptionsErrMsg;
            return {};
         }
         branchState = EBranchState::kRegex;
         d.fUseRegex = true;
      } else if (arg.compare("--threads") == 0) {
         argState = EArgState::kThreads;
      } else if (arg.compare("--tasks-per-worker") == 0) {
         argState = EArgState::kTasksPerWorkerHint;
      } else if (arg.compare(0, 1, "-") == 0) {
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
      args.push_back(std::string(argv[i]));
   }

   return ParseArgs(args);
}