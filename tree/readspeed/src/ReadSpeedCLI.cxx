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

const auto usageText = "Usage:\n"
                       " rootreadspeed --files fname1 [fname2 ...]\n"
                       "               --trees tname1 [tname2 ...]\n"
                       "               (--all-branches | --branches bname1 [bname2 ...] | --branches-regex bregex1 "
                       "[bregex2 ...])\n"
                       "               [--threads nthreads]\n"
                       "               [--tasks-per-worker ntasks]\n"
                       " rootreadspeed (--help|-h)\n"
                       " \n"
                       " Use -h for usage help, --help for detailed information.\n";

const auto argUsageText =
   "Arguments:\n"
   " Specifying files and trees:\n"
   "   --files fname1 [fname2...]\n"
   "    The list of root files to read from.\n"
   "\n"
   "   --trees tname1 [tname2...]\n"
   "    The list of trees to read from the files. If only one tree is provided then it will\n"
   "    be used for all files. If multiple trees are specified, each tree is read from the\n"
   "    respective file.\n"
   "\n"
   "\n"
   " Specifying branches:\n"
   "  Branches can be specified using one of the following flags. Currently only one can be used\n"
   "  at a time.\n"
   "   --all-branches\n"
   "    Reads every branch from the specified files and trees.\n"
   "\n"
   "   --branches bname1 [bname2...]\n"
   "    Reads the branches with matching names. Will error if any of the branches are not found.\n"
   "\n"
   "   --branches-regex bregex1 [bregex2 ...]\n"
   "    Reads any branches with a name matching the provided regex. Will error if any provided\n"
   "    regex does not match at least one branch.\n"
   "\n"
   "\n"
   " Meta arguments:\n"
   "   --threads nthreads\n"
   "    The number of threads to use for file reading. Will automatically cap to the number of\n"
   "    available threads on the machine.\n"
   "\n"
   "   --tasks-per-worker ntasks\n"
   "    The number of tasks to generate for each worker thread when using multithreading.\n";

const auto fullUsageText =
   "Description:\n"
   " rootreadspeed is a tool used to help identify bottlenecks in root analysis programs\n"
   " by providing an idea of what throughput you can expect when reading ROOT files in\n"
   " certain configurations.\n"
   " \n"
   " It does this by providing information about the number of bytes read from your files,\n"
   " how long this takes, and the differnet throughputs in MB/s, both in total and per thread.\n"
   "\n"
   "\n"
   "Interpreting results:\n"
   " \n"
   " There are three possible scenarios when using rootreadspeed, namely:\n"
   " \n"
   "  -The 'Real Time' is significantly lower than your own analysis runtime.\n"
   "    This would imply your actual application code is dominating the runtime of your analysis,\n"
   "    ie. your analysis logic is taking up the time.\n"
   "    \n"
   "    The best way to decrease the runtime would be to optimize your code, attempt to parallelize\n"
   "    it onto multiple threads if possible, or use a machine with a more performant CPU.\n"
   "  \n"
   "  \n"
   "  -The 'Real Time' is significantly higher than 'CPU Time / number of threads'*.\n"
   "    If the real time is higher than the CPU time per core it implies the reading of data is the\n"
   "    bottleneck, as the CPU cores are wasting time waiting for data to arrive from your disk/drive\n"
   "    or network connection in order to decompress it.\n"
   "    \n"
   "    The best way to decrease your runtime would be transferring the data you need onto a faster\n"
   "    storage medium (ie. a faster disk/drive such as an SSD, or connecting to a faster network\n"
   "    for remote file access), or to use a compression algorithm with a higher compression ratio,\n"
   "    possibly at the cost of the decompression rate.\n"
   "    \n"
   "    Changing the number of threads is unlikely to help, and in fact using too many threads may\n"
   "    degrade performance if they make requests to different regions of your local storage. \n"
   "    \n"
   "    * If no '--threads' argument was provided this is 1, otherwise it is the minimum of the value\n"
   "      provided and the number of threads your CPU can run in parallel. It is worth noting that -\n"
   "      on shared systems or if running other heavy applications - the number of your own threads\n"
   "      running at any time may be lower than the limit due to demand on the CPU.\n"
   "  \n"
   "  \n"
   "  -The 'Real Time' is similar to 'CPU Time / number of threads'\n"
   "  -AND 'Compressed Throughput' is lower than expected for your storage medium:\n"
   "    This would imply that your CPU threads aren't decompressing data as fast as your storage medium\n"
   "    can provide it, and so decompression is the bottleneck.\n"
   "    \n"
   "    The best way to decrease your runtime would be to utilise a system with a faster CPU, or make use\n"
   "    use of more threads when running, or use a compression algorithm with a higher decompression rate,\n"
   "    possibly at the cost of some extra file size.\n"
   "\n"
   "\n"
   "A note on caching:\n"
   " If your data is stored on a local disk, the system may cache some/all of the file in memory after it is\n"
   " first read. If this is realistic of how your analysis will run - then there is no concern. However, if\n"
   " you expect to only read files once in a while - and as such the files are unlikely to be in the cache -\n"
   " consider clearing the cache before running rootreadspeed.\n"
   "  (On Linux this can be done by running 'echo 3 > /proc/sys/vm/drop_caches' as a superuser)\n";

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

   const unsigned int effectiveThreads = std::max(r.fThreadPoolSize, 1u);

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
   const auto argsProvided = args.size() >= 2;
   const auto helpUsed = argsProvided && (args[1] == "--help" || args[1] == "-h");
   const auto longHelpUsed = argsProvided && args[1] == "--help";

   if (!argsProvided || helpUsed) {
      std::cout << usageText;
      if (helpUsed)
         std::cout << "\n" << argUsageText;
      if (longHelpUsed)
         std::cout << "\n\n" << fullUsageText;
      std::cout << std::endl;

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