// Author: Enrico Guiraud, David Poulton 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ReadSpeedCLI.hxx"

#ifdef R__USE_IMT
#include <ROOT/TTreeProcessorMT.hxx> // for TTreeProcessorMT::SetTasksPerWorkerHint
#endif

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
   "    The list of trees to read from the files. If only one tree is provided then it will"
   "    be used for all files. If multiple trees are specified, each tree is read from the"
   "    respective file."
   "\n"
   "\n"
   " Specifying branches:\n"
   "  Branches can be specified using one of the following flags. Currently only one can be used"
   "  at a time.\n"
   "   --all-branches\n"
   "    Reads every branch from the specified files and trees."
   "\n"
   "   --branches bname1 [bname2...]\n"
   "    Reads the branches with matching names. Will error if any of the branches are not found."
   "\n"
   "   --branches-regex bregex1 [bregex2 ...]\n"
   "    Reads any branches with a name matching the provided regex. Will error if any provided"
   "    regex does not match at least one branch."
   "\n"
   "\n"
   " Meta arguments:\n"
   "   --threads nthreads\n"
   "    The number of threads to use for file reading. Will automatically cap to the number of"
   "    available threads on the machine."
   "\n"
   "   --tasks-per-worker ntasks\n"
   "    The number of tasks to generate for each worker thread when using multithreading.";

const auto fullUsageText =
   "Description:\n"
   " rootreadspeed is a tool used to help identify bottlenecks in root analysis programs"
   " by providing an idea of what throughput you can expect when reading ROOT files in"
   " certain configurations."
   " \n"
   " It does this by providing information about the number of bytes read from your files,"
   " how long this takes, and the different throughputs in MB/s, both in total and per thread."
   "\n"
   "\n"
   "Compressed vs Uncompressed Throughput:\n"
   " Throughput speeds are provided as compressed and uncompressed - ROOT files are usually"
   " saved in compressed format, so these will often differ. Compressed bytes is the total"
   " number of bytes read from TFiles during the readspeed test (possibly including meta-data)."
   " Uncompressed bytes is the number of bytes processed by reading the branch values in the TTree."
   " Throughput is calculated as the total number of bytes over the total runtime (including"
   " decompression time) in the uncompressed and compressed cases."
   "\n"
   "\n"
   "Interpreting results:\n"
   " \n"
   " There are three possible scenarios when using rootreadspeed, namely:"
   " \n"
   "  -The 'Real Time' is significantly lower than your own analysis runtime."
   "    This would imply your actual application code is dominating the runtime of your analysis,"
   "    ie. your analysis logic or framework is taking up the time."
   "    \n"
   "    The best way to decrease the runtime would be to optimize your code, attempt to parallelize"
   "    it onto multiple threads if possible, or use a machine with a more performant CPU."
   "    The best way to decrease the runtime would be to optimize your code (or the framework's),"
   "    parallelize it onto multiple threads if possible (for example with"
   "    RDataFrame and EnableImplicitMT) or switch to a machine with a more performant CPU."
   "  \n"
   "  \n"
   "  -The 'Real Time' is significantly higher than 'CPU Time / number of threads'*."
   "    If the real time is higher than the CPU time per core it implies the reading of data is the"
   "    bottleneck, as the CPU cores are wasting time waiting for data to arrive from your disk/drive"
   "    or network connection in order to decompress it."
   "    \n"
   "    The best way to decrease your runtime would be transferring the data you need onto a faster"
   "    storage medium (ie. a faster disk/drive such as an SSD, or connecting to a faster network"
   "    for remote file access), or to use a compression algorithm with a higher compression ratio,"
   "    possibly at the cost of the decompression rate."
   "    \n"
   "    Changing the number of threads is unlikely to help, and in fact using too many threads may"
   "    degrade performance if they make requests to different regions of your local storage. "
   "    \n"
   "    * If no '--threads' argument was provided this is 1, otherwise it is the minimum of the value"
   "      provided and the number of threads your CPU can run in parallel. It is worth noting that -"
   "      on shared systems or if running other heavy applications - the number of your own threads"
   "      running at any time may be lower than the limit due to demand on the CPU."
   "  \n"
   "  \n"
   "  -The 'Real Time' is similar to 'CPU Time / number of threads'"
   "  -AND 'Compressed Throughput' is lower than expected for your storage medium:"
   "    This would imply that your CPU threads aren't decompressing data as fast as your storage medium"
   "    can provide it, and so decompression is the bottleneck."
   "    \n"
   "    The best way to decrease your runtime would be to utilise a system with a faster CPU, or make use"
   "    use of more threads when running, or use a compression algorithm with a higher decompression rate"
   "    such as LZ4, possibly at the cost of some extra file size."
   "\n"
   "\n"
   "A note on caching:\n"
   " If your data is stored on a local disk, the system may cache some/all of the file in memory after it is"
   " first read. If this is realistic of how your analysis will run - then there is no concern. However, if"
   " you expect to only read files once in a while - and as such the files are unlikely to be in the cache -"
   " consider clearing the cache before running rootreadspeed."
   " On Linux this can be done by running 'echo 3 > /proc/sys/vm/drop_caches' as a superuser"
   " or a specific file can be dropped from the cache with"
   " `dd of=<FILENAME> oflag=nocache conv=notrunc,fdatasync count=0 > /dev/null 2>&1`."
   "\n"
   "\n"
   " Known overhead of TTreeReader, RDataFrame:\n"
   " `rootreadspeed` is designed to read all data present in the specified branches, trees and files at the highest "
   " possible speed. When the application bottleneck is not in the computations performed by analysis logic, higher-level "
   " interfaces built on top of TTree such as TTreeReader and RDataFrame are known to add a significant runtime overhead "
   " with respect to the runtimes reported by `rootreadspeed` (up to a factor 2). In realistic analysis applications it has "
   " been observed that a large part of that overhead is compensated by the ability of TTreeReader and RDataFrame to read "
   " branch values selectively, based on event cuts, and this overhead will be reduced significantly when using RDataFrame "
   " in conjunction with RNTuple.";

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
             << " MB/s/thread for " << effectiveThreads << " threads\n\n";

   const float cpuEfficiency = (r.fCpuTime / effectiveThreads) / r.fRealTime;

   std::cout << "CPU Efficiency: \t\t" << (cpuEfficiency * 100) << "%\n";
   std::cout << "Reading data is ";
   if (cpuEfficiency > 0.80f) {
      std::cout << "likely CPU bound (decompression).\n";
   } else if (cpuEfficiency < 0.50f) {
      std::cout << "likely I/O bound.\n";
   } else {
      std::cout << "likely balanced (more threads may help though).\n";
   }
   std::cout << "For details run with the --help command.\n";
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
#ifdef R__USE_IMT
            ROOT::TTreeProcessorMT::SetTasksPerWorkerHint(std::stoi(arg));
            argState = EArgState::kNone;
#else
            std::cerr << "ROOT was built without implicit multi-threading (IMT) support. The --tasks-per-worker option "
                         "will be ignored.\n";
#endif
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
