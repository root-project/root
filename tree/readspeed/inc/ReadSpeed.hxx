// Author: Enrico Guiraud, David Poulton 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOTREADSPEED
#define ROOTREADSPEED

#include <TFile.h>

#include <string>
#include <vector>
#include <regex>

namespace ReadSpeed {

struct Data {
   /// Either a single tree name common for all files, or one tree name per file.
   std::vector<std::string> fTreeNames;
   /// List of input files.
   std::vector<std::string> fFileNames;
   /// Branches to read.
   std::vector<std::string> fBranchNames;
   /// If the branch names should use regex matching.
   bool fUseRegex = false;
};

struct Result {
   /// Real time spent reading and decompressing all data, in seconds.
   double fRealTime;
   /// CPU time spent reading and decompressing all data, in seconds.
   double fCpuTime;
   /// Real time spent preparing the multi-thread workload.
   double fMTSetupRealTime;
   /// CPU time spent preparing the multi-thread workload.
   double fMTSetupCpuTime;
   /// Number of uncompressed bytes read in total from TTree branches.
   ULong64_t fUncompressedBytesRead;
   /// Number of compressed bytes read in total from the TFiles.
   ULong64_t fCompressedBytesRead;
   /// Size of ROOT's thread pool for the run (0 indicates a single-thread run with no thread pool present).
   unsigned int fThreadPoolSize;
};

struct EntryRange {
   Long64_t fStart = -1;
   Long64_t fEnd = -1;
};

struct ByteData {
   ULong64_t fUncompressedBytesRead;
   ULong64_t fCompressedBytesRead;
};

struct ReadSpeedRegex {
   std::string text;
   std::regex regex;

   bool operator<(const ReadSpeedRegex &other) const { return text < other.text; }
};

std::vector<std::string> GetMatchingBranchNames(const std::string &fileName, const std::string &treeName,
                                                const std::vector<ReadSpeedRegex> &regexes);

// Read branches listed in branchNames in tree treeName in file fileName, return number of uncompressed bytes read.
ByteData ReadTree(const std::string &treeName, const std::string &fileName, const std::vector<std::string> &branchNames,
                  EntryRange range = {-1, -1});

Result EvalThroughputST(const Data &d);

// Return a vector of EntryRanges per file, i.e. a vector of vectors of EntryRanges with outer size equal to
// d.fFileNames.
std::vector<std::vector<EntryRange>> GetClusters(const Data &d);

// Mimic the logic of TTreeProcessorMT::MakeClusters: merge entry ranges together such that we
// run around TTreeProcessorMT::GetTasksPerWorkerHint tasks per worker thread.
// TODO it would be better to expose TTreeProcessorMT's actual logic and call the exact same method from here
std::vector<std::vector<EntryRange>>
MergeClusters(std::vector<std::vector<EntryRange>> &&clusters, unsigned int maxTasksPerFile);

Result EvalThroughputMT(const Data &d, unsigned nThreads);

Result EvalThroughput(const Data &d, unsigned nThreads);

} // namespace ReadSpeed

#endif // ROOTREADSPEED
