// Author: Enrico Guiraud, David Poulton 2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ReadSpeed.hxx"

#include <ROOT/TSeq.hxx>
#include <ROOT/TThreadExecutor.hxx>
#include <ROOT/TTreeProcessorMT.hxx>  // for TTreeProcessorMT::GetTasksPerWorkerHint
#include <ROOT/InternalTreeUtils.hxx> // for ROOT::Internal::TreeUtils::GetTopLevelBranchNames
#include <ROOT/RSlotStack.hxx>
#include <TBranch.h>
#include <TStopwatch.h>
#include <TTree.h>

#include <algorithm>
#include <cassert>
#include <cmath> // std::ceil
#include <memory>
#include <stdexcept>
#include <set>
#include <iostream>

using namespace ReadSpeed;

std::vector<std::string> ReadSpeed::GetMatchingBranchNames(const std::string &fileName, const std::string &treeName,
                                                           const std::vector<ReadSpeedRegex> &regexes)
{
   const auto f = std::unique_ptr<TFile>(TFile::Open(fileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
   if (f == nullptr || f->IsZombie())
      throw std::runtime_error("Could not open file '" + fileName + '\'');
   std::unique_ptr<TTree> t(f->Get<TTree>(treeName.c_str()));
   if (t == nullptr)
      throw std::runtime_error("Could not retrieve tree '" + treeName + "' from file '" + fileName + '\'');

   const auto unfilteredBranchNames = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*t);
   std::set<ReadSpeedRegex> usedRegexes;
   std::vector<std::string> branchNames;

   auto filterBranchName = [regexes, &usedRegexes](const std::string &bName) {
      if (regexes.size() == 1 && regexes[0].text == ".*") {
         usedRegexes.insert(regexes[0]);
         return true;
      }

      const auto matchBranch = [&usedRegexes, bName](const ReadSpeedRegex &regex) {
         bool match = std::regex_match(bName, regex.regex);

         if (match)
            usedRegexes.insert(regex);

         return match;
      };

      const auto iterator = std::find_if(regexes.begin(), regexes.end(), matchBranch);
      return iterator != regexes.end();
   };
   std::copy_if(unfilteredBranchNames.begin(), unfilteredBranchNames.end(), std::back_inserter(branchNames),
                filterBranchName);

   if (branchNames.empty()) {
      std::cerr << "Provided branch regexes didn't match any branches in tree '" + treeName + "' from file '" +
                      fileName + ".\n";
      std::terminate();
   }
   if (usedRegexes.size() != regexes.size()) {
      std::string errString = "The following regexes didn't match any branches in tree '" + treeName + "' from file '" +
                              fileName + "', this is probably unintended:\n";
      for (const auto &regex : regexes) {
         if (usedRegexes.find(regex) == usedRegexes.end())
            errString += '\t' + regex.text + '\n';
      }
      std::cerr << errString;
      std::terminate();
   }

   return branchNames;
}

std::vector<std::vector<std::string>> GetPerFileBranchNames(const Data &d)
{
   auto treeIdx = 0;
   std::vector<std::vector<std::string>> fileBranchNames;

   std::vector<ReadSpeedRegex> regexes;
   if (d.fUseRegex)
      std::transform(d.fBranchNames.begin(), d.fBranchNames.end(), std::back_inserter(regexes), [](std::string text) {
         return ReadSpeedRegex{text, std::regex(text)};
      });

   for (const auto &fName : d.fFileNames) {
      std::vector<std::string> branchNames;
      if (d.fUseRegex)
         branchNames = GetMatchingBranchNames(fName, d.fTreeNames[treeIdx], regexes);
      else
         branchNames = d.fBranchNames;

      fileBranchNames.push_back(branchNames);

      if (d.fTreeNames.size() > 1)
         ++treeIdx;
   }
   
   return fileBranchNames;
}

ByteData SumBytes(const std::vector<ByteData> &bytesData) {
   const auto uncompressedBytes =
      std::accumulate(bytesData.begin(), bytesData.end(), 0ull,
                        [](ULong64_t sum, const ByteData &o) { return sum + o.fUncompressedBytesRead; });
   const auto compressedBytes =
      std::accumulate(bytesData.begin(), bytesData.end(), 0ull,
                        [](ULong64_t sum, const ByteData &o) { return sum + o.fCompressedBytesRead; });

   return {uncompressedBytes, compressedBytes};
};

// Read branches listed in branchNames in tree treeName in file fileName, return number of uncompressed bytes read.
ByteData ReadSpeed::ReadTree(TFile *f, const std::string &treeName, const std::vector<std::string> &branchNames,
                             EntryRange range)
{
   std::unique_ptr<TTree> t(f->Get<TTree>(treeName.c_str()));
   if (t == nullptr)
      throw std::runtime_error("Could not retrieve tree '" + treeName + "' from file '" + f->GetName() + '\'');

   t->SetBranchStatus("*", 0);

   std::vector<TBranch *> branches;
   for (const auto &bName : branchNames) {
      auto *b = t->GetBranch(bName.c_str());
      if (b == nullptr)
         throw std::runtime_error("Could not retrieve branch '" + bName + "' from tree '" + t->GetName() +
                                  "' in file '" + t->GetCurrentFile()->GetName() + '\'');

      b->SetStatus(1);
      branches.push_back(b);
   }

   const auto nEntries = t->GetEntries();
   if (range.fStart == -1ll)
      range = EntryRange{0ll, nEntries};
   else if (range.fEnd > nEntries)
      throw std::runtime_error("Range end (" + std::to_string(range.fEnd) + ") is beyond the end of tree '" +
                               t->GetName() + "' in file '" + t->GetCurrentFile()->GetName() + "' with " +
                               std::to_string(nEntries) + " entries.");

   ULong64_t bytesRead = 0;
   const ULong64_t fileStartBytes = f->GetBytesRead();
   for (auto e = range.fStart; e < range.fEnd; ++e)
      for (auto *b : branches)
         bytesRead += b->GetEntry(e);

   const ULong64_t fileBytesRead = f->GetBytesRead() - fileStartBytes;
   return {bytesRead, fileBytesRead};
}

Result ReadSpeed::EvalThroughputST(const Data &d)
{
   auto treeIdx = 0;
   auto fileIdx = 0;
   ULong64_t uncompressedBytesRead = 0;
   ULong64_t compressedBytesRead = 0;

   TStopwatch sw;
   const auto fileBranchNames = GetPerFileBranchNames(d);

   for (const auto &fileName : d.fFileNames) {
      auto f = std::unique_ptr<TFile>(TFile::Open(fileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
      if (f == nullptr || f->IsZombie())
         throw std::runtime_error("Could not open file '" + fileName + '\'');

      sw.Start(kFALSE);

      const auto byteData = ReadTree(f.get(), d.fTreeNames[treeIdx], fileBranchNames[fileIdx]);
      uncompressedBytesRead += byteData.fUncompressedBytesRead;
      compressedBytesRead += byteData.fCompressedBytesRead;

      if (d.fTreeNames.size() > 1)
         ++treeIdx;
      ++fileIdx;

      sw.Stop();
   }

   return {sw.RealTime(), sw.CpuTime(), 0., 0., uncompressedBytesRead, compressedBytesRead, 0};
}

// Return a vector of EntryRanges per file, i.e. a vector of vectors of EntryRanges with outer size equal to
// d.fFileNames.
std::vector<std::vector<EntryRange>> ReadSpeed::GetClusters(const Data &d)
{
   auto treeIdx = 0;
   const auto nFiles = d.fFileNames.size();
   std::vector<std::vector<EntryRange>> ranges(nFiles);
   for (auto fileIdx = 0u; fileIdx < nFiles; ++fileIdx) {
      const auto &fileName = d.fFileNames[fileIdx];
      std::unique_ptr<TFile> f(TFile::Open(fileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
      if (f == nullptr || f->IsZombie())
         throw std::runtime_error("There was a problem opening file '" + fileName + '\'');
      const auto &treeName = d.fTreeNames.size() > 1 ? d.fTreeNames[fileIdx] : d.fTreeNames[0];
      auto *t = f->Get<TTree>(treeName.c_str()); // TFile owns this TTree
      if (t == nullptr)
         throw std::runtime_error("There was a problem retrieving TTree '" + treeName + "' from file '" + fileName +
                                  '\'');

      const auto nEntries = t->GetEntries();
      auto it = t->GetClusterIterator(0);
      Long64_t start = 0;
      std::vector<EntryRange> rangesInFile;
      while ((start = it.Next()) < nEntries)
         rangesInFile.emplace_back(EntryRange{start, it.GetNextEntry()});
      ranges[fileIdx] = std::move(rangesInFile);
      if (d.fTreeNames.size() > 1)
         ++treeIdx;
   }
   return ranges;
}

// Mimic the logic of TTreeProcessorMT::MakeClusters: merge entry ranges together such that we
// run around TTreeProcessorMT::GetTasksPerWorkerHint tasks per worker thread.
// TODO it would be better to expose TTreeProcessorMT's actual logic and call the exact same method from here
std::vector<std::vector<EntryRange>>
ReadSpeed::MergeClusters(std::vector<std::vector<EntryRange>> &&clusters, unsigned int maxTasksPerFile)
{
   std::vector<std::vector<EntryRange>> mergedClusters(clusters.size());

   auto clustersIt = clusters.begin();
   auto mergedClustersIt = mergedClusters.begin();
   for (; clustersIt != clusters.end(); clustersIt++, mergedClustersIt++) {
      const auto nClustersInThisFile = clustersIt->size();
      const auto nFolds = nClustersInThisFile / maxTasksPerFile;
      // If the number of clusters is less than maxTasksPerFile
      // we take the clusters as they are
      if (nFolds == 0) {
         *mergedClustersIt = *clustersIt;
         continue;
      }
      // Otherwise, we have to merge clusters, distributing the reminder evenly
      // between the first clusters
      auto nReminderClusters = nClustersInThisFile % maxTasksPerFile;
      const auto &clustersInThisFile = *clustersIt;
      for (auto i = 0ULL; i < nClustersInThisFile; ++i) {
         const auto start = clustersInThisFile[i].fStart;
         // We lump together at least nFolds clusters, therefore
         // we need to jump ahead of nFolds-1.
         i += (nFolds - 1);
         // We now add a cluster if we have some reminder left
         if (nReminderClusters > 0) {
            i += 1U;
            nReminderClusters--;
         }
         const auto end = clustersInThisFile[i].fEnd;
         mergedClustersIt->emplace_back(EntryRange({start, end}));
      }
      assert(nReminderClusters == 0 && "This should never happen, cluster-merging logic is broken.");
   }

   return mergedClusters;
}

Result ReadSpeed::EvalThroughputMT(const Data &d, unsigned nThreads)
{
   ROOT::TThreadExecutor pool(nThreads);
   const auto actualThreads = ROOT::GetThreadPoolSize();
   if (actualThreads != nThreads)
      std::cerr << "Running with " << actualThreads << " threads even though " << nThreads << " were requested.\n";

   TStopwatch clsw;
   clsw.Start();
   const unsigned int maxTasksPerFile =
      std::ceil(float(ROOT::TTreeProcessorMT::GetTasksPerWorkerHint() * actualThreads) / float(d.fFileNames.size()));

   const auto rangesPerFile = MergeClusters(GetClusters(d), maxTasksPerFile);
   clsw.Stop();

   const size_t nranges =
      std::accumulate(rangesPerFile.begin(), rangesPerFile.end(), 0u, [](size_t s, auto &r) { return s + r.size(); });
   std::cout << "Total number of tasks: " << nranges << '\n';

   const auto fileBranchNames = GetPerFileBranchNames(d);

   ROOT::Internal::RSlotStack slotStack(actualThreads);
   std::vector<int> lastFileIdxs;
   std::vector<std::unique_ptr<TFile>> lastTFiles;
   for (unsigned int i = 0; i < actualThreads; ++i) {
      lastFileIdxs.push_back(-1);
      lastTFiles.push_back(std::make_unique<TFile>());
   }

   auto processFile = [&](int fileIdx) {
      const auto &fileName = d.fFileNames[fileIdx];
      const auto &treeName = d.fTreeNames.size() > 1 ? d.fTreeNames[fileIdx] : d.fTreeNames[0];
      const auto &branchNames = fileBranchNames[fileIdx];

      auto readRange = [&](const EntryRange &range) -> ByteData {
         auto slotIndex = slotStack.GetSlot();
         auto &file = lastTFiles[slotIndex];
         auto &lastIndex = lastFileIdxs[slotIndex];

         if (lastIndex != fileIdx) {
            file.reset(TFile::Open(fileName.c_str(), "READ_WITHOUT_GLOBALREGISTRATION"));
            lastIndex = fileIdx;
         }

         if (file == nullptr || file->IsZombie())
            throw std::runtime_error("Could not open file '" + fileName + '\'');

         auto result = ReadTree(file.get(), treeName, branchNames, range);

         slotStack.ReturnSlot(slotIndex);

         return result;
      };

      const auto byteData = pool.MapReduce(readRange, rangesPerFile[fileIdx], SumBytes);

      return byteData;
   };

   TStopwatch sw;
   sw.Start();
   const auto totalByteData = pool.MapReduce(processFile, ROOT::TSeqUL{d.fFileNames.size()}, SumBytes);
   sw.Stop();

   return {sw.RealTime(),
           sw.CpuTime(),
           clsw.RealTime(),
           clsw.CpuTime(),
           totalByteData.fUncompressedBytesRead,
           totalByteData.fCompressedBytesRead,
           actualThreads};
}

Result ReadSpeed::EvalThroughput(const Data &d, unsigned nThreads)
{
   if (d.fTreeNames.empty()) {
      std::cerr << "Please provide at least one tree name\n";
      std::terminate();
   }
   if (d.fFileNames.empty()) {
      std::cerr << "Please provide at least one file name\n";
      std::terminate();
   }
   if (d.fBranchNames.empty()) {
      std::cerr << "Please provide at least one branch name\n";
      std::terminate();
   }
   if (d.fTreeNames.size() != 1 && d.fTreeNames.size() != d.fFileNames.size()) {
      std::cerr << "Please provide either one tree name or as many as the file names\n";
      std::terminate();
   }

   return nThreads > 0 ? EvalThroughputMT(d, nThreads) : EvalThroughputST(d);
}
