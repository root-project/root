/**
 \file ROOT/RTTreeDS.hxx
 \ingroup dataframe
 \author Vincenzo Eduardo Padulano
 \date 2024-12
*/

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_INTERNAL_RDF_RTTREEDS
#define ROOT_INTERNAL_RDF_RTTREEDS

#include "TTree.h"
#include "TTreeReader.h"
#include "TFile.h"

#include "ROOT/InternalTreeUtils.hxx" // GetTopLevelBranchNames
#include "ROOT/RDataSource.hxx"

#include "ROOT/RDF/RLoopManager.hxx" // GetBranchNames
#include "ROOT/RDF/Utils.hxx"        // GetBranchOrLeafTypeName
#include "ROOT/RDF/RTreeColumnReader.hxx"
#include "ROOT/RVec.hxx"

#include "TClassEdit.h"

#include <memory>
#include <string>
#include <vector>
#include <string_view>
#include <array>
#include <iostream>
#include <sstream>

// Begin forward decls

namespace ROOT {
class RDataFrame;
}

namespace ROOT::Detail::RDF {
class RLoopManager;
}

namespace ROOT::RDF {
class RSampleInfo;
}

namespace ROOT::RDF::Experimental {
class RSample;
}

namespace ROOT::TreeUtils {
struct RFriendInfo;
}

class TDirectory;

// End forward decls

namespace ROOT::Internal::RDF {

class RTTreeDS final : public ROOT::RDF::RDataSource {
   std::vector<std::string> fBranchNamesWithDuplicates{};
   std::vector<std::string> fBranchNamesWithoutDuplicates{};
   std::vector<std::string> fTopLevelBranchNames{};

   std::shared_ptr<TTree> fTree{};

   std::unique_ptr<TTreeReader> fTreeReader{};

   std::vector<std::unique_ptr<TChain>> fFriends{};

   std::tuple<bool, std::string, RTreeUntypedArrayColumnReader::ECollectionType>
   GetCollectionInfo(const std::string &typeName) const
   {
      const auto beginType = typeName.substr(0, typeName.find_first_of('<') + 1);

      // Find TYPE from ROOT::RVec<TYPE>
      if (auto pos = beginType.find("RVec<"); pos != std::string::npos) {
         const auto begin = typeName.find_first_of('<', pos) + 1;
         const auto end = typeName.find_last_of('>');
         const auto innerTypeName = typeName.substr(begin, end - begin);
         if (innerTypeName.find("bool") != std::string::npos)
            return {true, innerTypeName, RTreeUntypedArrayColumnReader::ECollectionType::kRVecBool};
         else
            return {true, innerTypeName, RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
      }

      // Find TYPE from std::array<TYPE,N>
      if (auto pos = beginType.find("array<"); pos != std::string::npos) {
         const auto begin = typeName.find_first_of('<', pos) + 1;
         const auto end = typeName.find_last_of('>');
         const auto arrTemplArgs = typeName.substr(begin, end - begin);
         const auto lastComma = arrTemplArgs.find_last_of(',');
         return {true, arrTemplArgs.substr(0, lastComma), RTreeUntypedArrayColumnReader::ECollectionType::kStdArray};
      }

      return {false, "", RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
   }

   ROOT::RDF::RSampleInfo
   CreateSampleInfo(const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const final;

   void RunFinalChecks(bool nodesLeftNotRun) const final
   {
      if (fTreeReader->GetEntryStatus() != TTreeReader::kEntryBeyondEnd && nodesLeftNotRun) {
         // something went wrong in the TTreeReader event loop
         throw std::runtime_error("An error was encountered while processing the data. TTreeReader status code is: " +
                                  std::to_string(fTreeReader->GetEntryStatus()));
      }
   }

   bool ValidRead(TTreeReader::EEntryStatus entryStatus) const
   {
      switch (entryStatus) {
      case TTreeReader::kEntryValid: return true;
      case TTreeReader::kIndexedFriendNoMatch: return true;
      case TTreeReader::kMissingBranchWhenSwitchingTree: return true;
      default: return false;
      }
   }

   void Setup(std::shared_ptr<TTree> &&tree, const ROOT::TreeUtils::RFriendInfo *friendInfo = nullptr);

public:
   RTTreeDS(std::shared_ptr<TTree> tree);
   RTTreeDS(std::shared_ptr<TTree> tree, const ROOT::TreeUtils::RFriendInfo &friendInfo);
   RTTreeDS(std::string_view treeName, TDirectory *dirPtr);
   RTTreeDS(std::string_view treeName, std::string_view fileNameGlob);
   RTTreeDS(std::string_view treeName, const std::vector<std::string> &fileNameGlobs);

   // Rule of five
   RTTreeDS(const RTTreeDS &) = delete;
   RTTreeDS &operator=(const RTTreeDS &) = delete;
   RTTreeDS(RTTreeDS &&) = delete;
   RTTreeDS &operator=(RTTreeDS &&) = delete;
   ~RTTreeDS() final = default;

   void Initialize() final
   {
      if (fGlobalEntryRange.has_value() && fGlobalEntryRange->first <= std::numeric_limits<Long64_t>::max() &&
          fGlobalEntryRange->second <= std::numeric_limits<Long64_t>::max() && fTreeReader &&
          fTreeReader->SetEntriesRange(fGlobalEntryRange->first, fGlobalEntryRange->second) !=
             TTreeReader::kEntryValid) {
         throw std::logic_error("Something went wrong in initializing the TTreeReader.");
      }
   }

   void Finalize() final
   {
      // At the end of the event loop, reset the TTreeReader to be ready for
      // a possible new run.
      if (fTreeReader)
         fTreeReader->Restart();
   }

   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final
   {
      auto treeOrChain = fTreeReader->GetTree();
      assert(treeOrChain != nullptr);

      // End of dataset or entry range
      if (fTreeReader->GetCurrentEntry() >= treeOrChain->GetEntriesFast() - 1 ||
          (fGlobalEntryRange.has_value() &&
           (static_cast<ULong64_t>(fTreeReader->GetCurrentEntry()) >= fGlobalEntryRange->first &&
            static_cast<ULong64_t>(fTreeReader->GetCurrentEntry()) == fGlobalEntryRange->second - 1))) {
         // Place the TTreeReader beyond the end of the dataset, so RunFinalChecks can work properly
         fTreeReader->Next();
         return {};
      }

      if (auto chain = dynamic_cast<TChain *>(treeOrChain)) {
         // We are either at a complete new beginning (entry == -1) or at the
         // end of processing of the previous tree in the chain. Go to the next
         // entry, which should always be the first entry in a tree. This allows
         // to get the proper tree offset for the range.
         fTreeReader->Next();
         if (!ValidRead(fTreeReader->GetEntryStatus()))
            return {};
         auto treeOffsets = chain->GetTreeOffset();
         auto treeNumber = chain->GetTreeNumber();
         const ULong64_t thisTreeBegin = treeOffsets[treeNumber];
         const ULong64_t thisTreeEnd = treeOffsets[treeNumber + 1];
         // Restrict the range to the global range if available
         const ULong64_t rangeBegin =
            fGlobalEntryRange.has_value() ? std::max(thisTreeBegin, fGlobalEntryRange->first) : thisTreeBegin;
         const ULong64_t rangeEnd =
            fGlobalEntryRange.has_value() ? std::min(thisTreeEnd, fGlobalEntryRange->second) : thisTreeEnd;
         return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeEnd}};
      } else {
         // Restrict the range to the global range if available
         const ULong64_t rangeBegin = fGlobalEntryRange.has_value() ? std::max(0ull, fGlobalEntryRange->first) : 0ull;
         const ULong64_t rangeEnd =
            fGlobalEntryRange.has_value()
               ? std::min(static_cast<ULong64_t>(treeOrChain->GetEntries()), fGlobalEntryRange->second)
               : static_cast<ULong64_t>(treeOrChain->GetEntries());
         return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeEnd}};
      }
   }

   const std::vector<std::string> &GetColumnNames() const { return fBranchNamesWithDuplicates; }

   bool HasColumn(std::string_view colName) const
   {
      return std::find(fBranchNamesWithDuplicates.begin(), fBranchNamesWithDuplicates.end(), colName) !=
             fBranchNamesWithDuplicates.end();
   }

   std::string GetTypeName(std::string_view colName) const final
   {
      auto colTypeName = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*fTree, std::string(colName));
      if (TClassEdit::IsSTLCont(colTypeName) == ROOT::ESTLType::kSTLvector) {
         std::vector<std::string> split;
         int dummy;
         TClassEdit::GetSplit(colTypeName.c_str(), split, dummy);
         auto &valueType = split[1];
         colTypeName = "ROOT::VecOps::RVec<" + valueType + ">";
      }
      return colTypeName;
   }

   std::string GetTypeNameWithOpts(std::string_view colName, bool vector2RVec) const final
   {
      auto colTypeName = ROOT::Internal::RDF::GetBranchOrLeafTypeName(*fTree, std::string(colName));
      if (vector2RVec && TClassEdit::IsSTLCont(colTypeName) == ROOT::ESTLType::kSTLvector) {
         std::vector<std::string> split;
         int dummy;
         TClassEdit::GetSplit(colTypeName.c_str(), split, dummy);
         auto &valueType = split[1];
         colTypeName = "ROOT::VecOps::RVec<" + valueType + ">";
      }
      return colTypeName;
   }

   bool SetEntry(unsigned int, ULong64_t entry) final
   {
      // The first entry of each tree in a chain is read in GetEntryRanges, we avoid repeating it here
      if (fTreeReader->GetCurrentEntry() != static_cast<Long64_t>(entry))
         fTreeReader->SetEntry(entry);
      return ValidRead(fTreeReader->GetEntryStatus());
   }

   Record_t GetColumnReadersImpl(std::string_view /* name */, const std::type_info & /* ti */)
   {
      // This datasource uses the newer GetColumnReaders() API
      return {};
   }

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int /*slot*/, std::string_view colName, const std::type_info &tid) final
   {
      // CAVEAT: the TTreeReader passed here must stay the same for the entire execution of the program
      // because the column readers will be cached in RLoopManager. Otherwise the column reader would
      // have a dangling refernce to the TTreeReader.
      // Early return for the case of opaque column reader requested by FilterAvailable/FilterMissing
      if (tid == typeid(void))
         return std::make_unique<RTreeOpaqueColumnReader>(*fTreeReader, colName);

      const auto typeName = ROOT::Internal::RDF::TypeID2TypeName(tid);
      if (auto &&[toConvert, innerTypeName, collType] = GetCollectionInfo(typeName); toConvert)
         return std::make_unique<RTreeUntypedArrayColumnReader>(*fTreeReader, colName, innerTypeName, collType);
      else
         return std::make_unique<RTreeUntypedValueColumnReader>(*fTreeReader, colName, typeName);
   }

   std::string GetLabel() final { return "TTreeDS"; }

   TTree *GetTree()
   {
      assert(fTree);
      return fTree.get();
   }

   const std::vector<std::string> &GetTopLevelFieldNames() const final { return fTopLevelBranchNames; }
   const std::vector<std::string> &GetColumnNamesNoDuplicates() const final { return fBranchNamesWithoutDuplicates; }

   void InitializeWithOpts(const std::set<std::string> &suppressErrorsForMissingBranches)
   {
      Initialize();
      if (fTreeReader)
         fTreeReader->SetSuppressErrorsForMissingBranches(
            std::vector<std::string>(suppressErrorsForMissingBranches.begin(), suppressErrorsForMissingBranches.end()));
   }

   std::string DescribeDataset() final
   {
      const auto treeName = fTree->GetName();
      const auto isTChain = dynamic_cast<TChain *>(fTree.get()) ? true : false;
      const auto treeType = isTChain ? "TChain" : "TTree";
      const auto isInMemory = !isTChain && !fTree->GetCurrentFile() ? true : false;
      const auto friendInfo = ROOT::Internal::TreeUtils::GetFriendInfo(*fTree);
      const auto hasFriends = friendInfo.fFriendNames.empty() ? false : true;
      std::stringstream ss;
      ss << "Dataframe from " << treeType;
      if (*treeName != 0) {
         ss << " " << treeName;
      }
      if (isInMemory) {
         ss << " (in-memory)";
      } else {
         const auto files = ROOT::Internal::TreeUtils::GetFileNamesFromTree(*fTree);
         const auto numFiles = files.size();
         if (numFiles == 1) {
            ss << " in file " << files[0];
         } else {
            ss << " in files\n";
            for (auto i = 0u; i < numFiles; i++) {
               ss << "  " << files[i];
               if (i < numFiles - 1)
                  ss << '\n';
            }
         }
      }
      if (hasFriends) {
         const auto numFriends = friendInfo.fFriendNames.size();
         if (numFriends == 1) {
            ss << "\nwith friend\n";
         } else {
            ss << "\nwith friends\n";
         }
         for (auto i = 0u; i < numFriends; i++) {
            const auto nameAlias = friendInfo.fFriendNames[i];
            const auto files = friendInfo.fFriendFileNames[i];
            const auto numFiles = files.size();
            const auto subnames = friendInfo.fFriendChainSubNames[i];
            ss << "  " << nameAlias.first;
            if (nameAlias.first != nameAlias.second)
               ss << " (" << nameAlias.second << ")";
            // case: TTree as friend
            if (numFiles == 1) {
               ss << " " << files[0];
            }
            // case: TChain as friend
            else {
               ss << '\n';
               for (auto j = 0u; j < numFiles; j++) {
                  ss << "    " << subnames[j] << " " << files[j];
                  if (j < numFiles - 1)
                     ss << '\n';
               }
            }
            if (i < numFriends - 1)
               ss << '\n';
         }
      }
      return ss.str();
   }

   std::string AsString() final { return "TTree data source"; }

   std::size_t GetNFiles() const final
   {
      if (dynamic_cast<TChain *>(fTree.get()))
         return ROOT::Internal::TreeUtils::GetFileNamesFromTree(*fTree).size();
      const auto *f = fTree->GetCurrentFile();
      return f ? 1 : 0;
   }

   void ProcessMT(ROOT::Detail::RDF::RLoopManager &lm) final;
};

ROOT::RDataFrame FromTTree(std::string_view treeName, std::string_view fileNameGlob);
ROOT::RDataFrame FromTTree(std::string_view treeName, const std::vector<std::string> &fileNameGlobs);

} // namespace ROOT::Internal::RDF

#endif
