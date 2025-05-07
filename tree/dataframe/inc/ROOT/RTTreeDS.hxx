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

#include "ROOT/RDataSource.hxx"

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <string_view>

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

namespace ROOT::Internal::TreeUtils {
class RNoCleanupNotifier;
}

class TChain;
class TDirectory;
class TTree;
class TTreeReader;

// End forward decls

namespace ROOT::Internal::RDF {

class RTTreeDS final : public ROOT::RDF::RDataSource {
   std::vector<std::string> fBranchNamesWithDuplicates{};
   std::vector<std::string> fBranchNamesWithoutDuplicates{};
   std::vector<std::string> fTopLevelBranchNames{};

   std::shared_ptr<TTree> fTree;

   std::unique_ptr<TTreeReader> fTreeReader;

   std::vector<std::unique_ptr<TChain>> fFriends;

   // Should be needed mostly for MT runs, but we keep it here to document and align the existing functionality
   // from RLoopManager. See https://github.com/root-project/root/pull/10729
   std::unique_ptr<ROOT::Internal::TreeUtils::RNoCleanupNotifier> fNoCleanupNotifier;

   ROOT::RDF::RSampleInfo
   CreateSampleInfo(unsigned int,
                    const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const final;

   void RunFinalChecks(bool nodesLeftNotRun) const final;

   void Setup(std::shared_ptr<TTree> &&tree, const ROOT::TreeUtils::RFriendInfo *friendInfo = nullptr);

   std::vector<std::pair<ULong64_t, ULong64_t>> GetTTreeEntryRange(TTree &tree);
   std::vector<std::pair<ULong64_t, ULong64_t>> GetTChainEntryRange(TChain &chain);

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
   ~RTTreeDS() final; // Define destructor where data member types are defined

   void Initialize() final;

   void Finalize() final;

   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;

   const std::vector<std::string> &GetColumnNames() const final { return fBranchNamesWithDuplicates; }

   bool HasColumn(std::string_view colName) const final
   {
      return std::find(fBranchNamesWithDuplicates.begin(), fBranchNamesWithDuplicates.end(), colName) !=
             fBranchNamesWithDuplicates.end();
   }

   std::string GetTypeName(std::string_view colName) const final;

   std::string GetTypeNameWithOpts(std::string_view colName, bool vector2RVec) const final;

   bool SetEntry(unsigned int, ULong64_t entry) final;

   Record_t GetColumnReadersImpl(std::string_view /* name */, const std::type_info & /* ti */) final
   {
      // This datasource uses the newer GetColumnReaders() API
      return {};
   }

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int, std::string_view, const std::type_info &) final
   {
      // This data source creates column readers via CreateColumnReader
      throw std::runtime_error("GetColumnReaders should not be called on this data source, something wrong happened!");
   }

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> CreateColumnReader(unsigned int slot, std::string_view col,
                                                                            const std::type_info &tid,
                                                                            TTreeReader *treeReader) final;

   std::string GetLabel() final { return "TTreeDS"; }

   TTree *GetTree();

   const std::vector<std::string> &GetTopLevelFieldNames() const final { return fTopLevelBranchNames; }

   const std::vector<std::string> &GetColumnNamesNoDuplicates() const final { return fBranchNamesWithoutDuplicates; }

   void InitializeWithOpts(const std::set<std::string> &suppressErrorsForMissingBranches) final;

   std::string DescribeDataset() final;

   std::string AsString() final { return "TTree data source"; }

   std::size_t GetNFiles() const final;

   void ProcessMT(ROOT::Detail::RDF::RLoopManager &lm) final;
};

ROOT::RDataFrame FromTTree(std::string_view treeName, std::string_view fileNameGlob);
ROOT::RDataFrame FromTTree(std::string_view treeName, const std::vector<std::string> &fileNameGlobs);

} // namespace ROOT::Internal::RDF

#endif
