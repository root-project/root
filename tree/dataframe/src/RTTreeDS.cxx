#include <sstream>

#include <ROOT/RTTreeDS.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RLoopManager.hxx> // GetBranchNames
#include <ROOT/RDF/RTreeColumnReader.hxx>
#include <ROOT/RDF/Utils.hxx> // GetBranchOrLeafTypeName

#include <TClassEdit.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <ROOT/RFriendInfo.hxx>
#include <ROOT/InternalTreeUtils.hxx> // GetTopLevelBranchNames

#ifdef R__USE_IMT
#include <TROOT.h>
#include <TEntryList.h>
#include <ROOT/TTreeProcessorMT.hxx>
#include <ROOT/RSlotStack.hxx>
#endif

namespace {
bool ValidRead(TTreeReader::EEntryStatus entryStatus)
{
   switch (entryStatus) {
   case TTreeReader::kEntryValid: return true;
   case TTreeReader::kIndexedFriendNoMatch: return true;
   case TTreeReader::kMissingBranchWhenSwitchingTree: return true;
   default: return false;
   }
}

std::tuple<bool, std::string, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType>
GetCollectionInfo(const std::string &typeName)
{
   const auto beginType = typeName.substr(0, typeName.find_first_of('<') + 1);

   // Find TYPE from ROOT::RVec<TYPE>
   if (auto pos = beginType.find("RVec<"); pos != std::string::npos) {
      const auto begin = typeName.find_first_of('<', pos) + 1;
      const auto end = typeName.find_last_of('>');
      const auto innerTypeName = typeName.substr(begin, end - begin);
      if (innerTypeName.find("bool") != std::string::npos)
         return {true, innerTypeName, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVecBool};
      else
         return {true, innerTypeName, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
   }

   // Find TYPE from std::array<TYPE,N>
   if (auto pos = beginType.find("array<"); pos != std::string::npos) {
      const auto begin = typeName.find_first_of('<', pos) + 1;
      const auto end = typeName.find_last_of('>');
      const auto arrTemplArgs = typeName.substr(begin, end - begin);
      const auto lastComma = arrTemplArgs.find_last_of(',');
      return {true, arrTemplArgs.substr(0, lastComma),
              ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kStdArray};
   }

   return {false, "", ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
}
} // namespace

// Destructor is defined here, where the data member types are actually available
ROOT::Internal::RDF::RTTreeDS::~RTTreeDS()
{
   if (fNoCleanupNotifier && fTree)
      // fNoCleanupNotifier was created only if the input TTree is a TChain
      fNoCleanupNotifier->RemoveLink(*static_cast<TChain *>(fTree.get()));
};

void ROOT::Internal::RDF::RTTreeDS::Setup(std::shared_ptr<TTree> &&tree, const ROOT::TreeUtils::RFriendInfo *friendInfo)
{
   fTree = tree;

   if (friendInfo) {
      fFriends = ROOT::Internal::TreeUtils::MakeFriends(*friendInfo);
      for (std::size_t i = 0ul; i < fFriends.size(); i++) {
         const auto &thisFriendAlias = friendInfo->fFriendNames[i].second;
         fTree->AddFriend(fFriends[i].get(), thisFriendAlias.c_str());
      }
   }

   if (fBranchNamesWithDuplicates.empty())
      fBranchNamesWithDuplicates = ROOT::Internal::RDF::GetBranchNames(*fTree);
   if (fBranchNamesWithoutDuplicates.empty())
      fBranchNamesWithoutDuplicates = ROOT::Internal::RDF::GetBranchNames(*fTree, /*allowDuplicates*/ false);
   if (fTopLevelBranchNames.empty())
      fTopLevelBranchNames = ROOT::Internal::TreeUtils::GetTopLevelBranchNames(*fTree);
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::shared_ptr<TTree> tree)
{
   assert(tree && "No tree passed to the constructor of RTTreeDS!");
   Setup(std::move(tree));
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::shared_ptr<TTree> tree, const ROOT::TreeUtils::RFriendInfo &friendInfo)
{
   assert(tree && "No tree passed to the constructor of RTTreeDS!");
   Setup(std::move(tree), &friendInfo);
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::string_view treeName, TDirectory *dirPtr)
{
   if (!dirPtr) {
      throw std::runtime_error("RDataFrame: invalid TDirectory when constructing the data source.");
   }
   const std::string treeNameInt(treeName);
   auto tree = dirPtr->Get<TTree>(treeName.data());
   if (!tree) {
      throw std::runtime_error("RDataFrame: TTree dataset '" + std::string(treeName) + "' cannot be found in '" +
                               dirPtr->GetName() + "'.");
   }
   Setup(ROOT::Internal::RDF::MakeAliasedSharedPtr(tree));

   // We keep the existing functionality from RLoopManager, until proven not necessary.
   // See https://github.com/root-project/root/pull/10729
   // The only constructors that were using this functionality are the ones taking a TDirectory * and the ones taking
   // a file name or list of file names, see
   // https://github.com/root-project/root/blob/f8b8277627f08cb79d71cec1006b219a82ae273c/tree/dataframe/src/RDataFrame.cxx
   if (auto ch = dynamic_cast<TChain *>(fTree.get()); ch && !fNoCleanupNotifier) {
      fNoCleanupNotifier = std::make_unique<ROOT::Internal::TreeUtils::RNoCleanupNotifier>();
      fNoCleanupNotifier->RegisterChain(*ch);
   }
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::string_view treeName, std::string_view fileNameGlob)
{
   std::string treeNameInt{treeName};
   std::string fileNameGlobInt{fileNameGlob};
   auto chain = ROOT::Internal::TreeUtils::MakeChainForMT(treeNameInt.c_str());
   chain->Add(fileNameGlobInt.c_str());

   Setup(std::move(chain));
   // We keep the existing functionality from RLoopManager, until proven not necessary.
   // See https://github.com/root-project/root/pull/10729
   // The only constructors that were using this functionality are the ones taking a TDirectory * and the ones taking
   // a file name or list of file names, see
   // https://github.com/root-project/root/blob/f8b8277627f08cb79d71cec1006b219a82ae273c/tree/dataframe/src/RDataFrame.cxx
   if (auto ch = dynamic_cast<TChain *>(fTree.get()); ch && !fNoCleanupNotifier) {
      fNoCleanupNotifier = std::make_unique<ROOT::Internal::TreeUtils::RNoCleanupNotifier>();
      fNoCleanupNotifier->RegisterChain(*ch);
   }
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::string_view treeName, const std::vector<std::string> &fileNameGlobs)
{
   std::string treeNameInt(treeName);
   auto chain = ROOT::Internal::TreeUtils::MakeChainForMT(treeNameInt);
   for (auto &&f : fileNameGlobs)
      chain->Add(f.c_str());

   Setup(std::move(chain));

   // We keep the existing functionality from RLoopManager, until proven not necessary.
   // See https://github.com/root-project/root/pull/10729
   // The only constructors that were using this functionality are the ones taking a TDirectory * and the ones taking
   // a file name or list of file names, see
   // https://github.com/root-project/root/blob/f8b8277627f08cb79d71cec1006b219a82ae273c/tree/dataframe/src/RDataFrame.cxx
   if (auto ch = dynamic_cast<TChain *>(fTree.get()); ch && !fNoCleanupNotifier) {
      fNoCleanupNotifier = std::make_unique<ROOT::Internal::TreeUtils::RNoCleanupNotifier>();
      fNoCleanupNotifier->RegisterChain(*ch);
   }
}

ROOT::RDataFrame ROOT::Internal::RDF::FromTTree(std::string_view treeName, std::string_view fileNameGlob)
{
   return ROOT::RDataFrame(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(treeName, fileNameGlob));
}

ROOT::RDataFrame
ROOT::Internal::RDF::FromTTree(std::string_view treeName, const std::vector<std::string> &fileNameGlobs)
{
   return ROOT::RDataFrame(std::make_unique<ROOT::Internal::RDF::RTTreeDS>(treeName, fileNameGlobs));
}

ROOT::RDF::RSampleInfo ROOT::Internal::RDF::RTTreeDS::CreateSampleInfo(
   unsigned int, const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const
{
   // one GetTree to retrieve the TChain, another to retrieve the underlying TTree
   auto *tree = fTreeReader->GetTree()->GetTree();
   // tree might be missing e.g. when a file in a chain does not exist
   if (!tree)
      return ROOT::RDF::RSampleInfo{};

   const std::string treename = ROOT::Internal::TreeUtils::GetTreeFullPaths(*tree)[0];
   auto *file = tree->GetCurrentFile();
   const std::string fname = file != nullptr ? file->GetName() : "#inmemorytree#";

   std::pair<Long64_t, Long64_t> range = fTreeReader->GetEntriesRange();
   R__ASSERT(range.first >= 0);
   if (range.second == -1) {
      range.second = tree->GetEntries(); // convert '-1', i.e. 'until the end', to the actual entry number
   }
   // If the tree is stored in a subdirectory, treename will be the full path to it starting with the root directory '/'
   const std::string &id = fname + (treename.rfind('/', 0) == 0 ? "" : "/") + treename;
   if (sampleMap.empty()) {
      return RSampleInfo(id, range);
   } else {
      if (sampleMap.find(id) == sampleMap.end())
         throw std::runtime_error("Full sample identifier '" + id + "' cannot be found in the available samples.");
      return RSampleInfo(id, range, sampleMap.at(id));
   }
}

void ROOT::Internal::RDF::RTTreeDS::ProcessMT(ROOT::Detail::RDF::RLoopManager &lm)
{
#ifdef R__USE_IMT
   ROOT::Internal::RSlotStack slotStack(fNSlots);
   std::atomic<ULong64_t> entryCount(0ull);

   const auto &entryList = fTree->GetEntryList() ? *fTree->GetEntryList() : TEntryList();
   const auto &suppressErrorsForMissingBranches = lm.GetSuppressErrorsForMissingBranches();
   auto tp{fGlobalEntryRange.has_value()
              ? std::make_unique<ROOT::TTreeProcessorMT>(*fTree, fNSlots, fGlobalEntryRange.value(),
                                                         suppressErrorsForMissingBranches)
              : std::make_unique<ROOT::TTreeProcessorMT>(*fTree, entryList, fNSlots, suppressErrorsForMissingBranches)};

   tp->Process([&lm, &slotStack, &entryCount](TTreeReader &treeReader) {
      lm.TTreeThreadTask(treeReader, slotStack, entryCount);
   });

   if (fGlobalEntryRange.has_value()) {
      auto &&[begin, end] = fGlobalEntryRange.value();
      auto &&processedEntries = entryCount.load();
      if ((end - begin) > processedEntries) {
         Warning("RDataFrame::Run",
                 "RDataFrame stopped processing after %lld entries, whereas an entry range (begin=%lld,end=%lld) was "
                 "requested. Consider adjusting the end value of the entry range to a maximum of %lld.",
                 processedEntries, begin, end, begin + processedEntries);
      }
   }
#else
   (void)lm;
#endif
}

std::size_t ROOT::Internal::RDF::RTTreeDS::GetNFiles() const
{
   assert(fTree && "The internal TTree is not available, something went wrong.");
   if (dynamic_cast<TChain *>(fTree.get()))
      return ROOT::Internal::TreeUtils::GetFileNamesFromTree(*fTree).size();

   return fTree->GetCurrentFile() ? 1 : 0;
}

std::string ROOT::Internal::RDF::RTTreeDS::DescribeDataset()
{
   assert(fTree && "The internal TTree is not available, something went wrong.");
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

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
ROOT::Internal::RDF::RTTreeDS::CreateColumnReader(unsigned int /*slot*/, std::string_view col, const std::type_info &ti,
                                                  TTreeReader *treeReader)
{
   // In a single thread run, use the TTreeReader data member.
   if (fTreeReader) {
      treeReader = fTreeReader.get();
   }

   // The TTreeReader might still not be available if CreateColumnReader was called before the start of the computation
   // graph execution, e.g. in AddDSColumns.
   if (!treeReader)
      return nullptr;

   if (ti == typeid(void))
      return std::make_unique<ROOT::Internal::RDF::RTreeOpaqueColumnReader>(*treeReader, col);

   const auto typeName = ROOT::Internal::RDF::TypeID2TypeName(ti);
   if (auto &&[toConvert, innerTypeName, collType] = GetCollectionInfo(typeName); toConvert)
      return std::make_unique<ROOT::Internal::RDF::RTreeUntypedArrayColumnReader>(*treeReader, col, innerTypeName,
                                                                                  collType);
   else
      return std::make_unique<ROOT::Internal::RDF::RTreeUntypedValueColumnReader>(*treeReader, col, typeName);
}

bool ROOT::Internal::RDF::RTTreeDS::SetEntry(unsigned int, ULong64_t entry)
{
   // The first entry of each tree in a chain is read in GetEntryRanges, we avoid repeating it here
   if (fTreeReader->GetCurrentEntry() != static_cast<Long64_t>(entry))
      fTreeReader->SetEntry(entry);
   return ValidRead(fTreeReader->GetEntryStatus());
}

std::string ROOT::Internal::RDF::RTTreeDS::GetTypeNameWithOpts(std::string_view colName, bool vector2RVec) const
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

std::string ROOT::Internal::RDF::RTTreeDS::GetTypeName(std::string_view colName) const
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

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::Internal::RDF::RTTreeDS::GetTTreeEntryRange(TTree &tree)
{
   // Restrict the range to the global range if available
   const ULong64_t rangeBegin = fGlobalEntryRange.has_value() ? std::max(0ull, fGlobalEntryRange->first) : 0ull;
   const ULong64_t rangeEnd = fGlobalEntryRange.has_value()
                                 ? std::min(static_cast<ULong64_t>(tree.GetEntries()), fGlobalEntryRange->second)
                                 : static_cast<ULong64_t>(tree.GetEntries());
   return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeEnd}};
}

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::Internal::RDF::RTTreeDS::GetTChainEntryRange(TChain &chain)
{
   // We are either at a complete new beginning (entry == -1) or at the
   // end of processing of the previous tree in the chain. Go to the next
   // entry, which should always be the first entry in a tree. This allows
   // to get the proper tree offset for the range.
   fTreeReader->Next();
   if (!ValidRead(fTreeReader->GetEntryStatus()))
      return {};
   auto treeOffsets = chain.GetTreeOffset();
   auto treeNumber = chain.GetTreeNumber();
   const ULong64_t thisTreeBegin = treeOffsets[treeNumber];
   const ULong64_t thisTreeEnd = treeOffsets[treeNumber + 1];
   // Restrict the range to the global range if available
   const ULong64_t rangeBegin =
      fGlobalEntryRange.has_value() ? std::max(thisTreeBegin, fGlobalEntryRange->first) : thisTreeBegin;
   const ULong64_t rangeEnd =
      fGlobalEntryRange.has_value() ? std::min(thisTreeEnd, fGlobalEntryRange->second) : thisTreeEnd;
   return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeEnd}};
}

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::Internal::RDF::RTTreeDS::GetEntryRanges()
{
   assert(fTreeReader && "TTreeReader is not available, this should never happen.");
   auto treeOrChain = fTreeReader->GetTree();
   assert(treeOrChain && "Could not retrieve TTree from TTreeReader, something went wrong.");

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
      return GetTChainEntryRange(*chain);
   } else {
      return GetTTreeEntryRange(*treeOrChain);
   }
}

void ROOT::Internal::RDF::RTTreeDS::Finalize()
{
   // At the end of the event loop, reset the TTreeReader to be ready for
   // a possible new run.
   if (fTreeReader)
      fTreeReader.reset();
}

void ROOT::Internal::RDF::RTTreeDS::Initialize()
{
   if (fNSlots == 1) {
      assert(!fTreeReader);
      fTreeReader = std::make_unique<TTreeReader>(fTree.get(), fTree->GetEntryList(), /*warnAboutLongerFriends*/ true);
      if (fGlobalEntryRange.has_value() && fGlobalEntryRange->first <= std::numeric_limits<Long64_t>::max() &&
          fGlobalEntryRange->second <= std::numeric_limits<Long64_t>::max() && fTreeReader &&
          fTreeReader->SetEntriesRange(fGlobalEntryRange->first, fGlobalEntryRange->second) !=
             TTreeReader::kEntryValid) {
         throw std::logic_error("Something went wrong in initializing the TTreeReader.");
      }
   }
}

void ROOT::Internal::RDF::RTTreeDS::InitializeWithOpts(const std::set<std::string> &suppressErrorsForMissingBranches)
{
   Initialize();
   if (fTreeReader)
      fTreeReader->SetSuppressErrorsForMissingBranches(suppressErrorsForMissingBranches);
}

void ROOT::Internal::RDF::RTTreeDS::RunFinalChecks(bool nodesLeftNotRun) const
{
   if (fTreeReader->GetEntryStatus() != TTreeReader::kEntryBeyondEnd && nodesLeftNotRun) {
      // something went wrong in the TTreeReader event loop
      throw std::runtime_error("An error was encountered while processing the data. TTreeReader status code is: " +
                               std::to_string(fTreeReader->GetEntryStatus()));
   }
}

TTree *ROOT::Internal::RDF::RTTreeDS::GetTree()
{
   assert(fTree);
   return fTree.get();
}
