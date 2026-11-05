#include <sstream>

#include <ROOT/RTTreeDS.hxx>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RLoopManager.hxx> // GetBranchNames
#include <ROOT/RDF/RTreeColumnReader.hxx>
#include <ROOT/RDF/Utils.hxx> // GetBranchOrLeafTypeName

#include <TBranchObject.h>
#include <TClassEdit.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TFriendElement.h>
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

   // Find TYPE from std::vector<TYPE>
   if (auto pos = beginType.find("vector<"); pos != std::string::npos) {
      const auto begin = typeName.find_first_of('<', pos) + 1;
      const auto end = typeName.find_last_of('>');
      const auto innerTypeName = typeName.substr(begin, end - begin);
      return {true, innerTypeName, ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kStdVector};
   }

   return {false, "", ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType::kRVec};
}

bool ContainsLeaf(const std::set<TLeaf *> &leaves, TLeaf *leaf)
{
   return (leaves.find(leaf) != leaves.end());
}

///////////////////////////////////////////////////////////////////////////////
/// This overload does not check whether the leaf/branch is already in bNamesReg. In case this is a friend leaf/branch,
/// `allowDuplicates` controls whether we add both `friendname.bname` and `bname` or just the shorter version.
void InsertBranchName(std::set<std::string> &bNamesReg, std::vector<std::string> &bNames, const std::string &branchName,
                      const std::string &friendName, bool allowDuplicates)
{
   if (!friendName.empty()) {
      // In case of a friend tree, users might prepend its name/alias to the branch names
      const auto friendBName = friendName + "." + branchName;
      if (bNamesReg.insert(friendBName).second)
         bNames.push_back(friendBName);
   }

   if (allowDuplicates || friendName.empty()) {
      if (bNamesReg.insert(branchName).second)
         bNames.push_back(branchName);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// This overload makes sure that the TLeaf has not been already inserted.
void InsertBranchName(std::set<std::string> &bNamesReg, std::vector<std::string> &bNames, const std::string &branchName,
                      const std::string &friendName, std::set<TLeaf *> &foundLeaves, TLeaf *leaf, bool allowDuplicates)
{
   const bool canAdd = allowDuplicates ? true : !ContainsLeaf(foundLeaves, leaf);
   if (!canAdd) {
      return;
   }

   InsertBranchName(bNamesReg, bNames, branchName, friendName, allowDuplicates);

   foundLeaves.insert(leaf);
}

void ExploreBranch(TTree &t, std::set<std::string> &bNamesReg, std::vector<std::string> &bNames, TBranch *b,
                   std::string prefix, std::string &friendName, bool allowDuplicates)
{
   // We want to avoid situations of overlap between the prefix and the
   // sub-branch name that might happen when the branch is composite, e.g.
   // prefix=reco_Wdecay2_from_tbar_4vect_NOSYS.fCoordinates.
   // subBranchName=fCoordinates.fPt
   // which would lead to a repetition of fCoordinates in the output branch name
   // Boundary to search for the token before the last dot
   auto prefixEndingDot = std::string::npos;
   if (!prefix.empty() && prefix.back() == '.')
      prefixEndingDot = prefix.size() - 2;
   std::string lastPrefixToken{};
   if (auto prefixLastRealDot = prefix.find_last_of('.', prefixEndingDot); prefixLastRealDot != std::string::npos)
      lastPrefixToken = prefix.substr(prefixLastRealDot + 1, prefixEndingDot - prefixLastRealDot);

   for (auto sb : *b->GetListOfBranches()) {
      TBranch *subBranch = static_cast<TBranch *>(sb);
      auto subBranchName = std::string(subBranch->GetName());
      auto fullName = prefix + subBranchName;

      if (auto subNameFirstDot = subBranchName.find_first_of('.'); subNameFirstDot != std::string::npos) {
         // Concatenate the prefix to the sub-branch name without overlaps
         if (!lastPrefixToken.empty() && lastPrefixToken == subBranchName.substr(0, subNameFirstDot))
            fullName = prefix + subBranchName.substr(subNameFirstDot + 1);
      }

      std::string newPrefix;
      if (!prefix.empty())
         newPrefix = fullName + ".";

      ExploreBranch(t, bNamesReg, bNames, subBranch, newPrefix, friendName, allowDuplicates);

      auto branchDirectlyFromTree = t.GetBranch(fullName.c_str());
      if (!branchDirectlyFromTree)
         branchDirectlyFromTree = t.FindBranch(fullName.c_str()); // try harder
      if (branchDirectlyFromTree)
         InsertBranchName(bNamesReg, bNames, std::string(branchDirectlyFromTree->GetFullName()), friendName,
                          allowDuplicates);

      if (bNamesReg.find(subBranchName) == bNamesReg.end() && t.GetBranch(subBranchName.c_str()))
         InsertBranchName(bNamesReg, bNames, subBranchName, friendName, allowDuplicates);
   }
}

void GetBranchNamesImpl(TTree &t, std::set<std::string> &bNamesReg, std::vector<std::string> &bNames,
                        std::set<TTree *> &analysedTrees, std::string &friendName, bool allowDuplicates)
{
   std::set<TLeaf *> foundLeaves;
   if (!analysedTrees.insert(&t).second) {
      return;
   }

   const auto branches = t.GetListOfBranches();
   // Getting the branches here triggered the read of the first file of the chain if t is a chain.
   // We check if a tree has been successfully read, otherwise we throw (see ROOT-9984) to avoid further
   // operations
   if (!t.GetTree()) {
      std::string err("GetBranchNames: error in opening the tree ");
      err += t.GetName();
      throw std::runtime_error(err);
   }
   if (branches) {
      for (auto b : *branches) {
         TBranch *branch = static_cast<TBranch *>(b);
         const auto branchName = std::string(branch->GetName());
         if (branch->IsA() == TBranch::Class()) {
            // Leaf list
            auto listOfLeaves = branch->GetListOfLeaves();
            if (listOfLeaves->GetEntriesUnsafe() == 1) {
               auto leaf = static_cast<TLeaf *>(listOfLeaves->UncheckedAt(0));
               InsertBranchName(bNamesReg, bNames, branchName, friendName, foundLeaves, leaf, allowDuplicates);
            }

            for (auto leaf : *listOfLeaves) {
               auto castLeaf = static_cast<TLeaf *>(leaf);
               const auto leafName = std::string(leaf->GetName());
               const auto fullName = branchName + "." + leafName;
               InsertBranchName(bNamesReg, bNames, fullName, friendName, foundLeaves, castLeaf, allowDuplicates);
            }
         } else if (branch->IsA() == TBranchObject::Class()) {
            // TBranchObject
            ExploreBranch(t, bNamesReg, bNames, branch, branchName + ".", friendName, allowDuplicates);
            InsertBranchName(bNamesReg, bNames, branchName, friendName, allowDuplicates);
         } else {
            // TBranchElement
            // Check if there is explicit or implicit dot in the name

            bool dotIsImplied = false;
            auto be = dynamic_cast<TBranchElement *>(b);
            if (!be)
               throw std::runtime_error("GetBranchNames: unsupported branch type");
            // TClonesArray (3) and STL collection (4)
            if (be->GetType() == 3 || be->GetType() == 4)
               dotIsImplied = true;

            if (dotIsImplied || branchName.back() == '.')
               ExploreBranch(t, bNamesReg, bNames, branch, "", friendName, allowDuplicates);
            else
               ExploreBranch(t, bNamesReg, bNames, branch, branchName + ".", friendName, allowDuplicates);

            InsertBranchName(bNamesReg, bNames, branchName, friendName, allowDuplicates);
         }
      }
   }

   // The list of friends needs to be accessed via GetTree()->GetListOfFriends()
   // (and not via GetListOfFriends() directly), otherwise when `t` is a TChain we
   // might not recover the list correctly (https://github.com/root-project/root/issues/6741).
   auto friendTrees = t.GetTree()->GetListOfFriends();

   if (!friendTrees)
      return;

   for (auto friendTreeObj : *friendTrees) {
      auto friendTree = ((TFriendElement *)friendTreeObj)->GetTree();

      std::string frName;
      auto alias = t.GetFriendAlias(friendTree);
      if (alias != nullptr)
         frName = std::string(alias);
      else
         frName = std::string(friendTree->GetName());

      GetBranchNamesImpl(*friendTree, bNamesReg, bNames, analysedTrees, frName, allowDuplicates);
   }
}

///////////////////////////////////////////////////////////////////////////////
/// Get all the branches names, including the ones of the friend trees
std::vector<std::string> RetrieveDatasetSchema(TTree &t, bool allowDuplicates = true)
{
   std::set<std::string> bNamesSet;
   std::vector<std::string> bNames;
   std::set<TTree *> analysedTrees;
   std::string emptyFrName = "";
   GetBranchNamesImpl(t, bNamesSet, bNames, analysedTrees, emptyFrName, allowDuplicates);
   return bNames;
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
      fBranchNamesWithDuplicates = RetrieveDatasetSchema(*fTree);
   if (fBranchNamesWithoutDuplicates.empty())
      fBranchNamesWithoutDuplicates = RetrieveDatasetSchema(*fTree, /*allowDuplicates*/ false);
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
      return RSampleInfo(id, range, nullptr, tree->GetEntries());
   } else {
      if (sampleMap.find(id) == sampleMap.end())
         throw std::runtime_error("Full sample identifier '" + id + "' cannot be found in the available samples.");
      return RSampleInfo(id, range, sampleMap.at(id), tree->GetEntries());
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

   tp->Process([this, &lm, &slotStack, &entryCount](TTreeReader &treeReader) {
      lm.TTreeThreadTask(
         treeReader, slotStack, entryCount,
         [this](std::size_t maxBulkSize, std::uint64_t &bulkBegin, std::uint64_t &rangeEnd, void *extraArg) {
            return this->PickBulkSize(maxBulkSize, bulkBegin, rangeEnd, extraArg);
         });
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

namespace {

using TypeInfoRef = std::reference_wrapper<const std::type_info>;
struct TypeInfoRefHash {
   std::size_t operator()(TypeInfoRef id) const { return id.get().hash_code(); }
};

struct TypeInfoRefEqualComp {
   bool operator()(TypeInfoRef left, TypeInfoRef right) const { return left.get() == right.get(); }
};

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
MakePrimaryColumnReader(std::string_view col, const std::type_info &ti, TTreeReader *treeReader,
                        std::size_t maxBulkSize)
{

   if (ti == typeid(void))
      return std::make_unique<ROOT::Internal::RDF::RTreeOpaqueColumnReader>(*treeReader, col, maxBulkSize);
   const auto typeName = ROOT::Internal::RDF::TypeID2TypeName(ti);

   if (auto &&[toConvert, innerTypeName, collType] = GetCollectionInfo(typeName); toConvert) {
      const std::type_info &innerTypeInfo = ROOT::Internal::RDF::TypeName2TypeID(innerTypeName);
      return std::make_unique<ROOT::Internal::RDF::RTreeGenericCollectionColumnReader>(
         *treeReader, col, innerTypeName, innerTypeInfo, collType, maxBulkSize);
   } else {
      return std::make_unique<ROOT::Internal::RDF::RTreeGenericValueColumnReader>(*treeReader, col, typeName, ti,
                                                                                  maxBulkSize);
   }
}
} // namespace

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
ROOT::Internal::RDF::RTTreeDS::CreateColumnReader(unsigned int /*slot*/, std::string_view col, const std::type_info &ti,
                                                  TTreeReader *treeReader, std::size_t maxBulkSize)
{
   // In a single thread run, use the TTreeReader data member.
   if (fTreeReader) {
      treeReader = fTreeReader.get();
   }

   // The TTreeReader might still not be available if CreateColumnReader was called before the start of the computation
   // graph execution, e.g. in AddDSColumns.
   if (!treeReader)
      return nullptr;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> colReader{
      MakePrimaryColumnReader(col, ti, treeReader, maxBulkSize)};

   if (TDirectory *treeDir = treeReader->GetTree()->GetDirectory(); treeDir) {
      using Map_t = std::unordered_map<std::string, std::pair<std::string, unsigned int>>;
      const std::string bitmaskMapName =
         std::string{"R_rdf_column_to_bitmask_mapping_"} + treeReader->GetTree()->GetName();
      if (Map_t const *columnMaskMap = treeDir->Get<Map_t>(bitmaskMapName.c_str()); columnMaskMap) {
         if (auto it = columnMaskMap->find(std::string(col)); it != columnMaskMap->end()) {
            return std::make_unique<RMaskedColumnReader>(*treeReader, std::move(colReader), it->second.first,
                                                         it->second.second, maxBulkSize);
         }
      }
   }

   return colReader;
}

ROOT::Internal::RDF::RMaskedEntryRange
ROOT::Internal::RDF::RTTreeDS::CheckFilters(unsigned int, Long64_t bulkBeginEntry, std::size_t bulkSize)
{
   return ROOT::Internal::RDF::RMaskedEntryRange{bulkSize, true, static_cast<std::uint64_t>(bulkBeginEntry)};
   // if (ROOT::IsImplicitMTEnabled())
   //    return ROOT::Internal::RDF::RMaskedEntryRange{bulkSize, true, static_cast<std::uint64_t>(bulkBeginEntry)};

   // assert(fTreeReader);
   // auto *tree = fTreeReader->GetTree();
   // assert(tree);
   // auto *entryList = tree->GetEntryList();
   // if (!entryList)
   //    return ROOT::Internal::RDF::RMaskedEntryRange{bulkSize, true, static_cast<std::uint64_t>(bulkBeginEntry)};

   // Long64_t nEntriesInList = entryList->GetN();
   // Long64_t bulkEnd = bulkBeginEntry + bulkSize;
   // auto ret = ROOT::Internal::RDF::RMaskedEntryRange{bulkSize, false, static_cast<std::uint64_t>(bulkBeginEntry)};
   // for (Long64_t idx = 0; idx < nEntriesInList; idx++) {
   //    Long64_t datasetEntryFromList = entryList->GetEntry(idx);
   //    // If we found an entry within the bulk range, we enable its reading
   //    if (datasetEntryFromList >= bulkBeginEntry && datasetEntryFromList < bulkEnd)
   //       ret[datasetEntryFromList - bulkBeginEntry] = true;
   // }
   // return ret;
}

bool ROOT::Internal::RDF::RTTreeDS::CheckValidRead(unsigned int, std::uint64_t entry)
{
   // The first entry of each tree in a chain is read in GetEntryRanges, we avoid repeating it here
   if (fTreeReader->GetCurrentEntry() != static_cast<Long64_t>(entry))
      fTreeReader->SetEntry(entry);
   return ValidRead(fTreeReader->GetEntryStatus());
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

   // auto *entryList = chain.GetEntryList();
   // if(!entryList)
   //    return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeEnd}};

   // auto localEntry = entryList->GetEntry(rangeBegin);
   // auto curEntry = rangeBegin;
   // while (localEntry < static_cast<decltype(localEntry)>(rangeEnd) && localEntry != -1) {
   //    ++curEntry;
   //    localEntry = entryList->GetEntry(curEntry);
   // }

   // return std::vector<std::pair<ULong64_t, ULong64_t>>{{rangeBegin, rangeBegin + curEntry}};
}

namespace {

bool IsReaderAtEndOfDataset(TTreeReader &treeReader, TTree &dataset,
                            const std::optional<std::pair<ULong64_t, ULong64_t>> &globalEntryRange)
{
   if (treeReader.GetCurrentEntry() >= dataset.GetEntriesFast() - 1)
      return true;

   if (globalEntryRange.has_value() &&
       (static_cast<ULong64_t>(treeReader.GetCurrentEntry()) >= globalEntryRange->first &&
        static_cast<ULong64_t>(treeReader.GetCurrentEntry()) == globalEntryRange->second - 1))
      return true;

   if (auto *entryList = treeReader.GetEntryList(); entryList && treeReader.GetCurrentEntry() >= entryList->GetN() - 1)
      return true;

   return false;
}
} // namespace

std::vector<std::pair<ULong64_t, ULong64_t>> ROOT::Internal::RDF::RTTreeDS::GetEntryRanges()
{
   assert(fTreeReader && "TTreeReader is not available, this should never happen.");
   auto treeOrChain = fTreeReader->GetTree();
   assert(treeOrChain && "Could not retrieve TTree from TTreeReader, something went wrong.");

   // End of dataset or entry range
   if (IsReaderAtEndOfDataset(*fTreeReader, *treeOrChain, fGlobalEntryRange)) {
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
   if (ROOT::IsImplicitMTEnabled())
      return;
   assert(!fTreeReader);
   fTreeReader = std::make_unique<TTreeReader>(fTree.get(), fTree->GetEntryList(), /*warnAboutLongerFriends*/ true);
   if (fGlobalEntryRange.has_value() && fGlobalEntryRange->first <= std::numeric_limits<Long64_t>::max() &&
       fGlobalEntryRange->second <= std::numeric_limits<Long64_t>::max() && fTreeReader &&
       fTreeReader->SetEntriesRange(fGlobalEntryRange->first, fGlobalEntryRange->second) != TTreeReader::kEntryValid) {
      throw std::logic_error("Something went wrong in initializing the TTreeReader.");
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

namespace {

/// Calculate how many more entries we have in this tree, taking TEntryLists into account if present
Long64_t RemainingEntriesInCurrentTree(TTreeReader &r, std::uint64_t &bulkBegin, std::uint64_t &rangeEnd)
{
   auto *entryList = r.GetEntryList();

   if (!entryList) {
      const auto treeOffset = r.GetTree()->GetChainOffset();
      const auto localEntry = bulkBegin - treeOffset;
      // auto it = r.GetTree()->GetTree()->GetClusterIterator(localEntry);
      // const auto clusterEnd = (it.Next(), it.Next()); // calling it twice to get the beginning of the _next_ cluster
      const auto curTreeEntries = r.GetTree()->GetTree()->GetEntries();
      return curTreeEntries - localEntry;
   }

   // In the MT case, TTreeProcessorMT creates one thread-local TTreeReader per task, which has already taken into
   // account the TEntryList boundaries. bulkBegin and rangeEnd will be "localised" to the valid range of this
   // TTreeReader within the boundaries of the associated TEntryList
   if (ROOT::IsImplicitMTEnabled())
      return rangeEnd - bulkBegin;

   const Long64_t nEntriesInList = entryList->GetN();
   const Long64_t curTreeEntries = r.GetTree()->GetTree()->GetEntries();
   Long64_t entryListEntriesInTree = 0u;
   Long64_t newBulkBegin = -1;
   for (Long64_t idx = 0; idx < nEntriesInList; idx++) {
      Long64_t datasetEntryFromList = entryList->GetEntry(idx);
      if (datasetEntryFromList >= static_cast<Long64_t>(bulkBegin) && datasetEntryFromList < curTreeEntries) {
         if (newBulkBegin == -1)
            newBulkBegin = idx;
         entryListEntriesInTree++;
      }
   }
   bulkBegin = newBulkBegin;
   rangeEnd = bulkBegin + entryListEntriesInTree;
   return entryListEntriesInTree;

   // // Otherwise we have a TEntryList and we must:
   // // 1. convert bulkBegin (i.e. the index in the entry list) to the actual local entry number
   // auto localEntry = entryList->GetEntry(bulkBegin);
   // // 2. find end of cluster according to TTree
   // // auto it = r.GetTree()->GetTree()->GetClusterIterator(localEntry);
   // // const auto clusterEnd = (it.Next(), it.Next()); // calling it twice to get the beginning of the _next_ cluster
   // // 3. find end of current tree
   // const auto curTreeEntries = r.GetTree()->GetTree()->GetEntries();
   // // 3. walk TEntryList to find out how many of its entries are within that cluster boundary
   // Long64_t entryListEntriesInTree = 0u;
   // while (localEntry < curTreeEntries && localEntry != -1) {
   //    ++entryListEntriesInTree;
   //    ++bulkBegin;
   //    localEntry = entryList->GetEntry(bulkBegin);
   // }

   // return entryListEntriesInTree;
}

/// Calculate how many more entries are available in each friend.
/// Return the minimum across all friends.
Long64_t RemainingEntriesInFriends(TTreeReader &r, std::uint64_t bulkBegin)
{
   Long64_t minEntriesInFriends = TTree::kMaxEntries;

   auto *t = r.GetTree();
   auto *friends = t->GetListOfFriends();
   if (!friends || friends->GetSize() == 0)
      return minEntriesInFriends;

   for (auto *friendElement : *friends) {
      auto *fr = static_cast<TFriendElement *>(friendElement)->GetTree();
      assert(fr);
      // We do not consider friends with an index
      if (fr->GetTreeIndex())
         continue;
      const Long64_t treeOffset = fr->GetChainOffset();
      const Long64_t localEntry = bulkBegin - treeOffset;
      const Long64_t entriesInSubTree = fr->GetTree()->GetEntries();
      const Long64_t remainingEntriesInSubTree = entriesInSubTree - localEntry;
      minEntriesInFriends = std::min(minEntriesInFriends, remainingEntriesInSubTree);
   }

   return minEntriesInFriends;
}

/// Select a reasonable bulk size depending on the state of the TTree event loop
/// The main heuristic is that we try to return a number that's as large as possible while staying within the same
/// TTree cluster (the TTreeCache is tree-wide (not per-branch) and we don't want it to switch back and forth as we
/// read entries for each branch across cluster boundaries).
std::size_t
PickBulkSizeForTree(TTreeReader &r, std::uint64_t &bulkBegin, std::uint64_t &rangeEnd, std::size_t maxEventsPerBulk)
{
   const Long64_t remainingEntriesInCluster = RemainingEntriesInCurrentTree(r, bulkBegin, rangeEnd);
   // we don't want to cross into a new friend tree while reading a bulk: downstream logic cannot handle it
   const Long64_t minRemainingEntriesInFriends = RemainingEntriesInFriends(r, bulkBegin);
   const Long64_t remainingEntriesInRange = rangeEnd - bulkBegin;

   auto bulkSize = std::min({static_cast<Long64_t>(maxEventsPerBulk), remainingEntriesInCluster,
                             minRemainingEntriesInFriends, remainingEntriesInRange});

   return bulkSize;
}
} // namespace

std::size_t ROOT::Internal::RDF::RTTreeDS::PickBulkSize(std::size_t maxBulkSize, std::uint64_t &bulkBegin,
                                                        std::uint64_t &rangeEnd, void *extraArg)
{
   TTreeReader *treeReader{nullptr};

   if (!ROOT::IsImplicitMTEnabled()) {
      // With IMT disabled, the TTreeReader is owned by RTTreeDS
      assert(!extraArg);
      assert(fTreeReader);
      treeReader = fTreeReader.get();
   } else {
      // With IMT enabled, we expect a TTreeReader coming from the TTreeProcessorMT
      assert(extraArg);
      treeReader = reinterpret_cast<TTreeReader *>(extraArg);
   }

   return PickBulkSizeForTree(*treeReader, bulkBegin, rangeEnd, maxBulkSize);
}
