#include <ROOT/RDataFrame.hxx>
#include <ROOT/RTTreeDS.hxx>
#include <ROOT/RFriendInfo.hxx>
#include <ROOT/RDF/RLoopManager.hxx>

#ifdef R__USE_IMT
#include <TROOT.h>
#include <TEntryList.h>
#include <ROOT/TTreeProcessorMT.hxx>
#include <ROOT/RSlotStack.hxx>
#endif

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

#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled())
      return;
#endif
   fTreeReader = std::make_unique<TTreeReader>(fTree.get(), fTree->GetEntryList(), /*warnAboutLongerFriends*/ true);
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
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::string_view treeName, std::string_view fileNameGlob)
{
   std::string treeNameInt{treeName};
   std::string fileNameGlobInt{fileNameGlob};
   auto chain = ROOT::Internal::TreeUtils::MakeChainForMT(treeNameInt.c_str());
   chain->Add(fileNameGlobInt.c_str());

   Setup(std::move(chain));
}

ROOT::Internal::RDF::RTTreeDS::RTTreeDS(std::string_view treeName, const std::vector<std::string> &fileNameGlobs)
{
   std::string treeNameInt(treeName);
   auto chain = ROOT::Internal::TreeUtils::MakeChainForMT(treeNameInt);
   for (auto &&f : fileNameGlobs)
      chain->Add(f.c_str());

   Setup(std::move(chain));
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
   const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &sampleMap) const
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
   auto &&suppErrs = lm.GetSuppressErrorsForMissingBranches();
   auto &&suppressErrorsForMissingBranches = std::vector<std::string>(suppErrs.begin(), suppErrs.end());
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
