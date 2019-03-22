#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RCustomColumnBase.hxx"
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RRangeBase.hxx"
#include "ROOT/RDF/RSlotStack.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#include "RtypesCore.h" // Long64_t
#include "TBranchElement.h"
#include "TBranchObject.h"
#include "TEntryList.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TROOT.h" // IsImplicitMTEnabled
#include "TTreeReader.h"

#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ROOT::Detail::RDF;
using namespace ROOT::Internal::RDF;

bool ContainsLeaf(const std::set<TLeaf *> &leaves, TLeaf *leaf)
{
   return (leaves.find(leaf) != leaves.end());
}

///////////////////////////////////////////////////////////////////////////////
/// This overload does not perform any check on the duplicates.
/// It is used for TBranch objects.
void UpdateList(std::set<std::string> &bNamesReg, ColumnNames_t &bNames, const std::string &branchName,
                const std::string &friendName)
{

   if (!friendName.empty()) {
      // In case of a friend tree, users might prepend its name/alias to the branch names
      const auto friendBName = friendName + "." + branchName;
      if (bNamesReg.insert(friendBName).second)
         bNames.push_back(friendBName);
   }

   if (bNamesReg.insert(branchName).second)
      bNames.push_back(branchName);
}

///////////////////////////////////////////////////////////////////////////////
/// This overloads makes sure that the TLeaf has not been already inserted.
void UpdateList(std::set<std::string> &bNamesReg, ColumnNames_t &bNames, const std::string &branchName,
                const std::string &friendName, std::set<TLeaf *> &foundLeaves, TLeaf *leaf, bool allowDuplicates)
{
   const bool canAdd = allowDuplicates ? true : !ContainsLeaf(foundLeaves, leaf);
   if (!canAdd) {
      return;
   }

   UpdateList(bNamesReg, bNames, branchName, friendName);

   foundLeaves.insert(leaf);
}

void ExploreBranch(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames, TBranch *b, std::string prefix,
                   std::string &friendName)
{
   for (auto sb : *b->GetListOfBranches()) {
      TBranch *subBranch = static_cast<TBranch *>(sb);
      auto subBranchName = std::string(subBranch->GetName());
      auto fullName = prefix + subBranchName;

      std::string newPrefix;
      if (!prefix.empty())
         newPrefix = fullName + ".";

      ExploreBranch(t, bNamesReg, bNames, subBranch, newPrefix, friendName);

      if (t.GetBranch(fullName.c_str())) {
         UpdateList(bNamesReg, bNames, fullName, friendName);

      } else if (t.GetBranch(subBranchName.c_str())) {
         UpdateList(bNamesReg, bNames, subBranchName, friendName);
      }
   }
}

void GetBranchNamesImpl(TTree &t, std::set<std::string> &bNamesReg, ColumnNames_t &bNames,
                        std::set<TTree *> &analysedTrees, std::string &friendName, bool allowDuplicates)
{
   std::set<TLeaf *> foundLeaves;
   if (!analysedTrees.insert(&t).second) {
      return;
   }

   const auto branches = t.GetListOfBranches();
   if (branches) {
      for (auto b : *branches) {
         TBranch *branch = static_cast<TBranch *>(b);
         const auto branchName = std::string(branch->GetName());
         if (branch->IsA() == TBranch::Class()) {
            // Leaf list
            auto listOfLeaves = branch->GetListOfLeaves();
            if (listOfLeaves->GetEntries() == 1) {
               auto leaf = static_cast<TLeaf *>(listOfLeaves->At(0));
               const auto leafName = std::string(leaf->GetName());
               if (leafName == branchName) {
                  UpdateList(bNamesReg, bNames, branchName, friendName, foundLeaves, leaf, allowDuplicates);
               }
            }

            for (auto leaf : *listOfLeaves) {
               auto castLeaf = static_cast<TLeaf *>(leaf);
               const auto leafName = std::string(leaf->GetName());
               const auto fullName = branchName + "." + leafName;
               UpdateList(bNamesReg, bNames, fullName, friendName, foundLeaves, castLeaf, allowDuplicates);
            }
         } else if (branch->IsA() == TBranchObject::Class()) {
            // TBranchObject
            ExploreBranch(t, bNamesReg, bNames, branch, branchName + ".", friendName);
            UpdateList(bNamesReg, bNames, branchName, friendName);
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
               ExploreBranch(t, bNamesReg, bNames, branch, "", friendName);
            else
               ExploreBranch(t, bNamesReg, bNames, branch, branchName + ".", friendName);

            UpdateList(bNamesReg, bNames, branchName, friendName);
         }
      }
   }

   auto friendTrees = t.GetListOfFriends();

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
ColumnNames_t ROOT::Internal::RDF::GetBranchNames(TTree &t, bool allowDuplicates)
{
   std::set<std::string> bNamesSet;
   ColumnNames_t bNames;
   std::set<TTree *> analysedTrees;
   std::string emptyFrName = "";
   GetBranchNamesImpl(t, bNamesSet, bNames, analysedTrees, emptyFrName, allowDuplicates);
   return bNames;
}


RLoopManager::RLoopManager(TTree *tree, const ColumnNames_t &defaultBranches)
   : fTree(std::shared_ptr<TTree>(tree, [](TTree *) {})), fDefaultColumns(defaultBranches),
     fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kROOTFilesMT : ELoopType::kROOTFiles)
{
}

RLoopManager::RLoopManager(ULong64_t nEmptyEntries)
   : fNEmptyEntries(nEmptyEntries), fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kNoFilesMT : ELoopType::kNoFiles)
{
}

RLoopManager::RLoopManager(std::unique_ptr<RDataSource> ds, const ColumnNames_t &defaultBranches)
   : fDefaultColumns(defaultBranches), fNSlots(RDFInternal::GetNSlots()),
     fLoopType(ROOT::IsImplicitMTEnabled() ? ELoopType::kDataSourceMT : ELoopType::kDataSource),
     fDataSource(std::move(ds))
{
   fDataSource->SetNSlots(fNSlots);
}

/// Run event loop with no source files, in parallel.
void RLoopManager::RunEmptySourceMT()
{
#ifdef R__USE_IMT
   RSlotStack slotStack(fNSlots);
   // Working with an empty tree.
   // Evenly partition the entries according to fNSlots. Produce around 2 tasks per slot.
   const auto nEntriesPerSlot = fNEmptyEntries / (fNSlots * 2);
   auto remainder = fNEmptyEntries % (fNSlots * 2);
   std::vector<std::pair<ULong64_t, ULong64_t>> entryRanges;
   ULong64_t start = 0;
   while (start < fNEmptyEntries) {
      ULong64_t end = start + nEntriesPerSlot;
      if (remainder > 0) {
         ++end;
         --remainder;
      }
      entryRanges.emplace_back(start, end);
      start = end;
   }

   // Each task will generate a subrange of entries
   auto genFunction = [this, &slotStack](const std::pair<ULong64_t, ULong64_t> &range) {
      auto slot = slotStack.GetSlot();
      InitNodeSlots(nullptr, slot);
      for (auto currEntry = range.first; currEntry < range.second; ++currEntry) {
         RunAndCheckFilters(slot, currEntry);
      }
      CleanUpTask(slot);
      slotStack.ReturnSlot(slot);
   };

   ROOT::TThreadExecutor pool;
   pool.Foreach(genFunction, entryRanges);

#endif // not implemented otherwise
}

/// Run event loop with no source files, in sequence.
void RLoopManager::RunEmptySource()
{
   InitNodeSlots(nullptr, 0);
   for (ULong64_t currEntry = 0; currEntry < fNEmptyEntries && fNStopsReceived < fNChildren; ++currEntry) {
      RunAndCheckFilters(0, currEntry);
   }
   CleanUpTask(0u);
}

/// Run event loop over one or multiple ROOT files, in parallel.
void RLoopManager::RunTreeProcessorMT()
{
#ifdef R__USE_IMT
   RSlotStack slotStack(fNSlots);
   const auto &entryList = fTree->GetEntryList() ? *fTree->GetEntryList() : TEntryList();
   auto tp = std::make_unique<ROOT::TTreeProcessorMT>(*fTree, entryList);

   std::atomic<ULong64_t> entryCount(0ull);

   tp->Process([this, &slotStack, &entryCount](TTreeReader &r) -> void {
      auto slot = slotStack.GetSlot();
      InitNodeSlots(&r, slot);
      const auto entryRange = r.GetEntriesRange(); // we trust TTreeProcessorMT to call SetEntriesRange
      const auto nEntries = entryRange.second - entryRange.first;
      auto count = entryCount.fetch_add(nEntries);
      // recursive call to check filters and conditionally execute actions
      while (r.Next()) {
         RunAndCheckFilters(slot, count++);
      }
      CleanUpTask(slot);
      slotStack.ReturnSlot(slot);
   });
#endif // no-op otherwise (will not be called)
}

/// Run event loop over one or multiple ROOT files, in sequence.
void RLoopManager::RunTreeReader()
{
   TTreeReader r(fTree.get(), fTree->GetEntryList());
   if (0 == fTree->GetEntriesFast())
      return;
   InitNodeSlots(&r, 0);

   // recursive call to check filters and conditionally execute actions
   // in the non-MT case processing can be stopped early by ranges, hence the check on fNStopsReceived
   while (r.Next() && fNStopsReceived < fNChildren) {
      RunAndCheckFilters(0, r.GetCurrentEntry());
   }
   CleanUpTask(0u);
}

/// Run event loop over data accessed through a DataSource, in sequence.
void RLoopManager::RunDataSource()
{
   R__ASSERT(fDataSource != nullptr);
   fDataSource->Initialise();
   auto ranges = fDataSource->GetEntryRanges();
   while (!ranges.empty()) {
      InitNodeSlots(nullptr, 0u);
      fDataSource->InitSlot(0u, 0ull);
      for (const auto &range : ranges) {
         auto end = range.second;
         for (auto entry = range.first; entry < end; ++entry) {
            if (fDataSource->SetEntry(0u, entry)) {
               RunAndCheckFilters(0u, entry);
            }
         }
      }
      CleanUpTask(0u);
      fDataSource->FinaliseSlot(0u);
      ranges = fDataSource->GetEntryRanges();
   }
   fDataSource->Finalise();
}

/// Run event loop over data accessed through a DataSource, in parallel.
void RLoopManager::RunDataSourceMT()
{
#ifdef R__USE_IMT
   R__ASSERT(fDataSource != nullptr);
   RSlotStack slotStack(fNSlots);
   ROOT::TThreadExecutor pool;

   // Each task works on a subrange of entries
   auto runOnRange = [this, &slotStack](const std::pair<ULong64_t, ULong64_t> &range) {
      const auto slot = slotStack.GetSlot();
      InitNodeSlots(nullptr, slot);
      fDataSource->InitSlot(slot, range.first);
      const auto end = range.second;
      for (auto entry = range.first; entry < end; ++entry) {
         if (fDataSource->SetEntry(slot, entry)) {
            RunAndCheckFilters(slot, entry);
         }
      }
      CleanUpTask(slot);
      fDataSource->FinaliseSlot(slot);
      slotStack.ReturnSlot(slot);
   };

   fDataSource->Initialise();
   auto ranges = fDataSource->GetEntryRanges();
   while (!ranges.empty()) {
      pool.Foreach(runOnRange, ranges);
      ranges = fDataSource->GetEntryRanges();
   }
   fDataSource->Finalise();
#endif // not implemented otherwise (never called)
}

/// Execute actions and make sure named filters are called for each event.
/// Named filters must be called even if the analysis logic would not require it, lest they report confusing results.
void RLoopManager::RunAndCheckFilters(unsigned int slot, Long64_t entry)
{
   for (auto &actionPtr : fBookedActions)
      actionPtr->Run(slot, entry);
   for (auto &namedFilterPtr : fBookedNamedFilters)
      namedFilterPtr->CheckFilters(slot, entry);
   for (auto &callback : fCallbacks)
      callback(slot);
}

/// Build TTreeReaderValues for all nodes
/// This method loops over all filters, actions and other booked objects and
/// calls their `InitRDFValues` methods. It is called once per node per slot, before
/// running the event loop. It also informs each node of the TTreeReader that
/// a particular slot will be using.
void RLoopManager::InitNodeSlots(TTreeReader *r, unsigned int slot)
{
   for (auto &ptr : fBookedActions)
      ptr->InitSlot(r, slot);
   for (auto &ptr : fBookedFilters)
      ptr->InitSlot(r, slot);
   for (auto &callback : fCallbacksOnce)
      callback(slot);
}

/// Initialize all nodes of the functional graph before running the event loop.
/// This method is called once per event-loop and performs generic initialization
/// operations that do not depend on the specific processing slot (i.e. operations
/// that are common for all threads).
void RLoopManager::InitNodes()
{
   EvalChildrenCounts();
   for (auto column : fCustomColumns)
      column->InitNode();
   for (auto &filter : fBookedFilters)
      filter->InitNode();
   for (auto &range : fBookedRanges)
      range->InitNode();
   for (auto &ptr : fBookedActions)
      ptr->Initialize();
}

/// Perform clean-up operations. To be called at the end of each event loop.
void RLoopManager::CleanUpNodes()
{
   fMustRunNamedFilters = false;

   // forget RActions and detach TResultProxies
   for (auto &ptr : fBookedActions)
      ptr->Finalize();

   fRunActions.insert(fRunActions.begin(), fBookedActions.begin(), fBookedActions.end());
   fBookedActions.clear();

   // reset children counts
   fNChildren = 0;
   fNStopsReceived = 0;
   for (auto &ptr : fBookedFilters)
      ptr->ResetChildrenCount();
   for (auto &ptr : fBookedRanges)
      ptr->ResetChildrenCount();

   fCallbacks.clear();
   fCallbacksOnce.clear();
}

/// Perform clean-up operations. To be called at the end of each task execution.
void RLoopManager::CleanUpTask(unsigned int slot)
{
   for (auto &ptr : fBookedActions)
      ptr->FinalizeSlot(slot);
   for (auto &ptr : fBookedFilters)
      ptr->ClearTask(slot);
}

/// Declare to the interpreter type aliases and other entities required by RDF jitted nodes.
/// This method clears the `fToJitDeclare` member variable.
void RLoopManager::JitDeclarations()
{
   gInterpreter->Declare(fToJitDeclare.c_str());
   fToJitDeclare.clear();
}

/// Add RDF nodes that require just-in-time compilation to the computation graph.
/// This method also invokes JitDeclarations() if needed, and clears the `fToJitExec` member variable.
void RLoopManager::Jit()
{
   JitDeclarations();

   auto error = TInterpreter::EErrorCode::kNoError;
   gInterpreter->Calc(fToJitExec.c_str(), &error);
   if (TInterpreter::EErrorCode::kNoError != error) {
      std::string exceptionText =
         "An error occurred while jitting. The lines above might indicate the cause of the crash\n";
      throw std::runtime_error(exceptionText.c_str());
   }
   fToJitExec.clear();
}

/// Trigger counting of number of children nodes for each node of the functional graph.
/// This is done once before starting the event loop. Each action sends an `increase children count` signal
/// upstream, which is propagated until RLoopManager. Each time a node receives the signal, in increments its
/// children counter. Each node only propagates the signal once, even if it receives it multiple times.
/// Named filters also send an `increase children count` signal, just like actions, as they always execute during
/// the event loop so the graph branch they belong to must count as active even if it does not end in an action.
void RLoopManager::EvalChildrenCounts()
{
   for (auto &actionPtr : fBookedActions)
      actionPtr->TriggerChildrenCount();
   for (auto &namedFilterPtr : fBookedNamedFilters)
      namedFilterPtr->TriggerChildrenCount();
}

unsigned int RLoopManager::GetNextID()
{
   static unsigned int id = 0;
   ++id;
   return id;
}

/// Start the event loop with a different mechanism depending on IMT/no IMT, data source/no data source.
/// Also perform a few setup and clean-up operations (jit actions if necessary, clear booked actions after the loop...).
void RLoopManager::Run()
{
   if (!fToJitExec.empty())
      Jit();

   InitNodes();

   switch (fLoopType) {
   case ELoopType::kNoFilesMT: RunEmptySourceMT(); break;
   case ELoopType::kROOTFilesMT: RunTreeProcessorMT(); break;
   case ELoopType::kDataSourceMT: RunDataSourceMT(); break;
   case ELoopType::kNoFiles: RunEmptySource(); break;
   case ELoopType::kROOTFiles: RunTreeReader(); break;
   case ELoopType::kDataSource: RunDataSource(); break;
   }

   CleanUpNodes();
}

/// Return the list of default columns -- empty if none was provided when constructing the RDataFrame
const ColumnNames_t &RLoopManager::GetDefaultColumnNames() const
{
   return fDefaultColumns;
}

TTree *RLoopManager::GetTree() const
{
   return fTree.get();
}

void RLoopManager::Book(RDFInternal::RActionBase *actionPtr)
{
   fBookedActions.emplace_back(actionPtr);
}

void RLoopManager::Deregister(RDFInternal::RActionBase *actionPtr)
{
   RDFInternal::Erase(actionPtr, fRunActions);
   RDFInternal::Erase(actionPtr, fBookedActions);
}

void RLoopManager::Book(RFilterBase *filterPtr)
{
   fBookedFilters.emplace_back(filterPtr);
   if (filterPtr->HasName()) {
      fBookedNamedFilters.emplace_back(filterPtr);
      fMustRunNamedFilters = true;
   }
}

void RLoopManager::Deregister(RFilterBase *filterPtr)
{
   RDFInternal::Erase(filterPtr, fBookedFilters);
   RDFInternal::Erase(filterPtr, fBookedNamedFilters);
}

void RLoopManager::Book(RRangeBase *rangePtr)
{
   fBookedRanges.emplace_back(rangePtr);
}

void RLoopManager::Deregister(RRangeBase *rangePtr)
{
   RDFInternal::Erase(rangePtr, fBookedRanges);
}

// dummy call, end of recursive chain of calls
bool RLoopManager::CheckFilters(unsigned int, Long64_t)
{
   return true;
}

/// Call `FillReport` on all booked filters
void RLoopManager::Report(ROOT::RDF::RCutFlowReport &rep) const
{
   for (const auto &fPtr : fBookedNamedFilters)
      fPtr->FillReport(rep);
}

void RLoopManager::RegisterCallback(ULong64_t everyNEvents, std::function<void(unsigned int)> &&f)
{
   if (everyNEvents == 0ull)
      fCallbacksOnce.emplace_back(std::move(f), fNSlots);
   else
      fCallbacks.emplace_back(everyNEvents, std::move(f), fNSlots);
}

std::vector<std::string> RLoopManager::GetFiltersNames()
{
   std::vector<std::string> filters;
   for (auto &filter : fBookedFilters) {
      auto name = (filter->HasName() ? filter->GetName() : "Unnamed Filter");
      filters.push_back(name);
   }
   return filters;
}

std::vector<RDFInternal::RActionBase *> RLoopManager::GetAllActions()
{
   std::vector<RDFInternal::RActionBase *> actions;
   actions.insert(actions.begin(), fBookedActions.begin(), fBookedActions.end());
   actions.insert(actions.begin(), fRunActions.begin(), fRunActions.end());
   return actions;
}

std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> RLoopManager::GetGraph()
{
   std::string name;
   if (fDataSource) {
      name = fDataSource->GetLabel();
   } else if (fTree) {
      name = fTree->GetName();
   } else {
      name = std::to_string(fNEmptyEntries);
   }

   auto thisNode = std::make_shared<ROOT::Internal::RDF::GraphDrawing::GraphNode>(name);
   thisNode->SetRoot();
   thisNode->SetCounter(0);
   return thisNode;
}

////////////////////////////////////////////////////////////////////////////
/// Return all valid TTree::Branch names (caching results for subsequent calls).
/// Never use fBranchNames directy, always request it through this method.
const ColumnNames_t &RLoopManager::GetBranchNames()
{
   if (fValidBranchNames.empty() && fTree) {
      fValidBranchNames = RDFInternal::GetBranchNames(*fTree, /*allowRepetitions=*/true);
   }
   return fValidBranchNames;
}
