// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h" // R__USE_IMT
#include "ROOT/RCutFlowReport.hxx"
#include "ROOT/RDFNodes.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RDataSource.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/RStringView.hxx"
#include "TTree.h"
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include <limits.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "RtypesCore.h" // Long64_t
#include "TError.h"
#include "TInterpreter.h"
#include "TROOT.h" // IsImplicitMTEnabled
#include "TTreeReader.h"

class TDirectory;
namespace ROOT {
class TSpinMutex;
} // namespace ROOT

using namespace ROOT::Detail::RDF;
using namespace ROOT::Internal::RDF;

namespace ROOT {
namespace Internal {
namespace RDF {

RActionBase::RActionBase(RLoopManager &lm, const unsigned int nSlots) : fLoopManager(lm), fNSlots(nSlots)
{
}

void RJittedAction::Run(unsigned int slot, Long64_t entry)
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->Run(slot, entry);
}

void RJittedAction::Initialize()
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->Initialize();
}

void RJittedAction::InitSlot(TTreeReader *r, unsigned int slot)
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->InitSlot(r, slot);
}

void RJittedAction::TriggerChildrenCount()
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->TriggerChildrenCount();
}

void RJittedAction::FinalizeSlot(unsigned int slot)
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->FinalizeSlot(slot);
}

void RJittedAction::Finalize()
{
   R__ASSERT(fConcreteAction != nullptr);
   fConcreteAction->Finalize();
}

void *RJittedAction::PartialUpdate(unsigned int slot)
{
   R__ASSERT(fConcreteAction != nullptr);
   return fConcreteAction->PartialUpdate(slot);
}

bool RJittedAction::HasRun() const
{
   if (fConcreteAction != nullptr) {
      return fConcreteAction->HasRun();
   } else {
      // The action has not been JITted. This means that it has not run.
      return false;
   }
}

// Some extern instaniations to speed-up compilation/interpretation time
// These are not active if c++17 is enabled because of a bug in our clang
// See ROOT-9499.
#if __cplusplus < 201703L
template class TColumnValue<int>;
template class TColumnValue<unsigned int>;
template class TColumnValue<char>;
template class TColumnValue<unsigned char>;
template class TColumnValue<float>;
template class TColumnValue<double>;
template class TColumnValue<Long64_t>;
template class TColumnValue<ULong64_t>;
template class TColumnValue<std::vector<int>>;
template class TColumnValue<std::vector<unsigned int>>;
template class TColumnValue<std::vector<char>>;
template class TColumnValue<std::vector<unsigned char>>;
template class TColumnValue<std::vector<float>>;
template class TColumnValue<std::vector<double>>;
template class TColumnValue<std::vector<Long64_t>>;
template class TColumnValue<std::vector<ULong64_t>>;
#endif
} // end NS RDF
} // end NS Internal
} // end NS ROOT

RCustomColumnBase::RCustomColumnBase(RLoopManager &lm, std::string_view name, const unsigned int nSlots,
                                     const bool isDSColumn)
   : fLoopManager(lm), fName(name), fNSlots(nSlots), fIsDataSourceColumn(isDSColumn)
{
}

// pin vtable. Work around cling JIT issue.
RCustomColumnBase::~RCustomColumnBase() = default;

std::string RCustomColumnBase::GetName() const
{
   return fName;
}

void RCustomColumnBase::InitNode()
{
   fLastCheckedEntry = std::vector<Long64_t>(fNSlots, -1);
}

void RJittedCustomColumn::InitSlot(TTreeReader *r, unsigned int slot)
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   fConcreteCustomColumn->InitSlot(r, slot);
}

void *RJittedCustomColumn::GetValuePtr(unsigned int slot)
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   return fConcreteCustomColumn->GetValuePtr(slot);
}

const std::type_info &RJittedCustomColumn::GetTypeId() const
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   return fConcreteCustomColumn->GetTypeId();
}

void RJittedCustomColumn::Update(unsigned int slot, Long64_t entry)
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   fConcreteCustomColumn->Update(slot, entry);
}

void RJittedCustomColumn::ClearValueReaders(unsigned int slot)
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   fConcreteCustomColumn->ClearValueReaders(slot);
}

void RJittedCustomColumn::InitNode()
{
   R__ASSERT(fConcreteCustomColumn != nullptr);
   fConcreteCustomColumn->InitNode();
}

RFilterBase::RFilterBase(RLoopManager &lm, std::string_view name, const unsigned int nSlots)
   : fLoopManager(lm), fLastResult(nSlots), fAccepted(nSlots), fRejected(nSlots), fName(name), fNSlots(nSlots)
{
}

RLoopManager &RFilterBase::GetLoopManager() const
{
   return fLoopManager;
}

bool RFilterBase::HasName() const
{
   return !fName.empty();
}

std::string RFilterBase::GetName() const
{
   return fName;
}

void RFilterBase::FillReport(ROOT::RDF::RCutFlowReport &rep) const
{
   if (fName.empty()) // FillReport is no-op for unnamed filters
      return;
   const auto accepted = std::accumulate(fAccepted.begin(), fAccepted.end(), 0ULL);
   const auto all = accepted + std::accumulate(fRejected.begin(), fRejected.end(), 0ULL);
   rep.AddCut({fName, accepted, all});
}

void RFilterBase::InitNode()
{
   fLastCheckedEntry = std::vector<Long64_t>(fNSlots, -1);
   if (!fName.empty()) // if this is a named filter we care about its report count
      ResetReportCount();
}

void RJittedFilter::SetFilter(std::unique_ptr<RFilterBase> f)
{
   fConcreteFilter = std::move(f);
}

void RJittedFilter::InitSlot(TTreeReader *r, unsigned int slot)
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->InitSlot(r, slot);
}

bool RJittedFilter::CheckFilters(unsigned int slot, Long64_t entry)
{
   R__ASSERT(fConcreteFilter != nullptr);
   return fConcreteFilter->CheckFilters(slot, entry);
}

void RJittedFilter::Report(ROOT::RDF::RCutFlowReport &cr) const
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->Report(cr);
}

void RJittedFilter::PartialReport(ROOT::RDF::RCutFlowReport &cr) const
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->PartialReport(cr);
}

void RJittedFilter::FillReport(ROOT::RDF::RCutFlowReport &cr) const
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->FillReport(cr);
}

void RJittedFilter::IncrChildrenCount()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->IncrChildrenCount();
}

void RJittedFilter::StopProcessing()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->StopProcessing();
}

void RJittedFilter::ResetChildrenCount()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->ResetChildrenCount();
}

void RJittedFilter::TriggerChildrenCount()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->TriggerChildrenCount();
}

void RJittedFilter::ResetReportCount()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->ResetReportCount();
}

void RJittedFilter::ClearValueReaders(unsigned int slot)
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->ClearValueReaders(slot);
}

void RJittedFilter::InitNode()
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->InitNode();
}

void RJittedFilter::AddFilterName(std::vector<std::string> &filters)
{
   if (fConcreteFilter == nullptr) {
      // No event loop performed yet, but the JITTING must be performed.
      GetLoopManager().BuildJittedNodes();
   }
   fConcreteFilter->AddFilterName(filters);
}

unsigned int &TSlotStack::GetCount()
{
   const auto tid = std::this_thread::get_id();
   {
      ROOT::TRWSpinLockReadGuard rg(fRWLock);
      auto it = fCountMap.find(tid);
      if (fCountMap.end() != it)
         return it->second;
   }

   {
      ROOT::TRWSpinLockWriteGuard rg(fRWLock);
      return (fCountMap[tid] = 0U);
   }
}
unsigned int &TSlotStack::GetIndex()
{
   const auto tid = std::this_thread::get_id();

   {
      ROOT::TRWSpinLockReadGuard rg(fRWLock);
      if (fIndexMap.end() != fIndexMap.find(tid))
         return fIndexMap[tid];
   }

   {
      ROOT::TRWSpinLockWriteGuard rg(fRWLock);
      return (fIndexMap[tid] = std::numeric_limits<unsigned int>::max());
   }
}

void TSlotStack::ReturnSlot(unsigned int slotNumber)
{
   auto &index = GetIndex();
   auto &count = GetCount();
   R__ASSERT(count > 0U && "TSlotStack has a reference count relative to an index which will become negative.");
   count--;
   if (0U == count) {
      index = std::numeric_limits<unsigned int>::max();
      ROOT::TRWSpinLockWriteGuard guard(fRWLock);
      fBuf[fCursor++] = slotNumber;
      R__ASSERT(fCursor <= fBuf.size() &&
                "TSlotStack assumes that at most a fixed number of values can be present in the "
                "stack. fCursor is greater than the size of the internal buffer. This violates "
                "such assumption.");
   }
}

unsigned int TSlotStack::GetSlot()
{
   auto &index = GetIndex();
   auto &count = GetCount();
   count++;
   if (std::numeric_limits<unsigned int>::max() != index)
      return index;
   ROOT::TRWSpinLockWriteGuard guard(fRWLock);
   R__ASSERT(fCursor > 0 &&
             "TSlotStack assumes that a value can be always obtained. In this case fCursor is <=0 and this "
             "violates such assumption.");
   index = fBuf[--fCursor];
   return index;
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
   TSlotStack slotStack(fNSlots);
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
}

/// Run event loop over one or multiple ROOT files, in parallel.
void RLoopManager::RunTreeProcessorMT()
{
#ifdef R__USE_IMT
   TSlotStack slotStack(fNSlots);
   auto tp = std::make_unique<ROOT::TTreeProcessorMT>(*fTree);

   tp->Process([this, &slotStack](TTreeReader &r) -> void {
      auto slot = slotStack.GetSlot();
      InitNodeSlots(&r, slot);
      // recursive call to check filters and conditionally execute actions
      while (r.Next()) {
         RunAndCheckFilters(slot, r.GetCurrentEntry());
      }
      CleanUpTask(slot);
      slotStack.ReturnSlot(slot);
   });
#endif // no-op otherwise (will not be called)
}

/// Run event loop over one or multiple ROOT files, in sequence.
void RLoopManager::RunTreeReader()
{
   TTreeReader r(fTree.get());
   if (0 == fTree->GetEntriesFast())
      return;
   InitNodeSlots(&r, 0);

   // recursive call to check filters and conditionally execute actions
   // in the non-MT case processing can be stopped early by ranges, hence the check on fNStopsReceived
   while (r.Next() && fNStopsReceived < fNChildren) {
      RunAndCheckFilters(0, r.GetCurrentEntry());
   }
   fTree->GetEntry(0);
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
   TSlotStack slotStack(fNSlots);
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
   // booked branches must be initialized first because other nodes might need to point to the values they encapsulate
   for (auto &bookedBranch : fBookedCustomColumns)
      bookedBranch.second->InitSlot(r, slot);
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
   for (auto &filter : fBookedFilters)
      filter->InitNode();
   for (auto &customColumn : fBookedCustomColumns)
      customColumn.second->InitNode();
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
      ptr->ClearValueReaders(slot);
   for (auto &pair : fBookedCustomColumns)
      pair.second->ClearValueReaders(slot);
}

/// Jit all actions that required runtime column type inference, and clean the `fToJit` member variable.
void RLoopManager::BuildJittedNodes()
{
   auto error = TInterpreter::EErrorCode::kNoError;
   gInterpreter->Calc(fToJit.c_str(), &error);
   if (TInterpreter::EErrorCode::kNoError != error) {
      std::string exceptionText =
         "An error occurred while jitting. The lines above might indicate the cause of the crash\n";
      throw std::runtime_error(exceptionText.c_str());
   }
   fToJit.clear();
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

unsigned int RLoopManager::GetNextID() const
{
   static unsigned int id = 0;
   ++id;
   return id;
}

/// Start the event loop with a different mechanism depending on IMT/no IMT, data source/no data source.
/// Also perform a few setup and clean-up operations (jit actions if necessary, clear booked actions after the loop...).
void RLoopManager::Run()
{
   if (!fToJit.empty())
      BuildJittedNodes();

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

RLoopManager &RLoopManager::GetLoopManager()
{
   return *this;
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

void RLoopManager::Book(const RCustomColumnBasePtr_t &columnPtr)
{
   const auto &name = columnPtr->GetName();
   fBookedCustomColumns[name] = columnPtr;
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
bool RLoopManager::CheckFilters(int, unsigned int)
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

std::vector<RDFInternal::RActionBase *> RLoopManager::GetAllActions(){
   std::vector<RDFInternal::RActionBase *> actions;
   actions.insert(actions.begin(), fBookedActions.begin(), fBookedActions.end());
   actions.insert(actions.begin(), fRunActions.begin(), fRunActions.end());
   return actions;
}

RRangeBase::RRangeBase(RLoopManager &lm, unsigned int start, unsigned int stop, unsigned int stride,
                       const unsigned int nSlots)
   : fLoopManager(lm), fStart(start), fStop(stop), fStride(stride), fNSlots(nSlots)
{
}

RLoopManager &RRangeBase::GetLoopManager() const
{
   return fLoopManager;
}

void RRangeBase::ResetCounters()
{
   fLastCheckedEntry = -1;
   fNProcessedEntries = 0;
   fHasStopped = false;
}
