// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h" // R__USE_IMT
#include "ROOT/TDFNodes.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#ifdef R__USE_IMT
#include "ROOT/TThreadExecutor.hxx"
#endif
#include "RtypesCore.h" // Long64_t
#include "TROOT.h"      // IsImplicitMTEnabled
#include "TTreeReader.h"

#include <cassert>
#include <mutex>
#include <numeric> // std::accumulate
#include <string>
class TDirectory;
class TTree;
using namespace ROOT::Detail::TDF;

namespace ROOT {
namespace Internal {
namespace TDF {

TActionBase::TActionBase(ROOT::Detail::TDF::TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches)
   : fImplPtr(implPtr), fTmpBranches(tmpBranches)
{
}

} // end NS TDF
} // end NS Internal
} // end NS ROOT

TCustomColumnBase::TCustomColumnBase(TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches, const std::string &name)
   : fImplPtr(implPtr), fTmpBranches(tmpBranches), fName(name){};

BranchNames_t TCustomColumnBase::GetTmpBranches() const
{
   return fTmpBranches;
}

std::string TCustomColumnBase::GetName() const
{
   return fName;
}

TDataFrameImpl *TCustomColumnBase::GetImplPtr() const
{
   return fImplPtr;
}

TFilterBase::TFilterBase(TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches, const std::string &name)
   : fImplPtr(implPtr), fTmpBranches(tmpBranches), fName(name){};

TDataFrameImpl *TFilterBase::GetImplPtr() const
{
   return fImplPtr;
}

BranchNames_t TFilterBase::GetTmpBranches() const
{
   return fTmpBranches;
}

bool TFilterBase::HasName() const
{
   return !fName.empty();
};

void TFilterBase::PrintReport() const
{
   if (fName.empty()) // PrintReport is no-op for unnamed filters
      return;
   const auto accepted = std::accumulate(fAccepted.begin(), fAccepted.end(), 0ULL);
   const auto all = accepted + std::accumulate(fRejected.begin(), fRejected.end(), 0ULL);
   double perc = accepted;
   if (all > 0) perc /= all;
   perc *= 100.;
   Printf("%-10s: pass=%-10lld all=%-10lld -- %8.3f %%", fName.c_str(), accepted, all, perc);
}

// This is an helper class to allow to pick a slot without resorting to a map
// indexed by thread ids.
// WARNING: this class does not work as a regular stack. The size is
// fixed at construction time and no blocking is foreseen.
class TSlotStack {
private:
   unsigned int fCursor;
   std::vector<unsigned int> fBuf;
   ROOT::TSpinMutex fMutex;

public:
   TSlotStack() = delete;
   TSlotStack(unsigned int size) : fCursor(size), fBuf(size) { std::iota(fBuf.begin(), fBuf.end(), 0U); }
   void Push(unsigned int slotNumber);
   unsigned int Pop();
};

void TSlotStack::Push(unsigned int slotNumber)
{
   std::lock_guard<ROOT::TSpinMutex> guard(fMutex);
   fBuf[fCursor++] = slotNumber;
   assert(fCursor <= fBuf.size() && "TSlotStack assumes that at most a fixed number of values can be present in the "
                                    "stack. fCursor is greater than the size of the internal buffer. This violates "
                                    "such assumption.");
}

unsigned int TSlotStack::Pop()
{
   assert(fCursor > 0 &&
          "TSlotStack assumes that a value can be always popped. fCursor is <=0 and this violates such assumption.");
   std::lock_guard<ROOT::TSpinMutex> guard(fMutex);
   return fBuf[--fCursor];
}

TDataFrameImpl::TDataFrameImpl(TTree *tree, const BranchNames_t &defaultBranches)
   : fTree(std::shared_ptr<TTree>(tree, [](TTree *) {})), fDefaultBranches(defaultBranches),
     fNSlots(ROOT::Internal::TDF::GetNSlots())
{
}

TDataFrameImpl::TDataFrameImpl(Long64_t nEmptyEntries)
   : fNEmptyEntries(nEmptyEntries), fNSlots(ROOT::Internal::TDF::GetNSlots())
{
}

void TDataFrameImpl::RunAndCheckFilters(unsigned int slot, Long64_t entry)
{
   for (auto &actionPtr : fBookedActions) actionPtr->Run(slot, entry);
   for (auto &namedFilterPtr : fBookedNamedFilters) namedFilterPtr->CheckFilters(slot, entry);
}

void TDataFrameImpl::Run()
{
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) {
      TSlotStack slotStack(fNSlots);
      CreateSlots(fNSlots);

      if (fNEmptyEntries > 0) {
         // Working with an empty tree.
         // Evenly partition the entries according to fNSlots
         const auto nEntriesPerSlot = fNEmptyEntries / fNSlots;
         auto remainder = fNEmptyEntries % fNSlots;
         std::vector<std::pair<Long64_t, Long64_t>> entryRanges;
         Long64_t start = 0;
         while (start < fNEmptyEntries) {
            Long64_t end = start + nEntriesPerSlot;
            if (remainder > 0) {
               ++end;
               --remainder;
            }
            entryRanges.emplace_back(start, end);
            start = end;
         }

         // Each task will generate a subrange of entries
         auto genFunction = [this, &slotStack](const std::pair<Long64_t, Long64_t> &range) {
            auto slot = slotStack.Pop();
            BuildAllReaderValues(nullptr, slot);
            for (auto currEntry = range.first; currEntry < range.second; ++currEntry) {
               RunAndCheckFilters(slot, currEntry);
            }
            slotStack.Push(slot);
         };

         ROOT::TThreadExecutor pool;
         pool.Foreach(genFunction, entryRanges);
      } else {
         using ttpmt_t = ROOT::TTreeProcessorMT;
         std::unique_ptr<ttpmt_t> tp;
         tp.reset(new ttpmt_t(*fTree));

         tp->Process([this, &slotStack](TTreeReader &r) -> void {
            auto slot = slotStack.Pop();
            BuildAllReaderValues(&r, slot);
            // recursive call to check filters and conditionally execute actions
            while (r.Next()) {
               RunAndCheckFilters(slot, r.GetCurrentEntry());
            }
            slotStack.Push(slot);
         });
      }
   } else {
#endif // R__USE_IMT
      CreateSlots(1);
      if (fNEmptyEntries > 0) {
         BuildAllReaderValues(nullptr, 0);
         for (Long64_t currEntry = 0; currEntry < fNEmptyEntries && fNStopsReceived < fNChildren; ++currEntry) {
            RunAndCheckFilters(0, currEntry);
         }
      } else {
         TTreeReader r(fTree.get());
         BuildAllReaderValues(&r, 0);

         // recursive call to check filters and conditionally execute actions
         // in the non-MT case processing can be stopped early by ranges, hence the check on fNStopsReceived
         while (r.Next() && fNStopsReceived < fNChildren) {
            RunAndCheckFilters(0, r.GetCurrentEntry());
         }
      }
#ifdef R__USE_IMT
   }
#endif // R__USE_IMT

   fHasRunAtLeastOnce = true;
   // forget actions
   fBookedActions.clear();
   // make all TResultProxies ready
   for (auto readiness : fResProxyReadiness) {
      *readiness.get() = true;
   }
   // forget TResultProxies
   fResProxyReadiness.clear();
}

/// Build TTreeReaderValues for all nodes
///
/// This method loops over all filters, actions and other booked objects and
/// calls their `BuildReaderValues` methods. It is called once per node per slot, before
/// running the event loop. It also informs each node of the TTreeReader that
/// a particular slot will be using.
void TDataFrameImpl::BuildAllReaderValues(TTreeReader *r, unsigned int slot)
{
   // booked branches must be initialized first
   // because actions and filters might need to point to the values encapsulate
   for (auto &bookedBranch : fBookedBranches) bookedBranch.second->BuildReaderValues(r, slot);
   for (auto &ptr : fBookedActions) ptr->BuildReaderValues(r, slot);
   for (auto &ptr : fBookedFilters) ptr->BuildReaderValues(r, slot);
}

/// Initialize all nodes of the functional graph before running the event loop
///
/// This method loops over all filters, actions and other booked objects and
/// calls their `CreateSlots` methods. It is called once per node before running the
/// event loop. The main effect is to inform all nodes of the number of slots
/// (i.e. workers) that will be used to perform the event loop.
void TDataFrameImpl::CreateSlots(unsigned int nSlots)
{
   for (auto &ptr : fBookedActions) ptr->CreateSlots(nSlots);
   for (auto &ptr : fBookedFilters) ptr->CreateSlots(nSlots);
   for (auto &bookedBranch : fBookedBranches) bookedBranch.second->CreateSlots(nSlots);
}

TDataFrameImpl *TDataFrameImpl::GetImplPtr()
{
   return this;
}

const BranchNames_t &TDataFrameImpl::GetDefaultBranches() const
{
   return fDefaultBranches;
}

TTree *TDataFrameImpl::GetTree() const
{
   return fTree.get();
}

TCustomColumnBase *TDataFrameImpl::GetBookedBranch(const std::string &name) const
{
   auto it = fBookedBranches.find(name);
   return it == fBookedBranches.end() ? nullptr : it->second.get();
}

TDirectory *TDataFrameImpl::GetDirectory() const
{
   return fDirPtr;
}

std::string TDataFrameImpl::GetTreeName() const
{
   return fTree->GetName();
}

void TDataFrameImpl::Book(const ActionBasePtr_t &actionPtr)
{
   fBookedActions.emplace_back(actionPtr);
}

void TDataFrameImpl::Book(const FilterBasePtr_t &filterPtr)
{
   fBookedFilters.emplace_back(filterPtr);
   if (filterPtr->HasName()) {
      fBookedNamedFilters.emplace_back(filterPtr);
   }
}

void TDataFrameImpl::Book(const TmpBranchBasePtr_t &branchPtr)
{
   fBookedBranches[branchPtr->GetName()] = branchPtr;
}

void TDataFrameImpl::Book(const std::shared_ptr<bool> &readinessPtr)
{
   fResProxyReadiness.emplace_back(readinessPtr);
}

void TDataFrameImpl::Book(const RangeBasePtr_t &rangePtr)
{
   fBookedRanges.emplace_back(rangePtr);
}

// dummy call, end of recursive chain of calls
bool TDataFrameImpl::CheckFilters(int, unsigned int)
{
   return true;
}

unsigned int TDataFrameImpl::GetNSlots() const
{
   return fNSlots;
}

/// Call `PrintReport` on all booked filters
void TDataFrameImpl::Report() const
{
   for (const auto &fPtr : fBookedNamedFilters) fPtr->PrintReport();
}

TRangeBase::TRangeBase(TDataFrameImpl *implPtr, const BranchNames_t &tmpBranches, unsigned int start, unsigned int stop,
                       unsigned int stride)
   : fImplPtr(implPtr), fTmpBranches(tmpBranches), fStart(start), fStop(stop), fStride(stride)
{
}

TDataFrameImpl *TRangeBase::GetImplPtr() const
{
   return fImplPtr;
}

BranchNames_t TRangeBase::GetTmpBranches() const
{
   return fTmpBranches;
}
