// Author: Enrico Guiraud, Danilo Piparo CERN  12/2016

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "ROOT/TDataFrame.hxx"
#include "ROOT/TTreeProcessorMT.hxx"
#include "ROOT/TSpinMutex.hxx"
#include "TDirectory.h"
#include "TError.h" // Info
#include "TROOT.h" // IsImplicitMTEnabled, GetImplicitMTPoolSize
#include "TString.h" // Printf

#include <numeric> // std::accumulate
#include <thread>

namespace ROOT {

namespace Internal {

unsigned int GetNSlots() {
   unsigned int nSlots = 1;
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) nSlots = ROOT::GetImplicitMTPoolSize();
#endif // R__USE_IMT
   return nSlots;
}

void CheckTmpBranch(const std::string& branchName, TTree *treePtr)
{
   auto branch = treePtr->GetBranch(branchName.c_str());
   if (branch != nullptr) {
      auto msg = "branch \"" + branchName + "\" already present in TTree";
      throw std::runtime_error(msg);
   }
}

/// Returns local BranchNames or default BranchNames according to which one should be used
const BranchNames &PickBranchNames(unsigned int nArgs, const BranchNames &bl, const BranchNames &defBl)
{
   bool useDefBl = false;
   if (nArgs != bl.size()) {
      if (bl.size() == 0 && nArgs == defBl.size()) {
         useDefBl = true;
      } else {
         auto msg = "mismatch between number of filter arguments (" + std::to_string(nArgs) +
                    ") and number of branches (" + std::to_string(bl.size() ? bl.size() : defBl.size()) + ")";
         throw std::runtime_error(msg);
      }
   }

   return useDefBl ? defBl : bl;
}

void TDataFrameActionBase::CreateSlots(unsigned int nSlots) { fReaderValues.resize(nSlots); }

} // end NS Internal

namespace Detail {

TDataFrameBranchBase::TDataFrameBranchBase(std::weak_ptr<TDataFrameImpl> df, BranchNames branches, const std::string &name)
   : fFirstData(df), fTmpBranches(branches), fName(name) {};
BranchNames TDataFrameBranchBase::GetTmpBranches() const { return fTmpBranches; }
std::string TDataFrameBranchBase::GetName() const { return fName; }


TDataFrameFilterBase::TDataFrameFilterBase(std::weak_ptr<TDataFrameImpl> df, BranchNames branches, const std::string& name)
   : fFirstData(df), fTmpBranches(branches), fName(name) {};

std::weak_ptr<TDataFrameImpl> TDataFrameFilterBase::GetDataFrame() const { return fFirstData; }

BranchNames TDataFrameFilterBase::GetTmpBranches() const { return fTmpBranches; }

void TDataFrameFilterBase::CreateSlots(unsigned int nSlots)
{
   fReaderValues.resize(nSlots);
   fLastCheckedEntry.resize(nSlots);
   fLastResult.resize(nSlots);
   fAccepted.resize(nSlots);
   fRejected.resize(nSlots);
   // fAccepted and fRejected could be different than 0 if this is not the
   // first event-loop run using this filter
   std::fill(fAccepted.begin(), fAccepted.end(), 0);
   std::fill(fRejected.begin(), fRejected.end(), 0);
}

void TDataFrameFilterBase::PrintReport() const {
   if (fName.empty())
      return;
   const auto accepted = std::accumulate(fAccepted.begin(), fAccepted.end(), 0ULL);
   const auto all = accepted + std::accumulate(fRejected.begin(), fRejected.end(), 0ULL);
   double perc = accepted;
   if (all > 0)
      perc /= all;
   perc *= 100.;
   Printf("%-10s: pass=%-10lld all=%-10lld -- %8.3f %%",
          fName.c_str(), accepted, all, perc);
}


TDataFrameImpl::TDataFrameImpl(const std::string &treeName, TDirectory *dirPtr, const BranchNames &defaultBranches)
   : fTreeName(treeName), fDirPtr(dirPtr), fDefaultBranches(defaultBranches), fNSlots(ROOT::Internal::GetNSlots())
{
}

TDataFrameImpl::TDataFrameImpl(TTree &tree, const BranchNames &defaultBranches)
   : fTree(&tree), fDefaultBranches(defaultBranches), fNSlots(ROOT::Internal::GetNSlots())
{
}

void TDataFrameImpl::Run()
{
#ifdef R__USE_IMT
   if (ROOT::IsImplicitMTEnabled()) {
      const auto fileName = fTree ? static_cast<TFile *>(fTree->GetCurrentFile())->GetName() : fDirPtr->GetName();
      const std::string      treeName = fTree ? fTree->GetName() : fTreeName;
      ROOT::TTreeProcessorMT tp(fileName, treeName);
      ROOT::TSpinMutex       slotMutex;
      std::map<std::thread::id, unsigned int> slotMap;
      unsigned int globalSlotIndex = 0;
      CreateSlots(fNSlots);
      tp.Process([this, &slotMutex, &globalSlotIndex, &slotMap](TTreeReader &r) -> void {
         const auto   thisThreadID = std::this_thread::get_id();
         unsigned int slot;
         {
            std::lock_guard<ROOT::TSpinMutex> l(slotMutex);
            auto                              thisSlotIt = slotMap.find(thisThreadID);
            if (thisSlotIt != slotMap.end()) {
               slot = thisSlotIt->second;
            } else {
               slot                  = globalSlotIndex;
               slotMap[thisThreadID] = slot;
               ++globalSlotIndex;
            }
         }

         BuildAllReaderValues(r, slot);

         // recursive call to check filters and conditionally execute actions
         while (r.Next())
            for (auto &actionPtr : fBookedActions) actionPtr->Run(slot, r.GetCurrentEntry());
      });
   } else {
#endif // R__USE_IMT
      TTreeReader r;
      if (fTree) {
         r.SetTree(fTree);
      } else {
         r.SetTree(fTreeName.c_str(), fDirPtr);
      }

      CreateSlots(1);
      BuildAllReaderValues(r, 0);

      // recursive call to check filters and conditionally execute actions
      while (r.Next())
         for (auto &actionPtr : fBookedActions) actionPtr->Run(0, r.GetCurrentEntry());
#ifdef R__USE_IMT
   }
#endif // R__USE_IMT

   fHasRunAtLeastOnce = true;
   // forget actions and "detach" the action result pointers marking them ready
   // and forget them too
   fBookedActions.clear();
   for (auto readiness : fResProxyReadiness) {
      *readiness.get() = true;
   }
   fResProxyReadiness.clear();
}

// build reader values for all actions, filters and branches
void TDataFrameImpl::BuildAllReaderValues(TTreeReader &r, unsigned int slot)
{
   for (auto &ptr : fBookedActions) ptr->BuildReaderValues(r, slot);
   for (auto &ptr : fBookedFilters) ptr->BuildReaderValues(r, slot);
   for (auto &bookedBranch : fBookedBranches) bookedBranch.second->BuildReaderValues(r, slot);
}

// inform all actions filters and branches of the required number of slots
void TDataFrameImpl::CreateSlots(unsigned int nSlots)
{
   for (auto &ptr : fBookedActions) ptr->CreateSlots(nSlots);
   for (auto &ptr : fBookedFilters) ptr->CreateSlots(nSlots);
   for (auto &bookedBranch : fBookedBranches) bookedBranch.second->CreateSlots(nSlots);
}

std::weak_ptr<ROOT::Detail::TDataFrameImpl> TDataFrameImpl::GetDataFrame() const
{
   return fFirstData;
}

const BranchNames &TDataFrameImpl::GetDefaultBranches() const
{
   return fDefaultBranches;
}

const BranchNames TDataFrameImpl::GetTmpBranches() const
{
   return fTmpBranches;
}

TTree *TDataFrameImpl::GetTree() const
{
   if (fTree) {
      return fTree;
   } else {
      auto treePtr = static_cast<TTree *>(fDirPtr->Get(fTreeName.c_str()));
      return treePtr;
   }
}

const TDataFrameBranchBase &TDataFrameImpl::GetBookedBranch(const std::string &name) const
{
   return *fBookedBranches.find(name)->second.get();
}

void *TDataFrameImpl::GetTmpBranchValue(const std::string &branch, unsigned int slot, Long64_t entry)
{
   return fBookedBranches.at(branch)->GetValue(slot, entry);
}

TDirectory *TDataFrameImpl::GetDirectory() const
{
   return fDirPtr;
}

std::string TDataFrameImpl::GetTreeName() const
{
   return fTreeName;
}

void TDataFrameImpl::SetFirstData(const std::shared_ptr<TDataFrameImpl> &sp)
{
   fFirstData = sp;
}

void TDataFrameImpl::Book(Internal::ActionBasePtr_t actionPtr)
{
   fBookedActions.emplace_back(actionPtr);
}

void TDataFrameImpl::Book(ROOT::Detail::FilterBasePtr_t filterPtr)
{
   fBookedFilters.emplace_back(filterPtr);
}

void TDataFrameImpl::Book(TmpBranchBasePtr_t branchPtr)
{
   fBookedBranches[branchPtr->GetName()] = branchPtr;
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
void TDataFrameImpl::Report() const {
   for(const auto& fPtr : fBookedFilters)
      fPtr->PrintReport();
}


} // end NS Detail

} // end NS ROOT


