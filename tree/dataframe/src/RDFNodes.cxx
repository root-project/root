// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RCutFlowReport.hxx"
#include "ROOT/RDFNodes.hxx"
#include "ROOT/RDFUtils.hxx"
#include "ROOT/RStringView.hxx"
#include "RtypesCore.h" // Long64_t
#include "TError.h"

#include <memory>
#include <numeric>
#include <string>
#include <vector>

class TDirectory;
namespace ROOT {
class TSpinMutex;
} // namespace ROOT

using namespace ROOT::Detail::RDF;
using namespace ROOT::Internal::RDF;

namespace ROOT {
namespace Internal {
namespace RDF {

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

void RJittedAction::ClearValueReaders(unsigned int slot)
{
   R__ASSERT(fConcreteAction != nullptr);
   return fConcreteAction->ClearValueReaders(slot);
}

std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> RJittedAction::GetGraph()
{
   R__ASSERT(fConcreteAction != nullptr);
   return fConcreteAction->GetGraph();
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

RFilterBase::RFilterBase(RLoopManager *implPtr, std::string_view name, const unsigned int nSlots,
                         const RDFInternal::RBookedCustomColumns &customColumns)
   : RNodeBase(implPtr), fLastResult(nSlots), fAccepted(nSlots), fRejected(nSlots), fName(name), fNSlots(nSlots),
     fCustomColumns(customColumns)
{
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

void RJittedFilter::ClearTask(unsigned int slot)
{
   R__ASSERT(fConcreteFilter != nullptr);
   fConcreteFilter->ClearTask(slot);
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
      GetLoopManagerUnchecked()->BuildJittedNodes();
   }
   fConcreteFilter->AddFilterName(filters);
}

void RRangeBase::ResetCounters()
{
   fLastCheckedEntry = -1;
   fNProcessedEntries = 0;
   fHasStopped = false;
}
