// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RJittedFilter.hxx"

#include <cassert>

ROOT::Detail::RDF::RJittedFilter::RJittedFilter(RLoopManager *lm, std::string_view name,
                                                const std::vector<std::string> &variations)
   : ROOT::Detail::RDF::RFilterBase(lm, name, lm->GetNSlots(), RDFInternal::RColumnRegister(nullptr),
                                    /*columnNames*/ {}, variations)
{
   // Jitted nodes of the computation graph (e.g. RJittedAction, RJittedDefine) usually don't need to register
   // themselves with the RLoopManager: the _concrete_ nodes will be registered with the RLoopManager right before
   // the event loop, at jitting time, and that is good enough.
   // RJittedFilter is an exception: RLoopManager needs to know about what filters have been booked even before
   // the event loop in order to return a correct list from RLoopManager::GetFiltersNames().
   // So RJittedFilters register themselves with RLoopManager at construction time and deregister themselves
   // in SetFilter, i.e. when they are sure that the concrete filter has been instantiated in jitted code and it has
   // been registered with RLoopManager, making the RJittedFilter registration redundant.
   fLoopManager->Register(this);
}

ROOT::Detail::RDF::RJittedFilter::~RJittedFilter()
{
   // This should be a no-op in most sane cases: the RJittedFilter should already have been deregistered in SetFilter.
   // However, in the edge case in which the branch of the computation graph that included this RJittedFilter went out
   // of scope before any event loop ran (e.g. because of bad code logic or a user that changed their mind during
   // interactive usage), we need to make sure RJittedFilters get properly deregistered.
   fLoopManager->Deregister(this);
}

void ROOT::Detail::RDF::RJittedFilter::SetFilter(std::unique_ptr<RFilterBase> f)
{
   // the concrete filter has been registered with RLoopManager on creation, so let's deregister ourselves
   fLoopManager->Deregister(this);
   fConcreteFilter = std::move(f);
}

void ROOT::Detail::RDF::RJittedFilter::InitSlot(TTreeReader *r, unsigned int slot)
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->InitSlot(r, slot);
}

bool ROOT::Detail::RDF::RJittedFilter::CheckFilters(unsigned int slot, Long64_t entry)
{
   assert(fConcreteFilter != nullptr);
   return fConcreteFilter->CheckFilters(slot, entry);
}

void ROOT::Detail::RDF::RJittedFilter::Report(ROOT::RDF::RCutFlowReport &cr) const
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->Report(cr);
}

void ROOT::Detail::RDF::RJittedFilter::PartialReport(ROOT::RDF::RCutFlowReport &cr) const
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->PartialReport(cr);
}

void ROOT::Detail::RDF::RJittedFilter::FillReport(ROOT::RDF::RCutFlowReport &cr) const
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->FillReport(cr);
}

void ROOT::Detail::RDF::RJittedFilter::IncrChildrenCount()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->IncrChildrenCount();
}

void ROOT::Detail::RDF::RJittedFilter::StopProcessing()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->StopProcessing();
}

void ROOT::Detail::RDF::RJittedFilter::ResetChildrenCount()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->ResetChildrenCount();
}

void ROOT::Detail::RDF::RJittedFilter::TriggerChildrenCount()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->TriggerChildrenCount();
}

void ROOT::Detail::RDF::RJittedFilter::ResetReportCount()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->ResetReportCount();
}

void ROOT::Detail::RDF::RJittedFilter::FinalizeSlot(unsigned int slot)
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->FinalizeSlot(slot);
}

void ROOT::Detail::RDF::RJittedFilter::InitNode()
{
   assert(fConcreteFilter != nullptr);
   fConcreteFilter->InitNode();
}

void ROOT::Detail::RDF::RJittedFilter::AddFilterName(std::vector<std::string> &filters)
{
   if (fConcreteFilter == nullptr) {
      // No event loop performed yet, but the JITTING must be performed.
      GetLoopManagerUnchecked()->Jit();
   }
   fConcreteFilter->AddFilterName(filters);
}

std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> ROOT::Detail::RDF::RJittedFilter::GetGraph(
   std::unordered_map<void *, std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode>> &visitedMap)
{
   if (fConcreteFilter != nullptr) {
      // Here the filter exists, so it can be served
      return fConcreteFilter->GetGraph(visitedMap);
   }
   throw std::runtime_error("The Jitting should have been invoked before this method.");
}

std::shared_ptr<ROOT::Detail::RDF::RNodeBase>
ROOT::Detail::RDF::RJittedFilter::GetVariedFilter(const std::string &variationName)
{
   assert(fConcreteFilter != nullptr);
   return fConcreteFilter->GetVariedFilter(variationName);
}
