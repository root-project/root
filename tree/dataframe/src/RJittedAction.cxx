// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RBookedCustomColumns.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RJittedAction.hxx"
#include "TError.h"

using ROOT::Internal::RDF::RJittedAction;
using ROOT::Detail::RDF::RLoopManager;

RJittedAction::RJittedAction(RLoopManager &lm) : RActionBase(&lm, {}, ROOT::Internal::RDF::RBookedCustomColumns{}) { }

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

void RJittedAction::SetHasRun()
{
   R__ASSERT(fConcreteAction != nullptr);
   return fConcreteAction->SetHasRun();
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
