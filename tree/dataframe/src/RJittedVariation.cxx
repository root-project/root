// Author: Enrico Guiraud CERN  02/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RJittedVariation.hxx>

#include <cassert>

ROOT::Internal::RDF::RJittedVariation::~RJittedVariation() {}

void ROOT::Internal::RDF::RJittedVariation::InitSlot(TTreeReader *r, unsigned int slot)
{
   assert(fConcreteVariation != nullptr);
   fConcreteVariation->InitSlot(r, slot);
}

void *ROOT::Internal::RDF::RJittedVariation::GetValuePtr(unsigned int slot, const std::string &column,
                                                         const std::string &variation)
{
   assert(fConcreteVariation != nullptr);
   return fConcreteVariation->GetValuePtr(slot, column, variation);
}

const std::type_info &ROOT::Internal::RDF::RJittedVariation::GetTypeId() const
{
   assert(fConcreteVariation != nullptr);
   return fConcreteVariation->GetTypeId();
}

void ROOT::Internal::RDF::RJittedVariation::Update(unsigned int slot, Long64_t entry)
{
   assert(fConcreteVariation != nullptr);
   fConcreteVariation->Update(slot, entry);
}

void ROOT::Internal::RDF::RJittedVariation::FinalizeSlot(unsigned int slot)
{
   assert(fConcreteVariation != nullptr);
   fConcreteVariation->FinalizeSlot(slot);
}
