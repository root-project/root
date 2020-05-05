// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RJittedCustomColumn.hxx>
#include <TError.h> // R__ASSERT

using namespace ROOT::Detail::RDF;

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
