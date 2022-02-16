// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RJittedDefine.hxx>
#include <ROOT/RDF/RLoopManager.hxx>

#include <cassert>

using namespace ROOT::Detail::RDF;

RJittedDefine::~RJittedDefine()
{
   fLoopManager->Deregister(this);
}

void RJittedDefine::InitSlot(TTreeReader *r, unsigned int slot)
{
   assert(fConcreteDefine != nullptr);
   fConcreteDefine->InitSlot(r, slot);
}

void *RJittedDefine::GetValuePtr(unsigned int slot)
{
   assert(fConcreteDefine != nullptr);
   return fConcreteDefine->GetValuePtr(slot);
}

const std::type_info &RJittedDefine::GetTypeId() const
{
   assert(fConcreteDefine != nullptr);
   return fConcreteDefine->GetTypeId();
}

void RJittedDefine::Update(unsigned int slot, Long64_t entry)
{
   assert(fConcreteDefine != nullptr);
   fConcreteDefine->Update(slot, entry);
}

void RJittedDefine::Update(unsigned int slot, const ROOT::RDF::RSampleInfo &id)
{
   assert(fConcreteDefine != nullptr);
   fConcreteDefine->Update(slot, id);
}

void RJittedDefine::FinalizeSlot(unsigned int slot)
{
   assert(fConcreteDefine != nullptr);
   fConcreteDefine->FinalizeSlot(slot);
}

void RJittedDefine::MakeVariations(const std::vector<std::string> &variations)
{
   assert(fConcreteDefine != nullptr);
   return fConcreteDefine->MakeVariations(variations);
}

RDefineBase &RJittedDefine::GetVariedDefine(const std::string &variationName)
{
   assert(fConcreteDefine != nullptr);
   return fConcreteDefine->GetVariedDefine(variationName);
}
