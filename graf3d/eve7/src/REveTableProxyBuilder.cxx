// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TClass.h"
#include "TROOT.h"
#include <ROOT/REveTableProxyBuilder.hxx>
#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveDataCollection.hxx>
#include <ROOT/REveDataTable.hxx>
#include <ROOT/REveManager.hxx>

using namespace ROOT::Experimental;

REveTableProxyBuilder::REveTableProxyBuilder() : REveDataProxyBuilderBase(), fTable(nullptr)
{
    fTable = new REveDataTable("ProxyTable");
}

REveTableProxyBuilder::~REveTableProxyBuilder()
{
   fTable->Destroy();
   fTable = nullptr;
}

// reuse table product
void REveTableProxyBuilder::Clean()
{
}

void REveTableProxyBuilder::Build(const REveDataCollection* collection, REveElement* product, const REveViewContext* context)
{
   REveTableViewInfo* info = context->GetTableViewInfo();
   if (info->GetDisplayedCollection() != Collection()->GetElementId())
   {
      return;
   }

   if (product->NumChildren() == 0) {
      product->AddElement(fTable);
   }

   // printf("-----REveTableProxyBuilder::Build() body for %s (%p, %p)\n",collection->GetCName(), collection, Collection() );

   if (info->GetConfigChanged() || fTable->NumChildren() == 0) {
      fTable->DestroyElements();
      std::stringstream ss;
      REveTableHandle::Entries_t& tableEntries =  context->GetTableViewInfo()->RefTableEntries(collection->GetItemClass()->GetName());
      for (const REveTableEntry& spec : tableEntries) {
         auto c = new REveDataColumn(spec.fName.c_str());
         fTable->AddElement(c);
         using namespace std::string_literals;
         std::string exp  =  spec.fExpression;
         c->SetPrecision(spec.fPrecision);
         c->SetExpressionAndType(exp.c_str(), spec.fType);
         ss << c->GetFunctionExpressionString();
         ss << "\n";
      }
      // std::cout << ss.str();
      gROOT->ProcessLine(ss.str().c_str());
   }
   fTable->StampObjProps();
}



void REveTableProxyBuilder::SetCollection(REveDataCollection* collection)
{
   REveDataProxyBuilderBase::SetCollection(collection);
   fTable->SetCollection(collection);
}

void REveTableProxyBuilder::ModelChanges(const REveDataCollection::Ids_t&, REveDataProxyBuilderBase::Product*)
{
   // printf("REveTableProxyBuilder::ModelChanges\n");
   if (fTable) fTable->StampObjProps();
}

void REveTableProxyBuilder::ConfigChanged() {
   Build();
}
