// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveDataSimpleProxyBuilder.hxx>

#include <ROOT/REveDataCollection.hxx>
#include <ROOT/REveCompound.hxx>
#include <ROOT/REveScene.hxx>
#include <cassert>
#include <TClass.h>

using namespace ROOT::Experimental;

REveDataSimpleProxyBuilder::REveDataSimpleProxyBuilder()
{}

REveDataSimpleProxyBuilder::~REveDataSimpleProxyBuilder()
{
   for (auto &p : m_products) {
      for (auto &compound : p->m_elements->RefChildren())
         compound->DecDenyDestroy();
   }
}

void REveDataSimpleProxyBuilder::Clean()
{
   for (auto &p : m_products) {
      auto spbIt = fProductMap.find(p->m_elements);
      if (spbIt != fProductMap.end()) {
         REveElement *product = p->m_elements;
         for (auto &compound : product->RefChildren()) {
            REveCollectionCompound *collComp = dynamic_cast<REveCollectionCompound *>(compound);
            collComp->DestroyElements();
            (spbIt)->second->cache.push_back(collComp);
         }
      }
   }

   CleanLocal();
}

//______________________________________________________________________________
REveCollectionCompound*
REveDataSimpleProxyBuilder::CreateCompound(bool set_color, bool propagate_color_to_all_children)
{
   REveCollectionCompound *c = new REveCollectionCompound(Collection());
   c->IncDenyDestroy();
   c->CSCImplySelectAllChildren();
   c->SetPickable(true);
   if (set_color)
   {
      c->SetMainColor(Collection()->GetMainColor());
      c->SetMainTransparency(Collection()->GetMainTransparency());
   }
   if (propagate_color_to_all_children)
   {
      c->CSCApplyMainColorToAllChildren();
      c->CSCApplyMainTransparencyToAllChildren();
   }
   else
   {
      c->CSCApplyMainColorToMatchingChildren();
      c->CSCApplyMainTransparencyToMatchingChildren();
   }
   return c;
}

//______________________________________________________________________________
REveCompound *REveDataSimpleProxyBuilder::GetHolder(REveElement *product, int idx)
{
   SPBProduct *spb = nullptr;

   // printf("REveDataSimpleProxyBuilder::GetHolder begin %s %d \n", Collection()->GetCName(), idx);

   auto it = fProductMap.find(product);
   if (it != fProductMap.end()) {
      spb = it->second;
   } else {
      spb = new SPBProduct;
      fProductMap.emplace(product, spb);
   }

   REveCollectionCompound *itemHolder = nullptr;

   auto hit = spb->map.find(idx);
   if (hit != spb->map.end()) {
      itemHolder = hit->second;
      // printf("GetHolder already in map %d \n", idx);
   } else {

      if (!spb->cache.empty()) {
         itemHolder = spb->cache.front();
         spb->cache.pop_front();
      }

      if (!itemHolder) {
         itemHolder = CreateCompound(true, true);
         product->AddElement(itemHolder);
         // printf("Creating new holder\n");
      }

      spb->map.emplace(idx, itemHolder);
      itemHolder->SetMainColor(Collection()->GetDataItem(idx)->GetMainColor());
      std::string name = Form("%s %d", Collection()->GetCName(), idx);
      itemHolder->SetName(name);
   }

   return itemHolder;
}

void
REveDataSimpleProxyBuilder::Build(const REveDataCollection *collection,
                                  REveElement* product, const REveViewContext* vc)
{
   // printf("REveDataSimpleProxyBuilder::Build %s %d\n", collection->GetCName(), collection->GetNItems());
   auto size = collection->GetNItems();
   for (int index = 0; index < size; ++index)
   {
      const REveDataItem* di = Collection()->GetDataItem(index);
      if (di->GetRnrSelf() && !di->GetFiltered())
      {
         REveCompound *itemHolder = GetHolder(product, index);
         Build(collection->GetDataPtr(index), index, itemHolder, vc);
      }
   }
}

void
REveDataSimpleProxyBuilder::BuildViewType(const REveDataCollection* collection,
                                          REveElement* product, const std::string& viewType, const REveViewContext* vc)
{
   auto size = collection->GetNItems();
   for (int index = 0; index < size; ++index)
   {
      auto di = Collection()->GetDataItem(index);
      if (di->GetRnrSelf() && !di->GetFiltered())
      {
         REveCompound *itemHolder = GetHolder(product, index);
         BuildViewType(collection->GetDataPtr(index), index, itemHolder, viewType, vc);
      }
   }
}

//______________________________________________________________________________

namespace
{
   void applyColorAttrToChildren(REveElement* p) {
      for (auto &it: p->RefChildren())
      {
         REveElement* c = it;
         if (c->GetMainColor() != p->GetMainColor())
         {
            c->SetMainColor(p->GetMainColor());
         }
         applyColorAttrToChildren(c);
      }
   }
}

void REveDataSimpleProxyBuilder::ModelChanges(const REveDataCollection::Ids_t &iIds, Product *p)
{
   for (auto itemIdx : iIds) {
      const REveDataItem *item = Collection()->GetDataItem(itemIdx);
      bool visible = ((!item->GetFiltered()) && item->GetRnrSelf()) && Collection()->GetRnrSelf();

      auto sit = fProductMap.find(p->m_elements);
      if (sit == fProductMap.end())
         return;

      auto spb = sit->second;
      REveCompound *holder = nullptr;
      auto hmit = spb->map.find(itemIdx);
      if (hmit != spb->map.end())
         holder = hmit->second;

      bool createHolder = visible && !holder;

      if (createHolder) {
         holder = GetHolder(p->m_elements, itemIdx);

         if (HaveSingleProduct())
            Build(Collection()->GetDataPtr(itemIdx), itemIdx, holder, p->m_viewContext);
         else

            applyColorAttrToChildren(holder);
         p->m_elements->ProjectChild(holder);
      } else if (holder) {
         holder->SetRnrSelf(visible);
         holder->SetRnrChildren(visible);
         holder->SetMainColor(item->GetMainColor());
         applyColorAttrToChildren(holder);
         LocalModelChanges(itemIdx, holder, p->m_viewContext);
      }
   }
}
//______________________________________________________________________________

bool
REveDataSimpleProxyBuilder::VisibilityModelChanges(int idx, REveElement* iCompound, const std::string& viewType, const REveViewContext* vc)
{
   const REveDataItem *item = Collection()->GetDataItem(idx);
   bool returnValue = false;

   if (item->GetVisible() ) {
      if (HaveSingleProduct())
         Build(Collection()->GetDataPtr(idx), idx, iCompound, vc);
      else
         BuildViewType(Collection()->GetDataPtr(idx), idx, iCompound, viewType, vc);
      returnValue = true;
   }
   return returnValue;
}

//______________________________________________________________________________

void REveDataSimpleProxyBuilder::FillImpliedSelected(REveElement::Set_t &impSet, Product* p)
{
   for (auto &s : Collection()->GetItemList()->RefSelectedSet()) {
      // printf("Fill implied selected %d \n", s);
      auto spb = fProductMap[p->m_elements];
      auto it = spb->map.find(s);
      if (it != spb->map.end()) {
         // printf("Fill implied selected %s \n", it->second->GetCName());
         it->second->FillImpliedSelectedSet(impSet);
      }
   }
}

//==============================================================================

//==============================================================================

//==============================================================================

REveCollectionCompound::REveCollectionCompound(REveDataCollection* c)
{
   fCollection = c;
   SetSelectionMaster(fCollection);
}

//______________________________________________________________________________

REveCollectionCompound::~REveCollectionCompound()
{
   // deny destroy ??
}

//______________________________________________________________________________

REveElement *REveCollectionCompound::GetSelectionMaster()
{
   static const REveException eh("REveCollectionCompound::GetSelectionMaster()");

   if (!fCollection->GetScene()->IsAcceptingChanges())
      return fCollection->GetItemList();

   fCollection->GetItemList()->RefSelectedSet().clear();
   try {

      std::size_t found = fName.find_last_of(" ");
      if (found == std::string::npos)
      {
         throw(eh + TString::Format("Can't retrive item index from %s", fName.c_str()));
      }
      std::string idss = fName.substr(found + 1);
      int idx = stoi(idss);
      // printf("REveCollectionCompound::GetSelectionMaster %d\n", idx);
      fCollection->GetItemList()->RefSelectedSet().insert(idx);
   } catch (std::exception& e) {
       gEveLog << "REveCollectionCompound::GetSelectionMaster " << e.what() << std::endl;
       fCollection->GetItemList()->RefSelectedSet().insert(0);
   }
   return fCollection->GetItemList();
}
