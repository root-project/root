// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel, 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/REveDataProxyBuilderBase.hxx>
#include <ROOT/REveProjectionManager.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveCompound.hxx>

#include <cassert>


using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;


REveDataProxyBuilderBase::Product::Product(std::string iViewType, const REveViewContext* c) : m_viewType(iViewType), m_viewContext(c), m_elements(0)
{
   m_elements = new REveCompound("ProxyProduct", "", false);
   m_elements->IncDenyDestroy();
}

//______________________________________________________________________________


REveDataProxyBuilderBase::REveDataProxyBuilderBase(const std::string &type):
   m_type(type),
   m_collection(nullptr),
   //   m_interactionList(0),
   m_haveWindow(false)
{
}

REveDataProxyBuilderBase::Product::~Product()
{
   // remove product from projected scene (RhoPhi or RhoZ)
   for (auto i : m_elements->RefProjecteds())
   {
      REveElement *projected = i->GetProjectedAsElement();
      projected->GetMother()->RemoveElement(projected);
   }

   // XXXX This might break now ... m_elements will auto drestruct, no?
   // We don't set incdenydestroy or additional something, do we?

   // remove from 3D scenes
   if (m_elements->HasMother())
   {
      m_elements->GetMother()->RemoveElement(m_elements);
   }

   m_elements->Annihilate();
}

//------------------------------------------------------------------------------

void REveDataProxyBuilderBase::SetCollection(REveDataCollection* c)
{
   m_collection = c;
}

//------------------------------------------------------------------------------

/*
void
REveDataProxyBuilderBase::SetInteractionList(REveDataInteractionList* l, const std::string& purpose )
{
   // Called if willHandleInteraction() returns false. Purpose ignored by default.

   m_interactionList = l;
}
*/

//------------------------------------------------------------------------------

void REveDataProxyBuilderBase::Build()
{
   if (m_collection)
   {
      // printf("Base %p %s %s\n", m_collection, m_collection->GetCName(), m_type.c_str());
      try
      {
         auto itemSize = m_collection->GetNItems(); //cashed

         Clean();
         for (auto &pp: m_products)
         {
            // printf("build() %s \n", m_collection->GetCName());
            REveElement* elms = pp->m_elements;
            auto oldSize = elms->NumChildren();

            if (HaveSingleProduct())
            {
               Build(m_collection, elms, pp->m_viewContext);
            }
            else
            {
               BuildViewType(m_collection, elms, pp->m_viewType, pp->m_viewContext);
            }

            // Project all children of current product.
            // If product is not registered into any projection-manager,
            // this does nothing.
            REveProjectable* pable = dynamic_cast<REveProjectable*>(elms);
            if (pable->HasProjecteds())
            {
               // loop projected holders
               for (auto &prj: pable->RefProjecteds())
               {
                  REveProjectionManager *pmgr = prj->GetManager();
                  Float_t oldDepth = pmgr->GetCurrentDepth();
                  pmgr->SetCurrentDepth(m_layer);
                  Int_t cnt = 0;

                  REveElement *projectedAsElement = prj->GetProjectedAsElement();
                  auto parentIt = projectedAsElement->RefChildren().begin();
                  for (auto &prod: elms->RefChildren())
                  {
                      // reused projected holder
                     if (cnt < oldSize)
                     {
                        /*
                        // AMT no use case for this at the moment
                        if ((*parentIt)->NumChildren()) {
                            // update projected (mislleading name)
                           for ( REveElement::List_i pci = (*parentIt)->BeginChildren(); pci != (*parentIt)->EndChildren(); pci++)
                               pmgr->ProjectChildrenRecurse(*pci);
                        }
                        */
                        // import projectable
                        pmgr->SubImportChildren(prod, *parentIt);

                        ++parentIt;
                     }
                     else if (cnt < itemSize)
                     {
                        // new product holder
                        pmgr->SubImportElements(prod, projectedAsElement);
                     }
                     else
                     {
                        break;
                     }
                     ++cnt;
                  }
                  pmgr->SetCurrentDepth(oldDepth);
               }
            }

            /*
            if (m_interactionList && itemSize > oldSize)
            {
               auto elIt = elms->RefChildren().begin();
               for (size_t cnt = 0; cnt < itemSize; ++cnt, ++elIt)
               {
                  if (cnt >= oldSize )
                       m_interactionList->Added(*elIt, cnt);
               }
            }
            */
         }
      }
      catch (const std::runtime_error& iException)
      {
         std::cout << "Caught exception in build function for item " << m_collection->GetCName() << ":\n"
                              << iException.what() << std::endl;
         exit(1);
      }
   }
}

//------------------------------------------------------------------------------

void
REveDataProxyBuilderBase::Build(const REveDataCollection*, REveElement*, const REveViewContext*)
{
   assert("virtual Build(const REveEventItem*, REveElement*, const REveViewContext*) not implemented by inherited class");
}


void
REveDataProxyBuilderBase::BuildViewType(const REveDataCollection*, REveElement*, std::string, const REveViewContext*)
{
   assert("virtual BuildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType, const FWViewContext*) not implemented by inherited class");
}

//------------------------------------------------------------------------------

REveElement*
REveDataProxyBuilderBase::CreateProduct( std::string viewType, const REveViewContext* viewContext)
{
   if ( m_products.empty() == false)
   {
      if (HaveSingleProduct()) {
         return m_products.back()->m_elements;
      }
      else {

         for (auto &prod: m_products)
         {
            if (viewType == prod->m_viewType)
               return prod->m_elements;
         }
      }
   }

   auto product = new Product(viewType, viewContext);
   m_products.push_back(product);

   if (m_collection)
   {
      // debug info in eve browser
      product->m_elements->SetName(Form("product %s", m_collection->GetCName()));
   }
   return product->m_elements;
}

//------------------------------------------------------------------------------

namespace
{
   void applyColorAttrToChildren(REveElement* p) {
      for (auto &it: p->RefChildren())
      {
         REveElement* c = it;
         if (c->GetMainColor() != p->GetMainColor())
         {
            c->SetMainColor(p->GetMainColor());
            // printf("apply color %d to %s\n", p->GetMainColor(), c->GetCName());
         }
         applyColorAttrToChildren(c);
      }
   }
}

void
REveDataProxyBuilderBase::ModelChanges(const REveDataCollection::Ids_t& iIds, Product* p)
{
   printf("REveDataProxyBuilderBase::ModelChanges  %s \n",  m_collection->GetCName());
   REveElement* elms = p->m_elements;
   assert(m_collection && static_cast<int>(m_collection->GetNItems()) <= elms->NumChildren() && "can not use default modelChanges implementation");

   for (auto itemIdx: iIds)
   {
      REveDataItem *item = m_collection->GetDataItem(itemIdx);

      // printf("Edit compound for item index %d \n", itemIdx);
      // imitate FWInteractionList::modelChanges
      auto itElement = elms->RefChildren().begin();
      std::advance(itElement, itemIdx);
      REveElement* comp = *itElement;
      bool visible = (!item->GetFiltered()) && item->GetRnrSelf();
      comp->SetRnrSelf(visible);
      comp->SetRnrChildren(visible);

      if (item->GetMainColor() != comp->GetMainColor()) comp->SetMainColor(item->GetMainColor());
      applyColorAttrToChildren(comp);

      if (VisibilityModelChanges(itemIdx, comp, p->m_viewContext))
      {
         elms->ProjectChild(comp);
         printf("---REveDataProxyBuilderBase project child\n");
      }
      else
      {
         LocalModelChanges(itemIdx, comp, p->m_viewContext);
      }
   }
}

void
REveDataProxyBuilderBase::LocalModelChanges(int, REveElement*, const REveViewContext*)
{
   // Nothing to be done in base class.
   // Visibility, main color and main transparency are handled automatically throught compound.
}

//------------------------------------------------------------------------------

void
REveDataProxyBuilderBase::ModelChanges(const REveDataCollection::Ids_t& iIds)
{
  if(m_haveWindow) {
     for (auto &prod: m_products)
    {
       ModelChanges(iIds, prod);
    }
    m_modelsChanged = false;
  } else {
    m_modelsChanged = true;
  }
}

//______________________________________________________________________________
void
REveDataProxyBuilderBase::CollectionChanged(const REveDataCollection* /*iItem*/)
{
   if(m_haveWindow) {
      Build();
   }
}

//------------------------------------------------------------------------------

void
REveDataProxyBuilderBase::SetupAddElement(REveElement* el, REveElement* parent, bool color) const
{
   SetupElement(el, color);
   // AMT -- this temprary to get right tooltip
   el->SetName(parent->GetName());
   parent->AddElement(el);
}

/** This method is invoked to setup the per element properties of the various
    objects being drawn.
  */
void
REveDataProxyBuilderBase::SetupElement(REveElement* el, bool color) const
{
   el->CSCTakeMotherAsMaster();
   el->SetPickable(true);

   if (color)
   {
      el->CSCApplyMainColorToMatchingChildren();
      el->CSCApplyMainTransparencyToMatchingChildren();
      el->SetMainColor(m_collection->GetMainColor());
      el->SetMainTransparency(m_collection->GetMainTransparency());
   }
}



REveCompound*
REveDataProxyBuilderBase::CreateCompound(bool set_color, bool propagate_color_to_all_children) const
{
   REveCompound *c = new REveCompound();
   c->CSCImplySelectAllChildren();
   c->SetPickable(true);
   if (set_color)
   {
      c->SetMainColor(m_collection->GetMainColor());
      c->SetMainTransparency(m_collection->GetMainTransparency());
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

//------------------------------------------------------------------------------

void REveDataProxyBuilderBase::Clean()
{
   // Cleans local common element list.
   for (auto &prod: m_products)
   {
      if (prod->m_elements)
         prod->m_elements->DestroyElements();
   }

   CleanLocal();
}

void REveDataProxyBuilderBase::CleanLocal()
{
   // Cleans local common element list.
}

void REveDataProxyBuilderBase::CollectionBeingDestroyed(const REveDataCollection* /*iItem*/)
{
   m_collection = nullptr;

   CleanLocal();

   for (auto &prod: m_products)
   {

      // (*i)->m_scaleConnection.disconnect();
      delete prod;
   }

   m_products.clear();
}

bool REveDataProxyBuilderBase::VisibilityModelChanges(int, REveElement *, const REveViewContext *)
{
   return false;
}

void REveDataProxyBuilderBase::SetHaveAWindow(bool iHaveAWindow)
{
   m_haveWindow = iHaveAWindow;
}
