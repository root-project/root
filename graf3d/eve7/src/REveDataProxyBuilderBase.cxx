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


//______________________________________________________________________________
void REveDataProxyBuilderBase::SetCollection(REveDataCollection* c)
{
   m_collection = c;
}

//______________________________________________________________________________
/*
void
REveDataProxyBuilderBase::SetInteractionList(REveDataInteractionList* l, const std::string& purpose )
{
   // Called if willHandleInteraction() returns false. Purpose ignored by default.

   m_interactionList = l;
}
*/

//______________________________________________________________________________
void REveDataProxyBuilderBase::Build()
{
   if (m_collection)
   {
      printf("Base %p %s %s\n", m_collection, m_collection->GetCName(), m_type.c_str());
      try
      {
         size_t itemSize = (size_t)m_collection->GetNItems(); //cashed

         Clean();
         for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
         {
            printf("build() %s \n", m_collection->GetCName());
            REveElement* elms = (*i)->m_elements;
            size_t oldSize = elms->NumChildren();

            if (HaveSingleProduct())
            {
               Build(m_collection, elms, (*i)->m_viewContext);
            }
            else
            {
               BuildViewType(m_collection, elms, (*i)->m_viewType, (*i)->m_viewContext);
            }

            // Project all children of current product.
            // If product is not registered into any projection-manager,
            // this does nothing.
            REveProjectable* pable = dynamic_cast<REveProjectable*>(elms);
            if (pable->HasProjecteds())
            {
               // loop projected holders
               for (REveProjectable::ProjList_i pi = pable->BeginProjecteds(); pi != pable->EndProjecteds(); ++pi)
               {
                  REveProjectionManager *pmgr = (*pi)->GetManager();
                  Float_t oldDepth = pmgr->GetCurrentDepth();
                  pmgr->SetCurrentDepth(m_layer);
                  size_t cnt = 0;

                  REveElement* projectedAsElement = (*pi)->GetProjectedAsElement();
                  REveElement::List_i parentIt = projectedAsElement->BeginChildren();
                  for (REveElement::List_i prodIt = elms->BeginChildren(); prodIt != elms->EndChildren(); ++prodIt, ++cnt)
                  {
                      // reused projected holder
                     if (cnt < oldSize)
                     {
                        if ((*parentIt)->NumChildren()) {
                            // update projected (mislleading name)
                           for ( REveElement::List_i pci = (*parentIt)->BeginChildren(); pci != (*parentIt)->EndChildren(); pci++)
                               pmgr->ProjectChildrenRecurse(*parentIt);
                        }
                        else {
                            // import projectable
                           pmgr->SubImportChildren(*prodIt, *parentIt);
                        }

                        ++parentIt;
                     }
                     else if (cnt < itemSize)
                     {
                        // new product holder
                        pmgr->SubImportElements(*prodIt, projectedAsElement);
                     }
                     else
                     {
                        break;
                     }
                  }
                  pmgr->SetCurrentDepth(oldDepth);
               }
            }

            /*
            if (m_interactionList && itemSize > oldSize)
            {
               REveElement::List_i elIt = elms->BeginChildren();
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

//______________________________________________________________________________
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

//______________________________________________________________________________


REveElement*
REveDataProxyBuilderBase::CreateProduct( std::string viewType, const REveViewContext* viewContext)
{
   if ( m_products.empty() == false)
   {
      if (HaveSingleProduct()) {
         return m_products.back()->m_elements;
      }
      else {

         for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
         {
            if (viewType == (*i)->m_viewType)
               return (*i)->m_elements;
         }
      }
   }

   Product* product = new Product(viewType, viewContext);
   m_products.push_back(product);

   if (m_collection)
   {
      // debug info in eve browser
      product->m_elements->SetName(Form("product %s", m_collection->GetCName()));
   }
   return product->m_elements;
}

//______________________________________________________________________________

//namespace {
//   void applyVisAttrToChildren(REveElement* p) {
//      for (auto it = p->BeginChildren(); it != p->EndChildren(); ++it)
//      {
//         REveElement* c = *it;
//         if (c->GetMainColor() != p->GetMainColor())
//         {
//            c->SetMainColor(p->GetMainColor());
//            printf("apply color %d to %s\n", p->GetMainColor(), c->GetCName());
//         }
//         c->SetRnrSelf(p->GetRnrSelf());
//         applyVisAttrToChildren(c);
//      }
//   }
//}

void
REveDataProxyBuilderBase::ModelChanges(const REveDataCollection::Ids_t& iIds, Product* p)
{
   printf("REveDataProxyBuilderBase::ModelChanges  %s \n",  m_collection->GetCName());
   REveElement* elms = p->m_elements;
   assert(m_collection && static_cast<int>(m_collection->GetNItems()) <= elms->NumChildren() && "can not use default modelChanges implementation");

   for (REveDataCollection::Ids_t::const_iterator it = iIds.begin(); it != iIds.end(); ++it)
   {
      int itemIdx = *it;
      REveDataItem* item = m_collection->GetDataItem(itemIdx);

      // printf("Edit compound for item index %d \n", itemIdx);
      // imitate FWInteractionList::modelChanges
      REveElement::List_i itElement = elms->BeginChildren();
      std::advance(itElement, itemIdx);
      REveElement* comp = *itElement;
      comp->SetMainColor(item->GetMainColor());
      comp->SetRnrSelf(item->GetRnrSelf());

      // AMT temporary workaround for use of compunds
      // applyVisAttrToChildren(comp);

      if (VisibilityModelChanges(*it, *itElement, p->m_viewContext))
      {
         elms->ProjectChild(*itElement);
         printf("---REveDataProxyBuilderBase project child\n ");
      }
   }
}

//______________________________________________________________________________


void
REveDataProxyBuilderBase::ModelChanges(const REveDataCollection::Ids_t& iIds)
{
  if(m_haveWindow) {
    for (Product_it i = m_products.begin(); i!= m_products.end(); ++i)
    {
       ModelChanges(iIds, *i);
    }
    m_modelsChanged=false;
  } else {
    m_modelsChanged=true;
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
   el->SetMainColor(m_collection->GetMainColor());
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
   REveCompound* c = new REveCompound();
   c->CSCTakeMotherAsMaster();
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

//______________________________________________________________________________

void
REveDataProxyBuilderBase::Clean()
{
   // Cleans local common element list.
   for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
   {
      if ((*i)->m_elements)
         (*i)->m_elements->DestroyElements();
   }

   CleanLocal();
}

void
REveDataProxyBuilderBase::CleanLocal()
{
   // Cleans local common element list.
}

void
REveDataProxyBuilderBase::CollectionBeingDestroyed(const REveDataCollection* /*iItem*/)
{
   m_collection = 0;

   CleanLocal();

   for (Product_it i = m_products.begin(); i!= m_products.end(); i++)
   {

      // (*i)->m_scaleConnection.disconnect();
      delete (*i);
   }

   m_products.clear();
}

bool
REveDataProxyBuilderBase::VisibilityModelChanges(int, REveElement*, const REveViewContext*)
{
   return false;
}

void
REveDataProxyBuilderBase::SetHaveAWindow(bool iHaveAWindow)
{
   m_haveWindow = iHaveAWindow;
}
