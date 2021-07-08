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
#include <ROOT/RLogger.hxx>
#include <ROOT/REveScene.hxx>

#include <cassert>

using namespace ROOT::Experimental;


REveDataProxyBuilderBase::Product::Product(std::string iViewType, const REveViewContext* c) : m_viewType(iViewType), m_viewContext(c), m_elements(0)
{
   m_elements = new REveCompound("ProxyProduct", "", false);
   m_elements->IncDenyDestroy();
}

//______________________________________________________________________________


REveDataProxyBuilderBase::REveDataProxyBuilderBase():
   m_collection(nullptr),
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

void REveDataProxyBuilderBase::Build()
{
   if (m_collection)
   {
      // printf("REveDataProxyBuilderBase::Build %p %s products %lu\n", m_collection, m_collection->GetCName(), m_products.size());
      try
      {
         auto itemSize = m_collection->GetNItems(); //cashed

         Clean();

         if (!m_collection->GetRnrSelf())
            return;

         for (auto &pp: m_products)
         {
            REveElement* product = pp->m_elements;
            auto oldSize = product->NumChildren();
            if (HaveSingleProduct())
            {
               Build(m_collection, product, pp->m_viewContext);
            }
            else
            {
               BuildViewType(m_collection, product, pp->m_viewType, pp->m_viewContext);
            }

            // Project all children of current product.
            // If product is not registered into any projection-manager,
            // this does nothing.
            REveProjectable* pableProduct = dynamic_cast<REveProjectable*>(product);
            if (pableProduct->HasProjecteds())
            {
               // loop projected holders
               for (auto &projectedProduct: pableProduct->RefProjecteds())
               {
                  REveProjectionManager *pmgr = projectedProduct->GetManager();
                  Float_t oldDepth = pmgr->GetCurrentDepth();
                  pmgr->SetCurrentDepth(m_layer);
                  Int_t cnt = 0;
                  REveElement *projectedProductAsElement = projectedProduct->GetProjectedAsElement();
                  // printf("projectedProduct children %d, product children %d\n", projectedProductAsElement->NumChildren(), product->NumChildren());
                  auto parentIt = projectedProductAsElement->RefChildren().begin();
                  for (auto &holder: product->RefChildren())
                  {
                     // reused projected holder
                     if (cnt < oldSize) 
                     {
                        pmgr->SubImportChildren(holder, *parentIt);
                        ++parentIt;
                     } else if (cnt < itemSize) {
                        // new product holder
                        pmgr->SubImportElements(holder, projectedProductAsElement);
                     }
                     else {
                        break;
                     }
                     ++cnt;
                  }
                  pmgr->SetCurrentDepth(oldDepth);
               }
            }
         }
      }
      catch (const std::runtime_error &iException) {
         R__LOG_ERROR(REveLog()) << "Caught exception in build function for item " << m_collection->GetName() << ":\n"
                 << iException.what() << std::endl;
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
REveDataProxyBuilderBase::BuildViewType(const REveDataCollection*, REveElement*, const std::string&, const REveViewContext*)
{
   assert("virtual BuildViewType(const FWEventItem*, TEveElementList*, FWViewType::EType, const FWViewContext*) not implemented by inherited class");
}

//------------------------------------------------------------------------------

REveElement*
REveDataProxyBuilderBase::CreateProduct( const std::string& viewType, const REveViewContext* viewContext)
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
      product->m_elements->SetName(TString::Format("product %s viewtype %s", m_collection->GetCName(), viewType.c_str()).Data());
   }
   return product->m_elements;
}

//______________________________________________________________________________

void
REveDataProxyBuilderBase::LocalModelChanges(int, REveElement*, const REveViewContext*)
{
   // Nothing to be done in base class.
   // Visibility, main color and main transparency are handled automatically throught compound.
}
//------------------------------------------------------------------------------

void
REveDataProxyBuilderBase::FillImpliedSelected( REveElement::Set_t& impSet)
{
   for (auto &prod: m_products)
   {
      FillImpliedSelected(impSet, prod);
   }
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
REveDataProxyBuilderBase::SetupAddElement(REveElement* el, REveElement* parent, bool color)
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
REveDataProxyBuilderBase::SetupElement(REveElement* el, bool color)
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

//------------------------------------------------------------------------------

void REveDataProxyBuilderBase::ScaleChanged()
{
   for (auto &prod : m_products) {
      ScaleProduct(prod->m_elements, prod->m_viewType);
   }
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

bool REveDataProxyBuilderBase::VisibilityModelChanges(int, REveElement *, const std::string&, const REveViewContext *)
{
   return false;
}

void REveDataProxyBuilderBase::SetHaveAWindow(bool iHaveAWindow)
{
   m_haveWindow = iHaveAWindow;
}
