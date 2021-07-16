// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel, 2019

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveDataProxySimpleBuilder
#define ROOT7_REveDataProxySimpleBuilder

#include<list>
#include <ROOT/REveDataProxyBuilderBase.hxx>

namespace ROOT {
namespace Experimental {

class REveDataCollection;
class REveElement;

class REveCollectionCompound : public REveCompound // ?? Should this be in as REveDataSimpleProxyBuilder.hxx ?????
{
private:
   REveDataCollection *fCollection{nullptr};   
public:
   REveCollectionCompound(REveDataCollection *c);
   virtual ~REveCollectionCompound();
   virtual REveElement *GetSelectionMaster() override;
};

//
//____________________________________________________________________________________
//
class REveDataSimpleProxyBuilder : public REveDataProxyBuilderBase
{

public:
   REveDataSimpleProxyBuilder();
   virtual ~REveDataSimpleProxyBuilder();

   struct SPBProduct {
      std::map<int, REveCollectionCompound*> map;
      int lastChildIdx{0};
   }; 
   
   typedef  std::map<REveElement*, std::unique_ptr<SPBProduct*> > EProductMap_t;

protected:
   void Build(const REveDataCollection* iCollection, REveElement* product, const REveViewContext*) override;

   void BuildViewType(const REveDataCollection* iCollection, REveElement* product, const std::string& viewType, const REveViewContext*) override;

   // Called once for every item in collection, the void* points to the
   // item properly offset in memory.
   virtual void Build(const void* data, int index, REveElement* iCollectionHolder, const REveViewContext*) = 0;
   virtual void BuildViewType(const void* data, int index, REveElement* iCollectionHolder, const std::string& viewType, const REveViewContext*) = 0;

   void ModelChanges(const REveDataCollection::Ids_t& iIds, Product* p) override;
   void FillImpliedSelected(REveElement::Set_t& impSet, Product* p) override;
   void Clean() override; // Utility
   REveCollectionCompound* CreateCompound(bool set_color=true, bool propagate_color_to_all_children=false);

   //int GetItemIdxForCompound() const;
   bool VisibilityModelChanges(int idx, REveElement*, const std::string& viewType, const REveViewContext*) override;

   std::map<REveElement*, SPBProduct*> fProductMap;
   REveCompound* GetHolder(REveElement *product, int idx);

private:
   REveDataSimpleProxyBuilder(const REveDataSimpleProxyBuilder&); // stop default

   const REveDataSimpleProxyBuilder& operator=(const REveDataSimpleProxyBuilder&); // stop default
};
//==============================================================================


} // namespace Experimental
} // namespace ROOT

#endif
