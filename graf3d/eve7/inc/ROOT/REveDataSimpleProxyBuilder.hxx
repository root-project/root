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

#include <ROOT/REveDataProxyBuilderBase.hxx>

namespace ROOT {
namespace Experimental {

class REveDataCollection;
class REveElement;

class REveDataSimpleProxyBuilder : public REveDataProxyBuilderBase
{
public:
   REveDataSimpleProxyBuilder(const std::string &type);
   virtual ~REveDataSimpleProxyBuilder();

protected:
   void Build(const REveDataCollection* iCollection, REveElement* product, const REveViewContext*) override;
   void BuildViewType(const REveDataCollection* iCollection, REveElement* product, std::string viewType, const REveViewContext*) override;

   //called once for each collection in collection, the void* points to the
   // object properly offset in memory
   virtual void Build(const void* data, int index, REveElement* iCollectionHolder, const REveViewContext*) = 0;
   virtual void BuildViewType(const void* data, int index, REveElement* iCollectionHolder, std::string viewType, const REveViewContext*) = 0;

   void Clean() override;

private:
   REveDataSimpleProxyBuilder(const REveDataSimpleProxyBuilder&); // stop default

   const REveDataSimpleProxyBuilder& operator=(const REveDataSimpleProxyBuilder&); // stop default

   bool VisibilityModelChanges(int idx, REveElement*, const REveViewContext*) override;

};

} // namespace Experimental
} // namespace ROOT

#endif
