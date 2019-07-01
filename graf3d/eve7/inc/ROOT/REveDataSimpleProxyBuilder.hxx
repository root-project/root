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
   using REveDataProxyBuilderBase::Build;
   virtual void Build(const REveDataCollection* iCollection, REveElement* product, const REveViewContext*);
   virtual void BuildViewType(const REveDataCollection* iCollection, REveElement* product, std::string viewType, const REveViewContext*);

   //called once for each collection in collection, the void* points to the
   // object properly offset in memory
   virtual void Build(const void* data, unsigned int index, REveElement* iCollectionHolder, const REveViewContext*) = 0;
   virtual void BuildViewType(const void* data, unsigned int index, REveElement* iCollectionHolder, std::string viewType, const REveViewContext*) = 0;

   virtual void Clean();

private:
   REveDataSimpleProxyBuilder(const REveDataSimpleProxyBuilder&); // stop default

   const REveDataSimpleProxyBuilder& operator=(const REveDataSimpleProxyBuilder&); // stop default

   virtual bool VisibilityModelChanges(int idx, REveElement*,  const REveViewContext*);

};

} // namespace Experimental
} // namespace ROOT

#endif
