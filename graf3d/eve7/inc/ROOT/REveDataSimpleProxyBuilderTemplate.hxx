// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveDataProxySimpleBuilderTemplate
#define ROOT7_REveDataProxySimpleBuilderTemplate

#include <ROOT/REveDataSimpleProxyBuilder.hxx>


namespace ROOT {
namespace Experimental {

template <typename T>
class REveDataSimpleProxyBuilderTemplate : public REveDataSimpleProxyBuilder {

public:
   REveDataSimpleProxyBuilderTemplate() : REveDataSimpleProxyBuilder()
   {
   }

protected:
   using REveDataSimpleProxyBuilder::BuildItem;
   void BuildItem(const void *iData, int index, REveElement *itemHolder, const REveViewContext *context) override
   {
      if(iData) {
         BuildItem(*reinterpret_cast<const T*> (iData), index, itemHolder, context);
      }
   }

   virtual void BuildItem(const T & /*iData*/, int /*index*/, REveElement * /*itemHolder*/, const REveViewContext * /*context*/)
   {
      throw std::runtime_error("virtual Build(const T&, int, REveElement&, const REveViewContext*) not implemented by inherited class.");
   }

   using REveDataSimpleProxyBuilder::BuildItemViewType;
   void BuildItemViewType(const void *iData, int index, REveElement *itemHolder, const std::string& viewType, const REveViewContext *context) override
   {
      if(iData) {
         BuildItemViewType(*reinterpret_cast<const T*> (iData), index, itemHolder, viewType, context);
      }
   }

   virtual void BuildItemViewType(const T & /*iData*/, int /*index*/, REveElement * /*itemHolder*/, const std::string& /*viewType*/, const REveViewContext * /*context*/)
   {
      throw std::runtime_error("virtual BuildViewType(const T&, int, REveElement&, const REveViewContext*) not implemented by inherited class.");
   }

private:
   REveDataSimpleProxyBuilderTemplate(const REveDataSimpleProxyBuilderTemplate&); // stop default

   const REveDataSimpleProxyBuilderTemplate& operator=(const REveDataSimpleProxyBuilderTemplate&); // stop default
};

} // namespace Experimental
} // namespace ROOT

#endif
