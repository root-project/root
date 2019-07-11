#ifndef ROOT7_REveDataProxySimpleBuilderTemplate
#define ROOT7_REveDataProxySimpleBuilderTemplate


#include <ROOT/REveDataSimpleProxyBuilder.hxx>


namespace ROOT {
namespace Experimental {

template <typename T>
class REveDataSimpleProxyBuilderTemplate : public REveDataSimpleProxyBuilder {

public:
   REveDataSimpleProxyBuilderTemplate() : REveDataSimpleProxyBuilder("GL")
   {
   }

protected:
   using REveDataSimpleProxyBuilder::Build;
   void Build(const void *iData, int index, REveElement *itemHolder, const REveViewContext *context) override
   {
      if(iData) {
         Build(*reinterpret_cast<const T*> (iData), index, itemHolder, context);
      }
   }

   virtual void Build(const T & /*iData*/, int /*index*/, REveElement * /*itemHolder*/, const REveViewContext * /*context*/)
   {
      throw std::runtime_error("virtual Build(const T&, int, REveElement&, const REveViewContext*) not implemented by inherited class.");
   }

   using REveDataSimpleProxyBuilder::BuildViewType;
   void BuildViewType(const void *iData, int index, REveElement *itemHolder, std::string viewType, const REveViewContext *context) override
   {
      if(iData) {
         BuildViewType(*reinterpret_cast<const T*> (iData), index, itemHolder, viewType, context);
      }
   }

   virtual void BuildViewType(const T & /*iData*/, int /*index*/, REveElement * /*itemHolder*/, std::string /*viewType*/, const REveViewContext * /*context*/)
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
