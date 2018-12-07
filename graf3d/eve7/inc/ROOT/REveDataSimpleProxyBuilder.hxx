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
   REveDataSimpleProxyBuilder(std::string type);
   virtual ~REveDataSimpleProxyBuilder();

protected:
   using REveDataProxyBuilderBase::Build;
   virtual void Build(const REveDataCollection* iCollection, REveElement* product, const REveViewContext*);
   virtual void BuildViewType(const REveDataCollection* iCollection, REveElement* product, std::string viewType, const REveViewContext*);

   //called once for each collection in collection, the void* points to the
   // object properly offset in memory
   virtual void Build(const void* data, REveElement* iCollectionHolder, const REveViewContext*) = 0;
   virtual void BuildViewType(const void* data, REveElement* iCollectionHolder, std::string viewType, const REveViewContext*) = 0;

   virtual void Clean();

private:
   REveDataSimpleProxyBuilder(const REveDataSimpleProxyBuilder&); // stop default

   const REveDataSimpleProxyBuilder& operator=(const REveDataSimpleProxyBuilder&); // stop default

   virtual bool VisibilityModelChanges(int idx, REveElement*,  const REveViewContext*);


   // ---------- member data --------------------------------

   ClassDef(REveDataSimpleProxyBuilder, 0);
};

} // namespace Experimental
} // namespace ROOT

#endif
