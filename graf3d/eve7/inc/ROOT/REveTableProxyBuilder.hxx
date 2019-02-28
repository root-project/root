#ifndef ROOT7_REveTableProxyBuilder
#define ROOT7_REveTableProxyBuilder

#include <ROOT/REveDataProxyBuilderBase.hxx>

namespace ROOT {
namespace Experimental {

class REveDataTable;
class REveTableInfo;

class REveTableProxyBuilder : public REveDataProxyBuilderBase
{
private:
   //TableHandle::TableEntries m_specs;
   REveDataTable* fTable; // cached

public:
   REveTableProxyBuilder() : REveDataProxyBuilderBase("Table"), fTable(0) {}
   virtual bool WillHandleInteraction() const { return true; }

   using REveDataProxyBuilderBase::ModelChanges;
   virtual void ModelChanges(const REveDataCollection::Ids_t&, REveDataProxyBuilderBase::Product* p);

   using REveDataProxyBuilderBase::Build;
   virtual void Build(const REveDataCollection* collection, REveElement* product, const REveViewContext* context);

   void DisplayedCollectionChanged(ElementId_t);


   virtual void CleanLocal();
};
}
}

#endif
