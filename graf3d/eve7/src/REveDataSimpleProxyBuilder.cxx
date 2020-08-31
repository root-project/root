#include <ROOT/REveDataSimpleProxyBuilder.hxx>

// user include files
#include <ROOT/REveDataCollection.hxx>
#include <ROOT/REveCompound.hxx>
#include <assert.h>

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

REveDataSimpleProxyBuilder::REveDataSimpleProxyBuilder(const std::string &type) : REveDataProxyBuilderBase(type)
{
}


REveDataSimpleProxyBuilder::~REveDataSimpleProxyBuilder()
{
}

void
REveDataSimpleProxyBuilder::Clean()
{
   for (auto &p: m_products)
   {
      if (p->m_elements)
      {
         REveElement *elms = p->m_elements;
         for (auto &c: elms->RefChildren())
            c->DestroyElements();
      }
   }

   CleanLocal();
}

//______________________________________________________________________________

void
REveDataSimpleProxyBuilder::Build(const REveDataCollection *collection,
                                  REveElement* product, const REveViewContext* vc)
{
   auto size = collection->GetNItems();
   auto pIdx = product->RefChildren().begin();
   for (int index = 0; index < size; ++index)
   {
      const REveDataItem* di = Collection()->GetDataItem(index);
      REveElement *itemHolder = nullptr;

      if (index <  product->NumChildren())
      {
         itemHolder = *pIdx;
         itemHolder->SetRnrSelfChildren(true, true);
         ++pIdx;
      }
      else
      {
         itemHolder = CreateCompound(true, true);
         itemHolder->SetMainColor(collection->GetMainColor());
         itemHolder->SetName(Form("%s %d", collection->GetCName(), index));

         product->AddElement(itemHolder);
      }


      if (di->GetRnrSelf() && !di->GetFiltered())
      {
         Build(collection->GetDataPtr(index), index, itemHolder, vc);
      }
   }
}

void
REveDataSimpleProxyBuilder::BuildViewType(const REveDataCollection* collection,
                                          REveElement* product, std::string viewType, const REveViewContext* vc)
{
   auto size = collection->GetNItems();
   auto pIdx = product->RefChildren().begin();
   for (int index = 0; index < size; ++index)
   {
      auto di = Collection()->GetDataItem(index);
      REveElement* itemHolder = nullptr;

      if (index <  product->NumChildren())
      {
         itemHolder = *pIdx;
         itemHolder->SetRnrSelfChildren(true, true);
         ++pIdx;
      }
      else
      {
         itemHolder = CreateCompound(true, true);
         itemHolder->SetMainColor(collection->GetMainColor());
         itemHolder->SetName(Form("%s %d", collection->GetCName(), index));

         product->AddElement(itemHolder);
      }


      if (di->GetRnrSelf() && !di->GetFiltered())
      {
         BuildViewType(collection->GetDataPtr(index), index, itemHolder, viewType, vc);
      }
      }
}

//______________________________________________________________________________

namespace
{
   void applyColorAttrToChildren(REveElement* p) {
      for (auto &it: p->RefChildren())
      {
         REveElement* c = it;
         if (c->GetMainColor() != p->GetMainColor())
         {
            c->SetMainColor(p->GetMainColor());
            printf("apply color %d to %s\n", p->GetMainColor(), c->GetCName());
         }
         applyColorAttrToChildren(c);
      }
   }
}

void
REveDataSimpleProxyBuilder::ModelChanges(const REveDataCollection::Ids_t& iIds, Product* p)
{
   printf("REveDataSimple ProxyBuilderBase::ModelChanges >>>>> (%p)  %s \n", (void*)this, Collection()->GetCName());
   REveElement* elms = p->m_elements;
   assert(Collection() && static_cast<int>(Collection()->GetNItems()) <= elms->NumChildren() && "can not use default modelChanges implementation");

   printf("N indices %d \n",(int)iIds.size());
   for (auto itemIdx: iIds)
   {
      const REveDataItem* item = Collection()->GetDataItem(itemIdx);

      // printf("Edit compound for item index %d \n", itemIdx);
      // imitate FWInteractionList::modelChanges
      auto itElement = elms->RefChildren().begin();
      std::advance(itElement, itemIdx);
      REveElement* comp = *itElement;
      bool visible = (!item->GetFiltered()) && item->GetRnrSelf();
      comp->SetRnrSelf(visible);
      comp->SetRnrChildren(visible);

      // printf("comapre %d %d\n", item->GetMainColor(), comp->GetMainColor());
      if (item->GetMainColor() != comp->GetMainColor()) {
         //printf("ffffffffffffffffffffffff set color to comp \n");
         comp->SetMainColor(item->GetMainColor());

      }
      applyColorAttrToChildren(comp);

      if (VisibilityModelChanges(itemIdx, comp, p->m_viewContext))
      {
         elms->ProjectChild(comp);
         // printf("---REveDataProxyBuilderBase project child\n");
      }
      else
      {
         LocalModelChanges(itemIdx, comp, p->m_viewContext);
      }
   }
}

//______________________________________________________________________________

bool
REveDataSimpleProxyBuilder::VisibilityModelChanges(int idx, REveElement* iCompound, const REveViewContext* vc)
{
   const REveDataItem* item = Collection()->GetDataItem(idx);
   bool returnValue = false;
   if (item->GetRnrSelf() && iCompound->NumChildren()==0)
   {
      printf("REveDataSimpleProxyBuilder::VisibilityModelChanges BUILD %d \n", idx);
      Build(Collection()->GetDataPtr(idx), idx, iCompound, vc);
      returnValue=true;
   }
   return returnValue;
}
