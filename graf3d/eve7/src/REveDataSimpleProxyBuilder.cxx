#include <ROOT/REveDataSimpleProxyBuilder.hxx>

// user include files
#include <ROOT/REveDataClasses.hxx>
#include <ROOT/REveCompound.hxx>

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
         SetupAddElement(itemHolder, product, true);
         itemHolder->SetName(Form("compound %d", index));

      }
      auto di = Collection()->GetDataItem(index);
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
         SetupAddElement(itemHolder, product, true);
         itemHolder->SetName(Form("compound %d", index));

      }
      auto di = Collection()->GetDataItem(index);
      if (di->GetRnrSelf() && !di->GetFiltered())
      {
         BuildViewType(collection->GetDataPtr(index), index, itemHolder, viewType, vc);
      }
   }
}

//______________________________________________________________________________

bool
REveDataSimpleProxyBuilder::VisibilityModelChanges(int idx, REveElement* iCompound, const REveViewContext* vc)
{
   REveDataItem* item = Collection()->GetDataItem(idx);
   bool returnValue = false;
   if (item->GetRnrSelf() && iCompound->NumChildren()==0)
   {
      printf("REveDataSimpleProxyBuilder::VisibilityModelChanges BUILD %d \n", idx);
      Build(Collection()->GetDataPtr(idx), idx, iCompound, vc);
      returnValue=true;
   }
   return returnValue;
}
