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
   for (Product_it i = m_products.begin(); i != m_products.end(); ++i)
   {
      if ((*i)->m_elements)
      {
         REveElement* elms = (*i)->m_elements;
         for (REveElement::List_i it = elms->BeginChildren(); it != elms->EndChildren(); ++it)
            (*it)->DestroyElements();
      }
   }

   CleanLocal();
}

//______________________________________________________________________________

void
REveDataSimpleProxyBuilder::Build(const REveDataCollection* collection,
                                  REveElement* product, const REveViewContext* vc)
{
   size_t size = collection->GetNItems();
   REveElement::List_i pIdx = product->BeginChildren();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      REveElement* itemHolder = 0;
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
         Build(collection->GetDataPtr(index), itemHolder, vc);
      }
   }
}

void
REveDataSimpleProxyBuilder::BuildViewType(const REveDataCollection* collection,
                                          REveElement* product, std::string viewType, const REveViewContext* vc)
{
   size_t size = collection->GetNItems();
   REveElement::List_i pIdx = product->BeginChildren();
   for (int index = 0; index < static_cast<int>(size); ++index)
   {
      REveElement* itemHolder = 0;
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
         BuildViewType(collection->GetDataPtr(index), itemHolder, viewType, vc);
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
      Build(Collection()->GetDataPtr(idx), iCompound, vc);
      returnValue=true;
   }
   return returnValue;
}
