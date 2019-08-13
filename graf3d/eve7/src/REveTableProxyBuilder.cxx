#include <ROOT/REveTableProxyBuilder.hxx>
#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveDataClasses.hxx>
#include <ROOT/REveManager.hxx>



using namespace ROOT::Experimental;

REveTableProxyBuilder::REveTableProxyBuilder() : REveDataProxyBuilderBase("Table"), fTable(nullptr)
{
    fTable = new REveDataTable("ProxyTable");
}

REveTableProxyBuilder::~REveTableProxyBuilder()
{
   fTable->Destroy();
   fTable = nullptr;
}

// reuse table product
void REveTableProxyBuilder::Clean()
{
}

void REveTableProxyBuilder::Build(const REveDataCollection* collection, REveElement* product, const REveViewContext* context)
{
   REveTableViewInfo* info = context->GetTableViewInfo();
   if (info->GetDisplayedCollection() != Collection()->GetElementId())
   {
      return;
   }

   if (product->NumChildren() == 0) {
      product->AddElement(fTable);
   }

   // printf("-----REveTableProxyBuilder::Build() body for %s (%p, %p)\n",collection->GetCName(), collection, Collection() );

   if (info->GetConfigChanged() || fTable->NumChildren() == 0) {
      fTable->DestroyElements();
      auto tableEntries =  context->GetTableViewInfo()->RefTableEntries(collection->GetName());
      for (const REveTableEntry& spec : tableEntries) {
         auto c = new REveDataColumn(spec.fName.c_str());
         fTable->AddElement(c);
         using namespace std::string_literals;
         std::string exp  = "i."s + spec.fExpression + "()"s;
         c->SetExpressionAndType(exp.c_str(), spec.fType);
         c->SetPrecision(spec.fPrecision);
      }
   }
   fTable->StampObjProps();
}



void REveTableProxyBuilder::SetCollection(REveDataCollection* collection)
{
   REveDataProxyBuilderBase::SetCollection(collection);
   fTable->SetCollection(collection);
}

void REveTableProxyBuilder::ModelChanges(const REveDataCollection::Ids_t&, REveDataProxyBuilderBase::Product*)
{
   // printf("REveTableProxyBuilder::ModelChanges\n");
   if (fTable) fTable->StampObjProps();
}

void REveTableProxyBuilder::ConfigChanged() {
   Build();
}
