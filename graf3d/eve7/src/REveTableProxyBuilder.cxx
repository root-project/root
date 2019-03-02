#include <ROOT/REveTableProxyBuilder.hxx>
#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveDataClasses.hxx>
#include <ROOT/REveManager.hxx>



using namespace ROOT::Experimental;

void REveTableProxyBuilder::Build(const REveDataCollection* collection, REveElement* product, const REveViewContext* context)
{
   REveTableViewInfo* info = context->GetTableViewInfo();

   // printf("-----REveTableProxyBuilder::Build() body for %s (%p, %p)\n",collection->GetCName(), collection, Collection() );
   auto table = new REveDataTable("testTable");
   table->SetCollection(collection);
   product->AddElement(table);
   product->SetName("product for " + collection->GetName());
   auto tableEntries =  context->GetTableViewInfo()->RefTableEntries(collection->GetName());

   for (const REveTableEntry& spec : tableEntries) {
      auto c = new REveDataColumn(spec.fName.c_str());
      table->AddElement(c);
      std::string exp  = "i." + spec.fExpression + "()";
      c->SetExpressionAndType(exp.c_str(), spec.fType);
      c->SetPrecision(spec.fPrecision);
   }

   info->SetTableId(table->GetElementId());
   fTable = table;
}

void REveTableProxyBuilder::CleanLocal()
{
   // all product elements are destroyed in Clean(), here is just a reset of cached variable
   fTable = 0;
}

void REveTableProxyBuilder::ModelChanges(const REveDataCollection::Ids_t&, REveDataProxyBuilderBase::Product*)
{
   // printf("REveTableProxyBuilder::ModelChanges\n");
   if (fTable) fTable->StampObjProps();
}

void REveTableProxyBuilder::DisplayedCollectionChanged(ElementId_t id) {
   // printf("-- displayed collection changed %d (%p) \n", id, Collection());
   REveDataCollection* c = dynamic_cast<REveDataCollection*>(gEve->FindElementById(id));
   SetCollection(c);
   Build();
}
