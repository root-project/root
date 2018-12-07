#include <ROOT/REveTableProxyBuilder.hxx>
#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveViewContext.hxx>
#include <ROOT/REveDataClasses.hxx>



using namespace ROOT::Experimental;

void REveTableProxyBuilder::Build(const REveDataCollection* collection, REveElement* product, const REveViewContext* context)
{
   REveTableViewInfo* info = context->GetTableViewInfo();
   if (collection->GetElementId() != info->GetDisplayedCollection())
      return;

   printf("REveTableProxyBuilder::Build() body for %s (%p, %p)\n",collection->GetCName(), collection, Collection() );
   auto table = new REveDataTable("testTable");
   table->SetCollection(collection);
   product->AddElement(table);

   auto tableEntries =  context->GetTableViewInfo()->RefTableEntries(collection->GetName());

   for (const REveTableEntry& spec : tableEntries) {
      auto c = new REveDataColumn(spec.fName.c_str());
      table->AddElement(c);
      std::string exp  = "i." + spec.fExpression + "()";
      c->SetExpressionAndType(exp.c_str(), spec.fType);
      c->SetPrecision(spec.fPrecision);
   }

   m_table = table;
}

void REveTableProxyBuilder::ModelChanges(const REveDataCollection::Ids_t&, REveDataProxyBuilderBase::Product*)
{
   m_table->StampObjProps();
}

void REveTableProxyBuilder::DisplayedCollectionChanged(ElementId_t id) {
   printf("displayed collection changed %d (%p) \n", id, Collection());
   Build();
}
