// #include "/opt/build/root_builds/rootcling.debug/src/interpreter/cling/include/cling/Interpreter/LookupHelper.h"

void step2() 
{
   // TClass::GetClass("edm::Ref<edm::LazyGetter<SiStripCluster>,SiStripCluster,edm::FindValue<SiStripCluster> >");
   // This is too simple and leads to the auto parsing happening outside of the transaction.
   //TClass::GetClass("edm::Ref<Nothing<SiStripCluster> >");
   std::string output;
   gInterpreter->GetInterpreterTypeName("edm::Ref<Nothing<SiStripCluster> >",output);

#if 0
   const clang::Type *type = 0;
   const char *name = "edm::Ref<edm::LazyGetter<SiStripCluster>,SiStripCluster,edm::FindValue<SiStripCluster> >";
   name = "edm::Ref<Nothing<SiStripCluster> >";
   //name = "edm::Wrapper<Nothing<SiStripCluster> >";
   //name = "Nothing<SiStripCluster>";
//   cling::runtime::gCling->getLookupHelper().findScope("edm::Ref<edm::LazyGetter<SiStripCluster>,SiStripCluster,edm::FindValue<SiStripCluster> >",cling::LookupHelper::WithDiagnostics,&type,true);
   cling::runtime::gCling->getLookupHelper().findScope(name,cling::LookupHelper::WithDiagnostics,&type,true);
#endif

}

