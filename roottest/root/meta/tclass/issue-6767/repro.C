#include "repro.h"

int repro()
{
   gErrorIgnoreLevel = kPrint; // Get all the output even if .rootrc or env says otherwise

   // Check that our setup is reproducing the environment that caused the origin problem
   const char *name = "edm_test::FwdPtr<CaloTowerTest>";
   auto cl = TClass::GetClass(name);
   if (cl) {
      Error("GenCollectionProxy reproducer", "The class for %s was unexpectedly found", name);
      return 1;
   }
   TypeInfo_t *ti = gInterpreter->TypeInfo_Factory();
   gInterpreter->TypeInfo_Init(ti, name);
   if (!gInterpreter->TypeInfo_IsValid(ti)) {
      Error("GenCollectionProxy reproducer", "The TypeInfo_t for %s is invalid", name);
      return 2;
   }
   Long_t prop = gInterpreter->TypeInfo_Property(ti);

   if (! (prop&kIsClass) ) {
      Error("GenCollectionProxy reproducer", "The TypeInfo_t for %s is not marked as a class", name);
      return 3;
   }  else {
      Info("GenCollectionProxy reproducer", "R__ASSERT(! (prop&kIsClass) && \"Impossible code path\") would have triggered");
   }

   // Now check that we fixed the problem.
   cl = TClass::GetClass("pair<void*,short>");
   if (!cl) {
      Error("GenCollectionProxy reproducer", "Can't materialize the TClass for pair<void*,short>");
      return 4;
   }

   // Now check that we fixed the problem.
   cl = TClass::GetClass("pair<CaloTowerTest,short>");
   if (cl) {
      Error("GenCollectionProxy reproducer", "Was unexpectedly able to materialize the TClass for pair<CaloTowerTest,short>");
      return 5;
   }

   // Was crashing because of the pattern above ... and because the previous test failed.
   cl = TClass::GetClass("multimap<CaloTowerTest, short>");
   if (!cl) {
      Info("GenCollectionProxy reproducer", "Can't materialize the TClass for multimap<CaloTowerTest, short>");
      // Let this pass has it might actually be an improvement.
      // return 6;
   }
   if (cl && cl->GetCollectionProxy()) {
      Error("GenCollectionProxy reproducer", "Was unexpectedly able to materialize the CollectionProxy for multimap<CaloTowerTest, short>");
      return 7;
   }

   // Was crashing because of the pattern above being used in TGenCollectionProxy's Value.
   cl = TClass::GetClass("vector<edm_test::FwdPtr<CaloTowerTest>>");
   if (!cl) {
      Info ("GenCollectionProxy reproducer", "Can't materialize the TClass for vector<edm_test::FwdPtr<CaloTowerTest>>");
      // Let this pass has it might actually be an improvement.
      // return 8;
   }
   if (cl && cl->GetCollectionProxy()) {
      Error("GenCollectionProxy reproducer", "Was unexpectedly able to materialize the CollectionProxy for vector<edm_test::FwdPtr<CaloTowerTest>>");
      return 9;
   }


   auto oldcl = TClass::GetClass("vector<CaloTowerTest*>");
   if (!oldcl) {
      Error("Reload reproducer", "Can not get the vector<CaloTowerTest*> TClass");
      return 20;
   }

   auto oldinfo = oldcl->GetStreamerInfo();
   if (!oldinfo) {
      Error("Reload reproducer", "Can not get the vector<CaloTowerTest*> TStreamerInfo");
      return 21;
   }

   auto el = oldinfo->GetElement(0);
   if (!el) {
      Error("Reload reproducer", "Inconsistency for vector<CaloTowerTest*> StreamerInfo which has no elements");
      return 22;
   }

   if (oldcl != el->GetClassPointer()) {
      Error("Reload reproducer", "Inconsistency for vector<CaloTowerTest*> StreamerInfo, the This element is not pointing to the TClass");
      return 23;
   }

   cl = TClass::GetClass("pair<int, list<CaloTowerTest*>>");
   if (!cl) {
      Error("Reload reproducer", "Can not get the pair<int, list<CaloTowerTest*>> TClass)");
      return 24;
   }

   ///
   /// Loading the library
   ///

   if (0 != gSystem->Load("libRepro")) { // libCaloTowerCollections")) {
      Error("Reload reproducer", "Can not load libCaloTowerCollections");
      return 40;
   }

   cl = TClass::GetClass("vector<CaloTowerTest*>");
   if (!cl) {
      Error("Reload reproducer", "Can not get the vector<CaloTowerTest*> TClass after loading the library");
      return 41;
   }

   auto newinfo = cl->GetStreamerInfo();
   if (!newinfo) {
      Error("Reload reproducer", "Can not get the vector<CaloTowerTest*> TStreamerInfo after loading the library");
      return 42;
   }

   el = newinfo->GetElement(0);
   if (!el) {
      Error("Reload reproducer", "Inconsistency for vector<CaloTowerTest*> StreamerInfo which has no elements after loading the library");
      return 43;
   }

   if (cl != el->GetClassPointer()) {
      Error("Reload reproducer", "Inconsistency for vector<CaloTowerTest*> StreamerInfo, the This element is not pointing to the TClass after loading the library");
      if (oldcl != el->GetClassPointer()) {
         Error("Reload reproducer", "Inconsistency for vector<CaloTowerTest*> StreamerInfo, the This element is still pointing to the old TClass after loading the library");

      }
      return 44;
   }

   cl = TClass::GetClass("list<CaloTowerTest*>");
   if (!cl) {
      Error("Reload reproducer", "Can not get the list<CaloTowerTest*> TClass)");
      return 45;
   }

   auto listinfo = cl->GetStreamerInfo();
   if (!listinfo) {
      Error("Reload reproducer", "Can not get the list<CaloTowerTest*> TStreamerInfo after loading the library");
      return 46;
   }

   TFile tmpfile("empty.root", "RECREATE");
   newinfo->ForceWriteInfo(&tmpfile);
   listinfo->ForceWriteInfo(&tmpfile);
   tmpfile.Close();
   gSystem->Unlink("empty.root");

#if 0
   TClass::GetClass of a pair<something, vector<somethingelse>>
   and new code induced the creation of the emulated one
   so equi to TClass::GetClass("vector<somethingelse")->GetStreamerInfo().code

   Then load library (now we have new TClass with old StreamerInfo with stale ptr)

   newclass->GetStreamerInfo()->GetElement(0)->GetClassPointer() == newclass.


   TFile tmpfile("empty.root");
   criticalInfo->ForceWriteInfo(file);
#endif

   return 0;
}

