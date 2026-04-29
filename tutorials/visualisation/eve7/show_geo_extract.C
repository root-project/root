/// \file
/// \ingroup tutorial_eve7
/// Helper script for showing of extracted / simplified geometries.
/// The test macro how to create the shapes is in file write_geo_extract.C
/// \macro_code
///
/// \author Matevz Tadel

void show_geo_extract(const char *file = "testShapeExtract.root")
{
   auto eveMng = ROOT::Experimental::REveManager::Create();
   // eveMng->AllowMultipleRemoteConnections(false, false);

   TFile::Open(file);
   TIter next(gDirectory->GetListOfKeys());
   TKey *key;
   TString seName("ROOT::Experimental::REveGeoShapeExtract");

   while ((key = (TKey *)next())) {
      std::cout << "calss name " << key->GetClassName() << "\n";
      if (seName == key->GetClassName()) {
         std::cout << "got the extract name " << key->GetClassName() << "\n";
         auto gse = (ROOT::Experimental::REveGeoShapeExtract *)key->ReadObj();
         auto gs = ROOT::Experimental::REveGeoShape::ImportShapeExtract(gse, 0);
         eveMng->AddGlobalElement(gs);
      }
   }

   eveMng->Show();
}
