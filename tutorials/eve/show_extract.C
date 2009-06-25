// Helper script for showing of extracted / simplified geometries.
// By default shows a simplified ALICE geometry.

void show_extract(const char* file="http://root.cern.ch/files/alice_ESDgeometry.root")
{
  TEveManager::Create();

  TFile::Open(file);

  TIter next(gDirectory->GetListOfKeys());
  TKey* key;
  TString xxx("TEveGeoShapeExtract");

  while ((key = (TKey*) next()))
  {
    if (xxx == key->GetClassName())
    {
      TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) key->ReadObj();
      TEveGeoShape* gs = TEveGeoShape::ImportShapeExtract(gse, 0);
      gEve->AddGlobalElement(gs);
    }
  }

  gEve->Redraw3D(kTRUE);
}
