{
  TGeoManager::Import("alice.root");
  TGeoManager::Import("atlas.root");
  TGeoManager::Import("cms.root");
  TGeoManager::Import("hades.root");
  TGeoManager::Import("lhcbfull.root");
  p = TGeoManager::Import("lhcbnobool.root");
  (p==0);

}
