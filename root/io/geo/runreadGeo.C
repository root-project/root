{
  TGeoManager::Import("aleph.root");
  TGeoManager::Import("ams.root");
  TGeoManager::Import("alice.root");
  TGeoManager::Import("atlas.root");
  TGeoManager::Import("brahms.root"); 
  TGeoManager::Import("btev.root"); 
  TGeoManager::Import("cdf.root");
  TGeoManager::Import("cms.root");
  TGeoManager::Import("d0.root");
  TGeoManager::Import("e907.root"); 
  TGeoManager::Import("gem.root"); 
  if (0) TGeoManager::Import("hades.root");
  TGeoManager::Import("lhcbfull.root");
  if (0) TGeoManager::Import("lhcbnobool.root");
  if (0) TGeoManager::Import("phenix.root"); 
  TGeoManager::Import("phobos.root"); 
  TGeoManager::Import("tesla.root");
  p=TGeoManager::Import("star.root");
  (p==0);
}
