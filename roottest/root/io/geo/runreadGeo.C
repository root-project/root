{
  TGeoManager::Import("aleph.root");
  TGeoManager::Import("alice.root");
  TGeoManager::Import("ams.root");
  TGeoManager::Import("atlas.root");
  TGeoManager::Import("babar.root");
  TGeoManager::Import("brahms.root"); 
  TGeoManager::Import("btev.root"); 
  TGeoManager::Import("cdf.root");
  TGeoManager::Import("cms.root");
  TGeoManager::Import("d0.root");
  TGeoManager::Import("e907.root"); 
  TGeoManager::Import("gem.root"); 
  TGeoManager::Import("hades.root");
  TGeoManager::Import("lhcbfull.root");
  TGeoManager::Import("phenix.root"); 
  TGeoManager::Import("phobos.root"); 
  TGeoManager::Import("star.root");
  TGeoManager::Import("tesla.root");
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   0; // Insure success of the test (for Makefile)
#endif
}
