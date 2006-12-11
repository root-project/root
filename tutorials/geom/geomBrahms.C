//script drawing a detector geometry (here BRAHMS)
//by default the geometry is drawn using the GL viewer
//Using the TBrowser, you can select other components
//if the file containing the geometry is not found in the local
//directory, it is automatically read from the ROOT web site.
// Author: Rene Brun
   
void geomBrahms() {
   const char *fname = "brahms.root";
   if (!gSystem->AccessPathName(fname)) {
      TGeoManager::Import(fname);
   } else {
      printf("accessing %s file from http://root.cern.ch/files\n",fname);
      TGeoManager::Import(Form("http://root.cern.ch/files/%s",fname));
   }
   gGeoManager->GetVolume("CAVE")->Draw("ogl");
   new TBrowser;  
}
