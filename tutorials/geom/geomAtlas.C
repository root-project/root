//script drawing a detector geometry (here ATLAS)
//by default the geometry is drawn using the GL viewer
//Using the TBrowser, you can select other components
//if the file containing the geometry is not found in the local
//directory, it is automatically read from the ROOT web site.
// Author: Rene Brun

void geomAtlas() {
   TGeoManager::Import("http://root.cern.ch/files/atlas.root");
   //gGeoManager->DefaultColors();
   gGeoManager->SetMaxVisNodes(5000);
   //gGeoManager->SetVisLevel(4);
   gGeoManager->GetVolume("ATLS")->Draw("ogl");
   new TBrowser;
}
