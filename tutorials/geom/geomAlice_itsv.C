//script drawing a detector geometry (here ITSV from Alice)
//by default the geometry is drawn using the GL viewer
//Using the TBrowser, you can select other components
//if the file containing the geometry is not found in the local
//directory, it is automatically read from the ROOT web site.
// Author: Rene Brun

void geomAlice_itsv() {
   TGeoManager::Import("http://root.cern.ch/files/alice2.root");
   gGeoManager->DefaultColors();
   gGeoManager->GetVolume("ITSV")->Draw("ogl");
   new TBrowser;
}
