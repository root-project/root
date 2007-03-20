//script drawing a detector geometry (here ITSV from Alice)
//by default the geometry is drawn using the GL viewer
//Using the TBrowser, you can select other components
//if the file containing the geometry is not found in the local
//directory, it is automatically read from the ROOT web site.
// Author: Rene Brun
      
void geomAlice_itsv() {
   TGeoManager::Import("http://root.cern.ch/files/alice.root");
   //gGeoManager->DefaultColors();
   gGeoManager->GetVolume("IT56")->InvisibleAll();
   gGeoManager->GetVolume("I018")->InvisibleAll();
   gGeoManager->GetVolume("I090")->InvisibleAll();
   gGeoManager->GetVolume("I093")->InvisibleAll();
   gGeoManager->GetVolume("I099")->InvisibleAll();
   gGeoManager->GetVolume("I200")->InvisibleAll();
   gGeoManager->GetVolume("IC01")->InvisibleAll();
   gGeoManager->GetVolume("IC02")->InvisibleAll();
   gGeoManager->GetVolume("I651")->InvisibleAll();
   gGeoManager->GetVolume("ICY1")->SetTransparency(90);
   gGeoManager->GetVolume("ICY2")->SetTransparency(90);
   gGeoManager->GetVolume("I215")->SetTransparency(50);
   gGeoManager->GetVolume("I212")->SetTransparency(50);
   gGeoManager->GetVolume("ITSV")->Draw("ogl");
   new TBrowser;
}
