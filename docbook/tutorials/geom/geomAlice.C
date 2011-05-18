//script drawing a detector geometry (here ALICE)
//by default the geometry is drawn using the GL viewer
//Using the TBrowser, you can select other components
//if the file containing the geometry is not found in the local
//directory, it is automatically read from the ROOT web site.
// Author: Rene Brun
      
void geomAlice()
{
   TGeoManager::Import("http://root.cern.ch/files/alice.root");
   //gGeoManager->DefaultColors();
   gGeoManager->GetVolume("HUFL")->InvisibleAll();
   gGeoManager->GetVolume("HUWA")->InvisibleAll();
   gGeoManager->GetVolume("ITSV")->InvisibleAll();
   gGeoManager->GetVolume("ZDC")->InvisibleAll();
   gGeoManager->GetVolume("ZEM")->InvisibleAll();
   gGeoManager->GetVolume("XEN1")->InvisibleAll();
   gGeoManager->GetVolume("HBW1")->InvisibleAll();
   gGeoManager->GetVolume("HHW1")->InvisibleAll();
   gGeoManager->GetVolume("HHW3")->InvisibleAll();
   gGeoManager->GetVolume("HHW2")->InvisibleAll();
   gGeoManager->GetVolume("HHF1")->InvisibleAll();
   gGeoManager->GetVolume("HHF2")->InvisibleAll();
   gGeoManager->GetVolume("HPIL")->InvisibleAll();
   gGeoManager->GetVolume("HMBS")->InvisibleAll();
   gGeoManager->GetVolume("HHC1")->InvisibleAll();
   gGeoManager->GetVolume("L3MO")->InvisibleAll();
   gGeoManager->GetVolume("DY1")->SetTransparency(90);
   gGeoManager->GetVolume("DY2")->SetTransparency(90);
   gGeoManager->GetVolume("DY11")->SetTransparency(70);
   gGeoManager->GetVolume("DY22")->SetTransparency(70);
   gGeoManager->GetVolume("ALIC")->Draw("ogl");
   new TBrowser;
}
