/// \file
/// \ingroup tutorial_geom
/// Script drawing a detector geometry (here ITSV from Alice).
///
/// By default the geometry is drawn using the GL viewer
/// Using the TBrowser, you can select other components
/// if the file containing the geometry is not found in the local
/// directory, it is automatically read from the ROOT web site.
///
/// \image html geom_geomAlice_itsv.png width=800px
/// \macro_image (tcanvas_js)
/// \preview 
/// \macro_code
///
/// \author Rene Brun
/// \date August 2020

void geomAlice_itsv() {
   TGeoManager::Import("http://root.cern/files/alice2.root");
   gGeoManager->DefaultColors();
   gGeoManager->GetVolume("ITSV")->Draw("ogl");
   new TBrowser;
}
