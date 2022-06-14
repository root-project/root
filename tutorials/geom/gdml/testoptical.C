/// \file
/// \ingroup tutorial_geom
/// Tests importing/exporting optical surfaces from GDML
///
/// Optical surfaces, skin surfaces and border surfaces are imported in object arrays
/// stored by TGeoManager class. Optical surfaces do not store property arrays but point
/// to GDML matrices describing such properties. One can get the data for such property
/// like:
///   TGeoOpticalSurface *surf = geom->GetOpticalSurface("surf1");
///   const char *property = surf=>GetPropertyRef("REFLECTIVITY");
///   TGeoGDMLMatrix *m = geom->GetGDMLMatrix(property);
/// Skin surfaces and border surfaces can be retrieved from the TGeoManager object by using:
///   TObjArray *skin_array = geom->GetListOfSkinSurfaces();
///   TObjArra8 *border_array = geom->GetListOfBorderSurfaces();
/// Alternatively accessors by name can also be used: GetSkinSurface(name)/GetBorderSurface(name)
///
/// \author Andrei Gheata

#include <cassert>
#include <TObjArray.h>
#include <TROOT.h>
#include <TGeoOpticalSurface.h>
#include <TGeoManager.h>

double Checksum(TGeoManager *geom)
{
   double sum = 0.;
   TIter next(geom->GetListOfOpticalSurfaces());
   TGeoOpticalSurface *surf;
   while ((surf = (TGeoOpticalSurface *)next())) {
      sum += (double)surf->GetType() + (double)surf->GetModel() + (double)surf->GetFinish() + surf->GetValue();
      TString name = surf->GetName();
      sum += (double)name.Hash();
      name = surf->GetTitle();
      sum += (double)name.Hash();
   }
   return sum;
}

int testoptical()
{
   TString geofile = gROOT->GetTutorialDir() + "/geom/gdml/opticalsurfaces.gdml";
   geofile.ReplaceAll("\\", "/");
   TGeoManager::SetExportPrecision(8);
   TGeoManager::SetVerboseLevel(0);
   printf("=== Importing %s ...\n", geofile.Data());
   TGeoManager *geom = TGeoManager::Import(geofile);
   printf("=== List of GDML matrices:\n");
   geom->GetListOfGDMLMatrices()->Print();
   printf("=== List of optical surfaces:\n");
   geom->GetListOfOpticalSurfaces()->Print();
   printf("=== List of skin surfaces:\n");
   geom->GetListOfSkinSurfaces()->Print();
   printf("=== List of border surfaces:\n");
   geom->GetListOfBorderSurfaces()->Print();
   // Compute some checksum for optical surfaces
   double checksum1 = Checksum(geom);
   printf("=== Exporting as .gdml, then importing back\n");
   geom->Export("tmp.gdml");
   geom = TGeoManager::Import("tmp.gdml");
   double checksum2 = Checksum(geom);
   assert((checksum2 == checksum1) && "Exporting/importing as .gdml not OK");
   printf("=== Exporting as .root, then importing back\n");
   geom->Export("tmp.root");
   geom = TGeoManager::Import("tmp.root");
   double checksum3 = Checksum(geom);
   assert((checksum3 == checksum1) && "Exporting/importing as .root not OK");
   printf("all OK\n");
   return 0;
}
