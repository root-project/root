// A multi-threaded geometry that is still alive at exit is deleted by
// TROOT::EndOfProcessCleanups(). That runs after the static objects of libGeom
// are destroyed if libGeom was loaded after the TApplication was created, as
// in PyROOT. ~TGeoManager() must not use them (it used to lock the destroyed
// TGeoManager::fgMutex, which aborts on macOS).

#include <TApplication.h>
#include <TROOT.h>
#include <TSystem.h>

int main()
{
   new TApplication("test_geomanager_at_exit", nullptr, nullptr); // never deleted, as in PyROOT
   if (gSystem->Load("libGeom") < 0)
      return 1;
   gROOT->ProcessLine(R"(
      auto geom = new TGeoManager("geom", "geom");
      auto vacuum = new TGeoMedium("vacuum", 1, new TGeoMaterial("vacuum", 0, 0, 0));
      geom->SetTopVolume(geom->MakeBox("top", vacuum, 10, 10, 10));
      geom->CloseGeometry();
      geom->SetMaxThreads(2);
   )");
   return gROOT->ProcessLine("gGeoManager && gGeoManager->IsMultiThread();") ? 0 : 1;
}
