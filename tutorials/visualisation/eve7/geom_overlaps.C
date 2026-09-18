/// \file
/// \ingroup tutorial_eve_7
/// Browse the overlaps and extrusions of a geometry, and look at them in 3D.
///
/// The overlap check runs once, over the whole geometry, before the browser starts
/// serving -- TGeoVolume::CheckOverlaps() is recursive, so checking volumes one at a
/// time as somebody clicks around both repeats the work and costs seconds per click.
/// After the single pass the table is instant.
///
/// Pick a row and the 3D view shows that overlap and nothing else: the two volumes
/// (red and yellow, semi-transparent) and the points on the offending surface (cyan).
/// An overlap belongs to a *logical* volume, so if that volume is placed several times
/// the pair is drawn at every placement.
///
/// \macro_code
///
/// \author Alja Mrak-Tadel

#include <ROOT/REveElement.hxx>
#include <ROOT/REveGeoOverlaps.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveScene.hxx>
#include <ROOT/REveViewer.hxx>

#include "TFile.h"
#include "TGeoManager.h"
#include "TKey.h"

#include <cstring>
#include <iostream>

using namespace ROOT::Experimental;

TGeoManager *load_geometry(const char *fname)
{
   TFile *f = TFile::Open(fname);
   if (!f || f->IsZombie()) {
      std::cerr << "cannot open " << fname << "\n";
      return nullptr;
   }
   TIter next(f->GetListOfKeys());
   while (auto *key = (TKey *)next())
      if (!std::strcmp(key->GetClassName(), "TGeoManager"))
         return dynamic_cast<TGeoManager *>(key->ReadObj());
   std::cerr << "no TGeoManager in " << fname << "\n";
   return nullptr;
}

void geom_overlaps(const char *fname = "cmsSimGeo2026.root", double precision = 0.001)
{
   auto *geom = load_geometry(fname);
   if (!geom)
      return;

   auto *eveMng = REveManager::Create();
   eveMng->AllowMultipleRemoteConnections(false, false);

   auto *table = new REveGeoOverlapTable("Overlaps", "Overlaps and extrusions");

   // Before anything is served: the scan mutates TGeoManager global state and runs
   // its own thread pool, so it must not race the client threads.
   std::cout << ">>> checking overlaps (precision " << precision << " cm) ...\n";
   table->ScanOverlaps(geom, precision);
   std::cout << ">>> found " << table->GetNumOverlaps() << " overlaps\n";

   // the list lives in its own scene, bound to a viewer named GeoOverlapTable --
   // that name is what picks the client-side view
   auto *tableScene = eveMng->SpawnNewScene("Overlap List");
   tableScene->AddElement(table);
   auto *tableViewer = eveMng->SpawnNewViewer("GeoOverlapTable", "");
   tableViewer->AddScene(tableScene);

   // the shapes of whichever row is selected go into the event scene
   eveMng->GetEventScene()->AddElement(table->GetVizHolder());

   eveMng->Show();
}
