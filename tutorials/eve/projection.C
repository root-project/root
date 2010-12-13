// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstates usage of automatic 2D projections - class TEveProjectionManager.

const char* esd_geom_file_name =
   "http://root.cern.ch/files/alice_ESDgeometry.root";

void projection()
{
   TFile::SetCacheFileDir(".");
   TEveManager::Create();

   // camera
   TEveScene* s = gEve->SpawnNewScene("Projected Event");
   gEve->GetDefaultViewer()->AddScene(s);
   TGLViewer* v = gEve->GetDefaultGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TGLOrthoCamera* cam = (TGLOrthoCamera*) v->CurrentCamera();
   cam->SetZoomMinMax(0.2, 20);

   // projections
   TEveProjectionManager* mng =
      new TEveProjectionManager(TEveProjection::kPT_RPhi);
   s->AddElement(mng);
   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   axes->SetTitle("TEveProjections demo");
   s->AddElement(axes);
   gEve->AddToListTree(axes, kTRUE);
   gEve->AddToListTree(mng, kTRUE);

   // Simple geometry
   TFile* geom = TFile::Open(esd_geom_file_name, "CACHEREAD");
   if (!geom)
      return;

   TEveGeoShapeExtract* gse = (TEveGeoShapeExtract*) geom->Get("Gentle");
   TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse, 0);
   geom->Close();
   delete geom;
   gEve->AddGlobalElement(gsre);
   gEve->GetGlobalScene()->SetRnrState(kFALSE);
   mng->ImportElements(gsre);

   TEveLine* line = new TEveLine;
   line->SetMainColor(kGreen);
   for (Int_t i=0; i<160; ++i)
      line->SetNextPoint(120*sin(0.2*i), 120*cos(0.2*i), 80-i);
   gEve->AddElement(line);
   mng->ImportElements(line);
   line->SetRnrSelf(kFALSE);

   gEve->Redraw3D(kTRUE);
}
