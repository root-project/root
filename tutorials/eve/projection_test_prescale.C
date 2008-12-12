// @(#)root/eve:$Id: quad_test.C 25245 2008-08-25 21:44:09Z matevz $
// Author: Matevz Tadel

// Demonstates usage pre-scaling for automatic 2D projections.

const char* esd_geom_file_name = "http://root.cern.ch/files/alice_ESDgeometry.root";

void projection_test_prescale()
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
   TEveProjectionManager* mng = new TEveProjectionManager();
   {
      mng->SetProjection(TEveProjection::kPT_RPhi);
      TEveProjection* p = mng->GetProjection();
      p->AddPreScaleEntry(0, 0,   4);     // r scale 2 from 0
      p->AddPreScaleEntry(0, 45,  1);    // r scale 1 from 45
      p->AddPreScaleEntry(0, 310, 0.5);
      p->SetUsePreScale(1);
   }
   {
      mng->SetProjection(TEveProjection::kPT_RhoZ);
      TEveProjection* p = mng->GetProjection();
      // Increase silicon tracker
      p->AddPreScaleEntry(0, 0, 4);     // rho scale 4 from 0
      p->AddPreScaleEntry(1, 0, 4);     // z   scale 4 from 0
      // Normal for TPC
      p->AddPreScaleEntry(0, 45,  1);   // rho scale 1 from 45
      p->AddPreScaleEntry(1, 110, 1);   // z   scale 1 from 110
      // Reduce the rest
      p->AddPreScaleEntry(0, 310, 0.5);
      p->AddPreScaleEntry(1, 250, 0.5);
      p->SetUsePreScale(1);
   }
   mng->SetProjection(TEveProjection::kPT_RPhi);
   s->AddElement(mng);


   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
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
