// @(#)root/eve:$Id$
// Author: Matevz Tadel

// Demonstates usage pre-scaling for automatic 2D projections.

const char* esd_geom_file_name =
   "http://root.cern.ch/files/alice_ESDgeometry.root";

void projection_prescale()
{
   TFile::SetCacheFileDir(".");
   TEveManager::Create();

   TEveViewer *pev = gEve->SpawnNewViewer("Projections");

   // camera
   TEveScene* s = gEve->SpawnNewScene("Projected Geom");
   pev->AddScene(s);

   TGLViewer* pgv = pev->GetGLViewer();
   pgv->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TGLOrthoCamera& cam = (TGLOrthoCamera&) pgv->CurrentCamera();
   cam.SetZoomMinMax(0.2, 20);

   // projections
   TEveProjectionManager* mng = new TEveProjectionManager();
   {
      mng->SetProjection(TEveProjection::kPT_RPhi);
      TEveProjection* p = mng->GetProjection();
      p->AddPreScaleEntry(0, 0,   4);    // r scale 4 from 0
      p->AddPreScaleEntry(0, 45,  1);    // r scale 1 from 45
      p->AddPreScaleEntry(0, 310, 0.5);
      p->SetUsePreScale(kTRUE);
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
      p->SetUsePreScale(kTRUE);
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
   mng->ImportElements(gsre);

   TEveLine* line = new TEveLine;
   line->SetMainColor(kGreen);
   for (Int_t i=0; i<160; ++i)
      line->SetNextPoint(120*sin(0.2*i), 120*cos(0.2*i), 80-i);
   gEve->AddElement(line);
   mng->ImportElements(line);
   line->SetRnrSelf(kFALSE);


   //-------------------------------------------------------------------------
   // Scaled 3D "projection"
   //-------------------------------------------------------------------------

   TEveViewer *sev = gEve->SpawnNewViewer("Scaled 3D");
   TEveProjectionManager* smng =
      new TEveProjectionManager(TEveProjection::kPT_3D);
   TEveProjection* sp = smng->GetProjection();
   sp->SetUsePreScale(kTRUE);
   sp->AddPreScaleEntry(2,   0,  1);
   sp->AddPreScaleEntry(2, 100,  0.2);

   TEveScene* ss = gEve->SpawnNewScene("Scaled Geom");
   sev->AddScene(ss);
   ss->AddElement(smng);

   smng->ImportElements(gsre);

   //-------------------------------------------------------------------------

   gEve->GetBrowser()->GetTabRight()->SetTab(1);

   gEve->Redraw3D(kTRUE);
}
