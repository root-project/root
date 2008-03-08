const char* esd_geom_file_name = "http://root.cern.ch/files/alice_ESDgeometry.root";

void projection_test()
{
   TFile::SetCacheFileDir(".");
   TEveManager::Create();

   // camera
   TEveScene* s = gEve->SpawnNewScene("Projected Event");
   gEve->GetDefViewer()->AddScene(s);
   TGLViewer* v = (TGLViewer *)gEve->GetGLViewer();
   v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TGLCameraMarkupStyle* mup = v->GetCameraMarkup();
   if(mup) mup->SetShow(kFALSE);

   // projections
   TEveProjectionManager* mng = new TEveProjectionManager();
   gEve->AddElement(mng, s);
   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   axes->SetText("TEveProjections demo");
   axes->SetFontFile("comicbd");
   axes->SetFontSize(20);
   gEve->AddGlobalElement(axes);
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
   gsre->SetRnrState(kFALSE);

   TEveLine* line = new TEveLine;
   line->SetMainColor((Color_t)3);
   for (Int_t i=0; i<160; ++i)
      line->SetNextPoint(120*sin(0.2*i), 120*cos(0.2*i), 80-i);
   gEve->AddElement(line);
   mng->ImportElements(line);
   line->SetRnrSelf(kFALSE);

   gEve->Redraw3D(kTRUE);
}
