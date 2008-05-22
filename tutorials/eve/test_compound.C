TEveLine* random_line(TRandom& rnd, Int_t n, Float_t delta)
{
   TEveLine* line = new TEveLine;
   line->SetMainColor(kGreen);

   Float_t x = 0, y = 0, z = 0;
   for (Int_t i=0; i<n; ++i)
   {
      line->SetNextPoint(x, y, z);
      x += rnd.Uniform(0, delta);
      y += rnd.Uniform(0, delta);
      z += rnd.Uniform(0, delta);
   }

   return line;
}

void test_compound()
{
   TEveManager::Create();

   TEveCompound* cmp = new TEveCompound;
   cmp->SetMainColor(kGreen);
   gEve->AddElement(cmp);

   TRandom rnd(0);

   cmp->OpenCompound();

   cmp->AddElement(random_line(rnd, 20, 10));
   cmp->AddElement(random_line(rnd, 20, 10));

   TEveLine* line = random_line(rnd, 20, 12);
   line->SetMainColor(kRed);
   cmp->AddElement(line);

   cmp->CloseCompound();

   // Projected view
   TEveViewer *viewer = gEve->SpawnNewViewer("Projected");
   TEveScene  *scene  = gEve->SpawnNewScene("Projected Event");
   viewer->AddScene(scene);
   {
      TGLViewer* v = viewer->GetGLViewer();
      v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
      TGLCameraMarkupStyle* mup = v->GetCameraMarkup();
      if(mup) mup->SetShow(kFALSE);
   }

   // projections
   TEveProjectionManager* mng = new TEveProjectionManager();
   scene->AddElement(mng);
   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   axes->SetText("TEveProjections demo");
   axes->SetFontFile("comicbd");
   axes->SetFontSize(20);
   scene->AddElement(axes);
   gEve->AddToListTree(axes, kTRUE);
   gEve->AddToListTree(mng, kTRUE);

   mng->ImportElements(cmp);

   gEve->Redraw3D(kTRUE);
}
