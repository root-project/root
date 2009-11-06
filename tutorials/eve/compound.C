// @(#)root/eve:$Id: text_test.C 26717 2008-12-07 22:07:55Z matevz $
// Author: Matevz Tadel

// Demonstates usage of EVE compound objects - class TEveCompound.

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

void compound()
{
   TEveManager::Create();

   TEveLine* ml = new TEveLine;
   ml->SetMainColor(kRed);
   ml->SetLineStyle(2);
   ml->SetLineWidth(3);
   gEve->InsertVizDBEntry("BigLine", ml);

   TEveCompound* cmp = new TEveCompound;
   cmp->SetMainColor(kGreen);
   gEve->AddElement(cmp);

   TRandom rnd(0);

   cmp->OpenCompound();

   cmp->AddElement(random_line(rnd, 20, 10));
   cmp->AddElement(random_line(rnd, 20, 10));

   TEveLine* line = random_line(rnd, 20, 12);
   line->ApplyVizTag("BigLine");
   cmp->AddElement(line);

   cmp->CloseCompound();

   // Projected view
   TEveViewer *viewer = gEve->SpawnNewViewer("Projected");
   TEveScene  *scene  = gEve->SpawnNewScene("Projected Event");
   viewer->AddScene(scene);
   {
      TGLViewer* v = viewer->GetGLViewer();
      v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   }

   // projections
   TEveProjectionManager* mng = new TEveProjectionManager(TEveProjection::kPT_RPhi);
   scene->AddElement(mng);
   TEveProjectionAxes* axes = new TEveProjectionAxes(mng);
   scene->AddElement(axes);
   gEve->AddToListTree(axes, kTRUE);
   gEve->AddToListTree(mng, kTRUE);

   mng->ImportElements(cmp);

   gEve->Redraw3D(kTRUE);
}
