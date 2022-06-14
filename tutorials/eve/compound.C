/// \file
/// \ingroup tutorial_eve
/// Demonstrates usage of EVE compound objects - class TEveCompound.
///
/// \image html eve_compound.png
/// \macro_code
///
/// \author Matevz Tadel

TEveLine* random_line(TRandom& rnd, Int_t n, Float_t delta)
{
   auto line = new TEveLine;
   line->SetMainColor(kGreen);

   Float_t x = 0, y = 0, z = 0;
   for (Int_t i=0; i<n; ++i) {
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

   auto ml = new TEveLine;
   ml->SetMainColor(kRed);
   ml->SetLineStyle(2);
   ml->SetLineWidth(3);
   gEve->InsertVizDBEntry("BigLine", ml);

   auto cmp = new TEveCompound;
   cmp->SetMainColor(kGreen);
   gEve->AddElement(cmp);

   TRandom rnd(0);

   cmp->OpenCompound();

   cmp->AddElement(random_line(rnd, 20, 10));
   cmp->AddElement(random_line(rnd, 20, 10));

   auto line = random_line(rnd, 20, 12);
   line->ApplyVizTag("BigLine");
   cmp->AddElement(line);

   cmp->CloseCompound();

   // Projected view
   auto viewer = gEve->SpawnNewViewer("Projected");
   auto scene  = gEve->SpawnNewScene("Projected Event");
   viewer->AddScene(scene);
   {
      auto v = viewer->GetGLViewer();
      v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   }

   // projections
   auto mng = new TEveProjectionManager(TEveProjection::kPT_RPhi);
   scene->AddElement(mng);
   auto axes = new TEveProjectionAxes(mng);
   scene->AddElement(axes);
   gEve->AddToListTree(axes, kTRUE);
   gEve->AddToListTree(mng, kTRUE);

   mng->ImportElements(cmp);

   gEve->Redraw3D(kTRUE);
}
