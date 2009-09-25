// Preliminary demo for showing Timur's GL plots in EVE.
// Author: Matevz Tadel, July 2009

// The main purpose of this test is to demonstrate the problems with
// rendering of sevaral GL plots or with rendering of GL plots with
// other scene objects - the usage of depth buffer destroys
// depth-buffer coherence.

void glplot()
{
   TEveManager::Create();
   gEve->GetDefaultGLViewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);

   TF2 *f2 = new TF2("f2","x**2 + y**2 - x**3 -8*x*y**4", -1., 1.2, -1.5, 1.5);
   f2->SetFillColor(45);
   x = new TEvePlot3D("EvePlot - TF2");
   x->SetLogZ(kTRUE);
   x->SetPlot(f2,"glsurf4");
   x->RefMainTrans().MoveLF(2, 1);
   gEve->AddElement(x);


   TH3F *h31 = new TH3F("h31", "h31", 10, -1, 1, 10, -1, 1, 10, -1, 1);
   h31->FillRandom("gaus");
   h31->SetFillColor(2);
   x = new TEvePlot3D("EvePlot - TH3F");
   x->SetPlot(h31, "glbox");
   x->RefMainTrans().MoveLF(2, -1);
   gEve->AddElement(x);

   gEve->Redraw3D(kTRUE);
}
