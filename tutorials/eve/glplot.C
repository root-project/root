/// \file
/// \ingroup tutorial_eve
/// Preliminary demo for showing Timur's GL plots in EVE.
///
/// \image html eve_glplot.png
/// \macro_code
///
/// \author Matevz Tadel

void glplot()
{
   TEveManager::Create();
   gEve->GetDefaultGLViewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);

   auto f2 = new TF2("f2","x**2 + y**2 - x**3 -8*x*y**4", -1., 1.2, -1.5, 1.5);
   f2->SetFillColor(45);
   auto x = new TEvePlot3D("EvePlot - TF2");
   x->SetLogZ(kTRUE);
   x->SetPlot(f2,"glsurf4");
   x->RefMainTrans().MoveLF(2, 1);
   gEve->AddElement(x);

   auto h31 = new TH3F("h31", "h31", 10, -1, 1, 10, -1, 1, 10, -1, 1);
   auto gxy = new TF3("gaus2","xygaus");
   gxy->SetParameters(1,0,1,0,0.3);
   h31->FillRandom("gaus2");
   h31->SetFillColor(2);
   x = new TEvePlot3D("EvePlot - TH3F");
   x->SetPlot(h31, "glbox");
   x->RefMainTrans().MoveLF(2, -1);
   gEve->AddElement(x);

   gEve->Redraw3D(kTRUE);
}
