/// \file
/// \ingroup tutorial_eve
/// Demonstrates how to combine Timur's GL plots with other scene elements.
///
/// \image html eve_glplot_geom.png
/// \macro_code
///
/// \author Matevz Tadel

void glplot_geom()
{
   TEveManager::Create();

   TEveUtil::Macro("show_extract.C");

   auto h31 = new TH3F("h31", "h31", 20, -3, 3, 20, -3, 3, 20, -3, 3);
   auto gxy = new TF3("gaus2","xygaus");
   gxy->SetParameters(1,0,1,0,0.3);
   h31->FillRandom("gaus2");

   h31->SetFillColor(2);
   auto x = new TEvePlot3D("EvePlot - TH3F");
   x->SetPlot(h31, "glbox");
   x->RefMainTrans().Scale(800, 800, 1000);
   x->RefMainTrans().RotateLF(1, 3, TMath::PiOver2());
   gEve->AddElement(x);

   gEve->Redraw3D(kTRUE);
}
