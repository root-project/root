// Demonstrates how to combine Timur's GL plots with other scene elements.
// Author: Matevz Tadel, Sept 2009

void glplot_geom()
{
   TEveManager::Create();

   TEveUtil::Macro("show_extract.C");

   TH3F *h31 = new TH3F("h31", "h31", 20, -3, 3, 20, -3, 3, 20, -3, 3);
   h31->FillRandom("gaus", 20*20*20);
   h31->SetFillColor(2);
   x = new TEvePlot3D("EvePlot - TH3F");
   x->SetPlot(h31, "glbox");
   x->RefMainTrans().Scale(800, 800, 1000);
   x->RefMainTrans().RotateLF(1, 3, TMath::PiOver2());
   gEve->AddElement(x);

   gEve->Redraw3D(kTRUE);
}
