/// \file
/// \ingroup tutorial_graphics
/// \notebook -js
/// \preview Test the IsInside methods of various graphics primitives.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void inside()
{
   auto el = new TEllipse(0.75, 0.25, .2, .15, 45, 315, 62);
   el->Draw();

   auto gr = new TGraph();
   double gr_x1[5] = {0.1, 0.3388252, 0.03796561, 0.4176218, 0.1};
   double gr_y1[5] = {0.5, 0.9644737, 0.7776316, 0.6960526, 0.5};
   gr = new TGraph(5, gr_x1, gr_y1);
   gr->Draw("L");

   auto bx = new TBox(.7, .8, .9, .95);
   bx->Draw();

   auto pv = new TPave(.05, .1, .3, .2);
   pv->Draw();

   auto di = new TDiamond(.05, .25, .3, .4);
   di->Draw();

   auto cr = new TCrown(.5, .5, .1, .15);
   cr->SetFillColor(19);
   cr->Draw();

   for (int i = 0; i < 10000; i++) {
      double x = gRandom->Rndm();
      double y = gRandom->Rndm();
      auto p = new TMarker(x, y, 7);
      p->Draw();
      if (el->IsInside(x, y) || bx->IsInside(x, y) || pv->IsInside(x, y) || di->IsInside(x, y) || cr->IsInside(x, y) ||
          gr->IsInside(x, y)) {
         p->SetMarkerColor(kGreen);
      } else {
         p->SetMarkerColor(kRed);
      }
   }
}
