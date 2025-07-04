/// \file
/// \ingroup tutorial_graphics
/// \preview Example of canvas partitioning with ratios
/// The TPad::DivideRatios method enables partitioning of the canvas according to different
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void colorpads(TCanvas *C, int n) {
   TPad *p;
   for (int i =1; i<=n; i++) {
      p = (TPad*)C->cd(i);
      p->SetFillColor(100+i);
      p->Modified();
      p->Update();
   }
}

void canvas3() {
   // canvas divided 2x2 with 4 different ratios along X and Y
   auto C1 = new TCanvas("C1", "C1", 400, 400);
   C1->DivideRatios(2, 2, {.2,.8}, {.4,.6}); colorpads(C1,4);

   // canvas divided 2x3 with ratios defined only in X
   auto C2 = new TCanvas("C2", "C2", 400, 0, 400, 400);
   C2->DivideRatios(2, 3, {.2,.8}); colorpads(C2,6);

   // canvas divided 2x2 with no ratios defined
   auto C3 = new TCanvas("C3", "C3", 0, 455, 400, 400);
   C3->DivideRatios(2, 2); colorpads(C3,4);

   // canvas divided 2x2 with ratios and margins
   auto C4 = new TCanvas("C4", "C4", 400, 455, 400, 400);
   C4->DivideRatios(2, 2, {.2,.7}, {.4,.5}, 0.05, 0.01); colorpads(C4,4);
}
