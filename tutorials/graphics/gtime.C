/// \file
/// \ingroup tutorial_graphics
/// Example of a graph of data moving in time.
/// Use the canvas "File/Quit" to exit from this example
///
/// \macro_code
///
/// \author  Olivier Couet

void gtime() {
   TCanvas *c1 = new TCanvas("c1");
   const Int_t ng = 100;
   const Int_t kNMAX = 10000;
   Double_t *X = new Double_t[kNMAX];
   Double_t *Y = new Double_t[kNMAX];
   Int_t cursor = kNMAX;
   TGraph *g = new TGraph(ng);
   g->SetMarkerStyle(21);
   g->SetMarkerColor(kBlue);
   Double_t x = 0;

   while (1) {
      c1->Clear();
      if (cursor > kNMAX-ng) {
         for (Int_t i=0;i<ng;i++) {
            X[i] = x;
            Y[i] = sin(x);
            x   += 0.1;
         }
         g->Draw("alp");
         cursor = 0;
      } else {
         x += 0.1;
         X[cursor+ng] = x;
         Y[cursor+ng] = sin(x);
         cursor++;
         g->DrawGraph(ng,&X[cursor],&Y[cursor],"alp");
      }
      c1->Update();
      gSystem->ProcessEvents();
      gSystem->Sleep(10);
   }
}

