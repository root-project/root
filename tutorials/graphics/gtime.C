/// \file
/// \ingroup tutorial_graphics
/// Example of a graph of data moving in time.
/// Use the canvas "File/Quit" to exit from this example
///
/// \macro_code
///
/// \author  Olivier Couet

void gtime()
{
   auto c1 = (TCanvas*) gROOT->FindObject("c1");
   if (c1) delete c1;

   c1 = new TCanvas("c1");
   const Int_t ng = 100;
   const Int_t kNMAX = 10000;
   std::vector<Double_t> X(kNMAX), Y(kNMAX);
   Int_t cursor = kNMAX;
   TGraph *g = new TGraph(ng);
   g->SetMarkerStyle(21);
   g->SetMarkerColor(kBlue);
   Double_t x = 0;

   while (1) {
      c1->Clear();
      if (cursor > kNMAX-ng) {
         for (Int_t i = 0; i < ng; i++) {
            X[i] = x;
            Y[i] = sin(x);
            x += 0.1;
         }
         g->Draw("alp");
         cursor = 0;
      } else {
         x += 0.1;
         X[cursor+ng] = x;
         Y[cursor+ng] = TMath::Sin(x);
         cursor++;
         g->DrawGraph(ng, X.data()+cursor, Y.data()+cursor, "alp");
      }
      c1->Update();
      if (gSystem->ProcessEvents()) break;
      gSystem->Sleep(10);
   }
}
