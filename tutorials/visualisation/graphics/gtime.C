/// \file
/// \ingroup tutorial_graphics
/// Example of a graph of data moving in time.
/// Use the canvas "File/Quit ROOT" to exit from this example
///
/// \macro_code
///
/// \author  Olivier Couet

void gtime()
{
   auto c1 = (TCanvas *)gROOT->FindObject("c1");
   if (c1)
      delete c1;

   c1 = new TCanvas("c1");
   const Int_t ng = 100;
   const Int_t kNMAX = 10000;
   std::vector<Double_t> X(kNMAX), Y(kNMAX);
   Int_t cursor = kNMAX;
   Double_t x = 0, stepx = 0.1;

   while (!gSystem->ProcessEvents()) {
      if (cursor + ng >= kNMAX) {
         cursor = 0;
         for (Int_t i = 0; i < ng; i++) {
            X[i] = x;
            Y[i] = TMath::Sin(x);
            x += stepx;
         }
      } else {
         X[cursor + ng] = x;
         Y[cursor + ng] = TMath::Sin(x);
         x += stepx;
         cursor++;
      }

      TGraph *g = new TGraph(ng, X.data() + cursor, Y.data() + cursor);
      g->SetMarkerStyle(21);
      g->SetMarkerColor(kBlue);
      g->SetLineColor(kGreen);
      g->SetBit(kCanDelete); // let canvas delete graph when call TCanvas::Clear()

      c1->Clear();
      g->Draw("alp");
      c1->Update();

      gSystem->Sleep(10);
   }
}
