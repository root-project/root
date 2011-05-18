//This tutorial illustrates how to create an histogram with polygonal
//bins (TH2Poly). The bins are boxes.
//
//Author: Olivier Couet

{
   TCanvas *ch2p2 = new TCanvas("ch2p2","ch2p2",600,400);
   TH2Poly *h2p = new TH2Poly();
   h2p->SetName("Boxes");
   h2p->SetTitle("Boxes");
   gStyle->SetPalette(1);

   Int_t i,j;
   Int_t nx = 40;
   Int_t ny = 40;
   Double_t x1,y1,x2,y2;
   Double_t dx=0.2, dy=0.1;
   x1 = 0.;
   x2 = dx;

   for (i = 0; i<nx; i++) {
      y1 = 0.;
      y2 = dy;
      for (j = 0; j<ny; j++) {
         h2p->AddBin(x1, y1, x2, y2);
         y1 = y2;
         y2 = y2+y2*dy;
      }
      x1 = x2;
      x2 = x2+x2*dx;
   }

   TRandom ran;
   for (i = 0; i<300000; i++) {
      h2p->Fill(50*ran.Gaus(2.,1), ran.Gaus(2.,1));
   }

   h2p->Draw("COLZ");
   return ch2p2;
}
