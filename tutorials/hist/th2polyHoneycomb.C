//This tutorial illustrates how to create an histogram with hexagonal
//bins (TH2Poly), fill it and draw it using GL.
//
//Author: Olivier Couet

void th2polyHoneycomb(){
   gStyle->SetCanvasPreferGL(true);
   TH2Poly *hc = new TH2Poly();
   hc->Honeycomb(0,0,.1,25,25);
   gStyle->SetPalette(1);

   TRandom ran;
   for (int i = 0; i<30000; i++) {
      hc->Fill(ran.Gaus(2.,1), ran.Gaus(2.,1));
   }

   hc->Draw("gllego");
}
