/// \file
/// \ingroup tutorial_hist
/// \notebook
///
/// When an histogram is drawn with the option `COLZ`, a palette is automatically drawn
/// vertically on the right side of the plot. It is possible to move and resize this
/// vertical palette as shown on the left plot. The right plot demonstrates that, when the
/// width of the palette is larger than its height, the palette is automatically drawn
/// horizontally.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void movepalette()
{
   auto c = new TCanvas("", "",0,0,1100,550);
   c->Divide(2,1);
   gStyle->SetOptStat(0);

   auto h1 = new TH2D("h1","h1",40,-4,4,40,-20,20);
   auto h2 = new TH2D("h2","h2",40,-4,4,40,-20,20);
   float px, py;
   for (int i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py); h1->Fill(px,5*py);
      gRandom->Rannor(px,py); h2->Fill(px,5*py);
   }

   TPad *p1 = (TPad *)c->cd(1);
   TPad *p2 = (TPad *)c->cd(2);

   p1->SetRightMargin(0.15);
   auto palette1 = new TPaletteAxis(4.05,-15,4.5,15,h1);
   h1->GetListOfFunctions()->Add(palette1);

   p2->SetBottomMargin(0.2);
   auto palette2 = new TPaletteAxis(-3.,-25,3.,-23,h2);
   h2->GetListOfFunctions()->Add(palette2);

   p1->cd(); h1->Draw("colz");
   p2->cd(); h2->Draw("colz");
}