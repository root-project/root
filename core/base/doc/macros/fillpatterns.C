void box(Int_t pat, Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Draw an box using the fill pattern "pat" with the "pat" value
   // written on top.

   TBox b;
   b.SetFillColor(1);
   b.SetFillStyle(pat); b.DrawBox(x1,y1,x2,y2);
   b.SetFillStyle(0)  ; b.DrawBox(x1,y1,x2,y2);
   b.SetFillColor(0)  ; b.SetFillStyle(1000)  ;
   Double_t dx = (x2-x1)/3;
   Double_t dy = (y2-y1)/3;
   Double_t h  = (y2-y1)/2.5;
   b.DrawBox(x1+dx, y1+dy, x2-dx, y2-dy);
   b.SetFillStyle(0);
   b.DrawBox(x1+dx, y1+dy, x2-dx, y2-dy);

   TLatex l;
   l.SetTextAlign(22); l.SetTextSize(h);
   l.DrawLatex((x1+x2)/2, (y1+y2)/2, Form("%d",pat));
}

TCanvas * fillpatterns()
{
   // Fill patterns example. This macro shows the available fill patterns.
   // The first table displays the 25 fixed patterns. They cannot be
   // customized unlike the hatches displayed in the second table which be
   // cutomized using:
   //   - gStyle->SetHatchesSpacing() to define the spacing between hatches.
   //   - gStyle->SetHatchesLineWidth() to define the hatches line width.
   //
   // Author: Olivier Couet.

   TCanvas *Pat = new TCanvas("Fill Patterns", "",0,0,500,700);
   Pat->Range(0,0,1,1);
   Pat->SetBorderSize(2);
   Pat->SetFrameFillColor(0);
   Double_t bh = 0.059;
   Double_t db = 0.01;
   Double_t y  = 0.995;
   Int_t i,j=3001;

   // Fixed patterns.
   for (i=1; i<=5; i++) {
      box(j++, 0.01, y-bh, 0.19, y);
      box(j++, 0.21, y-bh, 0.39, y);
      box(j++, 0.41, y-bh, 0.59, y);
      box(j++, 0.61, y-bh, 0.79, y);
      box(j++, 0.81, y-bh, 0.99, y);
      y = y-bh-db;
   }

   // Hatches
   y = y-3*db;
   gStyle->SetHatchesSpacing(2.0);
   gStyle->SetHatchesLineWidth(1);
   Int_t j1 = 3144;
   Int_t j2 = 3305;
   Int_t j3 = 3350;
   Int_t j4 = 3490;
   Int_t j5 = 3609;
   for (i=1; i<=9; i++) {
      if (i==6) {j2 += 10; j3 += 1; j4 += 1; j5 += 10;}
      if (i==5) {j4 -= 10; j5 -= 1;}
      box(j1, 0.01, y-bh, 0.19, y);
      box(j2, 0.21, y-bh, 0.39, y);
      box(j3, 0.41, y-bh, 0.59, y);
      box(j4, 0.61, y-bh, 0.79, y);
      box(j5, 0.81, y-bh, 0.99, y);
      j1 += 100;
      j2 += 10;
      j3 += 1;
      j4 -= 9;
      j5 += 9;
      y = y-bh-db;
   }
   return Pat;
}
