#include <iostream>

#include "TColorGradient.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TH1F.h"

typedef TColorGradient::Point point_type;

void grad2()
{
   //Gradient fill with transparency and "SAME" option.
   //This macro works ONLY on MacOS X with --enable-cocoa.
   
   TCanvas *cnv = new TCanvas("cnv", "gradient_test", 100, 100, 800, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      std::cout<<"This macro works only on MacOS X with --enable-cocoa\n";
      delete cnv;
      return;
   }

   TH1F * hist = new TH1F("a", "b", 10, -2., 3.);
   TH1F * hist2 = new TH1F("c", "d", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   const Double_t locations[] = {0., 1.};
   new TColor(1001, 1., 0., 0., "red", 0.5);
   const Color_t idx1[] = {1001, kOrange};
   //Gradient from ROOT's kOrange and my own semi-transparent red color.
   TLinearGradient *grad1 = new TLinearGradient(1002, 2, locations, idx1);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   
   grad1->SetStartEnd(start, end);
   
   hist->SetFillColor(1002);
   
   new TColor(1003, 0., 1., 0., "green", 0.5);
   const Color_t idx2[] = {1003, kBlue};
   //Gradient from ROOT's kBlue and my own semi-transparent green color.
   TLinearGradient *grad2 = new TLinearGradient(1004, 2, locations, idx2);
   grad2->SetStartEnd(start, end);
   hist2->SetFillColor(1004);
   
   hist2->Draw();
   hist->Draw("SAME");
}
