//Author: Timur Pocheptsov, 19/03/2014

#include <iostream>
#include <cassert>
#include <vector>

#include "TColorGradient.h"
#include "TCanvas.h"
#include "TPie.h"

#include "customcolor.h"

void gradients()
{
   const UInt_t nSlices = 5;

   Int_t idx[3] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(idx) != 3) {
      Error("grad", "failed to create new custom colors");
      return;
   }

   const Double_t locations[] = {0., 1.};
   const Double_t rgbaData1[] = {1., 0.8, 0., 1., 1., 0.2, 0., 0.8};

   //A special color: radial gradient + transparency.
   TRadialGradient * const gradientFill1 = new TRadialGradient(idx[0], 2, locations, rgbaData1);
   gradientFill1->SetCoordinateMode(TColorGradient::kPadMode);
   gradientFill1->SetStartEndR1R2(TColorGradient::Point(0.5, 0.5), 0.2, TColorGradient::Point(0.5, 0.5), 0.6);

   Double_t values[nSlices] = {0.8, 1.2, 1.2, 0.8, 1.};
   Int_t colors[nSlices] = {idx[0], idx[0], idx[0], idx[0], idx[0]};

   TCanvas *c = new TCanvas("cpie","Gradient colours demo", 700, 700);
   c->cd();

   const Double_t rgbaData2[] = {0.2, 0.2, 0.2, 1., 0.8, 1., 0.9, 1.};
   TLinearGradient * const gradientFill2 = new TLinearGradient(idx[1], 2, locations, rgbaData2);
   gradientFill2->SetStartEnd(TColorGradient::Point(0, 0), TColorGradient::Point(1, 1));
   
   c->SetFillColor(idx[1]);
   
   TText * t = new TText(0.05, 0.7, "Can you see the text?");
   t->Draw();

   TPad * pad = new TPad("p", "p", 0., 0., 1., 1.);
   new TColor(idx[2], 1., 1., 1., "transparent_fill_color", 0.);
   pad->SetFillColor(idx[2]);
   pad->Draw();
   pad->cd();

   TPie *pie4 = new TPie("pie", "Pie", nSlices, values, colors);
   pie4->SetEntryRadiusOffset(2,.05);
   pie4->SetLabelsOffset(-0.08);
   pie4->SetRadius(0.4);
   pie4->Draw("rsc");
   
   c->Modified();
   c->Update();
}
