//Author: Timur Pocheptsov, 19/03/2014
//This macro shows how to create and use linear gradients to fill
//a histogram or a pad.
//Requires OpenGL: you either have to use gStyle->SetCanvasPreferGL(kTRUE)
//or set OpenGL.CanvasPreferGL to 1 in a $ROOTSYS/etc/system.rootrc.

//Includes for ACLiC:
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions:
#include "customcolors.h"

typedef TColorGradient::Point point_type;

//______________________________________________________________________
Color_t create_pad_frame_gradient()
{
   //We create a gradient with 4 steps - from dark (and semi-transparent)
   //gray to almost transparent (95%) white and to white and to dark gray again.
   Color_t idx[3] = {};
   if (FindFreeCustomColorIndices(3, idx) != 3)
      return -1;

   const Double_t locations[] = {0., 0.2, 0.8, 1.};
   
   new TColor(idx[0], 0.25, 0.25, 0.25, "special pad color1", 0.55);
   new TColor(idx[1], 1., 1., 1., "special pad color2", 0.05);
   Color_t colorIndices[4] = {idx[0], idx[1], idx[1], idx[0]};
   TLinearGradient * const grad = new TLinearGradient(idx[2], 4, locations, colorIndices);
   const point_type start(0., 0.);
   const point_type end(1., 0.);
   grad->SetStartEnd(start, end);
   
   return idx[2];
}

//______________________________________________________________________
Color_t create_pad_gradient()
{
   //We create two-steps gradient from ROOT's standard colors (38 and 30).
   const Color_t idx = FindFreeCustomColorIndex(1000);//Start lookup from 1000.
   if (idx == -1)
      return -1;

   const Double_t locations[] = {0., 1.};
   const Color_t colorIndices[2] = {30, 38};
   
   TLinearGradient * const grad = new TLinearGradient(idx, 2, locations, colorIndices);
   const point_type start(0., 0.);
   const point_type end(0., 1.);
   grad->SetStartEnd(start, end);
   
   return idx;
}

//______________________________________________________________________
void grad()
{
   const Color_t frameColor = create_pad_frame_gradient();
   if (frameColor == -1) {
      ::Error("grad", "failed to allocate a custom color");
      return;
   }
   
   const Color_t padColor = create_pad_gradient();
   if (padColor == -1) {
      ::Error("grad", "failed to allocate a custom color");
      return;//:( no way to cleanup palette now.
   }

   const Color_t histFill = FindFreeCustomColorIndex(padColor + 1);//Start lookup from the next.
   if (histFill == -1) {
      ::Error("grad", "failed to allocate a custom color");
      return;
   }

   gStyle->SetCanvasPreferGL(kTRUE);

   TCanvas * const cnv = new TCanvas("gradient test", "gradient test", 100, 100, 600, 600);
   if (!cnv->UseGL()) {
      ::Error("grad", "This macro requires OpenGL");
      delete cnv;
      return;
   }
   
   cnv->SetFillColor(padColor);
   cnv->SetFrameFillColor(frameColor);

   //Gradient to fill a histogramm
   const Color_t colorIndices[3] = {kYellow, kOrange, kRed};
   const Double_t lengths[3] = {0., 0.5, 1.};
   TLinearGradient * const grad = new TLinearGradient(histFill, 3, lengths, colorIndices);
   grad->SetStartEnd(point_type(0., 0.), point_type(0., 1.));
   
   TH1F * const hist = new TH1F("h11", "h11", 20, -3., 3.);
   hist->SetFillColor(histFill);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
