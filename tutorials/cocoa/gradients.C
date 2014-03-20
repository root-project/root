//Author: Timur Pocheptsov, 19/03/2014
//This macro requires OS X and ROOT compiled with --enable-cocoa to run.

//Features:
//1. Radial and linear gradients
//2. Transparent/semitransparent colours.
//3. Shadows.


//Includes for ACLiC:

#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TText.h"
#include "TPie.h"

//Cocoa aux. functions.
#include "customcolor.h"

void gradients()
{
   //Find free colour indices in the ROOT's palette for:
   //1. A radial gradient for TPie;
   //2. A linear gradient for TCanvas
   //3. A fully transparent fill color for a nested pad.

   Int_t colorIndices[3] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(colorIndices) != 3) {
      ::Error("grad", "failed to create new custom colors");
      return;
   }

   //Better names:
   const Int_t &radialFill = colorIndices[0];
   const Int_t &linearFill = colorIndices[1];
   const Int_t &transparentFill = colorIndices[2];
   
   //Create a canvas to check if we support gradients:
   TCanvas *c = new TCanvas("cpie","Gradient colours demo", 700, 700);
   //Before we allocated any new colour or created any object:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("gradients", "This macro requires OS X and ROOT build with --enable-cocoa");
      delete c;
      return;
   }
   
   c->cd();

   //Colour positions in the gradient colours (here I place colours at the
   //ends of 0-1 axis for simplicity):
   const Double_t locations[] = {0., 1.};
   
   //Linear gradient fill (with an axis angle == 45):
   const Double_t rgbaData1[] = {0.2, 0.2, 0.2, 1.,/*gray*/
                                 0.8, 1., 0.9, 1.  /*pale green*/};
   TLinearGradient * const gradientFill2 = new TLinearGradient(linearFill, 2, locations, rgbaData1);
   //45 degrees:
   gradientFill2->SetStartEnd(TColorGradient::Point(0, 0), TColorGradient::Point(1, 1));
   //Set as a background color in the canvas:
   c->SetFillColor(linearFill);

   //Draw a text in the canvas (the object above the text will be
   //semi-transparent):
   TText * t = new TText(0.05, 0.7, "Can you see the text?");
   t->Draw();
   
   //We create a nested pad on top to render a TPie in,
   //this way we still have a text (below) + TPie with
   //a fancy colour on top.
   TPad * pad = new TPad("p", "p", 0., 0., 1., 1.);
   
   //TPad itself is fully transparent:
   new TColor(transparentFill, 1., 1., 1., "transparent_fill_color", 0.);
   pad->SetFillColor(transparentFill);
   //Add our pad into the canvas:
   pad->Draw();
   pad->cd();


   //Radial gradient fill for a TPie object:
   const Double_t rgbaData2[] = {/*opaque orange at the start:*/1., 0.8, 0., 1.,
                                 /*transparent red at the end:*/1., 0.2, 0., 0.8};


   TRadialGradient * const gradientFill1 = new TRadialGradient(radialFill, 2,
                                                               locations, rgbaData2);
   //Parameters for a gradient fill:
   //the gradient is 'pad-related' - we calculate everything in a pad's
   //space and consider it as a NDC (so pad's rect is (0,0), (1,1)).
   gradientFill1->SetCoordinateMode(TColorGradient::kPadMode);
   gradientFill1->SetStartEndR1R2(TColorGradient::Point(0.5, 0.5), 0.2,
                                  TColorGradient::Point(0.5, 0.5), 0.6);


   const UInt_t nSlices = 5;
   //Values for a TPie (non-const, that's how TPie's ctor is declared):
   Double_t values[nSlices] = {0.8, 1.2, 1.2, 0.8, 1.};
   Int_t colors[nSlices] = {radialFill, radialFill, radialFill,
                            radialFill, radialFill};

   TPie *pie = new TPie("pie", "TPie:", nSlices, values, colors);
   //One slice is slightly shifted:
   pie->SetEntryRadiusOffset(2, 0.05);
   //Move labels to the center (to fit the pad's space):
   pie->SetLabelsOffset(-0.08);
   //
   pie->SetRadius(0.4);
   pie->Draw("rsc");
}
