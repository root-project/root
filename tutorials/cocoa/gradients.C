/// \file
/// \ingroup tutorial_cocoa
/// This macro requires OS X and ROOT
/// compiled with --enable-cocoa to run.
///
/// Features:
///  1. Radial and linear gradients
///  2. Transparent/semitransparent colours.
///  3. Shadows.
///
/// \macro_code
///
/// \author Timur Pocheptsov

//Includes for ACLiC:
#include "TColorGradient.h"
#include "TVirtualX.h"
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

   Color_t colorIndices[3] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(colorIndices) != 3) {
      ::Error("grad", "failed to create new custom colors");
      return;
   }

   //Better names:
   const Color_t &radialFill = colorIndices[0];
   const Color_t &linearFill = colorIndices[1];
   const Color_t &transparentFill = colorIndices[2];

   //Create a canvas to check if we have a right back-end which supports gradients:
   TCanvas * const c = new TCanvas("cpie","Gradient colours demo", 700, 700);
   //Before we allocated any new colour or created any object:
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("gradients", "This macro requires OS X and ROOT built with --enable-cocoa");
      delete c;
      return;
   }

   //Linear gradient is defined by: 1) colors (to interpolate between them),
   //2) coordinates for these colors along the gradient axis [0., 1.] (must be sorted!).
   //3) Start and end points for a gradient, you specify them in some NDC rect ([0,0 - 1,1]),
   //and this rect is either: bounding rect of your polygon/object to fill
   //(gradient->SetCoordinateMode(TColorGradient::kObjectBoundingMode))
   //or bounding rect of a pad (gradient->SetCoordinateMode(TColorGradient::kPadMode)).
   //kObjectBoundingMode is the default one.


   //Colour positions in the gradient's palette (here I place colors at the
   //ends of 0-1):
   const Double_t locations[] = {0., 1.};
   //Linear gradient fill (with an axis angle == 45):
   const Double_t rgbaData1[] = {0.2, 0.2, 0.2, 1.,/*gray*/
                                 0.8, 1., 0.9, 1.  /*pale green*/};
   TLinearGradient * const gradientFill1 = new TLinearGradient(linearFill, 2, locations, rgbaData1);
   //45 degrees:
   gradientFill1->SetStartEnd(TColorGradient::Point(0., 0.), TColorGradient::Point(1., 1.));
   //Set as a background color in the canvas:
   c->SetFillColor(linearFill);

   //Draw a text in the canvas (the object above the text will be
   //semi-transparent):
   TText * const t = new TText(0.05, 0.7, "Can you see the text?");
   t->Draw();

   //We create a nested pad on top to render a TPie in,
   //this way we still have a text (below) + TPie with
   //a fancy colour on top.
   TPad * const pad = new TPad("p", "p", 0., 0., 1., 1.);

   //TPad itself is fully transparent:
   new TColor(transparentFill, 1., 1., 1., "transparent_fill_color", 0.);
   pad->SetFillColor(transparentFill);
   //Add our pad into the canvas:
   pad->Draw();
   pad->cd();

   //Radial gradient fill for a TPie object:
   const Double_t rgbaData2[] = {/*opaque orange at the start:*/1., 0.8, 0., 1.,
                                 /*transparent red at the end:*/1., 0.2, 0., 0.8};

   //
   //With Quartz/Cocoa we support the "extended" radial gradient:
   //you can specify two centers and two radiuses - the start and
   //the end of your radial gradient (+ colors/positions as with a linear
   //gradient).
   //

   TRadialGradient * const gradientFill2 = new TRadialGradient(radialFill, 2,
                                                               locations, rgbaData2);
   //Parameters for a gradient fill:
   //the gradient is 'pad-related' - we calculate everything in a pad's
   //space and consider it as a NDC (so pad's rect is (0,0), (1,1)).
   gradientFill2->SetCoordinateMode(TColorGradient::kPadMode);
   //Centers for both circles are the same point.
   gradientFill2->SetStartEndR1R2(TColorGradient::Point(0.5, 0.5), 0.1,
                                  TColorGradient::Point(0.5, 0.5), 0.4);

   const UInt_t nSlices = 5;
   //Values for a TPie (non-const, that's how TPie's ctor is declared):
   Double_t values[nSlices] = {0.8, 1.2, 1.2, 0.8, 1.};
   Int_t colors[nSlices] = {radialFill, radialFill, radialFill,
                            radialFill, radialFill};

   TPie * const pie = new TPie("pie", "TPie:", nSlices, values, colors);
   //One slice is slightly shifted:
   pie->SetEntryRadiusOffset(2, 0.05);
   //Move labels to the center (to fit the pad's space):
   pie->SetLabelsOffset(-0.08);
   //
   pie->SetRadius(0.4);
   pie->Draw("rsc");
}
