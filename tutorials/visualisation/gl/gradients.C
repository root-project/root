/// \file
/// \ingroup tutorial_gl
/// Features:
///  1. Radial and linear gradients
///  2. Transparent/semitransparent colours.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors  Timur Pocheptsov, Sergey Linev

// Includes for ACLiC:
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TStyle.h"
#include "TText.h"
#include "TPie.h"

void gradients(bool gl = true)
{
   // Find free colour indices in the ROOT's palette for:
   // 1. A radial gradient for TPie;
   // 2. A linear gradient for TCanvas
   // 3. A fully transparent fill color for a nested pad.

   gStyle->SetCanvasPreferGL(gl);

   auto c = new TCanvas("cpie", "Gradient colours demo", 700, 700);
   // Before we allocated any new colour or created any object:
   if (!c->UseGL() && !c->IsWeb())
      ::Warning("gradients", "This macro requires either OpenGL or Web canvas to correctly handle gradient colors");

   // Linear gradient is defined by: 1) colors (to interpolate between them),
   // 2) coordinates for these colors along the gradient axis [0., 1.] (must be sorted!).
   // 3) Start and end points for a gradient, you specify them in some NDC rect ([0,0 - 1,1]),
   // and this rect is either: bounding rect of your polygon/object to fill
   //(gradient->SetCoordinateMode(TColorGradient::kObjectBoundingMode))
   // or bounding rect of a pad (gradient->SetCoordinateMode(TColorGradient::kPadMode)).
   // kObjectBoundingMode is the default one.

   // Draw a text in the canvas (the object above the text will be
   // semi-transparent):
   auto t = new TText(0.05, 0.7, "Can you see the text?");
   t->Draw();

   // We create a nested pad on top to render a TPie in,
   // this way we still have a text (below) + TPie with
   // a fancy colour on top.
   auto pad = new TPad("p", "p", 0., 0., 1., 1.);

   // TPad itself is fully transparent:
   auto transparentFill = TColor::GetColor((Float_t)1., 1., 1., 0.);
   pad->SetFillColor(transparentFill);
   // Add our pad into the canvas:
   pad->Draw();
   pad->cd();

   // Radial gradient fill for a TPie object:
   auto col3 = TColor::GetColor((Float_t)1., 0.8, 0., 1.);   /*opaque orange at the start:*/
   auto col4 = TColor::GetColor((Float_t)1., 0.2, 0., 0.65); /*transparent red at the end:*/

   //'Simple' radial gradient with radius 0.4
   auto radialFill = TColor::GetRadialGradient(0.4, {col3, col4});

   // Linear gradient fill (with an axis angle == 45):
   auto col1 = TColor::GetColor((Float_t)0.2, 0.2, 0.2);           /*gray*/
   auto col2 = TColor::GetColor((Float_t)0.8, 1., 0.9);            /*pale green*/
   auto linearFill = TColor::GetLinearGradient(45., {col1, col2}); // 45 degrees:

   // Set as a background color in the canvas:
   c->SetFillColor(linearFill);

   const UInt_t nSlices = 5;
   // Values for a TPie (non-const, that's how TPie's ctor is declared):
   Double_t values[nSlices] = {0.8, 1.2, 1.2, 0.8, 1.};
   Int_t colors[nSlices] = {radialFill, radialFill, radialFill, radialFill, radialFill};

   TPie *const pie = new TPie("pie", "TPie:", nSlices, values, colors);
   // One slice is slightly shifted:
   pie->SetEntryRadiusOffset(2, 0.05);
   // Move labels to the center (to fit the pad's space):
   pie->SetLabelsOffset(-0.08);
   //
   pie->SetRadius(0.4);
   pie->Draw("rsc");
}
