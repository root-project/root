/// \file
/// \ingroup tutorial_gl
/// This tutorial demonstrates how to use radial gradients,
/// custom colors, transparency.
/// Requires OpenGL or Web-based canvas
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors Timur Pocheptsov, Sergey Linev

// Includes for ACLiC:
#include "TColorGradient.h"
#include "TEllipse.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TError.h"

//______________________________________________________________________
Color_t CreateRandomGradientFill()
{
   std::vector<Int_t> colors;

   for (int n = 0; n < 2; ++n)
      colors.emplace_back(gRandom->Integer(10) + 2);

   auto indx = TColor::GetRadialGradient(0.5, colors);

   // example how to modify gradient
   auto gradient = dynamic_cast<TRadialGradient *>(gROOT->GetColor(indx));
   if (gradient) {
      // change center and radius
      gradient->SetRadialGradient({0.3, 0.3}, 0.7);
      // change alpha parameter for the colors

      gradient->SetColorAlpha(0, 0.2 + gRandom->Rndm() * 0.8);
      gradient->SetColorAlpha(1, 0.2 + gRandom->Rndm() * 0.8);
   } else {
      ::Error("CreateRandomGradientFill", "failed to find new gradient color with index %d", indx);
   }

   return indx;
}

//______________________________________________________________________
bool add_ellipse(const Double_t xC, const Double_t yC, const Double_t r)
{
   const Color_t newColor = CreateRandomGradientFill();
   if (newColor == -1) {
      ::Error("add_ellipse", "failed to find a new color index for a gradient fill");
      return false;
   }

   TEllipse *const newEllipse = new TEllipse(xC, yC, r, r);
   newEllipse->SetFillColor(newColor);
   newEllipse->Draw();

   return true;
}

//______________________________________________________________________
void radialgradients(bool gl = true)
{
   gRandom->SetSeed(4357);

   gStyle->SetCanvasPreferGL(gl);

   auto cnv = new TCanvas("radialgradients", "radial gradients", 800, 800);
   if (!cnv->UseGL() && !cnv->IsWeb())
      ::Warning("radialgradients",
                "This macro requires either OpenGL or Web canvas to correctly handle gradient colors");

   for (unsigned i = 0; i < 100; ++i)
      add_ellipse(gRandom->Rndm(), gRandom->Rndm(), 0.5 * gRandom->Rndm());
}
