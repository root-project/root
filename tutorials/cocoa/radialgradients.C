//Author: Timur Pocheptsov, 25/03/2014.
//This tutorial demonstrates how to use radial gradients,
//custom colors, transparency.
//Requires ROOT built for OS X with --enable-cocoa.


//Includes for ACLiC:
#include <cassert>
#include <cstdlib>

#include "TColorGradient.h"
#include "TVirtualX.h"
#include "TEllipse.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "TError.h"

//Cocoa aux. functions.
#include "customcolor.h"

namespace {

//Just some colors (rgba) to build our
//fancy gradients from.
const Double_t basicColors[][4] =
{
{1., 0.,  0., 1.},
{1., 0.3, 0., 1.},
{0., 0.,  1., 1.},
{1., 1.,  0., 1.},
{1., 0.,  1., 1.},
{0., 1.,  1., 1.},
{0., 1.,  0., 1.},
{0., 0.5,  0., 1.},
//transparent colors:
{1., 0.,  0., 0.5},
{1., 0.3, 0., 0.5},
{0., 0.,  1., 0.5},
{1., 1.,  0., 0.5},
{1., 0.,  1., 0.5},
{0., 1.,  1., 0.5},
{0., 1.,  0., 0.5},
{0., 0.5,  0., 0.5},
//and even more transparent:
{1., 0.,  0., 0.2},
{1., 0.3, 0., 0.2},
{0., 0.,  1., 0.2},
{1., 1.,  0., 0.2},
{1., 0.,  1., 0.2},
{0., 1.,  1., 0.2},
{0., 1.,  0., 0.2},
{0., 0.5,  0., 0.2}
};

const unsigned nBasicColors = sizeof basicColors / sizeof basicColors[0];

//______________________________________________________________________
Color_t CreateRandomGradientFill()
{
   Color_t idx[1] = {};
   if (ROOT::CocoaTutorials::FindFreeCustomColorIndices(idx) != 1)
      return -1;

   const Double_t * const fromRGBA = basicColors[(std::rand() % (nBasicColors / 2))];
   //With odd number of colors the last one is never selected :)
   const Double_t * const toRGBA = basicColors[nBasicColors / 2 + (std::rand() % (nBasicColors / 2))];

   const Double_t locations[] = {0., 1.};
   const Double_t rgbas[] = {fromRGBA[0], fromRGBA[1], fromRGBA[2], fromRGBA[3],
                             toRGBA[0], toRGBA[1], toRGBA[2], toRGBA[3]};

   TRadialGradient * const grad = new TRadialGradient(idx[0], 2, locations, rgbas);
   grad->SetRadialGradient(TColorGradient::Point(0.5, 0.5), 0.5);

   return idx[0];
}

//______________________________________________________________________
bool add_ellipse(const Double_t xC, const Double_t yC, const Double_t r)
{
   assert(gPad != nullptr && "add_ellipse, no pad to add ellipse");

   const Color_t newColor = CreateRandomGradientFill();
   if (newColor == -1) {
      ::Error("add_ellipse", "failed to find a new color index for a gradient fill");
      return false;
   }

   TEllipse * const newEllipse = new TEllipse(xC, yC, r, r);
   newEllipse->SetFillColor(newColor);
   newEllipse->Draw();

   return true;
}

}

//______________________________________________________________________
void radialgradients()
{
   gRandom->SetSeed(4357);//;)

   TCanvas * const cnv = new TCanvas("radial gradients", "radial gradients", 800, 800);
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("radialgradients",
              "this demo requires OS X and ROOT built with --enable-cocoa");
      delete cnv;
      return;
   }

   for (unsigned i = 0; i < 100; ++i)
      if (!add_ellipse(gRandom->Rndm(), gRandom->Rndm(), 0.5 * gRandom->Rndm()))
         break;

   cnv->Modified();
   cnv->Update();
}
