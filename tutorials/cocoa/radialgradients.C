//This tutorial demonstrates how to use radial gradients,
//custom colors, transparency.
//Requires ROOT built for OS X with --enable-cocoa.


//Includes for ACLiC:
#include <cassert>
#include <cstdlib>

#include "TColorGradient.h"
#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "TEllipse.h"
#include "TRandom.h"
#include "TCanvas.h"
#include "Rtypes.h"
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
   const Double_t *fromRGBA = basicColors[(std::rand() % (nBasicColors / 2))];
   const Double_t *toRGBA = basicColors[nBasicColors / 2 + (std::rand() % (nBasicColors / 2))];

   const Double_t locations[] = {0., 1.};
   const Double_t rgbas[] = {fromRGBA[0], fromRGBA[1], fromRGBA[2], fromRGBA[3],
                             toRGBA[0], toRGBA[1], toRGBA[2], toRGBA[3]};

   Color_t idx[1] = {};
   //I do not check the result here, too lazy:
   ROOT::CocoaTutorials::FindFreeCustomColorIndices(idx);
   TRadialGradient * const grad = new TRadialGradient(idx[0], 2, locations, rgbas);
   grad->SetRadialGradient(TColorGradient::Point(0.5, 0.5), 0.5);

   return idx[0];
}

//______________________________________________________________________
void add_ellipse(const Double_t xC, const Double_t yC, const Double_t r)
{
   assert(gPad != nullptr && "add_ellipse, no pad to add ellipse");

   TEllipse * const newEllipse = new TEllipse(xC, yC, r, r);
   
   newEllipse->SetFillColor(CreateRandomGradientFill());
   newEllipse->Draw();
}

}

//______________________________________________________________________
void radialgradients()
{
   TCanvas * const cnv = new TCanvas("radial gradients", "radial gradients", 800, 800);
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("radialgradients",
              "this demo requires OS X and ROOT built with --enable-cocoa");
      delete cnv;
      return;
   }

   for (unsigned i = 0; i < 100; ++i)
      add_ellipse(gRandom->Rndm(), gRandom->Rndm(), 0.5 * gRandom->Rndm());
   
   cnv->Modified();
   cnv->Update();
}
