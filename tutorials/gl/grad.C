//Author: Timur Pocheptsov, 19/03/2014.
//This macro demonstrates how to create and use linear gradients to fill
//a histogram or a pad.
//To use this macro you need OpenGL enabled in pad:
//either set OpenGL.CanvasPreferGL to 1 in $ROOTSYS/etc/system.rootrc;
//or call gStyle->SetCanvasPreferGL(kTRUE); before canvas created.

//Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TStyle.h"
#include "TError.h"
#include "TH1F.h"

//Aux. functions for tutorials/gl.
#include "customcolorgl.h"

//______________________________________________________________________
void grad()
{
   //1. Try to 'allocate' five indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003, ... but
   //I prefer to find free indices in the ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Color_t colorIndices[5] = {};
   if (ROOT::GLTutorials::FindFreeCustomColorIndices(colorIndices) != 5) {
      ::Error("grad", "failed to create new custom colors");
      return;
   }

   //Make sure we enabled OpenGL support in a canvas.
   gStyle->SetCanvasPreferGL(kTRUE);

   //2. Test if canvas supports OpenGL:
   TCanvas * const cnv = new TCanvas("gradient demo 1", "gradient demo 1", 100, 100, 600, 600);
   if (!cnv->UseGL()) {
      ::Error("grad", "This macro requires OpenGL");
      delete cnv;
      return;
   }

   typedef TColorGradient::Point Point;

   //3. Create custom colors:
   //Linear gradient is defined by: 1) colors (to interpolate between them),
   //2) coordinates for these colors along the gradient axis [0., 1.] (must be sorted!).
   //3) Start and end points for a gradient, you specify them in some NDC rect ([0,0 - 1,1]),
   //and this rect is either: bounding rect of your polygon/object to fill
   //(gradient->SetCoordinateMode(TColorGradient::kObjectBoundingMode))
   //or bounding rect of a pad (gradient->SetCoordinateMode(TColorGradient::kPadMode)).
   //kObjectBoundingMode is the default one.

   const Color_t &frameGradient = colorIndices[2];//This gradient is a mixture of colorIndices[0] and colorIndices[1]
   //Fill color for a pad frame:
   {
      new TColor(colorIndices[0], 0.25, 0.25, 0.25, "special pad color1", 0.55);
      new TColor(colorIndices[1], 1., 1., 1., "special pad color2", 0.05);

      //dark-white-white-dark:
      const Double_t locations[] = {0., 0.2, 0.8, 1.};
      const Color_t gradientIndices[4] = {colorIndices[0], colorIndices[1], colorIndices[1], colorIndices[0]};

      //Gradient for a pad's frame.
      TLinearGradient * const gradFill1 = new TLinearGradient(frameGradient, 4, locations, gradientIndices);
      //Horizontal:
      gradFill1->SetStartEnd(Point(0., 0.), Point(1., 0.));
   }

   //This gradient is a mixture of two standard colors.
   const Color_t &padGradient = colorIndices[3];
   //Fill color for a pad:
   {
      const Double_t locations[] = {0., 1.};
      const Color_t gradientIndices[4] = {30, 38};//We create a gradient from system colors.

      //Gradient for a pad.
      TLinearGradient * const gradFill2 = new TLinearGradient(padGradient, 2, locations, gradientIndices);
      //Vertical:
      gradFill2->SetStartEnd(Point(0., 0.), Point(0., 1.));
   }

   //Another gradient built from three standard colors.
   const Color_t &histGradient = colorIndices[4];
   //Fill color for a histogram:
   {
      const Color_t gradientIndices[3] = {kYellow, kOrange, kRed};
      const Double_t locations[3] = {0., 0.5, 1.};

      //Gradient for a histogram.
      TLinearGradient * const gradFill3 = new TLinearGradient(histGradient, 3, locations, gradientIndices);
      //Vertical:
      gradFill3->SetStartEnd(Point(0., 0.), Point(0., 1.));
   }

   cnv->SetFillColor(padGradient);
   cnv->SetFrameFillColor(frameGradient);

   TH1F * const hist = new TH1F("a1", "b1", 20, -3., 3.);
   hist->SetFillColor(histGradient);
   hist->FillRandom("gaus", 100000);
   hist->Draw();
}
