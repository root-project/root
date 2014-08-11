//Author: Timur Pocheptsov, 19/03/2014.
//Gradient fill with transparency and the "SAME" option.
//To use this macro you need OpenGL enabled in pad:
//either set OpenGL.CanvasPreferGL to 1 in $ROOTSYS/etc/system.rootrc;
//or call gStyle->SetCanvasPreferGL(kTRUE); before canvas created.

//Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TColor.h"
#include "TStyle.h"
#include "TH1F.h"

//Aux. functions for tutorials/gl.
#include "customcolorgl.h"

void grad2()
{
   //1. 'Allocate' four indices for our custom colors.
   //We can use hard-coded indices like 1001, 1002, 1003 ... but
   //I prefer to find free indices in the ROOT's color table
   //to avoid possible conflicts with other tutorials.
   Color_t freeIndices[4] = {};
   if (ROOT::GLTutorials::FindFreeCustomColorIndices(freeIndices) != 4) {
      ::Error("grad2", "can not allocate new custom colors");
      return;
   }

   //'Aliases' (instead of freeIndices[someIndex])
   const Color_t customRed = freeIndices[0], grad1 = freeIndices[1];
   const Color_t customGreen = freeIndices[2], grad2 = freeIndices[3];

   //Make sure canvas supports OpenGL.
   gStyle->SetCanvasPreferGL(kTRUE);

   //2. Check that we have a canvas with an OpenGL support.
   TCanvas * const cnv = new TCanvas("gradiend demo 2", "gradient demo 2", 100, 100, 800, 600);
   if (!cnv->UseGL()) {
      ::Error("grad2", "This macro requires OpenGL");
      delete cnv;
      return;
   }

   //3. Custom colors:
   //   a) Custom semi-transparent red.
   new TColor(customRed, 1., 0., 0., "red", 0.5);



   //   b) Gradient (from our semi-transparent red to ROOT's kOrange).
   //      Linear gradient is defined by: 1) colors (to interpolate between them),
   //      2) coordinates for these colors along the gradient axis [0., 1.] (must be sorted!).
   //      3) Start and end points for a gradient, you specify them in some NDC rect ([0,0 - 1,1]),
   //         and this rect is either: bounding rect of your polygon/object to fill
   //         (gradient->SetCoordinateMode(TColorGradient::kObjectBoundingMode))
   //         or bounding rect of a pad (gradient->SetCoordinateMode(TColorGradient::kPadMode)).
   //         kObjectBoundingMode is the default one.

   const Double_t locations[] = {0., 1.};
   const Color_t idx1[] = {customRed, kOrange};
   TLinearGradient * const gradFill1 = new TLinearGradient(grad1, 2, locations, idx1);

   typedef TColorGradient::Point Point;
   //Starting and ending points for a gradient fill (it's a vertical gradient):
   gradFill1->SetStartEnd(Point(0., 0.), Point(0., 1.));

   //   c) Custom semi-transparent green.
   new TColor(customGreen, 0., 1., 0., "green", 0.5);

   //   d) Gradient from ROOT's kBlue to our custom green.
   const Color_t idx2[] = {customGreen, kBlue};

   TLinearGradient * const gradFill2 = new TLinearGradient(grad2, 2, locations, idx2);
   //Vertical gradient fill.
   gradFill2->SetStartEnd(Point(0., 0.), Point(0., 1.));

   TH1F * const hist = new TH1F("a2", "b2", 10, -2., 3.);
   TH1F * const hist2 = new TH1F("c3", "d3", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(grad1);
   hist2->SetFillColor(grad2);

   hist2->Draw();
   hist->Draw("SAME");
}
