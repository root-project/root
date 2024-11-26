/// \file
/// \ingroup tutorial_gl
/// Gradient fill with transparency and the "SAME" option.
/// To use this macro you need OpenGL enabled in pad:
/// either set OpenGL.CanvasPreferGL to 1 in $ROOTSYS/etc/system.rootrc;
/// or call `gStyle->SetCanvasPreferGL(kTRUE);` before canvas created.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors  Timur Pocheptsov, Sergey Linev

// Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TError.h"
#include "TColor.h"
#include "TStyle.h"
#include "TH1F.h"

void grad2(bool gl = true)
{
   // Make sure canvas supports OpenGL.
   gStyle->SetCanvasPreferGL(gl);

   // 2. Check that we have a canvas with an OpenGL support.
   auto cnv = new TCanvas("grad2", "gradient demo 2", 100, 100, 800, 600);
   if (!cnv->UseGL() && !cnv->IsWeb())
      ::Warning("grad2", "This macro requires either OpenGL or Web canvas to correctly handle gradient colors");

   // 3. Custom colors:
   //    a) Custom semi-transparent red.
   auto customRed = TColor::GetColor((Float_t)1., 0., 0., 0.5);

   //  Custom semi-transparent green.
   auto customGreen = TColor::GetColor((Float_t)0., 1., 0., 0.5);

   // 4. Linear gradient colors
   //   b) Gradient (from our semi-transparent red to ROOT's kOrange).
   //      Linear gradient is defined by: 1) angle in grad
   //      2) colors (to interpolate between them),
   //  If necessary, TLinearGradient object can be retrieved and modified later

   auto grad1 = TColor::GetLinearGradient(90., {customRed, kOrange});

   // Vertical gradient fill.
   auto grad2 = TColor::GetLinearGradient(90., {customGreen, kBlue});

   auto hist = new TH1F("a2", "b2", 10, -2., 3.);
   auto hist2 = new TH1F("c3", "d3", 10, -3., 3.);
   hist->FillRandom("landau", 100000);
   hist2->FillRandom("gaus", 100000);

   hist->SetFillColor(grad1);
   hist2->SetFillColor(grad2);

   hist2->Draw();
   hist->Draw("SAME");
}
