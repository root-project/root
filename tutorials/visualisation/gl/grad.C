/// \file
/// \ingroup tutorial_gl
/// This macro demonstrates how to create and use linear gradients to fill
/// a histogram or a pad.
///
/// \macro_image(nobatch)
/// \macro_code
///
/// \authors  Timur Pocheptsov, Sergey Linev

// Includes for ACLiC (cling does not need them).
#include "TColorGradient.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TStyle.h"
#include "TError.h"
#include "TH1F.h"

//______________________________________________________________________
void grad(bool use_gl = true)
{
   // Make sure we enabled OpenGL support in a canvas.
   gStyle->SetCanvasPreferGL(use_gl);

   // Test if canvas supports OpenGL:
   TCanvas *cnv = new TCanvas("grad", "gradient demo 1", 100, 100, 600, 600);
   if (!cnv->UseGL() && !cnv->IsWeb())
      ::Warning("grad", "This macro requires either OpenGL or Web canvas to correctly handle gradient colors");

   // Create custom linear gradients.
   // Linear gradient is defined by:
   // 1) Direction in which gradient is changing (defined as angle in grad)
   // 2) colors (to interpolate between them), at least two of them
   // 3) alpha parameter for the colors (if not specified - used from TColor directly)
   // 4) coordinates for these colors along the gradient axis [0., 1.] (must be sorted!).

   auto fcol1 = TColor::GetColor((Float_t)0.25, 0.25, 0.25, 0.55); // special frame color 1
   auto fcol2 = TColor::GetColor((Float_t)1., 1., 1., 0.05);       // special frame color 2

   auto frameGradient = TColor::GetLinearGradient(0., {fcol1, fcol2, fcol2, fcol1}, {0., 0.2, 0.8, 1.});

   // This gradient is a mixture of two standard colors.
   auto padGradient = TColor::GetLinearGradient(0., {30, 38});

   // Another gradient built from three standard colors.
   auto histGradient = TColor::GetLinearGradient(45., {kYellow, kOrange, kRed});

   // Example of radial gradient, for stats box works properly only in web canvas
   // Here first argument is radius [0..1] and then list of colors
   auto statGradient = TColor::GetRadialGradient(0.5, {kBlue, kGreen});

   gStyle->SetStatColor(statGradient);

   cnv->SetFillColor(padGradient);
   cnv->SetFrameFillColor(frameGradient);

   TH1F *const hist = new TH1F("a1", "b1", 20, -3., 3.);
   hist->SetFillColor(histGradient);
   hist->FillRandom("gaus", 10000);
   hist->Draw();
}
