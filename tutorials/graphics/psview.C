/// \file
/// \ingroup tutorial_graphics
/// \notebook
/// An example how to display PS, EPS, PDF files in canvas.
/// To load a PS file in a TCanvas, the ghostscript program needs to be install.
/// - On most unix systems it is installed by default.
/// - On Windows it has to be installed from http://pages.cs.wisc.edu/~ghost/
///   also the place where gswin32c.exe sits should be added in the PATH. One
///   way to do it is:
///     1. Start the Control Panel
///     2. Double click on System
///     3, Open the "Advanced" tab
///     4. Click on the "Environment Variables" button
///     5. Find "Path" in "System variable list", click on it.
///     6. Click on the "Edit" button.
///     7. In the "Variable value" field add the path of gswin32c
///        (after a ";") it should be something like:
///        "C:\Program Files\gs\gs8.13\bin"
///     8. click "OK" as much as needed.
///
/// \macro_code
///
/// \author Valeriy Onoutchin

#include "TROOT.h"
#include "TCanvas.h"
#include "TImage.h"

void psview()
{
   // set to batch mode -> do not display graphics
   gROOT->SetBatch(1);

   // create a PostScript file
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/graphics/feynman.C");
   gROOT->Macro(dir);
   gPad->Print("feynman.eps");

   // back to graphics mode
   gROOT->SetBatch(0);

   // create an image from PS file
   TImage *ps = TImage::Open("feynman.eps");

   if (!ps) {
      printf("GhostScript (gs) program must be installed\n");
      return;
   }

   new TCanvas("psexam", "Example how to display PS file in canvas", 600, 400);
   TLatex *tex = new TLatex(0.06,0.9,"The picture below has been loaded from a PS file:");
   tex->Draw();

   TPad *eps = new TPad("eps", "eps", 0., 0., 1., 0.75);
   eps->Draw();
   eps->cd();
   ps->Draw("xxx");
}
