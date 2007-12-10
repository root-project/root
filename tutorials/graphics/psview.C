// An example how to display PS, EPS, PDF files in canvas
// To load a PS file in a TCanvas, the ghostscript program needs to be install.
// On most unix systems it is usually installed. On Windows it has to be
// installed from http://pages.cs.wisc.edu/~ghost/
//Author: Valeriy Onoutchin
   
#include "TROOT.h"
#include "TCanvas.h"
#include "TImage.h"

void psview()
{
   // set to batch mode -> do not display graphics
   gROOT->SetBatch(1);

   // create a PostScript file
   gROOT->Macro("feynman.C");
   gPad->Print("feynman.ps");

   // back to graphics mode
   gROOT->SetBatch(0);

   // create an image from PS file
   TImage *ps = TImage::Open("feynman.ps");

   if (!ps) {
      printf("GhostScript (gs) program must be installed\n");
      return;
   }

   new TCanvas("psexam", "Example how to display PS file in canvas", 500, 650);
   ps->Draw("xxx"); 
}
