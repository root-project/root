// An example how to display PS, EPS, PDF files in canvas
//Author: Valeriy Onoutchin
   
#include "TROOT.h"
#include "TCanvas.h"
#include "TImage.h"

void psview()
{
   // set to batch  mode -> do not display graphics
   gROOT->SetBatch(1);

   // create PostScript file psexam.ps
   gROOT->Macro("psexam.C");

   // back to graphics mode
   gROOT->SetBatch(0);

   // create an image from PS file
   TImage *ps = TImage::Open("psexam.ps");

   if (!ps) {
      printf("GhostScript (gs) program must be installed\n");
      return;
   }

   new TCanvas("psexam", "Example how to display PS file in canvas", 500, 650);
   ps->Draw("xxx"); 
}
