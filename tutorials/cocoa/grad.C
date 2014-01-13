//This macro shows how to create and use linear gradients to fill
//a histogram or a pad.
//It works ONLY on MacOS X with cocoa graphical back-end.
//This macro DOES NOT demonstrate a good memory management/error handling.


//These includes are for ACLiC.
#include "TColorGradient.h"
#include "TVirtualX.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TError.h"
#include "TH1F.h"

#include "freecustomcolor.h"

namespace ROOT {
namespace CocoaTutorials {

//______________________________________________________________________
Int_t create_pad_frame_gradient()
{
   //We create a gradient with 4 steps - from dark (and semi-transparent)
   //gray to almost transparent (95%) white and to white and to dark gray again.
   
   //Unfortunately, if function fails at some stage, we are not deleting custom colors created
   //before this failure.
   
   const Double_t locations[] = {0., 0.2, 0.8, 1.};
   
   const Int_t color1Index = FindFreeCustomColorIndex();
   if (color1Index == -1) {
      ::Error("create_pad_frame_gradient", "no free index found for a custom gray color");
      return -1;
   }
   
   new TColor(color1Index, 0.25, 0.25, 0.25, "special pad color1", 0.55);
   
   const Int_t color2Index = FindFreeCustomColorIndex();
   if (color2Index == -1) {
      ::Error("create_pad_frame_gradient", "no free index found for a custom white color");
      return -1;
   }
   
   new TColor(color2Index, 1., 1., 1., "special pad color2", 0.05);
   
   const Int_t gradientIndex = FindFreeCustomColorIndex();
   if (gradientIndex == -1) {
      ::Error("create_pad_frame_gradient", "no free index found for a resulting gradient color");
      return -1;
   }
   
   const Color_t colorIndices[4] = {color1Index, color2Index, color2Index, color1Index};
   new TColorGradient(gradientIndex, TColorGradient::kGDHorizontal, 4, locations, colorIndices);
   
   return gradientIndex;
}

//______________________________________________________________________
Int_t create_pad_gradient()
{
   //We create two-steps gradient from ROOT's standard colors (38 and 30).
   const Int_t gradientIndex = FindFreeCustomColorIndex();
   if (gradientIndex != -1) {
      const Double_t locations[] = {0., 1.};
      const Color_t colorIndices[4] = {38, 30};
      new TColorGradient(gradientIndex, TColorGradient::kGDVertical, 2, locations, colorIndices);
   } else
      ::Error("create_pad_gradient", "no free index found for a gradient color");
   
   return gradientIndex;
}

//______________________________________________________________________
Int_t create_histo_gradient()
{
   const Int_t gradientIndex = FindFreeCustomColorIndex();
   if (gradientIndex != -1) {
      //Gradient to fill a histogramm
      const Color_t colorIndices[3] = {kRed, kOrange, kYellow};
      const Double_t lengths[3] = {0., 0.5, 1.};
      new TColorGradient(gradientIndex, TColorGradient::kGDVertical, 3, lengths, colorIndices);
   } else
      ::Error("create_histo_gradient", "no free index found for a gradient color");
   
   return gradientIndex;
}

}//CocoaTutorials
}//ROOT

//______________________________________________________________________
void grad()
{
   using namespace ROOT::CocoaTutorials;

   const Int_t padFrameColorIndex = create_pad_frame_gradient();
   if (padFrameColorIndex == -1)//error message was issued already.
      return;
   
   const Int_t padColorIndex = create_pad_gradient();
   if (padColorIndex == -1)//error message was printed by create_pad_gradient.
      return;
   
   const Int_t fillColorIndex = create_histo_gradient();
   if (fillColorIndex == -1)
      return;

   TCanvas * const cnv = new TCanvas("cnv", "gradient test", 100, 100, 600, 600);
   //After canvas was created, gVirtualX should be non-null.
   if (gVirtualX && !gVirtualX->InheritsFrom("TGCocoa")) {
      ::Error("grad", "This macro works only on MacOS X with --enable-cocoa");
      delete cnv;
      return;
   }

   cnv->SetFillColor(padColorIndex);
   cnv->SetFrameFillColor(padFrameColorIndex);
   
   TH1F * const hist = new TH1F("a", "b", 20, -3., 3.);
   hist->SetFillColor(fillColorIndex);
   hist->FillRandom("gaus", 100000);
   hist->Draw();

   cnv->Update();
}
