/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <TPieSlice.h>

#include <Riostream.h>
#include <TError.h>
#include <TROOT.h>
#include <TVirtualPad.h>
#include <TArc.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TPaveText.h>
#include <TH1.h>

ClassImp(TPieSlice)


//______________________________________________________________________________
//
// A slice of a piechart, see the TPie class.
//
// This class describe the property of single
//


//______________________________________________________________________________
TPieSlice::TPieSlice() : TNamed(), TAttFill(), TAttLine()
{
   // This is the default constructor, used to create the standard.

   fPie = 0;
   fValue = 1;
   fRadiusOffset = 0;
   fIsActive = kFALSE;
}


//______________________________________________________________________________
TPieSlice::TPieSlice(const char *name, const char *title,
                     TPie *pie, Double_t val) :
                     TNamed(name, title), TAttFill(), TAttLine()
{
   // This constructor create a slice with a particular value.

   fPie = pie;
   fValue = val;
   fRadiusOffset = 0;
   fIsActive = kFALSE;
}


//______________________________________________________________________________
Int_t TPieSlice::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
   // Eval if the mouse is over the area associated with this slice.

   Int_t dist = 9999;

   if (fIsActive) {
      dist = 0;
      fIsActive = kFALSE;
      gPad->SetCursor(kHand);
   }

   return dist;
}


//______________________________________________________________________________
Double_t TPieSlice::GetRadiusOffset()
{
   // return the value of the offset in radial direction for this slice.

   return fRadiusOffset;
}


//______________________________________________________________________________
Double_t TPieSlice::GetValue()
{
   // Return the value of this slice.

   return fValue;
}


//______________________________________________________________________________
void TPieSlice::SavePrimitive(ostream &/*out*/, Option_t * /*opts*/)
{
   // Do nothing.
}


//______________________________________________________________________________
void TPieSlice::SetRadiusOffset(Double_t val)
{
   // Set the radial offset of this slice.

   fRadiusOffset = val;
   if (fRadiusOffset<.0) fRadiusOffset = .0;
}


//______________________________________________________________________________
void TPieSlice::SetValue(Double_t val)
{
   // Set the value for this slice.
   // Negative values are changed with its absolute value.

   fValue = val;
   if (fValue<.0) {
      Warning("SetValue","Invalid negative value. Absolute value taken");
      fValue *= -1;
   }

   fPie->MakeSlices(kTRUE);
}
