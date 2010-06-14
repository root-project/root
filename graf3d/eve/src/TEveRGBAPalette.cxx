// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveRGBAPalette.h"

#include "TColor.h"
#include "TStyle.h"
#include "TMath.h"

//______________________________________________________________________________
//
// A generic, speed-optimised mapping from value to RGBA color
// supporting different wrapping and range truncation modes.
//
// Flag fFixColorRange: specifies how the palette is mapped to signal values:
//  true  - LowLimit -> HighLimit
//  false - MinValue -> MaxValue


ClassImp(TEveRGBAPalette);

//______________________________________________________________________________
TEveRGBAPalette::TEveRGBAPalette() :
   TObject(), TQObject(),
   TEveRefCnt(),

   fLowLimit(0), fHighLimit(0), fMinVal(0), fMaxVal(0),

   fInterpolate     (kTRUE),
   fShowDefValue    (kTRUE),
   fFixColorRange   (kFALSE),
   fUnderflowAction (kLA_Cut),
   fOverflowAction  (kLA_Clip),

   fDefaultColor(-1),
   fUnderColor  (-1),
   fOverColor   (-1),

   fNBins(0), fCAMin(0), fCAMax(0), fColorArray(0)
{
   // Constructor.

   SetLimits(0, 1024);
   SetMinMax(0,  512);

   SetDefaultColor(0);
   SetUnderColor(1);
   SetOverColor(2);
}

//______________________________________________________________________________
TEveRGBAPalette::TEveRGBAPalette(Int_t min, Int_t max, Bool_t interp,
                                 Bool_t showdef, Bool_t fixcolrng) :
   TObject(), TQObject(),
   TEveRefCnt(),

   fLowLimit(0), fHighLimit(0), fMinVal(0), fMaxVal(0),

   fInterpolate     (interp),
   fShowDefValue    (showdef),
   fFixColorRange   (fixcolrng),
   fUnderflowAction (kLA_Cut),
   fOverflowAction  (kLA_Clip),

   fDefaultColor(-1),
   fUnderColor  (-1),
   fOverColor   (-1),

   fNBins(0), fCAMin(0), fCAMax(0), fColorArray(0)
{
   // Constructor.

   SetLimits(min, max);
   SetMinMax(min, max);

   SetDefaultColor(0);
   SetUnderColor(1);
   SetOverColor(2);
}

//______________________________________________________________________________
TEveRGBAPalette::~TEveRGBAPalette()
{
   // Destructor.

   delete [] fColorArray;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetupColor(Int_t val, UChar_t* pixel) const
{
   // Set RGBA color 'pixel' for signal-value 'val'.

   using namespace TMath;
   Float_t div  = Max(1, fCAMax - fCAMin);
   Int_t   nCol = gStyle->GetNumberOfColors();

   Float_t f;
   if      (val >= fCAMax) f = nCol - 1;
   else if (val <= fCAMin) f = 0;
   else                     f = (val - fCAMin)/div*(nCol - 1);

   if (fInterpolate) {
      Int_t  bin = (Int_t) f;
      Float_t f1 = f - bin, f2 = 1.0f - f1;
      TEveUtil::ColorFromIdx(f1, gStyle->GetColorPalette(bin),
                             f2, gStyle->GetColorPalette(Min(bin + 1, nCol - 1)),
                             pixel);
   } else {
      TEveUtil::ColorFromIdx(gStyle->GetColorPalette((Int_t) Nint(f)), pixel);
   }
}

//______________________________________________________________________________
void TEveRGBAPalette::SetupColorArray() const
{
   // Construct internal color array that maps signal value to RGBA color.

   if (fColorArray)
      delete [] fColorArray;

   if (fFixColorRange) {
      fCAMin = fLowLimit; fCAMax = fHighLimit;
   } else {
      fCAMin = fMinVal;   fCAMax = fMaxVal;
   }
   fNBins = fCAMax - fCAMin + 1;

   fColorArray = new UChar_t [4 * fNBins];
   UChar_t* p = fColorArray;
   for(Int_t v = fCAMin; v <= fCAMax; ++v, p+=4)
      SetupColor(v, p);
}

//______________________________________________________________________________
void TEveRGBAPalette::ClearColorArray()
{
   // Clear internal color array.

   if (fColorArray) {
      delete [] fColorArray;
      fColorArray = 0;
      fNBins = fCAMin = fCAMax = 0;
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetLimits(Int_t low, Int_t high)
{
   // Set low/high limits on signal value. Current min/max values are
   // clamped into the new limits.

   fLowLimit  = low;
   fHighLimit = high;

   if (fMaxVal < fLowLimit)  SetMax(fLowLimit);
   if (fMinVal < fLowLimit)  SetMin(fLowLimit);
   if (fMinVal > fHighLimit) SetMin(fHighLimit);
   if (fMaxVal > fHighLimit) SetMax(fHighLimit);

   ClearColorArray();
}

//______________________________________________________________________________
void TEveRGBAPalette::SetLimitsScaleMinMax(Int_t low, Int_t high)
{
   // Set low/high limits and rescale current min/max values.

   Float_t rng_old = fHighLimit - fLowLimit;
   Float_t rng_new = high - low;

   fMinVal = TMath::Nint(low + (fMinVal - fLowLimit)*rng_new/rng_old);
   fMaxVal = TMath::Nint(low + (fMaxVal - fLowLimit)*rng_new/rng_old);
   fLowLimit  = low;
   fHighLimit = high;

   ClearColorArray();
}

//______________________________________________________________________________
void TEveRGBAPalette::SetMin(Int_t min)
{
   // Set current min value.

   fMinVal = TMath::Min(min, fMaxVal);
   ClearColorArray();
}

//______________________________________________________________________________
void TEveRGBAPalette::SetMax(Int_t max)
{
   // Set current max value.

   fMaxVal = TMath::Max(max, fMinVal);
   ClearColorArray();
}

//______________________________________________________________________________
void TEveRGBAPalette::SetMinMax(Int_t min, Int_t max)
{
   // Set current min/max values.

   fMinVal = min;
   fMaxVal = max;
   ClearColorArray();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetInterpolate(Bool_t b)
{
   // Set interpolation flag. This determines how colors from ROOT's
   // palette are mapped into RGBA values for given signal.

   fInterpolate = b;
   ClearColorArray();
}

//______________________________________________________________________________
void TEveRGBAPalette::SetFixColorRange(Bool_t v)
{
   // Set flag specifying how the palette is mapped to signal values:
   //  true  - LowLimit -> HighLimit
   //  false - MinValue -> MaxValue

   fFixColorRange = v;
   ClearColorArray();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetDefaultColor(Color_t ci)
{
   // Set default color.

   fDefaultColor = ci;
   TEveUtil::ColorFromIdx(ci, fDefaultRGBA, kTRUE);
}

//______________________________________________________________________________
void TEveRGBAPalette::SetDefaultColorPixel(Pixel_t pix)
{
   // Set default color.

   SetDefaultColor(Color_t(TColor::GetColor(pix)));
}

//______________________________________________________________________________
void TEveRGBAPalette::SetDefaultColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set default color.

   fDefaultColor = Color_t(TColor::GetColor(r, g, b));
   fDefaultRGBA[0] = r;
   fDefaultRGBA[1] = g;
   fDefaultRGBA[2] = b;
   fDefaultRGBA[3] = a;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetUnderColor(Color_t ci)
{
   // Set underflow color.

   fUnderColor = ci;
   TEveUtil::ColorFromIdx(ci, fUnderRGBA, kTRUE);
}

//______________________________________________________________________________
void TEveRGBAPalette::SetUnderColorPixel(Pixel_t pix)
{
   // Set underflow color.

   SetUnderColor(Color_t(TColor::GetColor(pix)));
}

//______________________________________________________________________________
void TEveRGBAPalette::SetUnderColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set underflow color.

   fUnderColor = Color_t(TColor::GetColor(r, g, b));
   fUnderRGBA[0] = r;
   fUnderRGBA[1] = g;
   fUnderRGBA[2] = b;
   fUnderRGBA[3] = a;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveRGBAPalette::SetOverColor(Color_t ci)
{
   // Set overflow color.

   fOverColor = ci;
   TEveUtil::ColorFromIdx(ci, fOverRGBA, kTRUE);
}

//______________________________________________________________________________
void TEveRGBAPalette::SetOverColorPixel(Pixel_t pix)
{
   // Set overflow color.

   SetOverColor(Color_t(TColor::GetColor(pix)));
}

//______________________________________________________________________________
void TEveRGBAPalette::SetOverColorRGBA(UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set overflow color.

   fOverColor = Color_t(TColor::GetColor(r, g, b));
   fOverRGBA[0] = r;
   fOverRGBA[1] = g;
   fOverRGBA[2] = b;
   fOverRGBA[3] = a;
}

//______________________________________________________________________________
void TEveRGBAPalette::MinMaxValChanged()
{
   // Emit the "MinMaxValChanged()" signal.
   // This is NOT called automatically from SetMin/Max functions but
   // it IS called from TEveRGBAPaletteEditor after it changes the
   // min/max values.

   Emit("MinMaxValChanged()");
}
