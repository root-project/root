// @(#)root/base:$Id$
// Author: Timur Pocheptsov   20/3/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TColorGradient                                                       //
//                                                                      //
// TColorGradient extends basic TColor.                                 //
// Actually, this is not a simple color, but linear gradient + shadow   //
// for filled area. By inheriting from TColor, gradients can be placed  //
// inside gROOT's list of colors and use it in all TAttXXX descendants  //
// without modifying any existing code.                                 //
// Shadow, of course, is not a property of any color, and gradient is   //
// not, but this is the best way to add new attributes to filled area   //
// without re-writing all the graphics code.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <cassert>

#include "TColorGradient.h"
#include "TObjArray.h"
#include "TString.h"
#include "TError.h"
#include "TROOT.h"

ClassImp(TColorGradient)

//______________________________________________________________________________
TColorGradient::TColorGradient()
                   : fCoordinateMode(kObjectBoundingMode)
{
}

//______________________________________________________________________________
TColorGradient::TColorGradient(Color_t colorIndex, UInt_t nPoints, const Double_t *points,
                               const Color_t *indices, ECoordinateMode mode)
                   : fCoordinateMode(mode)
{
   //I have no way to validate parameters here, so it's up to user
   //to pass correct arguments.
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != 0 && "TColorGradient, points parameter is null");
   assert(indices != 0 && "TColorGradient, indices parameter is null");

   ResetColor(nPoints, points, indices);
   RegisterColor(colorIndex);
}

//______________________________________________________________________________
TColorGradient::TColorGradient(Color_t colorIndex, UInt_t nPoints, const Double_t *points,
                               const Double_t *colors, ECoordinateMode mode)
                  : fCoordinateMode(mode)
{
   //I have no way to validate parameters here, so it's up to user
   //to pass correct arguments.
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != 0 && "TColorGradient, points parameter is null");
   assert(colors != 0 && "TColorGradient, colors parameter is null");

   ResetColor(nPoints, points, colors);
   RegisterColor(colorIndex);
}

//______________________________________________________________________________
void TColorGradient::ResetColor(UInt_t nPoints, const Double_t *points, const Color_t *colorIndices)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != 0 && "ResetColor, points parameter is null");
   assert(colorIndices != 0 && "ResetColor, colorIndices parameter is null");

   fColorPositions.assign(points, points + nPoints);
   fColors.resize(nPoints * 4);//4 == rgba.

   Float_t rgba[4];
   for (UInt_t i = 0, pos = 0; i < nPoints; ++i, pos += 4) {
      const TColor *clearColor = gROOT->GetColor(colorIndices[i]);
      if (!clearColor || dynamic_cast<const TColorGradient *>(clearColor)) {
         //TColorGradient can not be a step in TColorGradient.
         Error("ResetColor", "Bad color for index %d, set to opaque black", colorIndices[i]);
         fColors[pos] = 0.;
         fColors[pos + 1] = 0.;
         fColors[pos + 2] = 0.;
         fColors[pos + 3] = 1.;//Alpha.
      } else {
         clearColor->GetRGB(rgba[0], rgba[1], rgba[2]);
         rgba[3] = clearColor->GetAlpha();
         fColors[pos] = rgba[0];
         fColors[pos + 1] = rgba[1];
         fColors[pos + 2] = rgba[2];
         fColors[pos + 3] = rgba[3];
      }
   }
}

//______________________________________________________________________________
void TColorGradient::ResetColor(UInt_t nPoints, const Double_t *points,
                                const Double_t *colors)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != 0 && "ResetColor, points parameter is null");
   assert(colors != 0 && "ResetColor, colors parameter is null");

   fColorPositions.assign(points, points + nPoints);
   fColors.assign(colors, colors + nPoints * 4);
}

//______________________________________________________________________________
void TColorGradient::SetCoordinateMode(ECoordinateMode mode)
{
   fCoordinateMode = mode;
}

//______________________________________________________________________________
TColorGradient::ECoordinateMode TColorGradient::GetCoordinateMode()const
{
   return fCoordinateMode;
}

//______________________________________________________________________________
TColorGradient::SizeType_t TColorGradient::GetNumberOfSteps()const
{
   //
   return fColorPositions.size();
}

//______________________________________________________________________________
const Double_t *TColorGradient::GetColorPositions()const
{
   //
   return &fColorPositions[0];
}

//______________________________________________________________________________
const Double_t *TColorGradient::GetColors()const
{
   //
   return &fColors[0];
}

//______________________________________________________________________________
void TColorGradient::RegisterColor(Color_t colorIndex)
{
   fNumber = colorIndex;
   SetName(TString::Format("Color%d", colorIndex));

   if (gROOT) {
      if (gROOT->GetColor(colorIndex)) {
         Warning("RegisterColor", "Color with index %d is already defined", colorIndex);
         return;
      }

      if (TObjArray *colors = (TObjArray*)gROOT->GetListOfColors()) {
         colors->AddAtAndExpand(this, colorIndex);
      } else {
         Error("RegisterColor", "List of colors is a null pointer in gROOT, color was not registered");
         return;
      }
   }
}

ClassImp(TLinearGradient)

//______________________________________________________________________________
TLinearGradient::TLinearGradient()
{
}

//______________________________________________________________________________
TLinearGradient::TLinearGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                                 const Color_t *colorIndices, ECoordinateMode mode)
                   : TColorGradient(newColor, nPoints, points, colorIndices, mode)
{
}

//______________________________________________________________________________
TLinearGradient::TLinearGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                                 const Double_t *colors, ECoordinateMode mode)
                   : TColorGradient(newColor, nPoints, points, colors, mode)
{
}

//______________________________________________________________________________
void TLinearGradient::SetStartEnd(const Point &p1, const Point &p2)
{
   fStart = p1;
   fEnd = p2;
}

//______________________________________________________________________________
const TColorGradient::Point &TLinearGradient::GetStart()const
{
   return fStart;
}

//______________________________________________________________________________
const TColorGradient::Point &TLinearGradient::GetEnd()const
{
   return fEnd;
}

ClassImp(TRadialGradient)

//______________________________________________________________________________
TRadialGradient::TRadialGradient()
{
}

//______________________________________________________________________________
TRadialGradient::TRadialGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                                 const Color_t *colorIndices, ECoordinateMode mode)
                   : TColorGradient(newColor, nPoints, points, colorIndices, mode)
{
}

//______________________________________________________________________________
TRadialGradient::TRadialGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                                 const Double_t *colors, ECoordinateMode mode)
                   : TColorGradient(newColor, nPoints, points, colors, mode)
{
}

//______________________________________________________________________________
TRadialGradient::EGradientType TRadialGradient::GetGradientType()const
{
   return fType;
}

//______________________________________________________________________________
void TRadialGradient::SetStartEndR1R2(const Point &p1, Double_t r1, const Point &p2, Double_t r2)
{
   fStart = p1;
   fR1 = r1;
   fEnd = p2;
   fR2 = r2;

   fType = kExtended;
}

//______________________________________________________________________________
const TColorGradient::Point &TRadialGradient::GetStart()const
{
   return fStart;
}

//______________________________________________________________________________
Double_t TRadialGradient::GetR1()const
{
   return fR1;
}

//______________________________________________________________________________
const TColorGradient::Point &TRadialGradient::GetEnd()const
{
   return fEnd;
}

//______________________________________________________________________________
Double_t TRadialGradient::GetR2()const
{
   return fR2;
}

//______________________________________________________________________________
void TRadialGradient::SetRadialGradient(const Point &center, Double_t radius)
{
   fStart = center;
   fR1 = radius;

   fType = kSimple;
}

//______________________________________________________________________________
const TColorGradient::Point &TRadialGradient::GetCenter()const
{
   return fStart;
}

//______________________________________________________________________________
Double_t TRadialGradient::GetRadius()const
{
   return fR1;
}
