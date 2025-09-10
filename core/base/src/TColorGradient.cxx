// @(#)root/base:$Id$
// Author: Timur Pocheptsov   20/3/2012

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TColorGradient
\ingroup Base
\ingroup GraphicsAtt

TColorGradient extends basic TColor.
Actually, this is not a simple color, but linear gradient + shadow
for filled area. By inheriting from TColor, gradients can be placed
inside gROOT's list of colors and use it in all TAttXXX descendants
without modifying any existing code.

Shadow, of course, is not a property of any color, and gradient is
not, but this is the best way to add new attributes to filled area
without re-writing all the graphics code.
*/

#include <cassert>

#include "TColorGradient.h"
#include "TObjArray.h"
#include "TString.h"
#include "TError.h"
#include "TROOT.h"


////////////////////////////////////////////////////////////////////////////////
/// There is no way to validate parameters here, so it's up to user
/// to pass correct arguments.

TColorGradient::TColorGradient(Color_t colorIndex, UInt_t nPoints, const Double_t *points,
                               const Color_t *indices, ECoordinateMode mode)
                   : fCoordinateMode(mode)
{
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != nullptr && "TColorGradient, points parameter is null");
   assert(indices != nullptr && "TColorGradient, indices parameter is null");

   ResetColor(nPoints, points, indices);
   RegisterColor(colorIndex);
}

////////////////////////////////////////////////////////////////////////////////
/// There is no way to validate parameters here, so it's up to user
/// to pass correct arguments.

TColorGradient::TColorGradient(Color_t colorIndex, UInt_t nPoints, const Double_t *points,
                               const Double_t *colors, ECoordinateMode mode)
                  : fCoordinateMode(mode)
{
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != nullptr && "TColorGradient, points parameter is null");
   assert(colors != nullptr && "TColorGradient, colors parameter is null");

   ResetColor(nPoints, points, colors);
   RegisterColor(colorIndex);
}

////////////////////////////////////////////////////////////////////////////////
/// Reset color.

void TColorGradient::ResetColor(UInt_t nPoints, const Double_t *points, const Color_t *colorIndices)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != nullptr && "ResetColor, points parameter is null");
   assert(colorIndices != nullptr && "ResetColor, colorIndices parameter is null");

   std::vector<Double_t> colors(nPoints * 4);

   for (UInt_t i = 0, pos = 0; i < nPoints; ++i, pos += 4) {
      const TColor *clearColor = gROOT->GetColor(colorIndices[i]);
      if (!clearColor) {
         Error("ResetColor", "Bad color for index %d, set to opaque black", colorIndices[i]);
         colors[pos] = 0.;
         colors[pos + 1] = 0.;
         colors[pos + 2] = 0.;
         colors[pos + 3] = 1.; //Alpha.
      } else {
         if (dynamic_cast<const TColorGradient *>(clearColor))
            Warning("ResetColor", "Gradient color index %d used as base for other gradient color", colorIndices[i]);
         colors[pos] = clearColor->GetRed();
         colors[pos + 1] = clearColor->GetGreen();
         colors[pos + 2] = clearColor->GetBlue();
         colors[pos + 3] = clearColor->GetAlpha();
      }
   }

   ResetColor(nPoints, points, colors.data());
}

////////////////////////////////////////////////////////////////////////////////
/// Reset color.

void TColorGradient::ResetColor(UInt_t nPoints, const Double_t *points,
                                const Double_t *colors)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != nullptr && "ResetColor, points parameter is null");
   assert(colors != nullptr && "ResetColor, colors parameter is null");

   fColorPositions.assign(points, points + nPoints);
   fColors.assign(colors, colors + nPoints * 4);

   Double_t sum[4] = { 0., 0., 0., 0. };
   for (unsigned n = 0; n < fColors.size(); ++n)
      sum[n % 4] += fColors[n];

   SetRGB(sum[0] / nPoints, sum[1] / nPoints, sum[2] / nPoints);
   SetAlpha(sum[3] / nPoints);
}

////////////////////////////////////////////////////////////////////////////////
/// Change alpha parameter of the color

void TColorGradient::SetColorAlpha(UInt_t indx, Double_t alpha)
{
   if (indx*4 + 3 < fColors.size())
      fColors[indx*4 + 3] = alpha;
}

////////////////////////////////////////////////////////////////////////////////
/// Return alpha parameter of selected color

Double_t TColorGradient::GetColorAlpha(UInt_t indx) const
{
   if (indx*4 + 3 < fColors.size())
      return fColors[indx*4 + 3];

   return 1.;
}

////////////////////////////////////////////////////////////////////////////////
/// Set coordinate mode.

void TColorGradient::SetCoordinateMode(ECoordinateMode mode)
{
   fCoordinateMode = mode;
}

////////////////////////////////////////////////////////////////////////////////
/// Get coordinate mode.

TColorGradient::ECoordinateMode TColorGradient::GetCoordinateMode()const
{
   return fCoordinateMode;
}

////////////////////////////////////////////////////////////////////////////////
/// Get number of steps.

TColorGradient::SizeType_t TColorGradient::GetNumberOfSteps()const
{
   return fColorPositions.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Get color positions

const Double_t *TColorGradient::GetColorPositions()const
{
   return fColorPositions.data();
}

////////////////////////////////////////////////////////////////////////////////
/// Get colors.

const Double_t *TColorGradient::GetColors()const
{
   return fColors.data();
}

////////////////////////////////////////////////////////////////////////////////
/// Register color

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

////////////////////////////////////////////////////////////////////////////////
/// Set end and start.

void TLinearGradient::SetStartEnd(const Point &p1, const Point &p2)
{
   fStart = p1;
   fEnd = p2;
}

////////////////////////////////////////////////////////////////////////////////
/// Get start.

const TColorGradient::Point &TLinearGradient::GetStart()const
{
   return fStart;
}

////////////////////////////////////////////////////////////////////////////////
/// Get end.

const TColorGradient::Point &TLinearGradient::GetEnd()const
{
   return fEnd;
}


////////////////////////////////////////////////////////////////////////////////
/// Get gradient type.

TRadialGradient::EGradientType TRadialGradient::GetGradientType()const
{
   return fType;
}

////////////////////////////////////////////////////////////////////////////////
/// Set start and end R1 and R2.

void TRadialGradient::SetStartEndR1R2(const Point &p1, Double_t r1, const Point &p2, Double_t r2)
{
   fStart = p1;
   fR1 = r1;
   fEnd = p2;
   fR2 = r2;

   fType = kExtended;
}

////////////////////////////////////////////////////////////////////////////////
/// Get start.

const TColorGradient::Point &TRadialGradient::GetStart()const
{
   return fStart;
}

////////////////////////////////////////////////////////////////////////////////
// Get R1.

Double_t TRadialGradient::GetR1()const
{
   return fR1;
}

////////////////////////////////////////////////////////////////////////////////
/// Get end.

const TColorGradient::Point &TRadialGradient::GetEnd()const
{
   return fEnd;
}

////////////////////////////////////////////////////////////////////////////////
/// Get R2.

Double_t TRadialGradient::GetR2()const
{
   return fR2;
}

////////////////////////////////////////////////////////////////////////////////
/// Set radial gradient.

void TRadialGradient::SetRadialGradient(const Point &center, Double_t radius)
{
   fStart = center;
   fR1 = radius;

   fType = kSimple;
}

////////////////////////////////////////////////////////////////////////////////
/// Get center.

const TColorGradient::Point &TRadialGradient::GetCenter()const
{
   return fStart;
}

////////////////////////////////////////////////////////////////////////////////
/// Get radius.

Double_t TRadialGradient::GetRadius()const
{
   return fR1;
}
