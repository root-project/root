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

/*
//______________________________________________________________________________
TColorGradient::TColorGradient()
                  : fGradientDirection(kGDVertical)
{
   //Default ctors, which does nothing and should not be used (it's for io only?).
}
*/

//______________________________________________________________________________
TColorGradient::TColorGradient(Color_t colorIndex, EGradientDirection dir, UInt_t nPoints, const Double_t *points, const Color_t *indices)
                   : fGradientDirection(dir)
{
   //I have no way to validate parameters here, so it's up to user
   //to pass correct arguments.
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != 0 && "TColorGradient, points parameter is null");
   assert(indices != 0 && "TColorGradient, indices parameter is null");

   ResetColor(dir, nPoints, points, indices);
   RegisterColor(colorIndex);
}

//______________________________________________________________________________
TColorGradient::TColorGradient(Color_t colorIndex, EGradientDirection dir, UInt_t nPoints, const Double_t *points, const Double_t *colors)
                   : fGradientDirection(dir)
{
   //I have no way to validate parameters here, so it's up to user
   //to pass correct arguments.
   assert(nPoints != 0 && "TColorGradient, number of points is 0");
   assert(points != 0 && "TColorGradient, points parameter is null");
   assert(colors != 0 && "TColorGradient, colors parameter is null");

   ResetColor(dir, nPoints, points, colors);
   RegisterColor(colorIndex);
}

//______________________________________________________________________________
void TColorGradient::ResetColor(EGradientDirection dir, UInt_t nPoints, const Double_t *points, const Color_t *colorIndices)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != 0 && "ResetColor, points parameter is null");
   assert(colorIndices != 0 && "ResetColor, colorIndices parameter is null");

   fGradientDirection = dir;
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
void TColorGradient::ResetColor(EGradientDirection dir, UInt_t nPoints, const Double_t *points, const Double_t *colors)
{
   assert(nPoints != 0 && "ResetColor, number of points is 0");
   assert(points != 0 && "ResetColor, points parameter is null");
   assert(colors != 0 && "ResetColor, colors parameter is null");

   fGradientDirection = dir;
   fColorPositions.assign(points, points + nPoints);
   fColors.assign(colors, colors + nPoints * 4);
}

//______________________________________________________________________________
TColorGradient::EGradientDirection TColorGradient::GetGradientDirection()const
{
   //
   return fGradientDirection;
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
