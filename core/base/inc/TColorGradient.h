// @(#)root/base:$Id$
//Author: Timur Pocheptsov   20/03/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TColorGradient
#define ROOT_TColorGradient


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TColorGradient                                                       //
//                                                                      //
// TColorGradient extends basic TColor.                                 //
// Actually, this is not a simple color, but linear or radial gradient  //
// for a filled area. By inheriting from TColor, gradients can be       //
// placed inside gROOT's list of colors and use it in all TAttXXX       //
// descendants without modifying any existing code.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include "Rtypes.h"

#include "TColor.h"


class TColorGradient : public TColor {
public:
   typedef std::vector<Color_t>::size_type SizeType_t;

   //TODO: Replace with enum class as soon as we have C++11 enabled by default.
   //CoordinateMode: both linear and radial gradients require some points - the
   //start and end points.
   //We can use either pad's rectangle as a coordinate system
   //or an object's bounding rect.
   enum ECoordinateMode {
      kPadMode,//NDC, in a pad's rectangle (pad is 0,0 - 1,1).
      kObjectBoundingMode //NDC in an object's bounding rect (this rect is 0,0 - 1, 1).
   };

   struct Point {
      Double_t fX;
      Double_t fY;

      Point()
         : fX(0.), fY(0.)
      {
      }

      Point(Double_t x, Double_t y)
         : fX(x), fY(y)
      {
      }
   };

private:
   //Positions of color nodes in a gradient, in NDC.
   std::vector<Double_t> fColorPositions;
   std::vector<Double_t> fColors;//RGBA values.

   //'default value' is kObjectBoundingMode.
   ECoordinateMode fCoordinateMode;

protected:
   TColorGradient();

   TColorGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                  const Color_t *colorIndices, ECoordinateMode mode = kObjectBoundingMode);
   TColorGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                  const Double_t *colors, ECoordinateMode mode = kObjectBoundingMode);

public:
   void ResetColor(UInt_t nPoints, const Double_t *points,
                   const Color_t *colorIndices);
   void ResetColor(UInt_t nPoints, const Double_t *points,
                   const Double_t *colorIndices);

   void SetCoordinateMode(ECoordinateMode mode);
   ECoordinateMode GetCoordinateMode() const;

   SizeType_t GetNumberOfSteps() const;
   const Double_t *GetColorPositions() const;
   const Double_t *GetColors() const;

private:
   void RegisterColor(Color_t colorIndex);

   ClassDefOverride(TColorGradient, 0); //Gradient fill.
};

class TLinearGradient : public TColorGradient {
public:
   //With C++11 we'll use inherited constructors!!!
   TLinearGradient();
   TLinearGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                   const Color_t *colorIndices, ECoordinateMode mode = kObjectBoundingMode);
   TLinearGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                   const Double_t *colors, ECoordinateMode mode = kObjectBoundingMode);

   //points are always in NDC (and also affected by fCoordinateMode).
   void SetStartEnd(const Point &p1, const Point &p2);
   const Point &GetStart() const;
   const Point &GetEnd() const;

private:
   Point fStart;
   Point fEnd;

   ClassDefOverride(TLinearGradient, 0); //Linear gradient fill.
};

//
//Radial gradient. Can be either "simple": you specify a center
//and radius in NDC coordinates (see comments about linear gradient
//and coordinate modes above), or "extended": you have two centers
//(start,end) and two radiuses (R1, R2) and interpolation between them;
//still start/end and radiuses are in NDC.
//

class TRadialGradient : public TColorGradient {
public:
   enum EGradientType {
      kSimple,
      kExtended
   };


   //With C++11 we'll use inherited constructors!!!
   TRadialGradient();
   TRadialGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                   const Color_t *colorIndices, ECoordinateMode mode = kObjectBoundingMode);
   TRadialGradient(Color_t newColor, UInt_t nPoints, const Double_t *points,
                   const Double_t *colors, ECoordinateMode mode = kObjectBoundingMode);


   EGradientType GetGradientType()const;


   //Extended gradient.
   void SetStartEndR1R2(const Point &p1, Double_t r1,
                        const Point &p2, Double_t r2);
   const Point &GetStart() const;
   Double_t GetR1() const;
   const Point &GetEnd() const;
   Double_t GetR2() const;

   //Simple radial gradient: the same as extended with
   //start == end, r1 = 0, r2 = radius.
   void SetRadialGradient(const Point &center, Double_t radius);
   const Point &GetCenter()const;
   Double_t GetRadius()const;

private:
   Point fStart;
   Double_t fR1 = 0.;
   Point fEnd;
   Double_t fR2 = 0.;

   EGradientType fType = kSimple;

   ClassDefOverride(TRadialGradient, 0); //Radial gradient fill.
};


#endif
