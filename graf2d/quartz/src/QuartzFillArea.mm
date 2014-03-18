// @(#)root/graf2d:$Id$
// Author: Olivier Couet, 23/01/2012

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <cassert>
#include <vector>

#include "QuartzFillArea.h"
#include "TColorGradient.h"
#include "QuartzLine.h"
#include "CocoaUtils.h"
#include "TVirtualX.h"
#include "RStipples.h"
#include "TError.h"
#include "TROOT.h"

//TODO: either use Color_t or use gVirtualX->GetLine/Fill/Color -
//not both, it's a complete mess now!

namespace ROOT {
namespace Quartz {

namespace Util = MacOSX::Util;

namespace {

const CGSize shadowOffset = CGSizeMake(10., 10.);
const CGFloat shadowBlur = 5.;

}//Unnamed namespace.

//______________________________________________________________________________
Bool_t SetFillColor(CGContextRef ctx, Color_t colorIndex)
{
   assert(ctx != 0 && "SetFillColor, ctx parameter is null");

   const TColor *color = gROOT->GetColor(colorIndex);
   
   //TGX11 selected color 0 (which is white).
   if (!color)
      color = gROOT->GetColor(kWhite);
   //???
   if (!color)
      return kFALSE;

   const CGFloat alpha = color->GetAlpha();

   Float_t rgb[3] = {};
   color->GetRGB(rgb[0], rgb[1], rgb[2]);
   CGContextSetRGBFillColor(ctx, rgb[0], rgb[1], rgb[2], alpha);
   
   return kTRUE;
}

//______________________________________________________________________________
void DrawPattern(void *data, CGContextRef ctx)
{
   assert(data != 0 && "DrawPattern, data parameter is null");
   assert(ctx != 0 && "DrawPattern, ctx parameter is null");

   //Draw a stencil pattern from gStipples
   const unsigned stencilIndex = *static_cast<unsigned *>(data);

   for (int i = 30, y = 0; i >= 0; i -= 2, ++y) {
      int x = 0;
      for (int j = 0; j < 8; ++j, ++x) {
         if (gStipples[stencilIndex][i] & (1 << j))
            CGContextFillRect(ctx, CGRectMake(x, y, 1, 1));
      }
      
      for (int j = 0; j < 8; ++j, ++x) {
         if (gStipples[stencilIndex][i + 1] & (1 << j))
            CGContextFillRect(ctx, CGRectMake(x, y, 1, 1));
      }
   }
}

//______________________________________________________________________________
bool SetFillPattern(CGContextRef ctx, const unsigned *patternIndex)
{
   assert(ctx != 0 && "SetFillPattern, ctx parameter is null");
   assert(patternIndex != 0 && "SetFillPattern, patternIndex parameter is null");

   const TColor *fillColor = gROOT->GetColor(gVirtualX->GetFillColor());
   if (!fillColor)
      fillColor = gROOT->GetColor(kWhite);
   
   if (!fillColor)
      return false;

   CGFloat rgba[] = {fillColor->GetRed(), fillColor->GetGreen(), fillColor->GetBlue(), fillColor->GetAlpha()};

   const Util::CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   if (!baseSpace.Get())
      return false;

   const Util::CFScopeGuard<CGColorSpaceRef> patternSpace(CGColorSpaceCreatePattern (baseSpace.Get()));
   if (!patternSpace.Get())
      return false;

   CGContextSetFillColorSpace(ctx, patternSpace.Get());

   CGPatternCallbacks callbacks = {0, &DrawPattern, 0};
   const Util::CFScopeGuard<CGPatternRef> pattern(CGPatternCreate((void*)patternIndex,
                                                  CGRectMake(0, 0, 16, 16),
                                                  CGAffineTransformIdentity, 16, 16,
                                                  kCGPatternTilingConstantSpacing,
                                                  false, &callbacks));

   if (!pattern.Get())
      return false;

   CGContextSetFillPattern(ctx, pattern.Get(), rgba);

   return true;
}

//______________________________________________________________________________
bool SetFillAreaParameters(CGContextRef ctx, unsigned *patternIndex)
{
   assert(ctx != 0 && "SetFillAreaParameters, ctx parameter is null");
   
   const unsigned fillStyle = gVirtualX->GetFillStyle() / 1000;
   
   //2 is hollow, 1 is solid and 3 is a hatch, !solid and !hatch - this is from O.C.'s code.
   if (fillStyle == 2 || (fillStyle != 1 && fillStyle != 3)) {
      if (!SetLineColor(ctx, gVirtualX->GetFillColor())) {
         ::Error("SetFillAreaParameters", "Line color for index %d was not found", int(gVirtualX->GetLineColor()));
         return false;
      }
   } else if (fillStyle == 1) {
      //Solid fill.
      if (!SetFillColor(ctx, gVirtualX->GetFillColor())) {
         ::Error("SetFillAreaParameters", "Fill color for index %d was not found", int(gVirtualX->GetFillColor()));
         return false;
      }
   } else {
      assert(patternIndex != 0 && "SetFillAreaParameters, pattern index in null");

      *patternIndex = gVirtualX->GetFillStyle() % 1000;
      //ROOT has 26 fixed patterns.
      if (*patternIndex > 25)
         *patternIndex = 2;

      if (!SetFillPattern(ctx, patternIndex)) {
         ::Error("SetFillAreaParameters", "SetFillPattern failed");
         return false;
      }
   }

   return true;
}
   
//______________________________________________________________________________
void DrawBox(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2, bool hollow)
{
   // Draw a box
            
   if (x1 > x2)
      std::swap(x1, x2);
   if (y1 > y2)
      std::swap(y1, y2);
   
   if (hollow)
      CGContextStrokeRect(ctx, CGRectMake(x1, y1, x2 - x1, y2 - y1));
   else
      CGContextFillRect(ctx, CGRectMake(x1, y1, x2 - x1, y2 - y1));
}

//______________________________________________________________________________
void DrawBoxGradient(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2,
                     const TColorGradient *extendedColor, const CGPoint &startPoint,
                     const CGPoint &endPoint, Bool_t drawShadow)
{
   assert(ctx != nullptr && "DrawBoxGradient, ctx parameter is null");
   assert(extendedColor != nullptr && "DrawBoxGradient, extendedColor parameter is null");
   assert(extendedColor->GetNumberOfSteps() != 0 && "DrawBoxGradient, no colors in extendedColor");
   
   if (drawShadow) {
      //To have shadow and gradient at the same time,
      //I first have to fill polygon, and after that
      //draw gradient (since gradient fills the whole area
      //with clip path and generates no shadow).
      CGContextSetRGBFillColor(ctx, 1., 1., 1., 0.25);
      CGContextSetShadow(ctx, shadowOffset, shadowBlur);
      CGContextFillRect(ctx, CGRectMake(x1, y1, x2 - x1, y2 - y1));
   }
   
   CGContextBeginPath(ctx);
   CGContextAddRect(ctx, CGRectMake(x1, y1, x2 - x1, y2 - y1));
   CGContextClosePath(ctx);
   CGContextClip(ctx);
   
   //Create a gradient.
   //TODO: must be a generic colorspace!!!
   const Util::CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   if (!baseSpace.Get()) {
      ::Error("DrawBoxGradient", "CGColorSpaceCreateDeviceRGB failed");
      return;
   }

   if (dynamic_cast<const TRadialGradient *>(extendedColor)) {
   
   } else if (dynamic_cast<const TLinearGradient *>(extendedColor)) {
      const Util::CFScopeGuard<CGGradientRef> gradient(CGGradientCreateWithColorComponents(baseSpace.Get(),
                                                       extendedColor->GetColors(),
                                                       extendedColor->GetColorPositions(),
                                                       extendedColor->GetNumberOfSteps()));
      if (!gradient.Get()) {
         ::Error("DrawBoxGradient", "CGGradientCreateWithColorComponents failed");
         return;
      }

      CGContextDrawLinearGradient(ctx, gradient.Get(), startPoint, endPoint,
                                  kCGGradientDrawsAfterEndLocation |
                                  kCGGradientDrawsBeforeStartLocation);
   } else {
      //TODO: to be implemented.
      assert(0 && "DrawBoxGradient, not implemented for this type of color gradient");
   }

}

//______________________________________________________________________________
void DrawFillArea(CGContextRef ctx, Int_t n, TPoint *xy, Bool_t shadow)
{
   // Draw a filled area through all points.
   // n         : number of points
   // xy        : list of points

   assert(ctx != 0 && "DrawFillArea, ctx parameter is null");
   assert(xy != 0 && "DrawFillArea, xy parameter is null");
                  
   CGContextBeginPath(ctx);
      
   CGContextMoveToPoint(ctx, xy[0].fX, xy[0].fY);
   for (Int_t i = 1; i < n; ++i) 
      CGContextAddLineToPoint(ctx, xy[i].fX, xy[i].fY);

   CGContextClosePath(ctx);
      
   const unsigned fillStyle = gVirtualX->GetFillStyle() / 1000;
   
   //2 is hollow, 1 is solid and 3 is a hatch, !solid and !hatch - this is from O.C.'s code.
   if (fillStyle == 2 || (fillStyle != 1 && fillStyle != 3)) {
      CGContextStrokePath(ctx);
   } else if (fillStyle == 1) {
      if (shadow)
         CGContextSetShadow(ctx, shadowOffset, shadowBlur);
      
      CGContextFillPath(ctx);
   } else {
      if (shadow)
         CGContextSetShadow(ctx, shadowOffset, shadowBlur);

      CGContextFillPath(ctx);
   }
}

//______________________________________________________________________________
void DrawFillAreaGradient(CGContextRef ctx, Int_t nPoints, const TPoint *xy,
                          const TColorGradient *extendedColor, const CGPoint &startPoint,
                          const CGPoint &endPoint, Bool_t drawShadow)
{
   using ROOT::MacOSX::Util::CFScopeGuard;

   assert(ctx != nullptr && "DrawFillAreaGradient, ctx parameter is null");
   assert(nPoints != 0 && "DrawFillAreaGradient, nPoints parameter is 0");
   assert(xy != nullptr && "DrawFillAreaGradient, xy parameter is null");
   assert(extendedColor != nullptr && "DrawFillAreaGradient, extendedColor parameter is null");

   const CFScopeGuard<CGMutablePathRef> path(CGPathCreateMutable());
   if (!path.Get()) {
      ::Error("DrawFillAreaGradient", "CGPathCreateMutable failed");
      return;
   }

   CGPathMoveToPoint(path.Get(), nullptr, xy[0].fX, xy[0].fY);
   for (Int_t i = 1; i < nPoints; ++i)
      CGPathAddLineToPoint(path.Get(), nullptr, xy[i].fX, xy[i].fY);
   
   CGPathCloseSubpath(path.Get());
   
   if (drawShadow) {
      //To have shadow and gradient at the same time,
      //I first have to fill polygon, and after that
      //draw gradient (since gradient fills the whole area
      //with clip path and generates no shadow).
      CGContextSetRGBFillColor(ctx, 1., 1., 1., 0.5);
      CGContextBeginPath(ctx);
      CGContextAddPath(ctx, path.Get());
      CGContextSetShadow(ctx, shadowOffset, shadowBlur);
      CGContextFillPath(ctx);
   }

   CGContextBeginPath(ctx);
   CGContextAddPath(ctx, path.Get());
   CGContextClip(ctx);

   //Create a gradient.
   //TODO: must be a generic RGB color space???
   const CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   if (!baseSpace.Get()) {
      ::Error("DrawFillAreaGradient", "CGColorSpaceCreateDeviceRGB failed");
      return;
   }
   
   if (dynamic_cast<const TLinearGradient *>(extendedColor)) {
      const CFScopeGuard<CGGradientRef> gradient(CGGradientCreateWithColorComponents(baseSpace.Get(),
                                                 extendedColor->GetColors(),
                                                 extendedColor->GetColorPositions(),
                                                 extendedColor->GetNumberOfSteps()));
      if (!gradient.Get()) {
         ::Error("DrawFillAreaGradient", "CGGradientCreateWithColorComponents failed");
         return;
      }
      
      CGContextDrawLinearGradient(ctx, gradient.Get(), startPoint, endPoint,
                                  kCGGradientDrawsAfterEndLocation |
                                  kCGGradientDrawsBeforeStartLocation);
   } else {
      //TODO: to be implemented.
      assert(0 && "DrawFillAreadGradient, not implemented for this type of a gradient fill");
   }
}

#pragma mark - Some aux. functions.

//______________________________________________________________________________
void FindBoundingBox(Int_t nPoints, const TPoint *xy, CGPoint &topLeft, CGPoint &bottomRight)
{
   assert(nPoints > 2 && "FindBoundingBox, invalid number of points in a polygon");
   assert(xy != 0 && "FindBoundingBox, parameter 'xy' is null");
   
   topLeft.x = xy[0].fX;
   topLeft.y = xy[0].fY;
   
   bottomRight = topLeft;
   
   for (Int_t i = 1; i < nPoints; ++i) {
      topLeft.x = std::min(topLeft.x, CGFloat(xy[i].fX));
      topLeft.y = std::min(topLeft.y, CGFloat(xy[i].fY));
      //
      bottomRight.x = std::max(bottomRight.x, CGFloat(xy[i].fX));
      bottomRight.y = std::max(bottomRight.y, CGFloat(xy[i].fY));
   }
}

//______________________________________________________________________________
void CalculateGradientPoints(const TColorGradient *extendedColor, const CGSize &sizeOfDrawable,
                             Int_t n, const TPoint *polygon, CGPoint &start, CGPoint &end)
{
   assert(extendedColor != 0 && "CalculateGradientPoints, parameter 'extendedColor' is null");
   assert(n > 2 && "CalculateGradientPoints, parameter 'n' is not a valid number of points");
   assert(polygon != 0 && "CalculateGradientPoints, parameter 'polygon' is null");
   
   //TODO: check-check-check-chek - it's just a test and can be wrong!!!
   
   const TColorGradient::ECoordinateMode mode = extendedColor->GetCoordinateMode();

   //TODO: that's stupid, but ... radial can iherit linear to make things easier :)
   if (const TLinearGradient * const lGrad = dynamic_cast<const TLinearGradient *>(extendedColor)) {
      start = CGPointMake(lGrad->GetStartPoint().fX, lGrad->GetStartPoint().fY);
      end.x = lGrad->GetEndPoint().fX;
      end.y = lGrad->GetEndPoint().fY;
   } else if (const TRadialGradient * const rGrad = dynamic_cast<const TRadialGradient *>(extendedColor)) {
      start.x = rGrad->GetStartPoint().fX;
      start.y = rGrad->GetStartPoint().fY;
      end.x = rGrad->GetEndPoint().fX;
      end.y = rGrad->GetEndPoint().fY;
   }
   
   if (mode == TColorGradient::kObjectBoundingMode) {
      //With Quartz we always work with something similar to 'kPadMode',
      //so convert start and end into this space.
      CGPoint topLeft = {}, bottomRight = {};
      Quartz::FindBoundingBox(n, polygon, topLeft, bottomRight);
      
      const CGFloat w = bottomRight.x - topLeft.x;
      const CGFloat h = bottomRight.y - topLeft.y;
      
      start.x = (w * start.x + topLeft.x);
      end.x *= (w * end.x + topLeft.x);
      
      start.y = (h * start.y + topLeft.y);
      end.y *= (h * end.y + topLeft.y);
   } else {
      start.x *= sizeOfDrawable.width;
      start.y *= sizeOfDrawable.height;
      end.x *= sizeOfDrawable.width;
      end.y *= sizeOfDrawable.height;
   }
}

//______________________________________________________________________________
void CalculateGradientPoints(const TColorGradient *extendedColor, const CGSize &sizeOfDrawable,
                             Int_t x1, Int_t y1, Int_t x2, Int_t y2, CGPoint &start, CGPoint &end)
{
   assert(extendedColor != 0 && "CalculateGradientPoints, parameter 'extendedColor' is null");

   TPoint polygon[4];
   polygon[0].fX = x1;
   polygon[0].fY = y1;
   polygon[1].fX = x2;
   polygon[1].fY = y1;
   polygon[2].fX = x2;
   polygon[2].fY = y2;
   polygon[3].fX = x1;
   polygon[3].fY = y2;
   
   CalculateGradientPoints(extendedColor, sizeOfDrawable, 4, polygon, start, end);
}


}//namespace Quartz
}//namespace ROOT
