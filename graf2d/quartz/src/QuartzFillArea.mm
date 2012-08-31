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


namespace ROOT {
namespace Quartz {

namespace Util = MacOSX::Util;

namespace {

const CGSize shadowOffset = CGSizeMake(10., 10.);
const CGFloat shadowBlur = 5.;

//______________________________________________________________________________
void InvertGradientPositions(std::vector<CGFloat> &positions)
{
   typedef std::vector<CGFloat>::size_type size_type;
   
   for (size_type i = 0; i < positions.size(); ++i)
      positions[i] = 1. - positions[i];
}

}//Unnamed namespace.

//______________________________________________________________________________
Bool_t SetFillColor(CGContextRef ctx, Color_t colorIndex)
{
   assert(ctx != 0 && "SetFillColor, ctx parameter is null");

   const TColor * const color = gROOT->GetColor(colorIndex);
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
      if (*patternIndex >= 26) {
         ::Error("SetFillAreaParameters", "Pattern index must be < 26");
         return false;
      }

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
void DrawBoxGradient(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2, const TColorGradient *extendedColor, Bool_t drawShadow)
{
   assert(ctx != nullptr && "DrawBoxGradient, ctx parameter is null");
   assert(extendedColor != nullptr && "DrawBoxGradient, extendedColor parameter is null");
   assert(extendedColor->GetNumberOfSteps() != 0 && "DrawBoxGradient, no colors in extendedColor");
   
   CGPoint startPoint = CGPointZero;
   CGPoint endPoint = CGPointZero;
   
   if (extendedColor->GetGradientDirection() == TColorGradient::kGDHorizontal) {
      startPoint.x = x1;
      endPoint.x = x2;
   } else {
      startPoint.y = y1;
      endPoint.y = y2;
   }
      
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
   const Util::CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   
   std::vector<CGFloat> positions(extendedColor->GetColorPositions(), extendedColor->GetColorPositions() + extendedColor->GetNumberOfSteps());
   InvertGradientPositions(positions);

   const Util::CFScopeGuard<CGGradientRef> gradient(CGGradientCreateWithColorComponents(baseSpace.Get(), extendedColor->GetColors(), &positions[0], extendedColor->GetNumberOfSteps()));
   CGContextDrawLinearGradient(ctx, gradient.Get(), startPoint, endPoint, 0);
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
void DrawFillAreaGradient(CGContextRef ctx, Int_t nPoints, const TPoint *xy, const TColorGradient *extendedColor, Bool_t drawShadow)
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

   //Calculate gradient's start and end point (either X or Y coordinate,
   //depending on gradient type). Also, fill CGPath object.
   CGPoint startPoint = CGPointZero;
   CGPoint endPoint = CGPointZero;
   
   if (extendedColor->GetGradientDirection() == TColorGradient::kGDHorizontal) {
      startPoint = CGPointMake(xy[0].fX, 0.);
      endPoint = CGPointMake(xy[0].fX, 0.);
   
      for (Int_t i = 1; i < nPoints; ++i) {
         startPoint.x = std::min(startPoint.x, CGFloat(xy[i].fX));
         endPoint.x = std::max(endPoint.x, CGFloat(xy[i].fX));
      }
   } else {
      startPoint = CGPointMake(0., xy[0].fY);
      endPoint = CGPointMake(0., xy[0].fY);
   
      for (Int_t i = 1; i < nPoints; ++i) {
         startPoint.y = std::min(startPoint.y, CGFloat(xy[i].fY));
         endPoint.y = std::max(endPoint.y, CGFloat(xy[i].fY));
      }
   }
   
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
   const CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   std::vector<CGFloat> positions(extendedColor->GetColorPositions(), extendedColor->GetColorPositions() + extendedColor->GetNumberOfSteps());
   InvertGradientPositions(positions);

   const CFScopeGuard<CGGradientRef> gradient(CGGradientCreateWithColorComponents(baseSpace.Get(), extendedColor->GetColors(), &positions[0],
                                                                                  extendedColor->GetNumberOfSteps()));
   CGContextDrawLinearGradient(ctx, gradient.Get(), startPoint, endPoint, 0);
}

}//namespace Quartz
}//namespace ROOT
