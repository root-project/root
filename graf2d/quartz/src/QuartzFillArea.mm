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

//ROOT has TColorGradient, TLinearGradient and TRadialGradient -
//they all specify parameters in NDC. But for rendering with
//Quartz I need more specific parameters, I calculate them here.

// GradientFactory deals with CGFloat being either float or double.
template<class SRC, class DST>
struct GradientFactory {
   static CGGradientRef CreateGradient(CGColorSpaceRef colorSpace,
                                       const TColorGradient *extendedColor)
   {
      assert(colorSpace != nullptr &&
             "GradientFactory::CreateGradient, parameter 'colorSpace' is null");
      assert(extendedColor != nullptr &&
             "GradientFactory::CreateGradient, parameter 'extendedColor' is null");
      const SRC *compStart = extendedColor->GetColors();
      const SRC *compEnd = compStart + extendedColor->GetNumberOfSteps() * 4;
      const std::vector<DST> convertedComponents(compStart, compEnd);
      const SRC *posStart = extendedColor->GetColorPositions();
      const SRC *posEnd = posStart + extendedColor->GetNumberOfSteps();
      const std::vector<DST> convertedPositions(posStart, posEnd);

      return CGGradientCreateWithColorComponents(colorSpace,
                                                 &convertedComponents[0],
                                                 &convertedPositions[0],
                                                 extendedColor->GetNumberOfSteps());
   }
};

template<class DST>
struct GradientFactory<DST, DST> {
   static CGGradientRef CreateGradient(CGColorSpaceRef colorSpace,
                                       const TColorGradient *extendedColor)
   {
      assert(colorSpace != nullptr &&
             "GradientFactory::CreateGradient, parameter 'colorSpace' is null");
      assert(extendedColor != nullptr &&
             "GradientFactory::CreateGradient, parameter 'extendedColor' is null");
      const DST *comps = extendedColor->GetColors();
      const DST *pos = extendedColor->GetColorPositions();
      return CGGradientCreateWithColorComponents(colorSpace, comps, pos,
                                                 extendedColor->GetNumberOfSteps());
   }
};

struct GradientParameters {
   //
   CGPoint fStartPoint;
   CGPoint fEndPoint;

   //Only for radial gradient fill:
   CGFloat fStartRadius;
   CGFloat fEndRadius;

   //For the 'simple' radial gradient we use
   //only fStartPoint (it's a center actually)
   //and fStartRadius.

   //Something else:...

   GradientParameters()
      : fStartPoint(), fEndPoint(), fStartRadius(0.), fEndRadius(0.)
   {
   }

   GradientParameters(const CGPoint &start, const CGPoint &end)
      : fStartPoint(start), fEndPoint(end), fStartRadius(0.), fEndRadius(0.)
   {
   }


   GradientParameters(const CGPoint &start, const CGPoint &end,
                      CGFloat startRadius, CGFloat endRadius)
      : fStartPoint(start), fEndPoint(end), fStartRadius(startRadius), fEndRadius(endRadius)
   {
   }

};

//______________________________________________________________________________
CGRect FindBoundingBox(Int_t nPoints, const TPoint *xy)
{
   //When calculating gradient parameters for TColorGradient::kObjectBoundingMode
   //we need a bounding rect for a polygon.
   assert(nPoints > 2 && "FindBoundingBox, invalid number of points in a polygon");
   assert(xy != nullptr && "FindBoundingBox, parameter 'xy' is null");

   CGPoint bottomLeft = {};
   bottomLeft.x = xy[0].fX;
   bottomLeft.y = xy[0].fY;

   CGPoint topRight = bottomLeft;

   for (Int_t i = 1; i < nPoints; ++i) {
      bottomLeft.x = std::min(bottomLeft.x, CGFloat(xy[i].fX));
      bottomLeft.y = std::min(bottomLeft.y, CGFloat(xy[i].fY));
      //
      topRight.x = std::max(topRight.x, CGFloat(xy[i].fX));
      topRight.y = std::max(topRight.y, CGFloat(xy[i].fY));
   }

   return CGRectMake(bottomLeft.x, bottomLeft.y,
                     topRight.x - bottomLeft.x,
                     topRight.y - bottomLeft.y);
}

//______________________________________________________________________________
template<class GradientType>
bool CalculateGradientStartEnd(const GradientType *grad,
                               const CGSize &sizeOfDrawable,
                               Int_t n, const TPoint *polygon,
                               GradientParameters &params)
{
   assert(grad != nullptr &&
          "CalculateGradientStartEnd, parameter 'grad' is null");
   assert(sizeOfDrawable.width > 0. && sizeOfDrawable.height > 0. &&
          "CalculateGradientStartEnd, invalid destination drawable size");
   assert(n > 2 &&
          "CalculateGradientStartEnd, parameter 'n' is not a valid number of points");
   assert(polygon != nullptr &&
          "CalculateGradientStartEnd, parameter 'polygon' is null");

   const TColorGradient::ECoordinateMode mode = grad->GetCoordinateMode();

   CGPoint start = CGPointMake(grad->GetStart().fX, grad->GetStart().fY);
   CGPoint end = CGPointMake(grad->GetEnd().fX, grad->GetEnd().fY);

   const CGRect &bbox = FindBoundingBox(n, polygon);

   if (!bbox.size.width || !bbox.size.height)
      return false;//Invalid polygon actually.

   if (mode == TColorGradient::kObjectBoundingMode) {
      //With Quartz we always work with something similar to 'kPadMode',
      //so convert start and end into this space.
      start.x = bbox.size.width * start.x + bbox.origin.x;
      end.x = bbox.size.width * end.x + bbox.origin.x;

      start.y = bbox.size.height * start.y + bbox.origin.y;
      end.y = bbox.size.height * end.y + bbox.origin.y;
   } else {
      start.x *= sizeOfDrawable.width;
      start.y *= sizeOfDrawable.height;
      end.x *= sizeOfDrawable.width;
      end.y *= sizeOfDrawable.height;
   }

   params.fStartPoint = start;
   params.fEndPoint = end;

   return true;
}

//______________________________________________________________________________
bool CalculateGradientRadiuses(const TRadialGradient *grad,
                               const CGSize &sizeOfDrawable,
                               Int_t n, const TPoint *polygon,
                               GradientParameters &params)
{
   assert(grad != nullptr && "CalculateGradientRadiuses, parameter 'grad' is null");
   assert(sizeOfDrawable.width > 0. && sizeOfDrawable.height > 0. &&
          "CalculateGradientRadiuses, invalid destination drawable size");
   assert(grad->GetGradientType() == TRadialGradient::kExtended &&
          "CalculateGradientRadiuses, extended radial gradient expected");
   assert(n > 2 && "CalculateGradientRadiuses, parameter 'n' is not a valid number of points");
   assert(polygon != nullptr &&
          "CalculateGradientRadiuses, parameter 'polygon' is null");


   const CGRect &bbox = FindBoundingBox(n, polygon);
   if (!bbox.size.width || !bbox.size.height)
      return false;//Invalid polygon actually.

   CGFloat startRadius = grad->GetR1();
   CGFloat endRadius = grad->GetR2();

   if (grad->GetCoordinateMode() == TColorGradient::kObjectBoundingMode) {
      const CGFloat scale = std::max(bbox.size.width, bbox.size.height);

      startRadius *= scale;
      endRadius *= scale;
   } else {
      const CGFloat scale = std::max(sizeOfDrawable.width, sizeOfDrawable.height);

      startRadius *= scale;
      endRadius *= scale;
   }

   params.fStartRadius = startRadius;
   params.fEndRadius = endRadius;

   return true;
}

//______________________________________________________________________________
bool CalculateSimpleRadialGradientParameters(const TRadialGradient *grad,
                                             const CGSize &sizeOfDrawable,
                                             Int_t n, const TPoint *polygon,
                                             GradientParameters &params)
{
   assert(grad != nullptr &&
          "CalculateSimpleRadialGradientParameters, parameter 'grad' is null");
   assert(grad->GetGradientType() == TRadialGradient::kSimple &&
          "CalculateSimpleRadialGradientParameters, invalid gradient type");
   assert(sizeOfDrawable.width > 0. && sizeOfDrawable.height > 0. &&
          "CCalculateSimpleRadialGradientParameters, invalid destination drawable size");
   assert(n > 2 &&
          "CalculateSimpleRadialGradientParameters, parameter 'n' is not a valid number of points");
   assert(polygon != nullptr &&
          "CalculateSimpleRadialGradientParameters, parameter 'polygon' is null");


   const CGRect &bbox = FindBoundingBox(n, polygon);
   if (!bbox.size.width || !bbox.size.height)
      return false;//Invalid polygon actually.


   CGFloat radius = grad->GetRadius();
   CGPoint center = CGPointMake(grad->GetCenter().fX, grad->GetCenter().fY);

   if (grad->GetCoordinateMode() == TColorGradient::kObjectBoundingMode) {
      radius *= std::max(bbox.size.width, bbox.size.height);
      center.x = bbox.size.width * center.x + bbox.origin.x;
      center.y = bbox.size.height * center.y + bbox.origin.y;
   } else {
      radius *= std::max(sizeOfDrawable.width, sizeOfDrawable.height);
      center.x *= sizeOfDrawable.width;
      center.y *= sizeOfDrawable.height;
   }

   params.fStartPoint = center;
   params.fStartRadius = radius;

   return true;
}

//______________________________________________________________________________
bool CalculateGradientParameters(const TColorGradient *extendedColor,
                                 const CGSize &sizeOfDrawable,
                                 Int_t n, const TPoint *polygon,
                                 GradientParameters &params)
{
   assert(extendedColor != nullptr &&
          "CalculateGradientParameters, parameter 'extendedColor' is null");
   assert(sizeOfDrawable.width > 0. && sizeOfDrawable.height > 0. &&
          "CalculateGradientParameters, invalid destination drawable size");
   assert(n > 2 && "CalculateGradientParameters, parameter 'n' is not a valid number of points");
   assert(polygon != nullptr &&
          "CalculateGradientParameters, parameter 'polygon' is null");

   if (const TLinearGradient * const gl = dynamic_cast<const TLinearGradient *>(extendedColor))
      return CalculateGradientStartEnd(gl, sizeOfDrawable, n, polygon, params);
   else if (const TRadialGradient * const gr = dynamic_cast<const TRadialGradient *>(extendedColor)) {
      if (gr->GetGradientType() == TRadialGradient::kSimple) {
         return CalculateSimpleRadialGradientParameters(gr, sizeOfDrawable, n, polygon, params);
      } else {
         if (CalculateGradientStartEnd(gr, sizeOfDrawable, n, polygon, params))
            return CalculateGradientRadiuses(gr, sizeOfDrawable, n, polygon, params);
         return false;
      }
   }

   assert(0 && "CalculateGradientParamters, unknown gradient type");

   return false;
}

}//Unnamed namespace.

//______________________________________________________________________________
Bool_t SetFillColor(CGContextRef ctx, Color_t colorIndex)
{
   assert(ctx != nullptr && "SetFillColor, ctx parameter is null");

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
   assert(data != nullptr && "DrawPattern, data parameter is null");
   assert(ctx != nullptr && "DrawPattern, ctx parameter is null");

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
   assert(ctx != nullptr && "SetFillPattern, ctx parameter is null");
   assert(patternIndex != nullptr && "SetFillPattern, patternIndex parameter is null");

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
   assert(ctx != nullptr && "SetFillAreaParameters, ctx parameter is null");

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
      assert(patternIndex != nullptr && "SetFillAreaParameters, pattern index in null");

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
void DrawFillArea(CGContextRef ctx, Int_t n, TPoint *xy, Bool_t shadow)
{
   // Draw a filled area through all points.
   // n         : number of points
   // xy        : list of points

   assert(ctx != nullptr && "DrawFillArea, ctx parameter is null");
   assert(xy != nullptr && "DrawFillArea, xy parameter is null");

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
void DrawPolygonWithGradientFill(CGContextRef ctx, const TColorGradient *extendedColor, const CGSize &sizeOfDrawable,
                                 Int_t nPoints, const TPoint *xy, Bool_t drawShadow)
{
   using ROOT::MacOSX::Util::CFScopeGuard;

   assert(ctx != nullptr && "DrawPolygonWithGradientFill, ctx parameter is null");
   assert(nPoints != 0 && "DrawPolygonWithGradientFill, nPoints parameter is 0");
   assert(xy != nullptr && "DrawPolygonWithGradientFill, xy parameter is null");
   assert(extendedColor != nullptr &&
          "DrawPolygonWithGradientFill, extendedColor parameter is null");

   if (!sizeOfDrawable.width || !sizeOfDrawable.height)
      return;

   const CFScopeGuard<CGMutablePathRef> path(CGPathCreateMutable());
   if (!path.Get()) {
      ::Error("DrawPolygonWithGradientFill", "CGPathCreateMutable failed");
      return;
   }

   //Create a gradient.
   const CFScopeGuard<CGColorSpaceRef> baseSpace(CGColorSpaceCreateDeviceRGB());
   if (!baseSpace.Get()) {
      ::Error("DrawPolygonWithGradientFill", "CGColorSpaceCreateDeviceRGB failed");
      return;
   }

   typedef GradientFactory<Double_t, CGFloat> Factory;
   const CFScopeGuard<CGGradientRef> gradient(Factory::CreateGradient(baseSpace.Get(), extendedColor));
   if (!gradient.Get()) {
      ::Error("DrawPolygonWithGradientFill", "CGGradientCreateWithColorComponents failed");
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

   GradientParameters params;
   if (!CalculateGradientParameters(extendedColor, sizeOfDrawable, nPoints, xy, params))
      return;

   const TRadialGradient * const gr = dynamic_cast<const TRadialGradient *>(extendedColor);
   if (gr && (params.fStartRadius || params.fEndRadius)) {
      if (gr->GetGradientType() == TRadialGradient::kSimple) {
         CGContextDrawRadialGradient(ctx, gradient.Get(), params.fStartPoint, 0.,
                                     params.fStartPoint, params.fStartRadius,
                                     kCGGradientDrawsAfterEndLocation |
                                     kCGGradientDrawsBeforeStartLocation);
      } else {
         CGContextDrawRadialGradient(ctx, gradient.Get(), params.fStartPoint, params.fStartRadius,
                                     params.fEndPoint, params.fEndRadius,
                                     kCGGradientDrawsAfterEndLocation |
                                     kCGGradientDrawsBeforeStartLocation);
      }
   } else {
      CGContextDrawLinearGradient(ctx, gradient.Get(),
                                  params.fStartPoint, params.fEndPoint,
                                  kCGGradientDrawsAfterEndLocation |
                                  kCGGradientDrawsBeforeStartLocation);
   }
}

}//namespace Quartz
}//namespace ROOT
