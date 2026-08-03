// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TAttMarker.h"

#include "QuartzMarker.h"
#include "QuartzLine.h"
#include "QuartzFillArea.h"
#include "TMath.h"

namespace ROOT {
namespace Quartz {

//______________________________________________________________________________
void DrawPolyMarker(CGContextRef ctx, unsigned nPoints, const TPoint *xy,
                    const TAttMarker &attmark, float scaleFactor)
{
   Int_t markerSize = 0;
   std::vector<TPoint> markerShape;

   auto markerType = attmark.GetMarkerShape(markerSize, markerShape, scaleFactor);
   auto markerLineWidth = TAttMarker::GetMarkerLineWidth(attmark.GetMarkerStyle());

   if (!Quartz::SetFillColor(ctx, attmark.GetMarkerColor()))
      return;

   Quartz::SetLineColor(ctx, attmark.GetMarkerColor()); //Can not fail (for coverity).
   Quartz::SetLineStyle(ctx, 1);
   Quartz::SetLineWidth(ctx, markerLineWidth);

   if (scaleFactor > 1.)
      CGContextScaleCTM(ctx, 1. / scaleFactor, 1. / scaleFactor);

   // The fast pixel markers need to be treated separately
   if (markerType == TAttMarker::kShapeSegments) {
      CGContextSetLineJoin(ctx, kCGLineJoinMiter);
      CGContextSetLineCap(ctx, kCGLineCapButt);
   } else {
      CGContextSetLineJoin(ctx, kCGLineJoinRound);
      CGContextSetLineCap(ctx, kCGLineCapRound);
   }

   switch(markerType) {
      case TAttMarker::kShapeDot:
         for (unsigned i = 0; i < nPoints; ++i)
            CGContextFillRect(ctx, CGRectMake(xy[i].fX, xy[i].fY, 1, 1));
         break;
      case TAttMarker::kShapeCircle:
      case TAttMarker::kShapeFilledCircle: {
         for (unsigned i = 0; i < nPoints; ++i) {
            const CGRect rect = CGRectMake(xy[i].fX - markerSize * 0.5, xy[i].fY - markerSize * 0.5, markerSize, markerSize);
            if (markerType == TAttMarker::kShapeCircle)
               CGContextStrokeEllipseInRect(ctx, rect);
            else
               CGContextFillEllipseInRect(ctx, rect);
         }
         break;
      }
      case TAttMarker::kShapePolyLine:
      case TAttMarker::kShapeFilledArea:
         for (unsigned i = 0; i < nPoints; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            CGContextBeginPath(ctx);
            CGContextMoveToPoint(ctx, x + markerShape[0].fX, y + markerShape[0].fY);
            for (std::size_t p = 1; p < markerShape.size(); p++)
               CGContextAddLineToPoint(ctx, x + markerShape[p].fX, y + markerShape[p].fY);

            if (markerType == TAttMarker::kShapePolyLine)
               CGContextStrokePath(ctx);
            else
               CGContextFillPath(ctx);
         }
         break;
      case TAttMarker::kShapeSegments:
         for (unsigned i = 0; i < nPoints; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            for (std::size_t p = 0; p < markerShape.size(); p += 2) {
               CGContextBeginPath(ctx);
               CGContextMoveToPoint(ctx, x + markerShape[p].fX, y + markerShape[p].fY);
               CGContextAddLineToPoint(ctx, x + markerShape[p + 1].fX, y + markerShape[p + 1].fY);
               CGContextStrokePath(ctx);
            }
         }
         break;
      case TAttMarker::kShapeTriangles:
         // filled triangles
         for (unsigned i = 0; i < nPoints; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            for(std::size_t t = 0; t < markerShape.size(); t += 3) {
               CGContextBeginPath(ctx);
               CGContextMoveToPoint(ctx, x + markerShape[t].fX, y + markerShape[t].fY);
               for (std::size_t p = t + 1; p < t + 3; p++)
                  CGContextAddLineToPoint(ctx, x + markerShape[p].fX, y + markerShape[p].fY);
               CGContextFillPath(ctx);
            }
         }
         break;
   }

   CGContextSetLineJoin(ctx, kCGLineJoinMiter);
   CGContextSetLineCap(ctx, kCGLineCapButt);
}

} //namespace Quartz
} //namespace ROOT
