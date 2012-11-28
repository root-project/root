// @(#)root/graf2d:$Id$
// Author: Olivier Couet, 24/01/2012

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <cassert>
#include <vector>

#include "CocoaUtils.h"
#include "TObjString.h"
#include "QuartzLine.h"
#include "RStipples.h"
#include "TObjArray.h"
#include "TString.h"
#include "TColor.h"
#include "TStyle.h"
#include "TROOT.h"

namespace ROOT {
namespace Quartz {

//______________________________________________________________________________
Bool_t SetLineColor(CGContextRef ctx, Color_t colorIndex)
{
   assert(ctx != 0 && "SetLineColor, ctx parameter is null");

   const TColor * const color = gROOT->GetColor(colorIndex);
   if (!color)
      return kFALSE;

   const CGFloat alpha = color->GetAlpha();
   Float_t rgb[3] = {};
   color->GetRGB(rgb[0], rgb[1], rgb[2]);
   CGContextSetRGBStrokeColor(ctx, rgb[0], rgb[1], rgb[2], alpha);

   return kTRUE;
}

//______________________________________________________________________________
void SetLineType(CGContextRef ctx, Int_t n, Int_t *dash)
{
   // Set the line type in the context ctx.
   //
   // n       - length of the dash list
   //           n <= 0 use solid lines
   //           n >  0 use dashed lines described by dash(n)
   //                 e.g. n = 4,dash = (6,3,1,3) gives a dashed-dotted line
   //                 with dash length 6 and a gap of 7 between dashes
   // dash(n) - dash segment lengths

   assert(ctx != 0 && "SetLineType, ctx parameter is null");

   if (n) {
      CGFloat lengths[n];
      for (int i = 0; i < n; i++)
         lengths[i] = dash[i];
      CGContextSetLineDash(ctx, 0, lengths, n);
   } else {
      CGContextSetLineDash(ctx, 0, NULL, 0);
   }
}
   
//______________________________________________________________________________
void SetLineStyle(CGContextRef ctx, Int_t lstyle)
{
   // Set current line style in the context ctx.
   assert(ctx != 0 && "SetLineStyle, ctx parameter is null");
      
   static Int_t dashed[2] = {3, 3};
   static Int_t dotted[2] = {1, 2};
   static Int_t dasheddotted[4] = {3, 4, 1, 4};
      
   if (lstyle <= 1 ) {
      SetLineType(ctx, 0, 0);
   } else if (lstyle == 2) {
      SetLineType(ctx, 2, dashed);
   } else if (lstyle == 3) {
      SetLineType(ctx, 2,dotted);
   } else if (lstyle == 4) {
      SetLineType(ctx, 4, dasheddotted);
   } else {
      TString st = (TString)gStyle->GetLineStyleString(lstyle);
      TObjArray *tokens = st.Tokenize(" ");
      Int_t nt;
      nt = tokens->GetEntries();
      std::vector<Int_t> linestyle(nt);
      for (Int_t j = 0; j<nt; j++) {
         Int_t it;
         sscanf(((TObjString*)tokens->At(j))->GetName(), "%d", &it);
         linestyle[j] = (Int_t)(it/4);
      }
      SetLineType(ctx, nt, &linestyle[0]);
      delete tokens;
   }
}

//______________________________________________________________________________
void SetLineWidth(CGContextRef ctx, Int_t width)
{
   // Set the line width in the context ctx.
   //
   // width - the line width in pixels
   assert(ctx != 0 && "SetLineWidth, ctx parameter is null");
   
            
   if (width < 0)
      return;

   CGContextSetLineWidth(ctx, width);
}

//______________________________________________________________________________
void DrawLine(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   assert(ctx != 0 && "DrawLine, ctx parameter is null");

   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, x1, y1);
   CGContextAddLineToPoint(ctx, x2, y2);
   CGContextStrokePath(ctx);
}


//______________________________________________________________________________
void DrawPolyLine(CGContextRef ctx, Int_t n, TPoint * xy)
{
   // Draw a line through all points.
   // n         : number of points
   // xy        : list of points
   
   assert(ctx != 0 && "DrawPolyLine, ctx parameter is null");
   assert(xy != 0 && "DrawPolyLine, xy parameter is null");
   
   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, xy[0].fX, xy[0].fY);
   for (Int_t i = 1; i < n; ++i)
      CGContextAddLineToPoint(ctx, xy[i].fX, xy[i].fY);
   
   if (xy[n - 1].fX == xy[0].fX && xy[n - 1].fY == xy[0].fY)
      CGContextClosePath(ctx);
   
   CGContextStrokePath(ctx);
}

}//namespace Quartz
}//namespace ROOT
