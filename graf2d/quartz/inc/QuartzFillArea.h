// @(#)root/graf2d:$Id$
// Author: Olivier Couet, 23/01/2012

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_QuartzFillArea
#define ROOT_QuartzFillArea

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// QuartzFillArea                                                       //
//                                                                      //
// Aux. functions to draw fill area.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include <Cocoa/Cocoa.h>

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

class TColorGradient;

namespace ROOT {
namespace Quartz {
   
void DrawBox(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2, Int_t mode);
void DrawBoxGradient(CGContextRef ctx, Int_t x1, Int_t y1, Int_t x2, Int_t y2, const TColorGradient *extendedColor, Bool_t drawShadow);

void DrawFillArea(CGContextRef ctx, Int_t n, TPoint * xy, Bool_t drawShadow);
void DrawFillAreaGradient(CGContextRef ctx, Int_t nPoints, const TPoint *xy, const TColorGradient *extendedColor, Bool_t drawShadow);

void SetFillStyle(CGContextRef ctx, Int_t style, 
                  Float_t r, Float_t g, Float_t b, Float_t a);
void SetStencilPattern(CGContextRef ctx, 
                       Float_t r, Float_t g, Float_t b, Float_t a);

}
}

#endif
