// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   02/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDimension
#define ROOT_TGDimension


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGDimension + TGPosition                                             //
//                                                                      //
// Two small classes that implement dimensions (width and height) and   //
// positions (x and y). They are trivial and their members are public.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TGDimension {
public:
   UInt_t  fWidth;       // width
   UInt_t  fHeight;      // height

   TGDimension() { fWidth = fHeight = 0; }
   TGDimension(UInt_t width, UInt_t height) { fWidth = width; fHeight = height; }
   TGDimension(const TGDimension &d) { fWidth = d.fWidth; fHeight = d.fHeight; }
   ClassDef(TGDimension,0)  // Dimension object (width, height)
};


class TGPosition {
public:
   Int_t  fX;         // x position
   Int_t  fY;         // y position

   TGPosition() { fX = fY = 0; }
   TGPosition(Int_t xc, Int_t yc) { fX = xc; fY = yc; }
   TGPosition(const TGPosition &p) { fX = p.fX; fY = p.fY; }
   ClassDef(TGPosition,0)  // Position object (x and y)
};

#endif
