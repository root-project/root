// @(#)root/gui:$Name:  $:$Id: TGDimension.h,v 1.2 2000/06/27 15:12:23 rdm Exp $
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

   Bool_t operator==(const TGDimension &b) const
      { return ((fWidth == b.fWidth) && (fHeight == b.fHeight)); }
   TGDimension operator-(const TGDimension &b) const
      { return TGDimension(fWidth - b.fWidth, fHeight - b.fHeight); }
   TGDimension operator+(const TGDimension &b) const
      { return TGDimension(fWidth + b.fWidth, fHeight + b.fHeight); }
   ClassDef(TGDimension,0)  // Dimension object (width, height)
};


class TGPosition {
public:
   Int_t  fX;         // x position
   Int_t  fY;         // y position

   TGPosition() { fX = fY = 0; }
   TGPosition(Int_t xc, Int_t yc) { fX = xc; fY = yc; }
   TGPosition(const TGPosition &p) { fX = p.fX; fY = p.fY; }

   Bool_t operator==(const TGPosition &b) const
      { return ((fX == b.fX) && (fY == b.fY)); }
   TGPosition operator-(const TGPosition &b) const
      { return TGPosition(fX - b.fX, fY - b.fY); }
   TGPosition operator+(const TGPosition &b) const
      { return TGPosition(fX + b.fX, fY + b.fY); }
   ClassDef(TGPosition,0)  // Position object (x and y are Int_t)
};

class TGLongPosition {
public:
   Long_t  fX;         // x position
   Long_t  fY;         // y position

   TGLongPosition() { fX = fY = 0; }
   TGLongPosition(Long_t xc, Long_t yc) { fX = xc; fY = yc; }
   TGLongPosition(const TGLongPosition &p) { fX = p.fX; fY = p.fY; }

   Bool_t operator==(const TGLongPosition &b) const
      { return ((fX == b.fX) && (fY == b.fY)); }
   TGLongPosition operator-(const TGLongPosition &b) const
      { return TGLongPosition(fX - b.fX, fY - b.fY); }
   TGLongPosition operator+(const TGLongPosition &b) const
      { return TGLongPosition(fX + b.fX, fY + b.fY); }
   ClassDef(TGLongPosition,0)  // Position object (x and y are Long_t)
};

class TGInsets {
public:
   Int_t  fL;    // left
   Int_t  fR;    // right
   Int_t  fT;    // top
   Int_t  fB;    // bottom

   TGInsets() { fL = fT = fR = fB = 0; }
   TGInsets(Int_t lf, Int_t rg, Int_t tp, Int_t bt)
      { fL = lf; fR = rg; fT = tp; fB = bt; }
   TGInsets(const TGInsets &in)
      { fL = in.fL; fR = in.fR; fT = in.fT; fB = in.fB; }

   Bool_t operator==(const TGInsets &in) const
      { return ((fL == in.fL) && (fR == in.fR) && (fT == in.fT) && (fB == in.fB)); }
   ClassDef(TGInsets,0)   // Inset (left, right, top, bottom)
};

#endif
