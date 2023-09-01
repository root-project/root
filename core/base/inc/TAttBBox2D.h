// @(#)root/base:$Id$
// Author: Anna-Pia Lohfink 27.3.2014

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttBBox2D
#define ROOT_TAttBBox2D

#include "GuiTypes.h"
#include "Rtypes.h"

class TPoint;
class TAttBBox2D {

public:
   virtual ~TAttBBox2D();
   virtual Rectangle_t     GetBBox()  = 0; //Get TopLeft Corner with width and height
   virtual TPoint          GetBBoxCenter() = 0;
   virtual void            SetBBoxCenter(const TPoint &p) = 0;
   virtual void            SetBBoxCenterX(const Int_t x) = 0;
   virtual void            SetBBoxCenterY(const Int_t y) = 0;
   virtual void            SetBBoxX1(const Int_t x) = 0; //set lhs of BB to value
   virtual void            SetBBoxX2(const Int_t x) = 0; //set rhs of BB to value
   virtual void            SetBBoxY1(const Int_t y) = 0; //set top of BB to value
   virtual void            SetBBoxY2(const Int_t y) = 0; //set bottom of BB to value

   ClassDef(TAttBBox2D,0);  //2D bounding box attributes
};

#endif
