// @(#)root/base:$Name$:$Id$
// Author: Rene Brun   05/12/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualPad.h"
#include "X3DBuffer.h"

Size3D gSize3D;

void **(*gThreadTsd)(void*,Int_t) = 0;
Int_t (*gThreadXAR)(const char *xact, Int_t nb, void **ar, Int_t *iret) = 0;

//______________________________________________________________________________
TVirtualPad *&gPad
{
   static TVirtualPad *currentPad = 0;
   if (!gThreadTsd)
      return currentPad;
   else
      return *(TVirtualPad**)(*gThreadTsd)(&currentPad,0);
}


ClassImp(TVirtualPad)

//______________________________________________________________________________
//
//  TVirtualPad is an abstract base class for the Pad and Canvas classes.
//

//______________________________________________________________________________
TVirtualPad::TVirtualPad() : TAttPad()
{
   // VirtualPad default constructor

   fResizing = kFALSE;
}

//______________________________________________________________________________
TVirtualPad::TVirtualPad(const char *, const char *, Float_t,
           Float_t, Float_t, Float_t, Color_t color, Short_t , Short_t)
          : TAttPad()
{
   // VirtualPad constructor

   fResizing = kFALSE;

   SetFillColor(color);
   SetFillStyle(1001);
}

//______________________________________________________________________________
TVirtualPad::~TVirtualPad()
{
   // VirtualPad destructor

}

