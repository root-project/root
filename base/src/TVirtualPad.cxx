// @(#)root/base:$Name:  $:$Id: TVirtualPad.cxx,v 1.2 2000/06/13 12:32:57 brun Exp $
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
TVirtualPad::TVirtualPad(const char *, const char *, Double_t,
           Double_t, Double_t, Double_t, Color_t color, Short_t , Short_t)
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

//______________________________________________________________________________
void TVirtualPad::Streamer(TBuffer &R__b)
{
   // Stream an object of class TVirtualPad.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TVirtualPad::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TObject::Streamer(R__b);
      TAttLine::Streamer(R__b);
      TAttFill::Streamer(R__b);
      TAttPad::Streamer(R__b);
      //====end of old versions
      
   } else {
      TVirtualPad::Class()->WriteBuffer(R__b,this);
   }
}

