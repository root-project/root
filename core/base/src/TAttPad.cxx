// @(#)root/base:$Id$
// Author: Rene Brun   04/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Strlen.h"
#include "TAttPad.h"
#include "TBuffer.h"
#include "TStyle.h"
#include "TClass.h"

ClassImp(TAttPad)

//______________________________________________________________________________
//
//  Manages default Pad attributes. Referenced by TStyle.
//

//______________________________________________________________________________
TAttPad::TAttPad()
{
   //constructor
   ResetAttPad();
}

//______________________________________________________________________________
TAttPad::~TAttPad()
{
   //destructor
}

//______________________________________________________________________________
void TAttPad::Copy(TAttPad &attpad) const
{
   //copy function
   attpad.fLeftMargin   = fLeftMargin;
   attpad.fRightMargin  = fRightMargin;
   attpad.fBottomMargin = fBottomMargin;
   attpad.fTopMargin    = fTopMargin;
   attpad.fXfile   = fXfile;
   attpad.fYfile   = fYfile;
   attpad.fAfile   = fAfile;
   attpad.fXstat   = fXstat;
   attpad.fYstat   = fYstat;
   attpad.fAstat   = fAstat;
   attpad.fFrameFillColor = fFrameFillColor;
   attpad.fFrameFillStyle = fFrameFillStyle;
   attpad.fFrameLineColor = fFrameLineColor;
   attpad.fFrameLineStyle = fFrameLineStyle;
   attpad.fFrameLineWidth = fFrameLineWidth;
   attpad.fFrameBorderSize= fFrameBorderSize;
   attpad.fFrameBorderMode= fFrameBorderMode;
}

//______________________________________________________________________________
void TAttPad::Print(Option_t *) const
{
   //print function
}

//______________________________________________________________________________
void TAttPad::ResetAttPad(Option_t *)
{
   //reset pad attributes
   fLeftMargin   = gStyle->GetPadLeftMargin();
   fRightMargin  = gStyle->GetPadRightMargin();
   fBottomMargin = gStyle->GetPadBottomMargin();
   fTopMargin    = gStyle->GetPadTopMargin();
   fXfile  = 2;
   fYfile  = 2;
   fAfile  = 1;
   fXstat  = 0.99;
   fYstat  = 0.99;
   fAstat  = 2;
   fFrameLineColor = gStyle->GetFrameLineColor();
   fFrameFillColor = gStyle->GetFrameFillColor();
   fFrameFillStyle = gStyle->GetFrameFillStyle();
   fFrameLineStyle = gStyle->GetFrameLineStyle();
   fFrameLineWidth = gStyle->GetFrameLineWidth();
   fFrameBorderSize= gStyle->GetFrameBorderSize();
   fFrameBorderMode= gStyle->GetFrameBorderMode();
}

//______________________________________________________________________________
void TAttPad::SetBottomMargin(Float_t margin)
{
//*-*-*-*-*-*-*-*-*Set Pad bottom margin in fraction of the pad height*-*-*-*
//*-*              ===================================================
   if (margin < 0 || margin >=1) margin = 0.1;
   if (margin + fTopMargin >= 1) return;
   fBottomMargin = margin;
}

//______________________________________________________________________________
void TAttPad::SetLeftMargin(Float_t margin)
{
//*-*-*-*-*-*-*-*-*Set Pad left margin in fraction of the pad width*-*-*-*-*
//*-*              ================================================
   if (margin < 0 || margin >=1) margin = 0.1;
   if (margin + fRightMargin >= 1) return;
   fLeftMargin = margin;
}

//______________________________________________________________________________
void TAttPad::SetRightMargin(Float_t margin)
{
//*-*-*-*-*-*-*-*-*Set Pad right margin in fraction of the pad width*-*-*-*-*
//*-*              =================================================
   if (margin < 0 || margin >=1) margin = 0.1;
   if (margin + fLeftMargin >= 1) return;
   fRightMargin = margin;
}

//______________________________________________________________________________
void TAttPad::SetTopMargin(Float_t margin)
{
//*-*-*-*-*-*-*-*-*Set Pad top margin in fraction of the pad height*-*-*-*-*
//*-*              ================================================
   if (margin < 0 || margin >=1) margin = 0.1;
   if (margin + fBottomMargin >= 1) return;
   fTopMargin = margin;
}

//______________________________________________________________________________
void TAttPad::SetMargin(Float_t left, Float_t right, Float_t bottom, Float_t top)
{
//*-*-*-*-*-*-*-*-*Set all margins

   SetLeftMargin(left);
   SetRightMargin(right);
   SetBottomMargin(bottom);
   SetTopMargin(top);
}

//______________________________________________________________________________
void TAttPad::Streamer(TBuffer &R__b)
{
   // Stream an object of class TAttPad.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 2) {
         R__b.ReadClassBuffer(TAttPad::Class(), this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      R__b >> fLeftMargin;
      R__b >> fRightMargin;
      R__b >> fBottomMargin;
      R__b >> fTopMargin;
      R__b >> fXfile;
      R__b >> fYfile;
      R__b >> fAfile;
      R__b >> fXstat;
      R__b >> fYstat;
      R__b >> fAstat;
      if (R__v > 1) {
         R__b >> fFrameFillColor;
         R__b >> fFrameLineColor;
         R__b >> fFrameFillStyle;
         R__b >> fFrameLineStyle;
         R__b >> fFrameLineWidth;
         R__b >> fFrameBorderSize;
         R__b >> fFrameBorderMode;
      }
      //====end of old versions

   } else {
      R__b.WriteClassBuffer(TAttPad::Class(),this);
   }
}
