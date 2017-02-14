// @(#)root/gpad:$Id$
// Author: Rene Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttCanvas
#define ROOT_TAttCanvas

#include "Rtypes.h"

class TAttCanvas {
private:
   Float_t    fXBetween;        ///< X distance between pads
   Float_t    fYBetween;        ///< Y distance between pads
   Float_t    fTitleFromTop;    ///< Y distance of Global Title from top
   Float_t    fXdate;           ///< X position where to draw the date
   Float_t    fYdate;           ///< X position where to draw the date
   Float_t    fAdate;           ///< Alignment for the date

public:
   TAttCanvas();
   virtual ~TAttCanvas();
   virtual void     Copy(TAttCanvas &attcanvas) const;
   Float_t          GetAdate() const { return fAdate;}
   Float_t          GetTitleFromTop() const { return fTitleFromTop;}
   Float_t          GetXBetween() const { return fXBetween;}
   Float_t          GetXdate() const { return fXdate;}
   Float_t          GetYBetween() const { return fYBetween;}
   Float_t          GetYdate() const { return fYdate;}
   virtual void     Print(Option_t *option="") const;
   virtual void     ResetAttCanvas(Option_t *option="");
   virtual void     SetAdate(Float_t adate) { fAdate=adate;}
   virtual void     SetTitleFromTop(Float_t titlefromtop)
                                        { fTitleFromTop=titlefromtop;}
   virtual void     SetXBetween(Float_t xbetween) { fXBetween=xbetween;}
   virtual void     SetXdate(Float_t xdate) { fXdate=xdate;}
   virtual void     SetYBetween(Float_t ybetween) { fYBetween=ybetween;}
   virtual void     SetYdate(Float_t ydate) { fYdate=ydate;}

   ClassDef(TAttCanvas,1)  //Canvas attributes
};

#endif

