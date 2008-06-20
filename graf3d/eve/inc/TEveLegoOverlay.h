// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveLegoOverlay
#define ROOT_TEveLegoOverlay

#include "TGLOverlay.h"
#include "TEveElement.h"
#include "TGLAxisPainter.h"

class TEveCaloLego;

class TEveLegoOverlay : public TGLOverlayElement,
                        public TEveElementList
{
private:
   TEveLegoOverlay(const TEveLegoOverlay&);            // Not implemented
   TEveLegoOverlay& operator=(const TEveLegoOverlay&); // Not implemented

   void DrawSlider(TGLRnrCtx& rnrCtx);

   Bool_t SetSliderVal(Event_t* event,TGLRnrCtx& rnrCtx );

protected:
   TEveCaloLego*  fCalo;

   Int_t          fActiveID;
   Color_t        fActiveCol;

   Float_t        fMenuW;
   Float_t        fButtonW;
   Float_t        fSliderH;    // slider height in % of viewport
   Float_t        fSliderPosY; // y position of slider bottom up

   Bool_t         fShowSlider;
   Float_t        fSliderVal;

   TGLAxisPainter fAxisPainter;
   TGLAxisAttrib fAxisAtt;

public:
   TEveLegoOverlay();
   virtual ~TEveLegoOverlay(){}

   virtual  Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual  Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec, Event_t* event);
   virtual  void   MouseLeave();

   virtual  void    Render(TGLRnrCtx& rnrCtx);

   TEveCaloLego* GetCaloLego() {return fCalo;}
   void SetCaloLego(TEveCaloLego* c) {fCalo = c;}

   ClassDef(TEveLegoOverlay, 0); // GL-overaly control GUI for TEveCaloLego.
};

#endif
