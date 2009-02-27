// @(#)root/eve:$Id$
// Author: Alja Mrak-Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloLegoOverlay
#define ROOT_TEveCaloLegoOverlay

#include "TGLCameraOverlay.h"
#include "TEveElement.h"

class TEveCaloLego;

class TEveCaloLegoOverlay : public TGLCameraOverlay,
                        public TEveElementList
{
private:
   TEveCaloLegoOverlay(const TEveCaloLegoOverlay&);            // Not implemented
   TEveCaloLegoOverlay& operator=(const TEveCaloLegoOverlay&); // Not implemented

   void DrawSlider(TGLRnrCtx& rnrCtx);

   Bool_t SetSliderVal(Event_t* event,TGLRnrCtx& rnrCtx );

   TString        fHeaderTxt;
   Bool_t         fHeaderSelected;

protected:
   TEveCaloLego*  fCalo;
   TAxis         *fSliderAxis;

   Color_t        fMainColor;

   Bool_t         fShowCamera;
   Bool_t         fShowPlane;

   // plane-value
   Float_t        fMenuW;
   Float_t        fButtonW;
   Float_t        fSliderH;    // slider height in % of viewport
   Float_t        fSliderPosY; // y position of slider bottom up
   Bool_t         fShowSlider;
   Float_t        fSliderVal;

   // event handling
   Int_t           fActiveID;
   Color_t         fActiveCol;

   virtual  void   RenderPlaneInterface(TGLRnrCtx& rnrCtx);
   virtual  void   RenderHeader(TGLRnrCtx& rnrCtx);

public:
   TEveCaloLegoOverlay();
   virtual ~TEveCaloLegoOverlay(){}

   // event handling
   virtual  Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual  Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec, Event_t* event);
   virtual  void   MouseLeave();

   //rendering
   virtual  void   Render(TGLRnrCtx& rnrCtx);

   TEveCaloLego* GetCaloLego() {return fCalo;}
   void SetCaloLego(TEveCaloLego* c) {fCalo = c;}

   void SetShowCamera (Bool_t x) { fShowCamera = x; }
   Bool_t GetShowCamera() const { return fShowCamera; }
   void SetShowPlane (Bool_t x) { fShowPlane = x; }
   Bool_t GetShowPlane() const { return fShowPlane; }

   void  SetHeaderTxt(const char *txt) {fHeaderTxt = txt; }
   const char* GetHeaderTxt() const { return fHeaderTxt; }

   ClassDef(TEveCaloLegoOverlay, 0); // GL-overaly control GUI for TEveCaloLego.
};

#endif
