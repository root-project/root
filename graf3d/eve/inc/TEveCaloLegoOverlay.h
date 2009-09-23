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

class TEveCaloLegoOverlay : public TGLCameraOverlay
{
private:
   TEveCaloLegoOverlay(const TEveCaloLegoOverlay&);            // Not implemented
   TEveCaloLegoOverlay& operator=(const TEveCaloLegoOverlay&); // Not implemented

   Bool_t SetSliderVal(Event_t* event,TGLRnrCtx& rnrCtx );


protected:
   void   RenderLogaritmicScales(TGLRnrCtx& rnrCtx);
   void   RenderPaletteScales(TGLRnrCtx& rnrCtx);
   void   RenderPlaneInterface(TGLRnrCtx& rnrCtx);
   void   RenderHeader(TGLRnrCtx& rnrCtx);

   TEveCaloLego*  fCalo; // model

   // 2D scales
   Bool_t         fShowScales;
   Color_t        fScaleColor;
   UChar_t        fScaleTransparency; //transaprency in %
   Double_t       fScaleCoordX;
   Double_t       fScaleCoordY;
   Double_t       fCellX;
   Double_t       fCellY;

   Color_t        fFrameColor;
   UChar_t        fFrameLineTransp;
   UChar_t        fFrameBgTransp;;

   // move of scales
   Int_t             fMouseX, fMouseY; //! last mouse position
   Bool_t            fInDrag;

   // text top right corner
   TString        fHeaderTxt;
   Bool_t         fHeaderSelected;

   // plane ojects
   TAxis         *fPlaneAxis;
   Color_t        fAxisPlaneColor;
   Bool_t         fShowPlane;
   // plane state
   Float_t        fMenuW;
   Float_t        fButtonW;
   Bool_t         fShowSlider;
   Float_t        fSliderH;    // slider height in % of viewport
   Float_t        fSliderPosY; // y position of slider bottom up
   Float_t        fSliderVal;
   // plane event-handling
   Int_t          fActiveID;
   Color_t        fActiveCol;


public:
   TEveCaloLegoOverlay();
   virtual ~TEveCaloLegoOverlay(){}

   //rendering
   virtual  void   Render(TGLRnrCtx& rnrCtx);

   // event handling
   virtual  Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual  Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec, Event_t* event);
   virtual  void   MouseLeave();


   TEveCaloLego* GetCaloLego() {return fCalo;}
   void          SetCaloLego(TEveCaloLego* c) {fCalo = c;}

   void          SetShowPlane (Bool_t x) { fShowPlane = x; }
   Bool_t        GetShowPlane() const { return fShowPlane; }

   void          SetHeaderTxt(const char *txt) {fHeaderTxt = txt; }
   const char*   GetHeaderTxt() const { return fHeaderTxt; }

   void          SetShowScales(Bool_t x) { fShowScales = x;}
   void          SetScaleColorTransparency(Color_t colIdx, UChar_t transp);
   void          SetScalePosition(Double_t x, Double_t y);

   void          SetFrameAttribs(Color_t frameCol, UChar_t lineTransp, UChar_t bgTransp);

   ClassDef(TEveCaloLegoOverlay, 0); // GL-overaly control GUI for TEveCaloLego.
};

#endif
