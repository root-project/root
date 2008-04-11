// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloLegoGL
#define ROOT_TEveCaloLegoGL

#include "TGLObject.h"
#include "TEveCaloData.h"
#include "TGLIncludes.h"
#include "TGLFontManager.h"
#include <map>

class TEveCaloLego;

class TEveCaloLegoGL : public TGLObject
{
public:
   enum EMode { kDetailed, kSimplified };
private:
   TEveCaloLegoGL(const TEveCaloLegoGL&);            // Not implemented
   TEveCaloLegoGL& operator=(const TEveCaloLegoGL&); // Not implemented

protected:
   Int_t   GetGridStep(Int_t axId, const TAxis* ax, TGLRnrCtx &rnrCtx) const;

   void    SetFont(Float_t l, TGLRnrCtx &rnrCtx) const;
   void    RnrText(const char* txt, Float_t x, Float_t y, Float_t z, 
                   const TGLFont &font, Int_t mode) const;

   void    DrawZAxis(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const;
   void    DrawZAxisSimplified(TGLRnrCtx &rnrCtx, Float_t x0, Float_t y0) const;
   void    DrawXYAxis(TGLRnrCtx &rnrCtx, Float_t x0, Float_t x1, Float_t y0, Float_t y1) const;
   void    DrawHistBase(TGLRnrCtx &rnrCtx) const;

   void    MakeQuad(Float_t x, Float_t y, Float_t z, 
                    Float_t xw, Float_t yw, Float_t zh) const;
   void    MakeDisplayList() const;

   mutable Bool_t                   fDLCacheOK;
   mutable std::map< Int_t, UInt_t> fDLMap;

   mutable EMode                    fMode;

   TEveCaloLego            *fM;  // Model object.

   mutable TGLFont          fNumFont;
   mutable TGLFont          fSymbolFont;
   mutable Int_t            fFontSize; // font size in pixels

   const   Float_t          fTMSize; // tick mark size

   // grid density modes
   Float_t                  fMinBinWidth;
   Int_t                    fNBinSteps;
   Int_t*                   fBinSteps;

public:
   TEveCaloLegoGL();
   virtual ~TEveCaloLegoGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);

   virtual void  SetBBox();

   virtual Bool_t ShouldDLCache(const TGLRnrCtx & rnrCtx) const;
   virtual void   DLCacheDrop();
   virtual void   DLCachePurge();

   virtual void   DirectDraw(TGLRnrCtx & rnrCtx) const;
   void  DrawDetailed(TGLRnrCtx & rnrCtx) const;
   void  DrawSimplified(TGLRnrCtx & rnrCt) const;

   virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   virtual void ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEveCaloLegoGL, 0); // GL renderer class for TEveCaloLego.
};

#endif
