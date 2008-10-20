// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLCameraOverlay
#define ROOT_TGLCameraOverlay

#include "TGLOverlay.h"
#include "TGLAxisPainter.h"
#include "TAttAxis.h"

class TGLCameraOverlay : public TGLOverlayElement
{
public:
   enum EMode { kPlaneIntersect, kBar, kAxis };

private:
   TGLCameraOverlay(const TGLCameraOverlay&);            // Not implemented
   TGLCameraOverlay& operator=(const TGLCameraOverlay&); // Not implemented

protected:
   Bool_t         fShowOrthographic;
   Bool_t         fShowPerspective;

   EMode          fOrthographicMode;
   EMode          fPerspectiveMode;

   TGLAxisPainter fAxisPainter;
   TGLAxisAttrib   fAxisAtt;

   Float_t          fAxisExtend;
   TGLPlane      fExternalRefPlane;
   Bool_t           fUseExternalRefPlane;

  Double_t         fFrustum[4]; // cached

   void    RenderPlaneIntersect(TGLRnrCtx& rnrCtx, const TGLFont &font);
   void    RenderAxis(TGLRnrCtx& rnrCtx);
   void    RenderBar(TGLRnrCtx& rnrCtx, const TGLFont &font);

public:
   TGLCameraOverlay(Bool_t showOrtho=kTRUE, Bool_t showPersp=kFALSE);
   virtual ~TGLCameraOverlay() {}

   virtual  void   Render(TGLRnrCtx& rnrCtx);

   TGLAxisAttrib&  RefAxisAttrib() { return fAxisAtt; }
   Float_t GetAxisExtend() const { return fAxisExtend; }
   void    SetAxisExtend(Float_t x) { fAxisExtend = x; }

   TGLPlane& RefExternalRefPlane() { return fExternalRefPlane; }
   void  UseExternalRefPlane(Bool_t x) { fUseExternalRefPlane=x; }
   Bool_t GetUseExternalRefPlane() const { return fUseExternalRefPlane; }

   Int_t    GetPerspectiveMode() const { return fPerspectiveMode;}
   void     SetPerspectiveMode(EMode m) {fPerspectiveMode = m;}
   Int_t    GetOrthographicMode() const { return fOrthographicMode;}
   void     SetOrthographicMode(EMode m) {fOrthographicMode = m;}

   Bool_t   GetShowOrthographic() const { return fShowOrthographic; }
   void     SetShowOrthographic(Bool_t x) {fShowOrthographic =x;}
   Bool_t   GetShowPerspective() const { return fShowPerspective; }
   void     SetShowPerspective(Bool_t x) {fShowPerspective =x;}


   ClassDef(TGLCameraOverlay, 1); // Show coorinates of current camera frustum.
};

#endif
