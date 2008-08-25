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
private:
   TGLAxisPainter fAxisPainter;
   TGLAxisAttrib  fAxisAtt;

   Float_t        fAxisExtend;
   TGLPlane       fExternalRefPlane;
   Bool_t         fUseExternalRefPlane;

   Bool_t         fShowPerspective;
   Bool_t         fShowOrthographic;

   TGLCameraOverlay(const TGLCameraOverlay&);            // Not implemented
   TGLCameraOverlay& operator=(const TGLCameraOverlay&); // Not implemented

protected:
   void    RenderPerspective(TGLRnrCtx& rnrCtx, TGLVertex3 &v, const TGLFont &font);
   void    RenderOrthographic(TGLRnrCtx& rnrCtx);

public:
   TGLCameraOverlay();
   virtual ~TGLCameraOverlay() {}

   virtual  void   Render(TGLRnrCtx& rnrCtx);

   Bool_t   GetShowPerspective() const { return fShowPerspective; }
   void     SetShowPerspective(Bool_t x) {fShowPerspective =x;}

   Bool_t   GetShowOrthographic() const { return fShowOrthographic; }
   void     SetShowOrthographic(Bool_t x) {fShowOrthographic =x;}


   TGLAxisAttrib&  RefAxisAttrib() { return fAxisAtt; }
   Float_t GetAxisExtend() const { return fAxisExtend; }
   void    SetAxisExtend(Float_t x) { fAxisExtend = x; }

   TGLPlane& RefExternalRefPlane() { return fExternalRefPlane; }
   void  UseExternalRefPlane(Bool_t x) { fUseExternalRefPlane=x; }
   Bool_t GetUseExternalRefPlane() const { return fUseExternalRefPlane; }

   ClassDef(TGLCameraOverlay, 1); // Show coorinates of current camera frustum.
};

#endif
