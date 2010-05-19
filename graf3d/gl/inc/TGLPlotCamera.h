// @(#)root/gl:$Id$
// Author: Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPlotCamera
#define ROOT_TGLPlotCamera

#include "TGLUtil.h"
#include "TArcBall.h"
#include "TPoint.h"

class TGLPaintDevice;

class TGLPlotCamera
{
private:
   TGLPlotCamera(const TGLPlotCamera&);            // Not implemented
   TGLPlotCamera& operator=(const TGLPlotCamera&); // Not implemented

protected:
   TGLRect        fViewport;
   Double_t       fZoom;
   Double_t       fShift;
   Double_t       fOrthoBox[4];
   TGLVertex3     fCenter;
   TGLVector3     fTruck;
   TArcBall       fArcBall;
   TPoint         fMousePos;
   Bool_t         fVpChanged;

public:
   TGLPlotCamera();
   virtual ~TGLPlotCamera() {}

   void   SetViewport(const TGLRect &vp);

   void   SetViewVolume(const TGLVertex3 *box);
   void   StartRotation(Int_t px, Int_t py);
   void   RotateCamera(Int_t px, Int_t py);
   void   StartPan(Int_t px, Int_t py);
   void   Pan(Int_t px, Int_t py);
   void   ZoomIn();
   void   ZoomOut();
   void   SetCamera()const;
   void   Apply(Double_t phi, Double_t theta)const;
   Bool_t ViewportChanged()const{return fVpChanged;}
   Int_t  GetX()const;
   Int_t  GetY()const;
   Int_t  GetWidth()const;
   Int_t  GetHeight()const;

   ClassDef(TGLPlotCamera, 0); // Camera for plot-painters.
};

#endif
