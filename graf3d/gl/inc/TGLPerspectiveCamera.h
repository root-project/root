// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPerspectiveCamera
#define ROOT_TGLPerspectiveCamera

#include "TGLCamera.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPerspectiveCamera                                                 //
//                                                                      //
// Perspective projection camera - with characteristic foreshortening.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLPerspectiveCamera : public TGLCamera
{
private:
   // Fields
   Double_t    fFOV;

   static   Double_t fgFOVMin, fgFOVDefault, fgFOVMax;
   static   UInt_t   fgFOVDeltaSens;

public:
   TGLPerspectiveCamera();
   TGLPerspectiveCamera(const TGLVector3 & hAxis, const TGLVector3 & vAxis);
   virtual ~TGLPerspectiveCamera();

   virtual Bool_t IsPerspective() const { return kTRUE; }

   Double_t GetFOV() const { return fFOV; }
   void     SetFOV(Double_t fov) { fFOV = fov; }

   virtual void   Setup(const TGLBoundingBox & box, Bool_t reset=kTRUE);
   virtual void   Reset();
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2);
   using   TGLCamera::Truck;
   virtual Bool_t Truck(Int_t xDelta, Int_t yDelta, Bool_t mod1, Bool_t mod2);
   virtual void   Apply(const TGLBoundingBox & box, const TGLRect * pickRect = nullptr) const;

   // External scripting control
   virtual void Configure(Double_t fov, Double_t dolly, Double_t center[3],
                          Double_t hRotate, Double_t vRotate);

   ClassDef(TGLPerspectiveCamera,1) // Camera for perspective view.
};

#endif // ROOT_TGLPerspectiveCamera

