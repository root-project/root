// @(#)root/gl:$Name:  $:$Id: TGLPerspectiveCamera.h,v 1.1.1.1 2007/04/04 16:01:43 mtadel Exp $
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

#ifndef ROOT_TGLCamera
#include "TGLCamera.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLPerspectiveCamera                                                 //
//                                                                      //
// Perspective projection camera - with characteristic foreshortening.  //
//                                                                      //
// TODO: Currently constrains YOZ plane to be floor - this is never     //
// 'tipped'. While useful we really need to extend so can:              //
// i) Pick any one of the three natural planes of the world to be floor.//
// ii) Can use a free arcball style camera with no contraint - integrate//
// TArcBall.                                                            //
//////////////////////////////////////////////////////////////////////////

class TGLPerspectiveCamera : public TGLCamera
{
   private:
   // Fields
   TGLVector3  fHAxis;           //!
   TGLVector3  fVAxis;           //!

   // Set in Setup()
   Double_t    fDollyMin;        //!
   Double_t    fDollyDefault;    //!
   Double_t    fDollyMax;        //!

   Double_t    fFOV;             //!
   Double_t    fDolly;           //!
   Double_t    fVRotate;         //!
   Double_t    fHRotate;         //!
   TGLVertex3  fCenter;          //!
   TGLVector3  fTruck;           //!

   // These are fixed for any perspective camera
   static   Double_t fgFOVMin, fgFOVDefault, fgFOVMax;
   static   UInt_t   fgDollyDeltaSens, fgFOVDeltaSens;

public:
   TGLPerspectiveCamera(const TGLVector3 & hAxis, const TGLVector3 & vAxis);
   virtual ~TGLPerspectiveCamera();

   virtual void   Setup(const TGLBoundingBox & box, Bool_t reset=kTRUE);
   virtual void   Reset();
   virtual Bool_t Dolly(Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta);
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta);

   virtual void   Apply(const TGLBoundingBox & box, const TGLRect * pickRect = 0) const;

   // External scripting control
   void Configure(Double_t fov, Double_t dolly, Double_t center[3],
                  Double_t hRotate, Double_t vRotate);

   ClassDef(TGLPerspectiveCamera,0) // a perspective view camera
};

#endif // ROOT_TGLPerspectiveCamera

