// @(#)root/gl:$Name:  $:$Id: TGLPerspectiveCamera.h,v 1.3 2005/05/26 12:29:50 rdm Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

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

/*************************************************************************
 * TGLPerspectiveCamera - TODO
 *
 *
 *
 *************************************************************************/
class TGLPerspectiveCamera : public TGLCamera {
private:
   // Fields

   // Set in Setup()
   Double_t    fDollyMin;        //!
   Double_t    fDollyDefault;    //!
   Double_t    fDollyMax;        //!
   Double_t    fVolumeDiag;      //!

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
   TGLPerspectiveCamera();
   virtual ~TGLPerspectiveCamera();

   virtual void   Setup(const TGLBoundingBox & box);
   virtual void   Reset();
   virtual Bool_t Dolly(Int_t delta);
   virtual Bool_t Zoom (Int_t delta);
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta);
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta);
   virtual void   Apply(const TGLBoundingBox & box, const TGLRect * pickRect = 0);

   ClassDef(TGLPerspectiveCamera,0) // a perspective view camera
};

#endif // ROOT_TGLPerspectiveCamera

