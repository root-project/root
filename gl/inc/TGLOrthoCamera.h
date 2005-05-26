// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOrthoCamera
#define ROOT_TGLOrthoCamera

#ifndef ROOT_TGLCamera
#include "TGLCamera.h"
#endif

/*************************************************************************
 * TGLOrthoCamera - TODO
 *
 *
 *
 *************************************************************************/
class TGLOrthoCamera : public TGLCamera {
public:
   enum EType { kXOY, kYOZ, kXOZ };
private:
   // Fields  
   EType          fType;      //!

   // Set in SetLimits()
   Double_t       fZoomMin;    //!
   Double_t       fZoomDefault;//!
   Double_t       fZoomMax;    //!
   Double_t       fVolumeDiag; //!

   TGLVertex3     fCenter;    //!
   Double_t       fWidth;     //!
   Double_t       fHeight;    //!
   TGLMatrix      fMatrix;    //!
   Double_t       fZoom;      //!
   TGLVector3     fTruck;     //!
   
   static   UInt_t   fgZoomDeltaSens;
   
   // Methods
   void Init();
public:
   
   TGLOrthoCamera(EType type);
   virtual ~TGLOrthoCamera();

   virtual void   Setup(const TGLBoundingBox & box);
   virtual void   Reset();
   virtual Bool_t Dolly(Int_t delta);
   virtual Bool_t Zoom (Int_t delta);
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta);
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta);
   virtual void   Apply(const TGLBoundingBox & box, const TGLRect * pickRect = 0);
    
   ClassDef(TGLOrthoCamera,0) // an orthogonal view camera
};

#endif // ROOT_TGLOrthoCamera
