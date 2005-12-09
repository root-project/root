// @(#)root/gl:$Name:  $:$Id: TGLOrthoCamera.h,v 1.8 2005/11/22 18:05:46 brun Exp $
// Author:  Richard Maunder  25/05/2005

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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLOrthoCamera                                                       //
//                                                                      //
// Orthographic projection camera. Currently limited to three types     //
// defined at construction time - kXOY, kXOZ, kZOY - where this refers  //
// to the viewport plane axis - e.g. kXOY has X axis horizontal, Y      //
// vertical - i.e. looking down Z axis with Y vertical.                 //
//
// The plane types restriction could easily be removed to supported     //
// arbitary ortho projections along any axis/orientation with free      //
// rotations about them.                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGLOrthoCamera : public TGLCamera {
   public:
   enum EType { kXOY, kXOZ, kZOY  };
private:
   // Fields
   EType          fType;      //!

   // Limits - set in Setup()
   Double_t       fZoomMin;      //!
   Double_t       fZoomDefault;  //!
   Double_t       fZoomMax;      //!
   TGLBoundingBox fVolume;       //!

	// Current interaction
   Double_t       fZoom;      //!
   TGLVector3     fTruck;     //!
   TGLMatrix      fMatrix;    //!

   static   UInt_t   fgZoomDeltaSens;

   // Methods
   void Init();

public:

   // TODO: Convert this so define by pair of vectors as per perspective 
   // camera
   TGLOrthoCamera(EType type);
   virtual ~TGLOrthoCamera();

   virtual void   Setup(const TGLBoundingBox & box);
   virtual void   Reset();
   virtual Bool_t Dolly(Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta);
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta);
   virtual void   Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect = 0);

   // External scripting control
   void Configure(Double_t left, Double_t right, Double_t top, Double_t bottom);

   ClassDef(TGLOrthoCamera,0) // an orthogonal view camera
};

#endif // ROOT_TGLOrthoCamera
