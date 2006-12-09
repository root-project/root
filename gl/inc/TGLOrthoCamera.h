// @(#)root/gl:$Name:  $:$Id: TGLOrthoCamera.h,v 1.13 2006/08/31 13:42:14 couet Exp $
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

#ifndef ROOT_TArcBall
#include "TArcBall.h"
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
   enum EType { kXOY, kXOZ, kZOY  }; // pair of world axes aligned to h/v screen
private:
   // Fields
   EType          fType;         //! type (EType) - one of kXOY, kXOZ, kZOY

   // Limits - set in Setup()
   Double_t       fZoomMin;      //! minimum zoom factor
   Double_t       fZoomDefault;  //! default zoom factor
   Double_t       fZoomMax;      //! maximum zoom factor
   TGLBoundingBox fVolume;       //!

	// Current interaction
   Double_t       fZoom;         //! current zoom
   TGLVector3     fTruck;        //! current truck vector
   TGLMatrix      fMatrix;       //! orthographic orientation matrix

   static   UInt_t   fgZoomDeltaSens;

   //Stuff for TGLPlotPainter.
   Double_t       fShift;
   Double_t       fOrthoBox[4];
   TGLVertex3     fCenter;
   TArcBall       fArcBall;
   TPoint         fMousePos;
   Bool_t         fVpChanged;

   // Methods
   void Init();

public:

   // TODO: Convert this so define by pair of vectors as per perspective 
   // camera
   TGLOrthoCamera();
   TGLOrthoCamera(EType type);
   virtual ~TGLOrthoCamera();

   virtual void   Setup(const TGLBoundingBox & box, Bool_t reset=kTRUE);
   virtual void   Reset();
   virtual Bool_t Dolly(Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2);
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta);
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta);
   virtual void   Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect = 0) const;
   virtual void   Markup (TGLCameraMarkupStyle* ms) const;

   // External scripting control
   void Configure(Double_t left, Double_t right, Double_t top, Double_t bottom);

   //Stuff for TGLPlotPainter.
   void   SetViewport(Int_t context);
   void   SetViewVolume(const TGLVertex3 *box);
   void   StartRotation(Int_t px, Int_t py);
   void   RotateCamera(Int_t px, Int_t py);
   void   StartPan(Int_t px, Int_t py);
   void   Pan(Int_t px, Int_t py);
   void   ZoomIn();
   void   ZoomOut();
   void   SetCamera()const;
   void   Apply()const;

   Bool_t ViewportChanged()const{return fVpChanged;}
   Int_t  GetX()const;
   Int_t  GetY()const;
   Int_t  GetWidth()const;
   Int_t  GetHeight()const;

   ClassDef(TGLOrthoCamera,0) // an orthogonal view camera
};

#endif // ROOT_TGLOrthoCamera
