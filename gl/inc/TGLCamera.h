// @(#)root/gl:$Name:  $:$Id: TGLCamera.h,v 1.20 2006/02/26 16:08:10 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLCamera
#define ROOT_TGLCamera

#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TGLBoundingBox
#include "TGLBoundingBox.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

#include <assert.h>
#include <math.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLCamera                                                            //
//                                                                      //
// Abstract base camera class - concrete classes for orthographic and   //
// persepctive cameras derive from it. This class maintains values for  //
// the current:                                                         //
// i)   Viewport                                                        //
// ii)  Projection, modelview and clip matricies - extracted from GL    //
// iii) The 6 frustum planes                                            //
// iv)  Expanded frustum interest box                                   //
//                                                                      //
// It provides methods for various projection, overlap and intersection //
// tests for viewport and world locations, against the true frustum and //
// expanded interest box, and for extracting eye position and direction.//
//                                                                      //
// It also defines the pure virtual manipulation interface methods the  //
// concrete ortho and prespective classes must implement.               //
//////////////////////////////////////////////////////////////////////////

class TGLCamera
{
public:
   enum EFrustumPlane
   {
      kNear             = 0,
      kLeft             = 1,
      kRight            = 2,
      kTop              = 3,
      kBottom           = 4,
      kFar              = 5,
      kPlanesPerFrustum = 6
   };

private:
   // Fields

   // Debuging visual aids
   TGLBoundingBox   fPreviousInterestBox;  //! previous interest box (DEBUG)
   TGLBoundingBox   fInterestFrustum;      //! frustum basis of current interest box - NOT a true BB! (DEBUG)
   TGLBoundingBox   fInterestFrustumAsBox; //! frustum basis (as box) of current interest box (DEBUG)

   static const Double_t fgInterestBoxExpansion; //! expansion c.f. aligned current frustum box

   // Methods
   TGLBoundingBox Frustum(Bool_t asBox = kTRUE) const; // current frustum

   // Non-copyable class
   TGLCamera(const TGLCamera &);
   TGLCamera & operator=(const TGLCamera &);

protected:
   // Fields

   // Internal cached matrices and frustum planes
   mutable Bool_t    fCacheDirty;                      //! cached items dirty?
   mutable TGLMatrix fProjM;                           //! projection matrix        (cached)
   mutable TGLMatrix fModVM;                           //! modelView matrix         (cached)
   mutable TGLMatrix fClipM;                           //! object space clip matrix (cached)
   mutable TGLPlane fFrustumPlanes[kPlanesPerFrustum]; //! frustum planes           (cached)

   TGLRect   fViewport;    //! viewport (GL coords - origin bottom left)

   TGLBoundingBox   fInterestBox;          //! the interest box - created in UpdateInterest()
   mutable Double_t fLargestSeen;          //! largest box volume seen in OfInterest() - used when
                                           // bootstrapping interest box

   // Methods
   Bool_t     AdjustAndClampVal(Double_t & val, Double_t min, Double_t max,
                                Int_t screenShift, Int_t screenShiftRange,
                                Bool_t mod1, Bool_t mod2) const;

   // Internal cache update - const as the actual camera configuration is unaffected
   void       UpdateCache() const;

public:
   TGLCamera();
   virtual ~TGLCamera();

   void SetViewport(const TGLRect & viewport);

   // Camera manipulation interface (GL coord - origin bottom left)
   virtual void   Setup(const TGLBoundingBox & box, Bool_t reset=kTRUE) = 0;
   virtual void   Reset() = 0;
   // virtual void   Frame(const TGLBoundingBox & box) = 0; // TODO
   // virtual void   Frame(const TGLRec & rect) = 0; // TODO
   virtual Bool_t Dolly(Int_t delta, Bool_t mod1, Bool_t mod2) = 0;
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2) = 0;
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta) = 0;
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta) = 0;
   virtual void   Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect = 0) const = 0;

   // Current orientation and frustum
         TGLVertex3 EyePoint() const;
         TGLVector3 EyeDirection() const;
         TGLVertex3 FrustumCenter() const;
   const TGLPlane & FrustumPlane(EFrustumPlane plane) const;

   // Overlap / projection / intersection tests
   // Viewport is GL coorinate system - origin bottom/left
   EOverlap   FrustumOverlap (const TGLBoundingBox & box) const; // box/frustum overlap test
   EOverlap   ViewportOverlap(const TGLBoundingBox & box) const; // box/viewport overlap test
   TGLRect    ViewportRect   (const TGLBoundingBox & box, TGLBoundingBox::EFace face) const;
   TGLRect    ViewportRect   (const TGLBoundingBox & box, const TGLBoundingBox::EFace * face = 0) const;
   TGLVertex3 WorldToViewport(const TGLVertex3 & worldVertex) const;
   TGLVector3 WorldDeltaToViewport(const TGLVertex3 & worldRef, const TGLVector3 & worldDelta) const;
   TGLVertex3 ViewportToWorld(const TGLVertex3 & viewportVertex) const;
   TGLLine3   ViewportToWorld(Double_t viewportX, Double_t viewportY) const;
   TGLLine3   ViewportToWorld(const TPoint & viewport) const;
   TGLVector3 ViewportDeltaToWorld(const TGLVertex3 & worldRef, Double_t viewportXDelta, Double_t viewportYDelta) const;
   std::pair<Bool_t, TGLVertex3> ViewportPlaneIntersection(Double_t viewportX, Double_t viewportY, const TGLPlane & worldPlane) const;
   std::pair<Bool_t, TGLVertex3> ViewportPlaneIntersection(const TPoint & viewport, const TGLPlane & worldPlane) const;

   // Window to GL viewport conversion - invert Y
   void WindowToViewport(Int_t & /* x */, Int_t & y) const { y = fViewport.Height() - y; }
   void WindowToViewport(TPoint & point)             const { point.SetY(fViewport.Height() - point.GetY()); }
   void WindowToViewport(TGLRect & rect)             const { rect.Y() = fViewport.Height() - rect.Y(); }
   void WindowToViewport(TGLVertex3 & vertex)        const { vertex.Y() = fViewport.Height() - vertex.Y(); }

   // Cameras expanded-frustum interest box
   Bool_t OfInterest(const TGLBoundingBox & box, Bool_t checkSize) const;
   Bool_t UpdateInterest(Bool_t force);
   void   ResetInterest();

   // Debuging - draw frustum and interest boxes
   void  DrawDebugAids() const;

   ClassDef(TGLCamera,0); // abstract camera base class
};

inline const TGLPlane & TGLCamera::FrustumPlane(EFrustumPlane plane) const
{
   // Return one of the planes forming the camera frustum
   if (fCacheDirty) {
      Error("TGLCamera::FrustumBox()", "cache dirty");
   }
   return fFrustumPlanes[plane];
}


#endif // ROOT_TGLCamera
