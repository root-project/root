// @(#)root/gl:$Name:  $:$Id: TGLCamera.h,v 1.9 2005/06/21 16:54:17 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original by Timur Pocheptsov

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

#include <assert.h>
#include <math.h>

/*************************************************************************
 * TGLCamera - TODO
 *
 *
 *
 *************************************************************************/
class TGLCamera
{
private:
   // Fields
   enum
   {
      kNEAR    = 0,
      kLEFT    = 1,
      kRIGHT   = 2,
      kTOP     = 3,
      kBOTTOM  = 4,
      kFAR     = 5,
      kPlanesPerFrustum
   };

   // Frustum planes (cached)
   mutable TGLPlane fFrustumPlanes[kPlanesPerFrustum]; //!

   // Debuging visual aids
   TGLBoundingBox   fPreviousInterestBox;  //! previous interest box (DEBUG)
   TGLBoundingBox   fInterestFrustum;      //! frustum basis of current interest box - NOT a true BB! (DEBUG)
   TGLBoundingBox   fInterestFrustumAsBox; //! frustum basis (as box) of current interest box (DEBUG)

   
   static const Double_t fInterestBoxExpansion; //! expansion c.f. aligned current frustum box

   // Methods
   TGLBoundingBox Frustum(Bool_t asBox = kTRUE) const; // current frustum

   // Non-copyable class
   TGLCamera(const TGLCamera &);
   TGLCamera & operator=(const TGLCamera &);

protected:
   // Fields
   TGLRect   fViewport;    //! viewport (GL coords - origin bottom left)
   TGLMatrix fProjM;       //! projection matrix        (cached)
   TGLMatrix fModVM;       //! modelView matrix         (cached)
   TGLMatrix fClipM;       //! object space clip matrix (cached)
   Bool_t    fCacheDirty;  //! cached items dirty?

   TGLBoundingBox   fInterestBox;          //! the interest box - created in UpdateInterest()
   mutable Double_t fLargestInterest;      //! largest box diagonal - used when bootstrapping interest box

   // Methods
   Bool_t     AdjustAndClampVal(Double_t & val, Double_t min, Double_t max,
                                Int_t screenShift, Int_t screenShiftRange, 
                                Bool_t mod1, Bool_t mod2) const;
   void       UpdateCache();

public:
   TGLCamera();
   virtual ~TGLCamera();

   void SetViewport(const TGLRect & viewport);

   // Camera manipulation interface (GL coord - origin bottom left)
   virtual void   Setup(const TGLBoundingBox & box) = 0;
   virtual void   Reset() = 0;
   // virtual void   Frame(const TGLBoundingBox & box) = 0; // TODO
   // virtual void   Frame(const TGLRec & rect) = 0; // TODO
   virtual Bool_t Dolly(Int_t delta, Bool_t mod1, Bool_t mod2) = 0;
   virtual Bool_t Zoom (Int_t delta, Bool_t mod1, Bool_t mod2) = 0;
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta) = 0;
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta) = 0;
   virtual void   Apply(const TGLBoundingBox & sceneBox, const TGLRect * pickRect = 0) = 0;

   // Current orientation 
   TGLVertex3 EyePoint() const;
   TGLVector3 EyeDirection() const;

   // Projection and overlap tests
   EOverlap   FrustumOverlap (const TGLBoundingBox & box) const; // box/frustum overlap test
   EOverlap   ViewportOverlap(const TGLBoundingBox & box) const; // box/viewport overlap test
   TGLRect    ViewportSize   (const TGLBoundingBox & box) const; // project size of box on viewport
   TGLVector3 ProjectedShift (const TGLVertex3 & vertex, Int_t xDelta, Int_t yDelta) const;

   // Cameras expanded-frustum interest box
   Bool_t OfInterest(const TGLBoundingBox & box) const;
   Bool_t UpdateInterest(Bool_t force);
   void   ResetInterest();

   // Debuging - draw frustum and interest boxes
   void  DrawDebugAids() const;

   ClassDef(TGLCamera,0); // abstract camera base class
};

#endif // ROOT_TGLCamera
