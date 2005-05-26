// @(#)root/gl:$Name:$:$Id:$
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
      kPLANESPERFRUSTUM
   };

   // Frustum planes (cached)
   mutable TGLPlane fFrustumPlanes[kPLANESPERFRUSTUM]; //!

   // Methods
   TGLBoundingBox FrustumBox() const; // bounding box encapsulating frustum

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

   TGLBoundingBox   fInterestBox; //!
   mutable Double_t fLargestInterest;

   // Methods
   Bool_t AdjustAndClampVal(Double_t & val, Double_t min, Double_t max,
                            Int_t shift, Int_t shiftRange) const;
   void UpdateCache();

public:
   TGLCamera();
   virtual ~TGLCamera();

   void SetViewport(const TGLRect & viewport);

   // Camera manipulation interface (GL coord - origin bottom left)
   virtual void   Setup(const TGLBoundingBox & box) = 0;
   virtual void   Reset() = 0;
   // virtual void   Frame(const TGLBoundingBox & box) = 0; // TODO
   virtual Bool_t Dolly(Int_t delta) = 0;
   virtual Bool_t Zoom (Int_t delta) = 0;
   virtual Bool_t Truck(Int_t x, Int_t y, Int_t xDelta, Int_t yDelta) = 0;
   virtual Bool_t Rotate(Int_t xDelta, Int_t yDelta) = 0;
   virtual void   Apply(const TGLBoundingBox & box, const TGLRect * pickRect = 0) = 0;

   EOverlap FrustumOverlap (const TGLBoundingBox & box) const; // box/frustum overlap test
   EOverlap ViewportOverlap(const TGLBoundingBox & box) const; // box/viewport overlap test
   TGLRect  ViewportSize   (const TGLBoundingBox & box) const; // project size of box on viewport
   //Double_t NearVertexDistance(const TGLBoundingBox & box) const;
   //Double_t FarVertexDistance(const TGLBoundingBox & box) const;

   Bool_t OfInterest(const TGLBoundingBox & box) const;
   Bool_t UpdateInterest();
   void   ResetInterest();

   ClassDef(TGLCamera,0); // abstract camera base class
};

#endif // ROOT_TGLCamera
