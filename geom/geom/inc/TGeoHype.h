// @(#)root/geom:$Id$
// Author: Mihaela Gheata   20/11/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoHype
#define ROOT_TGeoHype

#include "TGeoTube.h"

class TGeoHype : public TGeoTube
{
protected :
// data members inherited from TGeoTube:
//   Double_t              fRmin; // inner radius at z=0
//   Double_t              fRmax; // outer radius at z=0
//   Double_t              fDz;   // half length
   Double_t              fStIn;   // Stereo angle for inner surface
   Double_t              fStOut;  // Stereo angle for inner surface

private :
// Precomputed parameters:
   Double_t              fTin;    // Tangent of stereo angle for inner surface
   Double_t              fTout;   // Tangent of stereo angle for outer surface
   Double_t              fTinsq;  // Squared tangent of stereo angle for inner surface
   Double_t              fToutsq; // Squared tangent of stereo angle for outer surface

   TGeoHype(const TGeoHype&) = delete;
   TGeoHype& operator=(const TGeoHype&) = delete;

public:
   // constructors
   TGeoHype();
   TGeoHype(Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz);
   TGeoHype(const char *name, Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz);
   TGeoHype(Double_t *params);
   // destructor
   virtual ~TGeoHype();
   // methods

   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   Int_t                 DistToHype(const Double_t *point, const Double_t *dir, Double_t *s, Bool_t inner, Bool_t in) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 64;}
   virtual Bool_t        GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const {return kFALSE;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   Double_t              GetStIn() const {return fStIn;}
   Double_t              GetStOut() const {return fStOut;}
   Bool_t                HasInner() const {return !TestShapeBit(kGeoRSeg);}
   Double_t              RadiusHypeSq(Double_t z, Bool_t inner) const;
   Double_t              ZHypeSq(Double_t r, Bool_t inner) const;
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   //virtual void          Paint(Option_t *option);
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   Double_t              SafetyToHype(const Double_t *point, Bool_t inner, Bool_t in) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetHypeDimensions(Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoHype, 1)         // hyperboloid class

};

#endif
