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

#ifndef ROOT_TGeoTube
#include "TGeoTube.h"
#endif

///////////////////////////////////////////////////////////////////////////////
//                                                                        
// TGeoHype - Hyperboloid class defined by 5 parameters. Bounded by:
//            - Two z planes at z=+/-dz
//            - Inner and outer lateral surfaces. These represent the surfaces 
//              described by the revolution of 2 hyperbolas about the Z axis:
//               r^2 - (t*z)^2 = a^2
//
//            r = distance between hyperbola and Z axis at coordinate z
//            t = tangent of the stereo angle (angle made by hyperbola
//                asimptotic lines and Z axis). t=0 means cylindrical surface.
//            a = distance between hyperbola and Z axis at z=0
//
//          The inner hyperbolic surface is described by:
//              r^2 - (tin*z)^2 = rin^2 
//           - absence of the inner surface (filled hyperboloid can be forced 
//             by rin=0 and sin=0
//          The outer hyperbolic surface is described by:
//              r^2 - (tout*z)^2 = rout^2
//  TGeoHype parameters: dz[cm], rin[cm], sin[deg], rout[cm], sout[deg].
//    MANDATORY conditions:
//           - rin < rout
//           - rout > 0
//           - rin^2 + (tin*dz)^2 > rout^2 + (tout*dz)^2
//                                                                        
///////////////////////////////////////////////////////////////////////////////


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
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   Int_t                 DistToHype(Double_t *point, Double_t *dir, Double_t *s, Bool_t inner) const;
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
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   Double_t              SafetyToHype(Double_t *point, Bool_t inner, Bool_t in) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   void                  SetHypeDimensions(Double_t rin, Double_t stin, Double_t rout, Double_t stout, Double_t dz);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoHype, 1)         // hyperboloid class

};

#endif
