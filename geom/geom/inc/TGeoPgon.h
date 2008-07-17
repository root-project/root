// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPgon
#define ROOT_TGeoPgon

#ifndef ROOT_TGeoPcon
#include "TGeoPcon.h"
#endif

  
////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoPgon - a polygone. It has at least 10 parameters :                 //
//            - the lower phi limit;                                      //
//            - the range in phi;                                         //
//            - the number of edges on each z plane;                      //
//            - the number of z planes (at least two) where the inner/outer //
//              radii are changing;                                       //
//            - z coordinate, inner and outer radius for each z plane     //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoPgon : public TGeoPcon
{
protected:
   // data members
   Int_t                 fNedges;    // number of edges (at least one)
   
   Int_t                 GetPhiCrossList(Double_t *point, Double_t *dir, Int_t istart, Double_t *sphi, Int_t *iphi, Double_t stepmax=TGeoShape::Big()) const;
   Bool_t                IsCrossingSlice(Double_t *point, Double_t *dir, Int_t iphi, Double_t sstart, Int_t &ipl, Double_t &snext, Double_t stepmax) const;
   void                  LocatePhi(Double_t *point, Int_t &ipsec) const;
   Double_t              Rpg(Double_t z, Int_t ipl, Bool_t inner, Double_t &a, Double_t &b) const;
   Double_t              Rproj(Double_t z,Double_t *point, Double_t *dir, Double_t cphi, Double_t sphi, Double_t &a, Double_t &b) const; 
   Bool_t                SliceCrossing(Double_t *point, Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi, Double_t &snext, Double_t stepmax) const;
   Bool_t                SliceCrossingIn(Double_t *point, Double_t *dir, Int_t ipl, Int_t nphi, Int_t *iphi, Double_t *sphi, Double_t &snext, Double_t stepmax) const;
   Bool_t                SliceCrossingZ(Double_t *point, Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi, Double_t &snext, Double_t stepmax) const;
   Bool_t                SliceCrossingInZ(Double_t *point, Double_t *dir, Int_t nphi, Int_t *iphi, Double_t *sphi, Double_t &snext, Double_t stepmax) const;

public:
   // constructors
   TGeoPgon();
   TGeoPgon(Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoPgon(const char *name, Double_t phi, Double_t dphi, Int_t nedges, Int_t nz);
   TGeoPgon(Double_t *params);
   // destructor
   virtual ~TGeoPgon();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 64+12*fNz;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   Int_t                 GetNedges() const   {return fNedges;}
   virtual Int_t         GetNmeshVertices() const;
   virtual Int_t         GetNsegments() const {return fNedges;}     
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const {return TGeoBBox::GetPointsOnSegments(npoints,array);}
   virtual void          InspectShape() const;
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   Double_t              SafetyToSegment(Double_t *point, Int_t ipl, Int_t iphi, Bool_t in, Double_t safphi, Double_t safmin=TGeoShape::Big()) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t *param);
   void                  SetNedges(Int_t ne) {if (ne>2) fNedges=ne;}
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoPgon, 1)         // polygone class 
};

#endif
