// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoPcon
#define ROOT_TGeoPcon

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

  
//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TGeoPcon - a composite polycone. It has at least 9 parameters :          //
//            - the lower phi limit;                                        //
//            - the range in phi;                                           //
//            - the number of z planes (at least two) where the inner/outer //
//              radii are changing;                                         //
//            - z coordinate, inner and outer radius for each z plane       //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

class TGeoPcon : public TGeoBBox
{
protected:
   // data members
   Int_t                 fNz;    // number of z planes (at least two)
   Double_t              fPhi1;  // lower phi limit 
   Double_t              fDphi;  // phi range
   Double_t             *fRmin;  //[fNz] pointer to array of inner radii 
   Double_t             *fRmax;  //[fNz] pointer to array of outer radii 
   Double_t             *fZ;     //[fNz] pointer to array of Z planes positions 
   
   // methods
   TGeoPcon(const TGeoPcon&); 
   TGeoPcon& operator=(const TGeoPcon&); 

public:
   // constructors
   TGeoPcon();
   TGeoPcon(Double_t phi, Double_t dphi, Int_t nz);
   TGeoPcon(const char *name, Double_t phi, Double_t dphi, Int_t nz);
   TGeoPcon(Double_t *params);
   // destructor
   virtual ~TGeoPcon();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual void          DefineSection(Int_t snum, Double_t z, Double_t rmin, Double_t rmax);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   Double_t              DistToSegZ(Double_t *point, Double_t *dir, Int_t &iz, Double_t c1, Double_t s1,
                                    Double_t c2, Double_t s2, Double_t cfio, Double_t sfio, Double_t cdfi) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 60+12*fNz;}
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetDphi() const {return fDphi;}
   Int_t                 GetNz() const   {return fNz;}
   virtual Int_t         GetNsegments() const;
   Double_t             *GetRmin() const {return fRmin;}
   Double_t              GetRmin(Int_t ipl) const;
   Double_t             *GetRmax() const {return fRmax;}
   Double_t              GetRmax(Int_t ipl) const;
   Double_t             *GetZ() const    {return fZ;}
   Double_t              GetZ(Int_t ipl) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual Int_t         GetNmeshVertices() const;
   virtual Bool_t        GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const {return kFALSE;}
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kTRUE;}
   virtual TBuffer3D    *MakeBuffer3D() const;
   Double_t             &Phi1()          {return fPhi1;}
   Double_t             &Dphi()          {return fDphi;}
   Double_t             &Rmin(Int_t ipl) {return fRmin[ipl];}
   Double_t             &Rmax(Int_t ipl) {return fRmax[ipl];}
   Double_t             &Z(Int_t ipl) {return fZ[ipl];}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   Double_t              SafetyToSegment(Double_t *point, Int_t ipl, Bool_t in=kTRUE, Double_t safmin=TGeoShape::Big()) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoPcon, 1)         // polycone class 
};

#endif
