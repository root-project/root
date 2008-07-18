// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoSphere
#define ROOT_TGeoSphere

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoSphere - spherical shell class. It takes 6 parameters :            //
//           - inner and outer radius Rmin, Rmax                          //
//           - the theta limits Tmin, Tmax                                //
//           - the phi limits Pmin, Pmax (the sector in phi is considered //
//             starting from Pmin to Pmax counter-clockwise               //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoSphere : public TGeoBBox
{
protected :
// data members
   Int_t                 fNz;     // number of z planes for drawing
   Int_t                 fNseg;   // number of segments for drawing
   Double_t              fRmin;   // inner radius
   Double_t              fRmax;   // outer radius
   Double_t              fTheta1; // lower theta limit
   Double_t              fTheta2; // higher theta limit
   Double_t              fPhi1;   // lower phi limit
   Double_t              fPhi2;   // higher phi limit
// methods

public:
   // constructors
   TGeoSphere();
   TGeoSphere(Double_t rmin, Double_t rmax, Double_t theta1=0, Double_t theta2=180,
              Double_t phi1=0, Double_t phi2=360);
   TGeoSphere(const char *name, Double_t rmin, Double_t rmax, Double_t theta1=0, Double_t theta2=180,
              Double_t phi1=0, Double_t phi2=360);
   TGeoSphere(Double_t *param, Int_t nparam=6);
   // destructor
   virtual ~TGeoSphere();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   Double_t              DistToSphere(Double_t *point, Double_t *dir, Double_t rsph, Bool_t check=kTRUE, Bool_t firstcross=kTRUE) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual const char   *GetAxisName(Int_t iaxis) const;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const {return 42;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   Int_t                 GetNumberOfDivisions() const {return fNseg;}
   virtual Bool_t        GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const {return kFALSE;}
   Int_t                 GetNz() const   {return fNz;}
   virtual Double_t      GetRmin() const {return fRmin;}
   virtual Double_t      GetRmax() const {return fRmax;}
   Double_t              GetTheta1() const {return fTheta1;}
   Double_t              GetTheta2() const {return fTheta2;}
   Double_t              GetPhi1() const {return fPhi1;}
   Double_t              GetPhi2() const {return fPhi2;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   Int_t                 IsOnBoundary(Double_t *point) const;
   Bool_t                IsPointInside(Double_t *point, Bool_t checkR=kTRUE, Bool_t checkTh=kTRUE, Bool_t checkPh=kTRUE) const;
   virtual TBuffer3D    *MakeBuffer3D() const;
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   void                  SetSphDimensions(Double_t rmin, Double_t rmax, Double_t theta1,
                                       Double_t theta2, Double_t phi1, Double_t phi2);
   virtual void          SetNumberOfDivisions(Int_t p);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoSphere, 1)         // sphere class
};

#endif
