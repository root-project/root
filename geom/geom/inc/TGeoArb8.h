// @(#)root/geom:$Id$
// Author: Andrei Gheata   24/10/01

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoArb8
#define ROOT_TGeoArb8

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif


////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoArb8 - a arbitrary trapezoid with less than 8 vertices standing on //
//   two paralel planes perpendicular to Z axis. Parameters :             //
//            - dz - half length in Z;                                    //
//            - xy[8][2] - vector of (x,y) coordinates of vertices        //
//               - first four points (xy[i][j], i<4, j<2) are the (x,y)   //
//                 coordinates of the vertices sitting on the -dz plane;  //
//               - last four points (xy[i][j], i>=4, j<2) are the (x,y)   //
//                 coordinates of the vertices sitting on the +dz plane;  //
//   The order of defining the vertices of an arb8 is the following :     //
//      - point 0 is connected with points 1,3,4                          //
//      - point 1 is connected with points 0,2,5                          //
//      - point 2 is connected with points 1,3,6                          //
//      - point 3 is connected with points 0,2,7                          //
//      - point 4 is connected with points 0,5,7                          //
//      - point 5 is connected with points 1,4,6                          //
//      - point 6 is connected with points 2,5,7                          //
//      - point 7 is connected with points 3,4,6                          //
//   Points can be identical in order to create shapes with less than     // 
//   8 vertices.                                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoArb8 : public TGeoBBox
{
protected:
   enum EGeoArb8Type {
//      kArb8Trd1 = BIT(25), // trd1 type
//      kArb8Trd2 = BIT(26), // trd2 type
      kArb8Trap = BIT(27), // planar surface trapezoid
      kArb8Tra  = BIT(28)  // general twisted trapezoid
   };
   // data members
   Double_t              fDz;          // half length in Z
   Double_t             *fTwist;       //! [4] tangents of twist angles 
   Double_t              fXY[8][2];    // list of vertices

   TGeoArb8(const TGeoArb8&); 
   TGeoArb8& operator=(const TGeoArb8&); 

public:
   // constructors
   TGeoArb8();
   TGeoArb8(Double_t dz, Double_t *vertices=0);
   TGeoArb8(const char *name, Double_t dz, Double_t *vertices=0);
   // destructor
   virtual ~TGeoArb8();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   void                  ComputeTwist();
   virtual Bool_t        Contains(Double_t *point) const;     
   Double_t              DistToPlane(Double_t *point, Double_t *dir, Int_t ipl, Bool_t in) const;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const;
   virtual void          GetBoundingCylinder(Double_t *param) const;
   virtual Int_t         GetByteCount() const {return 100;}
   Double_t              GetClosestEdge(Double_t *point, Double_t *vert, Int_t &isegment) const;
   virtual Bool_t        GetPointsOnFacet(Int_t /*index*/, Int_t /*npoints*/, Double_t * /*array*/) const;
   Double_t              GetDz() const {return fDz;}
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   static void           GetPlaneNormal(Double_t *p1, Double_t *p2, Double_t *p3, Double_t *norm);
   Double_t             *GetVertices() {return &fXY[0][0];}
   Double_t              GetTwist(Int_t iseg) const;
   virtual Bool_t        IsCylType() const {return kFALSE;}
   static Bool_t         IsSamePoint(Double_t *p1, Double_t *p2) {return (TMath::Abs(p1[0]-p2[0])<1.E-16 && TMath::Abs(p1[1]-p2[1])<1.E-16)?kTRUE:kFALSE;}
   static Bool_t         InsidePolygon(Double_t x, Double_t y, Double_t *pts);
   virtual void          InspectShape() const;
   Bool_t                IsTwisted() const {return (fTwist==0)?kFALSE:kTRUE;}
   Double_t              SafetyToFace(Double_t *point, Int_t iseg, Bool_t in) const;
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   void                  SetPlaneVertices(Double_t zpl, Double_t *vertices) const;
   virtual void          SetVertex(Int_t vnum, Double_t x, Double_t y);
   virtual void          SetDimensions(Double_t *param);
   void                  SetDz(Double_t dz) {fDz = dz;}
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoArb8, 1)         // arbitrary trapezoid with 8 vertices
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoTrap                                                               //
//                                                                        //
// Trap is a general trapezoid, i.e. one for which the faces perpendicular//
// to z are trapezia and their centres are not the same x, y. It has 11   //
// parameters: the half length in z, the polar angles from the centre of  //
// the face at low z to that at high z, H1 the half length in y at low z, //
// LB1 the half length in x at low z and y low edge, LB2 the half length  //
// in x at low z and y high edge, TH1 the angle w.r.t. the y axis from the//
// centre of low y edge to the centre of the high y edge, and H2, LB2,    //
// LH2, TH2, the corresponding quantities at high z.                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoTrap : public TGeoArb8
{
protected:
   // data members
   Double_t              fTheta; // theta angle
   Double_t              fPhi;   // phi angle
   Double_t              fH1;    // half length in y at low z
   Double_t              fBl1;   // half length in x at low z and y low edge
   Double_t              fTl1;   // half length in x at low z and y high edge
   Double_t              fAlpha1;// angle between centers of x edges an y axis at low z
   Double_t              fH2;    // half length in y at high z
   Double_t              fBl2;   // half length in x at high z and y low edge
   Double_t              fTl2;   // half length in x at high z and y high edge
   Double_t              fAlpha2;// angle between centers of x edges an y axis at low z

public:
   // constructors
   TGeoTrap();
   TGeoTrap(Double_t dz, Double_t theta, Double_t phi);
   TGeoTrap(Double_t dz, Double_t theta, Double_t phi, Double_t h1,
            Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
            Double_t tl2, Double_t alpha2);
   TGeoTrap(const char *name, Double_t dz, Double_t theta, Double_t phi, Double_t h1,
            Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
            Double_t tl2, Double_t alpha2);
   // destructor
   virtual ~TGeoTrap();
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   Double_t              GetTheta() const {return fTheta;}
   Double_t              GetPhi() const   {return fPhi;}
   Double_t              GetH1() const    {return fH1;}
   Double_t              GetBl1() const   {return fBl1;}
   Double_t              GetTl1() const   {return fTl1;}
   Double_t              GetAlpha1() const   {return fAlpha1;}
   Double_t              GetH2() const    {return fH2;}
   Double_t              GetBl2() const   {return fBl2;}
   Double_t              GetTl2() const   {return fTl2;}
   Double_t              GetAlpha2() const   {return fAlpha2;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          SetDimensions(Double_t *param);
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGeoTrap, 1)         // G3 TRAP shape
};

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoGtra                                                               //
//                                                                        //
// Gtra is a twisted general trapezoid, i.e. one for which the faces perpendicular//
// to z are trapezia and their centres are not the same x, y. It has 12   //
// parameters: the half length in z, the polar angles from the centre of  //
// the face at low z to that at high z, the twist angle, H1 the half length in y at low z, //
// LB1 the half length in x at low z and y low edge, LB2 the half length  //
// in x at low z and y high edge, TH1 the angle w.r.t. the y axis from the//
// centre of low y edge to the centre of the high y edge, and H2, LB2,    //
// LH2, TH2, the corresponding quantities at high z.                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoGtra : public TGeoTrap
{
protected:
   // data members
   Double_t          fTwistAngle; // twist angle in degrees
public:
   // constructors
   TGeoGtra();
   TGeoGtra(Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
            Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
            Double_t tl2, Double_t alpha2);
   TGeoGtra(const char *name, Double_t dz, Double_t theta, Double_t phi, Double_t twist, Double_t h1,
            Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2, 
            Double_t tl2, Double_t alpha2);
   // destructor
   virtual ~TGeoGtra();
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   Double_t              GetTwistAngle() const {return fTwistAngle;}
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SetDimensions(Double_t *param);
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
 
   ClassDef(TGeoGtra, 1)         // G3 GTRA shape
};

#endif
