/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST
// TGeoShape::Contains implemented by Mihaela Gheata

#ifndef ROOT_TGeoArb8
#define ROOT_TGeoArb8

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif


/*************************************************************************
 * TGeoArb8 - a arbitrary trapezoid with less than 8 vertices standing on
 *   two paralel planes perpendicular to Z axis. Parameters :
 *            - dz - half length in Z;
 *            - xy[8][2] - vector of (x,y) coordinates of vertices
 *               - first four points (xy[i][j], i<4, j<2) are the (x,y)
 *                 coordinates of the vertices sitting on the -dz plane;
 *               - last four points (xy[i][j], i>=4, j<2) are the (x,y)
 *                 coordinates of the vertices sitting on the +dz plane;
 *   The order of defining the vertices of an arb8 is the following :
 *      - point 0 is connected with points 1,3,4
 *      - point 1 is connected with points 0,2,5
 *      - point 2 is connected with points 1,3,6
 *      - point 3 is connected with points 0,2,7
 *      - point 4 is connected with points 0,5,7
 *      - point 5 is connected with points 1,4,6
 *      - point 6 is connected with points 2,5,7
 *      - point 7 is connected with points 3,4,6
 *   Points can be identical in order to create shapes with less than 
 *   8 vertices.
 *
 *************************************************************************/

class TGeoArb8 : public TGeoBBox
{
protected:
   enum EGeoArb8Type {
//      kArb8Trd1 = BIT(25), // trd1 type
//      kArb8Trd2 = BIT(26), // trd2 type
      kArb8Trap = BIT(27), // planar surface trapezoid
      kArb8Tra  = BIT(28), // general twisted trapezoid
   };
   // data members
   Double_t              fDz;          // half length in Z
   Double_t             *fTwist;       //[4] tangents of twist angles 
   Double_t              fXY[8][2];    // list of vertices
public:
   // constructors
   TGeoArb8();
   TGeoArb8(Double_t dz, Double_t *vertices=0);
   // destructor
   virtual ~TGeoArb8();
   // methods
   virtual void          ComputeBBox();
   void                  ComputeTwist();
   virtual Int_t         GetByteCount() {return 100;}
   Double_t              GetDz() {return fDz;}
   Double_t             *GetVertices() {return &fXY[0][0];}
   Bool_t                IsTwisted() {return (fTwist==0)?kFALSE:kTRUE;}
   void                  SetPlaneVertices(Double_t zpl, Double_t *vertices);
   virtual void          SetVertex(Int_t vnum, Double_t x, Double_t y);
   
   virtual Bool_t        Contains(Double_t *point);     
   Double_t              DistToPlane(Double_t *point, Double_t *dir, Int_t ipl);
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir);
   virtual void          Draw(Option_t *option);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const {return 0;}
   virtual void          InspectShape();
   virtual void          Paint(Option_t *option);
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point);
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option);
   virtual void          SetDimensions(Double_t *param);
   virtual void          SetPoints(Double_t *buff) const;
   virtual void          SetPoints(Float_t *buff) const;
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
   // destructor
   virtual ~TGeoTrap();
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   Double_t              GetTheta() {return fTheta;}
   Double_t              GetPhi()   {return fPhi;}
   Double_t              GetH1()    {return fH1;}
   Double_t              GetBl1()   {return fBl1;}
   Double_t              GetTl1()   {return fTl1;}
   Double_t              GetAlpha1()   {return fAlpha1;}
   Double_t              GetH2()    {return fH2;}
   Double_t              GetBl2()   {return fBl2;}
   Double_t              GetTl2()   {return fTl2;}
   Double_t              GetAlpha2()   {return fAlpha2;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;

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
   // destructor
   virtual ~TGeoGtra();
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const;
   Double_t              GetTwistAngle() {return fTwistAngle;}
  ClassDef(TGeoGtra, 1)         // G3 GTRA shape
};

#endif
