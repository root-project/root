/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata - Wed 24 Oct 2001 05:20:43 PM CEST

#ifndef ROOT_TGeoShape
#define ROOT_TGeoShape

#ifndef ROOT_TObject
#include "TObject.h"
#endif


// forward declarations
class TGeoParamCurve : public TObject
{
public:
   TGeoParamCurve() {;}
   virtual ~TGeoParamCurve() {}

   ClassDef(TGeoParamCurve, 1)
};

class TGeoBoolCombinator;
class TGeoMatrix;

/*************************************************************************
 * TGeoShape - base class for geometric shapes. Provides virtual methods
 *   for point and segment classification that has to be implemented by 
 *   all classes inheriting from it.
 *
 *************************************************************************/

class TGeoShape : public TObject
{
public:
enum EShapeType {
   kGeoNoShape = 0,
   kGeoVisX    = BIT(9),
   kGeoVisY    = BIT(10),
   kGeoVisZ    = BIT(11),
   kGeoRunTimeShape = BIT(12),
   kGeoInvalidShape = BIT(13),
   kGeoBox     = BIT(15),
   kGeoPara    = BIT(16),
   kGeoSph     = BIT(17),
   kGeoTube    = BIT(18),
   kGeoTubeSeg = BIT(19), 
   kGeoCone    = BIT(20),
   kGeoConeSeg = BIT(21),
   kGeoPcon    = BIT(22),
   kGeoPgon    = BIT(23),
   kGeoArb8    = BIT(24),
   kGeoEltu    = BIT(25),
   kGeoTrap    = BIT(26),
   kGeoCtub    = BIT(27),
   kGeoTrd1    = BIT(28),
   kGeoTrd2    = BIT(29),
   kGeoComb    = BIT(30)
};
static const Double_t kRadDeg;   // conversion factor rad->deg
static const Double_t kDegRad;   // conversion factor deg->rad
static const Double_t kBig;      // infinity
protected :
// data members
   Int_t                fShapeId;   // shape id
// methods

public:
   // constructors
   TGeoShape();
   // destructor
   virtual ~TGeoShape();
   // methods
   Int_t                 GetId()    {return fShapeId;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother) const  = 0;
   virtual const char   *GetName() const;
   virtual Int_t         GetByteCount()                          = 0;
   void                  SetId(Int_t id) {fShapeId = id;}

   static Double_t       ClosenessToCorner(Double_t *point, Bool_t in, Double_t *vertex,
                                           Double_t *normals, Double_t *cldir);
   virtual void          ComputeBBox()                           = 0;
   virtual Bool_t        Contains(Double_t *point)               = 0;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py) = 0;
   static Double_t       DistToCorner(Double_t *point, Double_t *dir, Bool_t in,
                                      Double_t *vertex, Double_t *norm, Int_t &inorm); 
   virtual Double_t      DistToOut(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) = 0;
   virtual Double_t      DistToIn(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=0, Double_t *safe=0) = 0;
   virtual Double_t      DistToSurf(Double_t *point, Double_t *dir) = 0;
   virtual void          Draw(Option_t *option)                  = 0;
   static Int_t          GetVertexNumber(Bool_t vx, Bool_t vy, Bool_t vz);
   Bool_t                IsRunTimeShape() const {return TestBit(kGeoRunTimeShape);}
   Bool_t                IsValid() const {return !TestBit(kGeoInvalidShape);}
   virtual void          InspectShape()                          = 0;
   virtual void          Paint(Option_t *option)                 = 0;
   virtual void          NextCrossing(TGeoParamCurve *c, Double_t *point) = 0;
   virtual Double_t      Safety(Double_t *point, Double_t *spoint, Option_t *option) = 0;
   virtual void          Sizeof3D() const                        = 0;
   virtual void          SetDimensions(Double_t *param)          = 0;
   virtual void          SetPoints(Double_t *buff) const         = 0;
   virtual void          SetPoints(Float_t *buff) const          = 0;
   void                  SetRuntime(Bool_t flag=kTRUE) {SetBit(kGeoRunTimeShape, flag);}
   Int_t                 ShapeDistancetoPrimitive(Int_t numpoints, Int_t px, Int_t py);
   
  ClassDef(TGeoShape, 0)           // base class for shapes
};

#endif

