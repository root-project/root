// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoShape
#define ROOT_TGeoShape

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

// forward declarations
class TGeoBoolCombinator;
class TGeoBBox;
class TGeoMatrix;
class TGeoHMatrix;
class TGeoVolume;
class TBuffer3D;

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoShape - base class for geometric shapes. Provides virtual methods  //
//   for point and segment classification that has to be implemented by   //
//   all classes inheriting from it.                                      //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoShape : public TNamed
{
private:
   static TGeoMatrix     *fgTransform;  // current transformation matrix that applies to shape
   static Double_t        fgEpsMch;     // Machine round-off error
public:
enum EShapeType {
   kBitMask32  = 0xffffffff,
   kGeoNoShape = 0,
   kGeoBad     = BIT(0),
   kGeoRSeg    = BIT(1),
   kGeoPhiSeg  = BIT(2),
   kGeoThetaSeg = BIT(3),
   kGeoVisX    = BIT(4),
   kGeoVisY    = BIT(5),
   kGeoVisZ    = BIT(6),
   kGeoRunTimeShape = BIT(7),
   kGeoInvalidShape = BIT(8),
   kGeoTorus   = BIT(9),
   kGeoBox     = BIT(10),
   kGeoPara    = BIT(11),
   kGeoSph     = BIT(12),
   kGeoTube    = BIT(13),
   kGeoTubeSeg = BIT(14), 
   kGeoCone    = BIT(15),
   kGeoConeSeg = BIT(16),
   kGeoPcon    = BIT(17),
   kGeoPgon    = BIT(18),
   kGeoArb8    = BIT(19),
   kGeoEltu    = BIT(20),
   kGeoTrap    = BIT(21),
   kGeoCtub    = BIT(22),
   kGeoTrd1    = BIT(23),
   kGeoTrd2    = BIT(24),
   kGeoComb    = BIT(25),
   kGeoClosedShape = BIT(26),
   kGeoXtru    = BIT(27),
   kGeoParaboloid = BIT(28),
   kGeoHalfSpace  = BIT(29),
   kGeoHype    = BIT(30),
   kGeoSavePrimitive = BIT(20)
};

protected :
// data members
   Int_t                 fShapeId;   // shape id
   UInt_t                fShapeBits; // shape bits
// methods
   virtual void          FillBuffer3D(TBuffer3D & buffer, Int_t reqSections, Bool_t localFrame) const;
   Int_t                 GetBasicColor() const;
   void                  SetOnBoundary(Bool_t /*flag=kTRUE*/) {;}
   void                  TransformPoints(Double_t *points, UInt_t NbPoints) const;

public:
   // constructors
   TGeoShape();
   TGeoShape(const char *name);
   // destructor
   virtual ~TGeoShape();
   // methods

   static Double_t       Big() {return 1.E30;}
   static TGeoMatrix    *GetTransform();
   static void           SetTransform(TGeoMatrix *matrix);
   static Double_t       Tolerance() {return 1.E-10;}
   static Double_t       ComputeEpsMch();
   static Double_t       EpsMch();
   virtual Double_t      Capacity() const                        = 0;
   void                  CheckShape(Int_t testNo, Int_t nsamples=10000, Option_t *option="");
   virtual void          ComputeBBox()                           = 0;
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm) = 0;
   virtual Bool_t        Contains(Double_t *point) const         = 0;
   virtual Bool_t        CouldBeCrossed(Double_t *point, Double_t *dir) const = 0;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py) = 0;
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const = 0;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const = 0;
   static Double_t       DistToPhiMin(Double_t *point, Double_t *dir, Double_t s1, Double_t c1, Double_t s2, Double_t c2, 
                                      Double_t sm, Double_t cm, Bool_t in=kTRUE);
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step)   = 0; 
   virtual void          Draw(Option_t *option=""); // *MENU*
   virtual void          ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual const char   *GetAxisName(Int_t iaxis) const = 0;
   virtual Double_t      GetAxisRange(Int_t iaxis, Double_t &xlo, Double_t &xhi) const = 0;
   virtual void          GetBoundingCylinder(Double_t *param) const = 0;
   virtual const TBuffer3D &GetBuffer3D(Int_t reqSections, Bool_t localFrame) const;
   virtual Int_t         GetByteCount() const                          = 0;
   virtual Bool_t        GetPointsOnSegments(Int_t npoints, Double_t *array) const = 0;
   virtual Int_t         GetFittingBox(const TGeoBBox *parambox, TGeoMatrix *mat, Double_t &dx, Double_t &dy, Double_t &dz) const = 0;
   Int_t                 GetId() const  {return fShapeId;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const  = 0;
   virtual void          GetMeshNumbers(Int_t &/*nvert*/, Int_t &/*nsegs*/, Int_t &/*npols*/) const {;}
   virtual const char   *GetName() const;
   virtual Int_t         GetNmeshVertices() const {return 0;}
   const char           *GetPointerName() const;
   virtual Bool_t        IsAssembly() const {return kFALSE;}
   virtual Bool_t        IsComposite() const {return kFALSE;}
   virtual Bool_t        IsCylType() const = 0;
   static  Bool_t        IsCloseToPhi(Double_t epsil, Double_t *point, Double_t c1, Double_t s1, Double_t c2, Double_t s2);
   static  Bool_t        IsCrossingSemiplane(Double_t *point, Double_t *dir, Double_t cphi, Double_t sphi, Double_t &snext, Double_t &rxy);
   static  Bool_t        IsSameWithinTolerance(Double_t a, Double_t b);
   static  Bool_t        IsSegCrossing(Double_t x1, Double_t y1, Double_t x2, Double_t y2,Double_t x3, Double_t y3,Double_t x4, Double_t y4);
   static  Bool_t        IsInPhiRange(Double_t *point, Double_t phi1, Double_t phi2);
   virtual Bool_t        IsReflected() const {return kFALSE;}
   Bool_t                IsRunTimeShape() const {return TestShapeBit(kGeoRunTimeShape);}
   Bool_t                IsValid() const {return !TestShapeBit(kGeoInvalidShape);}
   virtual Bool_t        IsValidBox() const                      = 0; 
   virtual void          InspectShape() const                    = 0;
   virtual TBuffer3D    *MakeBuffer3D() const {return 0;}
   static void           NormalPhi(Double_t *point, Double_t *dir, Double_t *norm, Double_t c1, Double_t s1, Double_t c2, Double_t s2);
   virtual void          Paint(Option_t *option="");
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const = 0;
   static  Double_t      SafetyPhi(Double_t *point, Bool_t in, Double_t phi1, Double_t phi2);
   virtual void          SetDimensions(Double_t *param)          = 0;
   void                  SetId(Int_t id) {fShapeId = id;}
   virtual void          SetPoints(Double_t *points) const         = 0;
   virtual void          SetPoints(Float_t *points) const          = 0;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const   = 0;
   void                  SetRuntime(Bool_t flag=kTRUE) {SetShapeBit(kGeoRunTimeShape, flag);}
   Int_t                 ShapeDistancetoPrimitive(Int_t numpoints, Int_t px, Int_t py) const;
   virtual void          Sizeof3D() const                        = 0;

   //----- bit manipulation
   void     SetShapeBit(UInt_t f, Bool_t set);
   void     SetShapeBit(UInt_t f) { fShapeBits |= f & kBitMask32; }
   void     ResetShapeBit(UInt_t f) { fShapeBits &= ~(f & kBitMask32); }
   Bool_t   TestShapeBit(UInt_t f) const { return (Bool_t) ((fShapeBits & f) != 0); }
   Int_t    TestShapeBits(UInt_t f) const { return (Int_t) (fShapeBits & f); }
   void     InvertShapeBit(UInt_t f) { fShapeBits ^= f & kBitMask32; }
   
   ClassDef(TGeoShape, 2)           // base class for shapes
};

#endif

