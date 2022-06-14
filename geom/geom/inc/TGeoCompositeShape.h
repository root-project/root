// @(#)root/geom:$Id$
// Author: Andrei Gheata   31/01/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoCompositeShape
#define ROOT_TGeoCompositeShape

#include "TGeoBBox.h"

/////////////////////////////////////////////////////////////////////////////
//                                                                         //
// TGeoCompositeShape - composite shape class. A composite shape contains  //
//   a list of primitive shapes, the list of corresponding transformations //
//   and a boolean finder handling boolean operations among components.    //
//                                                                         //
/////////////////////////////////////////////////////////////////////////////

class TGeoBoolNode;

class TGeoCompositeShape : public TGeoBBox
{
private :
// data members
   TGeoBoolNode         *fNode;             // top boolean node

protected:
   TGeoCompositeShape(const TGeoCompositeShape &) = delete;
   TGeoCompositeShape& operator=(const TGeoCompositeShape &) = delete;

public:
   // constructors
   TGeoCompositeShape();
   TGeoCompositeShape(const char *name, const char *expression);
   TGeoCompositeShape(const char *expression);
   TGeoCompositeShape(const char *name, TGeoBoolNode *node);
   // destructor
   virtual ~TGeoCompositeShape();
   // methods
   virtual Double_t      Capacity() const;
   virtual void          ClearThreadData() const;
   virtual void          CreateThreadData(Int_t nthreads);
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(const Double_t *point, const Double_t *dir, Double_t *norm);
   virtual void          ComputeNormal_v(const Double_t *points, const Double_t *dirs, Double_t *norms, Int_t vecsize);
   virtual Bool_t        Contains(const Double_t *point) const;
   virtual void          Contains_v(const Double_t *points, Bool_t *inside, Int_t vecsize) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromInside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual Double_t      DistFromOutside(const Double_t *point, const Double_t *dir, Int_t iact=1,
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual void          DistFromOutside_v(const Double_t *points, const Double_t *dirs, Double_t *dists, Int_t vecsize, Double_t *step) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv,
                                Double_t start, Double_t step);
   TGeoBoolNode         *GetBoolNode() const {return fNode;}
   virtual void          GetBoundingCylinder(Double_t * /*param*/) const {;}
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape * /*mother*/, TGeoMatrix * /*mat*/) const {return 0;}
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const;
   virtual Bool_t        GetPointsOnSegments(Int_t /*npoints*/, Double_t * /*array*/) const {return kFALSE;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsComposite() const {return kTRUE;}
   virtual Bool_t        IsCylType() const {return kFALSE;}
   void                  MakeNode(const char *expression);
   virtual Bool_t        PaintComposite(Option_t *option = "") const;
   void                  RegisterYourself();
   virtual Double_t      Safety(const Double_t *point, Bool_t in=kTRUE) const;
   virtual void          Safety_v(const Double_t *points, const Bool_t *inside, Double_t *safe, Int_t vecsize) const;
   virtual void          SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void          SetDimensions(Double_t * /*param*/) {;}
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          Sizeof3D() const;

   ClassDef(TGeoCompositeShape, 1)         // boolean composite shape
};



#endif

