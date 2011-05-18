// @(#)root/geom:$Id$
// Author: Andrei Gheata   02/06/05

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoShapeAssembly
#define ROOT_TGeoShapeAssembly

#ifndef ROOT_TGeoBBox
#include "TGeoBBox.h"
#endif

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoShapeAssembly - The shape encapsulating an assembly (union) of     //
//         volumes. Automatically created by TGeoVolumeAssembly class     //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoVolumeAssembly;

class TGeoShapeAssembly : public TGeoBBox
{
protected :
// data members
   Int_t                 fCurrent;  //! node number for current node
   Int_t                 fNext;     //! node number for next crossed node
   TGeoVolumeAssembly   *fVolume;   // assembly volume
   Bool_t                fBBoxOK;   // has bounding box been calculated

// methods
public:
   // constructors
   TGeoShapeAssembly();
   TGeoShapeAssembly(TGeoVolumeAssembly *vol);
   // destructor
   virtual ~TGeoShapeAssembly();
   // methods
   virtual void          ComputeBBox();
   virtual void          ComputeNormal(Double_t *point, Double_t *dir, Double_t *norm);
   virtual Bool_t        Contains(Double_t *point) const;
   virtual Int_t         DistancetoPrimitive(Int_t px, Int_t py);
   virtual Double_t      DistFromInside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual Double_t      DistFromOutside(Double_t *point, Double_t *dir, Int_t iact=1, 
                                   Double_t step=TGeoShape::Big(), Double_t *safe=0) const;
   virtual TGeoVolume   *Divide(TGeoVolume *voldiv, const char *divname, Int_t iaxis, Int_t ndiv, 
                                Double_t start, Double_t step);
   virtual TGeoShape    *GetMakeRuntimeShape(TGeoShape *mother, TGeoMatrix *mat) const;
   virtual void          GetMeshNumbers(Int_t &nvert, Int_t &nsegs, Int_t &npols) const;
   virtual Int_t         GetNmeshVertices() const {return 0;}
   virtual void          InspectShape() const;
   virtual Bool_t        IsAssembly() const {return kTRUE;}
   virtual Bool_t        IsCylType() const {return kFALSE;}
   void                  NeedsBBoxRecompute() {fBBoxOK = kFALSE;}
   void                  RecomputeBoxLast();
   virtual Double_t      Safety(Double_t *point, Bool_t in=kTRUE) const;
   virtual void          SavePrimitive(ostream &out, Option_t *option = "");
   virtual void          SetPoints(Double_t *points) const;
   virtual void          SetPoints(Float_t *points) const;
   virtual void          SetSegsAndPols(TBuffer3D &buff) const;

   ClassDef(TGeoShapeAssembly, 2)         // assembly shape
};

#endif
