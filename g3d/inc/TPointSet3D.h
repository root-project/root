// @(#)root/g3d:$Name:  $:$Id: TPointSet3D.h,v 1.3 2006/08/23 14:39:40 brun Exp $
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TPointSet3D
#define ROOT_TPointSet3D

#ifndef ROOT_TPolyMarker3D
#include "TPolyMarker3D.h"
#endif
#ifndef ROOT_TAttBBox
#include "TAttBBox.h"
#endif

#include "TRefArray.h"

class TPointSet3D : public TPolyMarker3D, public TAttBBox
{
protected:
   Bool_t    fOwnIds; //Flag specifying id-objects are owned by the point-set
   TRefArray fIds;    //User-provided point identifications

   TPointSet3D& operator=(const TPointSet3D&);

public:
   TPointSet3D() :
      TPolyMarker3D(), fOwnIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, m, opt), fOwnIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Float_t *p, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, p, m, opt), fOwnIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Double_t *p, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, p, m, opt), fOwnIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(const TPointSet3D &ps) :
      TPolyMarker3D(ps), TAttBBox(ps), fOwnIds(kFALSE), fIds(ps.fIds) {}

   virtual ~TPointSet3D();

   virtual void ComputeBBox();

   void     SetPointId(TObject* id);
   void     SetPointId(Int_t n, TObject* id);
   TObject* GetPointId(Int_t n) const { return fIds.At(n); }
   void     ClearIds();

   Bool_t GetOwnIds() const    { return fOwnIds; }
   void   SetOwnIds(Bool_t o)  { fOwnIds = o; }

   virtual void PointSelected(Int_t n);

   ClassDef(TPointSet3D,1) // TPolyMarker3D with direct OpenGL rendering.
}; // endclass TPointSet3D

#endif
