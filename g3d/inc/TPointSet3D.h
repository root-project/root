// @(#)root/g3d:$Name:  $:$Id: TPointSet3D.h,v 1.2 2006/04/07 09:20:43 rdm Exp $
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

class TPointSet3D : public TPolyMarker3D, public TAttBBox
{
protected:
   TObject** fIds;    //!User-provided point identifications
   Int_t     fNIds;   //!Number of allocated entries for user-ids
   Bool_t    fOwnIds; //!Flag specifying id-objects are owned by the point-set
   Bool_t    fFakeIds;//!Flag specifying ids are not TObject*

   TPointSet3D& operator=(const TPointSet3D&);

public:
   TPointSet3D() :
      TPolyMarker3D(), fIds(0), fNIds(0),
      fOwnIds(kFALSE), fFakeIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, m, opt), fIds(0), fNIds(0),
      fOwnIds(kFALSE), fFakeIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Float_t *p, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, p, m, opt), fIds(0), fNIds(0),
      fOwnIds(kFALSE), fFakeIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(Int_t n, Double_t *p, Marker_t m=1, Option_t *opt="") :
      TPolyMarker3D(n, p, m, opt), fIds(0), fNIds(0),
      fOwnIds(kFALSE), fFakeIds(kFALSE) { fName = "TPointSet3D"; }
   TPointSet3D(const TPointSet3D &ps) :
      TPolyMarker3D(ps), TAttBBox(), fIds(0), fNIds(0),
      fOwnIds(kFALSE), fFakeIds(kFALSE) { fName = "TPointSet3D"; }

   virtual ~TPointSet3D();

   virtual void ComputeBBox();

   void     SetPointId(TObject* id);
   void     SetPointId(Int_t n, TObject* id);
   TObject* GetPointId(Int_t n) const;
   void     ClearIds();

   Bool_t GetOwnIds() const    { return fOwnIds; }
   void   SetOwnIds(Bool_t o)  { fOwnIds = o; }
   Bool_t GetFakeIds() const   { return fFakeIds; }
   void   SetFakeIds(Bool_t f) { fFakeIds = f; }

   virtual void PointSelected(Int_t n);

   ClassDef(TPointSet3D,1) // TPolyMarker3D with direct OpenGL rendering.
}; // endclass TPointSet3D

#endif
