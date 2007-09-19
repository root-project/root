// @(#)root/g3d:$Id: TSocket.h,v 1.20 2005/07/29 14:26:51 rdm Exp $
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

public:
   TPointSet3D() :
     TPolyMarker3D() {}
   TPointSet3D(Int_t n, Marker_t marker=1, Option_t *option="") :
      TPolyMarker3D(n, marker, option) {}
   TPointSet3D(Int_t n, Float_t *p, Marker_t marker=1, Option_t *option="") :
      TPolyMarker3D(n, p, marker, option) {}
   TPointSet3D(Int_t n, Double_t *p, Marker_t marker=1, Option_t *option="") :
      TPolyMarker3D(n, p, marker, option) {}
   TPointSet3D(const TPointSet3D &polymarker) :
      TPolyMarker3D(polymarker), TAttBBox() {}

   virtual ~TPointSet3D() {}

   virtual void ComputeBBox();

   ClassDef(TPointSet3D,1) // TPolyMarker3D with direct OpenGL rendering.
}; // endclass TPointSet3D

#endif
