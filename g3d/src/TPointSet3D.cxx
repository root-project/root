// @(#)root/g3d:$Name:  $:$Id: TPointSet3D.cxx,v 1.3 2006/05/09 10:24:26 brun Exp $
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TPointSet3D.h"


//______________________________________________________________________
// TPointSet3D
//
// TPolyMarker3D using TPointSet3DGL for direct OpenGL rendering.
// Supports does not support exotic marker types, only
// full-circle (20) and square (21). Marker size is applied.
// If other marker type is specified, pixels are rendered and point-size
// is ignored.

ClassImp(TPointSet3D)

//______________________________________________________________________________
void TPointSet3D::ComputeBBox()
{
   //Compute the bounding box of this points set
   if (fN > 0) {
      Int_t    n = fN;
      Float_t* p = fP;
      BBoxInit();
      while (n--) {
         BBoxCheckPoint(p);
         p += 3;
      }
   } else {
      BBoxZero();
   }
}
