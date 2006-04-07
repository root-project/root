// $Header$

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

void TPointSet3D::ComputeBBox()
{
   if (fN > 0) {
      Int_t    n = fN;
      Float_t* p = fP;
      bbox_init();
      while (n--) {
	 bbox_check_point(p);
	 p += 3;
      }
   } else {
      bbox_zero();
   }
}
