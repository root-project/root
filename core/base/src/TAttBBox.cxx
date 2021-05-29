// @(#)root/base:$Id$
// Author: Matevz Tadel  7/4/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TAttBBox.h"

/** \class TAttBBox
\ingroup Base
\ingroup GraphicsAtt

Helper for management of bounding-box information.
Optionally used by classes that use direct OpenGL rendering
via `<Class>GL class`.
*/

ClassImp(TAttBBox);

////////////////////////////////////////////////////////////////////////////////
/// Allocate and prepare for incremental filling.

void TAttBBox::BBoxInit(Float_t infinity)
{
   if (fBBox == nullptr) fBBox = new Float_t[6];

   fBBox[0] =  infinity;   fBBox[1] = -infinity;
   fBBox[2] =  infinity;   fBBox[3] = -infinity;
   fBBox[4] =  infinity;   fBBox[5] = -infinity;
}

////////////////////////////////////////////////////////////////////////////////
/// Create cube of volume (2*epsilon)^3 at (x,y,z).
/// epsilon is zero by default.

void TAttBBox::BBoxZero(Float_t epsilon, Float_t x, Float_t y, Float_t z)
{
   if (fBBox == nullptr) fBBox = new Float_t[6];

   fBBox[0] = x - epsilon;   fBBox[1] = x + epsilon;
   fBBox[2] = y - epsilon;   fBBox[3] = y + epsilon;
   fBBox[4] = z - epsilon;   fBBox[5] = z + epsilon;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove BBox information.

void TAttBBox::BBoxClear()
{
   delete [] fBBox; fBBox = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Assert extents of all sides of the bounding-box are at least epsilon.

void TAttBBox::AssertBBoxExtents(Float_t epsilon)
{
   for (Int_t i=0; i<6; i+=2) {
      if (fBBox[i+1] - fBBox[i] < epsilon) {
         Float_t b  = 0.5*(fBBox[i] + fBBox[i+1]);
         fBBox[i]   = b - 0.5*epsilon;
         fBBox[i+1] = b + 0.5*epsilon;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Setup bounding box as cube with given extent and center position.

void TAttBBox::SetupBBoxCube(Float_t extent, Float_t x, Float_t y, Float_t z)
{
   BBoxZero(extent, x, y, z);
}
