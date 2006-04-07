// $Header$

#include "TAttBBox.h"


//______________________________________________________________________
// TAttBBox
//
// Helper for management of bounding-box information.
// Optionally used by classes that use direct OpenGL rendering
// via <Class>GLRenderer class.

ClassImp(TAttBBox)

void TAttBBox::bbox_init(Float_t infinity)
{
   // Allocate and prepare for incremental filling.

   if (fBBox == 0) fBBox = new Float_t[6];

   fBBox[0] =  infinity;   fBBox[1] = -infinity;
   fBBox[2] =  infinity;   fBBox[3] = -infinity;
   fBBox[4] =  infinity;   fBBox[5] = -infinity;  
}

void TAttBBox::bbox_zero(Float_t epsilon, Float_t x, Float_t y, Float_t z)
{
   // Create cube of volume (2*epsiolon)^3 at (x,y,z).
   // epsilon iz zero by default.

   if (fBBox == 0) fBBox = new Float_t[6];

   fBBox[0] = x - epsilon;   fBBox[1] = x + epsilon;
   fBBox[2] = y - epsilon;   fBBox[3] = y + epsilon;
   fBBox[4] = z - epsilon;   fBBox[5] = z + epsilon;
}

void TAttBBox::bbox_clear()
{
   delete [] fBBox; fBBox = 0;
}
