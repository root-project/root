// @(#)root/g3d:$Name:  $:$Id: TBRIK.cxx,v 1.2 2002/11/11 11:21:16 brun Exp $
// Author: Nenad Buncic 17/09/95 

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TBRIK.h"
#include "TNode.h"
#include "TVirtualPad.h"
#include "TBuffer3D.h"
#include "TGeometry.h"

ClassImp(TBRIK)


//______________________________________________________________________________
// Begin_Html <P ALIGN=CENTER> <IMG SRC="gif/brik.gif"> </P> End_Html
// BRIK is a box with faces perpendicular to the axes. It has 6 parameters:
//
//     - name       name of the shape
//     - title      shape's title
//     - material  (see TMaterial)
//     - dx         half-length of the box along the x-axis
//     - dy         half-length of the box along the y-axis
//     - dz         half-length of the box along the z-axis


//______________________________________________________________________________
TBRIK::TBRIK()
{
   // BRIK shape default constructor
}


//______________________________________________________________________________
TBRIK::TBRIK(const char *name, const char *title, const char *material, Float_t dx, Float_t dy, Float_t dz)
      : TShape (name, title,material)
{
   // BRIK shape normal constructor

   fDx = dx;
   fDy = dy;
   fDz = dz;
}


//______________________________________________________________________________
TBRIK::~TBRIK()
{
  // BRIK shape default destructor
}


//______________________________________________________________________________
Int_t TBRIK::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Compute distance from point px,py to a BRIK
   //
   // Compute the closest distance of approach from point px,py to each corner
   // point of the BRIK.

   const Int_t numPoints = 8;
   return ShapeDistancetoPrimitive(numPoints,px,py);
}

//______________________________________________________________________________
void TBRIK::Paint(Option_t *option)
{
   // Paint this 3-D shape with its current attributes

   // Allocate the necessary spage in gPad->fBuffer3D to store this shape
   Int_t NbPnts = 8;
   Int_t NbSegs = 12;
   Int_t NbPols = 6;
   TBuffer3D *buff = gPad->AllocateBuffer3D(3*NbPnts, 3*NbSegs, 6*NbPols);
   if (!buff) return;

   buff->fType = TBuffer3D::kBRIK;
   buff->fId   = this;

   // Fill gPad->fBuffer3D. Points coordinates are in Master space
   buff->fNbPnts = NbPnts;
   buff->fNbSegs = NbSegs;
   buff->fNbPols = NbPols;
   // In case of option "size" it is not necessary to fill the buffer
   if (buff->fOption == TBuffer3D::kSIZE) {
      buff->Paint(option);
      return;
   }

   SetPoints(buff->fPnts);

   TransformPoints(buff);

   // Basic colors: 0, 1, ... 7
   Int_t c = ((GetLineColor() % 8) - 1) * 4;
   if (c < 0) c = 0;

   buff->fSegs[ 0] = c   ; buff->fSegs[ 1] = 0   ; buff->fSegs[ 2] = 1   ;
   buff->fSegs[ 3] = c+1 ; buff->fSegs[ 4] = 1   ; buff->fSegs[ 5] = 2   ;
   buff->fSegs[ 6] = c+1 ; buff->fSegs[ 7] = 2   ; buff->fSegs[ 8] = 3   ;
   buff->fSegs[ 9] = c   ; buff->fSegs[10] = 3   ; buff->fSegs[11] = 0   ;
   buff->fSegs[12] = c+2 ; buff->fSegs[13] = 4   ; buff->fSegs[14] = 5   ;
   buff->fSegs[15] = c+2 ; buff->fSegs[16] = 5   ; buff->fSegs[17] = 6   ;
   buff->fSegs[18] = c+3 ; buff->fSegs[19] = 6   ; buff->fSegs[20] = 7   ;
   buff->fSegs[21] = c+3 ; buff->fSegs[22] = 7   ; buff->fSegs[23] = 4   ;
   buff->fSegs[24] = c   ; buff->fSegs[25] = 0   ; buff->fSegs[26] = 4   ;
   buff->fSegs[27] = c+2 ; buff->fSegs[28] = 1   ; buff->fSegs[29] = 5   ;
   buff->fSegs[30] = c+1 ; buff->fSegs[31] = 2   ; buff->fSegs[32] = 6   ;
   buff->fSegs[33] = c+3 ; buff->fSegs[34] = 3   ; buff->fSegs[35] = 7   ;

   buff->fPols[ 0] = c   ; buff->fPols[ 1] = 4   ;  buff->fPols[ 2] = 0  ;
   buff->fPols[ 3] = 9   ; buff->fPols[ 4] = 4   ;  buff->fPols[ 5] = 8  ;
   buff->fPols[ 6] = c+1 ; buff->fPols[ 7] = 4   ;  buff->fPols[ 8] = 1  ;
   buff->fPols[ 9] = 10  ; buff->fPols[10] = 5   ;  buff->fPols[11] = 9  ;
   buff->fPols[12] = c   ; buff->fPols[13] = 4   ;  buff->fPols[14] = 2  ;
   buff->fPols[15] = 11  ; buff->fPols[16] = 6   ;  buff->fPols[17] = 10 ;
   buff->fPols[18] = c+1 ; buff->fPols[19] = 4   ;  buff->fPols[20] = 3  ;
   buff->fPols[21] = 8   ; buff->fPols[22] = 7   ;  buff->fPols[23] = 11 ;
   buff->fPols[24] = c+2 ; buff->fPols[25] = 4   ;  buff->fPols[26] = 0  ;
   buff->fPols[27] = 3   ; buff->fPols[28] = 2   ;  buff->fPols[29] = 1  ;
   buff->fPols[30] = c+3 ; buff->fPols[31] = 4   ;  buff->fPols[32] = 4  ;
   buff->fPols[33] = 5   ; buff->fPols[34] = 6   ;  buff->fPols[35] = 7  ;

   // Paint gPad->fBuffer3D
   buff->Paint(option);
}


//______________________________________________________________________________
void TBRIK::SetPoints(Double_t *buff)
{
   // Create BRIK points

   if (buff) {
      buff[ 0] = -fDx ; buff[ 1] = -fDy ; buff[ 2] = -fDz;
      buff[ 3] = -fDx ; buff[ 4] =  fDy ; buff[ 5] = -fDz;
      buff[ 6] =  fDx ; buff[ 7] =  fDy ; buff[ 8] = -fDz;
      buff[ 9] =  fDx ; buff[10] = -fDy ; buff[11] = -fDz;
      buff[12] = -fDx ; buff[13] = -fDy ; buff[14] =  fDz;
      buff[15] = -fDx ; buff[16] =  fDy ; buff[17] =  fDz;
      buff[18] =  fDx ; buff[19] =  fDy ; buff[20] =  fDz;
      buff[21] =  fDx ; buff[22] = -fDy ; buff[23] =  fDz;
   }
}

//______________________________________________________________________________
void TBRIK::Sizeof3D() const
{
   // Return total X3D needed by TNode::ls (when called with option "x")

   gSize3D.numPoints += 8;
   gSize3D.numSegs   += 12;
   gSize3D.numPolys  += 6;
}
