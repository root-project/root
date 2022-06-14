// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveGridStepper.h"
#include "TEveTrans.h"

/** \class TEveGridStepper
\ingroup TEve
Provide discrete position coordinates for placement of objects on
regular grids.
*/

ClassImp(TEveGridStepper);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveGridStepper::TEveGridStepper(Int_t sm) :
   fMode(EStepMode_e(sm)),
   fCx(0), fCy(0), fCz(0), fNx(0), fNy(0), fNz(0),
   fDx(0), fDy(0), fDz(0), fOx(0), fOy(0), fOz(0)
{
   switch(fMode) {
      default:
      case kSM_XYZ:
         fLimitArr[0] = &fNx; fLimitArr[1] = &fNy; fLimitArr[2] = &fNz;
         fValueArr[0] = &fCx; fValueArr[1] = &fCy; fValueArr[2] = &fCz;
         break;
      case kSM_YXZ:
         fLimitArr[0] = &fNy; fLimitArr[1] = &fNx; fLimitArr[2] = &fNz;
         fValueArr[0] = &fCy; fValueArr[1] = &fCx; fValueArr[2] = &fCz;
         break;
      case kSM_XZY:
         fLimitArr[0] = &fNx; fLimitArr[1] = &fNz; fLimitArr[2] = &fNy;
         fValueArr[0] = &fCx; fValueArr[1] = &fCz; fValueArr[2] = &fCy;
         break;
   }

   fCx = fCy = fCz = 0;
   fNx = fNy = fNz = 16;
   fDx = fDy = fDz = 1;
   fOx = fOy = fOz = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reset position to origin.

void TEveGridStepper::Reset()
{
   fCx = fCy = fCz = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Subtract current position of 's' from origin of this.

void TEveGridStepper::Subtract(TEveGridStepper& s)
{
   fOx = -(s.fOx + s.fCx*s.fDx);
   fOy = -(s.fOy + s.fCy*s.fDy);
   fOz = -(s.fOz + s.fCz*s.fDz);
}

////////////////////////////////////////////////////////////////////////////////
/// Move to the next grid position.

Bool_t TEveGridStepper::Step()
{
   (*fValueArr[0])++;
   if (*fValueArr[0] >= *fLimitArr[0]) {
      *fValueArr[0] = 0; (*fValueArr[1])++;
      if (*fValueArr[1] >= *fLimitArr[1]) {
         *fValueArr[1] = 0; (*fValueArr[2])++;
         if (*fValueArr[2] >= *fLimitArr[2]) {
            return kFALSE;
         }
      }
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get current position.

void TEveGridStepper::GetPosition(Float_t* p)
{
   p[0] = fOx + fCx*fDx;
   p[1] = fOy + fCy*fDy;
   p[2] = fOz + fCz*fDz;
}

////////////////////////////////////////////////////////////////////////////////
/// Set position into the translation part of mx.

void TEveGridStepper::SetTrans(TEveTrans* mx)
{
   mx->SetPos(fOx + fCx*fDx, fOy + fCy*fDy, fOz + fCz*fDz);
}

////////////////////////////////////////////////////////////////////////////////
/// Set position into the translation part of mx and advance to the
/// next grid position.

void TEveGridStepper::SetTransAdvance(TEveTrans* mx)
{
   SetTrans(mx);
   Step();
}
