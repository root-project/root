// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveScalableStraightLineSet.h"
#include "TEveChunkManager.h"

//==============================================================================
//==============================================================================
// TEveScalableStraightLineSet
//==============================================================================

//______________________________________________________________________________
//
// Straight-line-set with extra scaling, useful for projectables that need
// to be scaled in accordance with an external object.


ClassImp(TEveScalableStraightLineSet);

//______________________________________________________________________________
TEveScalableStraightLineSet::TEveScalableStraightLineSet(const char* n, const char* t):
   TEveStraightLineSet (n, t),
   fCurrentScale(1.0)
{
   // Constructor.

   fScaleCenter[0] = 0;
   fScaleCenter[1] = 0;
   fScaleCenter[2] = 0;
}

//______________________________________________________________________________
void TEveScalableStraightLineSet::SetScaleCenter(Float_t x, Float_t y, Float_t z)
{
   // Set scale center.

   fScaleCenter[0] = x;
   fScaleCenter[1] = y;
   fScaleCenter[2] = z;
}

//______________________________________________________________________________
Double_t TEveScalableStraightLineSet::GetScale() const
{
   // Return current scale.

   return fCurrentScale;
}

//______________________________________________________________________________
void TEveScalableStraightLineSet::SetScale(Double_t scale)
{
   // Loop over line parameters and scale coordinates.

   TEveChunkManager::iterator li(GetLinePlex());
   while (li.next())
   {
      TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li();
      l.fV1[0] = fScaleCenter[0]+(l.fV1[0]-fScaleCenter[0])/fCurrentScale*scale;
      l.fV1[1] = fScaleCenter[1]+(l.fV1[1]-fScaleCenter[1])/fCurrentScale*scale;
      l.fV1[2] = fScaleCenter[2]+(l.fV1[2]-fScaleCenter[2])/fCurrentScale*scale;
      l.fV2[0] = fScaleCenter[0]+(l.fV2[0]-fScaleCenter[0])/fCurrentScale*scale;
      l.fV2[1] = fScaleCenter[1]+(l.fV2[1]-fScaleCenter[1])/fCurrentScale*scale;
      l.fV2[2] = fScaleCenter[2]+(l.fV2[2]-fScaleCenter[2])/fCurrentScale*scale;
   }
   fCurrentScale = scale;
}
