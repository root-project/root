// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveScalableStraightLineSet
#define ROOT_TEveScalableStraightLineSet

#include "TEveStraightLineSet.h"

class TEveScalableStraightLineSet : public TEveStraightLineSet
{
private:
   TEveScalableStraightLineSet(const TEveScalableStraightLineSet&);            // Not implemented
   TEveScalableStraightLineSet& operator=(const TEveScalableStraightLineSet&); // Not implemented

protected:
   Double_t      fCurrentScale;
   Float_t       fScaleCenter[3];

public:
   TEveScalableStraightLineSet(const char* n="ScalableStraightLineSet", const char* t="");
   virtual ~TEveScalableStraightLineSet() {}

   void SetScaleCenter(Float_t x, Float_t y, Float_t z);
   void SetScale(Double_t scale);

   Double_t GetScale() const;

   ClassDef(TEveScalableStraightLineSet, 1); // Straight-line-set with extra scaling.
};
#endif
