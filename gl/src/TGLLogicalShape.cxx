// @(#)root/gl:$Name:  $:$Id: TGLLogicalShape.cxx,v 1.4 2005/05/28 12:21:00 rdm Exp $
// Author:  Richard Maunder  25/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLLogicalShape.h"
#include "TGLDisplayListCache.h"

ClassImp(TGLLogicalShape)

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(ULong_t ID) :
   TGLDrawable(ID, kFALSE), // Logical shapes not DL cached by default at present
   fRef(0), fRefStrong(kFALSE)
{
}

//______________________________________________________________________________
TGLLogicalShape::~TGLLogicalShape()
{
   // Physical refs should have been cleared
   if (fRef > 0) {
      assert(kFALSE);
   }
}

//______________________________________________________________________________
void TGLLogicalShape::Purge()
{
   // Overload and clear out any costly geometry cache

   // Base work - purge the DL cache
   TGLDrawable::Purge();
}
