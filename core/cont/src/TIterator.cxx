// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TIterator                                                            //
//                                                                      //
// Iterator abstract base class. This base class provides the interface //
// for collection iterators.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TIterator.h"
#include "TError.h"


ClassImp(TIterator)

//______________________________________________________________________________
bool TIterator::operator!=(const TIterator &) const
{
   // Compare two iterator objects.
   // For backward compatibility reasons we have to provide this
   // default implementation.

   ::Warning("TIterator::operator!=", "this method must be overridden!");
   return false;
}

//______________________________________________________________________________
TObject *TIterator::operator*() const
{
   // Return current object or nullptr.
   // For backward compatibility reasons we have to provide this
   // default implementation.

   ::Warning("TIterator::operator*", "this method must be overridden!");
   return nullptr;
}
