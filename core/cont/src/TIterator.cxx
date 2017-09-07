// @(#)root/cont:$Id$
// Author: Fons Rademakers   13/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TIterator
\ingroup Containers
Iterator abstract base class. This base class provides the interface
for collection iterators.
*/

#include "TIterator.h"
#include "TError.h"

ClassImp(TIterator);

////////////////////////////////////////////////////////////////////////////////
/// Compare two iterator objects.
/// For backward compatibility reasons we have to provide this
/// default implementation.

Bool_t TIterator::operator!=(const TIterator &) const
{
   ::Warning("TIterator::operator!=", "this method must be overridden!");
   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Return current object or nullptr.
/// For backward compatibility reasons we have to provide this
/// default implementation.

TObject *TIterator::operator*() const
{
   ::Warning("TIterator::operator*", "this method must be overridden!");
   return nullptr;
}
