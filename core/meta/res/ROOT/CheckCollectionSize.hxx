/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CheckCollectionSize
#define ROOT_CheckCollectionSize

// Private header shared by libCore and libRIO.

#include "RtypesCore.h"
#include "TBuffer.h"
#include "TError.h"

namespace ROOT {
namespace Internal {

////////////////////////////////////////////////////////////////////////////////
/// Validate the number of elements of a collection that was just read from `buf`.
///
/// The collection size is stored as an Int_t but consumed as an unsigned value
/// (TVirtualCollectionProxy::Allocate, std::vector::resize), so a negative value
/// coming from corrupt data would turn into a request for billions of elements.
/// Report the problem from `location` and read the collection as empty instead.
inline Int_t CheckCollectionSize(TBuffer &buf, Int_t nElements, const char *location, const char *collectionName)
{
   if (R__unlikely(nElements < 0)) {
      ::Error(location,
              "Corrupt data: the collection of type %s read from %s has a negative number of elements (%d); "
              "reading it as empty.",
              collectionName ? collectionName : "(unknown)",
              buf.GetParent() ? buf.GetParent()->GetName() : "memory/socket", nElements);
      return 0;
   }
   return nElements;
}

} // namespace Internal
} // namespace ROOT

#endif
