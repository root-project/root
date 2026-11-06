/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ObjectWiseElementsUseBytes
#define ROOT_ObjectWiseElementsUseBytes

#include "TDataType.h"
#include "TDictionary.h"
#include "TGenCollectionProxy.h"

namespace ROOT {
namespace Internal {

////////////////////////////////////////////////////////////////////////////////
/// Return whether an element described by `value` occupies at least one byte on
/// file when streamed object-wise by a collection proxy.
///
/// Objects of class type can occupy zero bytes, e.g. with a custom streamer, and
/// pointers to fundamental types are not streamed at all.
inline bool ObjectWiseElementUsesBytes(const TGenCollectionProxy::Value *value)
{
   if (!value)
      return false;
   switch (value->fCase) {
   case kIsFundamental:
   case kIsEnum:
      // These kinds are not supported, and are not streamed.
      return value->fKind != kchar && value->fKind != kNoType_t && value->fKind != kOther_t;
   case TGenCollectionProxy::kBIT_ISSTRING:
   case 0U | kIsPointer | kIsClass:
   case 0U | kIsPointer | TGenCollectionProxy::kBIT_ISSTRING:
   case 0U | kIsPointer | TGenCollectionProxy::kBIT_ISTSTRING | kIsClass: return true;
   default: return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return whether each element of a collection proxy with the given key (for
/// maps, nullptr otherwise) and value descriptors occupies at least one byte on
/// file when streamed object-wise.
inline bool ObjectWiseElementsUseBytes(const TGenCollectionProxy::Value *key, const TGenCollectionProxy::Value *value)
{
   return ObjectWiseElementUsesBytes(key) || ObjectWiseElementUsesBytes(value);
}

} // namespace Internal
} // namespace ROOT

#endif
