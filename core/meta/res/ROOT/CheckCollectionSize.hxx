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
#include "TObjArray.h"
#include "TStreamerElement.h"
#include "TVirtualStreamerInfo.h"

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

////////////////////////////////////////////////////////////////////////////////
/// Validate the number of elements of a collection that was just read from `buf`.
///
/// In addition to the checks of the overload above, reject sizes larger than
/// the number of bytes left in the buffer if each element is known to occupy
/// at least one byte on file. Such a size can only come from corrupt data, and
/// unlike a negative one it is not caught by Allocate or resize, which would
/// try to allocate and initialise up to 2^31 elements.
///
/// \param[in] elementsUseBytes Callable returning whether each element occupies
///            at least one byte on file. It is only called for sizes larger than
///            the remaining bytes, so it does not slow down reading valid data.
template <typename ElementsUseBytes>
inline Int_t CheckCollectionSize(TBuffer &buf, Int_t nElements, const char *location, const char *collectionName,
                                 ElementsUseBytes &&elementsUseBytes)
{
   nElements = CheckCollectionSize(buf, nElements, location, collectionName);
   const Long64_t remaining = (Long64_t)buf.BufferSize() - buf.Length();
   // TBufferText and its subclasses do not keep the data in the buffer. TBufferSQL inherits from TBufferFile
   // without doing so either, but TTreeSQL does not support branches of collections or objects.
   if (R__unlikely(nElements > remaining) && buf.InheritsFrom("TBufferFile") && elementsUseBytes()) {
      ::Error(location,
              "Corrupt data: the collection of type %s read from %s has %d elements, but only %lld bytes are left "
              "in the buffer; reading it as empty.",
              collectionName ? collectionName : "(unknown)",
              buf.GetParent() ? buf.GetParent()->GetName() : "memory/socket", nElements, remaining);
      return 0;
   }
   return nElements;
}

////////////////////////////////////////////////////////////////////////////////
/// Return whether each element of a collection streamed member-wise with the
/// on-file layout `info` occupies at least one byte on file.
///
/// This is the case as soon as one of the data members is streamed as a
/// basic type or an array of them, a pointer written with an object tag, a
/// string or an STL collection. Members of class type are not considered: an
/// object can occupy zero bytes, e.g. with a custom streamer. Neither are
/// pointers marked `//->`, whose object is streamed in place.
inline bool MemberWiseElementsUseBytes(const TVirtualStreamerInfo *info)
{
   if (!info || !info->GetElements())
      return false;
   for (const auto *obj : *info->GetElements()) {
      const auto *element = dynamic_cast<const TStreamerElement *>(obj);
      // Artificial elements are set by I/O customization rules, they do not read from the buffer, and custom member
      // streamers can read anything.
      if (!element || dynamic_cast<const TStreamerArtificial *>(element) || element->GetStreamer())
         continue;
      const Int_t type = element->GetType();
      switch (type) {
      case TVirtualStreamerInfo::kObjectP:
      case TVirtualStreamerInfo::kTString:
      case TVirtualStreamerInfo::kTObject:
      case TVirtualStreamerInfo::kTNamed:
      case TVirtualStreamerInfo::kAnyP:
      case TVirtualStreamerInfo::kAnyPnoVT:
      case TVirtualStreamerInfo::kSTLp: return true;
      default:
         if (type > TVirtualStreamerInfo::kBase && type < TVirtualStreamerInfo::kObject) {
            // A basic type, an array of them (kOffsetL), or a pointer to an array of them (kOffsetP), which starts
            // with a byte telling whether the array is present.
            const Int_t basicType = type % TVirtualStreamerInfo::kOffsetL;
            if (basicType >= TVirtualStreamerInfo::kChar && basicType <= TVirtualStreamerInfo::kFloat16 &&
                basicType != TVirtualStreamerInfo::kLegacyChar)
               return true;
         } else if (type >= TVirtualStreamerInfo::kSTL && type <= TVirtualStreamerInfo::kSTLstring) {
            return true;
         }
      }
   }
   return false;
}

} // namespace Internal
} // namespace ROOT

#endif
