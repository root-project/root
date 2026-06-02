/// \file ROOT/RPageStorageS3.hxx
/// \author Jas Mehta <jasmehta805@gmail.com>
/// \date 2026-06-01

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RPageStorageS3
#define ROOT_RPageStorageS3

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>

#include <cstdint>
#include <string>

namespace ROOT {
namespace Experimental {
namespace Internal {

// clang-format off
/**
\class ROOT::Experimental::Internal::RNTupleAnchorS3
\ingroup NTuple
\brief Entry point for an RNTuple stored in S3-compatible object storage.

The anchor is serialized as a JSON object and stored at the base URL of the ntuple.
It contains the information needed to locate and read the header and footer envelopes.
The anchor is always the last object written during CommitDatasetImpl, ensuring atomicity:
if the anchor exists, the entire ntuple is complete.
*/
// clang-format on
struct RNTupleAnchorS3 {
   /// Allows evolving the anchor JSON schema in future versions
   std::uint32_t fVersionAnchor = 0;
   /// Version of the RNTuple binary format supported by the writer
   std::uint16_t fVersionEpoch = RNTuple::kVersionEpoch;
   std::uint16_t fVersionMajor = RNTuple::kVersionMajor;
   std::uint16_t fVersionMinor = RNTuple::kVersionMinor;
   std::uint16_t fVersionPatch = RNTuple::kVersionPatch;
   /// Pattern for resolving object IDs to full S3 URLs.
   /// ${baseurl} is replaced with the anchor URL, ${objid} with the numeric object ID.
   std::string fUrlTemplate;
   /// Object ID and byte offset of the compressed header within the S3 object
   std::uint64_t fHeaderObjId = 0;
   std::uint64_t fHeaderOffset = 0;
   /// Compressed and uncompressed sizes of the header envelope
   std::uint64_t fNBytesHeader = 0;
   std::uint64_t fLenHeader = 0;
   /// Object ID and byte offset of the compressed footer within the S3 object
   std::uint64_t fFooterObjId = 0;
   std::uint64_t fFooterOffset = 0;
   /// Compressed and uncompressed sizes of the footer envelope
   std::uint64_t fNBytesFooter = 0;
   std::uint64_t fLenFooter = 0;

   bool operator==(const RNTupleAnchorS3 &other) const;

   /// Serialize the anchor to a JSON string suitable for storage at the base URL
   std::string ToJSON() const;
   /// Deserialize the anchor from a JSON string. Returns an error on malformed or incompatible input.
   static RResult<RNTupleAnchorS3> CreateFromJSON(const std::string &json);
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
