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

#include <ROOT/RCurlConnection.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorage.hxx>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

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
   /// Pattern for resolving object IDs to full S3 URLs. ${baseurl} is replaced with the anchor URL,
   /// ${objid} with the numeric object ID. Defaults to the scheme this writer uses; the reader
   /// overrides it from the stored anchor.
   std::string fUrlTemplate = "${baseurl}/${objid}";
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

/// \brief Translate an ntpl+s3 URI into its plain HTTP(S) equivalent.
///
/// Accepts `ntpl+s3+http://host/bucket/path` and `ntpl+s3+https://host/bucket/path`, returning the
/// URL with the scheme replaced by http or https respectively. Throws RException on any other scheme.
std::string ParseS3Url(std::string_view uri);

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSinkS3
\ingroup NTuple
\brief Storage provider that writes ntuple pages into S3-compatible object storage.

Currently implements Mode B (one sealed page per S3 object, kTypeObject64 locators).
Mode A (multiple packed pages per object, kTypeMulti locators) will be added separately.

\warning The S3 backend is experimental and under active development.
*/
// clang-format on
class RPageSinkS3 : public ROOT::Internal::RPagePersistentSink {
private:
   /// HTTP base URL for this ntuple (derived from the s3 scheme URI); never has a trailing slash
   std::string fBaseUrl;
   /// One HTTP connection reused for every upload, so curl keeps it alive across objects on the same
   /// host instead of re-handshaking per object.
   ROOT::Internal::RCurlConnection fConnection;
   /// Object ID counter; incremented for each object written.
   std::uint64_t fObjectId{0};
   /// Tracks the number of bytes committed to the current cluster (reset in StageClusterImpl)
   std::uint64_t fNBytesCurrentCluster{0};
   /// Anchor metadata populated during the write path and uploaded last in CommitDatasetImpl
   RNTupleAnchorS3 fAnchor;

   /// Resolve a numeric object ID to its full HTTP URL
   std::string MakeObjectUrl(std::uint64_t objId) const;
   /// Upload raw bytes to the given S3 URL via an HTTP PUT request
   void PutObject(const std::string &url, const unsigned char *data, std::size_t size);

   /// Tag to select the internal constructor that takes an already-resolved base URL.
   struct RFromBaseUrl {};
   /// Internal constructor used by CloneAsHidden: the public constructor derives the base URL by parsing
   /// an s3 scheme URI, whereas a clone already has a resolved base URL to write under.
   RPageSinkS3(std::string_view ntupleName, std::string baseUrl, const ROOT::RNTupleWriteOptions &options,
               RFromBaseUrl);

protected:
   using RPagePersistentSink::InitImpl;
   void InitImpl(unsigned char *serializedHeader, std::uint32_t length) final;
   RNTupleLocator
   CommitSealedPageImpl(ROOT::DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) final;
   std::uint64_t StageClusterImpl() final;
   RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) final;
   using RPagePersistentSink::CommitDatasetImpl;
   ROOT::Internal::RNTupleLink CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) final;

public:
   RPageSinkS3(std::string_view ntupleName, std::string_view uri, const ROOT::RNTupleWriteOptions &options);
   ~RPageSinkS3() override;

   std::unique_ptr<ROOT::Internal::RPageSink>
   CloneAsHidden(std::string_view name, const ROOT::RNTupleWriteOptions &opts) const final;
}; // class RPageSinkS3

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
