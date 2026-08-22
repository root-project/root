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
#include <ROOT/RSpan.hxx>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
class RNTupleAnchorS3 {
   friend class RPageSinkS3;

private:
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
   /// Pattern for resolving clone (attribute-set) names to base URLs.
   /// ${baseurl} is replaced with the anchor URL, ${name} with the clone name.
   std::string fCloneTemplate = "${baseurl}/_clone/${name}";
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

public:
   RNTupleAnchorS3() = default;

   /// Deserialize the anchor from a JSON string. Returns an error on malformed or incompatible input.
   static RResult<RNTupleAnchorS3> CreateFromJSON(const std::string &json);
   /// Serialize the anchor to a JSON string suitable for storage at the base URL
   std::string ToJSON() const;

   bool operator==(const RNTupleAnchorS3 &other) const;
   bool operator!=(const RNTupleAnchorS3 &other) const { return !(*this == other); }

   std::uint32_t GetVersionAnchor() const { return fVersionAnchor; }
   std::uint16_t GetVersionEpoch() const { return fVersionEpoch; }
   std::uint16_t GetVersionMajor() const { return fVersionMajor; }
   std::uint16_t GetVersionMinor() const { return fVersionMinor; }
   std::uint16_t GetVersionPatch() const { return fVersionPatch; }
   const std::string &GetUrlTemplate() const { return fUrlTemplate; }
   const std::string &GetCloneTemplate() const { return fCloneTemplate; }
   std::uint64_t GetHeaderObjId() const { return fHeaderObjId; }
   std::uint64_t GetHeaderOffset() const { return fHeaderOffset; }
   std::uint64_t GetNBytesHeader() const { return fNBytesHeader; }
   std::uint64_t GetLenHeader() const { return fLenHeader; }
   std::uint64_t GetFooterObjId() const { return fFooterObjId; }
   std::uint64_t GetFooterOffset() const { return fFooterOffset; }
   std::uint64_t GetNBytesFooter() const { return fNBytesFooter; }
   std::uint64_t GetLenFooter() const { return fLenFooter; }
};

/// \brief Translate an ntpl+s3 URI into its plain HTTP(S) equivalent.
///
/// Accepts `ntpl+s3+http://host/bucket/path` and `ntpl+s3+https://host/bucket/path`, returning the
/// URL with the scheme replaced by http or https respectively. Returns an error result for any other
/// scheme or a malformed URI (rather than throwing), so callers on untrusted input can handle it.
RResult<std::string> ParseS3Url(std::string_view uri);

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSinkS3
\ingroup NTuple
\brief Storage provider that writes ntuple pages into S3-compatible object storage.

Currently implements Mode B (one sealed page per S3 object, kTypeObject64 locators).
Mode A (multiple packed pages per object, kTypeMulti locators) will be added separately.

Prefer calling RNTupleWriter::CommitDataset() explicitly to letting the writer's destructor do it: a
destructor cannot propagate an exception, so a failed footer or anchor upload is only logged and leaves
the ntuple without an anchor, i.e. unreadable.

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
   RPageSinkS3(std::string_view ntupleName, std::string_view baseUrl, const ROOT::RNTupleWriteOptions &options,
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

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSourceS3
\ingroup NTuple
\brief Storage provider that reads ntuple pages from S3-compatible object storage.

Counterpart of RPageSinkS3: implements Mode B reads (one sealed page per S3 object, kTypeObject64
locators). Pages are fetched one object at a time; batching them into concurrent GETs is left to a
follow-up.

The anchor is read in a single GET that deliberately asks for more bytes than it holds: its size is not
known up front, and a range running past the end of an object returns only what exists, so the length of
the reply is the size of the anchor.

\warning The S3 backend is experimental and under active development.
*/
// clang-format on
class RPageSourceS3 : public ROOT::Internal::RPageSource {
private:
   /// HTTP base URL for this ntuple (derived from the s3 scheme URI); never has a trailing slash
   std::string fBaseUrl;
   /// Connection used by everything that runs on the calling thread: the anchor, header and footer in
   /// LoadStructureImpl, the page lists in LoadPageListImpl and single pages in LoadSealedPageImpl.
   /// Reused across objects so curl keeps it alive instead of re-handshaking per object.
   ROOT::Internal::RCurlConnection fConnection;
   /// Connection used exclusively by LoadClusters(), which the cluster pool calls on its own I/O thread.
   /// A libcurl easy handle carries per-request state and must not be driven by two threads at once, so
   /// the prefetch path needs a handle of its own rather than sharing fConnection.
   ROOT::Internal::RCurlConnection fClusterConnection;
   /// Anchor metadata, fetched and parsed in LoadStructureImpl
   RNTupleAnchorS3 fAnchor;
   /// Populated by LoadStructureImpl and AttachImpl, moved out at the end of AttachImpl
   ROOT::Internal::RNTupleDescriptorBuilder fDescriptorBuilder;

   /// Resolve a numeric object ID to its full HTTP URL through the anchor's URL template
   std::string MakeObjectUrl(std::uint64_t objId) const;
   /// Download `size` bytes from `url` into the caller-provided `buffer` via an HTTP GET request.
   /// The connection is explicit because the caller's thread determines which one may be used.
   void GetObject(ROOT::Internal::RCurlConnection &connection, const std::string &url, unsigned char *buffer,
                  std::size_t size);
   /// Download at most `size` bytes from the start of `url`, returning how many arrived. Used to read an
   /// object of unknown length, where a short reply is the answer rather than an error.
   std::size_t GetObjectPrefix(ROOT::Internal::RCurlConnection &connection, const std::string &url,
                               unsigned char *buffer, std::size_t size);

   /// Tag to select the internal constructor that takes an already-resolved base URL.
   struct RFromBaseUrl {};
   /// Internal constructor used by CloneImpl: the public constructor derives the base URL by parsing an
   /// s3 scheme URI, whereas a clone of an open source already has one and must not re-parse it.
   RPageSourceS3(std::string_view ntupleName, std::string_view baseUrl, const ROOT::RNTupleReadOptions &options,
                 RFromBaseUrl);

   void LoadPageListImpl(const RNTupleLocator &locator, unsigned char *buffer) final;
   void LoadSealedPageImpl(const RNTupleLocator &locator, RSealedPage &sealedPage) final;

protected:
   void LoadStructureImpl() final;
   ROOT::RNTupleDescriptor AttachImpl() final;
   /// The cloned page source opens its own pair of HTTP connections to the same base URL.
   std::unique_ptr<RPageSource> CloneImpl() const final;

public:
   RPageSourceS3(std::string_view ntupleName, std::string_view uri, const ROOT::RNTupleReadOptions &options);
   ~RPageSourceS3() override;

   std::vector<std::unique_ptr<ROOT::Internal::RCluster>>
   LoadClusters(std::span<ROOT::Internal::RCluster::RKey> clusterKeys) final;

   void LoadStreamerInfo() final;

   std::unique_ptr<RPageSource> OpenWithDifferentAnchor(const ROOT::Internal::RNTupleLink &anchorLink,
                                                        const ROOT::RNTupleReadOptions &options = {}) final;
}; // class RPageSourceS3

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
