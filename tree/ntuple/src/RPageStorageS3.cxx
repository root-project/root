/// \file RPageStorageS3.cxx
/// \author Jas Mehta <jasmehta805@gmail.com>
/// \date 2026-06-01

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPageStorageS3.hxx>

#include <ROOT/RCluster.hxx>
#include <ROOT/RCurlConnection.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/StringUtils.hxx>

#include <nlohmann/json.hpp>
#include <xxhash.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RNTupleCompressor;
using ROOT::Internal::RNTupleDecompressor;
using ROOT::Internal::RNTupleSerializer;

/// Field-by-field equality check across all data members.
bool ROOT::Experimental::Internal::RNTupleAnchorS3::operator==(const RNTupleAnchorS3 &other) const
{
   return fVersionAnchor == other.fVersionAnchor && fVersionEpoch == other.fVersionEpoch &&
          fVersionMajor == other.fVersionMajor && fVersionMinor == other.fVersionMinor &&
          fVersionPatch == other.fVersionPatch && fUrlTemplate == other.fUrlTemplate &&
          fCloneTemplate == other.fCloneTemplate && fHeaderObjId == other.fHeaderObjId &&
          fHeaderOffset == other.fHeaderOffset && fNBytesHeader == other.fNBytesHeader &&
          fLenHeader == other.fLenHeader && fFooterObjId == other.fFooterObjId &&
          fFooterOffset == other.fFooterOffset && fNBytesFooter == other.fNBytesFooter &&
          fLenFooter == other.fLenFooter;
}

/// Serialize the anchor to a pretty-printed JSON string (2-space indent).
/// The checksum is computed over the compact canonical form of the data fields;
/// the stored JSON uses pretty-printing for readability.
std::string ROOT::Experimental::Internal::RNTupleAnchorS3::ToJSON() const
{
   nlohmann::json jsonAnchor;
   jsonAnchor["anchorVersion"] = fVersionAnchor;
   jsonAnchor["formatVersionEpoch"] = fVersionEpoch;
   jsonAnchor["formatVersionMajor"] = fVersionMajor;
   jsonAnchor["formatVersionMinor"] = fVersionMinor;
   jsonAnchor["formatVersionPatch"] = fVersionPatch;
   jsonAnchor["urlTemplate"] = fUrlTemplate;
   jsonAnchor["cloneTemplate"] = fCloneTemplate;
   jsonAnchor["headerObjId"] = fHeaderObjId;
   jsonAnchor["headerOffset"] = fHeaderOffset;
   jsonAnchor["nBytesHeader"] = fNBytesHeader;
   jsonAnchor["lenHeader"] = fLenHeader;
   jsonAnchor["footerObjId"] = fFooterObjId;
   jsonAnchor["footerOffset"] = fFooterOffset;
   jsonAnchor["nBytesFooter"] = fNBytesFooter;
   jsonAnchor["lenFooter"] = fLenFooter;

   auto canonical = jsonAnchor.dump(-1);
   jsonAnchor["checksum"] = XXH3_64bits(canonical.data(), canonical.size());
   return jsonAnchor.dump(2);
}

/// Construct an anchor from a JSON string.
/// The anchor version is checked first; if it does not match the current version,
/// parsing fails immediately. All remaining fields are extracted with jsonAnchor.at()
/// which throws on missing keys or type mismatches.
ROOT::RResult<ROOT::Experimental::Internal::RNTupleAnchorS3>
ROOT::Experimental::Internal::RNTupleAnchorS3::CreateFromJSON(const std::string &json)
{
   nlohmann::json jsonAnchor;
   try {
      jsonAnchor = nlohmann::json::parse(json);
   } catch (const nlohmann::json::parse_error &e) {
      return R__FAIL("cannot parse S3 anchor JSON: " + std::string(e.what()));
   }

   RNTupleAnchorS3 anchor;

   try {
      anchor.fVersionAnchor = jsonAnchor.at("anchorVersion").get<std::uint32_t>();
   } catch (const nlohmann::json::exception &e) {
      return R__FAIL("missing or invalid 'anchorVersion' in S3 anchor: " + std::string(e.what()));
   }

   if (anchor.fVersionAnchor != RNTupleAnchorS3().fVersionAnchor)
      return R__FAIL("unsupported S3 anchor version: " + std::to_string(anchor.fVersionAnchor));

   try {
      anchor.fVersionEpoch = jsonAnchor.at("formatVersionEpoch").get<std::uint16_t>();
      anchor.fVersionMajor = jsonAnchor.at("formatVersionMajor").get<std::uint16_t>();
      anchor.fVersionMinor = jsonAnchor.at("formatVersionMinor").get<std::uint16_t>();
      anchor.fVersionPatch = jsonAnchor.at("formatVersionPatch").get<std::uint16_t>();
      anchor.fUrlTemplate = jsonAnchor.at("urlTemplate").get<std::string>();
      anchor.fCloneTemplate = jsonAnchor.at("cloneTemplate").get<std::string>();
      anchor.fHeaderObjId = jsonAnchor.at("headerObjId").get<std::uint64_t>();
      anchor.fHeaderOffset = jsonAnchor.at("headerOffset").get<std::uint64_t>();
      anchor.fNBytesHeader = jsonAnchor.at("nBytesHeader").get<std::uint64_t>();
      anchor.fLenHeader = jsonAnchor.at("lenHeader").get<std::uint64_t>();
      anchor.fFooterObjId = jsonAnchor.at("footerObjId").get<std::uint64_t>();
      anchor.fFooterOffset = jsonAnchor.at("footerOffset").get<std::uint64_t>();
      anchor.fNBytesFooter = jsonAnchor.at("nBytesFooter").get<std::uint64_t>();
      anchor.fLenFooter = jsonAnchor.at("lenFooter").get<std::uint64_t>();
   } catch (const nlohmann::json::exception &e) {
      return R__FAIL("missing or invalid field in S3 anchor: " + std::string(e.what()));
   }

   if (!jsonAnchor.contains("checksum"))
      return R__FAIL("missing 'checksum' field in S3 anchor");

   std::uint64_t storedChecksum;
   try {
      storedChecksum = jsonAnchor.at("checksum").get<std::uint64_t>();
   } catch (const nlohmann::json::exception &e) {
      return R__FAIL("invalid 'checksum' field in S3 anchor: " + std::string(e.what()));
   }

   jsonAnchor.erase("checksum");
   auto canonical = jsonAnchor.dump(-1);
   auto computedChecksum = XXH3_64bits(canonical.data(), canonical.size());

   if (storedChecksum != computedChecksum)
      return R__FAIL("S3 anchor checksum mismatch");

   return anchor;
}

// S3 URI parsing

ROOT::RResult<std::string> ROOT::Experimental::Internal::ParseS3Url(std::string_view uri)
{
   const std::string uriStr(uri);

   // The base URL is a plain bucket/path prefix (MakeObjectUrl() appends "/<id>") and S3 authentication
   // comes from the environment via SigV4, not from the URL. Reject embedded userinfo, query strings,
   // and fragments rather than silently mishandling them.
   if (uriStr.find_first_of("@?#") != std::string::npos)
      return R__FAIL("S3 URI must not contain userinfo ('@'), a query ('?') or a fragment ('#'): " + uriStr);

   // The dedicated ntpl+s3 scheme marks an RNTuple stored natively as S3 objects, distinguishing it
   // from a ROOT file stored on S3 (which is opened through the S3 handler for s3:// URLs). Use
   // ntpl+s3+https:// in production; ntpl+s3+http:// targets local/testing endpoints such as MinIO and
   // transmits data unencrypted. The scheme is matched case-insensitively (RFC 3986), but the host,
   // bucket and key are kept verbatim because they are case-sensitive.
   std::string schemeLower;
   for (std::size_t i = 0; i < uriStr.size() && i < std::strlen("ntpl+s3+https://"); ++i)
      schemeLower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(uriStr[i]))));

   std::string httpScheme;
   std::size_t schemeLen = 0;
   if (ROOT::StartsWith(schemeLower, "ntpl+s3+https://")) {
      httpScheme = "https";
      schemeLen = std::strlen("ntpl+s3+https://");
   } else if (ROOT::StartsWith(schemeLower, "ntpl+s3+http://")) {
      httpScheme = "http";
      schemeLen = std::strlen("ntpl+s3+http://");
   } else {
      return R__FAIL("invalid S3 URI (expected ntpl+s3+http:// or ntpl+s3+https://): " + uriStr);
   }

   std::string hostAndPath = uriStr.substr(schemeLen);
   // Drop trailing slashes so MakeObjectUrl() never produces "//" in an object key and the anchor key
   // (the base URL itself) is not left ending in '/'.
   while (!hostAndPath.empty() && hostAndPath.back() == '/')
      hostAndPath.pop_back();

   // There must be a host after the scheme; check for emptiness once the trailing slashes are removed,
   // so a URI that is only slashes after the scheme (e.g. "ntpl+s3+http:///") is rejected as well.
   if (hostAndPath.empty())
      return R__FAIL("S3 URI has no host: " + uriStr);

   return httpScheme + "://" + hostAndPath;
}

// RPageSinkS3

ROOT::Experimental::Internal::RPageSinkS3::RPageSinkS3(std::string_view ntupleName, std::string_view uri,
                                                       const ROOT::RNTupleWriteOptions &options)
   : RPageSinkS3(ntupleName, ParseS3Url(uri).Unwrap(), options, RFromBaseUrl{})
{
}

ROOT::Experimental::Internal::RPageSinkS3::RPageSinkS3(std::string_view ntupleName, std::string_view baseUrl,
                                                       const ROOT::RNTupleWriteOptions &options, RFromBaseUrl)
   : RPagePersistentSink(ntupleName, options), fBaseUrl(baseUrl), fConnection(fBaseUrl)
{
   static std::once_flag once;
   std::call_once(once, []() {
      R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "The S3 backend is experimental and still under development. "
                                                  << "Do not store real data with this version of RNTuple!";
   });
   fConnection.SetCredentialsFromEnvironment();
   EnableDefaultMetrics("RPageSinkS3");
}

ROOT::Experimental::Internal::RPageSinkS3::~RPageSinkS3() = default;

std::string ROOT::Experimental::Internal::RPageSinkS3::MakeObjectUrl(std::uint64_t objId) const
{
   return fBaseUrl + "/" + std::to_string(objId);
}

void ROOT::Experimental::Internal::RPageSinkS3::PutObject(const std::string &url, const unsigned char *data,
                                                          std::size_t size)
{
   // All objects share fConnection; retarget it to this object's URL (via SetUrl) so curl can keep
   // the connection alive across uploads to the same host.
   fConnection.SetUrl(url).ThrowOnError();
   auto status = fConnection.SendPutReq(data, size);
   if (!status)
      throw ROOT::RException(R__FAIL("S3 PUT failed for " + url + ": " + status.fStatusMsg));
}

void ROOT::Experimental::Internal::RPageSinkS3::InitImpl(unsigned char *serializedHeader, std::uint32_t length)
{
   // fAnchor.fUrlTemplate keeps its default ("${baseurl}/${objid}").

   auto zipBuffer = MakeUninitArray<unsigned char>(length);
   auto szZipHeader =
      RNTupleCompressor::Zip(serializedHeader, length, GetWriteOptions().GetCompression(), zipBuffer.get());

   const auto headerObjId = fObjectId++;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      PutObject(MakeObjectUrl(headerObjId), zipBuffer.get(), szZipHeader);
   }

   fAnchor.fHeaderObjId = headerObjId;
   fAnchor.fHeaderOffset = 0;
   fAnchor.fNBytesHeader = szZipHeader;
   fAnchor.fLenHeader = length;
}

ROOT::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkS3::CommitSealedPageImpl(ROOT::DescriptorId_t,
                                                                const RPageStorage::RSealedPage &sealedPage)
{
   // Mode B: one S3 object per sealed page, located by a kTypeObject64 locator
   const auto pageObjId = fObjectId++;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      PutObject(MakeObjectUrl(pageObjId), reinterpret_cast<const unsigned char *>(sealedPage.GetBuffer()),
                sealedPage.GetBufferSize());
   }

   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeObject64);
   result.SetNBytesOnStorage(sealedPage.GetDataSize());
   result.SetPosition(ROOT::RNTupleLocatorObject64{pageObjId});
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.GetBufferSize());
   fNBytesCurrentCluster += sealedPage.GetBufferSize();
   return result;
}

std::uint64_t ROOT::Experimental::Internal::RPageSinkS3::StageClusterImpl()
{
   return std::exchange(fNBytesCurrentCluster, 0);
}

ROOT::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkS3::CommitClusterGroupImpl(unsigned char *serializedPageList,
                                                                  std::uint32_t length)
{
   auto bufPageListZip = MakeUninitArray<unsigned char>(length);
   auto szPageListZip =
      RNTupleCompressor::Zip(serializedPageList, length, GetWriteOptions().GetCompression(), bufPageListZip.get());

   const auto objId = fObjectId++;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      PutObject(MakeObjectUrl(objId), bufPageListZip.get(), szPageListZip);
   }

   RNTupleLocator result;
   result.SetType(RNTupleLocator::kTypeObject64);
   result.SetNBytesOnStorage(szPageListZip);
   result.SetPosition(ROOT::RNTupleLocatorObject64{objId});
   fCounters->fSzWritePayload.Add(static_cast<std::int64_t>(szPageListZip));
   return result;
}

ROOT::Internal::RNTupleLink
ROOT::Experimental::Internal::RPageSinkS3::CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length)
{
   auto bufFooterZip = MakeUninitArray<unsigned char>(length);
   auto szFooterZip =
      RNTupleCompressor::Zip(serializedFooter, length, GetWriteOptions().GetCompression(), bufFooterZip.get());

   const auto footerObjId = fObjectId++;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      PutObject(MakeObjectUrl(footerObjId), bufFooterZip.get(), szFooterZip);
   }

   fAnchor.fFooterObjId = footerObjId;
   fAnchor.fFooterOffset = 0;
   fAnchor.fNBytesFooter = szFooterZip;
   fAnchor.fLenFooter = length;

   // Upload the anchor LAST: once it exists at the base URL, a reader can assume the whole ntuple
   // is complete. Never upload it before all other objects are in place.
   const auto anchorJson = fAnchor.ToJSON();
   PutObject(fBaseUrl, reinterpret_cast<const unsigned char *>(anchorJson.data()), anchorJson.size());

   // An S3 ntuple is self-locating: its anchor always lives at the base URL, so there is no anchor
   // link to hand back here.
   return {};
}

std::unique_ptr<ROOT::Internal::RPageSink>
ROOT::Experimental::Internal::RPageSinkS3::CloneAsHidden(std::string_view name,
                                                         const ROOT::RNTupleWriteOptions &opts) const
{
   // Resolve the clone template so the hidden ntuple's objects and anchor live under a sub-prefix
   // that cannot collide with the main ntuple's numeric object keys.
   std::string cloneBaseUrl = fAnchor.fCloneTemplate;

   auto pos = cloneBaseUrl.find("${baseurl}");
   if (pos != std::string::npos)
      cloneBaseUrl.replace(pos, std::strlen("${baseurl}"), fBaseUrl);

   pos = cloneBaseUrl.find("${name}");
   if (pos != std::string::npos)
      cloneBaseUrl.replace(pos, std::strlen("${name}"), name);

   return std::unique_ptr<ROOT::Internal::RPageSink>(new RPageSinkS3(name, cloneBaseUrl, opts, RFromBaseUrl{}));
}

// RPageSourceS3

namespace {
/// How much of the anchor to ask for without knowing its size. Its schema is fixed and only the two URL
/// templates vary in length, so 8 KiB covers any realistic anchor in a single request; a larger one is
/// re-read at kMaxAnchorSize.
constexpr std::uint64_t kAnchorReadSize = 8 * 1024;

/// Ceiling on the anchor size, since the length ultimately comes from the remote endpoint and is used
/// as an allocation size.
constexpr std::uint64_t kMaxAnchorSize = 1024 * 1024;

/// The binary format stores an envelope's size in 48 bits (the low 16 of the same word hold the envelope
/// type), so nothing a writer can produce exceeds this. Sizes are clamped here before being summed into
/// a buffer length, no matter how large RNTupleReadOptions::GetMaxEnvelopeSize() is set.
constexpr std::uint64_t kFormatMaxEnvelopeSize = 1ULL << 48;
} // anonymous namespace

ROOT::Experimental::Internal::RPageSourceS3::RPageSourceS3(std::string_view ntupleName, std::string_view uri,
                                                           const ROOT::RNTupleReadOptions &options)
   : RPageSourceS3(ntupleName, ParseS3Url(uri).Unwrap(), options, RFromBaseUrl{})
{
}

ROOT::Experimental::Internal::RPageSourceS3::RPageSourceS3(std::string_view ntupleName, std::string_view baseUrl,
                                                           const ROOT::RNTupleReadOptions &options, RFromBaseUrl)
   : RPageSource(ntupleName, options), fBaseUrl(baseUrl), fMainConnection(fBaseUrl), fClusterConnection(fBaseUrl)
{
   fMainConnection.SetCredentialsFromEnvironment();
   fClusterConnection.SetCredentialsFromEnvironment();
   // Enable the counters before any I/O happens, so that LoadStructureImpl() can already use them.
   EnableDefaultMetrics("RPageSourceS3");
}

ROOT::Experimental::Internal::RPageSourceS3::~RPageSourceS3()
{
   // The cluster pool's I/O thread calls back into this source, so it has to be joined before any of
   // the members below (in particular fMainConnection) are destroyed.
   StopClusterPoolBackgroundThread();
}

std::string ROOT::Experimental::Internal::RPageSourceS3::MakeObjectUrl(std::uint64_t objId) const
{
   // ${baseurl} was already substituted once, when the anchor was read; only the per-object part is
   // left to do here.
   std::string url = fResolvedUrlTemplate;

   const auto pos = url.find("${objid}");
   if (pos != std::string::npos)
      url.replace(pos, std::strlen("${objid}"), std::to_string(objId));

   return url;
}

void ROOT::Experimental::Internal::RPageSourceS3::GetObject(ROOT::Internal::RCurlConnection &connection,
                                                            const std::string &url, unsigned char *buffer,
                                                            std::size_t size)
{
   // Retarget the connection rather than opening a new one, so curl keeps it alive across objects. The
   // caller picks which one: a curl easy handle serves one thread at a time.
   connection.SetUrl(url).ThrowOnError();

   // One range from offset 0 covers the whole object. RCurlConnection also copes with a server that
   // ignores the range and replies 200 with the full body.
   ROOT::Internal::RCurlConnection::RUserRange range;
   range.fDestination = buffer;
   range.fOffset = 0;
   range.fLength = size;

   auto status = connection.SendRangesReq(1, &range);
   if (!status)
      throw ROOT::RException(R__FAIL("S3 GET failed for " + url + ": " + status.fStatusMsg));
   if (range.fNBytesRecv != size) {
      throw ROOT::RException(R__FAIL("S3 GET for " + url + " returned " + std::to_string(range.fNBytesRecv) +
                                     " bytes, expected " + std::to_string(size)));
   }
}

std::size_t ROOT::Experimental::Internal::RPageSourceS3::GetShortObject(ROOT::Internal::RCurlConnection &connection,
                                                                        const std::string &url, unsigned char *buffer,
                                                                        std::size_t size)
{
   connection.SetUrl(url).ThrowOnError();

   // A range that runs past the end of an object is valid and simply returns fewer bytes, so asking for
   // more than the object holds is how its size is learned in a single request. Unlike GetObject(),
   // a short read is the expected outcome rather than an error.
   ROOT::Internal::RCurlConnection::RUserRange range;
   range.fDestination = buffer;
   range.fOffset = 0;
   range.fLength = size;

   auto status = connection.SendRangesReq(1, &range);
   if (!status)
      throw ROOT::RException(R__FAIL("S3 GET failed for " + url + ": " + status.fStatusMsg));
   return range.fNBytesRecv;
}

void ROOT::Experimental::Internal::RPageSourceS3::LoadStructureImpl()
{
   // The anchor lives at the base URL, and its size is not known upfront. Rather than a HEAD to learn the
   // size followed by a GET to fetch it, over-request in a single GET: the reply is truncated at the end
   // of the object, so the number of bytes received is the anchor size.
   auto anchorBuffer = MakeUninitArray<unsigned char>(kAnchorReadSize);
   auto anchorSize = GetShortObject(fMainConnection, fBaseUrl, anchorBuffer.get(), kAnchorReadSize);

   if (anchorSize == kAnchorReadSize) {
      // The anchor filled the request, so it may have been cut short. Re-read it once at the largest size
      // we are prepared to accept.
      anchorBuffer = MakeUninitArray<unsigned char>(kMaxAnchorSize);
      anchorSize = GetShortObject(fMainConnection, fBaseUrl, anchorBuffer.get(), kMaxAnchorSize);
      if (anchorSize == kMaxAnchorSize) {
         throw ROOT::RException(R__FAIL("S3 anchor at " + fBaseUrl + " is implausibly large (>= " +
                                        std::to_string(kMaxAnchorSize) + " bytes); refusing to read it"));
      }
   }

   const std::string anchorJson(reinterpret_cast<const char *>(anchorBuffer.get()), anchorSize);
   auto anchorResult = RNTupleAnchorS3::CreateFromJSON(anchorJson);
   if (!anchorResult)
      throw ROOT::RException(R__FORWARD_ERROR(anchorResult));
   fAnchor = anchorResult.Inspect();

   // ${baseurl} is fixed for the lifetime of this source, so resolve it once here rather than on every
   // object lookup. Mirrors the substitution in RPageSinkS3::CloneAsHidden().
   fResolvedUrlTemplate = fAnchor.GetUrlTemplate();
   if (const auto pos = fResolvedUrlTemplate.find("${baseurl}"); pos != std::string::npos)
      fResolvedUrlTemplate.replace(pos, std::strlen("${baseurl}"), fBaseUrl);

   // The envelope sizes are used to size a buffer and to offset into it, so bound them before summing.
   // The limit is a read option because a data set with an unusually large schema may legitimately need
   // it raised; it is capped at what the format itself can express, so that the sum cannot overflow no
   // matter what the caller asks for.
   const auto maxEnvelopeSize = std::min(fOptions.GetMaxEnvelopeSize(), kFormatMaxEnvelopeSize);
   if (fAnchor.GetNBytesHeader() > maxEnvelopeSize || fAnchor.GetNBytesFooter() > maxEnvelopeSize ||
       fAnchor.GetLenHeader() > maxEnvelopeSize || fAnchor.GetLenFooter() > maxEnvelopeSize) {
      throw ROOT::RException(R__FAIL("S3 anchor at " + fBaseUrl + " declares a header or footer larger than " +
                                     std::to_string(maxEnvelopeSize) +
                                     " bytes; raise RNTupleReadOptions::SetMaxEnvelopeSize() if this is genuine"));
   }

   fDescriptorBuilder.SetVersion(fAnchor.GetVersionEpoch(), fAnchor.GetVersionMajor(), fAnchor.GetVersionMinor(),
                                 fAnchor.GetVersionPatch());
   fDescriptorBuilder.SetOnDiskHeaderSize(fAnchor.GetNBytesHeader());
   fDescriptorBuilder.AddToOnDiskFooterSize(fAnchor.GetNBytesFooter());

   // Reserve enough space for the compressed and the uncompressed header/footer (see AttachImpl)
   const auto bufSize =
      fAnchor.GetNBytesHeader() + fAnchor.GetNBytesFooter() + std::max(fAnchor.GetLenHeader(), fAnchor.GetLenFooter());
   fStructureBuffer.fBuffer = MakeUninitArray<unsigned char>(bufSize);
   fStructureBuffer.fPtrHeader = fStructureBuffer.fBuffer.get();
   fStructureBuffer.fPtrFooter = fStructureBuffer.fBuffer.get() + fAnchor.GetNBytesHeader();

   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      GetObject(fMainConnection, MakeObjectUrl(fAnchor.GetHeaderObjId()),
                reinterpret_cast<unsigned char *>(fStructureBuffer.fPtrHeader), fAnchor.GetNBytesHeader());
      GetObject(fMainConnection, MakeObjectUrl(fAnchor.GetFooterObjId()),
                reinterpret_cast<unsigned char *>(fStructureBuffer.fPtrFooter), fAnchor.GetNBytesFooter());
      // One range read each for the anchor, the header and the footer.
      fCounters->fNRead.Add(3);
   }
}

ROOT::RNTupleDescriptor ROOT::Experimental::Internal::RPageSourceS3::AttachImpl()
{
   auto unzipBuf = reinterpret_cast<unsigned char *>(fStructureBuffer.fPtrFooter) + fAnchor.GetNBytesFooter();

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrHeader, fAnchor.GetNBytesHeader(), fAnchor.GetLenHeader(), unzipBuf);
   RNTupleSerializer::DeserializeHeader(unzipBuf, fAnchor.GetLenHeader(), fDescriptorBuilder);

   // The name locates nothing here, but comparing it asserts that the URL points at the data set the
   // caller meant. An empty name opts out.
   const auto &storedName = fDescriptorBuilder.GetDescriptor().GetName();
   if (!fNTupleName.empty() && storedName != fNTupleName) {
      throw ROOT::RException(
         R__FAIL("the S3 ntuple at " + fBaseUrl + " is named '" + storedName + "', not '" + fNTupleName + "'"));
   }

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrFooter, fAnchor.GetNBytesFooter(), fAnchor.GetLenFooter(), unzipBuf);
   RNTupleSerializer::DeserializeFooter(unzipBuf, fAnchor.GetLenFooter(), fDescriptorBuilder);

   return fDescriptorBuilder.MoveDescriptor();
}

void ROOT::Experimental::Internal::RPageSourceS3::LoadPageListImpl(const RNTupleLocator &locator, unsigned char *buffer)
{
   const auto objId = locator.GetPosition<ROOT::RNTupleLocatorObject64>().GetLocation();
   GetObject(fMainConnection, MakeObjectUrl(objId), buffer, locator.GetNBytesOnStorage());
}

void ROOT::Experimental::Internal::RPageSourceS3::LoadSealedPageImpl(const RNTupleLocator &locator,
                                                                     RSealedPage &sealedPage)
{
   Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
   const auto objId = locator.GetPosition<ROOT::RNTupleLocatorObject64>().GetLocation();
   // The const_cast is safe: the buffer belongs to the caller, who provided it for us to fill.
   auto buffer = static_cast<unsigned char *>(const_cast<void *>(sealedPage.GetBuffer()));
   GetObject(fMainConnection, MakeObjectUrl(objId), buffer, sealedPage.GetBufferSize());
}

std::vector<std::unique_ptr<ROOT::Internal::RCluster>>
ROOT::Experimental::Internal::RPageSourceS3::LoadClusters(std::span<ROOT::Internal::RCluster::RKey> clusterKeys)
{
   /// Where one page of a cluster lives in S3 and how much space it takes in the cluster buffer.
   struct RS3SealedPageLocator {
      ROOT::DescriptorId_t fColumnId = 0;
      ROOT::NTupleSize_t fPageNo = 0;
      std::uint64_t fObjId = 0;
      std::uint64_t fBufferSize = 0; ///< page payload + checksum (if available)
   };

   fCounters->fNClusterLoaded.Add(clusterKeys.size());

   std::vector<std::unique_ptr<ROOT::Internal::RCluster>> clusters;

   for (const auto &clusterKey : clusterKeys) {
      std::vector<RS3SealedPageLocator> onDiskPages;
      std::uint64_t clusterBufSize = 0;

      auto pageZeroMap = std::make_unique<ROOT::Internal::ROnDiskPageMap>();
      PrepareLoadCluster(
         clusterKey, *pageZeroMap,
         [&](ROOT::DescriptorId_t physicalColumnId, ROOT::NTupleSize_t pageNo,
             const ROOT::RClusterDescriptor::RPageInfo &pageInfo) {
            const auto &pageLocator = pageInfo.GetLocator();
            const auto objId = pageLocator.GetPosition<ROOT::RNTupleLocatorObject64>().GetLocation();
            const auto pageBufferSize = pageLocator.GetNBytesOnStorage() + pageInfo.HasChecksum() * kNBytesPageChecksum;
            onDiskPages.emplace_back(RS3SealedPageLocator{physicalColumnId, pageNo, objId, pageBufferSize});
            clusterBufSize += pageBufferSize;
         });

      auto clusterBuffer = new unsigned char[clusterBufSize];
      auto pageMap =
         std::make_unique<ROOT::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(clusterBuffer));

      {
         // One GET per page in Mode B; batching them concurrently is a follow-up. This runs on the
         // cluster pool's I/O thread, hence fClusterConnection: the caller may be using fMainConnection.
         Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
         auto pageBuffer = clusterBuffer;
         for (const auto &sealedLoc : onDiskPages) {
            ROOT::Internal::ROnDiskPage::Key key(sealedLoc.fColumnId, sealedLoc.fPageNo);
            pageMap->Register(key, ROOT::Internal::ROnDiskPage(pageBuffer, sealedLoc.fBufferSize));
            GetObject(fClusterConnection, MakeObjectUrl(sealedLoc.fObjId), pageBuffer, sealedLoc.fBufferSize);
            pageBuffer += sealedLoc.fBufferSize;
         }
      }

      fCounters->fNPageRead.Add(onDiskPages.size());
      fCounters->fSzReadPayload.Add(clusterBufSize);
      fCounters->fNRead.Add(onDiskPages.size());

      auto cluster = std::make_unique<ROOT::Internal::RCluster>(clusterKey.fClusterId);
      cluster->Adopt(std::move(pageMap));
      cluster->Adopt(std::move(pageZeroMap));
      for (auto colId : clusterKey.fPhysicalColumnSet)
         cluster->SetColumnAvailable(colId);
      clusters.emplace_back(std::move(cluster));
   }

   return clusters;
}

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::Internal::RPageSourceS3::CloneImpl() const
{
   // The clone opens its own connections, so clones can be read from concurrently.
   auto clone = std::unique_ptr<RPageSourceS3>(new RPageSourceS3(fNTupleName, fBaseUrl, fOptions, RFromBaseUrl{}));
   // Carry the anchor over: an attached clone skips LoadStructureImpl(), so without it MakeObjectUrl()
   // would resolve against a default-constructed template.
   clone->fAnchor = fAnchor;
   clone->fResolvedUrlTemplate = fResolvedUrlTemplate;
   return clone;
}

void ROOT::Experimental::Internal::RPageSourceS3::LoadStreamerInfo()
{
   R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "S3-backed sources have no associated StreamerInfo to load.";
}

std::unique_ptr<ROOT::Internal::RPageSource>
ROOT::Experimental::Internal::RPageSourceS3::OpenWithDifferentAnchor(const ROOT::Internal::RNTupleLink &,
                                                                     const ROOT::RNTupleReadOptions &)
{
   // An S3 ntuple is self-locating (its anchor is at the base URL), so there is no anchor link to follow
   // within the same storage container the way the file backend does.
   throw ROOT::RException(R__FAIL("OpenWithDifferentAnchor is not implemented for the S3 backend"));
}
