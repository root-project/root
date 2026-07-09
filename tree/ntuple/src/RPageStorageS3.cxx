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

#include <ROOT/RCurlConnection.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/StringUtils.hxx>

#include <nlohmann/json.hpp>

#include <cctype>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>

using ROOT::Internal::MakeUninitArray;
using ROOT::Internal::RNTupleCompressor;

/// Field-by-field equality check across all 14 anchor members.
/// Used to verify round-trip correctness in tests.
bool ROOT::Experimental::Internal::RNTupleAnchorS3::operator==(const RNTupleAnchorS3 &other) const
{
   return fVersionAnchor == other.fVersionAnchor && fVersionEpoch == other.fVersionEpoch &&
          fVersionMajor == other.fVersionMajor && fVersionMinor == other.fVersionMinor &&
          fVersionPatch == other.fVersionPatch && fUrlTemplate == other.fUrlTemplate &&
          fHeaderObjId == other.fHeaderObjId && fHeaderOffset == other.fHeaderOffset &&
          fNBytesHeader == other.fNBytesHeader && fLenHeader == other.fLenHeader &&
          fFooterObjId == other.fFooterObjId && fFooterOffset == other.fFooterOffset &&
          fNBytesFooter == other.fNBytesFooter && fLenFooter == other.fLenFooter;
}

/// Serialize the anchor to a pretty-printed JSON string (2-space indent).
/// nlohmann/json handles type conversion, string escaping, and uint64 precision.
/// The output is suitable for direct upload to S3 as the anchor object.
std::string ROOT::Experimental::Internal::RNTupleAnchorS3::ToJSON() const
{
   nlohmann::json jsonAnchor;
   jsonAnchor["anchorVersion"] = fVersionAnchor;
   jsonAnchor["formatVersionEpoch"] = fVersionEpoch;
   jsonAnchor["formatVersionMajor"] = fVersionMajor;
   jsonAnchor["formatVersionMinor"] = fVersionMinor;
   jsonAnchor["formatVersionPatch"] = fVersionPatch;
   jsonAnchor["urlTemplate"] = fUrlTemplate;
   jsonAnchor["headerObjId"] = fHeaderObjId;
   jsonAnchor["headerOffset"] = fHeaderOffset;
   jsonAnchor["nBytesHeader"] = fNBytesHeader;
   jsonAnchor["lenHeader"] = fLenHeader;
   jsonAnchor["footerObjId"] = fFooterObjId;
   jsonAnchor["footerOffset"] = fFooterOffset;
   jsonAnchor["nBytesFooter"] = fNBytesFooter;
   jsonAnchor["lenFooter"] = fLenFooter;
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
   // The hidden (attribute-set) ntuple is stored under a reserved "_clone" sub-prefix so its objects and
   // anchor can never collide with the main ntuple's numeric object keys ($baseurl/0, $baseurl/1, ...).
   std::string cloneBaseUrl = fBaseUrl + "/_clone/" + std::string(name);
   return std::unique_ptr<ROOT::Internal::RPageSink>(new RPageSinkS3(name, cloneBaseUrl, opts, RFromBaseUrl{}));
}
