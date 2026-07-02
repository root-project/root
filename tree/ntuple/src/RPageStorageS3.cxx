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

#include <nlohmann/json.hpp>

#include <string>

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
