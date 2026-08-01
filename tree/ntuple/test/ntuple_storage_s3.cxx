/// \file ntuple_storage_s3.cxx
/// \author Jas Mehta <jasmehta805@gmail.com>
/// \date 2026-06-01
/// \brief Unit tests for the S3 storage backend components (anchor serialization, write and read path).

#include "ntuple_test.hxx"
#include <ROOT/RPageStorageS3.hxx>
#include <ROOT/StringUtils.hxx>
#include <ROOT/TestSupport.hxx>

#include "TServerSocket.h"
#include "TSocket.h"
#include "TSystem.h"

#include <nlohmann/json.hpp>
#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using RNTupleAnchorS3 = ROOT::Experimental::Internal::RNTupleAnchorS3;

namespace {

/// Build a JSON object with all required anchor fields at their defaults.
nlohmann::json MakeAnchorJson()
{
   nlohmann::json jsonAnchor;
   jsonAnchor["anchorVersion"] = 0;
   jsonAnchor["formatVersionEpoch"] = ROOT::RNTuple::kVersionEpoch;
   jsonAnchor["formatVersionMajor"] = ROOT::RNTuple::kVersionMajor;
   jsonAnchor["formatVersionMinor"] = ROOT::RNTuple::kVersionMinor;
   jsonAnchor["formatVersionPatch"] = ROOT::RNTuple::kVersionPatch;
   jsonAnchor["urlTemplate"] = "${baseurl}/${objid}";
   jsonAnchor["cloneTemplate"] = "${baseurl}/_clone/${name}";
   jsonAnchor["headerObjId"] = 0;
   jsonAnchor["headerOffset"] = 0;
   jsonAnchor["nBytesHeader"] = 0;
   jsonAnchor["lenHeader"] = 0;
   jsonAnchor["footerObjId"] = 0;
   jsonAnchor["footerOffset"] = 0;
   jsonAnchor["nBytesFooter"] = 0;
   jsonAnchor["lenFooter"] = 0;
   return jsonAnchor;
}

/// Build JSON, add checksum, parse to anchor.
RNTupleAnchorS3 MakeAnchor(const nlohmann::json &jsonAnchor)
{
   auto canonicalJson = jsonAnchor.dump(-1);
   auto checksum = XXH3_64bits(canonicalJson.data(), canonicalJson.size());
   nlohmann::json jsonWithChecksum = jsonAnchor;
   jsonWithChecksum["checksum"] = checksum;
   auto result = RNTupleAnchorS3::CreateFromJSON(jsonWithChecksum.dump());
   return result.Inspect();
}

} // anonymous namespace

// ==================== RNTupleAnchorS3 Tests ====================

TEST(RNTupleAnchorS3, RoundTrip)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["formatVersionEpoch"] = 1;
   jsonAnchor["formatVersionMajor"] = 0;
   jsonAnchor["formatVersionMinor"] = 2;
   jsonAnchor["formatVersionPatch"] = 0;
   jsonAnchor["urlTemplate"] = "https://bucket.s3.us-east-1.amazonaws.com/data/${objid}";
   jsonAnchor["cloneTemplate"] = "${baseurl}/clones/${name}";
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["headerOffset"] = 0;
   jsonAnchor["nBytesHeader"] = 1200;
   jsonAnchor["lenHeader"] = 4096;
   jsonAnchor["footerObjId"] = 42;
   jsonAnchor["footerOffset"] = 0;
   jsonAnchor["nBytesFooter"] = 800;
   jsonAnchor["lenFooter"] = 2048;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   EXPECT_FALSE(json.empty());

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();

   EXPECT_EQ(orig, parsed);
   EXPECT_EQ(0u, parsed.GetVersionAnchor());
   EXPECT_EQ(1u, parsed.GetVersionEpoch());
   EXPECT_EQ(0u, parsed.GetVersionMajor());
   EXPECT_EQ(2u, parsed.GetVersionMinor());
   EXPECT_EQ(0u, parsed.GetVersionPatch());
   EXPECT_EQ("https://bucket.s3.us-east-1.amazonaws.com/data/${objid}", parsed.GetUrlTemplate());
   EXPECT_EQ("${baseurl}/clones/${name}", parsed.GetCloneTemplate());
   EXPECT_EQ(1u, parsed.GetHeaderObjId());
   EXPECT_EQ(0u, parsed.GetHeaderOffset());
   EXPECT_EQ(1200u, parsed.GetNBytesHeader());
   EXPECT_EQ(4096u, parsed.GetLenHeader());
   EXPECT_EQ(42u, parsed.GetFooterObjId());
   EXPECT_EQ(0u, parsed.GetFooterOffset());
   EXPECT_EQ(800u, parsed.GetNBytesFooter());
   EXPECT_EQ(2048u, parsed.GetLenFooter());
}

TEST(RNTupleAnchorS3, UnsupportedVersion)
{
   std::string json = R"({"anchorVersion": 99, "formatVersionEpoch": 1})";
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, MissingField)
{
   // Valid JSON but missing footer fields and cloneTemplate
   std::string json = R"({
     "anchorVersion": 0,
     "formatVersionEpoch": 1,
     "formatVersionMajor": 0,
     "formatVersionMinor": 2,
     "formatVersionPatch": 0,
     "urlTemplate": "test",
     "headerObjId": 1,
     "headerOffset": 0,
     "nBytesHeader": 100,
     "lenHeader": 200
   })";
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, SpecialCharsInUrl)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["urlTemplate"] = "https://example.com/path/with\"quotes/${objid}";
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 2;
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ(orig.GetUrlTemplate(), result.Inspect().GetUrlTemplate());
}

TEST(RNTupleAnchorS3, MalformedJson)
{
   auto result = RNTupleAnchorS3::CreateFromJSON("not json at all");
   EXPECT_FALSE(bool(result));

   result = RNTupleAnchorS3::CreateFromJSON("{incomplete");
   EXPECT_FALSE(bool(result));

   result = RNTupleAnchorS3::CreateFromJSON("");
   EXPECT_FALSE(bool(result));

   result = RNTupleAnchorS3::CreateFromJSON("   ");
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, ExtraFieldsDetectedByChecksum)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 500;
   jsonAnchor["lenHeader"] = 1000;
   jsonAnchor["footerObjId"] = 10;
   jsonAnchor["nBytesFooter"] = 300;
   jsonAnchor["lenFooter"] = 600;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   // Inject an unknown field; the checksum no longer matches because the reader hashes
   // all non-checksum fields, including unknown ones added by tampering.
   auto pos = json.rfind('}');
   json.insert(pos, ",\n  \"future_field\": 999");

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, LargeObjectIds)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 4294967296ULL; // 2^32 -- beyond uint32 range
   jsonAnchor["headerOffset"] = 0;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 9007199254740993ULL; // 2^53 + 1 -- beyond double precision
   jsonAnchor["footerOffset"] = 1099511627776ULL;   // 2^40
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(4294967296ULL, parsed.GetHeaderObjId());
   EXPECT_EQ(9007199254740993ULL, parsed.GetFooterObjId());
   EXPECT_EQ(1099511627776ULL, parsed.GetFooterOffset());
}

TEST(RNTupleAnchorS3, DefaultValues)
{
   auto orig = MakeAnchor(MakeAnchorJson());

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(0u, parsed.GetHeaderObjId());
   EXPECT_EQ(0u, parsed.GetNBytesHeader());
   EXPECT_EQ(0u, parsed.GetLenHeader());
   EXPECT_EQ(0u, parsed.GetFooterObjId());
   EXPECT_EQ(0u, parsed.GetNBytesFooter());
   EXPECT_EQ(0u, parsed.GetLenFooter());
}

TEST(RNTupleAnchorS3, UrlTemplateDefault)
{
   // A freshly constructed anchor carries the writer's default object-naming scheme.
   EXPECT_EQ("${baseurl}/${objid}", RNTupleAnchorS3().GetUrlTemplate());
}

TEST(RNTupleAnchorS3, BackslashInUrl)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["urlTemplate"] = "C:\\Users\\data\\${objid}";
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 2;
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ("C:\\Users\\data\\${objid}", result.Inspect().GetUrlTemplate());
}

TEST(RNTupleAnchorS3, MissingAnchorVersion)
{
   std::string json = R"({
     "formatVersionEpoch": 1,
     "formatVersionMajor": 0,
     "formatVersionMinor": 0,
     "formatVersionPatch": 0,
     "urlTemplate": "test",
     "headerObjId": 1,
     "headerOffset": 0,
     "nBytesHeader": 100,
     "lenHeader": 200,
     "footerObjId": 2,
     "footerOffset": 0,
     "nBytesFooter": 50,
     "lenFooter": 100
   })";
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, Equality)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 2;
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto a = MakeAnchor(jsonAnchor);
   auto b = MakeAnchor(jsonAnchor);
   EXPECT_EQ(a, b);

   auto jsonAnchor2 = jsonAnchor;
   jsonAnchor2["headerObjId"] = 99;
   auto c = MakeAnchor(jsonAnchor2);
   EXPECT_NE(a, c);

   auto jsonAnchor3 = jsonAnchor;
   jsonAnchor3["cloneTemplate"] = "${baseurl}/other/${name}";
   auto d = MakeAnchor(jsonAnchor3);
   EXPECT_NE(a, d);
}

TEST(RNTupleAnchorS3, ToJSONProducesValidJson)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 5;
   jsonAnchor["nBytesHeader"] = 500;
   jsonAnchor["lenHeader"] = 1000;
   jsonAnchor["footerObjId"] = 10;
   jsonAnchor["nBytesFooter"] = 300;
   jsonAnchor["lenFooter"] = 600;

   auto anchor = MakeAnchor(jsonAnchor);
   auto json = anchor.ToJSON();

   // Basic structural checks for valid JSON
   EXPECT_EQ('{', json.front());
   EXPECT_EQ('}', json.back());
   EXPECT_NE(std::string::npos, json.find("\"anchorVersion\""));
   EXPECT_NE(std::string::npos, json.find("\"formatVersionEpoch\""));
   EXPECT_NE(std::string::npos, json.find("\"urlTemplate\""));
   EXPECT_NE(std::string::npos, json.find("\"headerObjId\""));
   EXPECT_NE(std::string::npos, json.find("\"footerObjId\""));
   EXPECT_NE(std::string::npos, json.find("\"nBytesHeader\""));
   EXPECT_NE(std::string::npos, json.find("\"lenHeader\""));
   EXPECT_NE(std::string::npos, json.find("\"nBytesFooter\""));
   EXPECT_NE(std::string::npos, json.find("\"lenFooter\""));
}

TEST(RNTupleAnchorS3, NewlinesAndTabsInUrl)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["urlTemplate"] = "https://example.com/path\twith\ttabs\nand\nnewlines/${objid}";
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 2;
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   // Verify the JSON doesn't contain literal tabs/newlines inside the string value
   // (they should be escaped as \t and \n)
   auto urlPos = json.find("\"urlTemplate\"");
   ASSERT_NE(std::string::npos, urlPos);
   auto colonPos = json.find(':', urlPos);
   auto openQuote = json.find('"', colonPos + 1);
   auto closeQuote = openQuote + 1;
   while (closeQuote < json.size() && json[closeQuote] != '"') {
      if (json[closeQuote] == '\\')
         ++closeQuote; // skip escaped char
      ++closeQuote;
   }
   std::string rawUrlValue = json.substr(openQuote + 1, closeQuote - openQuote - 1);
   // Should contain escaped sequences, not literal control chars
   EXPECT_NE(std::string::npos, rawUrlValue.find("\\t"));
   EXPECT_NE(std::string::npos, rawUrlValue.find("\\n"));

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ(orig.GetUrlTemplate(), result.Inspect().GetUrlTemplate());
}

TEST(RNTupleAnchorS3, WrongFieldType)
{
   // anchorVersion is a string instead of an integer
   std::string json = R"({
     "anchorVersion": "not_a_number",
     "formatVersionEpoch": 1
   })";
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, EmptyUrlTemplate)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["urlTemplate"] = "";
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;
   jsonAnchor["footerObjId"] = 2;
   jsonAnchor["nBytesFooter"] = 50;
   jsonAnchor["lenFooter"] = 100;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ("", result.Inspect().GetUrlTemplate());
}

TEST(RNTupleAnchorS3, JsonArray)
{
   // Valid JSON but wrong type (array, not object)
   auto result = RNTupleAnchorS3::CreateFromJSON("[1, 2, 3]");
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, MaxUint64Values)
{
   // Test boundary values for all uint64 fields
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = UINT64_MAX;
   jsonAnchor["headerOffset"] = UINT64_MAX;
   jsonAnchor["nBytesHeader"] = UINT64_MAX;
   jsonAnchor["lenHeader"] = UINT64_MAX;
   jsonAnchor["footerObjId"] = UINT64_MAX;
   jsonAnchor["footerOffset"] = UINT64_MAX;
   jsonAnchor["nBytesFooter"] = UINT64_MAX;
   jsonAnchor["lenFooter"] = UINT64_MAX;

   auto orig = MakeAnchor(jsonAnchor);
   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(UINT64_MAX, parsed.GetHeaderObjId());
   EXPECT_EQ(UINT64_MAX, parsed.GetHeaderOffset());
   EXPECT_EQ(UINT64_MAX, parsed.GetNBytesHeader());
   EXPECT_EQ(UINT64_MAX, parsed.GetLenHeader());
   EXPECT_EQ(UINT64_MAX, parsed.GetFooterObjId());
   EXPECT_EQ(UINT64_MAX, parsed.GetFooterOffset());
   EXPECT_EQ(UINT64_MAX, parsed.GetNBytesFooter());
   EXPECT_EQ(UINT64_MAX, parsed.GetLenFooter());
}

// ==================== Checksum Tests ====================

TEST(RNTupleAnchorS3, ChecksumMismatch)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 42;
   jsonAnchor["nBytesHeader"] = 100;
   jsonAnchor["lenHeader"] = 200;

   auto anchor = MakeAnchor(jsonAnchor);
   auto json = anchor.ToJSON();

   // Corrupt a data field while keeping the old checksum
   auto pos = json.find("\"nBytesHeader\": 100");
   ASSERT_NE(std::string::npos, pos);
   json.replace(pos, std::strlen("\"nBytesHeader\": 100"), "\"nBytesHeader\": 999");

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, MissingChecksumRejected)
{
   // An anchor without a checksum field must be rejected.
   std::string json = R"({
     "anchorVersion": 0,
     "formatVersionEpoch": 0,
     "formatVersionMajor": 1,
     "formatVersionMinor": 0,
     "formatVersionPatch": 0,
     "urlTemplate": "${baseurl}/${objid}",
     "cloneTemplate": "${baseurl}/_clone/${name}",
     "headerObjId": 0,
     "headerOffset": 0,
     "nBytesHeader": 0,
     "lenHeader": 0,
     "footerObjId": 0,
     "footerOffset": 0,
     "nBytesFooter": 0,
     "lenFooter": 0
   })";

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, ChecksumDeterministic)
{
   auto jsonAnchor = MakeAnchorJson();
   jsonAnchor["headerObjId"] = 1;
   jsonAnchor["footerObjId"] = 5;
   jsonAnchor["nBytesHeader"] = 80;
   jsonAnchor["lenHeader"] = 100;
   jsonAnchor["nBytesFooter"] = 150;
   jsonAnchor["lenFooter"] = 200;

   auto anchor = MakeAnchor(jsonAnchor);
   auto json1 = anchor.ToJSON();
   auto json2 = anchor.ToJSON();
   EXPECT_EQ(json1, json2) << "ToJSON must produce identical output for the same data";

   auto result = RNTupleAnchorS3::CreateFromJSON(json1);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   auto json3 = result.Inspect().ToJSON();
   EXPECT_EQ(json1, json3);
}

TEST(RNTupleAnchorS3, WrongChecksumType)
{
   RNTupleAnchorS3 anchor;
   auto json = anchor.ToJSON();

   // Replace the numeric checksum value with a string
   auto pos = json.find("\"checksum\":");
   ASSERT_NE(std::string::npos, pos);
   auto valStart = json.find(':', pos) + 1;
   while (valStart < json.size() && json[valStart] == ' ')
      ++valStart;
   auto valEnd = valStart;
   while (valEnd < json.size() && json[valEnd] != '\n' && json[valEnd] != ',')
      ++valEnd;
   json.replace(valStart, valEnd - valStart, " \"not_a_number\"");

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

// ==================== ParseS3Url Tests ====================

using ROOT::Experimental::Internal::ParseS3Url;

TEST(RPageSinkS3, ParseS3UrlHttp)
{
   EXPECT_EQ("http://localhost:9000/mybucket/path", ParseS3Url("ntpl+s3+http://localhost:9000/mybucket/path").Unwrap());
}

TEST(RPageSinkS3, ParseS3UrlHttps)
{
   EXPECT_EQ("https://s3.cern.ch/mybucket/path", ParseS3Url("ntpl+s3+https://s3.cern.ch/mybucket/path").Unwrap());
}

TEST(RPageSinkS3, ParseS3UrlInvalid)
{
   // Non-S3 schemes, the bare s3:// (left to ROOT's S3 file handler), the old s3+http(s):// forms,
   // and ntpl+s3:// without a transport all yield an error result (so Unwrap() throws).
   EXPECT_THROW(ParseS3Url("http://example.com").Unwrap(), ROOT::RException);
   EXPECT_THROW(ParseS3Url("daos://pool/container").Unwrap(), ROOT::RException);
   EXPECT_THROW(ParseS3Url("").Unwrap(), ROOT::RException);
   EXPECT_THROW(ParseS3Url("s3://bucket/path").Unwrap(), ROOT::RException);
   EXPECT_THROW(ParseS3Url("s3+https://host/bucket/path").Unwrap(), ROOT::RException);
   EXPECT_THROW(ParseS3Url("ntpl+s3://host/bucket/path").Unwrap(), ROOT::RException);
   // A scheme followed only by slashes has no host either (trailing slashes are stripped before the
   // emptiness check).
   EXPECT_THROW(ParseS3Url("ntpl+s3+http:///").Unwrap(), ROOT::RException);
}

TEST(RPageSinkS3, ParseS3UrlTrailingSlash)
{
   // A trailing slash must not leak into object keys (MakeObjectUrl appends "/<id>") or the anchor key.
   EXPECT_EQ("http://localhost:9000/bucket/path", ParseS3Url("ntpl+s3+http://localhost:9000/bucket/path/").Unwrap());
   EXPECT_EQ("https://s3.cern.ch/bucket", ParseS3Url("ntpl+s3+https://s3.cern.ch/bucket/").Unwrap());
}

TEST(RPageSinkS3, ParseS3UrlCaseInsensitiveScheme)
{
   // The scheme is matched case-insensitively; the host/bucket/key case is preserved verbatim.
   EXPECT_EQ("http://Host:9000/MyBucket/Path", ParseS3Url("NTPL+S3+HTTP://Host:9000/MyBucket/Path").Unwrap());
   EXPECT_EQ("https://Host/MyBucket/Path", ParseS3Url("Ntpl+S3+Https://Host/MyBucket/Path").Unwrap());
}

TEST(RPageSinkS3, ParseS3UrlAwsAndCeph)
{
   // AWS (any region, path-style or virtual-hosted) and Ceph/MinIO endpoints all work through the
   // explicit ntpl+s3+https:// form: the user supplies the full host, which is passed through verbatim.
   EXPECT_EQ("https://s3.eu-west-1.amazonaws.com/bucket/data", // AWS path-style, regional
             ParseS3Url("ntpl+s3+https://s3.eu-west-1.amazonaws.com/bucket/data").Unwrap());
   EXPECT_EQ("https://bucket.s3.eu-west-1.amazonaws.com/data", // AWS virtual-hosted style
             ParseS3Url("ntpl+s3+https://bucket.s3.eu-west-1.amazonaws.com/data").Unwrap());
   EXPECT_EQ("https://s3.cern.ch/bucket/data", // Ceph RGW (CERN)
             ParseS3Url("ntpl+s3+https://s3.cern.ch/bucket/data").Unwrap());
}

TEST(RPageSinkS3, ParseS3UrlRejectsUnsupportedComponents)
{
   EXPECT_THROW(ParseS3Url("ntpl+s3+https://KEY:SECRET@host/bucket/path").Unwrap(), ROOT::RException); // userinfo
   EXPECT_THROW(ParseS3Url("ntpl+s3+http://host/bucket/path?versionId=1").Unwrap(), ROOT::RException); // query
   EXPECT_THROW(ParseS3Url("ntpl+s3+http://host/bucket/path#section").Unwrap(), ROOT::RException);     // fragment
   EXPECT_THROW(ParseS3Url("ntpl+s3+http://").Unwrap(), ROOT::RException);                             // no host
}

// ==================== RPageSinkS3 Wire-Level Tests (mock HTTP server) ====================

// These tests stand up a loopback TServerSocket and point an RPageSinkS3 at it, so the exact HTTP
// PUT requests the write path emits can be inspected with no live S3 service (they always run in
// CI). The mock-server idiom mirrors net/curl/test/curl_connection.cxx.
namespace {

/// Read one HTTP request (request line + headers + body) from an accepted socket, reply with the
/// given status (e.g. "200 OK"), and return the request-target (the path from the request line).
std::string ServeOneRequest(TSocket *sock, const char *status, std::string &headers, std::string &body)
{
   headers.clear();
   body.clear();

   // Read up to and including the end-of-headers marker, byte by byte.
   const char *eof = "\r\n\r\n";
   const std::size_t eofLen = std::strlen(eof);
   std::size_t nextInEof = 0;
   char c;
   while (sock->RecvRaw(&c, 1) > 0) {
      headers.push_back(c);
      if (c == eof[nextInEof]) {
         if (++nextInEof == eofLen)
            break;
      } else {
         nextInEof = 0;
      }
   }

   std::string lower(headers);
   std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });

   // libcurl uploads with "Expect: 100-continue"; acknowledge before reading the body.
   if (lower.find("expect: 100-continue") != std::string::npos) {
      const char *cont = "HTTP/1.1 100 Continue\r\n\r\n";
      sock->SendRaw(cont, std::strlen(cont));
   }

   std::size_t contentLength = 0;
   if (auto pos = lower.find("content-length: "); pos != std::string::npos) {
      auto valStart = pos + std::strlen("content-length: ");
      auto valEnd = lower.find("\r\n", valStart);
      contentLength = std::stoul(lower.substr(valStart, valEnd - valStart));
   }
   if (contentLength > 0) {
      body.resize(contentLength);
      sock->RecvRaw(&body[0], contentLength);
   }

   // This mock closes the socket after each request, so tell curl not to keep the connection alive
   // (the sink reuses one connection; without this curl could try to reuse a socket we just closed).
   const std::string response =
      std::string("HTTP/1.1 ") + status + "\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
   sock->SendRaw(response.data(), response.size());

   // The request line is "PUT /target HTTP/1.1"; return the middle token.
   std::string target;
   if (auto sp1 = headers.find(' '); sp1 != std::string::npos) {
      if (auto sp2 = headers.find(' ', sp1 + 1); sp2 != std::string::npos)
         target = headers.substr(sp1 + 1, sp2 - sp1 - 1);
   }
   return target;
}

} // anonymous namespace

TEST(RPageSinkS3Wire, WriteIssuesExpectedPuts)
{
   TServerSocket server(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string host = server.GetLocalInetAddress().GetHostAddress();
   const std::string basePath = "/wirebucket/wiretest";
   const std::string uri = "ntpl+s3+http://" + host + ":" + std::to_string(server.GetLocalPort()) + basePath;

   // Dummy credentials so curl signs every PUT (SigV4 Authorization header). The requests only reach
   // the loopback mock server in this test, never a real S3 service.
   gSystem->Setenv("S3_ACCESS_KEY", "dummykey");
   gSystem->Setenv("S3_SECRET_KEY", "dummysecret");
   gSystem->Setenv("S3_REGION", "us-east-1");

   struct Request {
      std::string fPath;
      std::string fHeaders;
      std::string fBody;
   };
   std::vector<Request> requests;

   // The sink reuses one connection, but this mock replies with "Connection: close", so curl opens a
   // fresh connection per object. Serve them on a background thread until the anchor (the request
   // whose target is exactly the base path) arrives last.
   std::thread serverThread([&] {
      for (;;) {
         TSocket *sock = server.Accept();
         if (!sock || sock == reinterpret_cast<TSocket *>(-1))
            break;
         Request req;
         req.fPath = ServeOneRequest(sock, "200 OK", req.fHeaders, req.fBody);
         sock->Close();
         requests.push_back(std::move(req));
         if (requests.back().fPath == basePath)
            break;
      }
   });

   {
      // The sink ctor emits a one-time (std::call_once) experimental warning; allow it. It is
      // optional because it only fires on the first sink construction in the whole process.
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      auto model = ROOT::RNTupleModel::Create();
      auto fldValue = model->MakeField<int>("value");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "wire", uri);
      for (int i = 0; i < 20; ++i) {
         *fldValue = i;
         writer->Fill();
      }
   } // writer destroyed here -> footer + anchor PUTs

   serverThread.join();

   gSystem->Unsetenv("S3_ACCESS_KEY");
   gSystem->Unsetenv("S3_SECRET_KEY");
   gSystem->Unsetenv("S3_REGION");

   // At minimum: header, one page, page list, footer, anchor.
   ASSERT_GE(requests.size(), 5u);

   for (const auto &req : requests) {
      // Every object is uploaded with a SigV4-signed HTTP PUT.
      EXPECT_EQ(0u, req.fHeaders.find("PUT ")) << req.fHeaders.substr(0, 32);
      std::string lower(req.fHeaders);
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });
      EXPECT_NE(std::string::npos, lower.find("authorization: aws4-hmac-sha256"))
         << "no SigV4 Authorization header on " << req.fPath;
   }

   // Object 0 is the header, written first.
   EXPECT_EQ(basePath + "/0", requests.front().fPath);
   // Every request but the last targets a data object at <base>/<id>; the anchor is last, at <base>.
   for (std::size_t i = 0; i + 1 < requests.size(); ++i)
      EXPECT_EQ(0u, requests[i].fPath.rfind(basePath + "/", 0)) << "unexpected object key " << requests[i].fPath;
   EXPECT_EQ(basePath, requests.back().fPath);
   // The anchor body is the JSON document the reader bootstraps from.
   EXPECT_NE(std::string::npos, requests.back().fBody.find("\"footerObjId\""));
   EXPECT_NE(std::string::npos, requests.back().fBody.find("\"urlTemplate\""));
   EXPECT_NE(std::string::npos, requests.back().fBody.find("\"checksum\""));
}

TEST(RPageSinkS3Wire, PutErrorThrows)
{
   TServerSocket server(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string host = server.GetLocalInetAddress().GetHostAddress();
   const std::string uri =
      "ntpl+s3+http://" + host + ":" + std::to_string(server.GetLocalPort()) + "/wirebucket/wireerr";

   gSystem->Setenv("S3_ACCESS_KEY", "dummykey");
   gSystem->Setenv("S3_SECRET_KEY", "dummysecret");
   gSystem->Setenv("S3_REGION", "us-east-1");

   // Reject the first upload (the header, written during writer construction) with 403.
   std::thread serverThread([&] {
      TSocket *sock = server.Accept();
      if (sock && sock != reinterpret_cast<TSocket *>(-1)) {
         std::string headers, body;
         ServeOneRequest(sock, "403 Forbidden", headers, body);
         sock->Close();
      }
   });

   // Allow the one-time (std::call_once) experimental warning the sink ctor may emit; it is optional
   // because it only fires on the first sink construction in the process.
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

   // The header PUT fails, so RPageSinkS3::PutObject throws out of writer construction.
   EXPECT_THROW(
      {
         auto model = ROOT::RNTupleModel::Create();
         model->MakeField<int>("value");
         auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "wire", uri);
      },
      ROOT::RException);

   serverThread.join();

   gSystem->Unsetenv("S3_ACCESS_KEY");
   gSystem->Unsetenv("S3_SECRET_KEY");
   gSystem->Unsetenv("S3_REGION");
}

TEST(RPageSinkS3Wire, CloneAsHiddenWritesUnderClonePrefix)
{
   TServerSocket server(0, false, TServerSocket::kDefaultBacklog, -1, ESocketBindOption::kInaddrLoopback);
   const std::string host = server.GetLocalInetAddress().GetHostAddress();
   const std::string basePath = "/wirebucket/wireclone";
   const std::string uri = "ntpl+s3+http://" + host + ":" + std::to_string(server.GetLocalPort()) + basePath;
   const std::string clonePrefix = basePath + "/_clone/attr";

   gSystem->Setenv("S3_ACCESS_KEY", "dummykey");
   gSystem->Setenv("S3_SECRET_KEY", "dummysecret");
   gSystem->Setenv("S3_REGION", "us-east-1");

   // Capture the target path of every PUT the clone issues. The clone writes its whole ntuple and its
   // anchor is last, targeting exactly the clone prefix -- use that as the stop condition.
   std::vector<std::string> paths;
   std::thread serverThread([&] {
      for (;;) {
         TSocket *sock = server.Accept();
         if (!sock || sock == reinterpret_cast<TSocket *>(-1))
            break;
         std::string headers, body;
         std::string path = ServeOneRequest(sock, "200 OK", headers, body);
         sock->Close();
         paths.push_back(path);
         if (paths.back() == clonePrefix)
            break;
      }
   });

   {
      // The sink ctor emits the one-time (std::call_once) experimental warning; allow it (optional).
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      ROOT::RNTupleWriteOptions opts;
      auto model = ROOT::RNTupleModel::Create();

      // The main sink only acts as the factory for the hidden clone; we drive the clone itself so its
      // PUT targets reveal where CloneAsHidden routes the hidden ntuple.
      ROOT::Experimental::Internal::RPageSinkS3 mainSink("main", uri, opts);
      auto cloneSink = mainSink.CloneAsHidden("attr", opts);
      cloneSink->Init(*model);
      cloneSink->CommitDataset();
   }

   serverThread.join();

   gSystem->Unsetenv("S3_ACCESS_KEY");
   gSystem->Unsetenv("S3_SECRET_KEY");
   gSystem->Unsetenv("S3_REGION");

   ASSERT_FALSE(paths.empty());
   // The clone's own object counter starts at 0, under its reserved sub-prefix.
   EXPECT_EQ(clonePrefix + "/0", paths.front());
   // Every object the clone writes stays under "$baseurl/_clone/attr", so it can never collide with the
   // main ntuple's numeric object keys ($baseurl/0, $baseurl/1, ...).
   for (const auto &p : paths)
      EXPECT_EQ(0u, p.rfind(clonePrefix, 0)) << "clone object escaped the _clone prefix: " << p;
   // The clone's anchor is written last, at exactly the clone's base URL.
   EXPECT_EQ(clonePrefix, paths.back());
}

// ==================== RPageSourceS3 Wire-Level Tests (mock HTTP server) ====================

// The read path needs a mock that answers HEAD and GET, not just PUT, so these tests run against a
// loopback server backed by an in-memory object store: the sink's PUTs populate it and the source's GETs
// read it back. That makes a complete write-then-read round trip possible with no S3 service at all, so
// it runs in CI.
class RPageSourceS3Wire : public ::testing::Test {
protected:
   /// The mock's object store: request path -> object contents.
   std::unordered_map<std::string, std::string> fStore;

   void SetUp() override
   {
      fServer = std::make_unique<TServerSocket>(0, false, TServerSocket::kDefaultBacklog, -1,
                                                ESocketBindOption::kInaddrLoopback);
      fHost = fServer->GetLocalInetAddress().GetHostAddress();
      fPort = fServer->GetLocalPort();

      // Dummy credentials so that curl signs every request (SigV4). They only ever reach the loopback
      // mock server, never a real S3 service.
      gSystem->Setenv("S3_ACCESS_KEY", "dummykey");
      gSystem->Setenv("S3_SECRET_KEY", "dummysecret");
      gSystem->Setenv("S3_REGION", "us-east-1");

      StartServer();
   }

   void TearDown() override
   {
      StopServer();
      gSystem->Unsetenv("S3_ACCESS_KEY");
      gSystem->Unsetenv("S3_SECRET_KEY");
      gSystem->Unsetenv("S3_REGION");
   }

   /// The ntpl+s3 URI addressing `path` on the mock server.
   std::string Uri(const std::string &path) const
   {
      return "ntpl+s3+http://" + fHost + ":" + std::to_string(fPort) + path;
   }

   /// Serve requests until StopServer() is called. Started automatically; call this again after a
   /// StopServer() to resume, which is how a test modifies fStore without racing the server thread.
   void StartServer()
   {
      fDone.store(false);
      fServerThread = std::thread([this] {
         while (!fDone.load()) {
            TSocket *sock = fServer->Accept();
            if (!sock || sock == reinterpret_cast<TSocket *>(-1))
               break;
            if (fDone.load()) {
               sock->Close();
               break;
            }
            ServeOneS3Request(sock);
            sock->Close();
         }
      });
   }

   /// Stop serving and join the thread. Accept() blocks, so this also opens a throw-away connection to
   /// wake it up. Safe to call when the server is already stopped.
   void StopServer()
   {
      if (!fServerThread.joinable())
         return;
      fDone.store(true);
      {
         TSocket wakeUp(fHost.c_str(), fPort);
         wakeUp.Close();
      }
      fServerThread.join();
   }

private:
   std::unique_ptr<TServerSocket> fServer;
   std::string fHost;
   int fPort = 0;
   std::atomic<bool> fDone{false};
   std::thread fServerThread;

   /// Serve one HTTP request against fStore: PUT stores the body under the request path, HEAD answers
   /// with the stored object's size, and GET returns the stored body. Unknown paths get a 404, other
   /// methods a 405.
   void ServeOneS3Request(TSocket *sock)
   {
      // Read up to and including the end-of-headers marker, byte by byte.
      std::string headers;
      const char *eof = "\r\n\r\n";
      const std::size_t eofLen = std::strlen(eof);
      std::size_t nextInEof = 0;
      char c;
      while (sock->RecvRaw(&c, 1) > 0) {
         headers.push_back(c);
         if (c == eof[nextInEof]) {
            if (++nextInEof == eofLen)
               break;
         } else {
            nextInEof = 0;
         }
      }

      // The request line is "METHOD /target HTTP/1.1".
      const auto requestLine = ROOT::Split(headers.substr(0, headers.find("\r\n")), " ", /*skipEmpty=*/true);
      const std::string method = requestLine.size() > 0 ? requestLine[0] : "";
      const std::string path = requestLine.size() > 1 ? requestLine[1] : "";

      std::string lower(headers);
      std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char ch) { return std::tolower(ch); });

      // libcurl uploads with "Expect: 100-continue"; acknowledge before reading the body.
      if (lower.find("expect: 100-continue") != std::string::npos) {
         const char *cont = "HTTP/1.1 100 Continue\r\n\r\n";
         sock->SendRaw(cont, std::strlen(cont));
      }

      std::string body;
      if (auto pos = lower.find("content-length: "); pos != std::string::npos) {
         auto valStart = pos + std::strlen("content-length: ");
         auto valEnd = lower.find("\r\n", valStart);
         const auto contentLength = std::stoul(lower.substr(valStart, valEnd - valStart));
         if (contentLength > 0) {
            body.resize(contentLength);
            sock->RecvRaw(&body[0], contentLength);
         }
      }

      // Every reply announces "Connection: close" because this mock closes the socket after each
      // request; curl then opens a fresh connection instead of reusing one we already closed. The
      // source's GETs carry a Range header, which this mock ignores: it always returns the whole object,
      // which RCurlConnection handles as the "server ignored the range" case.
      std::string response;
      if (method == "PUT") {
         fStore[path] = body;
         response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      } else if (method == "HEAD" || method == "GET") {
         auto it = fStore.find(path);
         if (it == fStore.end()) {
            response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
         } else {
            response = "HTTP/1.1 200 OK\r\nContent-Length: " + std::to_string(it->second.size()) +
                       "\r\nConnection: close\r\n\r\n";
            if (method == "GET")
               response += it->second;
         }
      } else {
         response = "HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      }

      sock->SendRaw(response.data(), response.size());
   }
};

/// Allows the experimental warning the sink emits once per process (std::call_once), which makes it
/// optional: it only fires on the first sink construction.
#define ALLOW_EXPERIMENTAL_WARNING(diags)   \
   ROOT::TestSupport::CheckDiagsRAII diags; \
   diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFull=*/false)

// Reading with the cluster cache on goes through LoadClusters() on the cluster pool's I/O thread; with
// it off, pages are read one at a time through LoadSealedPageImpl() on the calling thread. Both paths
// have to produce the same values, so the round trip is run against each.
class RPageSourceS3ClusterCache : public RPageSourceS3Wire,
                                  public ::testing::WithParamInterface<ROOT::RNTupleReadOptions::EClusterCache> {};

TEST_P(RPageSourceS3ClusterCache, RoundTrip)
{
   const auto uri = Uri("/wirebucket/roundtrip");

   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "wire", uri);
      for (int i = 0; i < 20; ++i) {
         *fldX = i;
         writer->Fill();
      }
   } // writer destroyed here -> footer + anchor PUTs

   ROOT::RNTupleReadOptions options;
   options.SetClusterCache(GetParam());

   auto reader = ROOT::RNTupleReader::Open("wire", uri, options);
   EXPECT_EQ(20u, reader->GetNEntries());

   auto viewX = reader->GetView<int>("x");
   for (int i = 0; i < 20; ++i)
      EXPECT_EQ(i, viewX(i));
}

INSTANTIATE_TEST_SUITE_P(RPageSourceS3Wire, RPageSourceS3ClusterCache,
                         ::testing::Values(ROOT::RNTupleReadOptions::EClusterCache::kOn,
                                           ROOT::RNTupleReadOptions::EClusterCache::kOff),
                         [](const auto &info) {
                            return info.param == ROOT::RNTupleReadOptions::EClusterCache::kOn ? "ClusterCacheOn"
                                                                                              : "ClusterCacheOff";
                         });

TEST_F(RPageSourceS3Wire, RoundTripManyPagesPerCluster)
{
   const auto uri = Uri("/wirebucket/manypages");

   // Cap pages at 256 bytes (64 ints) but leave the cluster size at its default, so one cluster holds
   // many pages from two interleaved columns. LoadClusters() packs them all into a single cluster buffer
   // at successive offsets, which is where an off-by-one in the buffer cursor would show up.
   ROOT::RNTupleWriteOptions writeOptions;
   writeOptions.SetMaxUnzippedPageSize(256);
   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto fldY = model->MakeField<int>("y");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "manypages", uri, writeOptions);
      for (int i = 0; i < 1000; ++i) {
         *fldX = i;
         *fldY = -i;
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("manypages", uri);
   EXPECT_EQ(1000u, reader->GetNEntries());

   // Guard the premise of the test: one cluster, many pages per column.
   const auto &desc = reader->GetDescriptor();
   ASSERT_EQ(1u, desc.GetNClusters());
   const auto columnId = desc.FindPhysicalColumnId(desc.FindFieldId("x"), 0, 0);
   EXPECT_GT(desc.GetClusterDescriptor(0).GetPageRange(columnId).GetPageInfos().size(), 1u);

   auto viewX = reader->GetView<int>("x");
   auto viewY = reader->GetView<int>("y");
   for (int i = 0; i < 1000; ++i) {
      EXPECT_EQ(i, viewX(i));
      EXPECT_EQ(-i, viewY(i));
   }
}

TEST_F(RPageSourceS3Wire, RoundTripManyClusters)
{
   const auto uri = Uri("/wirebucket/manyclusters");

   // A tiny target cluster size produces many clusters, so LoadClusters() iterates over its outer loop.
   ROOT::RNTupleWriteOptions writeOptions;
   writeOptions.SetApproxZippedClusterSize(1024);
   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "manyclusters", uri, writeOptions);
      for (int i = 0; i < 5000; ++i) {
         *fldX = static_cast<float>(i);
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("manyclusters", uri);
   EXPECT_EQ(5000u, reader->GetNEntries());
   ASSERT_GT(reader->GetDescriptor().GetNClusters(), 1u);

   auto viewX = reader->GetView<float>("x");
   for (int i = 0; i < 5000; ++i)
      EXPECT_FLOAT_EQ(static_cast<float>(i), viewX(i));
}

TEST_F(RPageSourceS3Wire, RoundTripManyClusterGroups)
{
   const auto uri = Uri("/wirebucket/clustergroups");

   // Committing a cluster group writes its own page list object, so Attach() has to walk several cluster
   // groups and issue one LoadPageListImpl() per group rather than a single one.
   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "clustergroups", uri);
      for (int group = 0; group < 4; ++group) {
         for (int i = 0; i < 25; ++i) {
            *fldX = group * 25 + i;
            writer->Fill();
         }
         writer->CommitCluster(/*commitClusterGroup=*/true);
      }
   }

   auto reader = ROOT::RNTupleReader::Open("clustergroups", uri);
   EXPECT_EQ(100u, reader->GetNEntries());
   EXPECT_GT(reader->GetDescriptor().GetNClusterGroups(), 1u);

   auto viewX = reader->GetView<int>("x");
   for (int i = 0; i < 100; ++i)
      EXPECT_EQ(i, viewX(i));
}

TEST_F(RPageSourceS3Wire, RoundTripWithoutPageChecksums)
{
   const auto uri = Uri("/wirebucket/nochecksum");

   // Page buffers are sized as payload + HasChecksum() * kNBytesPageChecksum. Every other test covers
   // the checksummed case (the default), so this one covers the other branch of that arithmetic.
   ROOT::RNTupleWriteOptions writeOptions;
   writeOptions.SetEnablePageChecksums(false);
   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "nochecksum", uri, writeOptions);
      for (int i = 0; i < 100; ++i) {
         *fldX = i * 3;
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("nochecksum", uri);
   EXPECT_EQ(100u, reader->GetNEntries());

   auto viewX = reader->GetView<int>("x");
   for (int i = 0; i < 100; ++i)
      EXPECT_EQ(i * 3, viewX(i));
}

TEST_F(RPageSourceS3Wire, CloneReadsFromItsOwnConnection)
{
   const auto uri = Uri("/wirebucket/clone");

   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "clone", uri);
      for (int i = 0; i < 50; ++i) {
         *fldX = i;
         writer->Fill();
      }
   }

   ROOT::Experimental::Internal::RPageSourceS3 source("clone", uri, ROOT::RNTupleReadOptions());
   source.Attach();
   ASSERT_EQ(50u, source.GetNEntries());

   // Clone() copies the descriptor of an attached source, so the clone must come back attached without
   // re-running LoadStructureImpl(), yet still be able to do I/O over its own connections.
   auto clone = source.Clone();
   ASSERT_EQ(50u, clone->GetNEntries());

   ROOT::DescriptorId_t columnId;
   {
      auto descGuard = clone->GetSharedDescriptorGuard();
      columnId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("x"), 0, 0);
   }

   // A null buffer asks only for the size; the second call transfers and verifies the checksum, so
   // reaching the end of this block proves the clone read real bytes from the mock.
   RPageStorage::RSealedPage sealedPage;
   clone->LoadSealedPage(columnId, ROOT::RNTupleLocalIndex(0, 0), sealedPage);
   ASSERT_GT(sealedPage.GetBufferSize(), 0u);

   auto buffer = MakeUninitArray<unsigned char>(sealedPage.GetBufferSize());
   sealedPage.SetBuffer(buffer.get());
   clone->LoadSealedPage(columnId, ROOT::RNTupleLocalIndex(0, 0), sealedPage);
   EXPECT_EQ(50u, sealedPage.GetNElements());
}

TEST_F(RPageSourceS3Wire, HonoursAnchorUrlTemplate)
{
   const std::string basePath = "/wirebucket/template";
   const auto uri = Uri(basePath);

   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "template", uri);
      for (int i = 0; i < 30; ++i) {
         *fldX = i * 7;
         writer->Fill();
      }
   }

   StopServer();

   // Relocate every data object under a "data/" segment and rewrite the anchor to describe the new
   // layout. The writer only ever emits the default template, so this is the only way to prove that the
   // source resolves the stored one instead of assuming "<base>/<objid>".
   std::unordered_map<std::string, std::string> relocated;
   for (const auto &[key, value] : fStore) {
      if (key == basePath)
         continue; // the anchor itself stays at the base path
      ASSERT_EQ(0u, key.rfind(basePath + "/", 0));
      relocated[basePath + "/data/" + key.substr(basePath.size() + 1)] = value;
   }
   ASSERT_FALSE(relocated.empty());

   auto jsonAnchor = nlohmann::json::parse(fStore[basePath]);
   jsonAnchor.erase("checksum");
   jsonAnchor["urlTemplate"] = "${baseurl}/data/${objid}";
   const auto canonicalJson = jsonAnchor.dump(-1);
   jsonAnchor["checksum"] = XXH3_64bits(canonicalJson.data(), canonicalJson.size());

   relocated[basePath] = jsonAnchor.dump(2);
   fStore = std::move(relocated);

   StartServer();

   auto reader = ROOT::RNTupleReader::Open("template", uri);
   EXPECT_EQ(30u, reader->GetNEntries());

   auto viewX = reader->GetView<int>("x");
   for (int i = 0; i < 30; ++i)
      EXPECT_EQ(i * 7, viewX(i));
}

TEST_F(RPageSourceS3Wire, ChecksNTupleName)
{
   const auto uri = Uri("/wirebucket/named");

   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "signal", uri);
      for (int i = 0; i < 10; ++i) {
         *fldX = i;
         writer->Fill();
      }
   }

   // The name locates nothing in S3, but asking for the wrong one usually means the URL is wrong, so it
   // is reported rather than ignored.
   EXPECT_THROW(ROOT::RNTupleReader::Open("background", uri), ROOT::RException);

   // The matching name works, and so does an empty one for a caller that does not know it up front.
   {
      auto reader = ROOT::RNTupleReader::Open("signal", uri);
      EXPECT_EQ(10u, reader->GetNEntries());
   }
   {
      ROOT::Experimental::Internal::RPageSourceS3 source("", uri, ROOT::RNTupleReadOptions());
      source.Attach();
      EXPECT_EQ(10u, source.GetNEntries());
   }
}

TEST_F(RPageSourceS3Wire, MissingPageObjectFails)
{
   const std::string basePath = "/wirebucket/lostpage";
   const auto uri = Uri(basePath);

   {
      ALLOW_EXPERIMENTAL_WARNING(diags);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "lostpage", uri);
      for (int i = 0; i < 20; ++i) {
         *fldX = i;
         writer->Fill();
      }
   }

   // Take the server down before touching the store, so the map is not read and written concurrently.
   StopServer();

   // Object 0 is the header and the anchor lives at the base path, so dropping object 1 leaves the
   // metadata intact: Attach() succeeds and the failure surfaces later, when a page is actually read.
   ASSERT_EQ(1u, fStore.erase(basePath + "/1"));

   StartServer();

   // The cluster cache is turned off deliberately. With it on, the failing GET happens inside
   // LoadClusters() on the cluster pool's I/O thread, and RClusterPool::ExecReadClusters() does not catch
   // exceptions, so the throw would terminate the process instead of reaching the caller. With the cache
   // off the page is read synchronously and the exception propagates as it should.
   ROOT::RNTupleReadOptions readOptions;
   readOptions.SetClusterCache(ROOT::RNTupleReadOptions::EClusterCache::kOff);

   auto reader = ROOT::RNTupleReader::Open("lostpage", uri, readOptions);
   EXPECT_EQ(20u, reader->GetNEntries());
   auto viewX = reader->GetView<int>("x");
   EXPECT_THROW(viewX(0), ROOT::RException);
}

TEST_F(RPageSourceS3Wire, ReadMissingAnchorFails)
{
   // fStore is empty: nothing was ever written under this prefix, so the anchor GET gets a 404.
   EXPECT_THROW(ROOT::RNTupleReader::Open("test", Uri("/wirebucket/missing")), ROOT::RException);
}

// ==================== Integration Tests (credential-gated) ====================

// These run against a real S3 service over https (CERN Ceph, AWS) and are skipped unless S3_ENDPOINT,
// S3_BUCKET, S3_ACCESS_KEY and S3_SECRET_KEY are all set. S3_ENDPOINT is a bare host, without a scheme.
class RPageS3IntegrationTest : public ::testing::Test {
protected:
   std::string fEndpoint;
   std::string fBucket;

   void SetUp() override
   {
      const char *endpoint = gSystem->Getenv("S3_ENDPOINT");
      const char *bucket = gSystem->Getenv("S3_BUCKET");
      const char *accessKey = gSystem->Getenv("S3_ACCESS_KEY");
      const char *secretKey = gSystem->Getenv("S3_SECRET_KEY");
      if (!endpoint || !bucket || !accessKey || !secretKey)
         GTEST_SKIP() << "S3 credentials not set; skipping integration test";
      fEndpoint = endpoint;
      fBucket = bucket;
   }

   std::string MakeUri(const std::string &prefix) const
   {
      return "ntpl+s3+https://" + fEndpoint + "/" + fBucket + "/" + prefix;
   }
};

TEST_F(RPageS3IntegrationTest, RoundTripSimple)
{
   const auto uri = MakeUri("roundtrip_simple");
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<int>("y");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "test", uri);
      for (int i = 0; i < 1000; ++i) {
         *fldX = static_cast<float>(i) * 0.5f;
         *fldY = i * i;
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("test", uri);
   EXPECT_EQ(1000u, reader->GetNEntries());
   auto viewX = reader->GetView<float>("x");
   auto viewY = reader->GetView<int>("y");
   for (int i = 0; i < 1000; ++i) {
      EXPECT_FLOAT_EQ(static_cast<float>(i) * 0.5f, viewX(i));
      EXPECT_EQ(i * i, viewY(i));
   }
}

TEST_F(RPageS3IntegrationTest, RoundTripStrings)
{
   const auto uri = MakeUri("roundtrip_strings");
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      auto model = ROOT::RNTupleModel::Create();
      auto fldName = model->MakeField<std::string>("name");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "strings", uri);
      for (int i = 0; i < 100; ++i) {
         *fldName = "entry_" + std::to_string(i);
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("strings", uri);
   EXPECT_EQ(100u, reader->GetNEntries());
   auto viewName = reader->GetView<std::string>("name");
   for (int i = 0; i < 100; ++i)
      EXPECT_EQ("entry_" + std::to_string(i), viewName(i));
}

TEST_F(RPageS3IntegrationTest, RoundTripEmpty)
{
   // No entries means no pages and no cluster groups; the read path has to cope with that.
   const auto uri = MakeUri("roundtrip_empty");
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "empty", uri);
   }

   auto reader = ROOT::RNTupleReader::Open("empty", uri);
   EXPECT_EQ(0u, reader->GetNEntries());
}

TEST_F(RPageS3IntegrationTest, RoundTripMultipleClusters)
{
   // A small cluster size forces many clusters, so LoadClusters() runs repeatedly (and from the cluster
   // pool's background thread) rather than just once.
   const auto uri = MakeUri("roundtrip_clusters");
   ROOT::RNTupleWriteOptions opts;
   opts.SetApproxZippedClusterSize(1024);
   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.optionalDiag(kWarning, "[ROOT.NTuple]", "experimental", /*matchFullMessage=*/false);

      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "clusters", uri, opts);
      for (int i = 0; i < 10000; ++i) {
         *fldX = static_cast<float>(i);
         writer->Fill();
      }
   }

   auto reader = ROOT::RNTupleReader::Open("clusters", uri);
   EXPECT_EQ(10000u, reader->GetNEntries());
   auto viewX = reader->GetView<float>("x");
   for (int i = 0; i < 10000; ++i)
      EXPECT_FLOAT_EQ(static_cast<float>(i), viewX(i));
}
