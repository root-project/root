/// \file ntuple_storage_s3.cxx
/// \author Jas Mehta <jasmehta805@gmail.com>
/// \date 2026-06-01
/// \brief Unit tests for the S3 storage backend components (anchor serialization).

#include "ntuple_test.hxx"
#include <ROOT/RPageStorageS3.hxx>
#include <ROOT/TestSupport.hxx>

#include "TServerSocket.h"
#include "TSocket.h"
#include "TSystem.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using RNTupleAnchorS3 = ROOT::Experimental::Internal::RNTupleAnchorS3;

// ==================== RNTupleAnchorS3 Tests ====================

TEST(RNTupleAnchorS3, RoundTrip)
{
   RNTupleAnchorS3 orig;
   orig.fVersionAnchor = 0;
   orig.fVersionEpoch = 1;
   orig.fVersionMajor = 0;
   orig.fVersionMinor = 2;
   orig.fVersionPatch = 0;
   orig.fUrlTemplate = "https://bucket.s3.us-east-1.amazonaws.com/data/${objid}";
   orig.fHeaderObjId = 1;
   orig.fHeaderOffset = 0;
   orig.fNBytesHeader = 1200;
   orig.fLenHeader = 4096;
   orig.fFooterObjId = 42;
   orig.fFooterOffset = 0;
   orig.fNBytesFooter = 800;
   orig.fLenFooter = 2048;

   auto json = orig.ToJSON();
   EXPECT_FALSE(json.empty());

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();

   EXPECT_EQ(orig, parsed);
   EXPECT_EQ(0u, parsed.fVersionAnchor);
   EXPECT_EQ(1u, parsed.fVersionEpoch);
   EXPECT_EQ(0u, parsed.fVersionMajor);
   EXPECT_EQ(2u, parsed.fVersionMinor);
   EXPECT_EQ(0u, parsed.fVersionPatch);
   EXPECT_EQ("https://bucket.s3.us-east-1.amazonaws.com/data/${objid}", parsed.fUrlTemplate);
   EXPECT_EQ(1u, parsed.fHeaderObjId);
   EXPECT_EQ(0u, parsed.fHeaderOffset);
   EXPECT_EQ(1200u, parsed.fNBytesHeader);
   EXPECT_EQ(4096u, parsed.fLenHeader);
   EXPECT_EQ(42u, parsed.fFooterObjId);
   EXPECT_EQ(0u, parsed.fFooterOffset);
   EXPECT_EQ(800u, parsed.fNBytesFooter);
   EXPECT_EQ(2048u, parsed.fLenFooter);
}

TEST(RNTupleAnchorS3, UnsupportedVersion)
{
   std::string json = R"({"anchorVersion": 99, "formatVersionEpoch": 1})";
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   EXPECT_FALSE(bool(result));
}

TEST(RNTupleAnchorS3, MissingField)
{
   // Valid JSON but missing footer fields
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
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "https://example.com/path/with\"quotes/${objid}";
   orig.fHeaderObjId = 1;
   orig.fNBytesHeader = 100;
   orig.fLenHeader = 200;
   orig.fFooterObjId = 2;
   orig.fNBytesFooter = 50;
   orig.fLenFooter = 100;

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ(orig.fUrlTemplate, result.Inspect().fUrlTemplate);
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

TEST(RNTupleAnchorS3, ExtraFieldsIgnored)
{
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "${baseurl}/${objid}";
   orig.fHeaderObjId = 1;
   orig.fNBytesHeader = 500;
   orig.fLenHeader = 1000;
   orig.fFooterObjId = 10;
   orig.fNBytesFooter = 300;
   orig.fLenFooter = 600;

   auto json = orig.ToJSON();
   // Inject an unknown field before the closing brace
   auto pos = json.rfind('}');
   json.insert(pos, ",\n  \"future_field\": 999");

   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ(orig, result.Inspect());
}

TEST(RNTupleAnchorS3, LargeObjectIds)
{
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "${baseurl}/${objid}";
   orig.fHeaderObjId = 4294967296ULL; // 2^32 -- beyond uint32 range
   orig.fHeaderOffset = 0;
   orig.fNBytesHeader = 100;
   orig.fLenHeader = 200;
   orig.fFooterObjId = 9007199254740993ULL; // 2^53 + 1 -- beyond double precision
   orig.fFooterOffset = 1099511627776ULL;   // 2^40
   orig.fNBytesFooter = 50;
   orig.fLenFooter = 100;

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(4294967296ULL, parsed.fHeaderObjId);
   EXPECT_EQ(9007199254740993ULL, parsed.fFooterObjId);
   EXPECT_EQ(1099511627776ULL, parsed.fFooterOffset);
}

TEST(RNTupleAnchorS3, DefaultValues)
{
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "${baseurl}/${objid}";

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(0u, parsed.fHeaderObjId);
   EXPECT_EQ(0u, parsed.fNBytesHeader);
   EXPECT_EQ(0u, parsed.fLenHeader);
   EXPECT_EQ(0u, parsed.fFooterObjId);
   EXPECT_EQ(0u, parsed.fNBytesFooter);
   EXPECT_EQ(0u, parsed.fLenFooter);
}

TEST(RNTupleAnchorS3, UrlTemplateDefault)
{
   // A freshly constructed anchor carries the writer's default object-naming scheme.
   EXPECT_EQ("${baseurl}/${objid}", RNTupleAnchorS3().fUrlTemplate);
}

TEST(RNTupleAnchorS3, BackslashInUrl)
{
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "C:\\Users\\data\\${objid}";
   orig.fHeaderObjId = 1;
   orig.fNBytesHeader = 100;
   orig.fLenHeader = 200;
   orig.fFooterObjId = 2;
   orig.fNBytesFooter = 50;
   orig.fLenFooter = 100;

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ("C:\\Users\\data\\${objid}", result.Inspect().fUrlTemplate);
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
   RNTupleAnchorS3 a;
   a.fUrlTemplate = "${baseurl}/${objid}";
   a.fHeaderObjId = 1;
   a.fNBytesHeader = 100;
   a.fLenHeader = 200;
   a.fFooterObjId = 2;
   a.fNBytesFooter = 50;
   a.fLenFooter = 100;

   RNTupleAnchorS3 b = a;
   EXPECT_EQ(a, b);

   b.fHeaderObjId = 99;
   EXPECT_FALSE(a == b);
}

TEST(RNTupleAnchorS3, ToJSONProducesValidJson)
{
   RNTupleAnchorS3 anchor;
   anchor.fUrlTemplate = "${baseurl}/${objid}";
   anchor.fHeaderObjId = 5;
   anchor.fNBytesHeader = 500;
   anchor.fLenHeader = 1000;
   anchor.fFooterObjId = 10;
   anchor.fNBytesFooter = 300;
   anchor.fLenFooter = 600;

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
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "https://example.com/path\twith\ttabs\nand\nnewlines/${objid}";
   orig.fHeaderObjId = 1;
   orig.fNBytesHeader = 100;
   orig.fLenHeader = 200;
   orig.fFooterObjId = 2;
   orig.fNBytesFooter = 50;
   orig.fLenFooter = 100;

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
   EXPECT_EQ(orig.fUrlTemplate, result.Inspect().fUrlTemplate);
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
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "";
   orig.fHeaderObjId = 1;
   orig.fNBytesHeader = 100;
   orig.fLenHeader = 200;
   orig.fFooterObjId = 2;
   orig.fNBytesFooter = 50;
   orig.fLenFooter = 100;

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   EXPECT_EQ("", result.Inspect().fUrlTemplate);
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
   RNTupleAnchorS3 orig;
   orig.fUrlTemplate = "${baseurl}/${objid}";
   orig.fHeaderObjId = UINT64_MAX;
   orig.fHeaderOffset = UINT64_MAX;
   orig.fNBytesHeader = UINT64_MAX;
   orig.fLenHeader = UINT64_MAX;
   orig.fFooterObjId = UINT64_MAX;
   orig.fFooterOffset = UINT64_MAX;
   orig.fNBytesFooter = UINT64_MAX;
   orig.fLenFooter = UINT64_MAX;

   auto json = orig.ToJSON();
   auto result = RNTupleAnchorS3::CreateFromJSON(json);
   ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
   const auto &parsed = result.Inspect();
   EXPECT_EQ(UINT64_MAX, parsed.fHeaderObjId);
   EXPECT_EQ(UINT64_MAX, parsed.fHeaderOffset);
   EXPECT_EQ(UINT64_MAX, parsed.fNBytesHeader);
   EXPECT_EQ(UINT64_MAX, parsed.fLenHeader);
   EXPECT_EQ(UINT64_MAX, parsed.fFooterObjId);
   EXPECT_EQ(UINT64_MAX, parsed.fFooterOffset);
   EXPECT_EQ(UINT64_MAX, parsed.fNBytesFooter);
   EXPECT_EQ(UINT64_MAX, parsed.fLenFooter);
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
