/// \file ntuple_storage_s3.cxx
/// \author Jas Mehta <jasmehta805@gmail.com>
/// \date 2026-06-01
/// \brief Unit tests for the S3 storage backend components (anchor serialization).

#include "ntuple_test.hxx"
#include <ROOT/RPageStorageS3.hxx>

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
