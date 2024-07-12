#include "ntuple_test.hxx"

#include <Byteswap.h>
#include <TVirtualStreamerInfo.h>

TEST(RNTuple, SerializeInt)
{
   std::int32_t value;
   unsigned char buffer[4];

   EXPECT_EQ(4, RNTupleSerializer::SerializeInt32(42, buffer));
   EXPECT_EQ(4, RNTupleSerializer::DeserializeInt32(buffer, value));
   EXPECT_EQ(42, value);

   EXPECT_EQ(4, RNTupleSerializer::SerializeInt32(-1, buffer));
   EXPECT_EQ(4, RNTupleSerializer::DeserializeInt32(buffer, value));
   EXPECT_EQ(-1, value);
}

TEST(RNTuple, SerializeString)
{
   std::string s;
   EXPECT_EQ(4, RNTupleSerializer::SerializeString(s, nullptr));
   s = "xyz";
   unsigned char buffer[8];
   EXPECT_EQ(7, RNTupleSerializer::SerializeString(s, buffer));

   try {
      RNTupleSerializer::DeserializeString(nullptr, 0, s).Unwrap();
      FAIL() << "too small buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   try {
      RNTupleSerializer::DeserializeString(buffer, 6, s).Unwrap();
      FAIL() << "too small buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   EXPECT_EQ(7, RNTupleSerializer::DeserializeString(buffer, 8, s).Unwrap());
   EXPECT_EQ("xyz", s);
   s.clear();
   EXPECT_EQ(4, RNTupleSerializer::SerializeString(s, buffer));
   s = "other content";
   EXPECT_EQ(4, RNTupleSerializer::DeserializeString(buffer, 8, s).Unwrap());
   EXPECT_TRUE(s.empty());
}

TEST(RNTuple, SerializeColumnType)
{
   EColumnType type = EColumnType::kUnknown;
   unsigned char buffer[2];

   try {
      RNTupleSerializer::SerializeColumnType(type, buffer);
      FAIL() << "unexpected column type should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected column type"));
   }

   RNTupleSerializer::SerializeUInt16(5000, buffer);
   try {
      RNTupleSerializer::DeserializeColumnType(buffer, type).Unwrap();
      FAIL() << "unexpected on disk column type should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected on-disk column type"));
   }

   for (int i = 1; i < static_cast<int>(EColumnType::kMax); ++i) {
      RNTupleSerializer::SerializeColumnType(static_cast<EColumnType>(i), buffer);
      RNTupleSerializer::DeserializeColumnType(buffer, type);
      EXPECT_EQ(i, static_cast<int>(type));
   }
}

TEST(RNTuple, SerializeFieldStructure)
{
   ENTupleStructure structure = ENTupleStructure::kInvalid;
   unsigned char buffer[2];

   try {
      RNTupleSerializer::SerializeFieldStructure(structure, buffer);
      FAIL() << "unexpected field structure value should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected field structure"));
   }

   RNTupleSerializer::SerializeUInt16(5000, buffer);
   try {
      RNTupleSerializer::DeserializeFieldStructure(buffer, structure).Unwrap();
      FAIL() << "unexpected on disk field structure value should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected on-disk field structure value"));
   }

   for (int i = 0; i < static_cast<int>(ENTupleStructure::kInvalid); ++i) {
      RNTupleSerializer::SerializeFieldStructure(static_cast<ENTupleStructure>(i), buffer);
      RNTupleSerializer::DeserializeFieldStructure(buffer, structure);
      EXPECT_EQ(i, static_cast<int>(structure));
   }
}

TEST(RNTuple, SerializeExtraTypeInfoId)
{
   EExtraTypeInfoIds id{EExtraTypeInfoIds::kInvalid};

   unsigned char buffer[4];

   try {
      RNTupleSerializer::SerializeExtraTypeInfoId(id, buffer);
      FAIL() << "unexpected field structure value should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unexpected extra type info"));
   }

   RNTupleSerializer::SerializeUInt32(5000, buffer);
   // unexpected on-disk ID should set the output value to "invalid"
   id = EExtraTypeInfoIds::kStreamerInfo;
   RNTupleSerializer::DeserializeExtraTypeInfoId(buffer, id).Unwrap();
   EXPECT_EQ(EExtraTypeInfoIds::kInvalid, id);

   for (int i = 0; i < static_cast<int>(EExtraTypeInfoIds::kInvalid); ++i) {
      RNTupleSerializer::SerializeExtraTypeInfoId(static_cast<EExtraTypeInfoIds>(i), buffer);
      RNTupleSerializer::DeserializeExtraTypeInfoId(buffer, id);
      EXPECT_EQ(i, static_cast<int>(id));
   }
}

TEST(RNTuple, SerializeEnvelope)
{
   try {
      RNTupleSerializer::DeserializeEnvelope(nullptr, 0, 137).Unwrap();
      FAIL() << "too small envelope should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   struct {
      std::uint64_t typeAndSize = 137;
      std::uint64_t payload = 0;
      std::uint64_t xxhash3 = 0;
   } testEnvelope;
#ifndef R__BYTESWAP
   // big endian
   testEnvelope.typeAndSize = RByteSwap<8>::bswap(testEnvelope.typeAndSize);
#endif

   EXPECT_EQ(8u, RNTupleSerializer::SerializeEnvelopePostscript(reinterpret_cast<unsigned char *>(&testEnvelope), 16));
   testEnvelope.xxhash3 = 0;
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope), 137).Unwrap();
      FAIL() << "XxHash-3 mismatch should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("XxHash-3"));
   }
#ifndef R__BYTESWAP
   // big endian
   testEnvelope.typeAndSize = RByteSwap<8>::bswap(137);
#else
   testEnvelope.typeAndSize = 137;
#endif

   EXPECT_EQ(8u, RNTupleSerializer::SerializeEnvelopePostscript(reinterpret_cast<unsigned char *>(&testEnvelope), 16));
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope), 138).Unwrap();
      FAIL() << "unsupported envelope type should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("envelope type mismatch"));
   }
#ifndef R__BYTESWAP
   // big endian
   testEnvelope.typeAndSize = RByteSwap<8>::bswap(137);
#else
   testEnvelope.typeAndSize = 137;
#endif

   EXPECT_EQ(8u, RNTupleSerializer::SerializeEnvelopePostscript(reinterpret_cast<unsigned char *>(&testEnvelope), 16));
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope) - 1, 137).Unwrap();
      FAIL() << "too small envelope buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("envelope buffer size too small"));
   }

#ifndef R__BYTESWAP
   // big endian
   testEnvelope.typeAndSize -= uint64_t(9) << 40;
#else
   testEnvelope.typeAndSize -= 9 << 16;
#endif
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope), 137).Unwrap();
      FAIL() << "too small envelope buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid envelope, too short"));
   }

#ifndef R__BYTESWAP
   // big endian
   testEnvelope.typeAndSize = RByteSwap<8>::bswap(137);
#else
   testEnvelope.typeAndSize = 137;
#endif
   std::uint64_t xxhash3_write;
   RNTupleSerializer::SerializeEnvelopePostscript(reinterpret_cast<unsigned char *>(&testEnvelope), 16, xxhash3_write);
   std::uint64_t xxhash3_read;
   EXPECT_EQ(8u,
             RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope), 137, xxhash3_read).Unwrap());
   EXPECT_EQ(xxhash3_write, xxhash3_read);
}

TEST(RNTuple, SerializeFrame)
{
   std::uint64_t frameSize;
   std::uint32_t nitems;

   try {
      RNTupleSerializer::DeserializeFrameHeader(nullptr, 0, frameSize).Unwrap();
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   unsigned char buffer[14];
   EXPECT_EQ(8u, RNTupleSerializer::SerializeRecordFramePreamble(buffer));
   try {
      RNTupleSerializer::SerializeFramePostscript(buffer, 7);
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(0u, RNTupleSerializer::SerializeFramePostscript(buffer, 10));

   try {
      RNTupleSerializer::DeserializeFrameHeader(buffer, 8, frameSize).Unwrap();
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(8u, RNTupleSerializer::DeserializeFrameHeader(buffer, 10, frameSize, nitems).Unwrap());
   EXPECT_EQ(10u, frameSize);
   EXPECT_EQ(1u, nitems);

   EXPECT_EQ(12u, RNTupleSerializer::SerializeListFramePreamble(2, buffer));
   try {
      RNTupleSerializer::SerializeFramePostscript(buffer, 11);
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(0u, RNTupleSerializer::SerializeFramePostscript(buffer, 14));

   try {
      RNTupleSerializer::DeserializeFrameHeader(buffer, 11, frameSize, nitems).Unwrap();
      FAIL() << "too short list frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(12u, RNTupleSerializer::DeserializeFrameHeader(buffer, 14, frameSize, nitems).Unwrap());
   EXPECT_EQ(14u, frameSize);
   EXPECT_EQ(2u, nitems);
}

TEST(RNTuple, SerializeLongFrame)
{
   std::uint64_t frameSize;
   std::uint32_t nitems;

   const std::size_t payloadSize = 1024 * 1024 * 1024;
   const std::size_t bufSize = payloadSize + 12;

   auto buffer = std::make_unique<char[]>(bufSize);
   buffer[12] = 'x';
   EXPECT_EQ(12u, RNTupleSerializer::SerializeListFramePreamble(2, buffer.get()));
   EXPECT_EQ(0u, RNTupleSerializer::SerializeFramePostscript(buffer.get(), payloadSize + 12));

   try {
      RNTupleSerializer::DeserializeFrameHeader(buffer.get(), payloadSize, frameSize).Unwrap();
      FAIL() << "too small frame should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(12u, RNTupleSerializer::DeserializeFrameHeader(buffer.get(), bufSize, frameSize, nitems).Unwrap());
   EXPECT_EQ(bufSize, frameSize);
   EXPECT_EQ(2u, nitems);
   EXPECT_EQ('x', buffer[12]);
}

TEST(RNTuple, SerializeFeatureFlags)
{
   std::vector<std::uint64_t> flags;
   unsigned char buffer[16];

   EXPECT_EQ(8u, RNTupleSerializer::SerializeFeatureFlags(flags, nullptr));
   EXPECT_EQ(8u, RNTupleSerializer::SerializeFeatureFlags(flags, buffer));
   EXPECT_EQ(8u, RNTupleSerializer::DeserializeFeatureFlags(buffer, 8, flags).Unwrap());
   ASSERT_EQ(1u, flags.size());
   EXPECT_EQ(0, flags[0]);

   flags[0] = 1;
   EXPECT_EQ(8u, RNTupleSerializer::SerializeFeatureFlags(flags, buffer));
   EXPECT_EQ(8u, RNTupleSerializer::DeserializeFeatureFlags(buffer, 8, flags).Unwrap());
   ASSERT_EQ(1u, flags.size());
   EXPECT_EQ(1, flags[0]);

   flags.push_back(std::uint64_t(-2));
   try {
      RNTupleSerializer::SerializeFeatureFlags(flags, buffer);
      FAIL() << "negative feature flag should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("out of bounds"));
   }

   flags[1] = 2;
   EXPECT_EQ(16u, RNTupleSerializer::SerializeFeatureFlags(flags, buffer));
   EXPECT_EQ(16u, RNTupleSerializer::DeserializeFeatureFlags(buffer, 16, flags).Unwrap());
   ASSERT_EQ(2u, flags.size());
   EXPECT_EQ(1, flags[0]);
   EXPECT_EQ(2, flags[1]);

   try {
      RNTupleSerializer::DeserializeFeatureFlags(nullptr, 0, flags).Unwrap();
      FAIL() << "too small buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   try {
      RNTupleSerializer::DeserializeFeatureFlags(buffer, 12, flags).Unwrap();
      FAIL() << "too small buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
}

TEST(RNTuple, SerializeLocator)
{
   unsigned char buffer[16];
   RNTupleLocator locator;
   locator.fPosition = 1U;
   locator.fBytesOnStorage = 2;

   EXPECT_EQ(12u, RNTupleSerializer::SerializeLocator(locator, nullptr));
   EXPECT_EQ(12u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator.fBytesOnStorage = static_cast<std::uint32_t>(-1);
   try {
      RNTupleSerializer::SerializeLocator(locator, buffer);
      FAIL() << "too big locator should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too large"));
   }

   locator = RNTupleLocator{};
   try {
      RNTupleSerializer::DeserializeLocator(buffer, 3, locator).Unwrap();
      FAIL() << "too short locator buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   try {
      RNTupleSerializer::DeserializeLocator(buffer, 11, locator).Unwrap();
      FAIL() << "too short locator buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(12u, RNTupleSerializer::DeserializeLocator(buffer, 12, locator).Unwrap());
   EXPECT_EQ(1u, locator.GetPosition<std::uint64_t>());
   EXPECT_EQ(2u, locator.fBytesOnStorage);
   EXPECT_EQ(RNTupleLocator::kTypeFile, locator.fType);

   locator.fPosition.emplace<std::string>("X");
   locator.fType = RNTupleLocator::kTypeURI;
   EXPECT_EQ(5u, RNTupleSerializer::SerializeLocator(locator, nullptr));
   EXPECT_EQ(5u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator = RNTupleLocator{};
   try {
      RNTupleSerializer::DeserializeLocator(buffer, 4, locator).Unwrap();
      FAIL() << "too short locator buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(5u, RNTupleSerializer::DeserializeLocator(buffer, 5, locator).Unwrap());
   EXPECT_EQ(0u, locator.fBytesOnStorage);
   EXPECT_EQ(RNTupleLocator::kTypeURI, locator.fType);
   EXPECT_EQ("X", locator.GetPosition<std::string>());

   locator.fPosition.emplace<std::string>("abcdefghijkl");
   EXPECT_EQ(16u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator = RNTupleLocator{};
   EXPECT_EQ(16u, RNTupleSerializer::DeserializeLocator(buffer, 16, locator).Unwrap());
   EXPECT_EQ("abcdefghijkl", locator.GetPosition<std::string>());

   locator.fType = RNTupleLocator::kTypeDAOS;
   locator.fPosition.emplace<RNTupleLocatorObject64>(RNTupleLocatorObject64{1337U});
   locator.fBytesOnStorage = 420420U;
   locator.fReserved = 0x5a;
   EXPECT_EQ(16u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator = RNTupleLocator{};
   EXPECT_EQ(16u, RNTupleSerializer::DeserializeLocator(buffer, 16, locator).Unwrap());
   EXPECT_EQ(locator.fType, RNTupleLocator::kTypeDAOS);
   EXPECT_EQ(locator.fBytesOnStorage, 420420U);
   EXPECT_EQ(locator.fReserved, 0x5a);
   EXPECT_EQ(1337U, locator.GetPosition<RNTupleLocatorObject64>().fLocation);

   std::int32_t *head = reinterpret_cast<std::int32_t *>(buffer);
#ifndef R__BYTESWAP
   // on big endian system
   *head = *head | 0x3;
#else
   *head = (0x3 << 24) | *head;
#endif
   try {
      RNTupleSerializer::DeserializeLocator(buffer, 16, locator).Unwrap();
      FAIL() << "unsupported locator type should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported locator type"));
   }
}

TEST(RNTuple, SerializeEnvelopeLink)
{
   RNTupleSerializer::REnvelopeLink link;
   link.fLength = 42;
   link.fLocator.fPosition = 137U;
   link.fLocator.fBytesOnStorage = 7;

   unsigned char buffer[20];
   EXPECT_EQ(20u, RNTupleSerializer::SerializeEnvelopeLink(link, nullptr));
   EXPECT_EQ(20u, RNTupleSerializer::SerializeEnvelopeLink(link, buffer));
   try {
      RNTupleSerializer::DeserializeEnvelopeLink(buffer, 3, link).Unwrap();
      FAIL() << "too short envelope link buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   RNTupleSerializer::REnvelopeLink reconstructedLink;
   EXPECT_EQ(20u, RNTupleSerializer::DeserializeEnvelopeLink(buffer, 20, reconstructedLink).Unwrap());
   EXPECT_EQ(link.fLength, reconstructedLink.fLength);
   EXPECT_EQ(link.fLocator, reconstructedLink.fLocator);
}

TEST(RNTuple, SerializeClusterSummary)
{
   RNTupleSerializer::RClusterSummary summary;
   summary.fFirstEntry = 42;
   summary.fNEntries = 137;

   unsigned char buffer[28];
   ASSERT_EQ(24u, RNTupleSerializer::SerializeClusterSummary(summary, nullptr));
   EXPECT_EQ(24u, RNTupleSerializer::SerializeClusterSummary(summary, buffer));
   RNTupleSerializer::RClusterSummary reco;
   try {
      RNTupleSerializer::DeserializeClusterSummary(buffer, 23, reco).Unwrap();
      FAIL() << "too short cluster summary should fail";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(24u, RNTupleSerializer::DeserializeClusterSummary(buffer, 24, reco).Unwrap());
   EXPECT_EQ(summary.fFirstEntry, reco.fFirstEntry);
   EXPECT_EQ(summary.fNEntries, reco.fNEntries);
   EXPECT_EQ(summary.fColumnGroupID, reco.fColumnGroupID);

   summary.fColumnGroupID = 13;
   ASSERT_EQ(28u, RNTupleSerializer::SerializeClusterSummary(summary, nullptr));
   EXPECT_EQ(28u, RNTupleSerializer::SerializeClusterSummary(summary, buffer));
   try {
      RNTupleSerializer::DeserializeClusterSummary(buffer, 27, reco).Unwrap();
      FAIL() << "too short cluster summary should fail";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(28u, RNTupleSerializer::DeserializeClusterSummary(buffer, 28, reco).Unwrap());
   EXPECT_EQ(summary.fFirstEntry, reco.fFirstEntry);
   EXPECT_EQ(summary.fNEntries, reco.fNEntries);
   EXPECT_EQ(summary.fColumnGroupID, reco.fColumnGroupID);
}

TEST(RNTuple, SerializeClusterGroup)
{
   RNTupleSerializer::RClusterGroup group;
   group.fMinEntry = 7;
   group.fEntrySpan = 84600;
   group.fNClusters = 42;
   group.fPageListEnvelopeLink.fLength = 42;
   group.fPageListEnvelopeLink.fLocator.fPosition = 137U;
   group.fPageListEnvelopeLink.fLocator.fBytesOnStorage = 7;

   unsigned char buffer[52];
   ASSERT_EQ(48u, RNTupleSerializer::SerializeClusterGroup(group, nullptr));
   EXPECT_EQ(48u, RNTupleSerializer::SerializeClusterGroup(group, buffer));
   RNTupleSerializer::RClusterGroup reco;
   try {
      RNTupleSerializer::DeserializeClusterGroup(buffer, 47, reco).Unwrap();
      FAIL() << "too short cluster group should fail";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(48u, RNTupleSerializer::DeserializeClusterGroup(buffer, 48, reco).Unwrap());
   EXPECT_EQ(group.fMinEntry, reco.fMinEntry);
   EXPECT_EQ(group.fEntrySpan, reco.fEntrySpan);
   EXPECT_EQ(group.fNClusters, reco.fNClusters);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLength, reco.fPageListEnvelopeLink.fLength);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.GetPosition<std::uint64_t>(),
             reco.fPageListEnvelopeLink.fLocator.GetPosition<std::uint64_t>());
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fBytesOnStorage, reco.fPageListEnvelopeLink.fLocator.fBytesOnStorage);

   // Test frame evolution
   auto pos = buffer;
   pos += RNTupleSerializer::SerializeRecordFramePreamble(pos);
   pos += RNTupleSerializer::SerializeUInt64(group.fMinEntry, pos);
   pos += RNTupleSerializer::SerializeUInt64(group.fEntrySpan, pos);
   pos += RNTupleSerializer::SerializeUInt32(group.fNClusters, pos);
   pos += RNTupleSerializer::SerializeEnvelopeLink(group.fPageListEnvelopeLink, pos);
   pos += RNTupleSerializer::SerializeUInt16(7, pos);
   pos += RNTupleSerializer::SerializeFramePostscript(buffer, pos - buffer);
   pos += RNTupleSerializer::SerializeUInt16(13, pos);
   EXPECT_EQ(50u, RNTupleSerializer::DeserializeClusterGroup(buffer, 50, reco).Unwrap());
   EXPECT_EQ(group.fMinEntry, reco.fMinEntry);
   EXPECT_EQ(group.fEntrySpan, reco.fEntrySpan);
   EXPECT_EQ(group.fNClusters, reco.fNClusters);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLength, reco.fPageListEnvelopeLink.fLength);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.GetPosition<std::uint64_t>(),
             reco.fPageListEnvelopeLink.fLocator.GetPosition<std::uint64_t>());
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fBytesOnStorage, reco.fPageListEnvelopeLink.fLocator.fBytesOnStorage);
   std::uint16_t remainder;
   RNTupleSerializer::DeserializeUInt16(buffer + 48, remainder);
   EXPECT_EQ(7u, remainder);
   RNTupleSerializer::DeserializeUInt16(buffer + 50, remainder);
   EXPECT_EQ(13u, remainder);
}

TEST(RNTuple, SerializeEmptyHeader)
{
   RNTupleDescriptorBuilder builder;
   builder.SetNTuple("ntpl", "");
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(0)
      .FieldName("")
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());
   auto desc = builder.MoveDescriptor();
   auto context = RNTupleSerializer::SerializeHeader(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto buffer = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeader(buffer.get(), desc);

   RNTupleSerializer::DeserializeHeader(buffer.get(), context.GetHeaderSize(), builder);
}


TEST(RNTuple, SerializeHeader)
{
   RNTupleDescriptorBuilder builder;
   builder.SetNTuple("ntpl", "");
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(0)
      .FieldName("")
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(42)
      .FieldName("pt")
      .Structure(ENTupleStructure::kLeaf)
      .MakeDescriptor()
      .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(24)
                       .FieldName("ptAlias")
                       .Structure(ENTupleStructure::kLeaf)
                       .ProjectionSourceId(42)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(137)
      .FieldName("jet")
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(13)
      .FieldName("eta")
      .Structure(ENTupleStructure::kLeaf)
      .MakeDescriptor()
      .Unwrap());
   builder.AddFieldLink(0, 42);
   builder.AddFieldLink(0, 24);
   builder.AddFieldLink(0, 137);
   builder.AddFieldLink(137, 13);
   builder.AddFieldProjection(42, 24);
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(23)
                        .PhysicalColumnId(23)
                        .FieldId(42)
                        .Type(EColumnType::kReal32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(100)
                        .PhysicalColumnId(23)
                        .FieldId(24)
                        .Type(EColumnType::kReal32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(17)
                        .PhysicalColumnId(17)
                        .FieldId(137)
                        .Type(EColumnType::kIndex32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(40)
                        .PhysicalColumnId(40)
                        .FieldId(137)
                        .Type(EColumnType::kByte)
                        .Index(1)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddExtraTypeInfo(RExtraTypeInfoDescriptorBuilder()
                               .ContentId(EExtraTypeInfoIds::kStreamerInfo)
                               .Content("xyz")
                               .MoveDescriptor()
                               .Unwrap());

   auto desc = builder.MoveDescriptor();
   auto context = RNTupleSerializer::SerializeHeader(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto buffer = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeader(buffer.get(), desc);

   RNTupleSerializer::DeserializeHeader(buffer.get(), context.GetHeaderSize(), builder);

   desc = builder.MoveDescriptor();
   auto ptAliasFieldId = desc.FindFieldId("ptAlias");
   auto colId = desc.FindLogicalColumnId(ptAliasFieldId, 0);
   EXPECT_TRUE(desc.GetColumnDescriptor(colId).IsAliasColumn());
   auto ptFieldId = desc.FindFieldId("pt");
   EXPECT_EQ(desc.FindLogicalColumnId(ptFieldId, 0), desc.GetColumnDescriptor(colId).GetPhysicalId());
   EXPECT_TRUE(desc.GetFieldDescriptor(ptAliasFieldId).IsProjectedField());
   EXPECT_EQ(ptFieldId, desc.GetFieldDescriptor(ptAliasFieldId).GetProjectionSourceId());
   EXPECT_FALSE(desc.GetFieldDescriptor(ptFieldId).IsProjectedField());
   EXPECT_EQ(1u, desc.GetNExtraTypeInfos());
   const auto &extraTypeInfoDesc = *desc.GetExtraTypeInfoIterable().begin();
   EXPECT_EQ(EExtraTypeInfoIds::kStreamerInfo, extraTypeInfoDesc.GetContentId());
   EXPECT_EQ(0u, extraTypeInfoDesc.GetTypeVersionFrom());
   EXPECT_EQ(0u, extraTypeInfoDesc.GetTypeVersionTo());
   EXPECT_TRUE(extraTypeInfoDesc.GetTypeName().empty());
   EXPECT_STREQ("xyz", extraTypeInfoDesc.GetContent().c_str());
}


TEST(RNTuple, SerializeFooter)
{
   RNTupleDescriptorBuilder builder;
   builder.SetNTuple("ntpl", "");
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(0)
      .FieldName("")
      .Structure(ENTupleStructure::kRecord)
      .MakeDescriptor()
      .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
      .FieldId(42)
      .FieldName("tag")
      .Structure(ENTupleStructure::kLeaf)
      .MakeDescriptor()
      .Unwrap());
   builder.AddFieldLink(0, 42);
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(17)
                        .PhysicalColumnId(17)
                        .FieldId(42)
                        .Type(EColumnType::kIndex32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());

   ROOT::Experimental::RClusterDescriptor::RColumnRange columnRange;
   ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   RClusterDescriptorBuilder clusterBuilder;
   clusterBuilder.ClusterId(84).FirstEntryIndex(0).NEntries(100);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange;
   pageRange.fPhysicalColumnId = 17;
   // Two pages adding up to 100 elements, one with checksum one without
   pageInfo.fNElements = 40;
   pageInfo.fLocator.fPosition = 7000U;
   pageInfo.fHasChecksum = true;
   pageRange.fPageInfos.emplace_back(pageInfo);
   pageInfo.fNElements = 60;
   pageInfo.fLocator.fPosition = 8000U;
   pageInfo.fHasChecksum = false;
   pageRange.fPageInfos.emplace_back(pageInfo);
   clusterBuilder.CommitColumnRange(17, 0, 100, pageRange);
   builder.AddCluster(clusterBuilder.MoveDescriptor().Unwrap());
   RClusterGroupDescriptorBuilder cgBuilder;
   RNTupleLocator cgLocator;
   cgLocator.fPosition = 1337U;
   cgLocator.fBytesOnStorage = 42;
   cgBuilder.ClusterGroupId(256).PageListLength(137).PageListLocator(cgLocator).NClusters(1).EntrySpan(100);
   std::vector<DescriptorId_t> clusterIds{84};
   cgBuilder.AddClusters(clusterIds);
   builder.AddClusterGroup(cgBuilder.MoveDescriptor().Unwrap());

   auto desc = builder.MoveDescriptor();
   auto context = RNTupleSerializer::SerializeHeader(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto bufHeader = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeader(bufHeader.get(), desc);

   std::vector<DescriptorId_t> physClusterIDs;
   for (const auto &c : desc.GetClusterIterable()) {
      physClusterIDs.emplace_back(context.MapClusterId(c.GetId()));
   }
   EXPECT_EQ(desc.GetNClusters(), physClusterIDs.size());
   context.MapClusterGroupId(256);

   auto sizePageList = RNTupleSerializer::SerializePageList(nullptr, desc, physClusterIDs, context);
   EXPECT_GT(sizePageList, 0);
   auto bufPageList = std::make_unique<unsigned char []>(sizePageList);
   EXPECT_EQ(sizePageList, RNTupleSerializer::SerializePageList(bufPageList.get(), desc, physClusterIDs, context));

   auto sizeFooter = RNTupleSerializer::SerializeFooter(nullptr, desc, context);
   EXPECT_GT(sizeFooter, 0);
   auto bufFooter = std::make_unique<unsigned char []>(sizeFooter);
   EXPECT_EQ(sizeFooter, RNTupleSerializer::SerializeFooter(bufFooter.get(), desc, context));

   RNTupleSerializer::DeserializeHeader(bufHeader.get(), context.GetHeaderSize(), builder);
   RNTupleSerializer::DeserializeFooter(bufFooter.get(), sizeFooter, builder);
   desc = builder.MoveDescriptor();

   EXPECT_EQ(1u, desc.GetNClusterGroups());
   const auto &clusterGroupDesc = desc.GetClusterGroupDescriptor(0);
   EXPECT_EQ(0u, clusterGroupDesc.GetMinEntry());
   EXPECT_EQ(100u, clusterGroupDesc.GetEntrySpan());
   EXPECT_EQ(1u, clusterGroupDesc.GetNClusters());
   EXPECT_EQ(137u, clusterGroupDesc.GetPageListLength());
   EXPECT_EQ(1337u, clusterGroupDesc.GetPageListLocator().GetPosition<std::uint64_t>());
   EXPECT_EQ(42u, clusterGroupDesc.GetPageListLocator().fBytesOnStorage);
   EXPECT_EQ(1u, desc.GetNClusters());
   EXPECT_EQ(0u, desc.GetNActiveClusters());

   RNTupleSerializer::DeserializePageList(bufPageList.get(), sizePageList, 0, desc);
   const auto &verify = desc.GetClusterGroupDescriptor(0);
   EXPECT_EQ(0u, verify.GetMinEntry());
   EXPECT_EQ(100u, verify.GetEntrySpan());
   EXPECT_EQ(1u, verify.GetNClusters());
   EXPECT_EQ(137u, verify.GetPageListLength());
   EXPECT_EQ(1337u, verify.GetPageListLocator().GetPosition<std::uint64_t>());
   EXPECT_EQ(42u, verify.GetPageListLocator().fBytesOnStorage);

   EXPECT_EQ(1u, desc.GetNActiveClusters());
   const auto &clusterDesc = desc.GetClusterDescriptor(0);
   EXPECT_EQ(0, clusterDesc.GetFirstEntryIndex());
   EXPECT_EQ(100, clusterDesc.GetNEntries());
   auto columnIds = clusterDesc.GetColumnRangeIterable();
   EXPECT_EQ(1u, columnIds.count());
   EXPECT_EQ(0, columnIds.begin()->first);
   columnRange = clusterDesc.GetColumnRange(0);
   EXPECT_EQ(100u, columnRange.fNElements);
   EXPECT_EQ(0u, columnRange.fFirstElementIndex);
   pageRange = clusterDesc.GetPageRange(0).Clone();
   EXPECT_EQ(2u, pageRange.fPageInfos.size());
   EXPECT_EQ(40u, pageRange.fPageInfos[0].fNElements);
   EXPECT_EQ(7000u, pageRange.fPageInfos[0].fLocator.GetPosition<std::uint64_t>());
   EXPECT_TRUE(pageRange.fPageInfos[0].fHasChecksum);
   EXPECT_EQ(60u, pageRange.fPageInfos[1].fNElements);
   EXPECT_EQ(8000u, pageRange.fPageInfos[1].fLocator.GetPosition<std::uint64_t>());
   EXPECT_FALSE(pageRange.fPageInfos[1].fHasChecksum);
}

TEST(RNTuple, SerializeFooterXHeader)
{
   RNTupleDescriptorBuilder builder;
   builder.SetNTuple("ntpl", "");
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(0)
                       .FieldName("")
                       .Structure(ENTupleStructure::kRecord)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(42)
                       .FieldName("field")
                       .TypeName("int32_t")
                       .Structure(ENTupleStructure::kLeaf)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddFieldLink(0, 42);
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(17)
                        .PhysicalColumnId(17)
                        .FieldId(42)
                        .Type(EColumnType::kInt32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());

   auto context = RNTupleSerializer::SerializeHeader(nullptr, builder.GetDescriptor());
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto bufHeader = std::make_unique<unsigned char[]>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeader(bufHeader.get(), builder.GetDescriptor());

   builder.BeginHeaderExtension();
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(43)
                       .FieldName("struct")
                       .Structure(ENTupleStructure::kRecord)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(44)
                       .FieldName("f")
                       .TypeName("float")
                       .Structure(ENTupleStructure::kLeaf)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(45)
                       .FieldName("i64")
                       .TypeName("int64_t")
                       .Structure(ENTupleStructure::kLeaf)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddFieldLink(0, 43);
   builder.AddFieldLink(43, 44);
   builder.AddFieldLink(0, 45);
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(18)
                        .PhysicalColumnId(18)
                        .FieldId(44)
                        .Type(EColumnType::kReal32)
                        .Index(0)
                        .FirstElementIndex(4200)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(19)
                        .PhysicalColumnId(19)
                        .FieldId(45)
                        .Type(EColumnType::kInt64)
                        .Index(0)
                        .FirstElementIndex(10000)
                        .MakeDescriptor()
                        .Unwrap());

   builder.AddField(RFieldDescriptorBuilder()
                       .FieldId(46)
                       .FieldName("projected")
                       .TypeName("float")
                       .Structure(ENTupleStructure::kLeaf)
                       .MakeDescriptor()
                       .Unwrap());
   builder.AddFieldLink(0, 46);
   builder.AddColumn(RColumnDescriptorBuilder()
                        .LogicalColumnId(20)
                        .PhysicalColumnId(18)
                        .FieldId(46)
                        .Type(EColumnType::kReal32)
                        .Index(0)
                        .MakeDescriptor()
                        .Unwrap());
   builder.AddExtraTypeInfo(RExtraTypeInfoDescriptorBuilder()
                               .ContentId(EExtraTypeInfoIds::kStreamerInfo)
                               .Content("xyz")
                               .MoveDescriptor()
                               .Unwrap());

   // Make sure late-added fields and the corresponding columns get an on-disk ID
   context.MapSchema(builder.GetDescriptor(), /*forHeaderExtension=*/true);

   auto desc = builder.MoveDescriptor();
   auto sizeFooter = RNTupleSerializer::SerializeFooter(nullptr, desc, context);
   EXPECT_GT(sizeFooter, 0);
   auto bufFooter = std::make_unique<unsigned char[]>(sizeFooter);
   EXPECT_EQ(sizeFooter, RNTupleSerializer::SerializeFooter(bufFooter.get(), desc, context));

   RNTupleSerializer::DeserializeHeader(bufHeader.get(), context.GetHeaderSize(), builder);
   RNTupleSerializer::DeserializeFooter(bufFooter.get(), sizeFooter, builder);

   desc = builder.MoveDescriptor();

   EXPECT_EQ(6u, desc.GetNFields());
   EXPECT_EQ(4u, desc.GetNLogicalColumns());
   EXPECT_EQ(3u, desc.GetNPhysicalColumns());
   auto xHeader = desc.GetHeaderExtension();
   EXPECT_TRUE(xHeader != nullptr);
   EXPECT_EQ(4u, xHeader->GetNFields());
   EXPECT_EQ(3u, xHeader->GetNLogicalColumns());
   EXPECT_EQ(2u, xHeader->GetNPhysicalColumns());
   auto fldIdStruct = desc.FindFieldId("struct");
   auto fldIdF = desc.FindFieldId("f", fldIdStruct);
   auto fldIdI64 = desc.FindFieldId("i64");
   EXPECT_EQ(1, fldIdStruct);
   EXPECT_EQ(2, fldIdF);
   EXPECT_EQ(3, fldIdI64);
   EXPECT_EQ(4, desc.FindFieldId("projected"));

   unsigned int counter = 0;
   for (const auto &c : desc.GetColumnIterable(fldIdF)) {
      EXPECT_EQ(c.GetFirstElementIndex(), 4200U);
      counter++;
   }
   EXPECT_EQ(1U, counter);
   for (const auto &c : desc.GetColumnIterable(fldIdI64)) {
      EXPECT_EQ(c.GetFirstElementIndex(), 10000U);
      counter++;
   }
   EXPECT_EQ(2U, counter);

   EXPECT_EQ(1u, desc.GetNExtraTypeInfos());
   const auto &extraTypeInfoDesc = *desc.GetExtraTypeInfoIterable().begin();
   EXPECT_EQ(EExtraTypeInfoIds::kStreamerInfo, extraTypeInfoDesc.GetContentId());
   EXPECT_EQ(0u, extraTypeInfoDesc.GetTypeVersionFrom());
   EXPECT_EQ(0u, extraTypeInfoDesc.GetTypeVersionTo());
   EXPECT_TRUE(extraTypeInfoDesc.GetTypeName().empty());
   EXPECT_STREQ("xyz", extraTypeInfoDesc.GetContent().c_str());
}

TEST(RNTuple, SerializeStreamerInfos)
{
   RNTupleSerializer::StreamerInfoMap_t infos;
   auto content = RNTupleSerializer::SerializeStreamerInfos(infos);
   EXPECT_TRUE(RNTupleSerializer::DeserializeStreamerInfos(content).Unwrap().empty());

   auto streamerInfo = RNTuple::Class()->GetStreamerInfo();
   infos[streamerInfo->GetNumber()] = streamerInfo;
   content = RNTupleSerializer::SerializeStreamerInfos(infos);
   auto result = RNTupleSerializer::DeserializeStreamerInfos(content).Unwrap();
   EXPECT_EQ(1u, result.size());
   EXPECT_STREQ("ROOT::Experimental::RNTuple", std::string(result.begin()->second->GetName()).c_str());
}
