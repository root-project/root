#include "ntuple_test.hxx"

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

TEST(RNTuple, SerializeEnvelope)
{
   try {
      RNTupleSerializer::DeserializeEnvelope(nullptr, 0).Unwrap();
      FAIL() << "too small envelope should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   struct {
      std::uint16_t writerVersion = static_cast<std::uint16_t>(1);
      std::uint16_t minVersion = static_cast<std::uint16_t>(1);
      std::uint32_t crc32 = 0;
   } testEnvelope;

   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope)).Unwrap();
      FAIL() << "CRC32 mismatch should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("CRC32"));
   }

   testEnvelope.writerVersion = testEnvelope.minVersion = 0;

   EXPECT_EQ(4u, RNTupleSerializer::SerializeEnvelopePostscript(
      reinterpret_cast<const unsigned char *>(&testEnvelope), 4, &testEnvelope.crc32));
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope)).Unwrap();
      FAIL() << "unsupported version should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too old"));
   }

   testEnvelope.writerVersion = RNTupleSerializer::kEnvelopeCurrentVersion;
   std::uint32_t crc32_write;
   RNTupleSerializer::SerializeEnvelopePostscript(
      reinterpret_cast<const unsigned char *>(&testEnvelope), 4, crc32_write, &testEnvelope.crc32);
   std::uint32_t crc32_read;
   EXPECT_EQ(4u, RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope), crc32_read).Unwrap());
   EXPECT_EQ(crc32_write, crc32_read);

   testEnvelope.writerVersion = RNTupleSerializer::kEnvelopeCurrentVersion + 1;
   testEnvelope.minVersion = RNTupleSerializer::kEnvelopeCurrentVersion + 1;
   EXPECT_EQ(4u, RNTupleSerializer::SerializeEnvelopePostscript(
      reinterpret_cast<const unsigned char *>(&testEnvelope), 4, &testEnvelope.crc32));
   try {
      RNTupleSerializer::DeserializeEnvelope(&testEnvelope, sizeof(testEnvelope)).Unwrap();
      FAIL() << "unsupported version should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too new"));
   }
}

TEST(RNTuple, SerializeFrame)
{
   std::uint32_t frameSize;
   std::uint32_t nitems;

   try {
      RNTupleSerializer::DeserializeFrameHeader(nullptr, 0, frameSize).Unwrap();
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   unsigned char buffer[12];
   EXPECT_EQ(4u, RNTupleSerializer::SerializeRecordFramePreamble(buffer));
   try {
      RNTupleSerializer::SerializeFramePostscript(buffer, 3);
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(0u, RNTupleSerializer::SerializeFramePostscript(buffer, 6));

   try {
      RNTupleSerializer::DeserializeFrameHeader(buffer, 4, frameSize).Unwrap();
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(4u, RNTupleSerializer::DeserializeFrameHeader(buffer, 6, frameSize, nitems).Unwrap());
   EXPECT_EQ(6u, frameSize);
   EXPECT_EQ(1u, nitems);

   EXPECT_EQ(4u, RNTupleSerializer::SerializeRecordFramePreamble(buffer));
   try {
      RNTupleSerializer::SerializeFramePostscript(buffer, -2);
      FAIL() << "too big frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too large"));
   }

   try {
      RNTupleSerializer::SerializeListFramePreamble(1 << 29, buffer);
      FAIL() << "too big list frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too large"));
   }
   EXPECT_EQ(8u, RNTupleSerializer::SerializeListFramePreamble(2, buffer));
   try {
      RNTupleSerializer::SerializeFramePostscript(buffer, 7);
      FAIL() << "too small frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(0u, RNTupleSerializer::SerializeFramePostscript(buffer, 12));

   try {
      RNTupleSerializer::DeserializeFrameHeader(buffer, 6, frameSize, nitems).Unwrap();
      FAIL() << "too short list frame should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(8u, RNTupleSerializer::DeserializeFrameHeader(buffer, 12, frameSize, nitems).Unwrap());
   EXPECT_EQ(12u, frameSize);
   EXPECT_EQ(2u, nitems);
}

TEST(RNTuple, SerializeFeatureFlags)
{
   std::vector<std::int64_t> flags;
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

   flags.push_back(-2);
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
   locator.fPosition = 1;
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

   locator.fPosition = 0;
   locator.fBytesOnStorage = 0;
   locator.fUrl = "xyz";
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
   EXPECT_EQ(1u, locator.fPosition);
   EXPECT_EQ(2u, locator.fBytesOnStorage);
   EXPECT_TRUE(locator.fUrl.empty());

   locator.fUrl = "X";
   EXPECT_EQ(5u, RNTupleSerializer::SerializeLocator(locator, nullptr));
   EXPECT_EQ(5u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator.fPosition = 42;
   locator.fBytesOnStorage = 42;
   locator.fUrl.clear();
   try {
      RNTupleSerializer::DeserializeLocator(buffer, 4, locator).Unwrap();
      FAIL() << "too short locator buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(5u, RNTupleSerializer::DeserializeLocator(buffer, 5, locator).Unwrap());
   EXPECT_EQ(0u, locator.fPosition);
   EXPECT_EQ(0u, locator.fBytesOnStorage);
   EXPECT_EQ("X", locator.fUrl);

   locator.fUrl = "abcdefghijkl";
   EXPECT_EQ(16u, RNTupleSerializer::SerializeLocator(locator, buffer));
   locator.fUrl.clear();
   EXPECT_EQ(16u, RNTupleSerializer::DeserializeLocator(buffer, 16, locator).Unwrap());
   EXPECT_EQ("abcdefghijkl", locator.fUrl);

   std::int32_t *head = reinterpret_cast<std::int32_t *>(buffer);
   *head = (0x3 << 24) | *head;
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
   link.fUnzippedSize = 42;
   link.fLocator.fPosition = 137;
   link.fLocator.fBytesOnStorage = 7;

   unsigned char buffer[16];
   EXPECT_EQ(16u, RNTupleSerializer::SerializeEnvelopeLink(link, nullptr));
   EXPECT_EQ(16u, RNTupleSerializer::SerializeEnvelopeLink(link, buffer));
   try {
      RNTupleSerializer::DeserializeEnvelopeLink(buffer, 3, link).Unwrap();
      FAIL() << "too short envelope link buffer should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }

   RNTupleSerializer::REnvelopeLink reconstructedLink;
   EXPECT_EQ(16u, RNTupleSerializer::DeserializeEnvelopeLink(buffer, 16, reconstructedLink).Unwrap());
   EXPECT_EQ(link.fUnzippedSize, reconstructedLink.fUnzippedSize);
   EXPECT_EQ(link.fLocator, reconstructedLink.fLocator);
}

TEST(RNTuple, SerializeClusterSummary)
{
   RNTupleSerializer::RClusterSummary summary;
   summary.fFirstEntry = 42;
   summary.fNEntries = 137;

   unsigned char buffer[24];
   ASSERT_EQ(20u, RNTupleSerializer::SerializeClusterSummary(summary, nullptr));
   EXPECT_EQ(20u, RNTupleSerializer::SerializeClusterSummary(summary, buffer));
   RNTupleSerializer::RClusterSummary reco;
   try {
      RNTupleSerializer::DeserializeClusterSummary(buffer, 19, reco).Unwrap();
      FAIL() << "too short cluster summary should fail";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(20u, RNTupleSerializer::DeserializeClusterSummary(buffer, 20, reco).Unwrap());
   EXPECT_EQ(summary.fFirstEntry, reco.fFirstEntry);
   EXPECT_EQ(summary.fNEntries, reco.fNEntries);
   EXPECT_EQ(summary.fColumnGroupID, reco.fColumnGroupID);

   summary.fColumnGroupID = 13;
   ASSERT_EQ(24u, RNTupleSerializer::SerializeClusterSummary(summary, nullptr));
   EXPECT_EQ(24u, RNTupleSerializer::SerializeClusterSummary(summary, buffer));
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
}

TEST(RNTuple, SerializeClusterGroup)
{
   RNTupleSerializer::RClusterGroup group;
   group.fNClusters = 42;
   group.fPageListEnvelopeLink.fUnzippedSize = 42;
   group.fPageListEnvelopeLink.fLocator.fPosition = 137;
   group.fPageListEnvelopeLink.fLocator.fBytesOnStorage = 7;

   unsigned char buffer[28];
   ASSERT_EQ(24u, RNTupleSerializer::SerializeClusterGroup(group, nullptr));
   EXPECT_EQ(24u, RNTupleSerializer::SerializeClusterGroup(group, buffer));
   RNTupleSerializer::RClusterGroup reco;
   try {
      RNTupleSerializer::DeserializeClusterGroup(buffer, 23, reco).Unwrap();
      FAIL() << "too short cluster group should fail";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too short"));
   }
   EXPECT_EQ(24u, RNTupleSerializer::DeserializeClusterGroup(buffer, 24, reco).Unwrap());
   EXPECT_EQ(group.fNClusters, reco.fNClusters);
   EXPECT_EQ(group.fPageListEnvelopeLink.fUnzippedSize, reco.fPageListEnvelopeLink.fUnzippedSize);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fPosition, reco.fPageListEnvelopeLink.fLocator.fPosition);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fBytesOnStorage, reco.fPageListEnvelopeLink.fLocator.fBytesOnStorage);

   // Test frame evolution
   auto pos = buffer;
   pos += RNTupleSerializer::SerializeRecordFramePreamble(pos);
   pos += RNTupleSerializer::SerializeUInt32(group.fNClusters, pos);
   pos += RNTupleSerializer::SerializeEnvelopeLink(group.fPageListEnvelopeLink, pos);
   pos += RNTupleSerializer::SerializeUInt16(7, pos);
   pos += RNTupleSerializer::SerializeFramePostscript(buffer, pos - buffer);
   pos += RNTupleSerializer::SerializeUInt16(13, pos);
   EXPECT_EQ(26u, RNTupleSerializer::DeserializeClusterGroup(buffer, 26, reco).Unwrap());
   EXPECT_EQ(group.fNClusters, reco.fNClusters);
   EXPECT_EQ(group.fPageListEnvelopeLink.fUnzippedSize, reco.fPageListEnvelopeLink.fUnzippedSize);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fPosition, reco.fPageListEnvelopeLink.fLocator.fPosition);
   EXPECT_EQ(group.fPageListEnvelopeLink.fLocator.fBytesOnStorage, reco.fPageListEnvelopeLink.fLocator.fBytesOnStorage);
   std::uint16_t remainder;
   RNTupleSerializer::DeserializeUInt16(buffer + 24, remainder);
   EXPECT_EQ(7u, remainder);
   RNTupleSerializer::DeserializeUInt16(buffer + 26, remainder);
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
   auto context = RNTupleSerializer::SerializeHeaderV1(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto buffer = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeaderV1(buffer.get(), desc);

   RNTupleSerializer::DeserializeHeaderV1(buffer.get(), context.GetHeaderSize(), builder);
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
   builder.AddFieldLink(0, 137);
   builder.AddFieldLink(137, 13);
   builder.AddColumn(23, 42, RColumnModel(EColumnType::kReal32, false), 0);
   builder.AddColumn(17, 137, RColumnModel(EColumnType::kIndex, true), 0);
   builder.AddColumn(40, 137, RColumnModel(EColumnType::kByte, true), 1);

   auto desc = builder.MoveDescriptor();
   auto context = RNTupleSerializer::SerializeHeaderV1(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto buffer = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeaderV1(buffer.get(), desc);

   RNTupleSerializer::DeserializeHeaderV1(buffer.get(), context.GetHeaderSize(), builder);
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
   builder.AddColumn(17, 42, RColumnModel(EColumnType::kIndex, true), 0);

   ROOT::Experimental::RClusterDescriptor::RColumnRange columnRange;
   ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   RClusterDescriptorBuilder clusterBuilder(84, 0, 100);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange;
   pageRange.fColumnId = 17;
   pageInfo.fNElements = 100;
   pageInfo.fLocator.fPosition = 7000;
   pageRange.fPageInfos.emplace_back(pageInfo);
   clusterBuilder.CommitColumnRange(17, 0, 100, pageRange);
   builder.AddClusterWithDetails(clusterBuilder.MoveDescriptor().Unwrap());
   RClusterGroupDescriptorBuilder cgBuilder;
   RNTupleLocator cgLocator;
   cgLocator.fPosition = 1337;
   cgLocator.fBytesOnStorage = 42;
   cgBuilder.ClusterGroupId(256).PageListLength(137).PageListLocator(cgLocator);
   cgBuilder.AddCluster(84);
   builder.AddClusterGroup(std::move(cgBuilder));

   auto desc = builder.MoveDescriptor();
   auto context = RNTupleSerializer::SerializeHeaderV1(nullptr, desc);
   EXPECT_GT(context.GetHeaderSize(), 0);
   auto bufHeader = std::make_unique<unsigned char []>(context.GetHeaderSize());
   context = RNTupleSerializer::SerializeHeaderV1(bufHeader.get(), desc);

   std::vector<DescriptorId_t> physClusterIDs;
   for (const auto &c : desc.GetClusterIterable()) {
      physClusterIDs.emplace_back(context.MapClusterId(c.GetId()));
   }
   EXPECT_EQ(desc.GetNClusters(), physClusterIDs.size());
   context.MapClusterGroupId(256);

   auto sizePageList = RNTupleSerializer::SerializePageListV1(nullptr, desc, physClusterIDs, context);
   EXPECT_GT(sizePageList, 0);
   auto bufPageList = std::make_unique<unsigned char []>(sizePageList);
   EXPECT_EQ(sizePageList, RNTupleSerializer::SerializePageListV1(bufPageList.get(), desc, physClusterIDs, context));

   auto sizeFooter = RNTupleSerializer::SerializeFooterV1(nullptr, desc, context);
   EXPECT_GT(sizeFooter, 0);
   auto bufFooter = std::make_unique<unsigned char []>(sizeFooter);
   EXPECT_EQ(sizeFooter, RNTupleSerializer::SerializeFooterV1(bufFooter.get(), desc, context));

   RNTupleSerializer::DeserializeHeaderV1(bufHeader.get(), context.GetHeaderSize(), builder);
   RNTupleSerializer::DeserializeFooterV1(bufFooter.get(), sizeFooter, builder);

   desc = builder.MoveDescriptor();

   EXPECT_EQ(1u, desc.GetNClusterGroups());
   const auto &clusterGroupDesc = desc.GetClusterGroupDescriptor(0);
   EXPECT_EQ(1u, clusterGroupDesc.GetNClusters());
   EXPECT_EQ(137u, clusterGroupDesc.GetPageListLength());
   EXPECT_EQ(1337u, clusterGroupDesc.GetPageListLocator().fPosition);
   EXPECT_EQ(42u, clusterGroupDesc.GetPageListLocator().fBytesOnStorage);

   std::vector<RClusterDescriptorBuilder> clusters = RClusterGroupDescriptorBuilder::GetClusterSummaries(desc, 0);
   RNTupleSerializer::DeserializePageListV1(bufPageList.get(), sizePageList, clusters);
   EXPECT_EQ(physClusterIDs.size(), clusters.size());
   for (std::size_t i = 0; i < clusters.size(); ++i) {
      desc.AddClusterDetails(clusters[i].MoveDescriptor().Unwrap());
   }

   EXPECT_EQ(1u, desc.GetNClusters());
   const auto &clusterDesc = desc.GetClusterDescriptor(0);
   EXPECT_EQ(0, clusterDesc.GetFirstEntryIndex());
   EXPECT_EQ(100, clusterDesc.GetNEntries());
   auto columnIds = clusterDesc.GetColumnIds();
   EXPECT_EQ(1u, columnIds.size());
   EXPECT_EQ(1u, columnIds.count(0));
   columnRange = clusterDesc.GetColumnRange(0);
   EXPECT_EQ(100u, columnRange.fNElements);
   EXPECT_EQ(0u, columnRange.fFirstElementIndex);
   pageRange = clusterDesc.GetPageRange(0).Clone();
   EXPECT_EQ(1u, pageRange.fPageInfos.size());
   EXPECT_EQ(100u, pageRange.fPageInfos[0].fNElements);
   EXPECT_EQ(7000u, pageRange.fPageInfos[0].fLocator.fPosition);
}
