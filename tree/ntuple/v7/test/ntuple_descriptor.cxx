#include "ntuple_test.hxx"

TEST(RNTuple, Descriptor)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("MyTuple", "Description", "Me", RNTupleVersion(1, 2, 3), ROOT::Experimental::RNTupleUuid());
   descBuilder.AddField(0, RNTupleVersion(), RNTupleVersion(), "", "", 0, ENTupleStructure::kRecord);
   descBuilder.AddField(1, RNTupleVersion(), RNTupleVersion(), "list", "std::vector<std::int32_t>",
                        0, ENTupleStructure::kCollection);
   descBuilder.AddFieldLink(0, 1);
   descBuilder.AddField(2, RNTupleVersion(), RNTupleVersion(), "list", "std::int32_t", 0, ENTupleStructure::kLeaf);
   descBuilder.AddFieldLink(1, 2);
   descBuilder.AddField(42, RNTupleVersion(), RNTupleVersion(), "x", "std::string", 0, ENTupleStructure::kLeaf);
   descBuilder.AddFieldLink(0, 42);
   descBuilder.AddColumn(3, 42, RNTupleVersion(), RColumnModel(EColumnType::kIndex, true), 0);
   descBuilder.AddColumn(4, 42, RNTupleVersion(), RColumnModel(EColumnType::kByte, true), 1);

   ROOT::Experimental::RClusterDescriptor::RColumnRange columnRange;
   ROOT::Experimental::RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   // Description of cluster #0
   descBuilder.AddCluster(0, RNTupleVersion(), 0, ROOT::Experimental::ClusterSize_t(100));
   columnRange.fColumnId = 3;
   columnRange.fFirstElementIndex = 0;
   columnRange.fNElements = 100;
   descBuilder.AddClusterColumnRange(0, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange0;
   pageRange0.fPageInfos.clear();
   pageRange0.fColumnId = 3;
   pageInfo.fNElements = 40;
   pageInfo.fLocator.fPosition = 0;
   pageRange0.fPageInfos.emplace_back(pageInfo);
   pageInfo.fNElements = 60;
   pageInfo.fLocator.fPosition = 1024;
   pageRange0.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(0, std::move(pageRange0));

   columnRange.fColumnId = 4;
   columnRange.fFirstElementIndex = 0;
   columnRange.fNElements = 300;
   descBuilder.AddClusterColumnRange(0, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange1;
   pageRange1.fColumnId = 4;
   pageInfo.fNElements = 200;
   pageInfo.fLocator.fPosition = 2048;
   pageRange1.fPageInfos.emplace_back(pageInfo);
   pageInfo.fNElements = 100;
   pageInfo.fLocator.fPosition = 4096;
   pageRange1.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(0, std::move(pageRange1));

   // Description of cluster #1
   descBuilder.AddCluster(1, RNTupleVersion(), 100, ROOT::Experimental::ClusterSize_t(1000));
   columnRange.fColumnId = 3;
   columnRange.fFirstElementIndex = 100;
   columnRange.fNElements = 1000;
   descBuilder.AddClusterColumnRange(1, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange2;
   pageRange2.fColumnId = 3;
   pageInfo.fNElements = 1000;
   pageInfo.fLocator.fPosition = 8192;
   pageRange2.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(1, std::move(pageRange2));

   columnRange.fColumnId = 4;
   columnRange.fFirstElementIndex = 300;
   columnRange.fNElements = 3000;
   descBuilder.AddClusterColumnRange(1, columnRange);
   ROOT::Experimental::RClusterDescriptor::RPageRange pageRange3;
   pageRange3.fColumnId = 4;
   pageInfo.fNElements = 3000;
   pageInfo.fLocator.fPosition = 16384;
   pageRange3.fPageInfos.emplace_back(pageInfo);
   descBuilder.AddClusterPageRange(1, std::move(pageRange3));

   const auto &reference = descBuilder.GetDescriptor();
   EXPECT_EQ("MyTuple", reference.GetName());
   EXPECT_EQ(1U, reference.GetVersion().GetVersionUse());
   EXPECT_EQ(2U, reference.GetVersion().GetVersionMin());
   EXPECT_EQ(3U, reference.GetVersion().GetFlags());

   auto szHeader = reference.SerializeHeader(nullptr);
   auto headerBuffer = new unsigned char[szHeader];
   reference.SerializeHeader(headerBuffer);
   auto szFooter = reference.SerializeFooter(nullptr);
   auto footerBuffer = new unsigned char[szFooter];
   reference.SerializeFooter(footerBuffer);

   const auto nbytesPostscript = RNTupleDescriptor::kNBytesPostscript;
   ASSERT_GE(szFooter, nbytesPostscript);
   std::uint32_t szPsHeader;
   std::uint32_t szPsFooter;
   RNTupleDescriptor::LocateMetadata(footerBuffer + szFooter - nbytesPostscript, szPsHeader, szPsFooter);
   EXPECT_EQ(szHeader, szPsHeader);
   EXPECT_EQ(szFooter, szPsFooter);

   RNTupleDescriptorBuilder reco;
   reco.SetFromHeader(headerBuffer);
   reco.AddClustersFromFooter(footerBuffer);
   EXPECT_EQ(reference, reco.GetDescriptor());

   EXPECT_EQ(NTupleSize_t(1100), reference.GetNEntries());
   EXPECT_EQ(NTupleSize_t(1100), reference.GetNElements(3));
   EXPECT_EQ(NTupleSize_t(3300), reference.GetNElements(4));

   EXPECT_EQ(DescriptorId_t(0), reference.GetFieldZeroId());
   EXPECT_EQ(DescriptorId_t(1), reference.FindFieldId("list", 0));
   EXPECT_EQ(DescriptorId_t(1), reference.FindFieldId("list"));
   EXPECT_EQ(DescriptorId_t(2), reference.FindFieldId("list", 1));
   EXPECT_EQ(DescriptorId_t(42), reference.FindFieldId("x", 0));
   EXPECT_EQ(DescriptorId_t(42), reference.FindFieldId("x"));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindFieldId("listX", 1));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindFieldId("list", 1024));

   EXPECT_EQ(DescriptorId_t(3), reference.FindColumnId(42, 0));
   EXPECT_EQ(DescriptorId_t(4), reference.FindColumnId(42, 1));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindColumnId(42, 2));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindColumnId(43, 0));

   EXPECT_EQ(DescriptorId_t(0), reference.FindClusterId(3, 0));
   EXPECT_EQ(DescriptorId_t(1), reference.FindClusterId(3, 100));
   EXPECT_EQ(ROOT::Experimental::kInvalidDescriptorId, reference.FindClusterId(3, 40000));

   delete[] footerBuffer;
   delete[] headerBuffer;
}
