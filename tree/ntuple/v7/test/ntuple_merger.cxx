#include "ntuple_test.hxx"

namespace {

// Reads an integer from a little-endian 4 byte buffer
std::int32_t ReadRawInt(void *ptr)
{
   std::int32_t val = *reinterpret_cast<std::int32_t *>(ptr);
#ifndef R__BYTESWAP
   // on big endian system
   auto x = (val & 0x0000FFFF) << 16 | (val & 0xFFFF0000) >> 16;
   return (x & 0x00FF00FF) << 8 | (x & 0xFF00FF00) >> 8;
#else
   return val;
#endif
}

} // anonymous namespace

TEST(RPageStorage, ReadSealedPages)
{
   FileRaii fileGuard("test_ntuple_sealed_pages.root");

   // Create an ntuple at least 2 clusters, one with 1 entry and one with 100000 entries.
   // Hence the second cluster should have more than a single page per column.  We write uncompressed
   // pages so that we can meaningfully peek into the content of read sealed pages later on.
   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<std::int32_t>("pt", 42);
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      RNTupleWriter ntuple(std::move(model),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
      ntuple.CommitCluster();
      for (unsigned i = 0; i < 100000; ++i) {
         *wrPt = i;
         ntuple.Fill();
      }
   }

   RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();
   const auto &descriptor = source.GetDescriptor();
   auto columnId = descriptor.FindColumnId(descriptor.FindFieldId("pt"), 0);
   RClusterIndex index(descriptor.FindClusterId(columnId, 0), 0);
   auto sealedPage = source.ReadSealedPage(columnId, index);
   ASSERT_EQ(1U, sealedPage.fNElements);
   ASSERT_EQ(4U, sealedPage.fSize);
   EXPECT_EQ(42, ReadRawInt(sealedPage.fBuffer.get()));

   // Check second, big cluster
   auto clusterId = descriptor.FindClusterId(columnId, 1);
   ASSERT_NE(clusterId, index.GetClusterId());
   const auto &clusterDesc = descriptor.GetClusterDescriptor(clusterId);
   const auto &pageRange = clusterDesc.GetPageRange(columnId);
   EXPECT_GT(pageRange.fPageInfos.size(), 1U);
   std::uint32_t firstElementInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      sealedPage = source.ReadSealedPage(columnId, RClusterIndex(clusterId, firstElementInPage));
      ASSERT_GE(sealedPage.fSize, 4U);
      EXPECT_EQ(firstElementInPage, ReadRawInt(sealedPage.fBuffer.get()));
      firstElementInPage += pi.fNElements;
   }
}


TEST(RFieldMerger, Merge)
{
   auto mergeResult = RFieldMerger::Merge(RFieldDescriptor(), RFieldDescriptor());
   EXPECT_FALSE(mergeResult);
}
