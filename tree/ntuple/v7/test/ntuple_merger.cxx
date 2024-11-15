#include "ntuple_test.hxx"

#include <TFileMerger.h>
#include <ROOT/TBufferMerger.hxx>
#include <gtest/gtest.h>
#include <string_view>
#include <unordered_map>
#include <zlib.h>
#include "gmock/gmock.h"

namespace {

// Reads an integer from a little-endian 4 byte buffer
std::int32_t ReadRawInt(const void *ptr)
{
   std::int32_t val = *reinterpret_cast<const std::int32_t *>(ptr);
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
      options.SetMaxUnzippedPageSize(4096);
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      writer->Fill();
      writer->CommitCluster();
      for (unsigned i = 0; i < 100000; ++i) {
         *wrPt = i;
         writer->Fill();
      }
   }

   RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();
   const auto fieldId = source.GetSharedDescriptorGuard()->FindFieldId("pt");
   auto columnId = source.GetSharedDescriptorGuard()->FindPhysicalColumnId(fieldId, 0, 0);

   // Check first cluster consisting of a single entry
   RClusterIndex index(source.GetSharedDescriptorGuard()->FindClusterId(columnId, 0), 0);
   RPageStorage::RSealedPage sealedPage;
   source.LoadSealedPage(columnId, index, sealedPage);
   ASSERT_EQ(1U, sealedPage.GetNElements());
   ASSERT_EQ(4U, sealedPage.GetDataSize());
   ASSERT_EQ(12U, sealedPage.GetBufferSize());
   auto buffer = std::make_unique<unsigned char[]>(sealedPage.GetBufferSize());
   sealedPage.SetBuffer(buffer.get());
   source.LoadSealedPage(columnId, index, sealedPage);
   ASSERT_EQ(1U, sealedPage.GetNElements());
   ASSERT_EQ(4U, sealedPage.GetDataSize());
   ASSERT_EQ(12U, sealedPage.GetBufferSize());
   EXPECT_EQ(42, ReadRawInt(sealedPage.GetBuffer()));

   // Check second, big cluster
   auto clusterId = source.GetSharedDescriptorGuard()->FindClusterId(columnId, 1);
   ASSERT_NE(clusterId, index.GetClusterId());
   const auto clusterDesc = source.GetSharedDescriptorGuard()->GetClusterDescriptor(clusterId).Clone();
   const auto &pageRange = clusterDesc.GetPageRange(columnId);
   EXPECT_GT(pageRange.fPageInfos.size(), 1U);
   std::uint32_t firstElementInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      sealedPage.SetBuffer(nullptr);
      source.LoadSealedPage(columnId, RClusterIndex(clusterId, firstElementInPage), sealedPage);
      buffer = std::make_unique<unsigned char[]>(sealedPage.GetBufferSize());
      sealedPage.SetBuffer(buffer.get());
      source.LoadSealedPage(columnId, RClusterIndex(clusterId, firstElementInPage), sealedPage);
      ASSERT_GE(sealedPage.GetBufferSize(), 12U);
      ASSERT_GE(sealedPage.GetDataSize(), 4U);
      EXPECT_EQ(firstElementInPage, ReadRawInt(sealedPage.GetBuffer()));
      firstElementInPage += pi.fNElements;
   }
}

TEST(RNTupleMerger, MergeSymmetric)
{
   // Write two test ntuples to be merged
   // These files are practically identical except that filed indices are interchanged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 567;
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      RNTupleMerger merger;
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }

   // Now check some information
   // ntuple1 has 10 entries
   // ntuple2 has 10 entries
   // ntuple3 has 20 entries, first 10 identical w/ ntuple1, second 10 identical w/ ntuple2
   {
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
      auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
      auto ntuple3 = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());

      auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("foo");

      auto bar1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto bar3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("bar");

      ntuple1->LoadEntry(1);
      ntuple2->LoadEntry(1);
      ntuple3->LoadEntry(1);
      ASSERT_NE(*foo1, *foo2);
      ASSERT_EQ(*foo1, *foo3);
      ASSERT_NE(*bar1, *bar2);
      ASSERT_EQ(*bar1, *bar3);

      ntuple3->LoadEntry(11);
      ASSERT_EQ(*foo2, *foo3);
      ASSERT_EQ(*bar2, *bar3);

      ntuple1->LoadEntry(9);
      ntuple2->LoadEntry(9);
      ntuple3->LoadEntry(9);
      ASSERT_NE(*foo1, *foo2);
      ASSERT_EQ(*foo1, *foo3);
      ASSERT_NE(*bar1, *bar2);
      ASSERT_EQ(*bar1, *bar3);

      ntuple3->LoadEntry(19);
      ASSERT_EQ(*foo2, *foo3);
      ASSERT_EQ(*bar2, *bar3);
   }
}

TEST(RNTupleMerger, MergeAsymmetric1)
{
   // Write two test ntuples to be merged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      // We expect this to fail in Filter and Strict mode since the fields between the sources do NOT match
      RNTupleMerger merger;
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeAsymmetric2)
{
   // Write two test ntuples to be merged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      // We expect this to fail in Filter and Strict mode since the fields between the sources do NOT match
      RNTupleMerger merger;
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeAsymmetric3)
{
   // Write two test ntuples to be merged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 567;
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      // We expect this to succeed except in all modes except Strict.
      RNTupleMerger merger;
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("Source RNTuple has extra fields"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeVector)
{
   // Write two test ntuples to be merged
   // These files are practically identical except that filed indices are interchanged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::vector<int>>("foo");
      auto fieldBar = model->MakeField<std::vector<int>>("bar");
      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath(), opts);
      for (size_t i = 0; i < 10; ++i) {
         fieldFoo->clear();
         fieldBar->clear();
         fieldFoo->push_back(i * 123);
         fieldFoo->push_back(i * 456);
         fieldBar->push_back(i * 789);
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<std::vector<int>>("bar");
      auto fieldFoo = model->MakeField<std::vector<int>>("foo");
      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath(), opts);
      for (size_t i = 0; i < 10; ++i) {
         fieldFoo->clear();
         fieldBar->clear();
         fieldFoo->push_back(i * 321);
         fieldBar->push_back(i * 654);
         fieldBar->push_back(i * 987);
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
      FileRaii fileGuard3("test_ntuple_merge_out.root");
      {
         // Gather the input sources
         std::vector<std::unique_ptr<RPageSource>> sources;
         sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
         sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
         std::vector<RPageSource *> sourcePtrs;
         for (const auto &s : sources) {
            sourcePtrs.push_back(s.get());
         }

         // Create the output
         auto opts = RNTupleWriteOptions();
         opts.SetCompression(0);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), opts);

         // Now Merge the inputs
         RNTupleMerger merger;
         RNTupleMergeOptions mopts;
         mopts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, *destination, mopts);
         EXPECT_TRUE(bool(res));
      }

      // Now check some information
      // ntuple1 has 10 entries
      // ntuple2 has 10 entries
      // ntuple3 has 20 entries, first 10 identical w/ ntuple1, second 10 identical w/ ntuple2
      {
         auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
         auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
         auto ntuple3 = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
         ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());

         auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("foo");
         auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("foo");
         auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("foo");

         auto bar1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("bar");
         auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("bar");
         auto bar3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("bar");

         ntuple1->LoadEntry(1);
         ntuple2->LoadEntry(1);
         ntuple3->LoadEntry(1);
         ASSERT_NE(foo1->size(), foo2->size());
         ASSERT_EQ(foo1->size(), foo3->size());
         ASSERT_NE(bar1->size(), bar2->size());
         ASSERT_EQ(bar1->size(), bar3->size());
         ASSERT_EQ(foo1->at(0), foo3->at(0));
         ASSERT_EQ(foo1->at(1), foo3->at(1));
         ASSERT_NE(foo1->at(0), foo2->at(0));
         ASSERT_EQ(bar1->at(0), bar3->at(0));
         ASSERT_NE(bar1->at(0), bar2->at(0));
         ASSERT_NE(bar2->at(0), bar3->at(0));

         ntuple3->LoadEntry(11);
         ASSERT_NE(foo1->size(), foo3->size());
         ASSERT_EQ(foo2->size(), foo3->size());
         ASSERT_NE(bar1->size(), bar3->size());
         ASSERT_EQ(bar2->size(), bar3->size());
         ASSERT_NE(foo1->at(0), foo2->at(0));
         ASSERT_NE(foo1->at(0), foo3->at(0));
         ASSERT_EQ(foo2->at(0), foo3->at(0));
         ASSERT_NE(bar1->at(0), bar2->at(0));
         ASSERT_NE(bar1->at(0), bar3->at(0));
         ASSERT_EQ(bar2->at(0), bar3->at(0));
         ASSERT_EQ(bar2->at(1), bar3->at(1));
      }
   }
}

TEST(RNTupleMerger, MergeInconsistentTypes)
{
   // Write two test ntuples to be merged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::string>("foo", "0");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = std::to_string(i * 123);
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<float>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 5.67;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Create the output
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());

      // Now Merge the inputs
      // We expect this to fail since the fields between the sources do NOT match
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         RNTupleMerger merger;
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("type incompatible"));
         }
      }
   }
}

TEST(RNTupleMerger, MergeThroughTFileMerger)
{
   // Write two test ntuples to be merged
   // These files are practically identical except that filed indices are interchanged
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 567;
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Now Merge the inputs through TFileMerger
      TFileMerger merger;
      merger.AddFile(fileGuard1.GetPath().c_str());
      merger.AddFile(fileGuard2.GetPath().c_str());
      merger.OutputFile(fileGuard3.GetPath().c_str());
      merger.PartialMerge(); // Merge closes and deletes the output
   }

   // Now check some information
   // ntuple1 has 10 entries
   // ntuple2 has 10 entries
   // ntuple3 has 20 entries, first 10 identical w/ ntuple1, second 10 identical w/ ntuple2
   {
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
      auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
      auto ntuple3 = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());

      auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("foo");

      auto bar1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto bar3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("bar");

      ntuple1->LoadEntry(1);
      ntuple2->LoadEntry(1);
      ntuple3->LoadEntry(1);
      ASSERT_NE(*foo1, *foo2);
      ASSERT_EQ(*foo1, *foo3);
      ASSERT_NE(*bar1, *bar2);
      ASSERT_EQ(*bar1, *bar3);

      ntuple3->LoadEntry(11);
      ASSERT_EQ(*foo2, *foo3);
      ASSERT_EQ(*bar2, *bar3);

      ntuple1->LoadEntry(9);
      ntuple2->LoadEntry(9);
      ntuple3->LoadEntry(9);
      ASSERT_NE(*foo1, *foo2);
      ASSERT_EQ(*foo1, *foo3);
      ASSERT_NE(*bar1, *bar2);
      ASSERT_EQ(*bar1, *bar3);

      ntuple3->LoadEntry(19);
      ASSERT_EQ(*foo2, *foo3);
      ASSERT_EQ(*bar2, *bar3);
   }
}

TEST(RNTupleMerger, MergeThroughTFileMergerIncremental)
{
   // Write two test ntuples to be merged
   // These files are practically identical except that filed indices are interchanged
   FileRaii fileGuardIn("test_ntuple_merge_in.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuardIn.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuardOut("test_ntuple_merge_out.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuardOut.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 567;
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   {
      // Now Merge the inputs through TFileMerger
      TFileMerger merger;
      merger.AddFile(fileGuardIn.GetPath().c_str());
      merger.OutputFile(fileGuardOut.GetPath().c_str(), "UPDATE");
      merger.PartialMerge(); // Merge closes and deletes the output
   }

   // Now check some information
   // ntupleIn has 10 entries
   // ntupleOut has 20 entries, second 10 identical w/ ntupleIn
   {
      auto ntupleIn = RNTupleReader::Open("ntuple", fileGuardIn.GetPath());
      auto ntupleOut = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
      ASSERT_EQ(2 * ntupleIn->GetNEntries(), ntupleOut->GetNEntries());

      auto fooIn = ntupleIn->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto fooOut = ntupleOut->GetModel().GetDefaultEntry().GetPtr<int>("foo");

      auto barIn = ntupleIn->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto barOut = ntupleOut->GetModel().GetDefaultEntry().GetPtr<int>("bar");

      ntupleIn->LoadEntry(1);
      ntupleOut->LoadEntry(1);
      ASSERT_NE(*fooIn, *fooOut);
      ASSERT_EQ(*fooOut, 567);
      ASSERT_NE(*barIn, *barOut);
      ASSERT_EQ(*barOut, 765);

      ntupleOut->LoadEntry(11);
      ASSERT_EQ(*fooIn, *fooOut);
      ASSERT_EQ(*barIn, *barOut);

      ntupleIn->LoadEntry(9);
      ntupleOut->LoadEntry(9);
      ASSERT_NE(*fooIn, *fooOut);
      ASSERT_EQ(*fooOut, 9 * 567);
      ASSERT_NE(*barIn, *barOut);
      ASSERT_EQ(*barOut, 9 * 765);

      ntupleOut->LoadEntry(19);
      ASSERT_EQ(*fooIn, *fooOut);
      ASSERT_EQ(*barIn, *barOut);
   }
}

TEST(RNTupleMerger, MergeThroughTFileMergerKey)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "TFileMerger", "Merging RNTuples is experimental");
   diags.requiredDiag(kError, "RNTuple::Merge", "Output file already has key, but not of type RNTuple!");
   diags.requiredDiag(kError, "TFileMerger", "Could NOT merge RNTuples!");
   diags.requiredDiag(kError, "TFileMerger", "error during merge of your ROOT files");

   // Write an ntuple to be merged, but the output file already has a key of the same name.
   FileRaii fileGuardIn("test_ntuple_merge_in.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuardIn.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuardOut("test_ntuple_merge_out.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuardOut.GetPath().c_str(), "RECREATE"));
      std::string ntuple = "ntuple";
      file->WriteObject(&ntuple, ntuple.c_str());
   }

   // Now merge the input
   {
      TFileMerger merger;
      merger.AddFile(fileGuardIn.GetPath().c_str());
      merger.OutputFile(fileGuardOut.GetPath().c_str(), "UPDATE");
      merger.PartialMerge();
   }
}

TEST(RNTupleMerger, MergeThroughTBufferMerger)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "TFileMerger", "Merging RNTuples is experimental");
   diags.requiredDiag(kWarning, "TBufferMergerFile", "not attached to the directory", false);

   FileRaii fileGuard("test_ntuple_merge_TBufferMerger.root");

   static constexpr int NumFiles = 10;
   {
      ROOT::TBufferMerger merger(fileGuard.GetPath().c_str());

      for (int i = 0; i < NumFiles; i++) {
         auto file1 = merger.GetFile();

         auto model = RNTupleModel::Create();
         auto pt = model->MakeField<float>("pt", 42.0);
         auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file1);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNClusters(), 10);
   EXPECT_EQ(reader->GetNEntries(), 10);
}

static bool VerifyValidZLIB(const void *buf, size_t bufsize, size_t tgtsize)
{
   // Mostly copy-pasted code from R__unzipZLIB
   auto tgt = std::make_unique<Bytef[]>(tgtsize);
   auto *src = reinterpret_cast<const uint8_t *>(buf);
   const auto HDRSIZE = 9;
   z_stream stream = {};
   stream.next_in = (Bytef *)(&src[HDRSIZE]);
   stream.avail_in = (uInt)bufsize - HDRSIZE;
   stream.next_out = tgt.get();
   stream.avail_out = (uInt)tgtsize;

   auto is_valid_header_zlib = [](const uint8_t *s) { return s[0] == 'Z' && s[1] == 'L' && s[2] == Z_DEFLATED; };
   if (!is_valid_header_zlib(src))
      return false;

   int err = inflateInit(&stream);
   if (err != Z_OK)
      return false;

   while ((err = inflate(&stream, Z_FINISH)) != Z_STREAM_END) {
      EXPECT_EQ(err, Z_OK);
      if (err != Z_OK)
         return false;
   }

   inflateEnd(&stream);

   return true;
}

enum class PageCompCheckType { kUncompressed, kZlib };

static bool VerifyPageCompression(const std::string_view fileName, PageCompCheckType checkType)
{
   // TODO(gparolini): eventually we want to do the following check:
   //   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   //   auto compSettings = reader->GetDescriptor().GetClusterDescriptor(0).GetColumnRange(0).fCompressionSettings;
   //   EXPECT_EQ(compSettings, kNewComp);
   // but right now we don't write the correct metadata when calling Merge() so we can't trust the advertised
   // compression settings to reflect the actual algorithm being used for compression.
   // Therefore, for now we do a more expensive check where we try to unzip the data using the expected
   // algorithm and verify that it works.
   auto source = RPageSource::Create("ntuple", fileName);
   source->Attach();
   auto descriptor = source->GetSharedDescriptorGuard();
   const auto &columnDesc = descriptor->GetColumnDescriptor(0);
   const auto colElement = ROOT::Experimental::Internal::RColumnElementBase::Generate(columnDesc.GetType());
   ROOT::Experimental::Internal::RPageStorage::RSealedPage sealedPage;
   source->LoadSealedPage(0, {0, 0}, sealedPage);
   auto buffer = std::make_unique<unsigned char[]>(sealedPage.GetBufferSize());
   sealedPage.SetBuffer(buffer.get());
   source->LoadSealedPage(0, {0, 0}, sealedPage);

   size_t uncompSize = sealedPage.GetNElements() * colElement->GetSize();
   if (checkType == PageCompCheckType::kZlib)
      return VerifyValidZLIB(sealedPage.GetBuffer(), sealedPage.GetDataSize(), uncompSize);
   else
      return sealedPage.GetDataSize() == uncompSize;
}

TEST(RNTupleMerger, ChangeCompression)
{
   FileRaii fileGuard("test_ntuple_merge_changecomp_in.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      for (size_t i = 0; i < 1000; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   constexpr auto kNewComp = 101;
   FileRaii fileGuardOutChecksum("test_ntuple_merge_changecomp_out.root");
   FileRaii fileGuardOutNoChecksum("test_ntuple_merge_changecomp_out_nock.root");
   FileRaii fileGuardOutUncomp("test_ntuple_merge_changecomp_out_uncomp.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Create the output
      auto writeOpts = RNTupleWriteOptions{};
      writeOpts.SetEnablePageChecksums(true);
      auto destinationChecksum = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutChecksum.GetPath(), writeOpts);
      auto destinationUncomp = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutUncomp.GetPath(), writeOpts);
      writeOpts.SetEnablePageChecksums(false);
      auto destinationNoChecksum =
         std::make_unique<RPageSinkFile>("ntuple", fileGuardOutNoChecksum.GetPath(), writeOpts);

      RNTupleMerger merger;
      auto opts = RNTupleMergeOptions{};
      opts.fCompressionSettings = kNewComp;
      merger.Merge(sourcePtrs, *destinationChecksum, opts);
      merger.Merge(sourcePtrs, *destinationNoChecksum, opts);
      opts.fCompressionSettings = 0;
      merger.Merge(sourcePtrs, *destinationUncomp, opts);
   }

   // Check that compression is the right one
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutChecksum.GetPath(), PageCompCheckType::kZlib));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutNoChecksum.GetPath(), PageCompCheckType::kZlib));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutUncomp.GetPath(), PageCompCheckType::kUncompressed));
}

TEST(RNTupleMerger, MergeLateModelExtension)
{
   // Write two test ntuples to be merged, with different models.
   // Use EMergingMode::kUnion so the output ntuple has all the fields of its inputs.
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::unordered_map<std::string, int>>("foo", 0);
      auto fieldVfoo = model->MakeField<std::vector<int>>("vfoo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath(), RNTupleWriteOptions());
      for (size_t i = 0; i < 10; ++i) {
         fieldFoo->insert(std::make_pair(std::to_string(i), i * 123));
         *fieldVfoo = {(int)i * 123};
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBaz = model->MakeField<int>("baz", 0);
      auto fieldFoo = model->MakeField<std::unordered_map<std::string, int>>("foo", 0);
      auto fieldVfoo = model->MakeField<std::vector<int>>("vfoo", 0);
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath(), wopts);
      for (size_t i = 0; i < 10; ++i) {
         *fieldBaz = i * 567;
         fieldFoo->insert(std::make_pair(std::to_string(i), i * 765));
         *fieldVfoo = {(int)i * 765};
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Create the output
      RNTupleWriteOptions wopts;
      wopts.SetCompression(0);
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);

      // Now Merge the inputs
      RNTupleMerger merger;
      auto opts = RNTupleMergeOptions{};
      opts.fCompressionSettings = 0;
      opts.fMergingMode = ENTupleMergingMode::kUnion;
      auto res = merger.Merge(sourcePtrs, *destination, opts);
      EXPECT_TRUE(bool(res));
   }

   {
      auto ntuple = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      auto foo = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_map<std::string, int>>("foo");
      auto vfoo = ntuple->GetModel().GetDefaultEntry().GetPtr<std::vector<int>>("vfoo");
      auto bar = ntuple->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto baz = ntuple->GetModel().GetDefaultEntry().GetPtr<int>("baz");

      for (int i = 0; i < 10; ++i) {
         ntuple->LoadEntry(i);
         ASSERT_EQ((*foo)[std::to_string(i)], i * 123);
         ASSERT_EQ((*vfoo)[0], i * 123);
         ASSERT_EQ(*bar, i * 321);
         ASSERT_EQ(*baz, 0);
      }
      for (int i = 10; i < 20; ++i) {
         ntuple->LoadEntry(i);
         ASSERT_EQ((*foo)[std::to_string(i - 10)], (i - 10) * 765);
         ASSERT_EQ((*vfoo)[0], (i - 10) * 765);
         ASSERT_EQ(*bar, 0);
         ASSERT_EQ(*baz, (i - 10) * 567);
      }
   }
}

TEST(RNTupleMerger, MergeCompression)
{
   // Verify that the compression of the output rntuple is the one we ask for
   FileRaii fileGuard1("test_ntuple_merge_comp_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto writeOpts = RNTupleWriteOptions();
      writeOpts.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath(), writeOpts);
      for (size_t i = 0; i < 100; ++i) {
         *fieldFoo = i * 123;
         *fieldBar = i * 321;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_comp_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar", 0);
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto writeOpts = RNTupleWriteOptions();
      writeOpts.SetCompression(404);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath(), writeOpts);
      for (size_t i = 0; i < 100; ++i) {
         *fieldFoo = i * 567;
         *fieldBar = i * 765;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   const auto kOutCompSettings = 101;
   FileRaii fileGuard3("test_ntuple_merge_comp_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      RNTupleMerger merger;
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         opts.fCompressionSettings = kOutCompSettings;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }

   EXPECT_TRUE(VerifyPageCompression(fileGuard3.GetPath(), PageCompCheckType::kZlib));

   {
      FileRaii fileGuard4("test_ntuple_merge_comp_out_tfilemerger.root");
      auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
      auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
      TFileMerger fileMerger(kFALSE, kFALSE);
      fileMerger.OutputFile(fileGuard4.GetPath().c_str(), "RECREATE", kOutCompSettings);
      fileMerger.AddFile(nt1.get());
      fileMerger.AddFile(nt2.get());
      fileMerger.Merge();

      EXPECT_TRUE(VerifyPageCompression(fileGuard4.GetPath(), PageCompCheckType::kZlib));
   }
}

TEST(RNTupleMerger, DifferentCompatibleRepresentations)
{
   // Verify that we can merge two RNTuples with fields that have different, but compatible, column representations.
   FileRaii fileGuard1("test_ntuple_merge_diff_rep_in_1.root");

   auto model = RNTupleModel::Create();
   auto pFoo = model->MakeField<double>("foo", 0);
   auto clonedModel = model->Clone();
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_diff_rep_in_2.root");

   {
      auto &fieldFooDbl = clonedModel->GetMutableField("foo");
      fieldFooDbl.SetColumnRepresentatives({{EColumnType::kReal32}});
      auto ntuple = RNTupleWriter::Recreate(std::move(clonedModel), "ntuple", fileGuard2.GetPath());
      auto e = ntuple->CreateEntry();
      auto pFoo2 = e->GetPtr<double>("foo");
      for (size_t i = 0; i < 10; ++i) {
         *pFoo2 = i * 567;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_diff_rep_out1.root");
   FileRaii fileGuard4("test_ntuple_merge_diff_rep_out2.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      auto sourcePtrs2 = sourcePtrs;

      // Now Merge the inputs. Do both with and without compression change
      RNTupleMerger merger;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         // TODO(gparolini): we want to support this in the future
         EXPECT_FALSE(bool(res));
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("different column type"));
         }
         // EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard4.GetPath(), RNTupleWriteOptions());
         auto res = merger.Merge(sourcePtrs, *destination);
         // TODO(gparolini): we want to support this in the future
         EXPECT_FALSE(bool(res));
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("different column type"));
         }
         // EXPECT_TRUE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MultipleRepresentations)
{
   // verify that we properly handle ntuples with multiple column representations
   FileRaii fileGuard1("test_ntuple_merge_multirep_in_1.root");

   {
      auto model = RNTupleModel::Create();
      auto fldPx = RFieldBase::Create("px", "float").Unwrap();
      fldPx->SetColumnRepresentatives({{EColumnType::kReal32}, {EColumnType::kReal16}});
      model->AddField(std::move(fldPx));
      auto ptrPx = model->GetDefaultEntry().GetPtr<float>("px");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      *ptrPx = 1.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Experimental::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
         const_cast<RFieldBase &>(writer->GetModel().GetConstField("px")), 1);
      *ptrPx = 2.0;
      writer->Fill();
   }

   // Now merge the inputs
   FileRaii fileGuard2("test_ntuple_merge_multirep_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      auto sourcePtrs2 = sourcePtrs;

      RNTupleMerger merger;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard2.GetPath(), RNTupleWriteOptions());
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         // TODO(gparolini): we want to support this in the future
         // XXX: this currently fails because of a mismatch in the number of columns of dst vs src.
         // Is this correct? Anyway the situation will likely change once we properly support different representation
         // indices...
         EXPECT_FALSE(bool(res));
         // EXPECT_TRUE(bool(res));
      }
   }
}

TEST(RNTupleMerger, Double32)
{
   // Verify that we can merge two RNTuples with fields that have different, but compatible, column representations.
   FileRaii fileGuard1("test_ntuple_merge_d32_in_1.root");

   {
      auto model = RNTupleModel::Create();
      auto pFoo = model->MakeField<Double32_t>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_d32_in_2.root");

   {
      auto model = RNTupleModel::Create();
      auto pFoo = model->MakeField<double>("foo", 0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pFoo = i * 321;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_d32_out1.root");
   FileRaii fileGuard4("test_ntuple_merge_d32_out2.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      auto sourcePtrs2 = sourcePtrs;

      // Now Merge the inputs. Do both with and without compression change
      RNTupleMerger merger;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto ntuple = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
         auto foo = ntuple->GetModel().GetDefaultEntry().GetPtr<Double32_t>("foo");

         for (int i = 0; i < 10; ++i) {
            ntuple->LoadEntry(i);
            ASSERT_DOUBLE_EQ(*foo, i * 123);
         }
         for (int i = 10; i < 20; ++i) {
            ntuple->LoadEntry(i);
            ASSERT_DOUBLE_EQ(*foo, (i - 10) * 321);
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard4.GetPath(), RNTupleWriteOptions());
         auto res = merger.Merge(sourcePtrs, *destination);
         EXPECT_TRUE(bool(res));
      }
      {
         auto ntuple = RNTupleReader::Open("ntuple", fileGuard4.GetPath());
         auto foo = ntuple->GetModel().GetDefaultEntry().GetPtr<double>("foo");

         for (int i = 0; i < 10; ++i) {
            ntuple->LoadEntry(i);
            ASSERT_DOUBLE_EQ(*foo, i * 123);
         }
         for (int i = 10; i < 20; ++i) {
            ntuple->LoadEntry(i);
            ASSERT_DOUBLE_EQ(*foo, (i - 10) * 321);
         }
      }
   }
}

TEST(RNTupleMerger, MergeProjectedFields)
{
   // Verify that the projected fields get treated properly by the merge (i.e. we don't try and merge the alias columns
   // but we preserve the projections)
   FileRaii fileGuard1("test_ntuple_merge_proj_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo", 0);
      auto projBar = RFieldBase::Create("bar", "int").Unwrap();
      model->AddProjectedField(std::move(projBar), [](const std::string &) { return "foo"; });
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_proj_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now Merge the inputs
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard2.GetPath(), RNTupleWriteOptions());
      RNTupleMerger merger;
      auto res = merger.Merge(sourcePtrs, *destination);
      EXPECT_TRUE(bool(res));
   }

   {
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
      auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
      ASSERT_EQ(ntuple1->GetNEntries() + ntuple1->GetNEntries(), ntuple2->GetNEntries());

      auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("foo");
      auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("foo");

      auto bar1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("bar");
      auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("bar");

      for (auto i = 0u; i < ntuple2->GetNEntries(); ++i) {
         ntuple1->LoadEntry(i % ntuple1->GetNEntries());
         ntuple2->LoadEntry(i);
         ASSERT_EQ(*foo1, *foo2);
         ASSERT_EQ(*bar1, *bar2);
      }
   }
}
