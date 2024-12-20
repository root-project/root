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
   auto wrPt = model->MakeField<std::int32_t>("pt");
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      options.SetMaxUnzippedPageSize(4096);
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      *wrPt = 42;
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBar = model->MakeField<int>("bar");
      auto fieldFoo = model->MakeField<int>("foo");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldFoo = model->MakeField<std::string>("foo");
      *fieldFoo = "0";
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = std::to_string(i * 123);
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<float>("foo");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBar = model->MakeField<int>("bar");
      auto fieldFoo = model->MakeField<int>("foo");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBar = model->MakeField<int>("bar");
      auto fieldFoo = model->MakeField<int>("foo");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
         *model->MakeField<float>("pt") = 42.0;
         auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file1);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNClusters(), 10);
   EXPECT_EQ(reader->GetNEntries(), 10);
}

static bool VerifyPageCompression(const std::string_view fileName, int expectedComp)
{
   // Check that the advertised compression is correct
   bool ok = true;
   {
      auto reader = RNTupleReader::Open("ntuple", fileName);
      auto compSettings = reader->GetDescriptor().GetClusterDescriptor(0).GetColumnRange(0).fCompressionSettings;
      if (compSettings != expectedComp) {
         std::cerr << "Advertised compression is wrong: " << compSettings << " instead of " << expectedComp << "\n";
         ok = false;
      }
   }

   // Check that the actual compression is correct
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

   // size_t uncompSize = sealedPage.GetNElements() * colElement->GetSize();
   int compAlgo = R__getCompressionAlgorithm((const unsigned char *)sealedPage.GetBuffer(), sealedPage.GetDataSize());
   if (compAlgo == ROOT::RCompressionSetting::EAlgorithm::kUndefined)
      compAlgo = 0;
   if (compAlgo != (expectedComp / 100)) {
      std::cerr << "Actual compression is wrong: " << compAlgo << " instead of " << (expectedComp / 100) << "\n";
      ok = false;
   }
   return ok;
}

TEST(RNTupleMerger, ChangeCompression)
{
   FileRaii fileGuard("test_ntuple_merge_changecomp_in.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      for (size_t i = 0; i < 1000; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   constexpr auto kNewComp = 404;
   FileRaii fileGuardOutChecksum("test_ntuple_merge_changecomp_out.root");
   FileRaii fileGuardOutNoChecksum("test_ntuple_merge_changecomp_out_nock.root");
   FileRaii fileGuardOutDiffComp("test_ntuple_merge_changecomp_out_diff.root");
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
      auto destinationDifferentComp =
         std::make_unique<RPageSinkFile>("ntuple", fileGuardOutDiffComp.GetPath(), writeOpts);
      writeOpts.SetCompression(kNewComp);
      auto destinationChecksum = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutChecksum.GetPath(), writeOpts);
      auto destinationNoChecksum =
         std::make_unique<RPageSinkFile>("ntuple", fileGuardOutNoChecksum.GetPath(), writeOpts);
      writeOpts.SetCompression(0);
      auto destinationUncomp = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutUncomp.GetPath(), writeOpts);
      writeOpts.SetEnablePageChecksums(false);

      RNTupleMerger merger;
      auto opts = RNTupleMergeOptions{};
      opts.fCompressionSettings = kNewComp;
      // This should fail because we specified a different compression than the sink
      auto res = merger.Merge(sourcePtrs, *destinationDifferentComp, opts);
      EXPECT_FALSE(bool(res));
      res = merger.Merge(sourcePtrs, *destinationChecksum, opts);
      EXPECT_TRUE(bool(res));
      res = merger.Merge(sourcePtrs, *destinationNoChecksum, opts);
      EXPECT_TRUE(bool(res));
      opts.fCompressionSettings = 0;
      res = merger.Merge(sourcePtrs, *destinationUncomp, opts);
      EXPECT_TRUE(bool(res));
   }

   // Check that compression is the right one
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutChecksum.GetPath(), kNewComp));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutNoChecksum.GetPath(), kNewComp));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutUncomp.GetPath(), 0));
}

TEST(RNTupleMerger, ChangeCompressionMixed)
{
   FileRaii fileGuard("test_ntuple_merge_changecomp_mixed_in.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::string>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      // Craft the input so that we have one column that ends up compressed (the indices) and one that is not (the
      // chars)
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = (char)(i + 'A');
         ntuple->Fill();
      }
   }

   FileRaii fileGuardOutChecksum("test_ntuple_merge_changecomp_mixed_out.root");
   FileRaii fileGuardOutDiffComp("test_ntuple_merge_changecomp_mixed_out_diff.root");
   FileRaii fileGuardOutNoChecksum("test_ntuple_merge_changecomp_mixed_out_nock.root");
   FileRaii fileGuardOutUncomp("test_ntuple_merge_changecomp_mixed_out_uncomp.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Create the output
      auto writeOpts = RNTupleWriteOptions{};
      writeOpts.SetEnablePageChecksums(true);
      auto destinationChecksum = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutChecksum.GetPath(), writeOpts);
      auto destinationNoChecksum =
         std::make_unique<RPageSinkFile>("ntuple", fileGuardOutNoChecksum.GetPath(), writeOpts);
      writeOpts.SetCompression(101);
      auto destinationDifferentComp =
         std::make_unique<RPageSinkFile>("ntuple", fileGuardOutDiffComp.GetPath(), writeOpts);
      writeOpts.SetCompression(0);
      auto destinationUncomp = std::make_unique<RPageSinkFile>("ntuple", fileGuardOutUncomp.GetPath(), writeOpts);
      writeOpts.SetEnablePageChecksums(false);

      RNTupleMerger merger;
      auto opts = RNTupleMergeOptions{};
      auto res = merger.Merge(sourcePtrs, *destinationChecksum, opts);
      EXPECT_TRUE(bool(res));
      res = merger.Merge(sourcePtrs, *destinationNoChecksum, opts);
      EXPECT_TRUE(bool(res));
      opts.fCompressionSettings = 101;
      res = merger.Merge(sourcePtrs, *destinationDifferentComp, opts);
      EXPECT_TRUE(bool(res));
      opts.fCompressionSettings = 0;
      res = merger.Merge(sourcePtrs, *destinationUncomp, opts);
      EXPECT_TRUE(bool(res));
   }

   // Check that compression is the right one
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutChecksum.GetPath(), 505));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutNoChecksum.GetPath(), 505));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutDiffComp.GetPath(), 101));
   EXPECT_TRUE(VerifyPageCompression(fileGuardOutUncomp.GetPath(), 0));
}

TEST(RNTupleMerger, MergeLateModelExtension)
{
   // Write two test ntuples to be merged, with different models.
   // Use EMergingMode::kUnion so the output ntuple has all the fields of its inputs.
   FileRaii fileGuard1("test_ntuple_merge_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::unordered_map<std::string, int>>("foo");
      auto fieldVfoo = model->MakeField<std::vector<int>>("vfoo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBaz = model->MakeField<int>("baz");
      auto fieldFoo = model->MakeField<std::unordered_map<std::string, int>>("foo");
      auto fieldVfoo = model->MakeField<std::vector<int>>("vfoo");
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
      auto fieldFoo = model->MakeField<int>("foo");
      auto fieldBar = model->MakeField<int>("bar");
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
      auto fieldBar = model->MakeField<int>("bar");
      auto fieldFoo = model->MakeField<int>("foo");
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
   const auto kOutCompSettings = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
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
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(kOutCompSettings);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         opts.fCompressionSettings = kOutCompSettings;
         auto res = merger.Merge(sourcePtrs, *destination, opts);
         EXPECT_TRUE(bool(res));
      }
   }

   EXPECT_TRUE(VerifyPageCompression(fileGuard3.GetPath(), kOutCompSettings));

   {
      FileRaii fileGuard4("test_ntuple_merge_comp_out_tfilemerger.root");
      auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
      auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
      TFileMerger fileMerger(kFALSE, kFALSE);
      fileMerger.OutputFile(fileGuard4.GetPath().c_str(), "RECREATE", kOutCompSettings);
      fileMerger.AddFile(nt1.get());
      fileMerger.AddFile(nt2.get());
      fileMerger.Merge();

      EXPECT_TRUE(VerifyPageCompression(fileGuard4.GetPath(), kOutCompSettings));
   }
}

TEST(RNTupleMerger, DifferentCompatibleRepresentations)
{
   // Verify that we can merge two RNTuples with fields that have different, but compatible, column representations.
   FileRaii fileGuard1("test_ntuple_merge_diff_rep_in_1.root");

   auto model = RNTupleModel::Create();
   auto pFoo = model->MakeField<double>("foo");
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
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(0);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
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
      auto pFoo = model->MakeField<Double32_t>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pFoo = i * 123;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_d32_in_2.root");

   {
      auto model = RNTupleModel::Create();
      auto pFoo = model->MakeField<double>("foo");
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
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(0);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
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
      auto fieldFoo = model->MakeField<int>("foo");
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

struct RNTupleMergerCheckEncoding : public ::testing::TestWithParam<std::tuple<int, int, int, bool>> {};

TEST_P(RNTupleMergerCheckEncoding, CorrectEncoding)
{
   const auto [compInput0, compInput1, compOutput, useDefaultComp] = GetParam();
   int expectedComp = useDefaultComp ? 505 : compOutput;

   // Verify that if the encoding of the inputs' fields is properly converted to match the output file's compression
   // (e.g. if we merge a compressed RNTuple with SplitInts and output to an uncompressed one, these should map to
   // Ints).
   FileRaii fileGuard1("test_ntuple_merger_enc_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldInt = model->MakeField<int>("int");
      auto fieldFloat = model->MakeField<float>("float");
      auto fieldVec = model->MakeField<std::vector<size_t>>("vec");
      auto writeOpts = RNTupleWriteOptions();
      writeOpts.SetCompression(compInput0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath(), writeOpts);
      for (size_t i = 0; i < 100; ++i) {
         *fieldInt = i * 123;
         *fieldFloat = i * 123;
         *fieldVec = std::vector<size_t>{i, 2 * i, 3 * i};
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merger_enc_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFloat = model->MakeField<float>("float");
      auto fieldVec = model->MakeField<std::vector<size_t>>("vec");
      auto fieldInt = model->MakeField<int>("int");
      auto writeOpts = RNTupleWriteOptions();
      writeOpts.SetCompression(compInput1);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath(), writeOpts);
      for (size_t i = 0; i < 100; ++i) {
         *fieldInt = i * 567;
         *fieldFloat = i * 567;
         *fieldVec = std::vector<size_t>{4 * i, 5 * i, 6 * i};
         ntuple->Fill();
      }
   }

   FileRaii fileGuard3("test_ntuple_merger_enc_out_3.root");
   {
      auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
      auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
      TFileMerger fileMerger(kFALSE, kFALSE);
      fileMerger.OutputFile(fileGuard3.GetPath().c_str(), "RECREATE", compOutput);
      fileMerger.AddFile(nt1.get());
      fileMerger.AddFile(nt2.get());
      // If `useDefaultComp` is true, it's as if we were calling hadd without a -f* flag
      if (useDefaultComp)
         fileMerger.SetMergeOptions(TString("default_compression"));
      fileMerger.Merge();

      EXPECT_TRUE(VerifyPageCompression(fileGuard3.GetPath(), expectedComp));
   }

   {
      auto reader = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      auto pInt = reader->GetView<int>("int");
      auto pFloat = reader->GetView<float>("float");
      auto pVec = reader->GetView<std::vector<size_t>>("vec");

      for (size_t i = 0; i < 100; ++i) {
         EXPECT_EQ(pInt(i), i * 123);
         EXPECT_FLOAT_EQ(pFloat(i), i * 123);
         std::vector<size_t> v{i, 2 * i, 3 * i};
         EXPECT_EQ(pVec(i), v);
      }
      for (size_t j = 100; j < 200; ++j) {
         size_t i = j - 100;
         EXPECT_EQ(pInt(j), i * 567);
         EXPECT_FLOAT_EQ(pFloat(j), i * 567);
         std::vector<size_t> v{4 * i, 5 * i, 6 * i};
         EXPECT_EQ(pVec(j), v);
      }
   }
}

INSTANTIATE_TEST_SUITE_P(Seq, RNTupleMergerCheckEncoding,
                         ::testing::Combine(
                            // compression of source 1
                            ::testing::Values(0, 101, 207, 404, 505),
                            // compression of source 2
                            ::testing::Values(0, 101, 207, 404, 505),
                            // compression of output TFile
                            ::testing::Values(0, 101, 207, 404, 505),
                            // use default compression
                            ::testing::Values(true, false)));
