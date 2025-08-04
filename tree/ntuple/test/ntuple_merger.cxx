#include "ntuple_test.hxx"

#include <TFileMerger.h>
#include <ROOT/TBufferMerger.hxx>
#include <gtest/gtest.h>
#include <string_view>
#include <unordered_map>
#include <zlib.h>
#include "gmock/gmock.h"
#include <TTree.h>
#include <TRandom3.h>

using ROOT::TestSupport::CheckDiagsRAII;

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
   RNTupleLocalIndex index(source.GetSharedDescriptorGuard()->FindClusterId(columnId, 0), 0);
   RPageStorage::RSealedPage sealedPage;
   source.LoadSealedPage(columnId, index, sealedPage);
   ASSERT_EQ(1U, sealedPage.GetNElements());
   ASSERT_EQ(4U, sealedPage.GetDataSize());
   ASSERT_EQ(12U, sealedPage.GetBufferSize());
   auto buffer = MakeUninitArray<unsigned char>(sealedPage.GetBufferSize());
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
   EXPECT_GT(pageRange.GetPageInfos().size(), 1U);
   std::uint32_t firstElementInPage = 0;
   for (const auto &pi : pageRange.GetPageInfos()) {
      sealedPage.SetBuffer(nullptr);
      source.LoadSealedPage(columnId, RNTupleLocalIndex(clusterId, firstElementInPage), sealedPage);
      buffer = MakeUninitArray<unsigned char>(sealedPage.GetBufferSize());
      sealedPage.SetBuffer(buffer.get());
      source.LoadSealedPage(columnId, RNTupleLocalIndex(clusterId, firstElementInPage), sealedPage);
      ASSERT_GE(sealedPage.GetBufferSize(), 12U);
      ASSERT_GE(sealedPage.GetDataSize(), 4U);
      EXPECT_EQ(firstElementInPage, ReadRawInt(sealedPage.GetBuffer()));
      firstElementInPage += pi.GetNElements();
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

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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

      // We expect this to fail in Filter and Strict mode since the fields between the sources do NOT match
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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

      // We expect this to fail in Filter and Strict mode since the fields between the sources do NOT match
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("missing the following field"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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

      // We expect this to succeed except in all modes except Strict.
      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(res);
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("Source RNTuple has extra fields"));
         }
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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

         RNTupleMergeOptions mopts;
         mopts.fMergingMode = mmode;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, mopts);
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
      RNTupleMerger merger{std::move(destination)};

      // We expect this to fail since the fields between the sources do NOT match
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, opts);
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

   // Now merge the inputs through TFileMerger
   FileRaii fileGuard3("test_ntuple_merge_out.root");
   {
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
      // Now merge the inputs through TFileMerger
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
   diags.requiredDiag(kError, "ROOT.NTuple.Merge", "Output file already has key, but not of type RNTuple!");
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

static bool VerifyPageCompression(const std::string_view fileName, std::uint32_t expectedComp)
{
   // Check that the advertised compression is correct
   bool ok = true;
   {
      auto reader = RNTupleReader::Open("ntuple", fileName);
      auto compSettings = *reader->GetDescriptor().GetClusterDescriptor(0).GetColumnRange(0).GetCompressionSettings();
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
   const auto colElement = ROOT::Internal::RColumnElementBase::Generate(columnDesc.GetType());
   RPageStorage::RSealedPage sealedPage;
   source->LoadSealedPage(0, {0, 0}, sealedPage);
   auto buffer = MakeUninitArray<unsigned char>(sealedPage.GetBufferSize());
   sealedPage.SetBuffer(buffer.get());
   source->LoadSealedPage(0, {0, 0}, sealedPage);

   std::uint32_t compAlgo =
      R__getCompressionAlgorithm((const unsigned char *)sealedPage.GetBuffer(), sealedPage.GetDataSize());
   if (compAlgo == ROOT::RCompressionSetting::EAlgorithm::kUndefined)
      compAlgo = 0;
   if (compAlgo != (expectedComp / 100)) {
      if (compAlgo == 0) {
         // This page might be uncompressed because compressing it wouldn't have saved space: check if that's the case.
         const auto nbytesComp =
            RNTupleCompressor::Zip(sealedPage.GetBuffer(), sealedPage.GetDataSize(), expectedComp, buffer.get());
         if (nbytesComp == sealedPage.GetDataSize()) {
            // Yep, the page was uncompressible. Accept that and return true.
            return true;
         }
         // Not good, the page should have been compressed! Fall through and follow the usual error flow.
      }
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

      auto opts = RNTupleMergeOptions{};
      opts.fCompressionSettings = kNewComp;
      {
         RNTupleMerger merger{std::move(destinationDifferentComp)};
         // This should fail because we specified a different compression than the sink
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         RNTupleMerger merger{std::move(destinationChecksum)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         RNTupleMerger merger{std::move(destinationNoChecksum)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         opts.fCompressionSettings = 0;
         RNTupleMerger merger{std::move(destinationUncomp)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
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

      auto opts = RNTupleMergeOptions{};
      {
         RNTupleMerger merger{std::move(destinationChecksum)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         RNTupleMerger merger{std::move(destinationNoChecksum)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         opts.fCompressionSettings = 101;
         RNTupleMerger merger{std::move(destinationDifferentComp)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         opts.fCompressionSettings = 0;
         RNTupleMerger merger{std::move(destinationUncomp)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
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

      auto opts = RNTupleMergeOptions{};
      opts.fCompressionSettings = 0;
      opts.fMergingMode = ENTupleMergingMode::kUnion;
      RNTupleMerger merger{std::move(destination)};
      auto res = merger.Merge(sourcePtrs, opts);
      EXPECT_TRUE(bool(res));
   }

   {
      auto ntuple = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      EXPECT_EQ(ntuple->GetNEntries(), 20);
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

      RNTupleMergeOptions opts;
      {
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(kOutCompSettings);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         opts.fCompressionSettings = kOutCompSettings;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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
      fieldFooDbl.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}});
      auto ntuple = RNTupleWriter::Recreate(std::move(clonedModel), "ntuple", fileGuard2.GetPath());
      auto e = ntuple->CreateEntry();
      auto pFoo2 = e->GetPtr<double>("foo");
      for (size_t i = 0; i < 10; ++i) {
         *pFoo2 = i * 567;
         ntuple->Fill();
      }
   }

   // Now merge the inputs. Do both with and without compression change
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

      {
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(0);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         // TODO(gparolini): we want to support this in the future
         EXPECT_FALSE(bool(res));
         if (res.GetError()) {
            EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("different column type"));
         }
         // EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard4.GetPath(), RNTupleWriteOptions());
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs);
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
      fldPx->SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}, {ROOT::ENTupleColumnType::kReal16}});
      model->AddField(std::move(fldPx));
      auto ptrPx = model->GetDefaultEntry().GetPtr<float>("px");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      *ptrPx = 1.0;
      writer->Fill();
      writer->CommitCluster();
      ROOT::Internal::RFieldRepresentationModifier::SetPrimaryColumnRepresentation(
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

      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard2.GetPath(), RNTupleWriteOptions());
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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

   // Now merge the inputs. Do both with and without compression change
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

      {
         auto wopts = RNTupleWriteOptions();
         wopts.SetCompression(0);
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), wopts);
         auto opts = RNTupleMergeOptions();
         opts.fCompressionSettings = 0;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
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
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs);
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

      // Now merge the inputs
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard2.GetPath(), RNTupleWriteOptions());
      RNTupleMerger merger{std::move(destination)};
      auto res = merger.Merge(sourcePtrs);
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

TEST(RNTupleMerger, MergeProjectedFieldsMultiple)
{
   // Verify that we correctly handle multiple projected fields
   FileRaii fileGuard1("test_ntuple_merge_proj_mul_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldInt = model->MakeField<int>("int");
      auto fieldFlt = model->MakeField<float>("flt");
      auto projIntProj = RFieldBase::Create("intProj", "int").Unwrap();
      model->AddProjectedField(std::move(projIntProj), [](const std::string &) { return "int"; });
      auto projFltProj = RFieldBase::Create("fltProj", "float").Unwrap();
      model->AddProjectedField(std::move(projFltProj), [](const std::string &) { return "flt"; });
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldInt = i * 123;
         *fieldFlt = i * 456;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_proj_mul_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard2.GetPath(), RNTupleWriteOptions());
      RNTupleMerger merger{std::move(destination)};
      auto res = merger.Merge(sourcePtrs);
      EXPECT_TRUE(bool(res));
   }

   {
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
      auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
      ASSERT_EQ(ntuple1->GetNEntries() + ntuple1->GetNEntries(), ntuple2->GetNEntries());
      const auto &desc1 = ntuple1->GetDescriptor();
      const auto nAliasColumns1 = desc1.GetNLogicalColumns() - desc1.GetNPhysicalColumns();
      ASSERT_EQ(nAliasColumns1, 2);
      const auto &desc2 = ntuple2->GetDescriptor();
      const auto nAliasColumns2 = desc2.GetNLogicalColumns() - desc2.GetNPhysicalColumns();
      ASSERT_EQ(nAliasColumns2, 2);

      auto int1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("int");
      auto int2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("int");
      auto intProj1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("intProj");
      auto intProj2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("intProj");

      auto flt1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      auto flt2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      auto fltProj1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<float>("fltProj");
      auto fltProj2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<float>("fltProj");

      for (auto i = 0u; i < ntuple2->GetNEntries(); ++i) {
         ntuple1->LoadEntry(i % ntuple1->GetNEntries());
         ntuple2->LoadEntry(i);
         ASSERT_EQ(*int1, *int2);
         ASSERT_EQ(*intProj1, *intProj2);
         ASSERT_FLOAT_EQ(*flt1, *flt2);
         ASSERT_FLOAT_EQ(*fltProj1, *fltProj2);
         ASSERT_FLOAT_EQ(*fltProj1, *flt1);
         ASSERT_FLOAT_EQ(*fltProj2, *flt2);
      }
   }
}

TEST(RNTupleMerger, MergeProjectedFieldsOnlyFirst)
{
   // Merge two files where the first has a projection and the second doesn't, and verify that we can
   // read the data from the second file with that projection.
   FileRaii fileGuard1("test_ntuple_merge_proj_onlyfirst_in_1.root");
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
   FileRaii fileGuard2("test_ntuple_merge_proj_onlyfirst_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<int>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *fieldFoo = i * 123;
         ntuple->Fill();
      }
   }

   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         FileRaii fileGuardOut("test_ntuple_merge_proj_onlyfirst_out.root");
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         RNTupleMerger merger{std::move(destination)};
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, opts);
         if (mmode != ENTupleMergingMode::kUnion) {
            EXPECT_FALSE(bool(res));
            continue;
         }
         EXPECT_TRUE(bool(res));

         auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
         auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
         auto ntuple3 = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());
         const auto &desc1 = ntuple1->GetDescriptor();
         const auto &desc2 = ntuple2->GetDescriptor();
         const auto &desc3 = ntuple3->GetDescriptor();
         const auto nAliasColumns1 = desc1.GetNLogicalColumns() - desc1.GetNPhysicalColumns();
         const auto nAliasColumns2 = desc2.GetNLogicalColumns() - desc2.GetNPhysicalColumns();
         const auto nAliasColumns3 = desc3.GetNLogicalColumns() - desc3.GetNPhysicalColumns();
         ASSERT_EQ(nAliasColumns1, 1);
         ASSERT_EQ(nAliasColumns2, 0);
         ASSERT_EQ(nAliasColumns3, 1);

         auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("foo");
         auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<int>("foo");
         auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("foo");

         auto bar1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<int>("bar");
         auto bar3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<int>("bar");

         for (auto i = 0u; i < ntuple1->GetNEntries(); ++i) {
            ntuple1->LoadEntry(i);
            ntuple3->LoadEntry(i);
            ASSERT_EQ(*foo1, *foo3);
            ASSERT_EQ(*bar1, *foo3);
            ASSERT_EQ(*bar1, *bar3);
         }
         for (auto i = 0u; i < ntuple2->GetNEntries(); ++i) {
            ntuple2->LoadEntry(i);
            ntuple3->LoadEntry(ntuple1->GetNEntries() + i);
            ASSERT_EQ(*foo2, *foo3);
            // we should be able to read the data from the second ntuple using the projection defined in the first.
            ASSERT_EQ(*foo2, *bar3);
         }
      }
   }
}

TEST(RNTupleMerger, MergeProjectedFieldsOnlySecond)
{
   // Merge two files where the second has a projection and the first doesn't, and verify that we can
   // read the data from the first file with that projection (only in union mode: in filter mode the new
   // projected field won't be added to the output)
   FileRaii fileGuard1("test_ntuple_merge_proj_onlysecond_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::vector<CustomStruct>>("foo");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < 10; ++i) {
         CustomStruct s;
         s.v1.push_back(i);
         s.s = std::to_string(i);
         *fieldFoo = {s};
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_merge_proj_onlysecond_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::vector<CustomStruct>>("foo");
      auto projBar = RFieldBase::Create("bar", "std::vector<CustomStruct>").Unwrap();
      const auto mapping = [](const std::string &name) {
         std::string replaced = name;
         replaced.replace(0, 3, "foo");
         return replaced;
      };
      model->AddProjectedField(std::move(projBar), mapping);
      // Add a second projection to test the case of multiple projections
      auto projBaz = RFieldBase::Create("baz", "std::vector<CustomStruct>").Unwrap();
      model->AddProjectedField(std::move(projBaz), mapping);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < 10; ++i) {
         CustomStruct s;
         s.v2.push_back({(float)i});
         s.b = static_cast<std::byte>(i);
         ntuple->Fill();
      }
   }

   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         FileRaii fileGuardOut("test_ntuple_merge_proj_onlysecond_out.root");
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         RNTupleMerger merger{std::move(destination)};
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, opts);
         if (mmode == ENTupleMergingMode::kStrict) {
            EXPECT_FALSE(bool(res));
            continue;
         }
         EXPECT_TRUE(bool(res));

         auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
         auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
         auto ntuple3 = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());
         const auto &desc1 = ntuple1->GetDescriptor();
         const auto &desc2 = ntuple2->GetDescriptor();
         const auto &desc3 = ntuple3->GetDescriptor();
         const auto nAliasColumns1 = desc1.GetNLogicalColumns() - desc1.GetNPhysicalColumns();
         const auto nAliasColumns2 = desc2.GetNLogicalColumns() - desc2.GetNPhysicalColumns();
         const auto nAliasColumns3 = desc3.GetNLogicalColumns() - desc3.GetNPhysicalColumns();
         ASSERT_EQ(nAliasColumns1, 0);
         // NOTE: CustomStruct decomposes into 10 columns
         ASSERT_EQ(nAliasColumns2, 20);
         ASSERT_EQ(nAliasColumns3, 20 * (mmode == ENTupleMergingMode::kUnion));

         if (mmode == ENTupleMergingMode::kFilter) {
            auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");
            auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");
            auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");

            auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("bar");
            EXPECT_THROW(ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("bar"),
                         ROOT::RException);
            auto baz2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("baz");
            EXPECT_THROW(ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("baz"),
                         ROOT::RException);

            for (auto i = 0u; i < ntuple1->GetNEntries(); ++i) {
               ntuple1->LoadEntry(i);
               ntuple3->LoadEntry(i);
               ASSERT_EQ(*foo1, *foo3);
            }
            for (auto i = 0u; i < ntuple2->GetNEntries(); ++i) {
               ntuple2->LoadEntry(i);
               ntuple3->LoadEntry(ntuple1->GetNEntries() + i);
               ASSERT_EQ(*foo2, *foo3);
               ASSERT_EQ(*bar2, *foo3);
            }
         } else {
            auto foo1 = ntuple1->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");
            auto foo2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");
            auto foo3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("foo");

            auto bar2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("bar");
            auto bar3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("bar");
            auto baz2 = ntuple2->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("baz");
            auto baz3 = ntuple3->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomStruct>>("baz");

            for (auto i = 0u; i < ntuple1->GetNEntries(); ++i) {
               ntuple1->LoadEntry(i);
               ntuple3->LoadEntry(i);
               ASSERT_EQ(*foo1, *foo3);
               // we should be able to read the data from the second ntuple using the projection defined in the first.
               ASSERT_EQ(*foo1, *bar3);
               ASSERT_EQ(*foo1, *baz3);
            }
            for (auto i = 0u; i < ntuple2->GetNEntries(); ++i) {
               ntuple2->LoadEntry(i);
               ntuple3->LoadEntry(ntuple1->GetNEntries() + i);
               ASSERT_EQ(*foo2, *foo3);
               ASSERT_EQ(*bar2, *foo3);
               ASSERT_EQ(*bar2, *bar3);
               ASSERT_EQ(*baz2, *foo3);
               ASSERT_EQ(*baz2, *baz3);
            }
         }
      }
   }
}

TEST(RNTupleMerger, MergeProjectedFieldsDifferent)
{
   // Merge two files where both the first and the second have a projection with the same name, but different source.
   // Should refuse to merge.
   FileRaii fileGuard1("test_ntuple_merge_proj_diff_in_1.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::vector<CustomStruct>>("foo");
      auto fieldBar = model->MakeField<std::vector<CustomStruct>>("bar");
      auto projBaz = RFieldBase::Create("baz", "std::vector<CustomStruct>").Unwrap();
      const auto mapping = [](const std::string &name) {
         std::string replaced = name;
         replaced.replace(0, 3, "bar");
         return replaced;
      };
      model->AddProjectedField(std::move(projBaz), mapping);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < 10; ++i) {
         CustomStruct s;
         s.v1.push_back(i);
         s.s = std::to_string(i);
         *fieldFoo = {s};
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_merge_proj_diff_in_2.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldFoo = model->MakeField<std::vector<CustomStruct>>("foo");
      auto fieldBar = model->MakeField<std::vector<CustomStruct>>("bar");
      auto projBaz = RFieldBase::Create("baz", "std::vector<CustomStruct>").Unwrap();
      const auto mapping = [](const std::string &name) {
         std::string replaced = name;
         replaced.replace(0, 3, "foo");
         return replaced;
      };
      model->AddProjectedField(std::move(projBaz), mapping);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < 10; ++i) {
         CustomStruct s;
         s.v2.push_back({(float)i});
         s.b = static_cast<std::byte>(i);
         ntuple->Fill();
      }
   }

   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         FileRaii fileGuardOut("test_ntuple_merge_proj_diff_out.root");
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         RNTupleMerger merger{std::move(destination)};
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, opts);
         ASSERT_FALSE(bool(res));
         EXPECT_THAT(res.GetError()->GetReport(), testing::HasSubstr("projected to a different field"));
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
         fileMerger.SetMergeOptions(TString("DefaultCompression"));
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

TEST(RNTupleMerger, MergeAsymmetric1TFileMerger)
{
   // Exactly the same test as MergeAsymmetric1, but passing through TFileMerger.

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

      // We expect this to fail in Filter and Strict mode since the fields between the sources do NOT match
      {
         auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
         auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
         TFileMerger fileMerger(kFALSE, kFALSE);
         fileMerger.OutputFile(fileGuard3.GetPath().c_str(), "RECREATE");
         fileMerger.AddFile(nt1.get());
         fileMerger.AddFile(nt2.get());
         fileMerger.SetMergeOptions(TString("rntuple.MergingMode=Filter"));
         CheckDiagsRAII diags;
         diags.requiredDiag(kError, "TFileMerger::Merge", "error during merge", false);
         diags.requiredDiag(kError, "ROOT.NTuple.Merge", "missing the following field", false);
         diags.requiredDiag(kError, "TFileMerger::MergeRecursive", "Could NOT merge RNTuples!", false);
         diags.optionalDiag(kWarning, "TFileMerger::MergeRecursive", "Merging RNTuples is experimental", false);
         auto res = fileMerger.Merge();
         EXPECT_FALSE(res);
      }
      {
         auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
         auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
         TFileMerger fileMerger(kFALSE, kFALSE);
         fileMerger.OutputFile(fileGuard3.GetPath().c_str(), "RECREATE");
         fileMerger.AddFile(nt1.get());
         fileMerger.AddFile(nt2.get());
         fileMerger.SetMergeOptions(TString("rntuple.MergingMode=Strict"));
         CheckDiagsRAII diags;
         diags.requiredDiag(kError, "TFileMerger::Merge", "error during merge", false);
         diags.requiredDiag(kError, "ROOT.NTuple.Merge", "missing the following field", false);
         diags.requiredDiag(kError, "TFileMerger::MergeRecursive", "Could NOT merge RNTuples!", false);
         diags.optionalDiag(kWarning, "TFileMerger::MergeRecursive", "Merging RNTuples is experimental", false);
         auto res = fileMerger.Merge();
         EXPECT_FALSE(res);
      }
      {
         auto nt1 = std::unique_ptr<TFile>(TFile::Open(fileGuard1.GetPath().c_str()));
         auto nt2 = std::unique_ptr<TFile>(TFile::Open(fileGuard2.GetPath().c_str()));
         TFileMerger fileMerger(kFALSE, kFALSE);
         fileMerger.OutputFile(fileGuard3.GetPath().c_str(), "RECREATE");
         fileMerger.AddFile(nt1.get());
         fileMerger.AddFile(nt2.get());
         fileMerger.SetMergeOptions(TString("rntuple.MergingMode=Union"));
         CheckDiagsRAII diags;
         diags.optionalDiag(kWarning, "TFileMerger::MergeRecursive", "Merging RNTuples is experimental", false);
         auto res = fileMerger.Merge();
         EXPECT_TRUE(res);
      }
   }
}

TEST(RNTupleMerger, SkipMissing)
{
   // Try merging various files, some containing RNTuples and some not; verify that we ignore the ones that don't.
   std::vector<FileRaii> fileGuards;
   for (int i = 0; i < 6; ++i) {
      auto &fileGuard =
         fileGuards.emplace_back(std::string("test_ntuple_merge_skipmissing_") + std::to_string(i) + ".root");

      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      if (i % 2) {
         auto model = RNTupleModel::Create();
         auto p = model->MakeField<std::string>("s");
         auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);
         for (int j = 0; j < 10; ++j) {
            *p = std::to_string(j + i);
            writer->Fill();
         }
      } else {
         auto tree = std::make_unique<TTree>("tree", "tree");
         std::string s;
         tree->Branch("s", &s);
         for (int j = 0; j < 10; ++j) {
            s = std::to_string(j + i);
            tree->Fill();
         }
         tree->Write();
      }
   }

   FileRaii fileOut("test_ntuple_merge_skipmissing_out.root");
   TFileMerger merger;
   merger.OutputFile(fileOut.GetPath().c_str());
   for (const auto &file : fileGuards) {
      merger.AddFile(file.GetPath().c_str());
   }

   bool ok = merger.PartialMerge();
   EXPECT_TRUE(ok);
}

struct RNTupleMergerDeferred : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(RNTupleMergerDeferred, MergeSecondDeferred)
{
   // Try merging 2 RNTuples, the first having a regular column "flt" and the second having a deferred column "flt".
   // Verify that the merged file contains the expected values.
   FileRaii fileGuard1("test_ntuple_merge_second_deferred_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_second_deferred_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with regular field "flt"
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt = i;
         writer->Fill();
      }
   }

   // Second RNTuple with deferred field "flt"
   {
      auto model = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = nEntriesPerFile + i;
         writer->Fill();
      }
      auto updater = writer->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt = writer->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = nEntriesPerFile + i;
         *pFlt = nEntriesPerFile + i;
         writer->Fill();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_second_deferred_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = ((int)i >= nEntriesPerFile && (int)i < nEntriesPerFile + firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i >= nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST_P(RNTupleMergerDeferred, MergeSecondDeferredTwoClusters)
{
   // Like MergeSecondDeferred but the deferred column in the second rntuple appears in a different cluster
   FileRaii fileGuard1("test_ntuple_merge_second_deferred_2clusters_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_second_deferred_2clusters_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with regular field "flt"
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt = i;
         writer->Fill();
      }
   }

   // Second RNTuple with deferred field "flt"
   {
      auto model = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = nEntriesPerFile + i;
         writer->Fill();
      }
      writer->CommitCluster();
      auto updater = writer->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt = writer->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = nEntriesPerFile + i;
         *pFlt = nEntriesPerFile + i;
         writer->Fill();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_second_deferred_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = ((int)i >= nEntriesPerFile && (int)i < nEntriesPerFile + firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i >= nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST_P(RNTupleMergerDeferred, MergeSecondDeferredTwoClustersUnaligned)
{
   // Like MergeSecondDeferredTwoClusters but the deferred column is not aligned with cluster boundaries
   FileRaii fileGuard1("test_ntuple_merge_second_deferred_2clusters_unaligned_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_second_deferred_2clusters_unaligned_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto nEntriesPerCluster = 4;
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with regular field "flt"
   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt = i;
         writer->Fill();
         if (i % nEntriesPerCluster == 0) {
            writer->CommitCluster();
         }
      }
   }

   // Second RNTuple with deferred field "flt"
   {
      auto model = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = nEntriesPerFile + i;
         writer->Fill();
         if (i > 0 && i % nEntriesPerCluster == 0) {
            writer->CommitCluster();
         }
      }
      auto updater = writer->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt = writer->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = nEntriesPerFile + i;
         *pFlt = nEntriesPerFile + i;
         writer->Fill();
         if (i % nEntriesPerCluster == 0) {
            writer->CommitCluster();
         }
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_second_deferred_unaligned_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = ((int)i >= nEntriesPerFile && (int)i < nEntriesPerFile + firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i >= nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST_P(RNTupleMergerDeferred, MergeFirstDeferred)
{
   // Try merging 2 RNTuples, the first having a late model extended field "flt" (with a deferred column) and the second
   // having a regular field "flt". Verify that the merged file contains the expected values.
   FileRaii fileGuard1("test_ntuple_merge_deferred_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_deferred_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with deferred field "flt"
   {
      auto model1 = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model1->MakeField<int>("int");
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer1 = RNTupleWriter::Recreate(std::move(model1), "ntuple", fileGuard1.GetPath(), wopts);
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = i;
         writer1->Fill();
      }
      auto updater = writer1->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt1 = writer1->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = i;
         *pFlt1 = i;
         writer1->Fill();
      }
   }

   // Second RNTuple with regular field "flt"
   {
      auto model2 = RNTupleModel::Create();
      auto pFlt2 = model2->MakeField<float>("flt");
      auto writer2 = RNTupleWriter::Recreate(std::move(model2), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt2 = nEntriesPerFile + i;
         writer2->Fill();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_deferred_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = (i < firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i < nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST_P(RNTupleMergerDeferred, MergeFirstDeferredTwoClusters)
{
   // Like MergeFirstDeferred but the deferred column in the first rntuple appears in a different cluster
   FileRaii fileGuard1("test_ntuple_merge_deferred_2clusters_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_deferred_2clusters_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with deferred field "flt"
   {
      auto model1 = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model1->MakeField<int>("int");
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer1 = RNTupleWriter::Recreate(std::move(model1), "ntuple", fileGuard1.GetPath(), wopts);
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = i;
         writer1->Fill();
      }
      writer1->CommitCluster();
      auto updater = writer1->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt1 = writer1->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = i;
         *pFlt1 = i;
         writer1->Fill();
      }
   }

   // Second RNTuple with regular field "flt"
   {
      auto model2 = RNTupleModel::Create();
      auto pFlt2 = model2->MakeField<float>("flt");
      auto writer2 = RNTupleWriter::Recreate(std::move(model2), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt2 = nEntriesPerFile + i;
         writer2->Fill();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_deferred_2clusters_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = (i < firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i < nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST_P(RNTupleMergerDeferred, MergeFirstDeferredTwoClustersUnaligned)
{
   // Like MergeFirstDeferredTwoClusters but the deferred column doesn't align to a cluster boundary
   FileRaii fileGuard1("test_ntuple_merge_deferred_2clusters_unaligned_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_deferred_2clusters_unaligned_in_2.root");

   const auto [nEntriesPerFile, outComp] = GetParam();
   constexpr auto nEntriesPerCluster = 4;
   constexpr auto firstDeferredIdx = 5;
   assert(nEntriesPerFile > firstDeferredIdx);

   // First RNTuple with late model extended field "flt"
   {
      auto model1 = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model1->MakeField<int>("int");
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer1 = RNTupleWriter::Recreate(std::move(model1), "ntuple", fileGuard1.GetPath(), wopts);
      for (int i = 0; i < firstDeferredIdx; ++i) {
         *pInt = i;
         writer1->Fill();
         if (i == nEntriesPerCluster)
            writer1->CommitCluster();
      }
      auto updater = writer1->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt1 = writer1->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = firstDeferredIdx; i < nEntriesPerFile; ++i) {
         *pInt = i;
         *pFlt1 = i;
         writer1->Fill();
         if (i % nEntriesPerCluster == 0)
            writer1->CommitCluster();
      }
   }

   // Second RNTuple with regular field "flt"
   {
      auto model2 = RNTupleModel::Create();
      auto pFlt2 = model2->MakeField<float>("flt");
      auto writer2 = RNTupleWriter::Recreate(std::move(model2), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < nEntriesPerFile; ++i) {
         *pFlt2 = nEntriesPerFile + i;
         writer2->Fill();
         if (i % nEntriesPerCluster == 0)
            writer2->CommitCluster();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_deferred_2clusters_unaligned_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(outComp);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2 * nEntriesPerFile);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = (i < firstDeferredIdx) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = ((int)i < nEntriesPerFile) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

INSTANTIATE_TEST_SUITE_P(Seq, RNTupleMergerDeferred,
                         testing::Combine(
                            // number of entries
                            testing::Values(6, 10, 100),
                            // compression
                            testing::Values(0, 505)));

TEST(RNTupleMerger, MergeDeferredAdvanced)
{
   // Try merging 3 RNTuples, the first 2 having a late-model extended field "flt" (where the second one has a deferred
   // column) and the third having a regular field "flt". Verify that the merged file contains the expected values.
   FileRaii fileGuard1("test_ntuple_merge_deferred_adv_in_1.root");
   FileRaii fileGuard2("test_ntuple_merge_deferred_adv_in_2.root");
   FileRaii fileGuard3("test_ntuple_merge_deferred_adv_in_3.root");

   // First RNTuple with late model extended field "flt" (column is not deferred because it's still entry 0)
   {
      auto model1 = RNTupleModel::Create();
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer1 = RNTupleWriter::Recreate(std::move(model1), "ntuple", fileGuard1.GetPath(), wopts);
      auto updater = writer1->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt1 = writer1->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = 0; i < 10; ++i) {
         *pFlt1 = i;
         writer1->Fill();
      }
   }

   // Second RNTuple with late model extended field "flt"
   {
      auto model2 = RNTupleModel::Create();
      // Add a non-late model extended field so we can write some entries before we extend the model and obtain
      // actual deferred columns in the extension header.
      auto pInt = model2->MakeField<int>("int");
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer2 = RNTupleWriter::Recreate(std::move(model2), "ntuple", fileGuard2.GetPath(), wopts);
      for (int i = 0; i < 5; ++i) {
         *pInt = 10 + i;
         writer2->Fill();
      }
      auto updater = writer2->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("flt", "float").Unwrap());
      updater->CommitUpdate();
      auto pFlt2 = writer2->GetModel().GetDefaultEntry().GetPtr<float>("flt");
      for (int i = 5; i < 10; ++i) {
         *pInt = 10 + i;
         *pFlt2 = 10 + i;
         writer2->Fill();
      }
   }

   // Third RNTuple with regular field "flt"
   {
      auto model3 = RNTupleModel::Create();
      auto pFlt3 = model3->MakeField<float>("flt");
      auto writer3 = RNTupleWriter::Recreate(std::move(model3), "ntuple", fileGuard3.GetPath());
      for (int i = 0; i < 10; ++i) {
         *pFlt3 = 20 + i;
         writer3->Fill();
      }
   }

   // Now merge them
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard3.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   FileRaii fileGuardOut("test_ntuple_merge_deferred_adv_out.root");
   auto wopts = RNTupleWriteOptions();
   wopts.SetCompression(0);
   auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
   RNTupleMerger merger{std::move(destination)};
   auto opts = RNTupleMergeOptions();
   opts.fMergingMode = ENTupleMergingMode::kUnion;
   auto res = merger.Merge(sourcePtrs, opts);
   EXPECT_TRUE(bool(res));

   auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 30);

   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("int");
   auto pFlt = reader->GetModel().GetDefaultEntry().GetPtr<float>("flt");
   for (auto i = 0u; i < reader->GetNEntries(); ++i) {
      reader->LoadEntry(i);
      float expectedFlt = (i >= 10 && i < 15) ? 0 : i;
      EXPECT_FLOAT_EQ(*pFlt, expectedFlt);
      int expectedInt = (i >= 10 && i < 20) * i;
      EXPECT_EQ(*pInt, expectedInt);
   }
}

TEST(RNTupleMerger, MergeIncrementalLMExt)
{
   // Create the input files:
   // File 0: f_0: int
   // File 1: f_0: int, f_1: float
   // File 2: f_0: int, f_1: float, f_2: string
   // File 3: f_0: int, f_1: float, f_2: string, f_3: int
   // ...
   // each file has 5 entries.
   std::vector<FileRaii> inputFiles;
   const auto nInputs = 12;
   auto model = RNTupleModel::Create();
   for (int fileIdx = 0; fileIdx < nInputs; ++fileIdx) {
      auto &fileGuard =
         inputFiles.emplace_back(std::string("test_ntuple_merge_incr_lmext_in_") + std::to_string(fileIdx) + ".root");

      // Each input gets a different model, so we can exercise the late model extension.
      // Just to have some variation, use different types depending on the field index
      const auto fieldName = std::string("f_") + std::to_string(fileIdx);
      switch (fileIdx % 3) {
      case 0: model->MakeField<int>(fieldName); break;
      case 1: model->MakeField<float>(fieldName); break;
      default: model->MakeField<std::string>(fieldName);
      }

      auto writer = RNTupleWriter::Recreate(model->Clone(), "ntpl", fileGuard.GetPath());

      // Fill the RNTuple with nFills per field
      const auto nFills = 5;
      const auto &entry = writer->GetModel().GetDefaultEntry();
      for (int fillIdx = 0; fillIdx < nFills; ++fillIdx) {
         for (int fieldIdx = 0; fieldIdx < fileIdx + 1; ++fieldIdx) {
            const auto fldName = std::string("f_") + std::to_string(fieldIdx);
            switch (fieldIdx % 3) {
            case 0: *entry.GetPtr<int>(fldName) = fileIdx + fillIdx + fieldIdx; break;
            case 1: *entry.GetPtr<float>(fldName) = fileIdx + fillIdx + fieldIdx; break;
            default: *entry.GetPtr<std::string>(fldName) = std::to_string(fileIdx + fillIdx + fieldIdx);
            }
         }
         writer->Fill();
      }
   }

   // Incrementally merge the inputs
   FileRaii fileGuard("test_ntuple_merge_incr_lmext.root");
   const auto compression = 0;

   {
      TFileMerger merger(kFALSE, kFALSE);
      merger.OutputFile(fileGuard.GetPath().c_str(), "RECREATE", compression);
      merger.SetMergeOptions(TString("rntuple.MergingMode=Union"));

      for (int i = 0; i < nInputs; ++i) {
         auto tfile = std::unique_ptr<TFile>(TFile::Open(inputFiles[i].GetPath().c_str(), "READ"));
         merger.AddFile(tfile.get());
         bool result =
            merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kNonResetable | TFileMerger::kKeepCompression);
         ASSERT_TRUE(result);
      }
   }

   // Now verify that the output file contains all the expected data.
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      const auto &desc = reader->GetDescriptor();
      for (int i = 0; i < nInputs; ++i) {
         const auto fieldId = desc.FindFieldId(std::string("f_") + std::to_string(i));
         EXPECT_NE(fieldId, ROOT::kInvalidDescriptorId);
         const auto &fdesc = desc.GetFieldDescriptor(fieldId);
         for (const auto &colId : fdesc.GetLogicalColumnIds()) {
            const auto &cdesc = desc.GetColumnDescriptor(colId);
            EXPECT_EQ(cdesc.GetFirstElementIndex(), (cdesc.GetIndex() == 0) * i * 5);
         }
      }

      // Merging always produces 1 cluster group.
      EXPECT_EQ(desc.GetNClusterGroups(), 1);
      // Merging doesn't change the number of clusters.
      EXPECT_EQ(desc.GetNClusters(), nInputs);

      const auto *xHeader = desc.GetHeaderExtension();
      EXPECT_NE(xHeader, nullptr);
      // All fields but the first should be in the extension header.
      EXPECT_EQ(xHeader->GetNFields(), nInputs - 1);

      RNTupleView<int> v_int[] = {
         reader->GetView<int>("f_0"),
         reader->GetView<int>("f_3"),
         reader->GetView<int>("f_6"),
         reader->GetView<int>("f_9"),
      };
      RNTupleView<float> v_float[] = {
         reader->GetView<float>("f_1"),
         reader->GetView<float>("f_4"),
         reader->GetView<float>("f_7"),
         reader->GetView<float>("f_10"),
      };
      RNTupleView<std::string> v_string[] = {
         reader->GetView<std::string>("f_2"),
         reader->GetView<std::string>("f_5"),
         reader->GetView<std::string>("f_8"),
         reader->GetView<std::string>("f_11"),
      };
      for (auto entryId : reader->GetEntryRange()) {
         int fileIdx = entryId / 5;
         int localEntryId = entryId % 5;

         for (int i = 0; i < nInputs / 3; ++i) {
            auto x0 = v_int[i](entryId);
            int expected_x0 = (entryId >= 15u * i) * (fileIdx + localEntryId + i * 3);
            EXPECT_EQ(x0, expected_x0);

            auto x1 = v_float[i](entryId);
            float expected_x1 = (entryId >= 5 + 15u * i) * (fileIdx + localEntryId + i * 3 + 1);
            EXPECT_FLOAT_EQ(x1, expected_x1);

            auto x2 = v_string[i](entryId);
            std::string expected_x2 =
               (entryId >= 10 + 15u * i) ? std::to_string(fileIdx + localEntryId + i * 3 + 2) : "";
            EXPECT_EQ(x2, expected_x2);
         }
      }
   }
}

TEST(RNTupleMerger, MergeIncrementalLMExtMemFile)
{
   // Same as MergeIncrementalLMExt but using TMemFiles
   std::vector<std::unique_ptr<TMemFile>> inputFiles;
   const auto nInputs = 12;
   auto model = RNTupleModel::Create();
   for (int fileIdx = 0; fileIdx < nInputs; ++fileIdx) {
      auto &file =
         inputFiles.emplace_back(new TMemFile((std::string("memFile_") + std::to_string(fileIdx)).c_str(), "CREATE"));

      // Each input gets a different model, so we can exercise the late model extension.
      // Just to have some variation, use different types depending on the field index
      const auto fieldName = std::string("f_") + std::to_string(fileIdx);
      switch (fileIdx % 3) {
      case 0: model->MakeField<int>(fieldName); break;
      case 1: model->MakeField<float>(fieldName); break;
      default: model->MakeField<std::string>(fieldName);
      }

      auto writer = RNTupleWriter::Append(model->Clone(), "ntpl", *file);

      // Fill the RNTuple with nFills per field
      const auto nFills = 5;
      const auto &entry = writer->GetModel().GetDefaultEntry();
      for (int fillIdx = 0; fillIdx < nFills; ++fillIdx) {
         for (int fieldIdx = 0; fieldIdx < fileIdx + 1; ++fieldIdx) {
            const auto fldName = std::string("f_") + std::to_string(fieldIdx);
            switch (fieldIdx % 3) {
            case 0: *entry.GetPtr<int>(fldName) = fileIdx + fillIdx + fieldIdx; break;
            case 1: *entry.GetPtr<float>(fldName) = fileIdx + fillIdx + fieldIdx; break;
            default: *entry.GetPtr<std::string>(fldName) = std::to_string(fileIdx + fillIdx + fieldIdx);
            }
         }
         writer->Fill();
      }
   }

   // Incrementally merge the inputs
   FileRaii fileGuard("test_ntuple_merge_incr_lmext_memfile.root");
   const auto compression = 505;

   {
      TFileMerger merger(kFALSE, kFALSE);
      merger.OutputFile(fileGuard.GetPath().c_str(), "RECREATE", compression);
      merger.SetMergeOptions(TString("rntuple.MergingMode=Union"));

      for (int i = 0; i < nInputs; ++i) {
         merger.AddFile(inputFiles[i].get());
         bool result =
            merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kNonResetable | TFileMerger::kKeepCompression);
         ASSERT_TRUE(result);
      }
   }

   // Now verify that the output file contains all the expected data.
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      const auto &desc = reader->GetDescriptor();
      for (int i = 0; i < nInputs; ++i) {
         const auto fieldId = desc.FindFieldId(std::string("f_") + std::to_string(i));
         EXPECT_NE(fieldId, ROOT::kInvalidDescriptorId);
         const auto &fdesc = desc.GetFieldDescriptor(fieldId);
         for (const auto &colId : fdesc.GetLogicalColumnIds()) {
            const auto &cdesc = desc.GetColumnDescriptor(colId);
            EXPECT_EQ(cdesc.GetFirstElementIndex(), (cdesc.GetIndex() == 0) * i * 5);
         }
      }

      // Merging always produces 1 cluster group.
      EXPECT_EQ(desc.GetNClusterGroups(), 1);
      // Merging doesn't change the number of clusters.
      EXPECT_EQ(desc.GetNClusters(), nInputs);

      const auto *xHeader = desc.GetHeaderExtension();
      EXPECT_NE(xHeader, nullptr);
      // All fields but the first should be in the extension header.
      EXPECT_EQ(xHeader->GetNFields(), nInputs - 1);

      RNTupleView<int> v_int[] = {
         reader->GetView<int>("f_0"),
         reader->GetView<int>("f_3"),
         reader->GetView<int>("f_6"),
         reader->GetView<int>("f_9"),
      };
      RNTupleView<float> v_float[] = {
         reader->GetView<float>("f_1"),
         reader->GetView<float>("f_4"),
         reader->GetView<float>("f_7"),
         reader->GetView<float>("f_10"),
      };
      RNTupleView<std::string> v_string[] = {
         reader->GetView<std::string>("f_2"),
         reader->GetView<std::string>("f_5"),
         reader->GetView<std::string>("f_8"),
         reader->GetView<std::string>("f_11"),
      };
      for (auto entryId : reader->GetEntryRange()) {
         int fileIdx = entryId / 5;
         int localEntryId = entryId % 5;

         for (int i = 0; i < nInputs / 3; ++i) {
            auto x0 = v_int[i](entryId);
            int expected_x0 = (entryId >= 15u * i) * (fileIdx + localEntryId + i * 3);
            EXPECT_EQ(x0, expected_x0);

            auto x1 = v_float[i](entryId);
            float expected_x1 = (entryId >= 5 + 15u * i) * (fileIdx + localEntryId + i * 3 + 1);
            EXPECT_FLOAT_EQ(x1, expected_x1);

            auto x2 = v_string[i](entryId);
            std::string expected_x2 =
               (entryId >= 10 + 15u * i) ? std::to_string(fileIdx + localEntryId + i * 3 + 2) : "";
            EXPECT_EQ(x2, expected_x2);
         }
      }
   }
}

TEST(RNTupleMerger, MergeLMExtBig)
{
   // Merge 2 RNTuples where the first one contains an extra field and the second one contains lots of elements.
   // This is meant to stress test the generation of zero pages when late model extending the merged rntuple.
   FileRaii fileGuard1("test_ntuple_merge_lmext_big_1.root");
   FileRaii fileGuard2("test_ntuple_merge_lmext_big_2.root");
   FileRaii fileGuardOut("test_ntuple_merge_lmext_big_out.root");

   {
      auto model = RNTupleModel::Create();
      auto pFoo = model->MakeField<double>("foo");
      auto pBar = model->MakeField<double>("bar");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (int i = 0; i < 50; ++i) {
         *pFoo = i;
         *pBar = 2 * i;
         writer->Fill();
      }
   }
   {
      auto model = RNTupleModel::Create();
      auto pFoo = model->MakeField<double>("foo");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (int i = 0; i < 50'000; ++i) {
         *pFoo = i;
         writer->Fill();
      }
   }

   // Merge the inputs
   {
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources)
         sourcePtrs.push_back(s.get());

      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), wopts);
      RNTupleMerger merger{std::move(destination)};
      auto opts = RNTupleMergeOptions();
      opts.fMergingMode = ENTupleMergingMode::kUnion;
      auto res = merger.Merge(sourcePtrs, opts);
      ASSERT_TRUE(bool(res));
   }

   // Now verify that the output file contains the expected data.
   {
      auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
      EXPECT_EQ(reader->GetNEntries(), 50'050);

      auto vFoo = reader->GetView<double>("foo");
      auto vBar = reader->GetView<double>("bar");
      EXPECT_DOUBLE_EQ(vFoo(0), 0);
      EXPECT_DOUBLE_EQ(vBar(0), 0);
      EXPECT_DOUBLE_EQ(vFoo(49), 49);
      EXPECT_DOUBLE_EQ(vBar(49), 98);
      EXPECT_DOUBLE_EQ(vFoo(50), 0);
      EXPECT_DOUBLE_EQ(vBar(50), 0);
      EXPECT_DOUBLE_EQ(vFoo(1000), 950);
      EXPECT_DOUBLE_EQ(vBar(1000), 0);
      EXPECT_DOUBLE_EQ(vFoo(50'049), 49999);
      EXPECT_DOUBLE_EQ(vBar(50'049), 0);
   }
}

TEST(RNTupleMerger, MergeEmptySchema)
{
   // Try merging two ntuples with an empty schema
   FileRaii fileGuard1("test_ntuple_merge_empty_1.root");
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuardOut("test_ntuple_merge_empty_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "ROOT.NTuple.Merge", "Output RNTuple 'ntuple' has no entries.");

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
   }

   // We expect the output ntuple to have no entries
   {
      auto ntupleOut = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
      ASSERT_EQ(ntupleOut->GetNEntries(), 0);
      // We expect to see only the zero field
      ASSERT_EQ(ntupleOut->GetDescriptor().GetNFields(), 1);
   }
}

TEST(RNTupleMerger, MergeFirstEmptySchema)
{
   // Try merging two ntuples, the first of which has an empty schema
   FileRaii fileGuard1("test_ntuple_merge_firstempty_1.root");
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_firstempty_2.root");
   {
      auto model = RNTupleModel::Create();
      auto pi = model->MakeField<int>("int");
      auto pf = model->MakeField<float>("flt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pi = i;
         *pf = i;
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuardOut("test_ntuple_merge_firstempty_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      // In Filter mode, we expect the output ntuple to have 10 entries but an empty schema
      {
         auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
         auto ntupleOut = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         ASSERT_EQ(ntupleOut->GetNEntries(), ntuple1->GetNEntries());
         ASSERT_EQ(ntupleOut->GetDescriptor().GetNFields(), 1);
      }

      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      // In Union mode, we expect the output ntuple to have the entries of the non-empty ntuple
      {
         auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
         auto ntupleOut = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         ASSERT_EQ(ntupleOut->GetNEntries(), ntuple2->GetNEntries());
         ASSERT_EQ(ntupleOut->GetDescriptor().GetNFields(), ntuple2->GetDescriptor().GetNFields());

         auto viewI = ntupleOut->GetView<int>("int");
         auto viewF = ntupleOut->GetView<float>("flt");
         for (auto idx : ntupleOut->GetEntryRange()) {
            EXPECT_EQ(viewI(idx), idx);
            EXPECT_FLOAT_EQ(viewF(idx), idx);
         }
      }

      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeSecondEmptySchema)
{
   // Try merging two ntuples, the second of which has an empty schema
   FileRaii fileGuard1("test_ntuple_merge_secondempty_1.root");
   {
      auto model = RNTupleModel::Create();
      auto pi = model->MakeField<int>("int");
      auto pf = model->MakeField<float>("flt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         *pi = i;
         *pf = i;
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_secondempty_2.root");
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuardOut("test_ntuple_merge_secondempty_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      // In Union mode we expect the output ntuple to the same fields as the first
      {
         auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
         auto ntupleOut = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         ASSERT_EQ(ntupleOut->GetNEntries(), ntuple1->GetNEntries());
         ASSERT_EQ(ntupleOut->GetDescriptor().GetNFields(), ntuple1->GetDescriptor().GetNFields());

         auto viewI = ntupleOut->GetView<int>("int");
         auto viewF = ntupleOut->GetView<float>("flt");
         for (auto idx : ntupleOut->GetEntryRange()) {
            EXPECT_EQ(viewI(idx), idx);
            EXPECT_FLOAT_EQ(viewF(idx), idx);
         }
      }

      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeStaggeredIncremental)
{
   // Minimal repro of ATLAS example:
   // https://gitlab.cern.ch/amete/rootparallelmerger/-/tree/master-rntuple-prototype

   constexpr auto kNEvents = 5;

   std::array<FileRaii, 4> fileGuardsIn{
      FileRaii("test_ntuple_merge_staggered_in1.root"),
      FileRaii("test_ntuple_merge_staggered_in2.root"),
      FileRaii("test_ntuple_merge_staggered_in3.root"),
      FileRaii("test_ntuple_merge_staggered_in4.root"),
   };
   FileRaii fileGuardMerged("test_ntuple_merge_staggered_merged.root");

   // Produce input files.
   // Evenly numbered files have only field "foo", oddly numbered have only "bar".
   for (unsigned fileIdx = 0; fileIdx < fileGuardsIn.size(); ++fileIdx) {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuardsIn[fileIdx].GetPath().c_str(), "RECREATE"));
      const auto fieldName = (fileIdx % 2) == 0 ? "foo" : "bar";
      auto model = RNTupleModel::Create();
      model->MakeField<float>(fieldName);
      auto writer = RNTupleWriter::Append(std::move(model), "ntuple", *file);
      auto pVal = writer->GetModel().GetDefaultEntry().GetPtr<float>(fieldName);

      for (int i = 0; i < kNEvents; ++i) {
         *pVal = fileIdx * kNEvents + i;
         writer->Fill();
      }
      file->Write();
   }

   // Merge the files
   TFileMerger merger(false, false);
   merger.OutputFile(fileGuardMerged.GetPath().c_str(), "RECREATE", 505);
   merger.SetMergeOptions(TString("rntuple.MergingMode=Union"));
   for (const auto &f : fileGuardsIn) {
      auto file = std::unique_ptr<TFile>(TFile::Open(f.GetPath().c_str(), "UPDATE"));
      merger.AddFile(file.get());
      bool ok = merger.PartialMerge(TFileMerger::kAllIncremental | TFileMerger::kKeepCompression);
      EXPECT_TRUE(ok);
   }

   // Verify values match.
   // We expect that "foo" and "bar" alternate having values or being 0 (each having kNEvents consecutively)
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuardMerged.GetPath().c_str(), "READ"));
      auto ntuple = file->Get<ROOT::RNTuple>("ntuple");
      ASSERT_NE(ntuple, nullptr);

      auto reader = RNTupleReader::Open(*ntuple);
      EXPECT_EQ(reader->GetNEntries(), kNEvents * fileGuardsIn.size());

      auto pnFoo = reader->GetModel().GetDefaultEntry().GetPtr<float>("foo");
      auto pnBar = reader->GetModel().GetDefaultEntry().GetPtr<float>("bar");
      for (auto idx : reader->GetEntryRange()) {
         reader->LoadEntry(idx);
         float expFoo = (idx / kNEvents) % 2 == 0 ? idx : 0;
         float expBar = (idx / kNEvents) % 2 == 1 ? idx : 0;
         EXPECT_FLOAT_EQ(*pnFoo, expFoo);
         EXPECT_FLOAT_EQ(*pnBar, expBar);
      }
   }
}

TEST(RNTupleMerger, MergeUntypedRecordEqual)
{
   // Merge 2 RNTuples with a single Untyped Record field with the same children in the same order.
   // It's supposed to work.

   FileRaii fileGuard1("test_ntuple_merge_untyped_equal_1.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<int>>("int"));
      children.push_back(std::make_unique<RField<float>>("float"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_untyped_equal_2.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<int>>("int"));
      children.push_back(std::make_unique<RField<float>>("float"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_untyped_equal_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_TRUE(bool(res));
      }
   }

   // Now check some information
   // ntuple1 has 10 entries
   // ntuple2 has 10 entries
   {
      auto ntuple1 = RNTupleReader::Open("ntuple", fileGuard1.GetPath());
      auto ntuple2 = RNTupleReader::Open("ntuple", fileGuard2.GetPath());
      auto ntuple3 = RNTupleReader::Open("ntuple", fileGuard3.GetPath());
      ASSERT_EQ(ntuple1->GetNEntries() + ntuple2->GetNEntries(), ntuple3->GetNEntries());
   }
}

TEST(RNTupleMerger, MergeUntypedRecordDifferent)
{
   // Merge 2 RNTuples with a single Untyped Record field with different children.
   // It's supposed to fail.

   FileRaii fileGuard1("test_ntuple_merge_untyped_diff_1.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<int>>("int"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_untyped_diff_2.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<float>>("float"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_untyped_diff_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
   }
}

TEST(RNTupleMerger, MergeUntypedSymmetric)
{
   // Merge 2 RNTuples with a single Untyped Record field with the same children but in different order.
   // It's supposed to fail.

   FileRaii fileGuard1("test_ntuple_merge_untyped_sym_1.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<int>>("int"));
      children.push_back(std::make_unique<RField<float>>("float"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   FileRaii fileGuard2("test_ntuple_merge_untyped_sym_2.root");
   {
      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> children;
      children.push_back(std::make_unique<RField<float>>("float"));
      children.push_back(std::make_unique<RField<int>>("int"));
      auto record = std::make_unique<ROOT::RRecordField>("record", std::move(children));
      model->AddField(std::move(record));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      for (size_t i = 0; i < 10; ++i) {
         ntuple->Fill();
      }
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_untyped_sym_out.root");
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      RNTupleMergeOptions opts;
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kFilter;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kUnion;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuard3.GetPath(), RNTupleWriteOptions());
         opts.fMergingMode = ENTupleMergingMode::kStrict;
         RNTupleMerger merger{std::move(destination)};
         auto res = merger.Merge(sourcePtrs, opts);
         EXPECT_FALSE(bool(res));
      }
   }
}

TEST(RNTupleMerger, GenerateZeroPagesIncremental)
{
   // Incrementally Union-merge RNTuples with alternating fields, so to trigger generation of zero pages.
   // This is a minimal reproducer of a bug uncovered by an ATLAS-like merging flow:
   // https://gitlab.cern.ch/amete/rootparallelmerger/-/tree/master-rntuple-prototype
   FileRaii fileGuardOut("test_ntuple_merge_zeropages_incr_out.root");

   {
      TFileMerger merger(kFALSE, kTRUE);
      merger.SetMergeOptions(TString("rntuple.MergingMode=Union"));
      merger.OutputFile(fileGuardOut.GetPath().c_str(), "RECREATE",
                        ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);

      auto writeOpts = RNTupleWriteOptions();

      // Produce data
      for (int idx : {0, 1, 2, 3, 4}) {
         auto file = std::unique_ptr<TMemFile>(new TMemFile(std::to_string(idx).c_str(), "CREATE"));

         auto model = ROOT::RNTupleModel::Create();
         const auto dataName = (idx % 2 == 0) ? "foo" : "bar";
         auto field = ROOT::RFieldBase::Create(dataName, "std::vector<int>").Unwrap();
         model->AddField(std::move(field));

         auto writer = ROOT::RNTupleWriter::Append(std::move(model), "ntuple", *file, writeOpts);

         std::vector<int> data;
         for (int i = 0; i < 5000; ++i) {
            auto entry = writer->GetModel().CreateBareEntry();
            data.clear();
            for (int j = 0; j < 1000; ++j) {
               data.push_back(i * j + j);
            }
            entry->BindRawPtr(dataName, &data);
            writer->Fill(*entry);
         }

         writer.reset();

         merger.AddFile(file.get());
         bool res = merger.PartialMerge(TFileMerger::kAllIncremental | TFileMerger::kKeepCompression);
         ASSERT_TRUE(res);

         file->Write();
      }
   }

   // Read back the data
   auto reader = ROOT::RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 25000);

   auto viewFoo = reader->GetView<std::vector<int>>("foo");
   auto viewBar = reader->GetView<std::vector<int>>("bar");

   for (int idx : {0, 1, 2, 3, 4}) {
      auto fooData = viewFoo(idx * 5000);
      auto barData = viewBar(idx * 5000);
      if (idx % 2 == 0) {
         ASSERT_EQ(fooData.size(), 1000);
         EXPECT_EQ(fooData[0], 0);
         EXPECT_EQ(fooData[500], 500);
         EXPECT_EQ(barData.size(), 0);

         fooData = viewFoo(idx * 5000 + 4999);
         barData = viewBar(idx * 5000 + 4999);
         ASSERT_EQ(fooData.size(), 1000);
         EXPECT_EQ(fooData[0], 0);
         EXPECT_EQ(fooData[1], 5000);
         EXPECT_EQ(fooData[999], 4995000);
         EXPECT_EQ(barData.size(), 0);
      } else {
         EXPECT_EQ(fooData.size(), 0);
         ASSERT_EQ(barData.size(), 1000);
         EXPECT_EQ(barData[0], 0);
         EXPECT_EQ(barData[500], 500);

         fooData = viewFoo(idx * 5000 + 4999);
         barData = viewBar(idx * 5000 + 4999);
         EXPECT_EQ(fooData.size(), 0);
         ASSERT_EQ(barData.size(), 1000);
         EXPECT_EQ(barData[0], 0);
         EXPECT_EQ(barData[1], 5000);
         EXPECT_EQ(barData[999], 4995000);
      }
   }
}

TEST(RNTupleMerger, MergeStreamerFields)
{
   // Merge two files with custom streamer fields
   FileRaii fileGuard1("test_ntuple_merge_streamer1.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      model->AddField(std::make_unique<ROOT::RStreamerField>("bar", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      auto fieldBar = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("bar");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
         CustomStruct bar;
         bar.a = i;
         *fieldBar = bar;
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_merge_streamer2.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("bar", "CustomStruct"));
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      auto fieldBar = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("bar");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
         CustomStruct bar;
         bar.a = 2.f * i;
         *fieldBar = bar;
         ntuple->Fill();
      }
   }
   
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         SCOPED_TRACE(std::string("with merging mode = ") + ToString(mmode));
         FileRaii fileGuardOut("test_ntuple_merge_streamer_out.root");
         {
            auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
            RNTupleMerger merger{std::move(destination)};
            RNTupleMergeOptions opts;
            opts.fMergingMode = mmode;
            auto res = merger.Merge(sourcePtrs, opts);
            ASSERT_TRUE(bool(res));
         }

         auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         EXPECT_EQ(reader->GetNEntries(), 20);
         auto pFoo = reader->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
         auto pBar = reader->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("bar");
         EXPECT_EQ(reader->GetModel().GetConstField("foo").GetStructure(), ROOT::ENTupleStructure::kStreamer);
         EXPECT_EQ(reader->GetModel().GetConstField("bar").GetStructure(), ROOT::ENTupleStructure::kStreamer);
         for (auto idx : reader->GetEntryRange()) {
            reader->LoadEntry(idx);
            ASSERT_EQ(pFoo->v1.size(), 1);
            EXPECT_FLOAT_EQ(pFoo->v1[0], idx % 10);
            EXPECT_EQ(pFoo->s, std::to_string(idx % 10));
            EXPECT_FLOAT_EQ(pBar->a, idx < 10 ? idx : 2.f * (idx - 10));
         }
      }
   }
}

TEST(RNTupleMerger, MergeStreamerFieldsFirstMissing)
{
   // Merge two files where the second has an additional streamer field - should fail except in Filter mode.
   FileRaii fileGuard1("test_ntuple_merge_streamer_firstmissing1.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
      }
   }
   FileRaii fileGuard2("test_ntuple_merge_streamer_firstmissing2.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      model->AddField(std::make_unique<ROOT::RStreamerField>("bar", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      auto fieldBar = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("bar");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
         CustomStruct bar;
         bar.a = 2.f * i;
         *fieldBar = bar;
         ntuple->Fill();
      }
   }
   
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         SCOPED_TRACE(std::string("with merging mode = ") + ToString(mmode));
         FileRaii fileGuardOut("test_ntuple_merge_streamer_firstmissing_out.root");
         {
            auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
            RNTupleMerger merger{std::move(destination)};
            RNTupleMergeOptions opts;
            opts.fMergingMode = mmode;
            auto res = merger.Merge(sourcePtrs, opts);
            if (mmode == ENTupleMergingMode::kFilter)
               EXPECT_TRUE(bool(res));
            else
               EXPECT_FALSE(bool(res));
         }
      }
   }
}

TEST(RNTupleMerger, MergeStreamerFieldsSecondMissing)
{
   // Merge two files where the first has an additional streamer field - should fail.
   FileRaii fileGuard1("test_ntuple_merge_streamer_secondmissing1.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      model->AddField(std::make_unique<ROOT::RStreamerField>("bar", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      auto fieldBar = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("bar");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
         CustomStruct bar;
         bar.a = 2.f * i;
         *fieldBar = bar;
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_merge_streamer_secondmissing2.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      auto fieldFoo = ntuple->GetModel().GetDefaultEntry().GetPtr<CustomStruct>("foo");
      for (int i = 0; i < 10; ++i) {
         CustomStruct foo;
         foo.v1.push_back(i);
         foo.s = std::to_string(i);
         *fieldFoo = foo;
         ntuple->Fill();
      }
   }
   {
      // Gather the input sources
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs;
      for (const auto &s : sources) {
         sourcePtrs.push_back(s.get());
      }

      // Now merge the inputs
      for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
         SCOPED_TRACE(std::string("with merging mode = ") + ToString(mmode));
         FileRaii fileGuardOut("test_ntuple_merge_streamer_secondmissing_out.root");
         {
            auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
            RNTupleMerger merger{std::move(destination)};
            RNTupleMergeOptions opts;
            opts.fMergingMode = mmode;
            auto res = merger.Merge(sourcePtrs, opts);
            EXPECT_FALSE(bool(res));
         }
      }
   }
}
