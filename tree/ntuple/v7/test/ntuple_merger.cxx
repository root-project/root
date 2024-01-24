#include "ntuple_test.hxx"

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
      RNTupleWriter ntuple(std::move(model), std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), options));
      ntuple.Fill();
      ntuple.CommitCluster();
      for (unsigned i = 0; i < 100000; ++i) {
         *wrPt = i;
         ntuple.Fill();
      }
   }

   RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();
   auto columnId =
      source.GetSharedDescriptorGuard()->FindPhysicalColumnId(source.GetSharedDescriptorGuard()->FindFieldId("pt"), 0);

   // Check first cluster consisting of a single entry
   RClusterIndex index(source.GetSharedDescriptorGuard()->FindClusterId(columnId, 0), 0);
   RPageStorage::RSealedPage sealedPage;
   source.LoadSealedPage(columnId, index, sealedPage);
   ASSERT_EQ(1U, sealedPage.fNElements);
   ASSERT_EQ(4U, sealedPage.fSize);
   auto buffer = std::make_unique<unsigned char[]>(sealedPage.fSize);
   sealedPage.fBuffer = buffer.get();
   source.LoadSealedPage(columnId, index, sealedPage);
   ASSERT_EQ(1U, sealedPage.fNElements);
   ASSERT_EQ(4U, sealedPage.fSize);
   EXPECT_EQ(42, ReadRawInt(sealedPage.fBuffer));

   // Check second, big cluster
   auto clusterId = source.GetSharedDescriptorGuard()->FindClusterId(columnId, 1);
   ASSERT_NE(clusterId, index.GetClusterId());
   const auto clusterDesc = source.GetSharedDescriptorGuard()->GetClusterDescriptor(clusterId).Clone();
   const auto &pageRange = clusterDesc.GetPageRange(columnId);
   EXPECT_GT(pageRange.fPageInfos.size(), 1U);
   std::uint32_t firstElementInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      buffer = std::make_unique<unsigned char[]>(pi.fLocator.fBytesOnStorage);
      sealedPage.fBuffer = buffer.get();
      source.LoadSealedPage(columnId, RClusterIndex(clusterId, firstElementInPage), sealedPage);
      ASSERT_GE(sealedPage.fSize, 4U);
      EXPECT_EQ(firstElementInPage, ReadRawInt(sealedPage.fBuffer));
      firstElementInPage += pi.fNElements;
   }
}

TEST(RFieldMerger, Merge)
{
   auto mergeResult = RFieldMerger::Merge(RFieldDescriptor(), RFieldDescriptor());
   EXPECT_FALSE(mergeResult);
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

      // Create the output
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      RNTupleMerger merger;
      EXPECT_NO_THROW(merger.Merge(sourcePtrs, *destination));
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

      // Create the output
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      // We expect this to fail since the fields between the sources do NOT match
      RNTupleMerger merger;
      EXPECT_THROW(merger.Merge(sourcePtrs, *destination), ROOT::Experimental::RException);
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

      // Create the output
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      // We expect this to fail since the fields between the sources do NOT match
      RNTupleMerger merger;
      EXPECT_THROW(merger.Merge(sourcePtrs, *destination), ROOT::Experimental::RException);
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

      // Create the output
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      // We expect this to fail since the fields between the sources do NOT match
      RNTupleMerger merger;
      EXPECT_THROW(merger.Merge(sourcePtrs, *destination), ROOT::Experimental::RException);
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
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
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
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
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
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      RNTupleMerger merger;
      EXPECT_NO_THROW(merger.Merge(sourcePtrs, *destination));
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

TEST(RNTupleMerger, MergeInconsistentTypes)
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
      RNTupleWriteOptions writeOpts;
      writeOpts.SetUseBufferedWrite(false);
      auto destination = RPageSink::Create("ntuple", fileGuard3.GetPath(), writeOpts);

      // Now Merge the inputs
      // We expect this to fail since the fields between the sources do NOT match
      RNTupleMerger merger;
      EXPECT_THROW(merger.Merge(sourcePtrs, *destination), ROOT::Experimental::RException);
   }
}
