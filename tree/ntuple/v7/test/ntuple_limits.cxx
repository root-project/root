#include "ntuple_test.hxx"

#include <limits>

// This test aims to exercise some limits of RNTuple that are expected to be upper bounds for realistic applications.
// The theoretical limits may be higher: for example, the specification supports up to 4B clusters per group, but the
// expectation is less than 10k. For good measure, we test up to 100k clusters per group below.
//
// By nature, such limit tests will use considerable resources. For that reason, we disable the tests by default to
// avoid running them in our CI. Locally they can be run by passing `--gtest_also_run_disabled_tests` to the gtest
// executable. This may be combined with `--gtest_filter` to select a particular test. For example, to run said test
// for many clusters in a single group, the invocation would be
// ```
// ./tree/ntuple/v7/test/ntuple_limits --gtest_also_run_disabled_tests --gtest_filter=*Limits_ManyClusters
// ```

TEST(RNTuple, Limits_ManyFields)
{
   // Writing and reading a model with 100k integer fields takes around 2.2s and seems to have slightly more than linear
   // complexity (200k fields take 7.5s).
   // Peak RSS is around 750MB.
   FileRaii fileGuard("test_ntuple_limits_manyFields.root");

   static constexpr int NumFields = 100'000;

   {
      auto model = RNTupleModel::Create();

      for (int i = 0; i < NumFields; i++) {
         std::string name = "f" + std::to_string(i);
         *model->MakeField<int>(name) = i;
      }

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();

   EXPECT_EQ(descriptor.GetNFields(), 1 + NumFields);
   EXPECT_EQ(model.GetConstFieldZero().GetSubFields().size(), NumFields);
   EXPECT_EQ(reader->GetNEntries(), 1);

   reader->LoadEntry(0);
   for (int i = 0; i < NumFields; i++) {
      auto valuePtr = model.GetDefaultEntry().GetPtr<int>("f" + std::to_string(i));
      EXPECT_EQ(*valuePtr, i);
   }
}

TEST(RNTuple, Limits_ManyClusters)
{
   // Writing and reading 500k clusters takes around 3.3s and seems to have benign scaling behavior.
   // (1M clusters take around 6.6s).
   // Peak RSS is around 850MB.
   FileRaii fileGuard("test_ntuple_limits_manyClusters.root");

   static constexpr int NumClusters = 500'000;

   {
      auto model = RNTupleModel::Create();
      auto id = model->MakeField<int>("id");

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      for (int i = 0; i < NumClusters; i++) {
         *id = i;
         writer->Fill();
         writer->CommitCluster();
      }
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();

   EXPECT_EQ(reader->GetNEntries(), NumClusters);
   EXPECT_EQ(descriptor.GetNClusters(), NumClusters);
   EXPECT_EQ(descriptor.GetNActiveClusters(), NumClusters);

   auto id = model.GetDefaultEntry().GetPtr<int>("id");
   for (int i = 0; i < NumClusters; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}

TEST(RNTuple, Limits_ManyClusterGroups)
{
   // Writing and reading 25k cluster groups takes around 1.7s and seems to have quadratic complexity
   // (50k cluster groups takes around 6.5s).
   // Peak RSS is around 275MB.
   FileRaii fileGuard("test_ntuple_limits_manyClusterGroups.root");

   static constexpr int NumClusterGroups = 25'000;

   {
      auto model = RNTupleModel::Create();
      auto id = model->MakeField<int>("id");

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      for (int i = 0; i < NumClusterGroups; i++) {
         *id = i;
         writer->Fill();
         writer->CommitCluster(/*commitClusterGroup=*/true);
      }
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();

   EXPECT_EQ(reader->GetNEntries(), NumClusterGroups);
   EXPECT_EQ(descriptor.GetNClusterGroups(), NumClusterGroups);
   EXPECT_EQ(descriptor.GetNClusters(), NumClusterGroups);

   auto id = model.GetDefaultEntry().GetPtr<int>("id");
   for (int i = 0; i < NumClusterGroups; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}

TEST(RNTuple, Limits_ManyPages)
{
   // Writing and reading 1M pages (of two elements each) takes around 1.3 and seems to have benign scaling behavior
   // (2M pages take 2.6s).
   // Peak RSS is around 600MB.
   FileRaii fileGuard("test_ntuple_limits_manyPages.root");

   static constexpr int NumPages = 1'000'000;
   static constexpr int NumEntries = NumPages * 2;

   {
      auto model = RNTupleModel::Create();
      auto id = model->MakeField<int>("id");
      RNTupleWriteOptions options;
      // Two elements per page.
      options.SetInitialUnzippedPageSize(8);
      options.SetMaxUnzippedPageSize(8);

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      for (int i = 0; i < NumEntries; i++) {
         *id = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();
   auto fieldId = descriptor.FindFieldId("id");
   auto columnId = descriptor.FindPhysicalColumnId(fieldId, 0, 0);

   EXPECT_EQ(reader->GetNEntries(), NumEntries);
   EXPECT_EQ(descriptor.GetNClusters(), 1);
   EXPECT_EQ(descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos.size(), NumPages);

   auto id = model.GetDefaultEntry().GetPtr<int>("id");
   for (int i = 0; i < NumEntries; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}

TEST(RNTuple, Limits_ManyPagesOneEntry)
{
   // Writing and reading 1M pages (of four elements each) takes around 2.4s and seems to have benign scaling behavior
   // (2M pages take around 4.8s).
   // Peak RSS is around 625MB.
   FileRaii fileGuard("test_ntuple_limits_manyPagesOneEntry.root");

   static constexpr int NumPages = 1'000'000;
   static constexpr int NumElements = NumPages * 4;

   {
      auto model = RNTupleModel::Create();
      auto ids = model->MakeField<std::vector<int>>("ids");
      RNTupleWriteOptions options;
      // Four elements per page (must fit two 64-bit indices!)
      options.SetInitialUnzippedPageSize(16);
      options.SetMaxUnzippedPageSize(16);

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      for (int i = 0; i < NumElements; i++) {
         ids->push_back(i);
      }
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();
   auto fieldId = descriptor.FindFieldId("ids");
   auto subFieldId = descriptor.FindFieldId("_0", fieldId);
   auto columnId = descriptor.FindPhysicalColumnId(subFieldId, 0, 0);

   EXPECT_EQ(reader->GetNEntries(), 1);
   EXPECT_EQ(descriptor.GetNClusters(), 1);
   EXPECT_EQ(descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos.size(), NumPages);

   auto ids = model.GetDefaultEntry().GetPtr<std::vector<int>>("ids");
   reader->LoadEntry(0);
   EXPECT_EQ(ids->size(), NumElements);
   for (int i = 0; i < NumElements; i++) {
      EXPECT_EQ((*ids)[i], i);
   }
}

TEST(RNTuple, DISABLED_Limits_LargePage)
{
   // Writing and reading one page with 600M elements takes around 18s and seems to have linear complexity
   // (900M elements take 27s)
   // Peak RSS is around 14 GB.
   FileRaii fileGuard("test_ntuple_limits_largePage.root");

   // clang-format off
   static constexpr int NumElements = 600'000'000;
   // clang-format on

   {
      auto model = RNTupleModel::Create();
      auto id = model->MakeField<std::uint64_t>("id");
      RNTupleWriteOptions options;
      static constexpr std::size_t Size = NumElements * sizeof(std::uint64_t);
      options.SetMaxUnzippedClusterSize(Size);
      options.SetApproxZippedClusterSize(Size);
      options.SetMaxUnzippedPageSize(Size);
      options.SetUseBufferedWrite(false);

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      for (int i = 0; i < NumElements; i++) {
         *id = i;
         writer->Fill();
      }
   }

   RNTupleReadOptions options;
   options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);
   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath(), options);
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();
   auto fieldId = descriptor.FindFieldId("id");
   auto columnId = descriptor.FindPhysicalColumnId(fieldId, 0, 0);

   EXPECT_EQ(reader->GetNEntries(), NumElements);
   EXPECT_EQ(descriptor.GetNClusters(), 1);
   EXPECT_EQ(descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos.size(), 1);
   EXPECT_GT(descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos[0].fLocator.GetNBytesOnStorage(),
             static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()));

   auto id = model.GetDefaultEntry().GetPtr<std::uint64_t>("id");
   for (int i = 0; i < NumElements; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}

TEST(RNTuple, DISABLED_Limits_LargePageOneEntry)
{
   // Writing and reading one page with 100M elements takes around 1.7s and seems to have linear complexity (200M
   // elements take 3.5s, 400M elements take around 7s).
   // Peak RSS is around 1.4GB.
   FileRaii fileGuard("test_ntuple_limits_largePageOneEntry.root");

   static constexpr int NumElements = 100'000'000;

   {
      auto model = RNTupleModel::Create();
      auto ids = model->MakeField<std::vector<int>>("ids");
      RNTupleWriteOptions options;
      static constexpr std::size_t Size = NumElements * sizeof(int);
      options.SetMaxUnzippedClusterSize(Size);
      options.SetApproxZippedClusterSize(Size);
      options.SetMaxUnzippedPageSize(Size);

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
      for (int i = 0; i < NumElements; i++) {
         ids->push_back(i);
      }
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();
   const auto &model = reader->GetModel();
   auto fieldId = descriptor.FindFieldId("ids");
   auto subFieldId = descriptor.FindFieldId("_0", fieldId);
   auto columnId = descriptor.FindPhysicalColumnId(subFieldId, 0, 0);

   EXPECT_EQ(reader->GetNEntries(), 1);
   EXPECT_EQ(descriptor.GetNClusters(), 1);
   EXPECT_EQ(descriptor.GetClusterDescriptor(0).GetPageRange(columnId).fPageInfos.size(), 1);

   auto ids = model.GetDefaultEntry().GetPtr<std::vector<int>>("ids");
   reader->LoadEntry(0);
   EXPECT_EQ(ids->size(), NumElements);
   for (int i = 0; i < NumElements; i++) {
      EXPECT_EQ((*ids)[i], i);
   }
}
