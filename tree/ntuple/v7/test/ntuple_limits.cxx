#include "ntuple_test.hxx"

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

TEST(RNTuple, DISABLED_Limits_ManyFields)
{
   // Writing and reading a model with 10k integer fields takes around 30s and seems to have more than quadratic
   // complexity (5k fields take 6s).
   FileRaii fileGuard("test_ntuple_limits_manyFields.root");

   static constexpr int NumFields = 10'000;

   {
      auto model = RNTupleModel::Create();

      for (int i = 0; i < NumFields; i++) {
         std::string name = "f" + std::to_string(i);
         model->MakeField<int>(name.c_str(), i);
      }

      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto descriptor = reader->GetDescriptor();
   auto model = reader->GetModel();

   EXPECT_EQ(descriptor->GetNFields(), 1 + NumFields);
   EXPECT_EQ(model->GetFieldZero().GetSubFields().size(), NumFields);
   EXPECT_EQ(reader->GetNEntries(), 1);

   reader->LoadEntry(0);
   for (int i = 0; i < NumFields; i++) {
      auto *valuePtr = model->GetDefaultEntry()->Get<int>("f" + std::to_string(i));
      EXPECT_EQ(*valuePtr, i);
   }
}

TEST(RNTuple, DISABLED_Limits_ManyClusters)
{
   // Writing and reading 100k clusters takes between 80s - 100s and seems to have more than quadratic complexity
   // (50k clusters take less than 15s).
   FileRaii fileGuard("test_ntuple_limits_manyClusters.root");

   static constexpr int NumClusters = 100'000;

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
   auto descriptor = reader->GetDescriptor();
   auto model = reader->GetModel();

   EXPECT_EQ(reader->GetNEntries(), NumClusters);
   EXPECT_EQ(descriptor->GetNClusters(), NumClusters);
   EXPECT_EQ(descriptor->GetNActiveClusters(), NumClusters);

   auto *id = model->GetDefaultEntry()->Get<int>("id");
   for (int i = 0; i < NumClusters; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}

TEST(RNTuple, DISABLED_Limits_ManyClusterGroups)
{
   // Writing and reading 100k cluster groups takes between 100s - 110s and seems to have more than quadratic complexity
   // (50k cluster groups takes less than 20s).
   FileRaii fileGuard("test_ntuple_limits_manyClusterGroups.root");

   static constexpr int NumClusterGroups = 100'000;

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
   auto descriptor = reader->GetDescriptor();
   auto model = reader->GetModel();

   EXPECT_EQ(reader->GetNEntries(), NumClusterGroups);
   EXPECT_EQ(descriptor->GetNClusterGroups(), NumClusterGroups);
   EXPECT_EQ(descriptor->GetNClusters(), NumClusterGroups);

   auto *id = model->GetDefaultEntry()->Get<int>("id");
   for (int i = 0; i < NumClusterGroups; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(*id, i);
   }
}
