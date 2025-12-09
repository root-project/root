#include "ntuple_test.hxx"

#include <algorithm>

TEST(RNTuple, MultiStreamBasics)
{
   FileRaii fileGuard("test_ntuple_multi_stream_basics.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("pt");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      writer->Fill();
   }

   {
      auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
      reader->EnableMetrics();
      reader->LoadEntry(0);
      reader->LoadEntry(1);
      EXPECT_EQ(
         1, reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RClusterPool.nCluster")->GetValueAsInt());
      auto ctrClusterLoaded = reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nClusterLoaded");
      auto nClusterLoaded = ctrClusterLoaded->GetValueAsInt();
      reader->LoadEntry(0);
      EXPECT_GT(ctrClusterLoaded->GetValueAsInt(), nClusterLoaded);
   }

   {
      auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
      reader->EnableMetrics();
      auto token = reader->CreateActiveEntryToken();
      EXPECT_EQ(ROOT::kInvalidDescriptorId, token.GetEntryNumber());

      try {
         token.SetEntryNumber(2);
         FAIL() << "out of bounds entry number should fail";
      } catch (const ROOT::RException &e) {
         EXPECT_THAT(e.what(), testing::HasSubstr("out of range"));
      }

      token.SetEntryNumber(0);
      reader->LoadEntry(0);
      reader->LoadEntry(1);
      EXPECT_EQ(
         2, reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RClusterPool.nCluster")->GetValueAsInt());
      auto ctrClusterLoaded = reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nClusterLoaded");
      auto nClusterLoaded = ctrClusterLoaded->GetValueAsInt();
      reader->LoadEntry(0);
      EXPECT_EQ(ctrClusterLoaded->GetValueAsInt(), nClusterLoaded);
   }
}

TEST(RNTuple, MultiStreamRefcount)
{
   FileRaii fileGuard("test_ntuple_multi_stream_refcount.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("pt");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      writer->Fill();
   }

   {
      auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
      reader->EnableMetrics();

      auto t1 = reader->CreateActiveEntryToken();
      t1.SetEntryNumber(0);
      auto t2{t1};
      EXPECT_EQ(0, t2.GetEntryNumber());

      auto ctrPageUnsealed = reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageUnsealed");
      auto ctrNPage = reader->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RPagePool.nPage");
      reader->LoadEntry(0);
      reader->LoadEntry(1);
      EXPECT_EQ(2, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(2, ctrNPage->GetValueAsInt());
      t1.Reset();
      reader->LoadEntry(0);
      EXPECT_EQ(2, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(1, ctrNPage->GetValueAsInt());
      reader->LoadEntry(1);
      EXPECT_EQ(3, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(2, ctrNPage->GetValueAsInt());
      t2.Reset(); // cluster 0 unpinned
      reader->LoadEntry(0);
      EXPECT_EQ(3, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(1, ctrNPage->GetValueAsInt());
      reader->LoadEntry(1);
      EXPECT_EQ(4, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(1, ctrNPage->GetValueAsInt());
   }
}

TEST(RNTuple, MultiStreamDestruct)
{
   FileRaii fileGuard("test_ntuple_multi_stream_destruct.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("pt");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto token = reader->CreateActiveEntryToken();
   reader.reset();
   EXPECT_NO_THROW(token.SetEntryNumber(0));
   EXPECT_NO_THROW(token.Reset());
}

TEST(RNTuple, MultiStreamSwap)
{
   FileRaii fileGuard("test_ntuple_multi_stream_swap.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<float>("pt");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      writer->Fill();
   }

   auto r1 = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto r2 = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   r1->EnableMetrics();
   r2->EnableMetrics();

   auto t1 = r1->CreateActiveEntryToken();
   t1.SetEntryNumber(0);
   auto t2 = r2->CreateActiveEntryToken();
   t2.SetEntryNumber(1);

   std::swap(t1, t2);
   EXPECT_EQ(1u, t1.GetEntryNumber()); // Now belongs to r2
   EXPECT_EQ(0u, t2.GetEntryNumber()); // Now belongs to r1

   {
      auto ctrPageUnsealed = r1->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageUnsealed");
      auto ctrNPage = r1->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RPagePool.nPage");
      r1->LoadEntry(0);
      r1->LoadEntry(1);
      EXPECT_EQ(2, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(2, ctrNPage->GetValueAsInt());
   }

   {
      auto ctrPageUnsealed = r2->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.nPageUnsealed");
      auto ctrNPage = r2->GetMetrics().GetCounter("RNTupleReader.RPageSourceFile.RPagePool.nPage");
      r2->LoadEntry(1);
      r2->LoadEntry(0);
      EXPECT_EQ(2, ctrPageUnsealed->GetValueAsInt());
      EXPECT_EQ(2, ctrNPage->GetValueAsInt());
   }
}
