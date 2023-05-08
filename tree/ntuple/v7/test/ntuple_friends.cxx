#include "ntuple_test.hxx"

TEST(RPageStorageFriends, Null)
{
   FileRaii fileGuard1("test_ntuple_friends_null1.root");
   FileRaii fileGuard2("test_ntuple_friends_null2.root");

   auto model1 = RNTupleModel::Create();
   auto model2 = RNTupleModel::Create();
   {
      auto ntuple1 = RNTupleWriter::Recreate(std::move(model1), "ntpl1", fileGuard1.GetPath());
      auto ntuple2 = RNTupleWriter::Recreate(std::move(model2), "ntpl2", fileGuard2.GetPath());
   }

   std::vector<std::unique_ptr<RPageSource>> realSources;
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl1", fileGuard1.GetPath(), RNTupleReadOptions()));
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl2", fileGuard2.GetPath(), RNTupleReadOptions()));
   RPageSourceFriends friendSource("myNTuple", realSources);
   friendSource.Attach();
   EXPECT_EQ(0u, friendSource.GetNEntries());
}


TEST(RPageStorageFriends, Empty)
{
   std::span<RNTupleReader::ROpenSpec> ntuples;
   auto reader = RNTupleReader::OpenFriends(ntuples);
   EXPECT_EQ(0u, reader->GetNEntries());
   EXPECT_EQ(0u, reader->GetModel()->GetFieldZero()->GetOnDiskId());
   EXPECT_EQ(0u, std::distance(reader->GetModel()->GetDefaultEntry()->begin(),
                               reader->GetModel()->GetDefaultEntry()->end()));
   EXPECT_EQ(0u, reader->GetDescriptor()->GetNColumns());
   EXPECT_EQ(1u, reader->GetDescriptor()->GetNFields()); // The zero field
   EXPECT_EQ(0u, reader->GetDescriptor()->GetNClusters());
}


TEST(RPageStorageFriends, Basic)
{
   FileRaii fileGuard1("test_ntuple_friends_basic1.root");
   FileRaii fileGuard2("test_ntuple_friends_basic2.root");

   auto model1 = RNTupleModel::Create();
   auto fieldPt = model1->MakeField<float>("pt", 42.0);

   auto model2 = RNTupleModel::Create();
   auto fieldEta = model2->MakeField<float>("eta", 24.0);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model1), "ntpl1", fileGuard1.GetPath());
      *fieldPt = 1.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *fieldPt = 2.0;
      ntuple->Fill();
      *fieldPt = 3.0;
      ntuple->Fill();
   }
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model2), "ntpl2", fileGuard2.GetPath());
      *fieldEta = 4.0;
      ntuple->Fill();
      *fieldEta = 5.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *fieldEta = 6.0;
      ntuple->Fill();
   }

   std::vector<RNTupleReader::ROpenSpec> friends{
      {"ntpl1", fileGuard1.GetPath()},
      {"ntpl2", fileGuard2.GetPath()} };
   auto ntuple = RNTupleReader::OpenFriends(friends);
   EXPECT_EQ(3u, ntuple->GetNEntries());

   auto clone = ntuple->Clone();
   EXPECT_EQ(3u, clone->GetNEntries());

   auto viewPt = clone->GetView<float>("ntpl1.pt");
   auto viewEta = clone->GetView<float>("ntpl2.eta");

   EXPECT_DOUBLE_EQ(1.0, viewPt(0));
   EXPECT_DOUBLE_EQ(2.0, viewPt(1));
   EXPECT_DOUBLE_EQ(3.0, viewPt(2));

   EXPECT_DOUBLE_EQ(4.0, viewEta(0));
   EXPECT_DOUBLE_EQ(5.0, viewEta(1));
   EXPECT_DOUBLE_EQ(6.0, viewEta(2));
}


TEST(RPageStorageFriends, FailOnNtupleNameClash)
{
   FileRaii fileGuard1("test_ntuple_friends_name1.root");
   FileRaii fileGuard2("test_ntuple_friends_name2.root");

   auto model1 = RNTupleModel::Create();
   auto model2 = RNTupleModel::Create();
   {
      auto ntuple1 = RNTupleWriter::Recreate(std::move(model1), "ntpl", fileGuard1.GetPath());
      auto ntuple2 = RNTupleWriter::Recreate(std::move(model2), "ntpl", fileGuard2.GetPath());
   }

   std::vector<std::unique_ptr<RPageSource>> realSources;
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl", fileGuard1.GetPath(), RNTupleReadOptions()));
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl", fileGuard2.GetPath(), RNTupleReadOptions()));
   RPageSourceFriends friendSource("myNTuple", realSources);
   EXPECT_THROW(friendSource.Attach(), ROOT::Experimental::RException);
}

TEST(RPageStorageFriends, FailOnEntryCountMismatch)
{
   FileRaii fileGuard1("test_ntuple_friends_count1.root");
   FileRaii fileGuard2("test_ntuple_friends_count2.root");

   auto model1 = RNTupleModel::Create();
   auto fieldPt = model1->MakeField<float>("pt", 42.0);
   auto model2 = RNTupleModel::Create();

   {
      auto ntuple1 = RNTupleWriter::Recreate(std::move(model1), "ntpl1", fileGuard1.GetPath());
      *fieldPt = 1.0;
      ntuple1->Fill();
      auto ntuple2 = RNTupleWriter::Recreate(std::move(model2), "ntpl2", fileGuard2.GetPath());
   }

   std::vector<std::unique_ptr<RPageSource>> realSources;
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl1", fileGuard1.GetPath(), RNTupleReadOptions()));
   realSources.emplace_back(std::make_unique<RPageSourceFile>("ntpl2", fileGuard2.GetPath(), RNTupleReadOptions()));
   RPageSourceFriends friendSource("myNTuple", realSources);
   EXPECT_THROW(friendSource.Attach(), ROOT::Experimental::RException);
}
