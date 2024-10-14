#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

TEST(RNTupleJoinProcessor, Basic)
{
   FileRaii fileGuard("test_ntuple_join_processor_basic.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   std::vector<RNTupleSourceSpec> ntuples;
   try {
      auto proc = RNTupleProcessor::CreateJoin(ntuples);
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }

   ntuples = {{"ntuple", fileGuard.GetPath()}};

   int nEntries = 0;
   auto proc = RNTupleProcessor::CreateJoin(ntuples);
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *x);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleJoinProcessor, Aligned)
{
   FileRaii fileGuard1("test_ntuple_join_processor_aligned1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple1", fileGuard1.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_join_processor_aligned2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple2", fileGuard2.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldY = {static_cast<float>(i * 0.2), 3.14, static_cast<float>(i * 1.3)};
         ntuple->Fill();
      }
   }
   try {
      std::vector<RNTupleSourceSpec> ntuples = {{"ntuple1", fileGuard1.GetPath()}, {"ntuple1", fileGuard1.GetPath()}};
      auto proc = RNTupleProcessor::CreateJoin(ntuples);
      FAIL() << "ntuples with the same name cannot be joined horizontally";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("horizontal joining of RNTuples with the same name is not allowed"));
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple1", fileGuard1.GetPath()}, {"ntuple2", fileGuard2.GetPath()}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples);

   std::vector<float> yExpected;

   int nEntries = 0;
   for (auto &entry : *proc) {
      EXPECT_FLOAT_EQ(nEntries, *entry.GetPtr<float>("x"));

      yExpected = {static_cast<float>(nEntries * 0.2), 3.14, static_cast<float>(nEntries * 1.3)};
      EXPECT_EQ(yExpected, *entry.GetPtr<std::vector<float>>("ntuple2#y"));

      ++nEntries;
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleJoinProcessor, IdenticalFieldNames)
{
   FileRaii fileGuard1("test_ntuple_join_processor_field_names1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple1", fileGuard1.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_join_processor_field_names2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple2", fileGuard2.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple1", fileGuard1.GetPath()}, {"ntuple2", fileGuard2.GetPath()}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples);

   int nEntries = 0;
   auto x = proc->GetEntry().GetPtr<float>("x");
   for (auto &entry : *proc) {
      EXPECT_FLOAT_EQ(*x, *entry.GetPtr<float>("ntuple2#x"));
      EXPECT_EQ(*x, nEntries);
      ++nEntries;
   }
   EXPECT_EQ(*x, proc->GetNEntriesProcessed() - 1);
}

TEST(RNTupleJoinProcessor, UnalignedBasic)
{
   FileRaii fileGuard1("test_ntuple_join_processor_unaligned_basic1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldI = model->MakeField<int>("i");
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple1", fileGuard1.GetPath());

      for (*fldI = 0; *fldI < 5; ++(*fldI)) {
         *fldX = *fldI * 0.5f;
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_join_processor_unaligned_basic2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldI = model->MakeField<int>("i");
      auto fldY = model->MakeField<float>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple2", fileGuard2.GetPath());

      for (*fldI = 0; *fldI < 5; ++(*fldI)) {
         if (*fldI % 2 == 1) {
            *fldY = *fldI * 0.2f;
            std::cout << *fldY << std::endl;
            ntuple->Fill();
         }
      }
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple1", fileGuard1.GetPath()}, {"ntuple2", fileGuard2.GetPath()}};
   auto proc = RNTupleProcessor::CreateJoin(ntuples, {"i"});

   int nEntries = 0;
   auto i = proc->GetEntry().GetPtr<int>("i");
   auto x = proc->GetEntry().GetPtr<float>("x");
   auto y = proc->GetEntry().GetPtr<float>("ntuple2#y");
   for (auto &entry : *proc) {
      std::cout << *i << ": x = " << *x << ", y = " << *y << std::endl;
      EXPECT_FLOAT_EQ(nEntries, *entry.GetPtr<int>("i"));

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      if (*i % 2 == 1) {
         EXPECT_FLOAT_EQ(static_cast<float>(*i * 0.2f), *y);
      } else {
         EXPECT_EQ(0., *y);
      }

      ++nEntries;
   }
}
