#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

TEST(RNTupleProcessor, Basic)
{
   FileRaii fileGuard("test_ntuple_processor_basic.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 10; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   std::vector<RNTupleSourceSpec> ntuples;
   try {
      auto proc = RNTupleProcessor::CreateChain(ntuples);
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }

   ntuples = {{"ntuple", fileGuard.GetPath()}};

   int nEntries = 0;
   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *x);
      EXPECT_EQ(proc->GetNEntriesProcessed(), proc->GetLocalEntryNumber());
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 10);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleProcessor, WithModel)
{
   FileRaii fileGuard("test_ntuple_processor_basic_with_model.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<float>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 10; ++i) {
         *fldX = static_cast<float>(i);
         *fldY = static_cast<float>(i) * 2.f;
         ntuple->Fill();
      }
   }

   auto model = RNTupleModel::Create();
   auto fldY = model->MakeField<float>("y");

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple", fileGuard.GetPath()}};

   auto proc = RNTupleProcessor::CreateChain(ntuples, std::move(model));
   for (const auto &entry : *proc) {
      try {
         auto x = entry.GetPtr<float>("x");
         FAIL() << "field \"x\" should not be present in the processor entry";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: x"));
      }
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()) * 2.f, *fldY);
   }
}

TEST(RNTupleProcessor, WithBareModel)
{
   FileRaii fileGuard("test_ntuple_processor_basic_with_model.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<float>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 10; ++i) {
         *fldX = static_cast<float>(i);
         *fldY = static_cast<float>(i) * 2.f;
         ntuple->Fill();
      }
   }

   auto model = RNTupleModel::CreateBare();
   model->MakeField<float>("y");
   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple", fileGuard.GetPath()}};

   auto proc = RNTupleProcessor::CreateChain(ntuples, std::move(model));
   for (const auto &entry : *proc) {
      try {
         auto x = entry.GetPtr<float>("x");
         FAIL() << "field \"x\" should not be present in the processor entry";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: x"));
      }
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()) * 2.f, *entry.GetPtr<float>("y"));
   }
}

TEST(RNTupleProcessor, SimpleChain)
{
   FileRaii fileGuard1("test_ntuple_processor_simple_chain1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_processor_simple_chain2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      for (unsigned i = 5; i < 8; ++i) {
         *fldX = static_cast<float>(i);
         *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
         ntuple->Fill();
      }
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple", fileGuard1.GetPath()}, {"ntuple", fileGuard2.GetPath()}};

   std::uint64_t nEntries = 0;
   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *x);

      auto y = entry.GetPtr<std::vector<float>>("y");
      std::vector<float> yExp = {static_cast<float>(proc->GetNEntriesProcessed()), static_cast<float>(nEntries * 2)};
      EXPECT_EQ(yExp, *y);

      if (proc->GetCurrentNTupleNumber() == 0) {
         EXPECT_EQ(proc->GetLocalEntryNumber(), proc->GetNEntriesProcessed());
      } else {
         EXPECT_EQ(proc->GetLocalEntryNumber() + 5, proc->GetNEntriesProcessed());
      }

      ++nEntries;
   }
   EXPECT_EQ(nEntries, 8);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleProcessor, SimpleChainWithModel)
{
   FileRaii fileGuard1("test_ntuple_processor_simple_chain_with_model1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x", 1.f);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      ntuple->Fill();
   }
   FileRaii fileGuard2("test_ntuple_processor_simple_chain_with_model2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x", 2.f);
      auto fldY = model->MakeField<int>("y", 404);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      ntuple->Fill();
   }
   FileRaii fileGuard3("test_ntuple_processor_simple_chain_with_model3.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x", 3.f);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard3.GetPath());
      ntuple->Fill();
   }

   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<float>("x");

   std::vector<RNTupleSourceSpec> ntuples = {
      {"ntuple", fileGuard1.GetPath()}, {"ntuple", fileGuard2.GetPath()}, {"ntuple", fileGuard3.GetPath()}};

   auto proc = RNTupleProcessor::CreateChain(ntuples, std::move(model));
   auto entry = proc->begin();
   *entry;
   EXPECT_EQ(1.f, *fldX);
   entry++;
   EXPECT_EQ(2.f, *fldX);
   try {
      (*entry).GetPtr<int>("y");
      FAIL() << "fields not specified by the provided model shoud not be part of the entry";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
   }
   ++entry;
   EXPECT_EQ(3.f, *fldX);
}

TEST(RNTupleProcessor, EmptyNTuples)
{
   FileRaii fileGuard1("test_ntuple_processor_empty_ntuples1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
   }
   FileRaii fileGuard2("test_ntuple_processor_empty_ntuples2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      for (unsigned i = 0; i < 2; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }
   FileRaii fileGuard3("test_ntuple_processor_empty_ntuples3.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard3.GetPath());
   }
   FileRaii fileGuard4("test_ntuple_processor_empty_ntuples4.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard4.GetPath());

      for (unsigned i = 2; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }
   FileRaii fileGuard5("test_ntuple_processor_empty_ntuples5.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard5.GetPath());
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple", fileGuard1.GetPath()},
                                             {"ntuple", fileGuard2.GetPath()},
                                             {"ntuple", fileGuard3.GetPath()},
                                             {"ntuple", fileGuard4.GetPath()},
                                             {"ntuple", fileGuard5.GetPath()}};

   std::uint64_t nEntries = 0;

   try {
      auto proc = RNTupleProcessor::CreateChain(ntuples);
      FAIL() << "creating a processor where the first RNTuple does not contain any entries should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("first RNTuple does not contain any entries"));
   }

   ntuples = {{"ntuple", fileGuard2.GetPath()},
              {"ntuple", fileGuard3.GetPath()},
              {"ntuple", fileGuard4.GetPath()},
              {"ntuple", fileGuard5.GetPath()}};

   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(nEntries), *x);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleProcessor, ChainUnalignedModels)
{
   FileRaii fileGuard1("test_ntuple_processor_chain_unaligned_models1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x", 0.);
      auto fldY = model->MakeField<char>("y", 'a');
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      ntuple->Fill();
   }
   FileRaii fileGuard2("test_ntuple_processor_chain_unaligned_models2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x", 1.);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      ntuple->Fill();
   }

   std::vector<RNTupleSourceSpec> ntuples = {{"ntuple", fileGuard1.GetPath()}, {"ntuple", fileGuard2.GetPath()}};

   auto proc = RNTupleProcessor::CreateChain(ntuples);
   auto entry = proc->begin();
   auto x = (*entry).GetPtr<float>("x");
   auto y = (*entry).GetPtr<char>("y");
   EXPECT_EQ(0., *x);
   EXPECT_EQ('a', *y);

   try {
      entry++;
      FAIL() << "trying to connect a new page source which doesn't have all initial fields is not supported";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field \"y\" not found in current RNTuple"));
   }
}
