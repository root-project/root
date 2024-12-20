#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

class RNTupleChainProcessorTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_chain_processor1.root", "test_ntuple_chain_processor2.root",
                                               "test_ntuple_chain_processor3.root",
                                               "test_ntuple_chain_processor4.root"};

   const std::string fNTupleName = "ntuple";

   void SetUp() override
   {
      {
         auto model = RNTupleModel::Create();
         auto fldX = model->MakeField<float>("x");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fFileNames[0]);

         for (unsigned i = 0; i < 5; i++) {
            *fldX = static_cast<float>(i);
            *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldX = model->MakeField<float>("x");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fFileNames[1]);

         for (unsigned i = 5; i < 8; i++) {
            *fldX = static_cast<float>(i);
            *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldX = model->MakeField<float>("x");
         // missing field y
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fFileNames[2]);

         for (unsigned i = 8; i < 15; i++) {
            *fldX = static_cast<float>(i);
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldX = model->MakeField<float>("x");
         auto fldY = model->MakeField<int>("y"); // different type for y
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fFileNames[3]);

         for (unsigned i = 15; i < 20; i++) {
            *fldX = static_cast<float>(i);
            *fldY = i * 2;
            ntuple->Fill();
         }
      }
   }

   void TearDown() override
   {
      for (const auto &fileName : fFileNames) {
         std::remove(fileName.c_str());
      }
   }
};

TEST(RNTupleChainProcessor, EmptySpec)
{
   std::vector<RNTupleOpenSpec> ntuples;
   try {
      auto proc = RNTupleProcessor::CreateChain(ntuples);
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }
}

TEST_F(RNTupleChainProcessorTest, SingleNTuple)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fFileNames[0]}};

   int nEntries = 0;
   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber());

      auto x = entry.GetPtr<float>("x");
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetLocalEntryNumber()), *x);
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleChainProcessorTest, Basic)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}};

   std::uint64_t nEntries = 0;
   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      if (proc->GetCurrentNTupleNumber() == 0) {
         EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber());
      } else {
         EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber() + 5);
      }

      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(nEntries - 1), *x);

      auto y = entry.GetPtr<std::vector<float>>("y");
      std::vector<float> yExp = {static_cast<float>(nEntries - 1), static_cast<float>((nEntries - 1) * 2)};
      EXPECT_EQ(yExp, *y);
   }
   EXPECT_EQ(nEntries, 8);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleChainProcessorTest, WithModel)
{
   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<float>("x");

   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}};

   auto proc = RNTupleProcessor::CreateChain(ntuples, std::move(model));
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(proc->GetNEntriesProcessed() - 1), *x);

      try {
         entry.GetPtr<std::vector<float>>("y");
         FAIL() << "fields not specified by the provided model shoud not be part of the entry";
      } catch (const ROOT::RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
      }
   }
}

TEST_F(RNTupleChainProcessorTest, WithBareModel)
{
   auto model = RNTupleModel::CreateBare();
   auto fldY = model->MakeField<std::vector<float>>("y");

   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}};

   auto proc = RNTupleProcessor::CreateChain(ntuples, std::move(model));
   for (const auto &entry : *proc) {
      auto y = entry.GetPtr<std::vector<float>>("y");
      std::vector<float> yExp = {static_cast<float>(proc->GetNEntriesProcessed() - 1),
                                 static_cast<float>((proc->GetNEntriesProcessed() - 1) * 2)};
      EXPECT_EQ(yExp, *y);

      try {
         entry.GetPtr<float>("x");
         FAIL() << "fields not specified by the provided model shoud not be part of the entry";
      } catch (const ROOT::RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: x"));
      }
   }
}

TEST_F(RNTupleChainProcessorTest, MissingFields)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[2]}};

   auto proc = RNTupleProcessor::CreateChain(ntuples);
   auto entry = proc->begin();

   while (proc->GetNEntriesProcessed() < 5) {
      auto x = (*entry).GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(proc->GetNEntriesProcessed() - 1), *x);
      entry++;
   }

   try {
      entry++;
      FAIL() << "having missing fields in subsequent ntuples should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field \"y\" not found in current RNTuple"));
   }
}

TEST_F(RNTupleChainProcessorTest, EmptyNTuples)
{
   FileRaii fileGuard("test_ntuple_processor_empty_ntuples.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fileGuard.GetPath());
   }

   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fileGuard.GetPath()}, {fNTupleName, fFileNames[0]}};

   std::uint64_t nEntries = 0;

   try {
      auto proc = RNTupleProcessor::CreateChain(ntuples);
      FAIL() << "creating a processor where the first RNTuple does not contain any entries should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("first RNTuple does not contain any entries"));
   }

   // Empty ntuples in the middle are just skipped (as long as their model complies)
   ntuples = {{fNTupleName, fFileNames[0]}, {fNTupleName, fileGuard.GetPath()}, {fNTupleName, fFileNames[1]}};

   auto proc = RNTupleProcessor::CreateChain(ntuples);
   for (const auto &entry : *proc) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(nEntries), *x);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 8);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}
