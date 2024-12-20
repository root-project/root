#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

TEST(RNTupleProcessor, EmptyNTuple)
{
   FileRaii fileGuard("test_ntuple_processor_empty.root");
   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   RNTupleOpenSpec ntuple{"ntuple", fileGuard.GetPath()};
   auto proc = RNTupleProcessor::Create(ntuple);

   int nEntries = 0;
   for ([[maybe_unused]] const auto &entry : *proc) {
      nEntries++;
   }
   EXPECT_EQ(0, nEntries);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

class RNTupleProcessorTest : public testing::Test {
protected:
   const std::string fFileName = "test_ntuple_processor.root";
   const std::string fNTupleName = "ntuple";

   void SetUp() override
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fFileName);

      for (unsigned i = 0; i < 5; i++) {
         *fldX = static_cast<float>(i);
         *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
         ntuple->Fill();
      }
   }
};

TEST_F(RNTupleProcessorTest, Base)
{
   RNTupleOpenSpec ntuple{fNTupleName, fFileName};
   auto proc = RNTupleProcessor::Create(ntuple);

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(nEntries - 1), *entry.GetPtr<float>("x"));

      std::vector<float> yExp{static_cast<float>(nEntries - 1), static_cast<float>((nEntries - 1) * 2)};
      EXPECT_EQ(yExp, *entry.GetPtr<std::vector<float>>("y"));
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, BaseWithModel)
{
   RNTupleOpenSpec ntuple{fNTupleName, fFileName};

   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<float>("x");

   auto proc = RNTupleProcessor::Create(ntuple, *model);

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(nEntries - 1), *fldX);

      try {
         entry.GetPtr<std::vector<float>>("y");
         FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
      } catch (const ROOT::RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
      }
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, BaseWithBareModel)
{
   RNTupleOpenSpec ntuple{fNTupleName, fFileName};

   auto model = RNTupleModel::CreateBare();
   model->MakeField<float>("x");

   auto proc = RNTupleProcessor::Create(ntuple, *model);

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetLocalEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(nEntries - 1), *entry.GetPtr<float>("x"));

      try {
         entry.GetPtr<std::vector<float>>("y");
         FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
      } catch (const ROOT::RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
      }
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}
