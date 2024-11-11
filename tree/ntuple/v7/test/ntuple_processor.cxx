#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

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
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *entry.GetPtr<float>("x"));

      std::vector<float> yExp{static_cast<float>(proc->GetNEntriesProcessed()),
                              static_cast<float>(proc->GetNEntriesProcessed() * 2)};
      EXPECT_EQ(yExp, *entry.GetPtr<std::vector<float>>("y"));

      EXPECT_EQ(proc->GetNEntriesProcessed(), proc->GetLocalEntryNumber());
      ++nEntries;
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
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *fldX);
      EXPECT_EQ(proc->GetNEntriesProcessed(), proc->GetLocalEntryNumber());

      try {
         entry.GetPtr<std::vector<float>>("y");
         FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
      }
      ++nEntries;
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
      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetNEntriesProcessed()), *entry.GetPtr<float>("x"));
      EXPECT_EQ(proc->GetNEntriesProcessed(), proc->GetLocalEntryNumber());

      try {
         entry.GetPtr<std::vector<float>>("y");
         FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
      }
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}
