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

TEST(RNTupleProcessor, PrepareJoinModel)
{
   auto primaryModel = RNTupleModel::Create();
   primaryModel->MakeField<int>("i");
   primaryModel->MakeField<float>("x");

   auto auxModel1 = RNTupleModel::Create();
   auxModel1->MakeField<int>("i");
   auxModel1->MakeField<float>("y");

   auto auxModel2 = RNTupleModel::Create();
   auxModel2->MakeField<int>("i");
   auxModel2->MakeField<float>("z");

   auto joinModel =
      RNTupleProcessor::PrepareJoinModel(*primaryModel, {auxModel1.get(), auxModel2.get()}, {"aux1", "aux2"});

   EXPECT_NO_THROW(joinModel->GetConstField("i"));
   EXPECT_NO_THROW(joinModel->GetConstField("x"));
   EXPECT_NO_THROW(joinModel->GetConstField("aux1.i"));
   EXPECT_NO_THROW(joinModel->GetConstField("aux1.y"));
   EXPECT_THROW(joinModel->GetConstField("y"), ROOT::RException);
   EXPECT_NO_THROW(joinModel->GetConstField("aux2.i"));
   EXPECT_NO_THROW(joinModel->GetConstField("aux2.z"));
   EXPECT_THROW(joinModel->GetConstField("z"), ROOT::RException);
   // TODO add test for projected fields
}

class RNTupleProcessorTest : public testing::Test {
protected:
   const std::array<std::string, 2> fFileNames{"test_ntuple_processor1.root", "test_ntuple_processor2.root"};
   const std::array<std::string, 2> fNTupleNames{"ntuple", "ntuple_aux"};

   void SetUp() override
   {
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldX = model->MakeField<float>("x");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[0], fFileNames[0]);

         for (unsigned i = 0; i < 5; i++) {
            *fldI = i;
            *fldX = static_cast<float>(i);
            *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[1], fFileNames[1]);

         for (unsigned i = 0; i < 5; ++i) {
            *fldI = i;
            *fldZ = i * 2.f;
            ntuple->Fill();
         }
      }
   }
};

TEST_F(RNTupleProcessorTest, Base)
{
   RNTupleOpenSpec ntuple{fNTupleNames[0], fFileNames[0]};
   auto proc = RNTupleProcessor::Create(ntuple);

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(nEntries - 1), *entry.GetPtr<float>("x"));

      std::vector<float> yExp{static_cast<float>(nEntries - 1), static_cast<float>((nEntries - 1) * 2)};
      EXPECT_EQ(yExp, *entry.GetPtr<std::vector<float>>("y"));
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, BaseWithModel)
{
   RNTupleOpenSpec ntuple{fNTupleNames[0], fFileNames[0]};

   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<float>("x");

   auto proc = RNTupleProcessor::Create(ntuple, std::move(model));

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());

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
   RNTupleOpenSpec ntuple{fNTupleNames[0], fFileNames[0]};

   auto model = RNTupleModel::CreateBare();
   model->MakeField<float>("x");

   auto proc = RNTupleProcessor::Create(ntuple, std::move(model));

   EXPECT_STREQ("ntuple", proc->GetProcessorName().c_str());

   {
      auto namedProc = RNTupleProcessor::Create(ntuple, "my_ntuple");
      EXPECT_STREQ("my_ntuple", namedProc->GetProcessorName().c_str());
   }

   int nEntries = 0;

   for (const auto &entry : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());

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

TEST_F(RNTupleProcessorTest, ChainedChain)
{
   std::vector<RNTupleOpenSpec> ntuples{{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}};

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(RNTupleProcessor::CreateChain(ntuples));
   innerProcs.push_back(RNTupleProcessor::Create(ntuples[0]));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   int nEntries = 0;

   for (const auto &entry [[maybe_unused]] : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*entry.GetPtr<int>("i"), proc->GetCurrentEntryNumber() % 5);
      EXPECT_EQ(static_cast<float>(*entry.GetPtr<int>("i")), *entry.GetPtr<float>("x"));
   }
   EXPECT_EQ(nEntries, 15);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, ChainedJoin)
{
   std::vector<RNTupleOpenSpec> ntuples{{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}};

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(RNTupleProcessor::CreateJoin(ntuples, {}));
   innerProcs.push_back(RNTupleProcessor::CreateJoin(ntuples, {"i"}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   int nEntries = 0;

   auto x = proc->GetEntry().GetPtr<float>("x");

   for (const auto &entry [[maybe_unused]] : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*entry.GetPtr<int>("i"), proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*entry.GetPtr<int>("i")), *x);
      EXPECT_EQ(*x * 2, *entry.GetPtr<float>("ntuple_aux.z"));
   }
   EXPECT_EQ(nEntries, 10);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}
