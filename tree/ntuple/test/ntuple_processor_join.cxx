#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

#include <TMemFile.h>

class RNTupleJoinProcessorTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_join_processor1.root", "test_ntuple_join_processor2.root",
                                               "test_ntuple_join_processor3.root", "test_ntuple_join_processor4.root"};

   const std::array<std::string, 4> fNTupleNames{"ntuple1", "ntuple2", "ntuple3", "ntuple4"};

   void SetUp() override
   {
      // The first ntuple is unaligned (fewer entries, but still ordered) with respect to the second and third.
      // The second and third ntuples are aligned with respect to each other.
      // The fourth ntuple has its entries shuffled on field 'i'. It has all entries present in the first ntuple (plus
      // additional), but not all entries present in the second or third.
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldJ = model->MakeField<int>("j");
         auto fldK = model->MakeField<int>("k");
         auto fldX = model->MakeField<float>("x");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[0], fFileNames[0]);

         for (unsigned i = 0; i < 5; ++i) {
            *fldI = i * 2;
            *fldJ = i;
            *fldK = i / 2;
            *fldX = *fldI * 0.5f;
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[1], fFileNames[1]);

         for (unsigned i = 0; i < 10; ++i) {
            *fldI = i;
            *fldY = {static_cast<float>(*fldI * 0.2), 3.14, static_cast<float>(*fldI * 1.3)};
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[2], fFileNames[2]);

         for (unsigned i = 0; i < 10; ++i) {
            *fldI = i;
            *fldZ = *fldI * 2.f;
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldJ = model->MakeField<int>("j");
         auto fldK = model->MakeField<int>("k");
         auto fldA = model->MakeField<float>("a");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[3], fFileNames[3]);

         for (const auto &i : {4, 14, 8, 11, 10, 0, 13, 1, 5, 6, 12, 2, 7}) {
            *fldI = i;
            *fldJ = *fldI / 2;
            *fldK = *fldJ / 2;
            *fldA = *fldI * 0.1f;
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

TEST_F(RNTupleJoinProcessorTest, Aligned)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}, {});

   auto i = proc->GetValuePtr<int>("i");
   auto y = proc->GetValuePtr<std::vector<float>>("y");
   auto z = proc->GetValuePtr<float>("ntuple3.z");

   int nEntries = 0;
   std::vector<float> yExpected;
   for (auto &_ : *proc) {
      EXPECT_EQ(++nEntries, proc->GetNEntriesProcessed());
      EXPECT_EQ(nEntries - 1, proc->GetCurrentEntryNumber());

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *y);

      EXPECT_FLOAT_EQ(*i * 2.f, *z);
   }

   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, IdenticalFieldNames)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}, {});

   auto iPrimary = proc->GetValuePtr<int>("i");
   auto iAux = proc->GetValuePtr<int>("ntuple3.i");

   for (auto &_ : *proc) {
      EXPECT_NE(iPrimary.GetPtr(), iAux.GetPtr());
      EXPECT_EQ(*iPrimary, *iAux);
   }

   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, UnalignedSingleJoinField)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {"i"});

   int nEntries = 0;
   auto i = proc->GetValuePtr<int>("i");
   auto x = proc->GetValuePtr<float>("x");
   auto y = proc->GetValuePtr<std::vector<float>>("ntuple2.y");
   std::vector<float> yExpected;
   for ([[maybe_unused]] auto &entry : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryNumber(), nEntries++);

      EXPECT_FLOAT_EQ(proc->GetCurrentEntryNumber() * 2, *i);
      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *y);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, UnalignedMultipleJoinFields)
{
   try {
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]},
                                   {"i", "j", "k", "l", "m"});
      FAIL() << "trying to create a join processor with more than four join fields should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("a maximum of four join fields is allowed"));
   }

   try {
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i", "i"});
      FAIL() << "trying to create a join processor with duplicate join fields should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("join fields must be unique"));
   }

   try {
      auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]},
                                               {"i", "j", "k"});
      proc->begin();
      FAIL() << "trying to use a join processor where not all join fields are present should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("could not find join field \"j\" in RNTuple \"ntuple2\""));
   }

   auto proc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i", "j", "k"});

   int nEntries = 0;
   auto i = proc->GetValuePtr<int>("i");
   auto x = proc->GetValuePtr<float>("x");
   auto a = proc->GetValuePtr<float>("ntuple4.a");
   for ([[maybe_unused]] auto &entry : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryNumber(), nEntries++);

      EXPECT_FLOAT_EQ(proc->GetCurrentEntryNumber() * 2, *i);
      EXPECT_FLOAT_EQ(*i * 0.5f, *x);
      EXPECT_EQ(*i * 0.1f, *a);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, MissingEntries)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[3], fFileNames[3]}, {"i"});

   auto i = proc->GetValuePtr<int>("i");
   auto a = proc->GetValuePtr<float>("ntuple4.a");
   std::vector<float> yExpected;

   auto procIter = proc->begin();
   EXPECT_EQ(*i * 0.1f, *a);
   ++procIter;
   EXPECT_EQ(*i * 0.1f, *a);
   ++procIter;
   EXPECT_EQ(2ULL, proc->GetCurrentEntryNumber());
   ++procIter;
   EXPECT_EQ(i.GetPtr(), nullptr);
   try {
      *i;
      FAIL() << "dereferencing a value from an invalid entry should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot read \"i\" because the entry it belongs to is invalid"));
   }
}

TEST_F(RNTupleJoinProcessorTest, WithModel)
{
   auto primaryModel = RNTupleModel::Create();
   auto fldI = primaryModel->MakeField<int>("i");
   auto fldX = primaryModel->MakeField<float>("x");

   auto auxModel = RNTupleModel::Create();
   auto fldY = auxModel->MakeField<std::vector<float>>("y");

   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {"i"},
                                            std::move(primaryModel), std::move(auxModel));

   auto i = proc->GetValuePtr<int>("i");
   auto x = proc->GetValuePtr<float>("x");
   auto y = proc->GetValuePtr<std::vector<float>>("ntuple2.y");

   try {
      proc->GetValuePtr<float>("ntuple2.z");
      FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: ntuple2.z"));
   }

   int nEntries = 0;
   std::vector<float> yExpected;
   for (auto &_ : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryNumber(), nEntries++);

      EXPECT_EQ(proc->GetCurrentEntryNumber() * 2, *i);
      EXPECT_EQ(*fldI, *i);

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);
      EXPECT_FLOAT_EQ(*fldX, *x);

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *y);
      EXPECT_EQ(*fldY, *y);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, WithBareModel)
{
   auto primaryModel = RNTupleModel::CreateBare();
   primaryModel->MakeField<int>("i");
   primaryModel->MakeField<float>("x");

   auto auxModel = RNTupleModel::CreateBare();
   auxModel->MakeField<std::vector<float>>("y");

   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {"i"},
                                            std::move(primaryModel), std::move(auxModel));

   auto i = proc->GetValuePtr<int>("i");
   auto x = proc->GetValuePtr<float>("x");
   auto y = proc->GetValuePtr<std::vector<float>>("ntuple2.y");

   try {
      proc->GetValuePtr<float>("ntuple2.z");
      FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: ntuple2.z"));
   }

   int nEntries = 0;
   std::vector<float> yExpected;
   for (auto &_ : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryNumber(), nEntries++);

      EXPECT_EQ(proc->GetCurrentEntryNumber() * 2, *i);

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *y);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, TMemFile)
{
   TMemFile memFile("test_ntuple_processor_join_tmemfile.root", "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fldI = model->MakeField<int>("i");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Append(std::move(model), "ntuple_aux", memFile);

      for (unsigned i = 0; i < 10; ++i) {
         *fldI = i;
         *fldY = {static_cast<float>(*fldI * 0.2), 3.14, static_cast<float>(*fldI * 1.3)};
         ntuple->Fill();
      }
   }

   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {"ntuple_aux", &memFile}, {"i"});

   int nEntries = 0;
   auto i = proc->GetValuePtr<int>("i");
   auto x = proc->GetValuePtr<float>("x");
   auto y = proc->GetValuePtr<std::vector<float>>("ntuple_aux.y");
   std::vector<float> yExpected;
   for ([[maybe_unused]] auto &entry : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryNumber(), nEntries++);

      EXPECT_FLOAT_EQ(proc->GetCurrentEntryNumber() * 2, *i);
      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *y);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}
