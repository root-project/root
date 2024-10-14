#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

class RNTupleJoinProcessorTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_join_processor1.root", "test_ntuple_join_processor2.root",
                                               "test_ntuple_join_processor3.root", "test_ntuple_join_processor4.root"};

   const std::array<std::string, 4> fNTupleNames{"ntuple1", "ntuple2", "ntuple3", "ntuple4"};

   void SetUp() override
   {
      // The first ntuple is unaligned (fewer entries, but still ordered) with respect to the second and third.
      // The second and third ntuples are aligned with respect to each other.
      // The fourth ntuple is larger than the first ntuple and its entries shuffled on field 'i'.
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldJ = model->MakeField<int>("j");
         auto fldK = model->MakeField<int>("k");
         auto fldX = model->MakeField<float>("x");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[0], fFileNames[0]);

         for (*fldI = 0; *fldI < 10; ++(*fldI)) {
            *fldJ = *fldI / 2;
            *fldK = *fldJ / 2;
            *fldX = *fldI * 0.5f;
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[1], fFileNames[1]);

         for (*fldI = 0; *fldI < 10; ++(*fldI)) {
            if (*fldI % 2 == 1) {
               *fldY = {static_cast<float>(*fldI * 0.2), 3.14, static_cast<float>(*fldI * 1.3)};
               ntuple->Fill();
            }
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[2], fFileNames[2]);

         for (*fldI = 0; *fldI < 10; ++(*fldI)) {
            if (*fldI % 2 == 1) {
               *fldZ = *fldI * 2.f;
               ntuple->Fill();
            }
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldJ = model->MakeField<int>("j");
         auto fldK = model->MakeField<int>("k");
         auto fldA = model->MakeField<float>("a");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[3], fFileNames[3]);

         for (const auto &i : {3, 14, 9, 8, 11, 10, 0, 13, 1, 5, 6, 12, 2, 4, 7}) {
            *fldI = i;
            *fldJ = *fldI / 2;
            *fldK = *fldJ / 2;
            if (*fldI % 2 == 1) {
               *fldA = *fldI * 0.1f;
               ntuple->Fill();
            }
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

TEST_F(RNTupleJoinProcessorTest, Basic)
{
   std::vector<RNTupleOpenSpec> ntuples;
   try {
      auto proc = RNTupleProcessor::CreateJoin(ntuples, {});
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }

   ntuples = {{fNTupleNames[0], fFileNames[0]}};

   int nEntries = 0;
   auto proc = RNTupleProcessor::CreateJoin(ntuples, {});
   for (const auto &entry : *proc) {
      auto i = entry.GetPtr<int>("i");
      EXPECT_EQ(proc->GetNEntriesProcessed(), *i);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 10);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, Aligned)
{
   try {
      std::vector<RNTupleOpenSpec> ntuples = {{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}};
      auto proc = RNTupleProcessor::CreateJoin(ntuples, {});
      FAIL() << "ntuples with the same name cannot be joined horizontally";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("horizontal joining of RNTuples with the same name is not allowed"));
   }

   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples, {});

   std::vector<float> yExpected;

   int nEntries = 0;
   for (auto &entry : *proc) {
      auto i = entry.GetPtr<int>("i");

      yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
      EXPECT_EQ(yExpected, *entry.GetPtr<std::vector<float>>("y"));

      EXPECT_FLOAT_EQ(*i * 2.f, *entry.GetPtr<float>("ntuple3.z"));

      ++nEntries;
   }
   EXPECT_EQ(nEntries, 5);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleJoinProcessorTest, IdenticalFieldNames)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples, {});

   auto i = proc->GetEntry().GetPtr<int>("i");
   for (auto &entry : *proc) {
      EXPECT_EQ(*i, *entry.GetPtr<int>("ntuple3.i"));
   }
}

TEST_F(RNTupleJoinProcessorTest, UnalignedSingleJoinField)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples, {"i"});

   int nEntries = 0;
   auto i = proc->GetEntry().GetPtr<int>("i");
   auto x = proc->GetEntry().GetPtr<float>("x");
   auto y = proc->GetEntry().GetPtr<std::vector<float>>("ntuple2.y");
   std::vector<float> yExpected;
   for (auto &entry : *proc) {
      EXPECT_FLOAT_EQ(nEntries, *entry.GetPtr<int>("i"));

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      if (*i % 2 == 1) {
         yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
         EXPECT_EQ(yExpected, *entry.GetPtr<std::vector<float>>("ntuple2.y"));
      } else {
         yExpected = {};
         EXPECT_EQ(yExpected, *y);
      }

      ++nEntries;
   }
}

TEST_F(RNTupleJoinProcessorTest, UnalignedMultipleJoinFields)
{
   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}};

   try {
      RNTupleProcessor::CreateJoin(ntuples, {"i", "j", "k", "l", "m"});
      FAIL() << "trying to create a join processor with more than four join fields should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("a maximum of four join fields is allowed"));
   }

   try {
      RNTupleProcessor::CreateJoin(ntuples, {"i", "i"});
      FAIL() << "trying to create a join processor with duplicate join fields should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("join fields must be unique"));
   }

   try {
      std::vector<RNTupleOpenSpec> unfitNTuples = {{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}};
      RNTupleProcessor::CreateJoin(unfitNTuples, {"i", "j", "k"});
      FAIL() << "trying to create a join processor where not all join fields are present should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("could not find join field \"j\" in RNTuple \"ntuple2\""));
   }

   auto proc = RNTupleProcessor::CreateJoin(ntuples, {"i", "j", "k"});

   int nEntries = 0;
   auto i = proc->GetEntry().GetPtr<int>("i");
   auto x = proc->GetEntry().GetPtr<float>("x");
   auto a = proc->GetEntry().GetPtr<float>("ntuple4.a");
   for (auto &entry : *proc) {
      EXPECT_FLOAT_EQ(nEntries, *entry.GetPtr<int>("i"));

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);

      if (*i % 2 == 1) {
         EXPECT_EQ(*i * 0.1f, *entry.GetPtr<float>("ntuple4.a"));
      } else {
         EXPECT_EQ(0.f, *a);
      }

      ++nEntries;
   }
}

TEST_F(RNTupleJoinProcessorTest, WithModel)
{
   auto model1 = RNTupleModel::Create();
   auto i = model1->MakeField<int>("i");
   auto x = model1->MakeField<float>("x");

   auto model2 = RNTupleModel::Create();
   auto y = model2->MakeField<std::vector<float>>("y");

   auto model3 = RNTupleModel::Create();
   auto z = model3->MakeField<float>("z");

   std::vector<std::unique_ptr<RNTupleModel>> models;
   models.push_back(std::move(model1));
   models.push_back(std::move(model2));
   models.push_back(std::move(model3));

   std::vector<RNTupleOpenSpec> ntuples = {
      {fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}};

   auto proc = RNTupleProcessor::CreateJoin(ntuples, {"i"}, std::move(models));

   std::vector<float> yExpected;
   int nEntries = 0;
   for (auto &entry : *proc) {
      EXPECT_EQ(nEntries, *i);
      EXPECT_EQ(*entry.GetPtr<int>("i"), *i);

      EXPECT_FLOAT_EQ(*i * 0.5f, *x);
      EXPECT_FLOAT_EQ(*entry.GetPtr<float>("x"), *x);

      if (*i % 2 == 1) {
         yExpected = {static_cast<float>(*i * 0.2), 3.14, static_cast<float>(*i * 1.3)};
         EXPECT_EQ(yExpected, *y);
         EXPECT_EQ(*entry.GetPtr<std::vector<float>>("ntuple2.y"), *y);
         EXPECT_FLOAT_EQ(static_cast<float>(*i * 2.f), *z);
         EXPECT_FLOAT_EQ(*entry.GetPtr<float>("ntuple3.z"), *z);
      } else {
         yExpected = {};
         EXPECT_EQ(yExpected, *y);
         EXPECT_EQ(0., *z);
      }

      try {
         entry.GetPtr<float>("ntuple2.z");
         FAIL() << "should not be able to access values from fields not present in the provided models";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: ntuple2.z"));
      }

      ++nEntries;
   }
}
