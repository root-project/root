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

   auto i = proc->GetView<int>("i");
   auto y = proc->GetView<std::vector<float>>("y");
   auto z = proc->GetView<float>("ntuple3.z");

   std::vector<float> yExpected;

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryIndex());

      yExpected = {static_cast<float>(i(idx) * 0.2), 3.14, static_cast<float>(i(idx) * 1.3)};
      EXPECT_EQ(yExpected, y(idx));

      EXPECT_FLOAT_EQ(i(idx) * 2.f, z(idx));
   }

   EXPECT_EQ(9, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleJoinProcessorTest, IdenticalFieldNames)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}, {});

   auto iPrimary = proc->GetView<int>("i");
   auto iAux = proc->GetView<int>("ntuple3.i");

   EXPECT_NE(&iPrimary.GetField(), &iAux.GetField());
   for (auto idx : *proc) {
      EXPECT_EQ(iPrimary(idx), iAux(idx));
   }

   EXPECT_EQ(9, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleJoinProcessorTest, UnalignedSingleJoinField)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {"i"});

   auto iPrimary = proc->GetView<int>("i");
   auto iAux = proc->GetView<int>("ntuple2.i");
   auto x = proc->GetView<float>("x");
   auto y = proc->GetView<std::vector<float>>("ntuple2.y");

   std::vector<float> yExpected;

   for (auto idx : *proc) {
      EXPECT_EQ(idx * 2, iPrimary(idx));
      EXPECT_EQ(iPrimary(idx), iAux(idx));
      EXPECT_FLOAT_EQ(iPrimary(idx) * 0.5f, x(idx));

      yExpected = {static_cast<float>(iPrimary(idx) * 0.2), 3.14, static_cast<float>(iPrimary(idx) * 1.3)};
      EXPECT_EQ(yExpected, y(idx));
   }

   EXPECT_EQ(4, proc->GetCurrentEntryIndex());
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
      auto proc =
         RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {"i", "j"});
      proc->begin();
      FAIL() << "trying to use a join processor where not all join fields are present should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("could not find join field \"j\" in auxiliary processor \"ntuple2\""));
   }

   auto proc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i", "j", "k"});

   auto i = proc->GetView<int>("i");
   auto x = proc->GetView<float>("x");
   auto a = proc->GetView<float>("ntuple4.a");

   for (auto idx : *proc) {
      EXPECT_EQ(proc->GetCurrentEntryIndex(), idx);

      EXPECT_FLOAT_EQ(proc->GetCurrentEntryIndex() * 2, i(idx));
      EXPECT_FLOAT_EQ(i(idx) * 0.5f, x(idx));
      EXPECT_EQ(i(idx) * 0.1f, a(idx));
   }

   EXPECT_EQ(4, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleJoinProcessorTest, MissingEntries)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[3], fFileNames[3]}, {"i"});

   auto i = proc->GetView<int>("i");
   auto a = proc->GetView<float>("ntuple4.a");
   std::vector<float> yExpected;

   EXPECT_TRUE(a.IsValid(0));
   EXPECT_EQ(i(0) * 0.1f, a(0));
   EXPECT_TRUE(a.IsValid(1));
   EXPECT_EQ(i(1) * 0.1f, a(1));
   EXPECT_TRUE(a.IsValid(2));
   EXPECT_EQ(i(2) * 0.1f, a(2));

   EXPECT_FALSE(a.IsValid(3));
   EXPECT_THAT([&a] { return a(3); },
               testing::ThrowsMessage<ROOT::RException>(testing::HasSubstr(
                  "cannot read from \"ntuple4.a\" at entry 3 because the field is invalid at this entry")));
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

   auto i = proc->GetView<int>("i");
   auto x = proc->GetView<float>("x");
   auto y = proc->GetView<std::vector<float>>("ntuple_aux.y");

   std::vector<float> yExpected;

   for (auto idx : *proc) {
      EXPECT_EQ(idx * 2, i(idx));

      EXPECT_FLOAT_EQ(i(idx) * 0.5f, x(idx));

      yExpected = {static_cast<float>(i(idx) * 0.2), 3.14, static_cast<float>(i(idx) * 1.3)};
      EXPECT_EQ(yExpected, y(idx));
   }

   EXPECT_EQ(4, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleJoinProcessorTest, PrintStructure)
{
   auto proc = RNTupleProcessor::CreateJoin({fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}, {});

   std::ostringstream os;
   proc->PrintStructure(os);

   const std::string exp = "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple2                     | | ntuple3                     |\n"
                           "| test_ntuple_join_process... | | test_ntuple_join_process... |\n"
                           "+-----------------------------+ +-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}
