#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

#include <TMemFile.h>

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

         for (unsigned i = 5; i < 10; i++) {
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

         for (unsigned i = 10; i < 15; i++) {
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
   try {
      auto proc = RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec>{});
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }
}

TEST_F(RNTupleChainProcessorTest, SingleNTuple)
{
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}});

   auto x = proc->GetView<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryIndex());

      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetCurrentEntryIndex()), x(idx));
   }
   EXPECT_EQ(4, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleChainProcessorTest, Basic)
{
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});

   EXPECT_STREQ("ntuple", proc->GetProcessorName().c_str());

   {
      auto namedProc =
         RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}}, "my_ntuple");
      EXPECT_STREQ("my_ntuple", namedProc->GetProcessorName().c_str());
   }

   auto x = proc->GetView<float>("x");
   auto y = proc->GetView<std::vector<float>>("y");

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryIndex());

      EXPECT_EQ(static_cast<float>(idx), x(idx));

      std::vector<float> yExp{x(idx), (x(idx) * 2)};
      EXPECT_EQ(yExp, y(idx));
   }
   EXPECT_EQ(9, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleChainProcessorTest, MissingFields)
{
   auto proc = RNTupleProcessor::CreateChain(
      {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[2]}, {fNTupleName, fFileNames[0]}});

   auto y = proc->GetView<std::vector<float>>("y");

   std::vector<float> yExp;

   for (auto idx : *proc) {
      if (idx >= 5 && idx < 10) {
         EXPECT_FALSE(y.IsValid(idx));
         EXPECT_THROW(y(idx), ROOT::RException);
      } else {
         yExp = {static_cast<float>(idx % 5), static_cast<float>((idx % 5) * 2)};
         EXPECT_TRUE(y.IsValid(idx));
         EXPECT_EQ(yExp, y(idx));
      }
   }

   EXPECT_EQ(14, proc->GetCurrentEntryIndex());
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

   // Empty ntuples are skipped (as long as their model complies)
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fileGuard.GetPath()},
                                              {fNTupleName, fFileNames[0]},
                                              {fNTupleName, fileGuard.GetPath()},
                                              {fNTupleName, fFileNames[1]}});

   auto x = proc->GetView<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(static_cast<float>(idx), x(idx));
   }
   EXPECT_EQ(9, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleChainProcessorTest, LoadRandomEntry)
{
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});
   auto x = proc->GetView<float>("x");

   EXPECT_EQ(3.f, x(3)); // start at the first processor in the chain
   EXPECT_EQ(9.f, x(9)); // jump to the next processor in the chain
   EXPECT_EQ(6.f, x(6)); // stay at the same processor
   EXPECT_EQ(2.f, x(2)); // jump back to the first processor in the chain
   try {
      x(10);
      FAIL() << "should not be able to read non-existent entries";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("index 10 out of bounds"));
   }
}

TEST_F(RNTupleChainProcessorTest, TMemFile)
{
   TMemFile memFile("test_ntuple_processor_chain_tmemfile_second.root", "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Append(std::move(model), fNTupleName, memFile);

      for (unsigned i = 5; i < 10; i++) {
         *fldX = static_cast<float>(i);
         *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
         ntuple->Fill();
      }
   }

   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, &memFile}});

   auto x = proc->GetView<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryIndex());

      EXPECT_EQ(static_cast<float>(idx), x(idx));
   }
   EXPECT_EQ(9, proc->GetCurrentEntryIndex());
}

TEST_F(RNTupleChainProcessorTest, PrintStructure)
{
   auto proc = RNTupleProcessor::CreateChain(
      {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}, {fNTupleName, fFileNames[2]}});

   std::ostringstream os;
   proc->PrintStructure(os);

   const std::string exp = "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_proces... |\n"
                           "+-----------------------------+\n"
                           "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_proces... |\n"
                           "+-----------------------------+\n"
                           "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_proces... |\n"
                           "+-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}
