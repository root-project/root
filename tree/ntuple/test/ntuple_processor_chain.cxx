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

   auto x = proc->GetValuePtr<float>("x");

   for (const auto &idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(proc->GetCurrentEntryNumber()), *x);
   }
   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleChainProcessorTest, Basic)
{
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});

   EXPECT_STREQ("ntuple", proc->GetProcessorName().c_str());

   {
      auto namedProc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}},
                                                     nullptr, "my_ntuple");
      EXPECT_STREQ("my_ntuple", namedProc->GetProcessorName().c_str());
   }

   auto x = proc->GetValuePtr<float>("x");
   auto y = proc->GetValuePtr<std::vector<float>>("y");

   for (const auto &idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_EQ(static_cast<float>(idx), *x);

      std::vector<float> yExp = {static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *y);
   }
   EXPECT_EQ(8, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleChainProcessorTest, WithModel)
{
   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<float>("x");

   auto proc =
      RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}}, std::move(model));

   auto x = proc->GetValuePtr<float>("x");

   try {
      proc->GetValuePtr<std::vector<float>>("y");
      FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: y"));
   }

   for (const auto &idx : *proc) {
      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
      EXPECT_EQ(fldX, x.GetPtr());
   }
}

TEST_F(RNTupleChainProcessorTest, WithBareModel)
{
   auto model = RNTupleModel::CreateBare();
   auto fldY = model->MakeField<std::vector<float>>("y");

   auto proc =
      RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}}, std::move(model));

   auto y = proc->GetValuePtr<std::vector<float>>("y");

   try {
      proc->GetValuePtr<float>("x");
      FAIL() << "fields not present in the model passed to the processor shouldn't be readable";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: x"));
   }

   for (const auto &idx : *proc) {
      std::vector<float> yExp = {static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *y);
   }
}

TEST_F(RNTupleChainProcessorTest, MissingFields)
{
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[2]}});
   auto iter = proc->begin();

   auto x = proc->GetValuePtr<float>("x");

   while (proc->GetNEntriesProcessed() < 5) {
      EXPECT_EQ(static_cast<float>(proc->GetNEntriesProcessed() - 1), *x);
      iter++;
   }

   try {
      iter++;
      FAIL() << "having missing fields in subsequent ntuples should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field \"y\" not found in the current RNTuple"));
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

   // Empty ntuples are skipped (as long as their model complies)
   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fileGuard.GetPath()},
                                              {fNTupleName, fFileNames[0]},
                                              {fNTupleName, fileGuard.GetPath()},
                                              {fNTupleName, fFileNames[1]}});

   auto x = proc->GetValuePtr<float>("x");

   for (const auto &idx : *proc) {
      EXPECT_EQ(static_cast<float>(idx), *x);
   }
   EXPECT_EQ(8, proc->GetNEntriesProcessed());
}

namespace ROOT::Experimental::Internal {
struct RNTupleProcessorEntryLoader {
   static ROOT::NTupleSize_t LoadEntry(RNTupleProcessor &processor, ROOT::NTupleSize_t entryNumber)
   {
      return processor.LoadEntry(entryNumber);
   }
};
} // namespace ROOT::Experimental::Internal

TEST_F(RNTupleChainProcessorTest, LoadRandomEntry)
{
   using ROOT::Experimental::Internal::RNTupleProcessorEntryLoader;

   auto proc = RNTupleProcessor::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});

   auto x = proc->GetValuePtr<float>("x");

   RNTupleProcessorEntryLoader::LoadEntry(*proc, 3);
   EXPECT_EQ(3.f, *x);
   EXPECT_EQ(0, proc->GetCurrentProcessorNumber());

   RNTupleProcessorEntryLoader::LoadEntry(*proc, 7);
   EXPECT_EQ(7.f, *x);
   EXPECT_EQ(1, proc->GetCurrentProcessorNumber());

   RNTupleProcessorEntryLoader::LoadEntry(*proc, 6);
   EXPECT_EQ(6.f, *x);
   EXPECT_EQ(1, proc->GetCurrentProcessorNumber());

   RNTupleProcessorEntryLoader::LoadEntry(*proc, 2);
   EXPECT_EQ(2.f, *x);
   EXPECT_EQ(0, proc->GetCurrentProcessorNumber());

   EXPECT_EQ(ROOT::kInvalidNTupleIndex, RNTupleProcessorEntryLoader::LoadEntry(*proc, 8));
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

   auto x = proc->GetValuePtr<float>("x");

   for (const auto &idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_EQ(static_cast<float>(idx), *x);
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}
