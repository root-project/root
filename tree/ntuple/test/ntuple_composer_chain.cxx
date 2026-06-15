#include "ntuple_test.hxx"

#include <TMemFile.h>

class RNTupleChainComposerTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_chain_composer1.root", "test_ntuple_chain_composer2.root",
                                               "test_ntuple_chain_composer3.root", "test_ntuple_chain_composer4.root"};

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

TEST(RNTupleChainComposer, EmptySpec)
{
   try {
      auto composer = RNTupleComposer::CreateChain(std::vector<RNTupleOpenSpec>{});
      FAIL() << "creating a composer without at least one RNTuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }
}

TEST_F(RNTupleChainComposerTest, SingleNTuple)
{
   auto composer = RNTupleComposer::CreateChain({{fNTupleName, fFileNames[0]}});

   auto x = composer->RequestField<float>("x");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(composer->GetCurrentEntryNumber()), *x);
   }
   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleChainComposerTest, Basic)
{
   auto composer = RNTupleComposer::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});

   EXPECT_STREQ("ntuple", composer->GetProcessorName().c_str());

   {
      auto namedProc =
         RNTupleComposer::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}}, "my_ntuple");
      EXPECT_STREQ("my_ntuple", namedProc->GetProcessorName().c_str());
   }

   auto x = composer->RequestField<float>("x");
   auto y = composer->RequestField<std::vector<float>>("y");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_EQ(static_cast<float>(idx), *x);

      std::vector<float> yExp = {static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *y);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleChainComposerTest, MissingFields)
{
   auto composer = RNTupleComposer::CreateChain(
      {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[2]}, {fNTupleName, fFileNames[1]}});

   auto x = composer->RequestField<float>("x");
   auto y = composer->RequestField<std::vector<float>>("y");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx % 5, static_cast<int>(*x) % 5);

      if (idx < 5 || idx >= 10) {
         EXPECT_TRUE(y.HasValue());
      } else {
         EXPECT_FALSE(y.HasValue());
      }
   }
   EXPECT_EQ(15, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleChainComposerTest, EmptyNTuples)
{
   FileRaii fileGuard("test_ntuple_composer_empty_ntuples.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<std::vector<float>>("y");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleName, fileGuard.GetPath());
   }

   std::vector<RNTupleOpenSpec> ntuples = {{fNTupleName, fileGuard.GetPath()}, {fNTupleName, fFileNames[0]}};

   // Empty ntuples are skipped (as long as their model complies)
   auto composer = RNTupleComposer::CreateChain({{fNTupleName, fileGuard.GetPath()},
                                                 {fNTupleName, fFileNames[0]},
                                                 {fNTupleName, fileGuard.GetPath()},
                                                 {fNTupleName, fFileNames[1]}});

   auto x = composer->RequestField<float>("x");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(static_cast<float>(idx), *x);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

namespace ROOT::Experimental::Internal {
struct RNTupleComposerEntryLoader {
   static ROOT::NTupleSize_t LoadEntry(RNTupleComposer &composer, ROOT::NTupleSize_t entryNumber)
   {
      composer.Connect(composer.fEntry->GetFieldIndices(), RNTupleProcessorProvenance(), /*updateFields=*/false);
      return composer.LoadEntry(entryNumber);
   }

   static void LoadUnfrozenEntry(RNTupleComposer &composer, ROOT::NTupleSize_t entryNumber)
   {
      composer.LoadEntry(entryNumber);
   }
};
} // namespace ROOT::Experimental::Internal

TEST_F(RNTupleChainComposerTest, LoadRandomEntry)
{
   using ROOT::Experimental::Internal::RNTupleComposerEntryLoader;
   auto composer = RNTupleComposer::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}});

   auto x = composer->RequestField<float>("x");

   RNTupleComposerEntryLoader::LoadEntry(*composer, 3);
   EXPECT_EQ(3.f, *x);
   EXPECT_EQ(0, composer->GetCurrentProcessorNumber());

   RNTupleComposerEntryLoader::LoadEntry(*composer, 9);
   EXPECT_EQ(9.f, *x);
   EXPECT_EQ(1, composer->GetCurrentProcessorNumber());

   RNTupleComposerEntryLoader::LoadEntry(*composer, 6);
   EXPECT_EQ(6.f, *x);
   EXPECT_EQ(1, composer->GetCurrentProcessorNumber());

   RNTupleComposerEntryLoader::LoadEntry(*composer, 2);
   EXPECT_EQ(2.f, *x);
   EXPECT_EQ(0, composer->GetCurrentProcessorNumber());

   EXPECT_EQ(ROOT::kInvalidNTupleIndex, RNTupleComposerEntryLoader::LoadEntry(*composer, 10));
}

TEST_F(RNTupleChainComposerTest, TMemFile)
{
   TMemFile memFile("test_ntuple_composer_chain_tmemfile_second.root", "RECREATE");
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

   auto composer = RNTupleComposer::CreateChain({{fNTupleName, fFileNames[0]}, {fNTupleName, &memFile}});

   auto x = composer->RequestField<float>("x");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_EQ(static_cast<float>(idx), *x);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleChainComposerTest, PrintStructure)
{
   auto composer = RNTupleComposer::CreateChain(
      {{fNTupleName, fFileNames[0]}, {fNTupleName, fFileNames[1]}, {fNTupleName, fFileNames[2]}});

   std::ostringstream os;
   composer->PrintStructure(os);

   const std::string exp = "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_compos... |\n"
                           "+-----------------------------+\n"
                           "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_compos... |\n"
                           "+-----------------------------+\n"
                           "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_chain_compos... |\n"
                           "+-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}
