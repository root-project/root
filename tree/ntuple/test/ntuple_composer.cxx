#include "ntuple_test.hxx"

#include <TMemFile.h>

using ROOT::Experimental::RNTupleComposer;

TEST(RNTupleComposer, EmptyNTuple)
{
   FileRaii fileGuard("test_ntuple_composer_empty .root");
   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto composer = RNTupleComposer::Create({"ntuple", fileGuard.GetPath()});
   auto processor = RNTupleProcessor(*composer);

   int nEntries = 0;
   for (auto idx [[maybe_unused]] : processor) {
      nEntries++;
   }
   EXPECT_EQ(0, nEntries);
   EXPECT_EQ(nEntries, processor.GetNEntriesProcessed());
}

TEST(RNTupleComposer, TMemFile)
{
   TMemFile memFile("test_ntuple_composer_tmemfile.root", "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Append(std::move(model), "ntuple", memFile);

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   auto composer = RNTupleComposer::Create({"ntuple", &memFile});

   auto x = composer->RequestField<float>("x");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
   }

   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST(RNTupleComposer, TDirectory)
{
   FileRaii fileGuard("test_ntuple_composer_tdirectoryfile.root");
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto dir = std::unique_ptr<TDirectory>(file->mkdir("a/b"));
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Append(std::move(model), "ntuple", *dir);

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   auto file = std::make_unique<TFile>(fileGuard.GetPath().c_str());
   auto composer = RNTupleComposer::Create({"a/b/ntuple", file.get()});
   auto x = composer->RequestField<float>("x");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
   }

   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

class RNTupleComposerTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_composer1.root ", "test_ntuple_composer2.root ",
                                               "test_ntuple_composer3.root ", "test_ntuple_composer4.root "};
   const std::array<std::string, 4> fNTupleNames{"ntuple", "ntuple_aux", "ntuple_aux", "ntuple_aux"};

   void SetUp() override
   {
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldX = model->MakeField<float>("x");
         auto fldY = model->MakeField<std::vector<float>>("y");
         auto fldStruct = model->MakeField<CustomStruct>("struct");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[0], fFileNames[0]);

         for (unsigned i = 0; i < 5; i++) {
            *fldI = i;
            *fldX = static_cast<float>(i);
            *fldY = {static_cast<float>(i), static_cast<float>(i * 2)};
            fldStruct->a = i * 1.f;
            ntuple->Fill();
         }
      }
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto fldStruct = model->MakeField<CustomStruct>("struct");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[1], fFileNames[1]);

         for (unsigned i = 0; i < 5; ++i) {
            *fldI = i;
            *fldZ = i * 2.f;
            fldStruct->a = i * 2.f;
            ntuple->Fill();
         }
      }
      // Same as above, but entries in reverse order
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto fldStruct = model->MakeField<CustomStruct>("struct");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[2], fFileNames[2]);

         for (int i = 4; i >= 0; --i) {
            *fldI = i;
            *fldZ = i * 3.f;
            fldStruct->a = i * 3.f;

            ntuple->Fill();
         }
      }
      // Same as above, but the second and fourth entry are missing
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto fldStruct = model->MakeField<CustomStruct>("struct");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[3], fFileNames[3]);

         for (unsigned i = 0; i < 5; ++i) {
            if (i % 2 == 1)
               continue;
            *fldI = i;
            *fldZ = i * 4.f;
            fldStruct->a = i * 4.f;
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

TEST_F(RNTupleComposerTest, Base)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto x = composer->RequestField<float>("x");
   // Check that `RequestField` also works with `void`.
   auto y = composer->RequestField<void>("y");

   try {
      composer->RequestField<float>("z");
      FAIL() << "registering fields that do not exist should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(),
                  testing::HasSubstr("cannot register field with name \"z\" because it is not present in the on-disk "
                                     "information of the RNTuple(s) this composition is created from"));
   }

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);

      std::vector<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *std::static_pointer_cast<std::vector<float>>(y.GetPtr()));
   }
   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, RequestFieldWithPtr)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto xPtr = std::make_shared<float>();
   auto x = composer->RequestField<float>("x", xPtr.get());

   auto xNewPtr = std::make_shared<float>();

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
      EXPECT_EQ(x.GetRawPtr(), xPtr.get());

      if (idx == 2) {
         x.BindRawPtr(xNewPtr.get());
         xPtr.swap(xNewPtr);
      }
   }
}

TEST_F(RNTupleComposerTest, RequestFieldWithVoidPtr)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto xPtr = std::make_shared<float>();
   auto x = composer->RequestField<void>("x", xPtr.get());

   auto xNewPtr = std::make_shared<float>();

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_FLOAT_EQ(static_cast<float>(idx), *std::static_pointer_cast<float>(x.GetPtr()));
      EXPECT_EQ(x.GetRawPtr(), xPtr.get());

      if (idx == 2) {
         x.BindRawPtr(xNewPtr.get());
         xPtr.swap(xNewPtr);
      }
   }
}

TEST_F(RNTupleComposerTest, RequestFieldWithTypeString)
{
   {
      auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_NO_THROW(composer->RequestField("y", "std::vector<float    >"));
   }
   {
      auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_NO_THROW(composer->RequestField("y", "std::vector<Float_t>"));
   }
   {
      auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_THROW(composer->RequestField("y", "std::vetor<float>"), ROOT::RException);
   }

   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});
   auto x = composer->RequestField("x", "float");
   auto yPtr = std::make_shared<std::vector<float>>();
   auto y = composer->RequestField("y", "std::vector<float>", yPtr.get());

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *std::static_pointer_cast<float>(x.GetPtr()));

      std::vector<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *std::static_pointer_cast<std::vector<float>>(y.GetPtr()));
   }
   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, AlternativeTypes)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto xAsDouble = composer->RequestField<double>("x");
   auto xAsFloat = composer->RequestField<float>("x");

   try {
      composer->RequestField<std::string>("x");
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("in-memory field x of type std::string is incompatible with "
                                                 "on-disk field x: incompatible on-disk type name float"));
   }

   auto yAsRVec = composer->RequestField<ROOT::RVec<float>>("y");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<double>(idx), *xAsDouble);
      EXPECT_FLOAT_EQ(idx, *xAsFloat);

      ROOT::RVec<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      for (std::size_t i = 0ul; i < yAsRVec->size(); ++i) {
         EXPECT_FLOAT_EQ(yExp[i], (*yAsRVec)[i]);
      }
   }
}

TEST_F(RNTupleComposerTest, Subfields)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto strct = composer->RequestField<CustomStruct>("struct");
   auto strct_a = composer->RequestField<float>("struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_FLOAT_EQ(idx, idx);
      EXPECT_FLOAT_EQ(strct->a, *strct_a);
   }
}

TEST_F(RNTupleComposerTest, PrintStructureSingle)
{
   auto composer = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   std::ostringstream os;
   composer->PrintStructure(os);

   const std::string exp = "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_composer1.root  |\n"
                           "+-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleComposerTest, ChainedChain)
{
   std::vector<std::unique_ptr<RNTupleComposer>> innerProcs;
   innerProcs.push_back(
      RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}}));
   innerProcs.push_back(
      RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}}));

   auto composer = RNTupleComposer::CreateChain(std::move(innerProcs));

   auto i = composer->RequestField<int>("i");
   auto z = composer->RequestField<float>("z");
   auto strct_a = composer->RequestField<float>("struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      if ((idx >= 5 && idx < 10) || idx >= 15) {
         EXPECT_EQ(*i, 4 - idx % 5);
         EXPECT_EQ(*z, (4 - idx % 5) * 3.f);
      } else {
         EXPECT_EQ(*i, idx % 5);
         EXPECT_EQ(*z, (idx % 5) * 2.f);
      }

      EXPECT_EQ(*strct_a, *z);
   }
   EXPECT_EQ(20, processor.GetNEntriesProcessed());

   auto zPtr = std::make_shared<float>();
   z.BindRawPtr(zPtr.get());
   auto aPtr = std::make_shared<float>();
   strct_a.BindRawPtr(aPtr.get());

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1 + 20, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());

      if ((idx >= 5 && idx < 10) || idx >= 15) {
         EXPECT_EQ(*i, 4 - idx % 5);
         EXPECT_EQ(*z, (4 - idx % 5) * 3.f);
      } else {
         EXPECT_EQ(*i, idx % 5);
         EXPECT_EQ(*z, (idx % 5) * 2.f);
      }

      EXPECT_EQ(*strct_a, *z);
      EXPECT_EQ(z.GetPtr().get(), zPtr.get());
      EXPECT_EQ(strct_a.GetPtr().get(), aPtr.get());
   }
   EXPECT_EQ(40, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, ChainedJoin)
{
   std::vector<std::unique_ptr<RNTupleComposer>> innerProcs;
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));

   auto composer = RNTupleComposer::CreateChain(std::move(innerProcs));

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
      EXPECT_EQ(*z, *strct_a);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, ChainedJoinUnaligned)
{
   std::vector<std::unique_ptr<RNTupleComposer>> innerProcs;
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[2], fFileNames[2]}, {"i"}));
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[2], fFileNames[2]}, {"i"}));

   auto composer = RNTupleComposer::CreateChain(std::move(innerProcs));

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
      EXPECT_EQ(*z, *strct_a);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, ChainedJoinMissingEntries)
{
   std::vector<std::unique_ptr<RNTupleComposer>> innerProcs;
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i"}));
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i"}));

   auto composer = RNTupleComposer::CreateChain(std::move(innerProcs));

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);

      if ((idx % 5) % 2 == 1) {
         EXPECT_FALSE(z.HasValue());
         EXPECT_FALSE(strct_a.HasValue());
      } else {
         EXPECT_TRUE(z.HasValue());
         EXPECT_TRUE(strct_a.HasValue());
         EXPECT_EQ(*x * 4, *z);
         EXPECT_EQ(*z, *strct_a);
      }
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedChain)
{
   auto primaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
      EXPECT_EQ(*z, *strct_a);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedChainUnaligned)
{
   auto primaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[2], fFileNames[2]}, {fNTupleNames[2], fFileNames[2]}});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {"i"});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
      EXPECT_EQ(*z, *strct_a);
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedChainMissingEntries)
{
   auto primaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[3], fFileNames[3]}, {fNTupleNames[3], fFileNames[3]}});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {"i"});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a = composer->RequestField<float>("ntuple_aux.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);

      if ((idx % 5) % 2 == 1) {
         EXPECT_FALSE(z.HasValue());
         EXPECT_FALSE(strct_a.HasValue());
      } else {
         EXPECT_TRUE(z.HasValue());
         EXPECT_TRUE(strct_a.HasValue());
         EXPECT_EQ(*x * 4, *z);
         EXPECT_EQ(*z, *strct_a);
      }
   }
   EXPECT_EQ(10, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedJoinComposedPrimary)
{
   auto primaryProc =
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleComposer::Create({fNTupleNames[2], fFileNames[2]}, "ntuple_aux2");

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"}, "joined_ntuple");

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z1 = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a1 = composer->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = composer->RequestField<float>("ntuple_aux2.z");
   auto strct_a2 = composer->RequestField<float>("ntuple_aux2.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);
      EXPECT_EQ(*x * 3, *z2);
      EXPECT_EQ(*x * 3, *strct_a2);
   }
   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedJoinComposedPrimaryMissingEntries)
{
   auto primaryProc =
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleComposer::Create({fNTupleNames[3], fFileNames[3]}, "ntuple_aux2");

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z1 = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a1 = composer->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = composer->RequestField<float>("ntuple_aux2.z");
   auto strct_a2 = composer->RequestField<float>("ntuple_aux2.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);

      if (idx % 2 == 1) {
         EXPECT_FALSE(z2.HasValue());
         EXPECT_FALSE(strct_a2.HasValue());
      } else {
         EXPECT_TRUE(z2.HasValue());
         EXPECT_TRUE(strct_a2.HasValue());
         EXPECT_EQ(*x * 4, *z2);
         EXPECT_EQ(*x * 4, *strct_a2);
      }
   }
   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedJoinComposedAuxiliary)
{
   auto primaryProc = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto auxProcIntermediate = RNTupleComposer::Create({fNTupleNames[2], fFileNames[2]}, "ntuple_aux2");

   auto auxProc = RNTupleComposer::CreateJoin(RNTupleComposer::Create({fNTupleNames[1], fFileNames[1]}),
                                              std::move(auxProcIntermediate), {"i"});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryProc), std::move(auxProc), {});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z1 = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a1 = composer->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = composer->RequestField<float>("ntuple_aux.ntuple_aux2.z");
   auto strct_a2 = composer->RequestField<float>("ntuple_aux.ntuple_aux2.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);
      EXPECT_EQ(*x * 3, *z2);
      EXPECT_EQ(*x * 3, *strct_a2);
   }

   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedJoinComposedAuxiliaryMissingEntries)
{
   auto primaryProc = RNTupleComposer::Create({fNTupleNames[0], fFileNames[0]});

   auto auxProcIntermediate = RNTupleComposer::Create({fNTupleNames[3], fFileNames[3]}, "ntuple_aux2");

   auto auxProc = RNTupleComposer::CreateJoin(RNTupleComposer::Create({fNTupleNames[1], fFileNames[1]}),
                                              std::move(auxProcIntermediate), {"i"});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryProc), std::move(auxProc), {});

   auto i = composer->RequestField<int>("i");
   auto x = composer->RequestField<float>("x");
   auto z1 = composer->RequestField<float>("ntuple_aux.z");
   auto strct_a1 = composer->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = composer->RequestField<float>("ntuple_aux.ntuple_aux2.z");
   auto strct_a2 = composer->RequestField<float>("ntuple_aux.ntuple_aux2.struct.a");

   auto processor = RNTupleProcessor(*composer);

   for (auto idx : processor) {
      EXPECT_EQ(idx + 1, processor.GetNEntriesProcessed());
      EXPECT_EQ(idx, composer->GetCurrentEntryNumber());
      EXPECT_EQ(*i, composer->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);

      if (idx % 2 == 1) {
         EXPECT_FALSE(z2.HasValue());
         EXPECT_FALSE(strct_a2.HasValue());
      } else {
         EXPECT_TRUE(z2.HasValue());
         EXPECT_TRUE(strct_a2.HasValue());
         EXPECT_EQ(*x * 4, *z2);
         EXPECT_EQ(*x * 4, *strct_a2);
      }
   }

   EXPECT_EQ(5, processor.GetNEntriesProcessed());
}

TEST_F(RNTupleComposerTest, JoinedJoinComposedSameName)
{
   auto primaryProc =
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleComposer::Create({fNTupleNames[2], fFileNames[2]});
   auto composer = RNTupleComposer::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"});

   try {
      composer->RequestField<float>("ntuple_aux.z");

      FAIL() << "creating an auxiliary composer where its name causes conflicts should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(),
                  testing::HasSubstr("ambiguous field name: \"ntuple_aux.z\" is present in the primary RNTupleComposer "
                                     "\"ntuple\", but may also refer to a field in the auxiliary RNTupleComposer named "
                                     "\"ntuple_aux\". To avoid this ambiguity, rename the auxiliary RNTupleComposer."));
   }
}

TEST_F(RNTupleComposerTest, PrintStructureChainedJoin)
{
   std::vector<std::unique_ptr<RNTupleComposer>> innerProcs;
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));
   innerProcs.push_back(
      RNTupleComposer::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));

   auto composer = RNTupleComposer::CreateChain(std::move(innerProcs));

   std::ostringstream os;
   composer->PrintStructure(os);

   const std::string exp = "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                           "+-----------------------------+ +-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleComposerTest, PrintStructureJoinedChain)
{
   auto primaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});
   auto auxiliaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto composer = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os;
   composer->PrintStructure(os);

   const std::string exp = "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                           "+-----------------------------+ +-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleComposerTest, PrintStructureJoinedChainAsymmetric)
{
   auto primaryChain =
      RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});
   auto auxiliaryChain = RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}});

   auto proc1 = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os1;
   proc1->PrintStructure(os1);

   const std::string exp1 = "+-----------------------------+ +-----------------------------+\n"
                            "| ntuple                      | | ntuple_aux                  |\n"
                            "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                            "+-----------------------------+ +-----------------------------+\n"
                            "+-----------------------------+\n"
                            "| ntuple                      |\n"
                            "| test_ntuple_composer1.root  |\n"
                            "+-----------------------------+\n";
   EXPECT_EQ(exp1, os1.str());

   primaryChain = RNTupleComposer::CreateChain({{fNTupleNames[0], fFileNames[0]}});
   auxiliaryChain = RNTupleComposer::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto proc2 = RNTupleComposer::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os2;
   proc2->PrintStructure(os2);

   const std::string exp2 = "+-----------------------------+ +-----------------------------+\n"
                            "| ntuple                      | | ntuple_aux                  |\n"
                            "| test_ntuple_composer1.root  | | test_ntuple_composer2.root  |\n"
                            "+-----------------------------+ +-----------------------------+\n"
                            "                                +-----------------------------+\n"
                            "                                | ntuple_aux                  |\n"
                            "                                | test_ntuple_composer2.root  |\n"
                            "                                +-----------------------------+\n";
   EXPECT_EQ(exp2, os2.str());
}
