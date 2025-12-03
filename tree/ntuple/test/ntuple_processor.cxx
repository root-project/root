#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

#include <TMemFile.h>

TEST(RNTupleProcessor, EmptyNTuple)
{
   FileRaii fileGuard("test_ntuple_processor_empty.root");
   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto proc = RNTupleProcessor::Create({"ntuple", fileGuard.GetPath()});

   int nEntries = 0;
   for (auto it = proc->begin(); it != proc->end(); it++) {
      nEntries++;
   }
   EXPECT_EQ(0, nEntries);
   EXPECT_EQ(nEntries, proc->GetNEntriesProcessed());
}

TEST(RNTupleProcessor, TMemFile)
{
   TMemFile memFile("test_ntuple_processor_tmemfile.root", "RECREATE");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Append(std::move(model), "ntuple", memFile);

      for (unsigned i = 0; i < 5; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   auto proc = RNTupleProcessor::Create({"ntuple", &memFile});

   auto x = proc->RequestField<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST(RNTupleProcessor, TDirectory)
{
   FileRaii fileGuard("test_ntuple_processor_tdirectoryfile.root");
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
   auto proc = RNTupleProcessor::Create({"a/b/ntuple", file.get()});
   auto x = proc->RequestField<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

class RNTupleProcessorTest : public testing::Test {
protected:
   const std::array<std::string, 4> fFileNames{"test_ntuple_processor1.root", "test_ntuple_processor2.root",
                                               "test_ntuple_processor3.root", "test_ntuple_processor4.root"};
   const std::array<std::string, 4> fNTupleNames{"ntuple", "ntuple_aux", "ntuple_aux", "ntuple_aux"};

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
      // Same as above, but entries in reverse order
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[2], fFileNames[2]);

         for (int i = 4; i >= 0; --i) {
            *fldI = i;
            *fldZ = i * 3.f;
            ntuple->Fill();
         }
      }
      // Same as above, but the second and fourth entry are missing
      {
         auto model = RNTupleModel::Create();
         auto fldI = model->MakeField<int>("i");
         auto fldZ = model->MakeField<float>("z");
         auto ntuple = RNTupleWriter::Recreate(std::move(model), fNTupleNames[3], fFileNames[3]);

         for (unsigned i = 0; i < 5; ++i) {
            if (i % 2 == 1)
               continue;
            *fldI = i;
            *fldZ = i * 4.f;
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

TEST_F(RNTupleProcessorTest, Base)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto x = proc->RequestField<float>("x");
   // Check that `RequestField` also works with `void`.
   auto y = proc->RequestField<void>("y");

   try {
      proc->RequestField<float>("z");
      FAIL() << "registering fields that do not exist should not be possible";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(),
                  testing::HasSubstr("cannot register field with name \"z\" because it is not present in the on-disk "
                                     "information of the RNTuple(s) this processor is created from"));
   }

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);

      std::vector<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *std::static_pointer_cast<std::vector<float>>(y.GetPtr()));
   }
   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, RequestFieldWithPtr)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto xPtr = std::make_shared<float>();
   auto x = proc->RequestField<float>("x", xPtr.get());

   auto xNewPtr = std::make_shared<float>();

   for (auto idx : *proc) {
      EXPECT_FLOAT_EQ(static_cast<float>(idx), *x);
      EXPECT_EQ(x.GetRawPtr(), xPtr.get());

      if (idx == 2) {
         x.BindRawPtr(xNewPtr.get());
         xPtr.swap(xNewPtr);
      }
   }
}

TEST_F(RNTupleProcessorTest, RequestFieldWithVoidPtr)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto xPtr = std::make_shared<float>();
   auto x = proc->RequestField<void>("x", xPtr.get());

   auto xNewPtr = std::make_shared<float>();

   for (auto idx : *proc) {
      EXPECT_FLOAT_EQ(static_cast<float>(idx), *std::static_pointer_cast<float>(x.GetPtr()));
      EXPECT_EQ(x.GetRawPtr(), xPtr.get());

      if (idx == 2) {
         x.BindRawPtr(xNewPtr.get());
         xPtr.swap(xNewPtr);
      }
   }
}

TEST_F(RNTupleProcessorTest, PrintStructureSingle)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   std::ostringstream os;
   proc->PrintStructure(os);

   const std::string exp = "+-----------------------------+\n"
                           "| ntuple                      |\n"
                           "| test_ntuple_processor1.root |\n"
                           "+-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleProcessorTest, ChainedChain)
{
   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}}));
   innerProcs.push_back(RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);
      EXPECT_EQ(static_cast<float>(*i), *x);
   }
   EXPECT_EQ(15, proc->GetNEntriesProcessed());

   auto xPtr = std::make_shared<float>();
   x.BindRawPtr(xPtr.get());

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1 + 15, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);
      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(x.GetPtr().get(), xPtr.get());
   }
   EXPECT_EQ(30, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, ChainedJoin)
{
   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, ChainedJoinUnaligned)
{
   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[2], fFileNames[2]}, {"i"}));
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[2], fFileNames[2]}, {"i"}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, ChainedJoinMissingEntries)
{
   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i"}));
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[3], fFileNames[3]}, {"i"}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);

      if ((idx % 5) % 2 == 1) {
         EXPECT_FALSE(z.HasValue());
      } else {
         EXPECT_TRUE(z.HasValue());
         EXPECT_EQ(*x * 4, *z);
      }
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedChain)
{
   auto primaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedChainUnaligned)
{
   auto primaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[2], fFileNames[2]}, {fNTupleNames[2], fFileNames[2]}});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {"i"});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedChainMissingEntries)
{
   auto primaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});

   auto auxiliaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[3], fFileNames[3]}, {fNTupleNames[3], fFileNames[3]}});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {"i"});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z = proc->RequestField<float>("ntuple_aux.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);

      if ((idx % 5) % 2 == 1) {
         EXPECT_FALSE(z.HasValue());
      } else {
         EXPECT_TRUE(z.HasValue());
         EXPECT_EQ(*x * 4, *z);
      }
   }
   EXPECT_EQ(10, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedPrimary)
{
   auto primaryProc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleProcessor::Create({fNTupleNames[2], fFileNames[2]}, "ntuple_aux2");

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"}, "joined_ntuple");

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z1 = proc->RequestField<float>("ntuple_aux.z");
   auto z2 = proc->RequestField<float>("ntuple_aux2.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 3, *z2);
   }
   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedPrimaryMissingEntries)
{
   auto primaryProc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleProcessor::Create({fNTupleNames[3], fFileNames[3]}, "ntuple_aux2");

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z1 = proc->RequestField<float>("ntuple_aux.z");
   auto z2 = proc->RequestField<float>("ntuple_aux2.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);

      if (idx % 2 == 1) {
         EXPECT_FALSE(z2.HasValue());
      } else {
         EXPECT_TRUE(z2.HasValue());
         EXPECT_EQ(*x * 4, *z2);
      }
   }
   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedAuxiliary)
{
   auto primaryProc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto auxProcIntermediate = RNTupleProcessor::Create({fNTupleNames[2], fFileNames[2]}, "ntuple_aux2");

   auto auxProc = RNTupleProcessor::CreateJoin(RNTupleProcessor::Create({fNTupleNames[1], fFileNames[1]}),
                                               std::move(auxProcIntermediate), {"i"});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z1 = proc->RequestField<float>("ntuple_aux.z");
   auto z2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 3, *z2);
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedAuxiliaryMissingEntries)
{
   auto primaryProc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto auxProcIntermediate = RNTupleProcessor::Create({fNTupleNames[3], fFileNames[3]}, "ntuple_aux2");

   auto auxProc = RNTupleProcessor::CreateJoin(RNTupleProcessor::Create({fNTupleNames[1], fFileNames[1]}),
                                               std::move(auxProcIntermediate), {"i"});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {});

   auto i = proc->RequestField<int>("i");
   auto x = proc->RequestField<float>("x");
   auto z1 = proc->RequestField<float>("ntuple_aux.z");
   auto z2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.z");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);

      if (idx % 2 == 1) {
         EXPECT_FALSE(z2.HasValue());
      } else {
         EXPECT_TRUE(z2.HasValue());
         EXPECT_EQ(*x * 4, *z2);
      }
   }

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedSameName)
{
   auto primaryProc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   try {
      auto auxProc = RNTupleProcessor::Create({fNTupleNames[2], fFileNames[2]});
      auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"});
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr("a field or nested auxiliary processor named \"ntuple_aux\" is already present as a field "
                            "in the primary processor; rename the auxiliary processor to avoid conflicts"));
   }
}

TEST_F(RNTupleProcessorTest, PrintStructureChainedJoin)
{
   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcs;
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));
   innerProcs.push_back(
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   std::ostringstream os;
   proc->PrintStructure(os);

   const std::string exp = "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                           "+-----------------------------+ +-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleProcessorTest, PrintStructureJoinedChain)
{
   auto primaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});
   auto auxiliaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os;
   proc->PrintStructure(os);

   const std::string exp = "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "+-----------------------------+ +-----------------------------+\n"
                           "| ntuple                      | | ntuple_aux                  |\n"
                           "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                           "+-----------------------------+ +-----------------------------+\n";
   EXPECT_EQ(exp, os.str());
}

TEST_F(RNTupleProcessorTest, PrintStructureJoinedChainAsymmetric)
{
   auto primaryChain =
      RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}, {fNTupleNames[0], fFileNames[0]}});
   auto auxiliaryChain = RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}});

   auto proc1 = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os1;
   proc1->PrintStructure(os1);

   const std::string exp1 = "+-----------------------------+ +-----------------------------+\n"
                            "| ntuple                      | | ntuple_aux                  |\n"
                            "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                            "+-----------------------------+ +-----------------------------+\n"
                            "+-----------------------------+\n"
                            "| ntuple                      |\n"
                            "| test_ntuple_processor1.root |\n"
                            "+-----------------------------+\n";
   EXPECT_EQ(exp1, os1.str());

   primaryChain = RNTupleProcessor::CreateChain({{fNTupleNames[0], fFileNames[0]}});
   auxiliaryChain = RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[1], fFileNames[1]}});

   auto proc2 = RNTupleProcessor::CreateJoin(std::move(primaryChain), std::move(auxiliaryChain), {});

   std::ostringstream os2;
   proc2->PrintStructure(os2);

   const std::string exp2 = "+-----------------------------+ +-----------------------------+\n"
                            "| ntuple                      | | ntuple_aux                  |\n"
                            "| test_ntuple_processor1.root | | test_ntuple_processor2.root |\n"
                            "+-----------------------------+ +-----------------------------+\n"
                            "                                +-----------------------------+\n"
                            "                                | ntuple_aux                  |\n"
                            "                                | test_ntuple_processor2.root |\n"
                            "                                +-----------------------------+\n";
   EXPECT_EQ(exp2, os2.str());
}
