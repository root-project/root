#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

#include <TMemFile.h>

#include <array>

#include <cstdio>

#include <tuple>

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

TEST_F(RNTupleProcessorTest, RequestFieldWithTypeString)
{
   {
      auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_NO_THROW(proc->RequestField("y", "std::vector<float    >"));
   }
   {
      auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_NO_THROW(proc->RequestField("y", "std::vector<Float_t>"));
   }
   {
      auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});
      EXPECT_THROW(proc->RequestField("y", "std::vetor<float>"), ROOT::RException);
   }

   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});
   auto x = proc->RequestField("x", "float");
   auto yPtr = std::make_shared<std::vector<float>>();
   auto y = proc->RequestField("y", "std::vector<float>", yPtr.get());

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());

      EXPECT_FLOAT_EQ(static_cast<float>(idx), *std::static_pointer_cast<float>(x.GetPtr()));

      std::vector<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      EXPECT_EQ(yExp, *std::static_pointer_cast<std::vector<float>>(y.GetPtr()));
   }
   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, AlternativeTypes)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto xAsDouble = proc->RequestField<double>("x");
   auto xAsFloat = proc->RequestField<float>("x");

   try {
      proc->RequestField<std::string>("x");
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("in-memory field x of type std::string is incompatible with "
                                                 "on-disk field x: incompatible on-disk type name float"));
   }

   auto yAsRVec = proc->RequestField<ROOT::RVec<float>>("y");

   for (auto idx : *proc) {
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

      EXPECT_FLOAT_EQ(static_cast<double>(idx), *xAsDouble);
      EXPECT_FLOAT_EQ(idx, *xAsFloat);

      ROOT::RVec<float> yExp{static_cast<float>(idx), static_cast<float>((idx) * 2)};
      for (std::size_t i = 0ul; i < yAsRVec->size(); ++i) {
         EXPECT_FLOAT_EQ(yExp[i], (*yAsRVec)[i]);
      }
   }
}

TEST_F(RNTupleProcessorTest, Subfields)
{
   auto proc = RNTupleProcessor::Create({fNTupleNames[0], fFileNames[0]});

   auto strct = proc->RequestField<CustomStruct>("struct");
   auto strct_a = proc->RequestField<float>("struct.a");

   for (auto idx : *proc) {
      EXPECT_FLOAT_EQ(idx, idx);
      EXPECT_FLOAT_EQ(strct->a, *strct_a);
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
      RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}}));
   innerProcs.push_back(
      RNTupleProcessor::CreateChain({{fNTupleNames[1], fFileNames[1]}, {fNTupleNames[2], fFileNames[2]}}));

   auto proc = RNTupleProcessor::CreateChain(std::move(innerProcs));

   auto i = proc->RequestField<int>("i");
   auto z = proc->RequestField<float>("z");
   auto strct_a = proc->RequestField<float>("struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      if ((idx >= 5 && idx < 10) || idx >= 15) {
         EXPECT_EQ(*i, 4 - idx % 5);
         EXPECT_EQ(*z, (4 - idx % 5) * 3.f);
      } else {
         EXPECT_EQ(*i, idx % 5);
         EXPECT_EQ(*z, (idx % 5) * 2.f);
      }

      EXPECT_EQ(*strct_a, *z);
   }
   EXPECT_EQ(20, proc->GetNEntriesProcessed());

   auto zPtr = std::make_shared<float>();
   z.BindRawPtr(zPtr.get());
   auto aPtr = std::make_shared<float>();
   strct_a.BindRawPtr(aPtr.get());

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1 + 20, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());

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
   EXPECT_EQ(40, proc->GetNEntriesProcessed());
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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
      EXPECT_EQ(*z, *strct_a);
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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
      EXPECT_EQ(*z, *strct_a);
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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z);
      EXPECT_EQ(*z, *strct_a);
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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 3, *z);
      EXPECT_EQ(*z, *strct_a);
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
   auto strct_a = proc->RequestField<float>("ntuple_aux.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

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
   auto strct_a1 = proc->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = proc->RequestField<float>("ntuple_aux2.z");
   auto strct_a2 = proc->RequestField<float>("ntuple_aux2.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);
      EXPECT_EQ(*x * 3, *z2);
      EXPECT_EQ(*x * 3, *strct_a2);
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
   auto strct_a1 = proc->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = proc->RequestField<float>("ntuple_aux2.z");
   auto strct_a2 = proc->RequestField<float>("ntuple_aux2.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

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
   auto strct_a1 = proc->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.z");
   auto strct_a2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

      EXPECT_EQ(static_cast<float>(*i), *x);
      EXPECT_EQ(*x * 2, *z1);
      EXPECT_EQ(*x * 2, *strct_a1);
      EXPECT_EQ(*x * 3, *z2);
      EXPECT_EQ(*x * 3, *strct_a2);
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
   auto strct_a1 = proc->RequestField<float>("ntuple_aux.struct.a");
   auto z2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.z");
   auto strct_a2 = proc->RequestField<float>("ntuple_aux.ntuple_aux2.struct.a");

   for (auto idx : *proc) {
      EXPECT_EQ(idx + 1, proc->GetNEntriesProcessed());
      EXPECT_EQ(idx, proc->GetCurrentEntryNumber());
      EXPECT_EQ(*i, proc->GetCurrentEntryNumber() % 5);

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

   EXPECT_EQ(5, proc->GetNEntriesProcessed());
}

TEST_F(RNTupleProcessorTest, JoinedJoinComposedSameName)
{
   auto primaryProc =
      RNTupleProcessor::CreateJoin({fNTupleNames[0], fFileNames[0]}, {fNTupleNames[1], fFileNames[1]}, {});

   auto auxProc = RNTupleProcessor::Create({fNTupleNames[2], fFileNames[2]});
   auto proc = RNTupleProcessor::CreateJoin(std::move(primaryProc), std::move(auxProc), {"i"});

   try {
      proc->RequestField<float>("ntuple_aux.z");

      FAIL() << "creating an auxiliary processor where its name causes conflicts should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr(
                                 "ambiguous field name: \"ntuple_aux.z\" is present in the primary RNTupleProcessor "
                                 "\"ntuple\", but may also refer to a field in the auxiliary RNTupleProcessor named "
                                 "\"ntuple_aux\". To avoid this ambiguity, rename the auxiliary RNTupleProcessor."));
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

// This test is a translation using RNTupleProcessor of the test
// introduced by https://github.com/root-project/root/pull/19322,
// to ensure that the TTree friendship mechanism works equivalently
// with the RNTuple join mechanism.
class GH16805ProcessorTest : public testing::Test {
protected:
   const std::vector<std::string> fStepZeroFiles{
      "gh16805_rntuple_stepzero_0.root",
      "gh16805_rntuple_stepzero_1.root"
   };

   const std::vector<std::string> fJoinFiles{
      "gh16805_rntuple_join_0.root",
      "gh16805_rntuple_join_1.root",
      "gh16805_rntuple_join_2.root"
   };
   const std::string fStepOneFile = "gh16805_rntuple_stepone.root";

   void WriteStepZero(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("stepZeroBr1");
      auto br2 = model->MakeField<int>("stepZeroBr2");

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "stepzero", fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         *br2 = 2 * i;
         writer->Fill();
      }
   }

   void WriteStepOne(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("stepOneBr1");

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "stepone", fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         writer->Fill();
      }
   }

   void WriteJoin(const std::string &fileName, int begin, int end)
   {
      auto model = ROOT::RNTupleModel::Create();

      auto br1 = model->MakeField<int>("joinBr1");
      auto br2 = model->MakeField<int>("joinBr2");

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "topLevelJoin", fileName);

      for (int i = begin; i < end; ++i) {
         *br1 = i;
         *br2 = 2 * i;
         writer->Fill();
      }
   }

   void SetUp() override
   {
      WriteStepZero(fStepZeroFiles[0], 0, 10);
      WriteStepZero(fStepZeroFiles[1], 10, 20);

      WriteJoin(fJoinFiles[0], 200, 207);
      WriteJoin(fJoinFiles[1], 207, 214);
      WriteJoin(fJoinFiles[2], 214, 220);

      WriteStepOne(fStepOneFile, 100, 120);
   }

   void TearDown() override
   {
      for (const auto &f : fStepZeroFiles)
         std::remove(f.c_str());

      for (const auto &f : fJoinFiles)
         std::remove(f.c_str());

      std::remove(fStepOneFile.c_str());
   }
};

TEST_F(GH16805ProcessorTest, JoinReading)
{
   std::vector<RNTupleOpenSpec> stepOneSpecs{
      {"stepone", fStepOneFile}
   };

   std::vector<RNTupleOpenSpec> stepZeroSpecs{
      {"stepzero", fStepZeroFiles[0]},
      {"stepzero", fStepZeroFiles[1]}
   };

   std::vector<RNTupleOpenSpec> joinSpecs{
      {"topLevelJoin", fJoinFiles[0]},
      {"topLevelJoin", fJoinFiles[1]},
      {"topLevelJoin", fJoinFiles[2]}
   };

   auto stepOneProc =
      RNTupleProcessor::CreateChain(stepOneSpecs, "stepone");

   auto stepZeroProc =
      RNTupleProcessor::CreateChain(stepZeroSpecs, "stepzero");

   auto joinProc =
      RNTupleProcessor::CreateChain(joinSpecs, "topLevelJoin");

   auto joinedWithJoin =
      RNTupleProcessor::CreateJoin(
         std::move(stepOneProc),
         std::move(joinProc),
         {}
      );

   auto joinedAll =
      RNTupleProcessor::CreateJoin(
         std::move(joinedWithJoin),
         std::move(stepZeroProc),
         {}
      );

   auto stepOneBr1 = joinedAll->RequestField<int>("stepOneBr1");
   auto joinBr1 = joinedAll->RequestField<int>("topLevelJoin.joinBr1");
   auto joinBr2 = joinedAll->RequestField<int>("topLevelJoin.joinBr2");
   auto stepZeroBr1 = joinedAll->RequestField<int>("stepzero.stepZeroBr1");
   auto stepZeroBr2 = joinedAll->RequestField<int>("stepzero.stepZeroBr2");

   std::size_t i = 0;

   for (auto idx : *joinedAll) {
      EXPECT_EQ(i, idx);

      EXPECT_EQ(static_cast<int>(i), *stepZeroBr1);
      EXPECT_EQ(static_cast<int>(2 * i), *stepZeroBr2);
      EXPECT_EQ(static_cast<int>(100 + i), *stepOneBr1);
      EXPECT_EQ(static_cast<int>(200 + i), *joinBr1);
      EXPECT_EQ(static_cast<int>(2 * (200 + i)), *joinBr2);

      ++i;
   }

   EXPECT_EQ(20u, i);
   EXPECT_EQ(20u, joinedAll->GetNEntriesProcessed());
}

// This test is a translation using RNTupleProcessor of the TTree test
// introduced by https://github.com/root-project/root/pull/20222,
// to ensure that the corresponding friendship logic works equivalently
// with the RNTuple join mechanism.
using GH20033ProcessorConfig = std::tuple<bool, bool, bool, bool>;

class GH20033ProcessorTest : public testing::TestWithParam<GH20033ProcessorConfig> {
protected:

   const std::array<std::string, 2> fStepZeroFiles{
      "gh20033_rntuple_stepzero_0.root",
      "gh20033_rntuple_stepzero_1.root"
   };

   const std::string fStepOneFile = "gh20033_rntuple_stepone.root";
   const std::string fStepTwoFile = "gh20033_rntuple_steptwo.root";
   const std::string fStepThreeFile = "gh20033_rntuple_stepthree.root";
   const std::string fStepFourFile = "gh20033_rntuple_stepfour.root";

   static void WriteStepZero(const std::string &fileName, int begin, int end)
   {
      auto model = RNTupleModel::Create();

      auto stepZeroBr1 = model->MakeField<int>("stepZeroBr1");
      auto stepZeroBr2 = model->MakeField<int>("stepZeroBr2");
      auto value = model->MakeField<int>("value");

      auto writer = RNTupleWriter::Recreate(std::move(model), "stepzero", fileName);

      for (int i = begin; i < end; ++i) {
         *stepZeroBr1 = i;
         *stepZeroBr2 = 2 * i;
         *value = i;
         writer->Fill();
      }
   }

   static void
   WriteStepFile(const std::string &fileName, std::string_view ntupleName, std::string_view fieldName, int offset)
   {
      auto model = RNTupleModel::Create();

      auto field = model->MakeField<int>(std::string(fieldName));

      auto value = model->MakeField<int>("value");

      auto writer = RNTupleWriter::Recreate(std::move(model), ntupleName, fileName);

      for (int i = 0; i < 20; ++i) {
         *field = offset + i;
         *value = offset + i;
         writer->Fill();
      }
   }

   void SetUp() override
   {

      WriteStepZero(fStepZeroFiles[0], 0, 10);
      WriteStepZero(fStepZeroFiles[1], 10, 20);

      WriteStepFile(fStepOneFile, "stepone", "stepOneBr1", 100);
      WriteStepFile(fStepTwoFile, "steptwo", "stepTwoBr1", 200);
      WriteStepFile(fStepThreeFile, "stepthree", "stepThreeBr1", 300);
      WriteStepFile(fStepFourFile, "stepfour", "stepFourBr1", 400);
   }

   void TearDown() override
   {
      for (const auto &fileName : fStepZeroFiles)
         std::remove(fileName.c_str());

      std::remove(fStepOneFile.c_str());
      std::remove(fStepTwoFile.c_str());
      std::remove(fStepThreeFile.c_str());
      std::remove(fStepFourFile.c_str());
   }

   std::unique_ptr<RNTupleProcessor>
   CreateStepProcessor(std::string_view ntupleName, std::string_view fileName, bool useChain)
   {
      if (useChain) {
         std::vector<RNTupleOpenSpec> specs{{std::string(ntupleName), std::string(fileName)}};
         return RNTupleProcessor::CreateChain(specs);
      }

      return RNTupleProcessor::Create({std::string(ntupleName), std::string(fileName)});
   }

   std::unique_ptr<RNTupleProcessor> CreateJoinedProcessor()
   {
      std::vector<RNTupleOpenSpec> stepZeroSpecs{{"stepzero", fStepZeroFiles[0]}, {"stepzero", fStepZeroFiles[1]}};

      auto stepZeroProc = RNTupleProcessor::CreateChain(stepZeroSpecs, "stepzero");

      const auto &[chainStepOne, chainStepTwo, chainStepThree, chainStepFour] = GetParam();

      auto stepOneProc = CreateStepProcessor("stepone", fStepOneFile, chainStepOne);
      auto stepTwoProc = CreateStepProcessor("steptwo", fStepTwoFile, chainStepTwo);
      auto stepThreeProc = CreateStepProcessor("stepthree", fStepThreeFile, chainStepThree);
      auto stepFourProc = CreateStepProcessor("stepfour", fStepFourFile, chainStepFour);

      auto joined = RNTupleProcessor::CreateJoin(std::move(stepFourProc), std::move(stepThreeProc), {});
      joined = RNTupleProcessor::CreateJoin(std::move(joined), std::move(stepTwoProc), {});
      joined = RNTupleProcessor::CreateJoin(std::move(joined), std::move(stepOneProc), {});
      joined = RNTupleProcessor::CreateJoin(std::move(joined), std::move(stepZeroProc), {});

      return joined;
   }
};

TEST_P(GH20033ProcessorTest, Regression)
{
   auto proc = CreateJoinedProcessor();

   auto stepFourBr1 = proc->RequestField<int>("stepFourBr1");
   auto stepThreeBr1 = proc->RequestField<int>("stepthree.stepThreeBr1");
   auto stepTwoBr1 = proc->RequestField<int>("steptwo.stepTwoBr1");
   auto stepOneBr1 = proc->RequestField<int>("stepone.stepOneBr1");
   auto stepZeroBr1 = proc->RequestField<int>("stepzero.stepZeroBr1");
   auto stepZeroBr2 = proc->RequestField<int>("stepzero.stepZeroBr2");

   std::size_t nEntries = 0;

   for (auto idx : *proc) {
      EXPECT_EQ(nEntries, idx);

      EXPECT_EQ(static_cast<int>(400 + idx), *stepFourBr1);
      EXPECT_EQ(static_cast<int>(300 + idx), *stepThreeBr1);
      EXPECT_EQ(static_cast<int>(200 + idx), *stepTwoBr1);
      EXPECT_EQ(static_cast<int>(100 + idx), *stepOneBr1);
      EXPECT_EQ(static_cast<int>(idx), *stepZeroBr1);
      EXPECT_EQ(static_cast<int>(2 * idx), *stepZeroBr2);

      ++nEntries;
   }

   EXPECT_EQ(20u, nEntries);
   EXPECT_EQ(20u, proc->GetNEntriesProcessed());
}

TEST_P(GH20033ProcessorTest, SameFieldName)
{
   auto proc = CreateJoinedProcessor();

   auto stepFourValue = proc->RequestField<int>("value");
   auto stepThreeValue = proc->RequestField<int>("stepthree.value");
   auto stepTwoValue = proc->RequestField<int>("steptwo.value");
   auto stepOneValue = proc->RequestField<int>("stepone.value");
   auto stepZeroValue = proc->RequestField<int>("stepzero.value");

   std::size_t nEntries = 0;

   for (auto idx : *proc) {
      EXPECT_EQ(nEntries, idx);

      EXPECT_EQ(static_cast<int>(400 + idx), *stepFourValue);
      EXPECT_EQ(static_cast<int>(300 + idx), *stepThreeValue);
      EXPECT_EQ(static_cast<int>(200 + idx), *stepTwoValue);
      EXPECT_EQ(static_cast<int>(100 + idx), *stepOneValue);
      EXPECT_EQ(static_cast<int>(idx), *stepZeroValue);

      ++nEntries;
   }

   EXPECT_EQ(20u, nEntries);
   EXPECT_EQ(20u, proc->GetNEntriesProcessed());
}

INSTANTIATE_TEST_SUITE_P(CreateVsCreateChain, GH20033ProcessorTest,
                         testing::Combine(testing::Bool(), testing::Bool(), testing::Bool(), testing::Bool()));
