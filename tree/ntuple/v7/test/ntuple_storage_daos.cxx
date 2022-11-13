#include "ntuple_test.hxx"
#include <ROOT/RPageStorageDaos.hxx>
#include "ROOT/TestSupport.hxx"
#include <iostream>

TEST(RPageStorageDaos, Basics)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kWarning, "in int daos_init()", "This RNTuple build uses libdaos_mock. Use only for testing!");
   diags.requiredDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos",
                      "The DAOS backend is experimental and still under development.", false);
   diags.requiredDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 1", false);

   std::string daosUri("daos://" R__DAOS_TEST_POOL "/container-test-1");

   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple-1", daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple-1", daosUri);
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel()->GetDefaultEntry()->Get<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
}

TEST(RPageStorageDaos, Extended)
{
   std::string daosUri("daos://" R__DAOS_TEST_POOL "/container-test-2");

   auto model = RNTupleModel::Create();
   auto wrVector = model->MakeField<std::vector<double>>("vector");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple-2", daosUri, options);
      constexpr unsigned int nEvents = 32000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         auto nVec = 1 + floor(rnd.Rndm() * 1000.);
         wrVector->resize(nVec);
         for (unsigned int n = 0; n < nVec; ++n) {
            auto val = 1 + rnd.Rndm() * 1000. - 500.;
            (*wrVector)[n] = val;
            chksumWrite += val;
         }
         ntuple->Fill();
         if (i % 1000 == 0)
            ntuple->CommitCluster();
      }
   }

   RNTupleReadOptions options;
   options.SetClusterBunchSize(5);
   auto ntuple = RNTupleReader::Open("ntuple-2", daosUri, options);
   auto rdVector = ntuple->GetModel()->GetDefaultEntry()->Get<std::vector<double>>("vector");

   double chksumRead = 0.0;
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      for (auto v : *rdVector)
         chksumRead += v;
   }
   EXPECT_EQ(chksumRead, chksumWrite);
}

TEST(RPageStorageDaos, Options)
{
   std::string daosUri("daos://" R__DAOS_TEST_POOL "/container-test-3");

   {
      auto model = RNTupleModel::Create();

      RNTupleWriteOptionsDaos options;
      options.SetObjectClass("UNKNOWN");
      try {
         auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple-3", daosUri, options);
         FAIL() << "unknown object class should throw";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("UNKNOWN"));
      }
   }

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);

      RNTupleWriteOptionsDaos options;
      options.SetObjectClass("RP_XSF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple-3", daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
   }

   auto readOptions = RNTupleReadOptions();
   readOptions.SetClusterBunchSize(3);
   ROOT::Experimental::Detail::RPageSourceDaos source("ntuple-3", daosUri, readOptions);
   source.Attach();
   EXPECT_STREQ("RP_XSF", source.GetObjectClass().c_str());
   EXPECT_EQ(3U, source.GetReadOptions().GetClusterBunchSize());
   EXPECT_EQ(1U, source.GetNEntries());
}

TEST(RPageStorageDaos, MultipleNTuplesPerContainer)
{
   std::string daosUri("daos://" R__DAOS_TEST_POOL "/container-test-4");

   RNTupleWriteOptions options;

   {
      auto modelA = RNTupleModel::Create();
      auto wrPt = modelA->MakeField<float>("pt", 34.0);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelA), "ntupleA", daosUri, options);
      ntuple->Fill();
      *wrPt = 160.0;
      ntuple->Fill();
   }
   {
      auto modelB = RNTupleModel::Create();
      auto wrPt = modelB->MakeField<float>("pt", 81.0);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelB), "ntupleB", daosUri, options);
      ntuple->Fill();
      *wrPt = 96.0;
      ntuple->Fill();
      *wrPt = 54.0;
      ntuple->Fill();
   }
   {
      auto ntupleA = RNTupleReader::Open("ntupleA", daosUri);
      auto ntupleB = RNTupleReader::Open("ntupleB", daosUri);
      EXPECT_EQ(2U, ntupleA->GetNEntries());
      EXPECT_EQ(3U, ntupleB->GetNEntries());

      {
         auto rdPt = ntupleA->GetModel()->GetDefaultEntry()->Get<float>("pt");
         ntupleA->LoadEntry(0);
         EXPECT_EQ(34.0, *rdPt);
         ntupleA->LoadEntry(1);
         EXPECT_EQ(160.0, *rdPt);
      }
      {
         auto rdPt = ntupleB->GetModel()->GetDefaultEntry()->Get<float>("pt");
         ntupleB->LoadEntry(0);
         EXPECT_EQ(81.0, *rdPt);
         ntupleB->LoadEntry(1);
         EXPECT_EQ(96.0, *rdPt);
         ntupleB->LoadEntry(2);
         EXPECT_EQ(54.0, *rdPt);
      }
   }

   // Nonexistent ntuple
   EXPECT_THROW(RNTupleReader::Open("ntupleC", daosUri), ROOT::Experimental::RException);
}
