#include "ntuple_test.hxx"
#include <ROOT/RPageStorageDaos.hxx>
#include "ROOT/TestSupport.hxx"
#include <iostream>

TEST(RPageStorageDaos, Basics)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kWarning, "in int daos_init()", "This RNTuple build uses libdaos_mock. Use only for testing!");
   diags.requiredDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos", "The DAOS backend is experimental and still under development.", false);
   diags.requiredDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 1", false);

   std::string daosUri("daos://" R__DAOS_TEST_POOL "/container-test-1");

   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", daosUri);
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
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", daosUri, options);
      constexpr unsigned int nEvents = 32000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         auto nVec = 1 + floor(rnd.Rndm() * 1000.);
         wrVector->resize(nVec);
         for (unsigned int n = 0; n < nVec; ++n) {
            auto val = 1 + rnd.Rndm()*1000. - 500.;
            (*wrVector)[n] = val;
            chksumWrite += val;
         }
         ntuple->Fill();
         if (i % 1000 == 0)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("f", daosUri);
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
         auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", daosUri, options);
         FAIL() << "unknown object class should throw";
      } catch (const RException& err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("UNKNOWN"));
      }
   }

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);

      RNTupleWriteOptionsDaos options;
      options.SetObjectClass("RP_XSF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
   }

   ROOT::Experimental::Detail::RPageSourceDaos source("ntuple", daosUri, RNTupleReadOptions());
   source.Attach();
   EXPECT_STREQ("RP_XSF", source.GetObjectClass().c_str());
   EXPECT_EQ(1U, source.GetNEntries());
}
