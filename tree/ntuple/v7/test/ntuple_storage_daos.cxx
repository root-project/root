#include "ntuple_test.hxx"
#include <ROOT/RPageStorageDaos.hxx>
#include "ROOT/TestSupport.hxx"
#include <iostream>
#include <unordered_map>

class RPageStorageDaos : public ::testing::Test {
private:
   static std::unordered_set<std::string> fContainerLabels;
   ROOT::TestSupport::CheckDiagsRAII fRootDiags;

protected:
   /// \brief Stores the test label in a class-wide collection and returns the DAOS URI ("daos://{pool}/{container}").
   /// The test label serves as the container identifier.
   static std::string RegisterLabel(std::string_view testLabel)
   {
      auto [strIt, _] = fContainerLabels.emplace(testLabel);
      static const std::string testPoolUriPrefix("daos://" R__DAOS_TEST_POOL "/");
      return {testPoolUriPrefix + *strIt};
   }

   void SetUp() override
   {
      // Initialized at the start of each test to expect diagnostic messages from TestSupport
      fRootDiags.requiredDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkDaos::RPageSinkDaos",
                              "The DAOS backend is experimental and still under development.", false);
      fRootDiags.requiredDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 1", false);
      fRootDiags.optionalDiag(kWarning, "in int daos_init()",
                              "This RNTuple build uses libdaos_mock. Use only for testing!");
   }

   static void TearDownTestSuite()
   {
#ifndef R__DAOS_TEST_MOCK
      const std::string sysCmd("daos cont destroy " R__DAOS_TEST_POOL " ");
      for (const auto &label : fContainerLabels) {
         system((sysCmd + label).data());
      }
#endif
   }
};

std::unordered_set<std::string> RPageStorageDaos::fContainerLabels{};

TEST_F(RPageStorageDaos, Basics)
{
   std::string daosUri = RegisterLabel("ntuple-test-basics");
   const std::string_view ntupleName("ntuple");
   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(0); // Disable caging mechanism.
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);

      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open(ntupleName, daosUri);
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel()->GetDefaultEntry()->Get<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
}

TEST_F(RPageStorageDaos, Extended)
{
   std::string daosUri = RegisterLabel("ntuple-test-extended");
   const std::string_view ntupleName("ntuple");
   auto model = RNTupleModel::Create();
   auto wrVector = model->MakeField<std::vector<double>>("vector");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
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
   auto ntuple = RNTupleReader::Open(ntupleName, daosUri, options);
   auto rdVector = ntuple->GetModel()->GetDefaultEntry()->Get<std::vector<double>>("vector");

   double chksumRead = 0.0;
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      for (auto v : *rdVector)
         chksumRead += v;
   }
   EXPECT_EQ(chksumRead, chksumWrite);
}

TEST_F(RPageStorageDaos, Options)
{
   std::string daosUri = RegisterLabel("ntuple-test-options");
   const std::string_view ntupleName("ntuple");
   {
      auto model = RNTupleModel::Create();

      RNTupleWriteOptionsDaos options;
      options.SetObjectClass("UNKNOWN");
      try {
         auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
         FAIL() << "unknown object class should throw";
      } catch (const RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("UNKNOWN"));
      }
   }

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);

      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(0);
      options.SetObjectClass("RP_XSF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
   }

   auto readOptions = RNTupleReadOptions();
   readOptions.SetClusterBunchSize(3);
   ROOT::Experimental::Detail::RPageSourceDaos source(ntupleName, daosUri, readOptions);
   source.Attach();
   EXPECT_STREQ("RP_XSF", source.GetObjectClass().c_str());
   EXPECT_EQ(3U, source.GetReadOptions().GetClusterBunchSize());
   EXPECT_EQ(1U, source.GetNEntries());
}

TEST_F(RPageStorageDaos, MultipleNTuplesPerContainer)
{
   std::string daosUri = RegisterLabel("ntuple-test-multiple");
   const std::string_view ntupleName1("ntuple1"), ntupleName2("ntuple2");

   RNTupleWriteOptionsDaos options;
   options.SetMaxCageSize(0);

   {
      auto model1 = RNTupleModel::Create();
      auto wrPt = model1->MakeField<float>("pt", 34.0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model1), ntupleName1, daosUri, options);
      ntuple->Fill();
      *wrPt = 160.0;
      ntuple->Fill();
   }
   {
      auto model2 = RNTupleModel::Create();
      auto wrPt = model2->MakeField<float>("pt", 81.0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model2), ntupleName2, daosUri, options);
      ntuple->Fill();
      *wrPt = 96.0;
      ntuple->Fill();
      *wrPt = 54.0;
      ntuple->Fill();
   }
   {
      auto ntuple1 = RNTupleReader::Open(ntupleName1, daosUri);
      auto ntuple2 = RNTupleReader::Open(ntupleName2, daosUri);
      EXPECT_EQ(2U, ntuple1->GetNEntries());
      EXPECT_EQ(3U, ntuple2->GetNEntries());

      {
         auto rdPt = ntuple1->GetModel()->GetDefaultEntry()->Get<float>("pt");
         ntuple1->LoadEntry(0);
         EXPECT_EQ(34.0, *rdPt);
         ntuple1->LoadEntry(1);
         EXPECT_EQ(160.0, *rdPt);
      }
      {
         auto rdPt = ntuple2->GetModel()->GetDefaultEntry()->Get<float>("pt");
         ntuple2->LoadEntry(0);
         EXPECT_EQ(81.0, *rdPt);
         ntuple2->LoadEntry(1);
         EXPECT_EQ(96.0, *rdPt);
         ntuple2->LoadEntry(2);
         EXPECT_EQ(54.0, *rdPt);
      }
   }

   // Nonexistent ntuple
   EXPECT_THROW(RNTupleReader::Open("ntuple3", daosUri), ROOT::Experimental::RException);
}

#ifdef R__USE_IMT
// This feature depends on RPageSinkBuf and the ability to issue a single `CommitSealedPageV()` call; thus, disable if
// ROOT was built with `-Dimt=OFF`
TEST_F(RPageStorageDaos, CagedPages)
{
   std::string daosUri = RegisterLabel("ntuple-test-caged");
   const std::string_view ntupleName("ntuple");
   ROOT::EnableImplicitMT();

   auto model = RNTupleModel::Create();
   auto wrVector = model->MakeField<std::vector<double>>("vector");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(4 * 64 * 1024);
      options.SetUseBufferedWrite(true);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
      constexpr unsigned int nEvents = 180000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         auto nVec = 1 + floor(rnd.Rndm() * 1000.);
         wrVector->resize(nVec);
         for (unsigned int n = 0; n < nVec; ++n) {
            auto val = 1 + rnd.Rndm() * 1000. - 500.;
            (*wrVector)[n] = val;
            chksumWrite += val;
         }
         ntuple->Fill();
      }
   }

   // Attempt to read all the entries written above as caged pages, with cluster cache turned on.
   {
      RNTupleReadOptions options;
      options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOn);
      options.SetClusterBunchSize(5);
      auto ntuple = RNTupleReader::Open(ntupleName, daosUri, options);
      auto rdVector = ntuple->GetModel()->GetDefaultEntry()->Get<std::vector<double>>("vector");

      double chksumRead = 0.0;
      for (auto entryId : *ntuple) {
         ntuple->LoadEntry(entryId);
         for (auto v : *rdVector)
            chksumRead += v;
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }

   // Wrongly attempt to read a single caged page when cluster cache is disabled.
   {
      RNTupleReadOptions options;
      options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);
      auto ntuple = RNTupleReader::Open(ntupleName, daosUri, options);
      EXPECT_THROW(ntuple->LoadEntry(1), ROOT::Experimental::RException);
   }
}
#endif
