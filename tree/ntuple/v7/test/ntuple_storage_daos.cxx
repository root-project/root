#include "ntuple_test.hxx"
#include <ROOT/RPageStorageDaos.hxx>

#include <TRandom3.h>

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
      fRootDiags.optionalDiag(kWarning, "ROOT::Experimental::Internal::RPageSinkDaos::RPageSinkDaos",
                              "The DAOS backend is experimental and still under development.", false);
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
   auto wrPt = model->MakeField<float>("pt");

   {
      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(0); // Disable caging mechanism.
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);

      *wrPt = 42.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open(ntupleName, daosUri);
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel().GetDefaultEntry().GetPtr<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);

   try {
      ntuple->LoadEntry(3);
      FAIL() << "loading a non-existing entry should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("entry with index 3 out of bounds"));
   }
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
   auto rdVector = ntuple->GetModel().GetDefaultEntry().GetPtr<std::vector<double>>("vector");

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
      } catch (const ROOT::RException &err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("UNKNOWN"));
      }
   }

   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("pt");

      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(0);
      options.SetObjectClass("RP_XSF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
      ntuple->Fill();
      ntuple->CommitCluster();
   }

   auto readOptions = RNTupleReadOptions();
   readOptions.SetClusterBunchSize(3);
   ROOT::Experimental::Internal::RPageSourceDaos source(ntupleName, daosUri, readOptions);
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
      auto wrPt = model1->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model1), ntupleName1, daosUri, options);
      *wrPt = 34.0;
      ntuple->Fill();
      *wrPt = 160.0;
      ntuple->Fill();
   }
   {
      auto model2 = RNTupleModel::Create();
      auto wrPt = model2->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model2), ntupleName2, daosUri, options);
      *wrPt = 81.0;
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
         auto rdPt = ntuple1->GetModel().GetDefaultEntry().GetPtr<float>("pt");
         ntuple1->LoadEntry(0);
         EXPECT_EQ(34.0, *rdPt);
         ntuple1->LoadEntry(1);
         EXPECT_EQ(160.0, *rdPt);
      }
      {
         auto rdPt = ntuple2->GetModel().GetDefaultEntry().GetPtr<float>("pt");
         ntuple2->LoadEntry(0);
         EXPECT_EQ(81.0, *rdPt);
         ntuple2->LoadEntry(1);
         EXPECT_EQ(96.0, *rdPt);
         ntuple2->LoadEntry(2);
         EXPECT_EQ(54.0, *rdPt);
      }
   }

   // Nonexistent ntuple
   EXPECT_THROW(RNTupleReader::Open("ntuple3", daosUri), ROOT::RException);
}

TEST_F(RPageStorageDaos, DisabledSamePageMerging)
{
   std::string daosUri = RegisterLabel("ntuple-test-disabled-same-page-merging");
   auto model = RNTupleModel::Create();
   *model->MakeField<float>("px") = 1.0;
   *model->MakeField<float>("py") = 1.0;
   RNTupleWriteOptionsDaos options;
   options.SetEnablePageChecksums(true);
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", daosUri, options);
   writer->Fill();
   writer.reset();

   auto reader = RNTupleReader::Open("ntpl", daosUri);
   EXPECT_EQ(1u, reader->GetNEntries());

   const auto &desc = reader->GetDescriptor();
   const auto pxColId = desc.FindPhysicalColumnId(desc.FindFieldId("px"), 0, 0);
   const auto pyColId = desc.FindPhysicalColumnId(desc.FindFieldId("py"), 0, 0);
   const auto clusterId = desc.FindClusterId(pxColId, 0);
   const auto &clusterDesc = desc.GetClusterDescriptor(clusterId);
   EXPECT_FALSE(clusterDesc.GetPageRange(pxColId).Find(0).fLocator.fPosition ==
                clusterDesc.GetPageRange(pyColId).Find(0).fLocator.fPosition);

   auto viewPx = reader->GetView<float>("px");
   auto viewPy = reader->GetView<float>("py");
   EXPECT_FLOAT_EQ(1.0, viewPx(0));
   EXPECT_FLOAT_EQ(1.0, viewPy(0));
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
   auto wrCnt = model->MakeField<std::uint32_t>("cnt");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptionsDaos options;
      options.SetMaxCageSize(4 * 64 * 1024);
      options.SetUseBufferedWrite(true);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, daosUri, options);
      constexpr unsigned int nEvents = 180000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         *wrCnt = i;
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
      auto rdVector = ntuple->GetModel().GetDefaultEntry().GetPtr<std::vector<double>>("vector");

      double chksumRead = 0.0;
      for (auto entryId : *ntuple) {
         ntuple->LoadEntry(entryId);
         for (auto v : *rdVector)
            chksumRead += v;
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }

   {
      RNTupleReadOptions options;
      options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);
      auto ntuple = RNTupleReader::Open(ntupleName, daosUri, options);
      // Attempt to read a caged page data when cluster cache is disabled.
      EXPECT_THROW(ntuple->LoadEntry(1), ROOT::RException);

      // However, loading a single sealed page should work
      auto pageSource = RPageSource::Create(ntupleName, daosUri, options);
      pageSource->Attach();
      const auto &desc = pageSource->GetSharedDescriptorGuard()->Clone();
      const auto colId = desc->FindPhysicalColumnId(desc->FindFieldId("cnt"), 0, 0);
      const auto clusterId = desc->FindClusterId(colId, 0);

      RPageStorage::RSealedPage sealedPage;
      pageSource->LoadSealedPage(colId, RClusterIndex{clusterId, 0}, sealedPage);
      EXPECT_GT(sealedPage.GetNElements(), 0);
      auto pageBuf = std::make_unique<unsigned char[]>(sealedPage.GetBufferSize());
      sealedPage.SetBuffer(pageBuf.get());
      pageSource->LoadSealedPage(colId, RClusterIndex{clusterId, 0}, sealedPage);

      auto colType = desc->GetColumnDescriptor(colId).GetType();
      auto elem = ROOT::Experimental::Internal::RColumnElementBase::Generate<std::uint32_t>(colType);
      auto page = pageSource->UnsealPage(sealedPage, *elem).Unwrap();
      EXPECT_GT(page.GetNElements(), 0);
      auto ptrData = static_cast<std::uint32_t *>(page.GetBuffer());
      for (std::uint32_t i = 0; i < page.GetNElements(); ++i) {
         EXPECT_EQ(i, *(ptrData + i));
      }
   }
}

TEST_F(RPageStorageDaos, Checksum)
{
   std::string daosUri = RegisterLabel("ntuple-test-checksum");
   CreateCorruptedRNTuple(daosUri);

   {
      IMTRAII _;

      auto reader = RNTupleReader::Open("ntpl", daosUri);
      EXPECT_EQ(1u, reader->GetNEntries());

      auto viewPx = reader->GetView<float>("px");
      auto viewPy = reader->GetView<float>("py");
      auto viewPz = reader->GetView<float>("pz");
      EXPECT_THROW(viewPz(0), ROOT::RException); // we run under IMT, even the valid column should fail
   }

   auto reader = RNTupleReader::Open("ntpl", daosUri);
   EXPECT_EQ(1u, reader->GetNEntries());

   auto viewPx = reader->GetView<float>("px");
   auto viewPy = reader->GetView<float>("py");
   auto viewPz = reader->GetView<float>("pz");
   EXPECT_THROW(viewPx(0), ROOT::RException);
   EXPECT_THROW(viewPy(0), ROOT::RException);
   EXPECT_FLOAT_EQ(3.0, viewPz(0));

   DescriptorId_t pxColId;
   DescriptorId_t pyColId;
   DescriptorId_t clusterId;
   auto pageSource = RPageSource::Create("ntpl", daosUri);
   pageSource->Attach();
   {
      auto descGuard = pageSource->GetSharedDescriptorGuard();
      pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0, 0);
      pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0, 0);
      clusterId = descGuard->FindClusterId(pxColId, 0);
   }
   RClusterIndex index{clusterId, 0};

   RPageStorage::RSealedPage sealedPage;
   constexpr std::size_t bufSize = 12;
   unsigned char buffer[bufSize];
   sealedPage.SetBuffer(buffer);
   EXPECT_THROW(pageSource->LoadSealedPage(pxColId, index, sealedPage), ROOT::RException);
   EXPECT_THROW(pageSource->LoadSealedPage(pyColId, index, sealedPage), ROOT::RException);
}
#endif
