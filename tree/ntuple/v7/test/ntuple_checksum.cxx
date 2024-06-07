#include "ntuple_test.hxx"

#include <cstring>

namespace {

void CreateCorruptedFile(const std::string &uri)
{
   RNTupleWriteOptions options;
   options.SetCompression(0);

   auto model = RNTupleModel::Create();
   auto ptrPx = model->MakeField<float>("px", 1.0);
   auto ptrPy = model->MakeField<float>("py", 2.0);
   model->Freeze();
   auto modelClone = model->Clone(); // required later to write the corrupted version
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", uri, options);
   writer->Fill();
   writer.reset();

   // Load sealed pages to memory
   auto pageSource = RPageSource::Create("ntpl", uri);
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
   const auto pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   RClusterIndex index{clusterId, 0};

   constexpr std::size_t bufSize = sizeof(float) + RPageStorage::kNBytesPageChecksum;
   unsigned char pxBuffer[bufSize];
   RPageStorage::RSealedPage pxSealedPage;
   pxSealedPage.SetBufferSize(bufSize);
   pxSealedPage.SetBuffer(pxBuffer);
   pageSource->LoadSealedPage(pxColId, index, pxSealedPage);
   EXPECT_EQ(bufSize, pxSealedPage.GetBufferSize());
   unsigned char pyBuffer[bufSize];
   RPageStorage::RSealedPage pySealedPage;
   pySealedPage.SetBufferSize(bufSize);
   pySealedPage.SetBuffer(pyBuffer);
   pageSource->LoadSealedPage(pyColId, index, pySealedPage);
   EXPECT_EQ(bufSize, pySealedPage.GetBufferSize());

   // Corrupt px sealed page's checksum
   memset(pxBuffer + sizeof(float), 0, RPageStorage::kNBytesPageChecksum);

   // Rewrite RNTuple with valid py page and corrupted px page
   auto pageSink = ROOT::Experimental::Internal::RPagePersistentSink::Create("ntpl", uri, options);
   pageSink->Init(*modelClone);
   pageSink->CommitSealedPage(pxColId, pxSealedPage);
   pageSink->CommitSealedPage(pyColId, pySealedPage);
   pageSink->CommitCluster(1);
   pageSink->CommitClusterGroup();
   pageSink->CommitDataset();
   modelClone.reset();
}

#ifdef R__USE_IMT
struct IMTRAII {
   IMTRAII() { ROOT::EnableImplicitMT(); }
   ~IMTRAII() { ROOT::DisableImplicitMT(); }
};
#endif

} // anonymous namespace

TEST(RNTupleChecksum, VerifyOnRead)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_read.root");

   CreateCorruptedFile(fileGuard.GetPath());

   for (auto co : {RNTupleReadOptions::EClusterCache::kOn, RNTupleReadOptions::EClusterCache::kOff}) {
      RNTupleReadOptions options;
      options.SetClusterCache(co);
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), options);
      EXPECT_EQ(1u, reader->GetNEntries());

      auto viewPx = reader->GetView<float>("px");
      auto viewPy = reader->GetView<float>("py");
      EXPECT_THROW(viewPx(0), RException);
      EXPECT_FLOAT_EQ(2.0, viewPy(0));
   }
}

#ifdef R__USE_IMT
TEST(RNTupleChecksum, VerifyOnReadImt)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_read.root");

   CreateCorruptedFile(fileGuard.GetPath());

   IMTRAII _;

   RNTupleReadOptions options;
   options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOn);
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), options);
   EXPECT_EQ(1u, reader->GetNEntries());

   auto viewPx = reader->GetView<float>("px");
   auto viewPy = reader->GetView<float>("py");
   try {
      viewPy(0);
      FAIL() << "now even reading py should fail because pages are unsealed in parallel";
   } catch (const RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("page checksum"));
   }
}
#endif // R__USE_IMT

TEST(RNTupleChecksum, VerifyOnLoad)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_load.root");

   CreateCorruptedFile(fileGuard.GetPath());

   RPageStorage::RSealedPage sealedPage;
   DescriptorId_t pxColId;
   DescriptorId_t pyColId;
   DescriptorId_t clusterId;
   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   pageSource->Attach();
   {
      auto descGuard = pageSource->GetSharedDescriptorGuard();
      pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
      pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0);
      clusterId = descGuard->FindClusterId(pxColId, 0);
   }
   RClusterIndex index{clusterId, 0};

   constexpr std::size_t bufSize = 12;
   pageSource->LoadSealedPage(pyColId, index, sealedPage);
   EXPECT_EQ(bufSize, sealedPage.GetBufferSize());
   unsigned char buffer[bufSize];
   sealedPage.SetBuffer(buffer);
   // no exception
   pageSource->LoadSealedPage(pyColId, index, sealedPage);

   EXPECT_THROW(pageSource->LoadSealedPage(pxColId, index, sealedPage), RException);
}

TEST(RNTupleChecksum, OmitPageChecksum)
{
   FileRaii fileGuard("test_ntuple_omit_page_checksum.root");

   auto model = RNTupleModel::Create();
   auto ptrPx = model->MakeField<float>("px", 1.0);
   RNTupleWriteOptions options;
   options.SetCompression(0);
   options.SetEnablePageChecksums(false);
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
   writer->Fill();
   writer.reset();

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   const auto &clusterDesc = descGuard->GetClusterDescriptor(clusterId);
   const auto pageInfo = clusterDesc.GetPageRange(pxColId).fPageInfos[0];
   EXPECT_EQ(4u, pageInfo.fLocator.fBytesOnStorage);
   EXPECT_FALSE(pageInfo.fHasChecksum);

   RPageStorage::RSealedPage sealedPage;
   pageSource->LoadSealedPage(pxColId, RClusterIndex{clusterId, 0}, sealedPage);
   EXPECT_FALSE(sealedPage.GetHasChecksum());
   EXPECT_EQ(4u, sealedPage.GetBufferSize());

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   auto viewPx = reader->GetView<float>("px");
   EXPECT_FLOAT_EQ(1.0, viewPx(0));
}

TEST(RNTupleChecksum, Merge)
{
   FileRaii fileGuard1("test_ntuple_checksum_merge1.root");
   FileRaii fileGuard2("test_ntuple_checksum_merge2.root");

   RNTupleWriteOptions options;
   options.SetCompression(0);

   CreateCorruptedFile(fileGuard1.GetPath());

   {
      auto model = RNTupleModel::Create();
      auto ptrPx = model->MakeField<float>("px", 3.0);
      auto ptrPy = model->MakeField<float>("py", 4.0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      writer->Fill();
   }

   FileRaii fileGuard3("test_ntuple_checksum_merge_out.root");
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntpl", fileGuard1.GetPath()));
   sources.push_back(RPageSource::Create("ntpl", fileGuard2.GetPath()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   auto destination = std::make_unique<RPageSinkFile>("ntpl", fileGuard3.GetPath(), options);
   RNTupleMerger merger;
   try {
      merger.Merge(sourcePtrs, *destination);
      FAIL() << "merging should fail due to checksum error";
   } catch (const RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("page checksum"));
   }
}
