#include "ntuple_test.hxx"

namespace {

void CreateCorruptedFile(const std::string &fileName)
{
   auto model = RNTupleModel::Create();
   auto ptrPx = model->MakeField<float>("px", 1.0);
   auto ptrPy = model->MakeField<float>("py", 2.0);
   RNTupleWriteOptions options;
   options.SetCompression(0);
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileName, options);
   writer->Fill();
   writer.reset();

   auto pageSource = RPageSource::Create("ntpl", fileName);
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   const auto &clusterDesc = descGuard->GetClusterDescriptor(clusterId);
   const auto pageInfo = clusterDesc.GetPageRange(pxColId).fPageInfos[0];
   EXPECT_EQ(4u, pageInfo.fLocator.fBytesOnStorage);
   EXPECT_TRUE(pageInfo.fHasChecksum);

   std::uint64_t wrongChecksum = 0;
   FILE *f = fopen(fileName.c_str(), "r+b");
   fseek(f, pageInfo.fLocator.GetPosition<std::uint64_t>() + 4, SEEK_SET);
   fwrite(&wrongChecksum, sizeof(wrongChecksum), 1, f);
   fclose(f);
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

      // TODO(jblomer): check that now even reading py fails because pages are unsealed in parallel
      // Requires fixing the cluster scheduler.
   }
}

#ifdef R__USE_IMT
TEST(RNTupleChecksum, VerifyOnReadImt)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_read.root");

   CreateCorruptedFile(fileGuard.GetPath());

   IMTRAII _;

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
