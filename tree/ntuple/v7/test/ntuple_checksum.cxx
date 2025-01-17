
#include "ntuple_test.hxx"

#include <cstring>

TEST(RNTupleChecksum, VerifyOnRead)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_read.root");

   CreateCorruptedRNTuple(fileGuard.GetPath());

   for (auto co : {RNTupleReadOptions::EClusterCache::kOn, RNTupleReadOptions::EClusterCache::kOff}) {
      RNTupleReadOptions options;
      options.SetClusterCache(co);
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), options);
      EXPECT_EQ(1u, reader->GetNEntries());

      auto viewPx = reader->GetView<float>("px");
      auto viewPy = reader->GetView<float>("py");
      auto viewPz = reader->GetView<float>("pz");
      EXPECT_THROW(viewPx(0), ROOT::RException);
      EXPECT_THROW(viewPy(0), ROOT::RException);
      EXPECT_FLOAT_EQ(3.0, viewPz(0));
   }
}

#ifdef R__USE_IMT
TEST(RNTupleChecksum, VerifyOnReadImt)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_read.root");

   CreateCorruptedRNTuple(fileGuard.GetPath());

   IMTRAII _;

   RNTupleReadOptions options;
   options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOn);
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath(), options);
   EXPECT_EQ(1u, reader->GetNEntries());

   auto viewPx = reader->GetView<float>("px");
   auto viewPy = reader->GetView<float>("py");
   auto viewPz = reader->GetView<float>("pz");
   try {
      viewPz(0);
      FAIL() << "now even reading pz should fail because pages are unsealed in parallel";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("page checksum"));
   }
}
#endif // R__USE_IMT

TEST(RNTupleChecksum, VerifyOnLoad)
{
   FileRaii fileGuard("test_ntuple_checksum_verify_load.root");

   CreateCorruptedRNTuple(fileGuard.GetPath());

   RPageStorage::RSealedPage sealedPage;
   DescriptorId_t pxColId;
   DescriptorId_t pyColId;
   DescriptorId_t pzColId;
   DescriptorId_t clusterId;
   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   pageSource->Attach();
   {
      auto descGuard = pageSource->GetSharedDescriptorGuard();
      pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0, 0);
      pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0, 0);
      pzColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("pz"), 0, 0);
      clusterId = descGuard->FindClusterId(pxColId, 0);
   }
   RNTupleLocalIndex index{clusterId, 0};

   constexpr std::size_t bufSize = 12;
   pageSource->LoadSealedPage(pzColId, index, sealedPage);
   EXPECT_EQ(bufSize, sealedPage.GetBufferSize());
   unsigned char buffer[bufSize];
   sealedPage.SetBuffer(buffer);
   // no exception
   pageSource->LoadSealedPage(pzColId, index, sealedPage);

   EXPECT_THROW(pageSource->LoadSealedPage(pxColId, index, sealedPage), ROOT::RException);
   EXPECT_THROW(pageSource->LoadSealedPage(pyColId, index, sealedPage), ROOT::RException);
}

TEST(RNTupleChecksum, OmitPageChecksum)
{
   FileRaii fileGuard("test_ntuple_omit_page_checksum.root");

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("px") = 1.0;
   RNTupleWriteOptions options;
   options.SetCompression(0);
   options.SetEnablePageChecksums(false);
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
   writer->Fill();
   writer.reset();

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0, 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   const auto &clusterDesc = descGuard->GetClusterDescriptor(clusterId);
   const auto pageInfo = clusterDesc.GetPageRange(pxColId).fPageInfos[0];
   EXPECT_EQ(4u, pageInfo.fLocator.GetNBytesOnStorage());
   EXPECT_FALSE(pageInfo.fHasChecksum);

   RPageStorage::RSealedPage sealedPage;
   pageSource->LoadSealedPage(pxColId, RNTupleLocalIndex{clusterId, 0}, sealedPage);
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

   CreateCorruptedRNTuple(fileGuard1.GetPath());

   {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("px") = 4.0;
      *model->MakeField<float>("py") = 5.0;
      *model->MakeField<float>("pz") = 6.0;
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
   RNTupleMerger merger{std::move(destination)};
   try {
      merger.Merge(sourcePtrs);
      FAIL() << "merging should fail due to checksum error";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("page checksum"));
   }
}
