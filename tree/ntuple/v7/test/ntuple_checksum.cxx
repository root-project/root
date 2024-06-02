#include "ntuple_test.hxx"

TEST(RNTupleChecksum, Verify)
{
   FileRaii fileGuard("test_ntuple_checksum_verify.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrPx = model->MakeField<float>("px", 1.0);
      auto ptrPy = model->MakeField<float>("py", 2.0);
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      writer->Fill();
   }

   {
      auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
      pageSource->Attach();
      auto descGuard = pageSource->GetSharedDescriptorGuard();
      const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
      const auto clusterId = descGuard->FindClusterId(pxColId, 0);
      const auto &clusterDesc = descGuard->GetClusterDescriptor(clusterId);
      const auto pageInfo = clusterDesc.GetPageRange(pxColId).fPageInfos[0];
      EXPECT_EQ(4u, pageInfo.fLocator.fBytesOnStorage);
      EXPECT_TRUE(pageInfo.fHasChecksum);

      std::uint64_t wrongChecksum = 0;
      FILE *f = fopen(fileGuard.GetPath().c_str(), "r+b");
      fseek(f, pageInfo.fLocator.GetPosition<std::uint64_t>() + 4, SEEK_SET);
      fwrite(&wrongChecksum, sizeof(wrongChecksum), 1, f);
      fclose(f);
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());

   auto viewPx = reader->GetView<float>("px");
   auto viewPy = reader->GetView<float>("py");
   EXPECT_THROW(viewPx(0), RException);
   EXPECT_FLOAT_EQ(2.0, viewPy(0));
}
