#include "ntuple_test.hxx"

TEST(RNTupleCompat, Epoch)
{
   FileRaii fileGuard("test_ntuple_compat_epoch.root");

   RNTuple ntpl;
   // The first 16 bit integer in the struct is the epoch
   std::uint16_t *versionEpoch = reinterpret_cast<uint16_t *>(&ntpl);
   *versionEpoch = *versionEpoch + 1;
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   file->WriteObject(&ntpl, "ntpl");
   file->Close();

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple with different epoch version should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported RNTuple epoch version"));
   }
}

TEST(RNTupleCompat, FeatureFlag)
{
   FileRaii fileGuard("test_ntuple_compat_feature_flag.root");

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetNTuple("ntpl", "");
   descBuilder.SetFeature(RNTupleDescriptor::kFeatureFlagTest);
   descBuilder.AddField(
      RFieldDescriptorBuilder::FromField(ROOT::Experimental::RFieldZero()).FieldId(0).MakeDescriptor().Unwrap());
   ASSERT_TRUE(static_cast<bool>(descBuilder.EnsureValidDescriptor()));

   auto writer =
      RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, RNTupleFileWriter::EContainerFormat::kTFile);
   RNTupleSerializer serializer;

   auto ctx = serializer.SerializeHeader(nullptr, descBuilder.GetDescriptor());
   auto buffer = std::make_unique<unsigned char[]>(ctx.GetHeaderSize());
   ctx = serializer.SerializeHeader(buffer.get(), descBuilder.GetDescriptor());
   writer->WriteNTupleHeader(buffer.get(), ctx.GetHeaderSize(), ctx.GetHeaderSize());

   auto szFooter = serializer.SerializeFooter(nullptr, descBuilder.GetDescriptor(), ctx);
   buffer = std::make_unique<unsigned char[]>(szFooter);
   serializer.SerializeFooter(buffer.get(), descBuilder.GetDescriptor(), ctx);
   writer->WriteNTupleFooter(buffer.get(), szFooter, szFooter);

   writer->Commit();
   // Call destructor to flush data to disk
   writer = nullptr;

   auto pageSource = RPageSource::Create("ntpl", fileGuard.GetPath());
   try {
      pageSource->Attach();
      FAIL() << "opening an RNTuple that uses an unsupported feature should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported format feature: 137"));
   }
}
