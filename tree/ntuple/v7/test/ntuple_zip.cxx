#include "ntuple_test.hxx"

TEST(RNTupleZip, Basics)
{
   RNTupleCompressor compressor;
   RNTupleDecompressor decompressor;

   std::string data = "xxxxxxxxxxxxxxxxxxxxxxxx";
   auto szZipped = compressor.Zip(data.data(), data.length(), 101);
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   decompressor.Unzip(compressor.GetZipBuffer(), szZipped, data.length(), unzipBuffer.get());
   EXPECT_EQ(data, std::string(unzipBuffer.get(), data.length()));

   // inplace decompression
   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char [data.length()]);
   memcpy(zipBuffer.get(), compressor.GetZipBuffer(), szZipped);
   decompressor.Unzip(zipBuffer.get(), szZipped, data.length());
   EXPECT_EQ(data, std::string(reinterpret_cast<char *>(zipBuffer.get()), data.length()));
}


TEST(RNTupleZip, Empty)
{
   RNTupleCompressor compressor;

   char x;
   EXPECT_EQ(0U, compressor.Zip(&x, 0, 0));
   EXPECT_EQ(0U, compressor.Zip(&x, 0, 101));

   // Don't crash
   RNTupleDecompressor().Unzip(&x, 0, 0, &x);
}


TEST(RNTupleZip, Uncompressed)
{
   RNTupleCompressor compressor;
   char X = 'x';
   EXPECT_EQ(1U, compressor.Zip(&X, 1, 0));
   RNTupleDecompressor().Unzip(compressor.GetZipBuffer(), 1, 1, &X);
   EXPECT_EQ('x', X);
}


TEST(RNTupleZip, Small)
{
   RNTupleCompressor compressor;
   char X = 'x';
   EXPECT_EQ(1U, compressor.Zip(&X, 1, 101));
   RNTupleDecompressor().Unzip(compressor.GetZipBuffer(), 1, 1, &X);
   EXPECT_EQ('x', X);
}


TEST(RNTupleZip, Large)
{
   constexpr unsigned int N = kMAXZIPBUF + 32;
   auto zipBuffer = std::make_unique<unsigned char[]>(N);
   auto unzipBuffer = std::make_unique<char[]>(N);
   std::string data(N, 'x');

   RNTupleCompressor compressor;
   RNTupleDecompressor decompressor;

   /// Trailing byte cannot be compressed, entire buffer returns uncompressed
   int nWrites = 0;
   auto szZip = compressor.Zip(data.data(), kMAXZIPBUF + 1, 101,
      [&zipBuffer, &nWrites](const void *buffer, size_t nbytes, size_t offset) {
         memcpy(zipBuffer.get() + offset, buffer, nbytes);
         nWrites++;
      });
   EXPECT_EQ(2, nWrites);
   EXPECT_EQ(static_cast<unsigned int>(kMAXZIPBUF) + 1, szZip);

   nWrites = 0;
   szZip = compressor.Zip(data.data(), data.length(), 101,
      [&zipBuffer, &nWrites](const void *buffer, size_t nbytes, size_t offset) {
         memcpy(zipBuffer.get() + offset, buffer, nbytes);
         nWrites++;
      });
   EXPECT_LT(szZip, N);
   EXPECT_EQ(2, nWrites);
   decompressor.Unzip(zipBuffer.get(), szZip, N, unzipBuffer.get());
   EXPECT_EQ(data, std::string(unzipBuffer.get(), N));
}

TEST(RNTupleWriter, TFilePtr) {
   FileRaii fileGuard("test_ntuple_zip_tfileptr_comp.root");
   {
      std::unique_ptr<TFile> file;
      auto model = RNTupleModel::Create();
      auto field = model->MakeField<float>("field");
      auto klassVec = model->MakeField<std::vector<CustomStruct>>("klassVec");
      RNTupleWriteOptions options;
      options.SetCompression(404);
      auto ntuple = std::make_unique<RNTupleWriter>(std::move(model),
         std::make_unique<RPageSinkFile>("ntuple", fileGuard.GetPath(), options, file
      ));
      for (int i = 0; i < 20000; i++) {
         *field = static_cast<float>(i);
         CustomStruct klass;
         klass.s = std::to_string(i);
         *klassVec = {klass};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
   std::ostringstream oss;
   ntuple->PrintInfo(ROOT::Experimental::ENTupleInfo::kStorageDetails, oss);
   EXPECT_THAT(oss.str(), testing::HasSubstr("Compression: 404"));
#endif
   auto rdField = ntuple->GetView<float>("field");
   auto klassVecField = ntuple->GetView<std::vector<CustomStruct>>("klassVec");
   EXPECT_EQ(20000, ntuple->GetNEntries());
   for (auto i : ntuple->GetEntryRange()) {
      ASSERT_EQ(static_cast<float>(i), rdField(i));
      ASSERT_EQ(std::to_string(i), klassVecField(i).at(0).s);
   }
}
