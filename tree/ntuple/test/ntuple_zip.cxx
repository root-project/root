#include "ntuple_test.hxx"

TEST(RNTupleZip, Basics)
{
   std::string data = "xxxxxxxxxxxxxxxxxxxxxxxx";
   auto zipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), 101, zipBuffer.get());
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), data.length()));
}

TEST(RNTupleZip, Empty)
{
   char x = 0;
   char z;
   EXPECT_EQ(0U, RNTupleCompressor::Zip(&x, 0, 0, &z));
   EXPECT_EQ(0U, RNTupleCompressor::Zip(&x, 0, 101, &z));

   // Don't crash
   RNTupleDecompressor::Unzip(&x, 0, 0, &x);
}

TEST(RNTupleZip, Uncompressed)
{
   char X = 'x';
   char Z;
   EXPECT_EQ(1U, RNTupleCompressor::Zip(&X, 1, 0, &Z));
   RNTupleDecompressor::Unzip(&Z, 1, 1, &X);
   EXPECT_EQ('x', X);
}

TEST(RNTupleZip, Small)
{
   char X = 'x';
   char Z;
   EXPECT_EQ(1U, RNTupleCompressor::Zip(&X, 1, 101, &Z));
   RNTupleDecompressor::Unzip(&Z, 1, 1, &X);
   EXPECT_EQ('x', X);
}

TEST(RNTupleZip, LargeWithOutputBuffer)
{
   constexpr unsigned int N = kMAXZIPBUF + 32;
   auto zipBuffer = MakeUninitArray<unsigned char>(N);
   auto unzipBuffer = MakeUninitArray<char>(N);
   std::string data(N, 'x');

   /// One byte more than a block: two blocks of about the same size, both compressed (no 1-byte trailing block,
   /// which could not be compressed and would leave the entire buffer uncompressed)
   auto szZip = RNTupleCompressor::Zip(data.data(), kMAXZIPBUF + 1, 101, zipBuffer.get());
   EXPECT_LT(szZip, static_cast<unsigned int>(kMAXZIPBUF) + 1);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZip, kMAXZIPBUF + 1, unzipBuffer.get());
   EXPECT_EQ(std::string_view(data.data(), kMAXZIPBUF + 1), std::string_view(unzipBuffer.get(), kMAXZIPBUF + 1));

   szZip = RNTupleCompressor::Zip(data.data(), data.length(), 101, zipBuffer.get());
   EXPECT_LT(szZip, N);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZip, N, unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), N));
}

TEST(RNTupleZip, LargeJustAboveBlocks)
{
   // Buffers just above one or two blocks: a page of 2^24 or 2^25 bytes must not be stored uncompressed
   for (std::size_t n : {std::size_t(kMAXZIPBUF) + 1, std::size_t(kMAXZIPBUF) + 2, std::size_t(1) << 25}) {
      auto zipBuffer = MakeUninitArray<unsigned char>(n);
      auto unzipBuffer = MakeUninitArray<char>(n);
      std::string data(n, 'x');
      for (int compression : {101, 207, 404, 505}) {
         auto szZip = RNTupleCompressor::Zip(data.data(), n, compression, zipBuffer.get());
         EXPECT_LT(szZip, n) << "size " << n << ", compression " << compression;
         RNTupleDecompressor::Unzip(zipBuffer.get(), szZip, n, unzipBuffer.get());
         EXPECT_EQ(data, std::string_view(unzipBuffer.get(), n));
      }
   }
}

TEST(RNTupleZip, CorruptedInput)
{
   std::string data = "xxxxxxxxxxxxxxxxxxxxxxxx";
   auto zipBuffer = MakeUninitArray<unsigned char>(data.length());
   auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), 101, zipBuffer.get());
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = MakeUninitArray<unsigned char>(data.length());
   // corrupt the buffer header
   memset(zipBuffer.get() + 1, 0xCD, 5);
   EXPECT_THROW(RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get()),
                ROOT::RException);
}
