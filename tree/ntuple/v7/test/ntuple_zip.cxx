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

   /// Trailing byte cannot be compressed, entire buffer returns uncompressed
   auto szZip = RNTupleCompressor::Zip(data.data(), kMAXZIPBUF + 1, 101, zipBuffer.get());
   EXPECT_EQ(static_cast<unsigned int>(kMAXZIPBUF) + 1, szZip);

   szZip = RNTupleCompressor::Zip(data.data(), data.length(), 101, zipBuffer.get());
   EXPECT_LT(szZip, N);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZip, N, unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), N));
}
