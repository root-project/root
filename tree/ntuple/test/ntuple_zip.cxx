#include "ntuple_test.hxx"

#ifdef R__HAS_LHC4CODEC
#include "ZipLHC4.h"

#include <cstdint>
#include <cstring>
#include <vector>
#endif

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

#ifdef R__HAS_LHC4CODEC
TEST(RNTupleZip, LHC4)
{
   std::string data = "xxxxxxxxxxxxxxxxxxxxxxxx";
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4,
                                                     ROOT::RCompressionSetting::ELevel::kDefaultLHC4);
   auto zipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), compression, zipBuffer.get());
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), data.length()));
}

TEST(RNTupleZip, LHC4Filters)
{
   R__SetLHC4Filters(1);
   R__SetLHC4Bwt(0);
   std::vector<std::uint32_t> column(1024);
   for (std::size_t i = 0; i < column.size(); ++i)
      column[i] = static_cast<std::uint32_t>(i);
   const auto *data = reinterpret_cast<const char *>(column.data());
   const auto dataLen = column.size() * sizeof(std::uint32_t);
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 6);
   auto zipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   auto szZipped = RNTupleCompressor::Zip(data, dataLen, compression, zipBuffer.get());
   EXPECT_LT(szZipped, dataLen);
   auto unzipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, dataLen, unzipBuffer.get());
   EXPECT_EQ(0, std::memcmp(data, unzipBuffer.get(), dataLen));
   R__SetLHC4Filters(0);
}

TEST(RNTupleZip, LHC4Bwt)
{
   R__SetLHC4Filters(0);
   R__SetLHC4Codec(kLHC4CodecBwt);
   std::string data(4096, 'a');
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 6);
   auto zipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), compression, zipBuffer.get());
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), data.length()));
   R__SetLHC4Codec(kLHC4CodecBeam);
}

TEST(RNTupleZip, LHC4FilterOptions)
{
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4Filters(1);
   R__SetLHC4FilterFallback(0);
   R__SetLHC4FilterRle(1);
   R__SetLHC4FilterDict(0);
   std::vector<std::uint32_t> column(1024);
   for (std::size_t i = 0; i < column.size(); ++i)
      column[i] = static_cast<std::uint32_t>(i);
   const auto *data = reinterpret_cast<const char *>(column.data());
   const auto dataLen = column.size() * sizeof(std::uint32_t);
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 6);
   auto zipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   auto szZipped = RNTupleCompressor::Zip(data, dataLen, compression, zipBuffer.get());
   EXPECT_LT(szZipped, dataLen);
   auto unzipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, dataLen, unzipBuffer.get());
   EXPECT_EQ(0, std::memcmp(data, unzipBuffer.get(), dataLen));
   R__SetLHC4Filters(0);
   R__SetLHC4FilterFallback(1);
   R__SetLHC4FilterDict(1);
}

TEST(RNTupleZip, LHC4ExternalCodec)
{
   if (!R__LHC4CodecAvailable(kLHC4CodecZstd))
      GTEST_SKIP() << "lhc4codec zstd backend not available";

   R__SetLHC4Codec(kLHC4CodecZstd);
   R__SetLHC4Filters(0);
   std::string data(4096, 'x');
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 6);
   auto zipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), compression, zipBuffer.get());
   EXPECT_LT(szZipped, data.length());
   auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get());
   EXPECT_EQ(data, std::string_view(unzipBuffer.get(), data.length()));
   R__SetLHC4Codec(kLHC4CodecBeam);
}

TEST(RNTupleZip, LHC4NativeCodecs)
{
   const int codecs[] = {kLHC4CodecMosaic, kLHC4CodecCrystal, kLHC4CodecOracle};
   std::string data(4096, 'x');
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 5);
   for (int codec : codecs) {
      ASSERT_TRUE(R__LHC4CodecAvailable(codec));
      R__SetLHC4Codec(codec);
      R__SetLHC4Filters(1);
      auto zipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
      auto szZipped = RNTupleCompressor::Zip(data.data(), data.length(), compression, zipBuffer.get());
      EXPECT_LT(szZipped, data.length()) << "codec " << codec;
      auto unzipBuffer = std::unique_ptr<char[]>(new char[data.length()]);
      RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, data.length(), unzipBuffer.get());
      EXPECT_EQ(data, std::string_view(unzipBuffer.get(), data.length())) << "codec " << codec;
   }
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4Filters(0);
}

TEST(RNTupleZip, LHC4Auto)
{
   R__SetLHC4Codec(kLHC4CodecAuto);
   R__SetLHC4AutoMinGainPct(1);
   R__SetLHC4Filters(1);
   std::vector<std::uint32_t> column(1024);
   for (std::size_t i = 0; i < column.size(); ++i)
      column[i] = static_cast<std::uint32_t>(i);
   const auto *data = reinterpret_cast<const char *>(column.data());
   const auto dataLen = column.size() * sizeof(std::uint32_t);
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 5);
   auto zipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   auto szZipped = RNTupleCompressor::Zip(data, dataLen, compression, zipBuffer.get());
   EXPECT_LT(szZipped, dataLen);
   auto unzipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, dataLen, unzipBuffer.get());
   EXPECT_EQ(0, std::memcmp(data, unzipBuffer.get(), dataLen));
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4Filters(0);
}

TEST(RNTupleZip, LHC4AutoMaxLevel)
{
   R__SetLHC4Codec(kLHC4CodecAuto);
   R__SetLHC4AutoMinGainPct(1);
   R__SetLHC4AutoMaxLevel(1);
   R__SetLHC4Filters(1);
   std::vector<std::uint32_t> column(1024);
   for (std::size_t i = 0; i < column.size(); ++i)
      column[i] = static_cast<std::uint32_t>(i);
   const auto *data = reinterpret_cast<const char *>(column.data());
   const auto dataLen = column.size() * sizeof(std::uint32_t);
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 5);
   auto zipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   auto szZipped = RNTupleCompressor::Zip(data, dataLen, compression, zipBuffer.get());
   EXPECT_LT(szZipped, dataLen);
   auto unzipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, dataLen, unzipBuffer.get());
   EXPECT_EQ(0, std::memcmp(data, unzipBuffer.get(), dataLen));
   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4AutoMaxLevel(0);
   R__SetLHC4Filters(0);
}

TEST(RNTupleZip, LHC4AutoCodecs)
{
   EXPECT_EQ(kLHC4AutoCodecsRoot, R__GetLHC4AutoCodecs());

   R__SetLHC4Codec(kLHC4CodecAuto);
   R__SetLHC4AutoMinGainPct(0);
   R__SetLHC4AutoCodecs(kLHC4CodecBit(kLHC4CodecBeam));
   R__SetLHC4Filters(1);
   R__ResetLHC4CompressStats();

   std::vector<std::uint32_t> column(1024);
   for (std::size_t i = 0; i < column.size(); ++i)
      column[i] = static_cast<std::uint32_t>(i);
   const auto *data = reinterpret_cast<const char *>(column.data());
   const auto dataLen = column.size() * sizeof(std::uint32_t);
   const int compression = ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kLHC4, 5);
   auto zipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   auto szZipped = RNTupleCompressor::Zip(data, dataLen, compression, zipBuffer.get());
   EXPECT_LT(szZipped, dataLen);
   auto unzipBuffer = std::unique_ptr<char[]>(new char[dataLen]);
   RNTupleDecompressor::Unzip(zipBuffer.get(), szZipped, dataLen, unzipBuffer.get());
   EXPECT_EQ(0, std::memcmp(data, unzipBuffer.get(), dataLen));

   R__LHC4AutoCodecStatsSummary stats{};
   R__GetLHC4AutoCodecStatsSummary(&stats);
   EXPECT_GT(stats.auto_selections, 0u);
   EXPECT_EQ(stats.codec_hits[kLHC4CodecBeam], stats.auto_selections);
   for (int i = 0; i < kLHC4CodecCount; ++i) {
      if (i != kLHC4CodecBeam)
         EXPECT_EQ(0u, stats.codec_hits[i]) << "codec " << i;
   }

   R__SetLHC4Codec(kLHC4CodecBeam);
   R__SetLHC4AutoCodecs(0);
   R__SetLHC4AutoMinGainPct(1);
   R__SetLHC4Filters(0);
   EXPECT_EQ(kLHC4AutoCodecsRoot, R__GetLHC4AutoCodecs());
}
#endif
