#include "gtest/gtest.h"

#include <ROOT/RNTupleZip.hxx>

#include <memory>
#include <string>
#include <utility>

using RNTupleCompressor = ROOT::Experimental::Detail::RNTupleCompressor;
using RNTupleDecompressor = ROOT::Experimental::Detail::RNTupleDecompressor;


TEST(RNTupleZip, Basics)
{
   RNTupleCompressor compressor;
   RNTupleDecompressor decompressor;

   std::string data = "xxxxxxxxxxxxxxxxxxxxxxxx";
   auto szZipped = compressor(data.data(), data.length(), 101);
   EXPECT_LT(szZipped, data.length());
   char unzipBuffer[data.length()];
   decompressor(compressor.GetZipBuffer(), szZipped, data.length(), unzipBuffer);
   EXPECT_EQ(data, std::string(unzipBuffer, data.length()));

   // inplace decompression
   unsigned char zipBuffer[data.length()];
   memcpy(zipBuffer, compressor.GetZipBuffer(), szZipped);
   decompressor(zipBuffer, szZipped, data.length());
   EXPECT_EQ(data, std::string(reinterpret_cast<char *>(zipBuffer), data.length()));
}


TEST(RNTupleZip, Empty)
{
   RNTupleCompressor compressor;

   char x;
   EXPECT_EQ(0U, compressor(&x, 0, 0));
   EXPECT_EQ(0U, compressor(&x, 0, 101));

   // Don't crash
   RNTupleDecompressor()(&x, 0, 0, &x);
}


TEST(RNTupleZip, Uncompressed)
{
   RNTupleCompressor compressor;
   char X = 'x';
   EXPECT_EQ(1U, compressor(&X, 1, 0));
   RNTupleDecompressor()(compressor.GetZipBuffer(), 1, 1, &X);
   EXPECT_EQ('x', X);
}


TEST(RNTupleZip, Small)
{
   RNTupleCompressor compressor;
   char X = 'x';
   char x = 0;
   EXPECT_EQ(1U, compressor(&X, 1, 101));
   RNTupleDecompressor()(compressor.GetZipBuffer(), 1, 1, &X);
   EXPECT_EQ('x', X);
}
