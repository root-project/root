#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <ROOT/RError.hxx>
#include <ROOT/RNTupleZip.hxx>

#include <RZip.h>

#include <cstring>

using ROOT::Internal::RNTupleCompressor;
using ROOT::Internal::RNTupleDecompressor;

TEST(RZip, HeaderBasics)
{
   unsigned char header[9] = {'Z', 'S', '\1', 1, 0, 0, 2, 1, 0};
   int srcsize = 0;
   int tgtsize = 0;

   EXPECT_EQ(0, R__unzip_header(&srcsize, header, &tgtsize));
   EXPECT_EQ(10, srcsize);
   EXPECT_EQ(258, tgtsize);
}

TEST(RZip, CorruptHeaderRNTuple)
{
   constexpr char content[] = {7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
   unsigned char blocks[50];
   unsigned char verify[50];
   static_assert(sizeof(content) < sizeof(blocks));

   const auto sz1 = RNTupleCompressor::Zip(content, sizeof(content), 101, blocks);
   EXPECT_LT(sz1, sizeof(content));

   RNTupleDecompressor::Unzip(blocks, sz1, sizeof(content), verify);
   EXPECT_EQ(0, memcmp(content, verify, sizeof(content)));

   try {
      RNTupleDecompressor::Unzip(blocks, 8, sizeof(content), verify);
      FAIL() << "too short input buffer should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("zip buffer too short"));
   }

   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 1, sizeof(content), verify);
      FAIL() << "too long input buffer should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unexpected trailing bytes in zip buffer"));
   }

   blocks[6]++;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1, sizeof(content) + 1, verify);
      FAIL() << "too long target size should throw";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unexpected length after unzipping the buffer"));
   }
   blocks[6]--;

   EXPECT_LT(sz1, sizeof(blocks) + 10);
   blocks[sz1] = 'Z';
   blocks[sz1 + 1] = 'S';
   blocks[sz1 + 2] = '\1';
   blocks[sz1 + 3] = 0;
   blocks[sz1 + 4] = 0;
   blocks[sz1 + 5] = 0;
   blocks[sz1 + 6] = 0;
   blocks[sz1 + 7] = 0;
   blocks[sz1 + 8] = 0;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "source length zero in the header should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 1;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "source size < target size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 11;
   blocks[sz1 + 6] = 12;
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "too big source size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }

   blocks[sz1 + 3] = 1;
   blocks[sz1 + 6] = sizeof(verify);
   try {
      RNTupleDecompressor::Unzip(blocks, sz1 + 10, sizeof(verify), verify);
      FAIL() << "too big target size should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("failed to unzip buffer header"));
   }
}
