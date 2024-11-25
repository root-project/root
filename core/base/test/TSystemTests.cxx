#include "gtest/gtest.h"

#include "TSystem.h"
#include "TString.h"

#include <string>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <streambuf>

TEST(TSystem, TempFile)
{
   TString fname = "root_test_";
   auto ftmp = gSystem->TempFileName(fname);

   EXPECT_TRUE(fname.Length() > 10);
   EXPECT_TRUE(ftmp != nullptr);

   std::string content = "test_temp_file_content";
   auto res_write = fwrite(content.data(), 1, content.length(), ftmp);
   EXPECT_EQ(res_write, content.length());

   auto res_close = fclose(ftmp);
   EXPECT_EQ(res_close, 0);

   std::ifstream fread(fname.Data());
   std::string str((std::istreambuf_iterator<char>(fread)), std::istreambuf_iterator<char>());
   EXPECT_STREQ(content.c_str(), str.c_str());

   gSystem->Unlink(fname);
}

TEST(TSystem, TempFileSuffix)
{
   TString fname = "root_suffix_test_";
   const char *suffix = ".txt";
   auto ftmp = gSystem->TempFileName(fname, nullptr, suffix);

   EXPECT_TRUE(fname.Length() > 16);
   EXPECT_TRUE(ftmp != nullptr);

   // check that suffix really at the end of the file name
   EXPECT_STREQ(fname(fname.Length() - strlen(suffix), strlen(suffix)).Data(), suffix);

   std::string content = "test_temp_file_content_suffix";
   auto res_write = fwrite(content.data(), 1, content.length(), ftmp);
   EXPECT_EQ(res_write, content.length());

   auto res_close = fclose(ftmp);
   EXPECT_EQ(res_close, 0);

   std::ifstream fread(fname.Data());
   std::string str((std::istreambuf_iterator<char>(fread)), std::istreambuf_iterator<char>());
   EXPECT_STREQ(content.c_str(), str.c_str());

   gSystem->Unlink(fname);
}

TEST(TSystem, CryptoRandom)
{
   // test with 512 bits, longer keys may not work

   const int len = 64;
   uint8_t buf[64];

   for (int n = 0; n < len; n++)
      buf[n] = 0;

   auto res = gSystem->GetCryptoRandom(buf, len);

   EXPECT_EQ(res, len);

   int nmatch = 0;

   for (int n = 0; n < len; n++)
      if (buf[n] == 0)
         nmatch++;

   // check that values in buffer changed
   EXPECT_TRUE(nmatch != len);

   for (int n = 0; n < len; n++)
      buf[n] = n;

   res = gSystem->GetCryptoRandom(buf, len);

   EXPECT_EQ(res, len);

   nmatch = 0;

   for (int n = 0; n < len; n++)
      if (buf[n] == n)
         nmatch++;

   // check that values in buffer changed
   EXPECT_TRUE(nmatch != len);
}
