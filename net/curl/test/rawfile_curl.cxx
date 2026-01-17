#include "gtest/gtest.h"

#include "ROOT/RCurlConnection.hxx"
#include "ROOT/RRawFileCurl.hxx"

#include <algorithm>

using ROOT::Internal::RRawFile;
using ROOT::Internal::RRawFileCurl;

TEST(RRawFileCurl, Stat)
{
   RRawFile::ROptions options;
   RRawFileCurl f("https://root.cern/files/davix.test", options);
   EXPECT_EQ(13u, f.GetSize());
}

TEST(RRawFileCurl, Redirect)
{
   RRawFile::ROptions options;
   // root.cern redirects HTTP requests to HTTPS
   RRawFileCurl f("http://root.cern/files/davix.test", options);
   EXPECT_EQ(13u, f.GetSize());
}

TEST(RRawFileCurl, Basics)
{
   std::string line;
   RRawFile::ROptions options;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("https://root.cern/files/davix.test", options));
   f->Readln(line);
   EXPECT_STREQ("Hello, World", line.c_str());

   std::unique_ptr<RRawFileCurl> f2(new RRawFileCurl("https://root.cern/files/davix.test.404", options));
   EXPECT_THROW(f2->Readln(line), std::runtime_error);
}

TEST(RRawFileCurl, Eof)
{
   char tail[4];
   tail[3] = '\0';
   RRawFile::ROptions options;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("https://root.cern/files/davix.test", options));
   auto nbytes = f->ReadAt(tail, 10, f->GetSize() - 3);
   EXPECT_EQ(3u, nbytes);
   EXPECT_STREQ("ld\n", tail);

   // Read past end of the file
   nbytes = f->ReadAt(nullptr, 1, f->GetSize());
   EXPECT_EQ(0u, nbytes);
}

TEST(RRawFileCurl, ReadV)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   char buffer[2] = {0, 0};
   RRawFile::RIOVec iovec[2];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 0;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = &buffer[1];
   iovec[1].fOffset = 11;
   iovec[1].fSize = 1;
   f->ReadV(iovec, 2);

   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(1U, iovec[1].fOutBytes);
   EXPECT_EQ('H', buffer[0]);
   EXPECT_EQ('d', buffer[1]);
}

TEST(RRawFileCurl, ReadVUnsorted)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   char buffer[4] = {0, 0, 0, 0};
   RRawFile::RIOVec iovec[4];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 10;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = &buffer[1];
   iovec[1].fOffset = 0;
   iovec[1].fSize = 1;
   iovec[2].fBuffer = &buffer[2];
   iovec[2].fOffset = 11;
   iovec[2].fSize = 3;
   iovec[3].fBuffer = &buffer[3];
   iovec[3].fOffset = 15;
   iovec[3].fSize = 1;
   f->ReadV(iovec, 4);

   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(1U, iovec[1].fOutBytes);
   EXPECT_EQ(2U, iovec[2].fOutBytes);
   EXPECT_EQ(0U, iovec[3].fOutBytes);
   EXPECT_EQ("lHd\n", std::string(buffer, sizeof(buffer)));
}

TEST(RRawFileCurl, ReadVEmpty)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   // Should not crash
   f->ReadV(nullptr, 0);

   RRawFile::RIOVec iovec[4];
   iovec[0].fBuffer = nullptr;
   iovec[0].fOffset = 0;
   iovec[0].fSize = 0;
   iovec[1].fBuffer = nullptr;
   iovec[1].fOffset = 0;
   iovec[1].fSize = 0;
   f->ReadV(iovec, 2); // Don't crash
   EXPECT_EQ(0U, iovec[0].fOutBytes);
   EXPECT_EQ(0U, iovec[1].fOutBytes);

   char buffer[2] = {0, 0};
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 0;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = nullptr;
   iovec[1].fOffset = 1;
   iovec[1].fSize = 0;
   iovec[2].fBuffer = nullptr;
   iovec[2].fOffset = 1;
   iovec[2].fSize = 0;
   iovec[3].fBuffer = &buffer[1];
   iovec[3].fOffset = 2;
   iovec[3].fSize = 1;
   f->ReadV(iovec, 4);
   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(0U, iovec[1].fOutBytes);
   EXPECT_EQ(0U, iovec[2].fOutBytes);
   EXPECT_EQ(1U, iovec[3].fOutBytes);
   EXPECT_EQ('H', buffer[0]);
   EXPECT_EQ('l', buffer[1]);

   // In this case, the 4 requests can be coalesced into a single combined range that gets returned
   iovec[3].fBuffer = &buffer[1];
   iovec[3].fOffset = 1;
   iovec[3].fSize = 1;
   f->ReadV(iovec, 4);
   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(0U, iovec[1].fOutBytes);
   EXPECT_EQ(0U, iovec[2].fOutBytes);
   EXPECT_EQ(1U, iovec[3].fOutBytes);
   EXPECT_EQ('H', buffer[0]);
   EXPECT_EQ('e', buffer[1]);
}

TEST(RRawFileCurl, Overlap1)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   char buffer[15];
   std::fill(std::begin(buffer), std::end(buffer), 0);
   RRawFile::RIOVec iovec[5];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 1;
   iovec[0].fSize = 3;
   iovec[1].fBuffer = &buffer[3];
   iovec[1].fOffset = 2;
   iovec[1].fSize = 3;
   iovec[2].fBuffer = &buffer[6];
   iovec[2].fOffset = 3;
   iovec[2].fSize = 3;
   iovec[3].fBuffer = &buffer[9];
   iovec[3].fOffset = 4;
   iovec[3].fSize = 3;
   iovec[4].fBuffer = &buffer[12];
   iovec[4].fOffset = 5;
   iovec[4].fSize = 3;
   f->ReadV(iovec, 5);
   EXPECT_EQ(3U, iovec[0].fOutBytes);
   EXPECT_EQ(3U, iovec[1].fOutBytes);
   EXPECT_EQ(3U, iovec[2].fOutBytes);
   EXPECT_EQ(3U, iovec[3].fOutBytes);
   EXPECT_EQ(3U, iovec[4].fOutBytes);
   EXPECT_EQ("ellllolo,o, , W", std::string(buffer, sizeof(buffer)));
}

TEST(RRawFileCurl, Overlap2)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   char buffer[14];
   std::fill(std::begin(buffer), std::end(buffer), 0);
   RRawFile::RIOVec iovec[5];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 0;
   iovec[0].fSize = 5;
   iovec[1].fBuffer = &buffer[5];
   iovec[1].fOffset = 1;
   iovec[1].fSize = 1;
   iovec[2].fBuffer = nullptr;
   iovec[2].fOffset = 2;
   iovec[2].fSize = 0;
   iovec[3].fBuffer = &buffer[6];
   iovec[3].fOffset = 4;
   iovec[3].fSize = 3;
   iovec[4].fBuffer = &buffer[9];
   iovec[4].fOffset = 3;
   iovec[4].fSize = 5;
   f->ReadV(iovec, 5);
   EXPECT_EQ(5U, iovec[0].fOutBytes);
   EXPECT_EQ(1U, iovec[1].fOutBytes);
   EXPECT_EQ(0U, iovec[2].fOutBytes);
   EXPECT_EQ(3U, iovec[3].fOutBytes);
   EXPECT_EQ(5U, iovec[4].fOutBytes);
   EXPECT_EQ("Helloeo, lo, W", std::string(buffer, sizeof(buffer)));
}

TEST(RRawFileCurl, SmallMaxRanges)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));
   f->GetConnection().SetMaxNRangesPerRequest(1);

   char buffer[3] = {0, 0, 0};
   RRawFile::RIOVec iovec[3];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 10;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = &buffer[1];
   iovec[1].fOffset = 12;
   iovec[1].fSize = 1;
   iovec[2].fBuffer = &buffer[2];
   iovec[2].fOffset = 14;
   iovec[2].fSize = 1;
   f->ReadV(iovec, 3);
   EXPECT_EQ(1u, iovec[0].fOutBytes);
   EXPECT_EQ(1u, iovec[1].fOutBytes);
   EXPECT_EQ(0u, iovec[2].fOutBytes);
   EXPECT_EQ("l\n", std::string(buffer, 2));
}

TEST(RRawFileCurl, ManyRanges)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileCurl> f(new RRawFileCurl("http://root.cern/files/davix.test", options));

   constexpr unsigned int N = 1000;
   char buffer[N];
   std::fill(std::begin(buffer), std::end(buffer), 0);
   RRawFile::RIOVec iovec[N];
   for (unsigned int i = 0; i < N; ++i) {
      iovec[i].fBuffer = &buffer[i];
      iovec[i].fOffset = 2 * i;
      iovec[i].fSize = 1;
   }
   f->ReadV(iovec, N);
   EXPECT_NE(0u, f->GetConnection().GetMaxNRangesPerRequest());
   EXPECT_LT(f->GetConnection().GetMaxNRangesPerRequest(), N);
   unsigned int i = 0;
   for (; i < 7; ++i)
      EXPECT_EQ(1u, iovec[i].fOutBytes);
   for (; i < N; ++i)
      EXPECT_EQ(0u, iovec[i].fOutBytes);
}
