#include "RConfigure.h"
#include "ROOT/RRawFileNetXNG.hxx"

#include <string>

#include "gtest/gtest.h"

using RRawFile = ROOT::Internal::RRawFile;
using RRawFileNetXNG = ROOT::Internal::RRawFileNetXNG;

TEST(RRawFileNetXNG, Idle)
{
   // Test construction and destruction if the URL is never opened
   RRawFile::ROptions options;
   RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/xrootd.test", options);
}

TEST(RRawFileNetXNG, Basics)
{
   std::string line;
   RRawFile::ROptions options;
   std::unique_ptr<RRawFileNetXNG> f(new RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/xrootd.test", options));
   f->Readln(line);
   EXPECT_STREQ("Hello, World", line.c_str());
   EXPECT_EQ(13u, f->GetSize());

   std::unique_ptr<RRawFileNetXNG> f2(new RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/xrootd.test.ENOENT", options));
   EXPECT_THROW(f2->Readln(line), std::runtime_error);
}


TEST(RRawFileNetXNG, Eof)
{
   char tail[4];
   tail[3] = '\0';
   RRawFile::ROptions options;
   std::unique_ptr<RRawFileNetXNG> f(new RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/xrootd.test", options));
   auto nbytes = f->ReadAt(tail, 10, f->GetSize() - 3);
   EXPECT_EQ(3u, nbytes);
   EXPECT_STREQ("ld\n", tail);
}


TEST(RRawFileNetXNG, ReadV)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   std::unique_ptr<RRawFileNetXNG> f(new RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/xrootd.test", options));

   char buffer[2];
   buffer[0] = buffer[1] = 0;
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
