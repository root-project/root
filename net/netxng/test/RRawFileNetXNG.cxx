#include "RConfigure.h"
#include "ROOT/RRawFileNetXNG.hxx"

#include <string>
#include <utility>

#include "gtest/gtest.h"

using RRawFile = ROOT::Internal::RRawFile;
using RRawFileNetXNG = ROOT::Internal::RRawFileNetXNG;

TEST(RRawFileNetXNG, Idle)
{
   // Test construction and destruction if the URL is never opened
   RRawFile::ROptions options;
   RRawFileNetXNG("root://eospublic.cern.ch//eos/root-eos/testfiles/xrootd.test", options);
}

TEST(RRawFileNetXNG, Basics)
{
   std::string line;
   RRawFile::ROptions options;
   auto f = std::make_unique<RRawFileNetXNG>("root://eospublic.cern.ch//eos/root-eos/testfiles/xrootd.test", options);
   f->Readln(line);
   EXPECT_STREQ("This file is used in the RRawFile unit tests", line.c_str());
   EXPECT_EQ(45u, f->GetSize());

   auto f2 = std::make_unique<RRawFileNetXNG>("root://eospublic.cern.ch//eos/root-eos/testfiles/xrootd.NOENT", options);
   EXPECT_THROW(f2->Readln(line), std::runtime_error);
}


TEST(RRawFileNetXNG, Eof)
{
   char tail[4];
   tail[3] = '\0';
   RRawFile::ROptions options;
   auto f = std::make_unique<RRawFileNetXNG>("root://eospublic.cern.ch//eos/root-eos/testfiles/xrootd.test", options);
   auto nbytes = f->ReadAt(tail, 10, f->GetSize() - 3);
   EXPECT_EQ(3u, nbytes);
   EXPECT_STREQ("ts\n", tail);
}


TEST(RRawFileNetXNG, ReadV)
{
   RRawFile::ROptions options;
   options.fBlockSize = 0;
   auto f = std::make_unique<RRawFileNetXNG>("root://eospublic.cern.ch//eos/root-eos/testfiles/xrootd.test", options);

   auto iovLimits = f->GetReadVLimits();
   EXPECT_EQ(static_cast<std::uint64_t>(-1), iovLimits.fMaxTotalSize);
   EXPECT_TRUE(iovLimits.HasReqsLimit());
   EXPECT_TRUE(iovLimits.HasSizeLimit());

   char buffer[2];
   buffer[0] = buffer[1] = 0;
   RRawFile::RIOVec iovec[2];
   iovec[0].fBuffer = &buffer[0];
   iovec[0].fOffset = 0;
   iovec[0].fSize = 1;
   iovec[1].fBuffer = &buffer[1];
   iovec[1].fOffset = 43;
   iovec[1].fSize = 1;
   f->ReadV(iovec, 2);

   EXPECT_EQ(1U, iovec[0].fOutBytes);
   EXPECT_EQ(1U, iovec[1].fOutBytes);
   EXPECT_EQ('T', buffer[0]);
   EXPECT_EQ('s', buffer[1]);
}
