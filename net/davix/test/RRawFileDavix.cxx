#include "RConfigure.h"
#include "ROOT/RRawFileDavix.hxx"

#include <string>

#include "gtest/gtest.h"

using namespace ROOT::Detail;

TEST(RRawFileDavix, Idle)
{
   // Test construction and destruction if the URL is never opened
   RRawFile::ROptions options;
   RRawFileDavix("http://root.cern.ch/files/davix.test", options);
}

TEST(RRawFileDavix, Basics)
{
   std::string line;
   RRawFile::ROptions options;
   options.fBlockSize = 1;
   std::unique_ptr<RRawFileDavix> f(new RRawFileDavix("http://root.cern.ch/files/davix.test", options));
   f->Readln(line);
   EXPECT_STREQ("Hello, World", line.c_str());
   EXPECT_EQ(13u, f->GetSize());

   std::unique_ptr<RRawFileDavix> f2(new RRawFileDavix("http://root.cern.ch/files/davix.test.404", options));
   EXPECT_THROW(f2->Readln(line), std::runtime_error);
}
