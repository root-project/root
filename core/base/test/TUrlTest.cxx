#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TUrl.h"

TEST(TUrl, FilePath)
{
   TUrl u1("/tmp/a.root", kTRUE);
   EXPECT_TRUE(u1.IsValid());

   //https://its.cern.ch/jira/browse/ROOT-5820
   TUrl u2("/tmp/a.root:/", kTRUE); // TFile.GetPath() returns a trailing :/
   EXPECT_TRUE(u2.IsValid());

   // ROOT-5430
   const char * ref_5430 = "file:///tmp/t.root";
   EXPECT_STREQ(TUrl("/tmp/t.root").GetUrl(), ref_5430);
   EXPECT_STREQ(TUrl("//tmp/t.root").GetUrl(), ref_5430);
   EXPECT_STREQ(TUrl("///tmp/t.root").GetUrl(), ref_5430);
}
