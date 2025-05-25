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
}
