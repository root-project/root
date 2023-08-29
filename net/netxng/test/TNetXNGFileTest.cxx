#include "RConfigure.h"

#include <string>

#include "gtest/gtest.h"

#include "TFile.h"

const char *fileName = "root://eospublic.cern.ch//eos/root-eos/h1/dstarmb.root";

TEST(TNetXNGFileTest, Plugin)
{
   auto f = TFile::Open(fileName);
   EXPECT_TRUE(f != nullptr);
   EXPECT_TRUE(f->IsOpen());
   EXPECT_FALSE(f->IsZombie());
   EXPECT_STREQ("TNetXNGFile", f->ClassName());
   delete f;
}

TEST(TNetXNGFileTest, DoubleClose)
{
   auto f = TFile::Open(fileName);
   EXPECT_TRUE(f != nullptr);
   EXPECT_FALSE(f->IsZombie());
   EXPECT_TRUE(f->IsOpen());
   EXPECT_STREQ("TNetXNGFile", f->ClassName());
   f->Close();
   EXPECT_FALSE(f->IsOpen());
   f->Close();
   EXPECT_FALSE(f->IsOpen());
   delete f;
}
