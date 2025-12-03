#include "gtest/gtest.h"

#include "TROOT.h"

#include <string>
#include <sstream>

TEST(TROOT, Version)
{
   ASSERT_TRUE(gROOT);

   std::string versionString = gROOT->GetVersion();
   EXPECT_EQ(7, versionString.size());

   std::stringstream versionStream(versionString);
   std::string buf;
   char del = '.';
   auto tokCounter = 0;
   const std::vector<size_t> refLength{1, 2, 2};
   while (getline(versionStream, buf, del)) {
      EXPECT_EQ(refLength[tokCounter++], buf.size());
   }
   EXPECT_EQ(3, tokCounter);
}