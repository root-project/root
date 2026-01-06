#include "gtest/gtest.h"

#include "TROOT.h"

#include <filesystem>
#include <sstream>
#include <string>
#include <iostream>

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

// TROOT::GetSharedLibDir() is fundamental to resolve all the directories
// relevant for ROOT, because it can be inferred without environment variables
// like ROOTSYS by locating libCore, which is loaded by definition when using
// ROOT. Therefore, GetSharedLibDir() serves as an anchor to resolve all other
// directories, using the correct relative paths for either the build or
// install tree. Given this fundamental role, we need to test that it works.
TEST(TROOT, GetSharedLibDir)
{
   namespace fs = std::filesystem;

   // Use std::filesystem for automatic path normalization.
   fs::path libDir = gROOT->GetSharedLibDir().Data();
   fs::path libDirRef = EXPECTED_SHARED_LIBRARY_DIR;

   EXPECT_EQ(libDir, libDirRef);
}
