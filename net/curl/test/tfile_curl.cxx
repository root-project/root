#include "gtest/gtest.h"

#include "TCurlFile.h"

#include <cstdio>
#include <memory>
#include <utility>

TEST(TCurlFile, GetSize)
{
   auto f = std::make_unique<TCurlFile>("https://root.cern.ch/files/tutorials/hsimple.root");
   EXPECT_LT(0, f->GetSize());
}

TEST(TCurlFile, Read)
{
   auto f = std::make_unique<TCurlFile>("https://root.cern.ch/files/tutorials/hsimple.root");
   auto obj = f->Get("hpx");
   EXPECT_TRUE(obj != nullptr);
}

TEST(TCurlFile, Cp)
{
   const char *localPath = "root_test_curl_file_cp.root";
   auto f = std::make_unique<TCurlFile>("https://root.cern.ch/files/tutorials/hsimple.root");
   EXPECT_TRUE(f->Cp(localPath, false /* progressbar */));
   remove(localPath);
}
