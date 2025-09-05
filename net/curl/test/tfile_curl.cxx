#include "gtest/gtest.h"

#include "TCurlFile.h"

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
