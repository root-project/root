#include "gtest/gtest.h"

#include "TCurlFile.h"
#include "TObject.h"

#include <memory>
#include <utility>

TEST(TCurlFile, GetSize)
{
   auto f = std::make_unique<TCurlFile>("https://root.cern.ch/files/galaxy.root");
   EXPECT_LT(0, f->GetSize());
}

TEST(TCurlFile, Read)
{
   auto f = std::make_unique<TCurlFile>("https://root.cern.ch/files/galaxy.root");
   auto obj = f->Get("n4254");
   EXPECT_TRUE(obj != nullptr);
}
