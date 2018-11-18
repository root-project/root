#include "RConfigure.h"
#include "ROOT/RRawFile.hxx"

#include <memory>

#include "gtest/gtest.h"

using namespace ROOT::Detail;

TEST(RRawFile, CreateAndDestroy)
{
   std::unique_ptr<RRawFile> f(RRawFile::Create("file:///dev/null"));
   EXPECT_EQ(0u, f->GetSize());
}
