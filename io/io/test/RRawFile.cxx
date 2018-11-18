#include "RConfigure.h"
#include "ROOT/RRawFile.hxx"

#include "gtest/gtest.h"

using namespace ROOT::Detail;

TEST(RRawFile, CreateAndDestroy)
{
   RRawFile *f = RRawFile::Create("bla");
   delete f;
}
