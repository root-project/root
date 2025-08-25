#include "gtest/gtest.h"

#include "ROOT/RRawFileCurl.hxx"

using ROOT::Internal::RRawFile;
using ROOT::Internal::RRawFileCurl;

TEST(RRawFileCurl, Basics)
{
   RRawFile::ROptions options;
   RRawFileCurl("http://root.cern/files/davix.test", options);
}
