#include "gtest/gtest.h"

#include <ROOT/RNTupleImporter.hxx>

TEST(RNTupleImporter, Basics)
{
   auto importer = ROOT::Experimental::RNTupleImporter::Create("", "", "");
}
