#include "gtest/gtest.h"

#include <ROOT/RNTupleImporter.hxx>

TEST(RNTupleImporter, Basics)
{
   auto importer =
      ROOT::Experimental::RNTupleImporter::Create("/data/B2HHH~zstd.root", "DecayTree", "test.root").Unwrap();
   importer->Import();
}
