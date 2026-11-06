// Tests for the RooFit JSON interface
// Authors: Jonas Rembser, CERN 12/2024

#include <RooFit/Detail/JSONInterface.h>

#include <TVectorD.h>

#include <gtest/gtest.h>

TEST(JSONInterface, MapsOfTVectorD)
{
   using RooFit::Detail::JSONNode;
   using RooFit::Detail::JSONTree;

   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &rootnode = tree->rootnode();

   rootnode.set_map();

   rootnode["map"] << std::map<std::string, TVectorD>{{"vec", TVectorD{3}}};
   rootnode["unordered_map"] << std::unordered_map<std::string, TVectorD>{{"vec", TVectorD{3}}};

   // For debugging:
   // std::stringstream ss;
   // rootnode.writeJSON(ss);
   // std::cout << ss.str() << std::endl;
}

// Values that are no containers have no children.
TEST(JSONInterface, ChildrenOfScalar)
{
   auto tree = RooFit::Detail::JSONTree::create(R"({"number": 5, "string": "s"})");
   for (const char *key : {"number", "string"}) {
      std::size_t nChildren = 0;
      for (auto const &child : tree->rootnode()[key].children()) {
         (void)child;
         ++nChildren;
      }
      EXPECT_EQ(nChildren, 0u) << key;
   }
}
