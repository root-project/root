#include "ROOT/RTree.hxx"
#include "ROOT/RTreeModel.hxx"
#include "ROOT/RTreeStorage.hxx"

#include "gtest/gtest.h"

#include <memory>
#include <utility>

using RInputTree = ROOT::Experimental::RInputTree;
using RTreeModel = ROOT::Experimental::RTreeModel;
using RTreeSource = ROOT::Experimental::Detail::RTreeSource;

TEST(RForestTree, Basics)
{
   auto model = std::make_shared<RTreeModel>();
   RInputTree tree(model, std::make_unique<RTreeSource>("T"));
   RInputTree tree2(std::make_unique<RTreeSource>("T"));
}
