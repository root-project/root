#include "ROOT/RTree.hxx"
#include "ROOT/RTreeModel.hxx"
#include "ROOT/RPageStorage.hxx"

#include "gtest/gtest.h"

#include <memory>
#include <utility>

using RInputTree = ROOT::Experimental::RInputTree;
using RTreeModel = ROOT::Experimental::RTreeModel;
using RPageSource = ROOT::Experimental::Detail::RPageSource;

TEST(RForestTree, Basics)
{
   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt");

   RInputTree tree(model, std::make_unique<RPageSource>("T"));
   RInputTree tree2(std::make_unique<RPageSource>("T"));
}
