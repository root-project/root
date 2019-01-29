#include "ROOT/RTree.hxx"
#include "ROOT/RTreeModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include "TFile.h"  // Remove me

#include "gtest/gtest.h"

#include <memory>
#include <utility>

using RInputTree = ROOT::Experimental::RInputTree;
using RTreeModel = ROOT::Experimental::RTreeModel;
using RPageSource = ROOT::Experimental::Detail::RPageSource;
using RPageSinkRoot = ROOT::Experimental::Detail::RPageSinkRoot;

TEST(RForestTree, Basics)
{
   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt");

   RInputTree tree(model, std::make_unique<RPageSource>("T"));
   RInputTree tree2(std::make_unique<RPageSource>("T"));
}

TEST(RForestTree, StorageRoot)
{
   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settings;
   settings.fFile = file;
   RPageSinkRoot sinkRoot("myTree", settings);

   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt");

   sinkRoot.Create(*model);
   file->Close();
}


TEST(RForestTree, RemoveMe)
{
   //TFile
}
