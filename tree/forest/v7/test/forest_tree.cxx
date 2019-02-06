#include "ROOT/RTree.hxx"
#include "ROOT/RTreeModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include "TFile.h"  // Remove me

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <utility>

using RInputTree = ROOT::Experimental::RInputTree;
using ROutputTree = ROOT::Experimental::ROutputTree;
using RTreeModel = ROOT::Experimental::RTreeModel;
using RPageSource = ROOT::Experimental::Detail::RPageSource;
using RPageSinkRoot = ROOT::Experimental::Detail::RPageSinkRoot;
using RPageSourceRoot = ROOT::Experimental::Detail::RPageSourceRoot;

TEST(RForestTree, Basics)
{
   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt");

   //RInputTree tree(model, std::make_unique<RPageSource>("T"));
   //RInputTree tree2(std::make_unique<RPageSource>("T"));
}

TEST(RForestTree, StorageRoot)
{
   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settingsWrite;
   settingsWrite.fFile = file;
   RPageSinkRoot sinkRoot("myTree", settingsWrite);

   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt", 42.0);
   auto fieldX = model->AddField<float>("energy");
   auto fieldStr = model->AddField<std::string>("string", "abc");

   //auto fieldFail = model->AddField<int>("jets");
   auto fieldJet = model->AddField<std::vector<float>>("jets" /* TODO(jblomer), {1.0, 2.0}*/);
   auto nnlo = model->AddField<std::vector<std::vector<float>>>("nnlo");

   sinkRoot.Create(model.get());
   sinkRoot.CommitDataset();
   file->Close();

   file = TFile::Open("test.root", "READ");
   RPageSourceRoot::RSettings settingsRead;
   settingsRead.fFile = file;
   RPageSourceRoot sourceRoot("myTree", settingsRead);
   sourceRoot.Attach();
   file->Close();
}


TEST(RForestTree, WriteRead)
{
   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settingsWrite;
   settingsWrite.fFile = file;
   settingsWrite.fTakeOwnership = true;

   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt", 42.0);
   auto fieldTag = model->AddField<std::string>("tag", "xyz");

   {
      ROutputTree tree(model, std::make_unique<RPageSinkRoot>("myTree", settingsWrite));
      tree.Fill();
   }

   *fieldPt = 0.0;
   fieldTag->clear();

   file = TFile::Open("test.root", "READ");
   RPageSourceRoot::RSettings settingsRead;
   settingsRead.fFile = file;
   settingsRead.fTakeOwnership = true;
   RInputTree tree(model, std::make_unique<RPageSourceRoot>("myTree", settingsRead));
   EXPECT_EQ(1U, tree.GetNEntries());
   tree.GetEntry(0);
   EXPECT_EQ(42.0, *fieldPt);
   EXPECT_STREQ("xyz", fieldTag->c_str());
}
