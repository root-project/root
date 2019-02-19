#include "ROOT/RTree.hxx"
#include "ROOT/RTreeModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include <TClass.h>
#include <TFile.h>

#include "gtest/gtest.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

using RInputTree = ROOT::Experimental::RInputTree;
using ROutputTree = ROOT::Experimental::ROutputTree;
using RTreeModel = ROOT::Experimental::RTreeModel;
using RPageSource = ROOT::Experimental::Detail::RPageSource;
using RPageSinkRoot = ROOT::Experimental::Detail::RPageSinkRoot;
using RPageSourceRoot = ROOT::Experimental::Detail::RPageSourceRoot;
using RTreeFieldBase = ROOT::Experimental::Detail::RTreeFieldBase;

TEST(RForestTree, Basics)
{
   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt");

   //RInputTree tree(model, std::make_unique<RPageSource>("T"));
   //RInputTree tree2(std::make_unique<RPageSource>("T"));
}

TEST(RForestTree, ReconstructModel)
{
   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settingsWrite;
   settingsWrite.fFile = file;
   settingsWrite.fTakeOwnership = true;

   auto model = std::make_shared<RTreeModel>();
   auto fieldPt = model->AddField<float>("pt", 42.0);
   auto fieldNnlo = model->AddField<std::vector<std::vector<float>>>("nnlo");
   auto fieldKlass = model->AddField<ROOT::Experimental::RForestTest>("klass");
   {
      RPageSinkRoot sinkRoot("myTree", settingsWrite);
      sinkRoot.Create(model.get());
      sinkRoot.CommitDataset();
   }

   file = TFile::Open("test.root", "READ");
   RPageSourceRoot::RSettings settingsRead;
   settingsRead.fFile = file;
   settingsRead.fTakeOwnership = true;
   RPageSourceRoot sourceRoot("myTree", settingsRead);
   sourceRoot.Attach();

   auto modelReconstructed = sourceRoot.GenerateModel();
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
   auto fieldEnergy = model->AddField<float>("energy", 7.0);
   auto fieldTag = model->AddField<std::string>("tag", "xyz");
   auto fieldJets = model->AddField<std::vector<float>>("jets");
   fieldJets->push_back(1.0);
   fieldJets->push_back(2.0);
   auto fieldNnlo = model->AddField<std::vector<std::vector<float>>>("nnlo");
   fieldNnlo->push_back(std::vector<float>());
   fieldNnlo->push_back(std::vector<float>{1.0});
   fieldNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
   auto fieldKlass = model->AddField<ROOT::Experimental::RForestTest>("klass");
   fieldKlass->s = "abc";

   {
      ROutputTree tree(model, std::make_unique<RPageSinkRoot>("myTree", settingsWrite));
      tree.Fill();
   }

   *fieldPt = 0.0;
   *fieldEnergy = 0.0;
   fieldTag->clear();
   fieldJets->clear();
   fieldNnlo->clear();
   fieldKlass->s.clear();

   file = TFile::Open("test.root", "READ");
   RPageSourceRoot::RSettings settingsRead;
   settingsRead.fFile = file;
   settingsRead.fTakeOwnership = true;
   RInputTree tree(model, std::make_unique<RPageSourceRoot>("myTree", settingsRead));
   EXPECT_EQ(1U, tree.GetNEntries());
   tree.GetEntry(0);

   EXPECT_EQ(42.0, *fieldPt);
   EXPECT_EQ(7.0, *fieldEnergy);
   EXPECT_STREQ("xyz", fieldTag->c_str());

   EXPECT_EQ(2U, fieldJets->size());
   EXPECT_EQ(1.0, (*fieldJets)[0]);
   EXPECT_EQ(2.0, (*fieldJets)[1]);

   EXPECT_EQ(3U, fieldNnlo->size());
   EXPECT_EQ(0U, (*fieldNnlo)[0].size());
   EXPECT_EQ(1U, (*fieldNnlo)[1].size());
   EXPECT_EQ(4U, (*fieldNnlo)[2].size());
   EXPECT_EQ(1.0, (*fieldNnlo)[1][0]);
   EXPECT_EQ(1.0, (*fieldNnlo)[2][0]);
   EXPECT_EQ(2.0, (*fieldNnlo)[2][1]);
   EXPECT_EQ(4.0, (*fieldNnlo)[2][2]);
   EXPECT_EQ(8.0, (*fieldNnlo)[2][3]);

   EXPECT_STREQ("abc", fieldKlass->s.c_str());
}

TEST(RForestTree, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RTreeField<float>::MyTypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RTreeField<std::vector<std::string>>::MyTypeName().c_str());
   EXPECT_STREQ("ROOT::Experimental::RForestTest",
                ROOT::Experimental::RTreeField<ROOT::Experimental::RForestTest>::MyTypeName().c_str());
}

namespace {
class RNoDictionary {};
} // namespace

TEST(RForestTree, TClass) {
   auto modelFail = std::make_shared<RTreeModel>();
   EXPECT_THROW(modelFail->AddField<RNoDictionary>("nodict"), std::runtime_error);

   auto model = std::make_shared<RTreeModel>();
   auto ptrKlass = model->AddField<ROOT::Experimental::RForestTest>("klass");

   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settingsWrite;
   settingsWrite.fFile = file;
   settingsWrite.fTakeOwnership = true;
   ROutputTree tree(model, std::make_unique<RPageSinkRoot>("myTree", settingsWrite));
}
