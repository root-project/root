#include "ROOT/RForest.hxx"
#include "ROOT/RForestModel.hxx"
#include "ROOT/RPageStorage.hxx"
#include "ROOT/RPageStorageRoot.hxx"

#include <TClass.h>
#include <TFile.h>
#include <TRandom3.h>

#include "gtest/gtest.h"

#include <exception>
#include <memory>
#include <string>
#include <utility>

using RInputForest = ROOT::Experimental::RInputForest;
using ROutputForest = ROOT::Experimental::ROutputForest;
using RForestModel = ROOT::Experimental::RForestModel;
using RPageSource = ROOT::Experimental::Detail::RPageSource;
using RPageSinkRoot = ROOT::Experimental::Detail::RPageSinkRoot;
using RPageSourceRoot = ROOT::Experimental::Detail::RPageSourceRoot;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;

TEST(RForest, Basics)
{
   auto model = std::make_shared<RForestModel>();
   auto fieldPt = model->AddField<float>("pt");

   //RInputTree tree(model, std::make_unique<RPageSource>("T"));
   //RInputTree tree2(std::make_unique<RPageSource>("T"));
}

TEST(RForest, ReconstructModel)
{
   auto model = std::make_shared<RForestModel>();
   auto fieldPt = model->AddField<float>("pt", 42.0);
   auto fieldNnlo = model->AddField<std::vector<std::vector<float>>>("nnlo");
   auto fieldKlass = model->AddField<ROOT::Experimental::RForestTest>("klass");
   {
      RPageSinkRoot sinkRoot("myTree", "test.root");
      sinkRoot.Create(model.get());
      sinkRoot.CommitDataset();
   }

   RPageSourceRoot sourceRoot("myTree", "test.root");
   sourceRoot.Attach();

   auto modelReconstructed = sourceRoot.GenerateModel();
   EXPECT_EQ(nullptr, modelReconstructed->GetDefaultEntry()->Get<float>("xyz"));
   auto vecPtr = modelReconstructed->GetDefaultEntry()->Get<std::vector<std::vector<float>>>("nnlo");
   EXPECT_TRUE(vecPtr != nullptr);
   // Don't crash
   vecPtr->push_back(std::vector<float>{1.0});
}

TEST(RForest, StorageRoot)
{
   TFile *file = TFile::Open("test.root", "RECREATE");
   RPageSinkRoot::RSettings settingsWrite;
   settingsWrite.fFile = file;
   RPageSinkRoot sinkRoot("myTree", settingsWrite);

   auto model = std::make_shared<RForestModel>();
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


TEST(RForest, WriteRead)
{
   auto model = std::make_shared<RForestModel>();
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
      ROutputForest forest(model, std::make_unique<RPageSinkRoot>("f", "test.root"));
      forest.Fill();
   }

   *fieldPt = 0.0;
   *fieldEnergy = 0.0;
   fieldTag->clear();
   fieldJets->clear();
   fieldNnlo->clear();
   fieldKlass->s.clear();

   RInputForest forest(model, std::make_unique<RPageSourceRoot>("f", "test.root"));
   EXPECT_EQ(1U, forest.GetNEntries());
   forest.GetEntry(0);

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

TEST(RForest, View)
{
   auto model = std::make_shared<RForestModel>();
   auto fieldPt = model->AddField<float>("pt", 42.0);
   auto fieldTag = model->AddField<std::string>("tag", "xyz");
   auto fieldJets = model->AddField<std::vector<float>>("jets");
   fieldJets->push_back(1.0);
   fieldJets->push_back(2.0);

   {
      ROutputForest forest(model, std::make_unique<RPageSinkRoot>("f", "test.root"));
      forest.Fill();
   }

   RInputForest forest(std::make_unique<RPageSourceRoot>("f", "test.root"));
   auto viewPt = forest.GetView<float>("pt");
   int i = 0;
   while (forest.ViewNext()) {
      EXPECT_EQ(42.0, viewPt());
      i++;
   }
   EXPECT_EQ(1, i);

   auto ctx = forest.GetViewContext();
   auto viewJets = forest.GetView<std::vector<float>>("jets", ctx.get());
   i = 0;
   while (ctx->Next()) {
      EXPECT_EQ(2U, viewJets().size());
      EXPECT_EQ(1.0, viewJets()[0]);
      EXPECT_EQ(2.0, viewJets()[1]);
      i++;
   }
   EXPECT_EQ(1, i);
}

TEST(RForest, Compositional)
{
   auto event_model = std::make_shared<RForestModel>();
   auto h1_px = event_model->AddField<float>("h1_px", 0.0);

   auto hit_model = std::make_shared<RForestModel>();
   auto hit_x = hit_model->AddField<float>("x", 0.0);
   auto hit_y = hit_model->AddField<float>("y", 0.0);

   auto track_model = std::make_shared<RForestModel>();
   auto track_energy = track_model->AddField<float>("energy", 0.0);

   /*auto hits =*/ track_model->AddCollection("hits", hit_model);
}

TEST(RForest, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::MyTypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RField<std::vector<std::string>>::MyTypeName().c_str());
   EXPECT_STREQ("ROOT::Experimental::RForestTest",
                ROOT::Experimental::RField<ROOT::Experimental::RForestTest>::MyTypeName().c_str());
}

namespace {
class RNoDictionary {};
} // namespace

TEST(RForest, TClass) {
   auto modelFail = std::make_shared<RForestModel>();
   EXPECT_THROW(modelFail->AddField<RNoDictionary>("nodict"), std::runtime_error);

   auto model = std::make_shared<RForestModel>();
   auto ptrKlass = model->AddField<ROOT::Experimental::RForestTest>("klass");

   ROutputForest forest(model, std::make_unique<RPageSinkRoot>("f", "test.root"));
}


TEST(RForest, Capture) {
   auto model = std::make_shared<RForestModel>();
   float pt;
   model->CaptureField("pt", &pt);
}

TEST(RForest, RealWorld1)
{
   // See https://github.com/olifre/root-io-bench/blob/master/benchmark.cpp
   auto model = RForestModel::Create();
   auto& fldEvent = model->AddFieldRef<std::uint32_t>("event");
   auto& fldEnergy = model->AddFieldRef<double>("energy");
   auto& fldTimes = model->AddFieldRef<std::vector<double>>("times");
   auto& fldIndices = model->AddFieldRef<std::vector<std::uint32_t>>("indices");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto forest = ROutputForest::Create(model, "f", "test.root");
      constexpr unsigned int nEvents = 60000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         fldEvent = i;
         fldEnergy = rnd.Rndm() * 1000.;

         chksumWrite += double(fldEvent);
         chksumWrite += fldEnergy;

         auto nTimes = 1 + floor(rnd.Rndm() * 1000.);
         fldTimes.resize(nTimes);
         for (unsigned int n = 0; n < nTimes; ++n) {
            fldTimes[n] = 1 + rnd.Rndm()*1000. - 500.;
            chksumWrite += fldTimes[n];
         }

         auto nIndices = 1 + floor(rnd.Rndm() * 1000.);
         fldIndices.resize(nIndices);
         for (unsigned int n = 0; n < nIndices; ++n) {
            fldIndices[n] = 1 + floor(rnd.Rndm() * 1000.);
            chksumWrite += double(fldIndices[n]);
         }

         forest->Fill();
      }
   }

   double chksumRead = 0.0;
   auto forest = RInputForest::Create(model, "f", "test.root");
   for (unsigned int i = 0; i < forest->GetNEntries(); ++i) {
      forest->GetEntry(i);
      chksumRead += double(fldEvent) + fldEnergy;
      for (auto t : fldTimes) chksumRead += t;
      for (auto ind : fldIndices) chksumRead += double(ind);
   }

   EXPECT_EQ(chksumRead, chksumWrite);
}
