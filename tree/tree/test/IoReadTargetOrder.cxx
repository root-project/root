/// \file IoReadTargetOrder.cxx
/// \brief Regression test for ROOT GitHub issue #23686.

#include "IoReadTargetOrder.hxx"

#include "gtest/gtest.h"

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <memory>
#include <vector>

static void AssertRuleRan(const ClusterIoRead &c, const char *context)
{
   EXPECT_EQ(1, c.ran) << context << ": read rule did not run (or ran more than once)";
   ASSERT_EQ(3u, c.v.size()) << context << ": v has wrong size";
   EXPECT_EQ(10u, c.v[0]) << context << ": v[0] wrong";
   EXPECT_EQ(20u, c.v[1]) << context << ": v[1] wrong";
   EXPECT_EQ(30u, c.v[2]) << context << ": v[2] wrong";
   EXPECT_EQ(99u, c.f) << context << ": f wrong";
   EXPECT_EQ(18u, c.q) << context << ": q wrong (expect v.size()*6)";
}

class IoReadTargetOrder : public ::testing::Test {
protected:
   static constexpr const char *kFileName = "test_ioread_target_order.root";

   void SetUp() override
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(kFileName, "RECREATE"));
      ASSERT_TRUE(file && !file->IsZombie());

      auto *t = new TTree("t", "t");
      ClusterIoReadV1 c;
      c.v = {10, 20, 30};
      c.f = 99;
      auto *cPtr = &c;

      auto *vec = new std::vector<ClusterIoReadV1>{c};
      auto *w = new WrapperIoRead<HolderIoRead<ClusterIoReadV1>>;
      w->obj.v.push_back(c);

      t->Branch("obj", &cPtr, 32000, 99);
      t->Branch("vec", &vec, 32000, 99);
      t->Branch("wrap", &w, 32000, 99);
      t->Branch("wrap.", &w, 32000, 99);
      t->Fill();
      file->Write();

      delete vec;
      delete w;
      file->Close();
   }

   void TearDown() override { std::remove(kFileName); }
};

TEST_F(IoReadTargetOrder, AllBranchLayouts)
{
   auto file = std::unique_ptr<TFile>(TFile::Open(kFileName));
   ASSERT_TRUE(file && !file->IsZombie());

   auto *t = file->Get<TTree>("t");
   ASSERT_NE(nullptr, t);

   ClusterIoRead *obj = nullptr;
   t->SetBranchAddress("obj", &obj);
   t->GetBranch("obj")->GetEntry(0);
   ASSERT_NE(nullptr, obj);
   AssertRuleRan(*obj, "obj");
   delete obj;

   std::vector<ClusterIoRead> *vec = nullptr;
   t->SetBranchAddress("vec", &vec);
   t->GetBranch("vec")->GetEntry(0);
   ASSERT_NE(nullptr, vec);
   ASSERT_EQ(1u, vec->size());
   AssertRuleRan(vec->at(0), "vec");
   delete vec;

   WrapperIoRead<HolderIoRead<ClusterIoRead>> *wrap = nullptr;
   t->SetBranchAddress("wrap", &wrap);
   t->GetBranch("wrap")->GetEntry(0);
   ASSERT_NE(nullptr, wrap);
   ASSERT_EQ(1u, wrap->obj.v.size());
   AssertRuleRan(wrap->obj.v.at(0), "wrap");
   delete wrap;

   WrapperIoRead<HolderIoRead<ClusterIoRead>> *wrapDot = nullptr;
   t->SetBranchAddress("wrap.", &wrapDot);
   t->GetBranch("wrap.")->GetEntry(0);
   ASSERT_NE(nullptr, wrapDot);
   ASSERT_EQ(1u, wrapDot->obj.v.size());
   AssertRuleRan(wrapDot->obj.v.at(0), "wrap.");
   delete wrapDot;
}
