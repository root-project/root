#include "ROOT/TDataFrame.hxx"
#include "TSystem.h"
#include "TTree.h"
#include "gtest/gtest.h"

using namespace ROOT::Experimental;

// fixture that creates two files with two trees of 10 events each. One has branch `x`, the other branch `y`, both ints.
class TDFAndFriends : public ::testing::Test {
protected:
   constexpr static auto kFile1 = "test_tdfandfriends.root";
   constexpr static auto kFile2 = "test_tdfandfriends2.root";

   static void SetUpTestCase()
   {
      TDataFrame d(10);
      d.Define("x", [] { return 1; }).Snapshot<int>("t", kFile1, {"x"});
      d.Define("y", [] { return 2; }).Snapshot<int>("t2", kFile2, {"y"});
   }

   static void TearDownTestCase()
   {
      gSystem->Unlink("test_tdfandfriends.root");
      gSystem->Unlink("test_tdfandfriends2.root");
   }
};

TEST_F(TDFAndFriends, FriendByFile)
{
   TFile f1(kFile1);
   TTree *t1 = static_cast<TTree *>(f1.Get("t"));
   t1->AddFriend("t2", kFile2);
   TDataFrame d(*t1);
   auto x = d.Min<int>("x");
   EXPECT_EQ(*x, 1);
   auto t = d.Take<int>("y");
   for (auto v : t)
      EXPECT_EQ(v, 2);
}

TEST_F(TDFAndFriends, FriendByPointer)
{
   TFile f1(kFile1);
   TTree *t1 = static_cast<TTree *>(f1.Get("t"));
   TFile f2(kFile2);
   TTree *t2 = static_cast<TTree *>(f2.Get("t2"));
   t1->AddFriend(t2);
   TDataFrame d(*t1);
   auto x = d.Min<int>("x");
   EXPECT_EQ(*x, 1);
   auto t = d.Take<int>("y");
   for (auto v : t)
      EXPECT_EQ(v, 2);
}
