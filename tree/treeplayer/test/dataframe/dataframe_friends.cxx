#include "ROOT/TDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "TSystem.h"
#include "TFile.h"
#include "TTree.h"
#include "gtest/gtest.h"

using namespace ROOT::Experimental;

// fixture that creates two files with two trees of 10 events each. One has branch `x`, the other branch `y`, both ints.
class TDFAndFriends : public ::testing::Test {
protected:
   constexpr static auto kFile1 = "test_tdfandfriends.root";
   constexpr static auto kFile2 = "test_tdfandfriends2.root";
   constexpr static auto kFile3 = "test_tdfandfriends3.root";
   static void SetUpTestCase()
   {
      TDataFrame d(4);
      d.Define("x", [] { return 1; }).Snapshot<int>("t", kFile1, {"x"});
      d.Define("y", [] { return 2; }).Snapshot<int>("t2", kFile2, {"y"});

      TFile f(kFile3, "RECREATE");
      TTree t("t3", "t3");
      float arr[4];
      t.Branch("arr", arr, "arr[4]/F");
      for (auto i : ROOT::TSeqU(4)) {
         for (auto j : ROOT::TSeqU(4)) {
            arr[j] = i + j;
         }
         t.Fill();
      }
      t.Write();
   }

   static void TearDownTestCase()
   {
      for (auto fileName : {kFile1, kFile2, kFile3})
         gSystem->Unlink(fileName);
   }
};

TEST_F(TDFAndFriends, FriendByFile)
{
   TFile f1(kFile1);
   TTree *t1 = static_cast<TTree *>(f1.Get("t"));
   t1->AddFriend("t2", kFile2);
   TDataFrame d(*t1);
   auto x = d.Min<int>("x");
   auto t = d.Take<int>("y");
   EXPECT_EQ(*x, 1);
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
   auto t = d.Take<int>("y");
   EXPECT_EQ(*x, 1);
   for (auto v : t)
      EXPECT_EQ(v, 2);
}

TEST_F(TDFAndFriends, FriendArrayByFile)
{
   TFile f1(kFile1);
   TTree *t1 = static_cast<TTree *>(f1.Get("t"));
   t1->AddFriend("t3", kFile3);
   TDataFrame d(*t1);

   int i(0);
   auto checkArr = [&i](std::array_view<float> av) {
      auto ifloat = float(i);
      EXPECT_EQ(ifloat, av[0]);
      EXPECT_EQ(ifloat + 1, av[1]);
      EXPECT_EQ(ifloat + 2, av[2]);
      EXPECT_EQ(ifloat + 3, av[3]);
      i++;
   };
   d.Foreach(checkArr, {"arr"});
}

TEST_F(TDFAndFriends, FriendArrayByPointer)
{
   TFile f1(kFile1);
   TTree *t1 = static_cast<TTree *>(f1.Get("t"));
   TFile f3(kFile3);
   TTree *t3 = static_cast<TTree *>(f3.Get("t3"));
   t1->AddFriend(t3);
   TDataFrame d(*t1);

   int i(0);
   auto checkArr = [&i](std::array_view<float> av) {
      auto ifloat = float(i);
      EXPECT_EQ(ifloat, av[0]);
      EXPECT_EQ(ifloat + 1, av[1]);
      EXPECT_EQ(ifloat + 2, av[2]);
      EXPECT_EQ(ifloat + 3, av[3]);
      i++;
   };
   d.Foreach(checkArr, {"arr"});
}
