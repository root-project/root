#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

#include <array>

#include "gtest/gtest.h"

TEST(ReaderArrayIterator, NonConst) {
   // test TTreeReaderArray's nested iterator type

   // create input TTree
   TTree tree("TTreeReaderArrayTree", "In-memory test tree");
   std::array<int, 6> values{{0, 1, 2, 3, 4, 5}};
   int n = 6;
   tree.Branch("n", &n);
   tree.Branch("arr", &values, "arr[n]/I");
   tree.Fill();
   n = 0;
   tree.Fill();
   tree.ResetBranchAddresses();

   // create TTreeReader
   TTreeReader r(&tree);
   TTreeReaderArray<int> arr(r, "arr");

   // tests on full entry
   r.Next();

   // begin and end, comparisons
   EXPECT_EQ(arr.begin(), arr.begin());
   EXPECT_EQ(arr.end(), arr.end());
   EXPECT_NE(arr.begin(), arr.end());

   // copy/move ctors
   auto it = arr.begin();
   const auto it2(it);
   auto it3(std::move(it));

   // increments, decrements, copy/move assignments
   EXPECT_EQ(++it, arr.begin() + 1);
   it = it3++;
   EXPECT_EQ(it += 1, it3);

   EXPECT_EQ((arr.begin() + 2) - 1, --(arr.begin() + 2));
   it = std::move(it3);
   auto it4 = it--;
   EXPECT_EQ(it4 -= 1, it);

   // operator+(int, iterator)
   3 + it;

   // comparisons
   auto e = arr.end();
   auto b = arr.begin();
   EXPECT_GT(e, b);
   EXPECT_LT(b, b + 2);
   EXPECT_LE(b + 1, b + 1);
   EXPECT_GE(b + 1, b + 1);

   // difference
   const std::ptrdiff_t length = arr.end() - arr.begin();
   EXPECT_EQ(arr.begin() + length, arr.end());
   const auto it5 = arr.begin() + 2;
   EXPECT_EQ(arr.begin() + (it5 - arr.begin()), arr.begin() + 2);

   // operator[]
   EXPECT_EQ(arr.begin()[2], *(arr.begin() + 2));

   // ranged for loop
   int truth = 0;
   for (auto v : arr)
      EXPECT_DOUBLE_EQ(v, truth++);

   // tests on empty entry
   r.Next();
   EXPECT_EQ(arr.begin(), arr.end());
}

TEST(ReaderArrayIterator, Const) {
   // test const TTreeReaderArray's nested const_iterator type

   // create input TTree
   TTree tree("TTreeReaderArrayTree", "In-memory test tree");
   std::array<int, 6> values{{0, 1, 2, 3, 4, 5}};
   int n = 6;
   tree.Branch("n", &n);
   tree.Branch("arr", &values, "arr[n]/I");
   tree.Fill();
   n = 0;
   tree.Fill();
   tree.ResetBranchAddresses();

   // create TTreeReader
   TTreeReader r(&tree);
   const TTreeReaderArray<int> arr(r, "arr");

   // tests on full entry
   r.Next();

   // begin and end, comparisons
   EXPECT_EQ(arr.begin(), arr.begin());
   EXPECT_EQ(arr.end(), arr.end());
   EXPECT_NE(arr.begin(), arr.end());

   // copy/move ctors
   auto it = arr.begin();
   const auto it2(it);
   (void)it2;
   auto it3(std::move(it));
   (void)it3;

   // increments, decrements
   EXPECT_EQ((arr.begin() + 2) - 1, arr.begin() + 1);

   // operator+(int, iterator)
   3 + it;

   // comparisons
   auto e = arr.end();
   auto b = arr.begin();
   EXPECT_GT(e, b);
   EXPECT_LT(b, b + 2);
   EXPECT_LE(b + 1, b + 1);
   EXPECT_GE(b + 1, b + 1);

   // difference
   const auto length = arr.end() - arr.begin();
   EXPECT_EQ(arr.begin() + length, arr.end());
   const auto it5 = arr.begin() + 2;
   EXPECT_EQ(arr.begin() + (it5 - arr.begin()), arr.begin() + 2);

   // operator[]
   EXPECT_EQ(arr.begin()[2], *(arr.begin() + 2));

   // ranged for loop
   int truth = 0;
   for (auto v : arr)
      EXPECT_EQ(v, truth++);

   // tests on empty entry
   r.Next();
   EXPECT_EQ(arr.begin(), arr.end());
}
