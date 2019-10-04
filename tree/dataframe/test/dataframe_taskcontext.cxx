#include "ROOT/RDF/TaskContext.hxx"

#include "gtest/gtest.h"

#include <array>

namespace RDFInt = ROOT::Internal::RDF;

TEST(RDataFrameTaskContext, Size)
{
   ROOT::Internal::RDF::TaskContextStorage storage(42);
   ROOT::Internal::RDF::FSVector<int> vec(storage, 17);

   EXPECT_EQ(storage.size(), 42u);
   EXPECT_EQ(vec.size(), 42u);
}

TEST(RDataFrameTaskContext, SetGetInt)
{
   ROOT::Internal::RDF::TaskContextStorage storage(42);
   ROOT::Internal::RDF::FSVector<int> vec(storage, 17);

   int count = 0;
   for (auto &v: vec) {
      EXPECT_EQ(v, 17);
      v = ++count;
   }
   count = 0;
   for (auto &v: vec) {
      EXPECT_EQ(v, ++count);
   }
}

struct S {
   S(const std::array<char, 3>& vals): fVals(vals) {}
   std::array<char, 3> fVals;
   bool operator==(const S& rhs) const { return fVals == rhs.fVals; }
};

TEST(RDataFrameTaskContext, SetGetStruct)
{
   ROOT::Internal::RDF::TaskContextStorage storage(42);
   std::array<char, 3> init{12, 13, 14};
   ROOT::Internal::RDF::FSVector<S> vec(storage, init);

   char count = 0;
   for (auto &v: vec) {
      EXPECT_EQ(v, init);
      ++count;
      v = S(std::array<char,3>{count, char(count + 1), char(count + 2)});
   }
   count = 0;
   for (auto &v: vec) {
      ++count;
      EXPECT_EQ(v.fVals[0], count);
      EXPECT_EQ(v.fVals[1], count + 1);
      EXPECT_EQ(v.fVals[2], count + 2);
   }
}

TEST(RDataFrameTaskContext, SetGetVecStruct)
{
   ROOT::Internal::RDF::TaskContextStorage storage(42);
   std::array<char, 3> init{12, 13, 14};
   ROOT::Internal::RDF::FSVector<std::vector<S>> vec(storage, std::vector<S>{init, init, init, init});

   char count = 0;
   for (auto &v: vec) {
      EXPECT_EQ(v[3], init);
      ++count;
      v[1] = S(std::array<char,3>{count, char(count + 1), char(count + 2)});
   }
   count = 0;
   for (auto &v: vec) {
      ++count;
      EXPECT_EQ(v[1].fVals[0], count);
      EXPECT_EQ(v[1].fVals[1], count + 1);
      EXPECT_EQ(v[1].fVals[2], count + 2);
   }
}



struct Huge {
   Huge(const std::array<long, 3>& vals): fBig(), fVals(vals) {}
   long fBig[1024*16];
   std::array<long, 3> fVals;
   bool operator==(const Huge& rhs) const { return fVals == rhs.fVals; }
};


TEST(RDataFrameTaskContext, SetGetHugeStruct)
{
   ROOT::Internal::RDF::TaskContextStorage storage(1024);
   std::array<long, 3> init{12, 13, 14};
   ROOT::Internal::RDF::FSVector<Huge> vec(storage, init);

   long count = 0;
   for (auto &v: vec) {
      EXPECT_EQ(v, init);
      ++count;
      v = Huge(std::array<long,3>{count, long(count + 1), long(count + 2)});
   }
   count = 0;
   for (auto &v: vec) {
      ++count;
      EXPECT_EQ(v.fVals[0], count);
      EXPECT_EQ(v.fVals[1], count + 1);
      EXPECT_EQ(v.fVals[2], count + 2);
   }
}

TEST(RDataFrameTaskContext, AllocationPattern)
{
   ROOT::Internal::RDF::TaskContextStorage storage(42);
   ROOT::Internal::RDF::FSVector<S> vec(storage, std::array<char,3>{'a', 'b', 'c'});
   auto prev = vec.begin();
   for (auto i = ++vec.begin(), e = vec.end(); i != e; ++i) {
      EXPECT_GE(abs(&*vec.begin() - &*i), 4 * 1024); // elements should be further away from begin() than 4k
      EXPECT_GE(abs(&*prev - &*i), 4 * 1024); // subsequent elements should be further apart than 4k
      prev = i;
   }
}
