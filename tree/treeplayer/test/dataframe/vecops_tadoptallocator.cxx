#include <ROOT/TVec.hxx>

#include <gtest/gtest.h>

#include <vector>
#include <iostream>
using namespace ROOT::Detail::VecOps;
using namespace ROOT::Experimental::VecOps;

TEST(TAdoptAllocator, ReusePointer)
{
   std::vector<double> vreference{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

   std::vector<double> vmodel(vreference);
   TAdoptAllocator<double> alloc0(vmodel.data(), vmodel.size());

   EXPECT_EQ(vmodel.data(), alloc0.allocate(123));

   TAdoptAllocator<double> alloc1(vmodel.data(), vmodel.size());

   std::vector<double, TAdoptAllocator<double>> v(alloc1);
   v.resize(vmodel.size());

   EXPECT_EQ(vmodel.data(), v.data());
   EXPECT_EQ(vmodel.size(), v.size());

   for (size_t i = 0; i < vmodel.size(); ++i) {
      EXPECT_EQ(vmodel[i], vreference[i]);
      EXPECT_EQ(v[i], vreference[i]);
   }

   v.emplace_back(3.);
   EXPECT_NE(vmodel.data(), v.data());

   auto res = 0;
   try {
      v.resize(std::numeric_limits<size_t>::max());
   } catch (...) {
      res = 1;
   }
   EXPECT_EQ(1, res);
}

class TCopySignal {
private:
   unsigned int *fCopyCount = nullptr;

public:
   TCopySignal(unsigned int &copyCount) : fCopyCount(&copyCount){};
   TCopySignal(const TCopySignal &other)
   {
      fCopyCount = other.fCopyCount;
      (*fCopyCount)++;
   }
};

TEST(TAdoptAllocator, NewAllocations)
{
   unsigned int copyCount = 0;
   std::vector<TCopySignal> model;
   model.reserve(8);
   for (int i = 0; i < 8; ++i) {
      model.emplace_back(copyCount);
   }

   EXPECT_EQ(0U, copyCount);

   unsigned int dummy;
   TAdoptAllocator<TCopySignal> alloc(model.data(), model.size());
   ROOT::Detail::VecOps::TVecImpl<TCopySignal> v(model.size(), dummy, alloc);

   EXPECT_EQ(0U, copyCount);
   v.emplace_back(copyCount);
   EXPECT_EQ(8U, copyCount);
}
