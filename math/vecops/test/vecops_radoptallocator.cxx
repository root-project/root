#include <ROOT/RAdoptAllocator.hxx>

#include <gtest/gtest.h>

#include <vector>
#include <iostream>
using namespace ROOT::Detail::VecOps;

TEST(RAdoptAllocator, ReusePointer)
{
   std::vector<double> vreference{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

   std::vector<double> vmodel(vreference);
   RAdoptAllocator<double> alloc0(vmodel.data());

   EXPECT_EQ(vmodel.data(), std::allocator_traits<RAdoptAllocator<double>>::allocate(alloc0, 123));

   RAdoptAllocator<double> alloc1(vmodel.data());

   std::vector<double, RAdoptAllocator<double>> v(alloc1);
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

TEST(RAdoptAllocator, NewAllocations)
{
   unsigned int copyCount = 0;
   std::vector<TCopySignal> model;
   model.reserve(8);
   for (int i = 0; i < 8; ++i) {
      model.emplace_back(copyCount);
   }

   EXPECT_EQ(0U, copyCount);

   unsigned int dummy;
   RAdoptAllocator<TCopySignal> alloc(model.data());
   std::vector<TCopySignal, RAdoptAllocator<TCopySignal>> v(model.size(), dummy, alloc);

   EXPECT_EQ(0U, copyCount);
   v.emplace_back(copyCount);
   EXPECT_EQ(8U, copyCount);
}

void CheckMove(std::vector<double, RAdoptAllocator<double>>&& vm, const std::vector<double, RAdoptAllocator<double>>& v) {
   EXPECT_TRUE(vm.get_allocator() == v.get_allocator());
}

TEST(RAdoptAllocator, Traits)
{
   std::vector<double> vmodel {1.,2.,3.};
   RAdoptAllocator<double> alloc0(vmodel.data());
   RAdoptAllocator<double> alloc1(vmodel.data());
   std::vector<double, RAdoptAllocator<double>> v0(3, 0., alloc0);
   std::vector<double, RAdoptAllocator<double>> v1(3, 0., alloc1);

   EXPECT_TRUE(v0.get_allocator() == v1.get_allocator()) << "Baseline test failed";

   std::vector<double, RAdoptAllocator<double>> v2;
   v2 = v0;
   EXPECT_FALSE(v0.get_allocator() == v2.get_allocator());

   swap(v1,v2);
   EXPECT_TRUE(v0.get_allocator() == v2.get_allocator());

   CheckMove(std::move(v0),v2);

}

