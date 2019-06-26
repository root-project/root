#include <gtest/gtest.h>
#include <TMVA/RTensor.hxx>

using namespace TMVA::Experimental;

TEST(RTensor, DereferenceOperator)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});

   auto it = x.begin();
   EXPECT_EQ(0, *it);

   it++;
   EXPECT_EQ(1, *it);
}

TEST(RTensor, PlusMinusEqualOperator)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});

   auto it = x.begin();
   EXPECT_EQ(0, *it);

   it += 3;
   EXPECT_EQ(3, *it);

   it -= 2;
   EXPECT_EQ(1, *it);
}

TEST(RTensor, LessGreaterEqualOperator)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});

   auto it = x.begin();
   auto it1 = x.begin();
   EXPECT_TRUE(it==it1);
   EXPECT_TRUE(it>=it1);
   EXPECT_TRUE(it<=it1);

   it += 2;
   EXPECT_TRUE(it>it1);
   EXPECT_TRUE(it1<it);
}

TEST(RTensor, AddSubtractOperator)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});

   auto it = x.begin();
   EXPECT_EQ(0, *it);

   auto it1 = it + 3;
   EXPECT_EQ(3, *it1);

   auto it2 = it1 - 1;
   EXPECT_EQ(2, *it2);
}

TEST(RTensor, Difference)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});

   auto it = x.begin() + 3;
   auto diff = it - x.begin();
   EXPECT_EQ(diff, 3);

   auto diff2 = x.begin() - it;
   EXPECT_EQ(diff2, -3);

   auto diff3 = x.end() - x.begin();
   EXPECT_EQ(diff3, 4);
}

TEST(RTensor, ComputeIndicesFromGlobalIndexRowMajor)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::RowMajor);
   for (std::size_t i = 0; i < 4; i++) {
      auto idx = Internal::ComputeIndicesFromGlobalIndex(x.GetShape(), x.GetMemoryLayout(), i);
      EXPECT_EQ(x(idx), data[i]);
   }
}

TEST(RTensor, ComputeIndicesFromGlobalIndexColumnMajor)
{
   float data[4] = {0, 2, 1, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::ColumnMajor);
   for (std::size_t i = 0; i < 4; i++) {
      auto idx = Internal::ComputeIndicesFromGlobalIndex(x.GetShape(), x.GetMemoryLayout(), i);
      EXPECT_EQ(x(idx), data[i]);
   }
}

TEST(RTensor, ComputeIndicesFromGlobalIndexSlice)
{
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3});
   auto y = x.Slice({{0, 2}, {1, 3}});
   float ref[4] = {1, 2, 4, 5};
   for (std::size_t i = 0; i < 4; i++) {
      auto idx = Internal::ComputeIndicesFromGlobalIndex(y.GetShape(), y.GetMemoryLayout(), i);
      EXPECT_EQ(y(idx), ref[i]);
   }
}

TEST(RTensor, BeginEndLoopRowMajor)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::RowMajor);
   auto globalIdx = 0u;
   for (auto it = x.begin(); it != x.end(); it++) {
      EXPECT_EQ(data[globalIdx], *it);
      globalIdx++;
   }
}

TEST(RTensor, BeginEndLoopColumnMajor)
{
   float data[4] = {0, 2, 1, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::ColumnMajor);
   auto globalIdx = 0u;
   for (auto it = x.begin(); it != x.end(); it++) {
      EXPECT_EQ(data[globalIdx], *it);
      globalIdx++;
   }
}

TEST(RTensor, BeginEndLoopSlice)
{
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3});
   auto y = x.Slice({{0, 2}, {1, 3}});
   auto globalIdx = 0u;
   float ref[4] = {1, 2, 4, 5};
   for (auto it = y.begin(); it != y.end(); it++) {
      EXPECT_EQ(ref[globalIdx], *it);
      globalIdx++;
   }
}

TEST(RTensor, RangeBasedLoop)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});
   auto globalIdx = 0u;
   for (auto y : x) {
      EXPECT_EQ(data[globalIdx], y);
      globalIdx++;
   }
}

TEST(RTensor, CopyRowMajor)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::RowMajor);
   auto y1 = x.Copy(MemoryLayout::RowMajor);
   auto y2 = x.Copy(MemoryLayout::ColumnMajor);

   for (std::size_t i = 0; i < 2; i++) {
      for (std::size_t j = 0; j < 2; j++) {
         EXPECT_EQ(x(i, j), y1(i, j));
         EXPECT_EQ(x(i, j), y2(i, j));
      }
   }
}

TEST(RTensor, CopyColumnMajor)
{
   float data[4] = {0, 2, 1, 3};
   RTensor<float> x(data, {2, 2}, MemoryLayout::ColumnMajor);
   auto y1 = x.Copy(MemoryLayout::RowMajor);
   auto y2 = x.Copy(MemoryLayout::ColumnMajor);

   for (std::size_t i = 0; i < 2; i++) {
      for (std::size_t j = 0; j < 2; j++) {
         EXPECT_EQ(x(i, j), y1(i, j));
         EXPECT_EQ(x(i, j), y2(i, j));
      }
   }
}

TEST(RTensor, CopySlice)
{
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> y(data, {2, 3});
   auto x = y.Slice({{0, 2}, {1, 3}});
   auto y1 = x.Copy(MemoryLayout::RowMajor);
   auto y2 = x.Copy(MemoryLayout::ColumnMajor);

   for (std::size_t i = 0; i < 2; i++) {
      for (std::size_t j = 0; j < 2; j++) {
         EXPECT_EQ(x(i, j), y1(i, j));
         EXPECT_EQ(x(i, j), y2(i, j));
         EXPECT_EQ(y(i, j + 1), x(i, j));
      }
   }
}

TEST(RTensor, StdForEach)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});
   auto increment = [](float& n) { n++; };
   std::for_each(x.begin(), x.end(), increment);

   for (std::size_t i = 0; i < 4; i++) {
      EXPECT_EQ(data[i], float(i) + 1.0);
   }
}

TEST(RTensor, IterateSlice)
{
   float data[4] = {0, 1, 2, 3};
   RTensor<float> x(data, {2, 2});
   auto y = x.Slice({{0, 2}, {0, 1}});
   for (auto& e : y) e++;
   EXPECT_EQ(data[0], 1);
   EXPECT_EQ(data[1], 1);
   EXPECT_EQ(data[2], 3);
   EXPECT_EQ(data[3], 3);
}
