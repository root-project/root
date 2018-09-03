#include <gtest/gtest.h>
#include <TMVA/RTensor.hxx>
#include <iostream> // DEBUG

using namespace TMVA::Experimental;

TEST(RTensor, AdoptMemory)
{
   float data[4] = {0, 0, 0, 0};
   RTensor<float> x(data, {4});
   for (size_t i = 0; i < 4; i++) {
      x(i) = (float)i;
   }
   for (size_t i = 0; i < 4; i++) {
      EXPECT_EQ((float)i, data[i]);
   }
}

TEST(RTensor, InitWithZeros)
{
   RTensor<float> x({2, 3});
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(0.0, x(i, j));
      }
   }
}

TEST(RTensor, GetElement)
{
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3});
   auto shape = x.GetShape();
   float count = 0.0;
   for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
         EXPECT_EQ(count, x.At({i, j}));
         EXPECT_EQ(count, x(i, j));
         count++;
      }
   }
}

TEST(RTensor, SetElement)
{
   RTensor<float> x({2, 3});
   auto shape = x.GetShape();
   float count = 0.0;
   for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
         x.At({i, j}) = count;
         x(i, j) = count;
         count++;
      }
   }
   auto data = x.GetData();
   for (size_t i = 0; i < shape.size(); i++) {
      EXPECT_EQ((float)i, *(data + i));
   }
}

TEST(RTensor, RowMajorMemoryOrder)
{
   // Layout:
   // 0, 1, 2
   // 3, 4, 5
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3}, MemoryOrder::RowMajor);

   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(count, x(i, j));
         count++;
      }
   }
}

TEST(RTensor, ColumnMajorMemoryOrder)
{
   // Layout:
   // 0, 2, 4
   // 1, 3, 5
   float data[6] = {0, 2, 4, 1, 3, 5};
   RTensor<float> x(data, {2, 3}, MemoryOrder::ColumnMajor);

   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(count, x(i, j));
         count++;
      }
   }
}
