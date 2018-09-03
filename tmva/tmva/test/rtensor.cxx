#include <gtest/gtest.h>
#include <TMVA/RTensor.hxx>

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

TEST(RTensor, SetElementRowMajor)
{
   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   RTensor<float> x({2, 3}, MemoryOrder::RowMajor);
   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(i, j) = count;
         count++;
      }
   }

   // Layout (physical):
   // 0, 3, 1, 4, 2, 5
   float ref[6] = {0, 1, 2, 3, 4, 5};
   float *data = x.GetData();
   for (size_t i = 0; i < 6; i++) {
      EXPECT_EQ(data[i], ref[i]);
   }
}

TEST(RTensor, SetElementColumnMajor)
{
   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   RTensor<float> x({2, 3}, MemoryOrder::ColumnMajor);
   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(i, j) = count;
         count++;
      }
   }

   // Layout (physical):
   // 0, 3, 1, 4, 2, 5
   float ref[6] = {0, 3, 1, 4, 2, 5};
   float *data = x.GetData();
   for (size_t i = 0; i < 6; i++) {
      EXPECT_EQ(data[i], ref[i]);
   }
}

TEST(RTensor, GetElementRowMajor)
{
   // Layout (physical):
   // 0, 3, 1, 4, 2, 5
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3}, MemoryOrder::RowMajor);

   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(x(i, j), count);
         count++;
      }
   }
}

TEST(RTensor, GetElementColumnMajor)
{
   // Layout (physical):
   // 0, 3, 1, 4, 2, 5
   float data[6] = {0, 3, 1, 4, 2, 5};
   RTensor<float> x(data, {2, 3}, MemoryOrder::ColumnMajor);

   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(x(i, j), count);
         count++;
      }
   }
}
