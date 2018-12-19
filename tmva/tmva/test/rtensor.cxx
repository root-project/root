#include <gtest/gtest.h>
#include <TMVA/RTensor.hxx>

using namespace TMVA::Experimental;

TEST(RTensor, GetElement)
{
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3});
   auto shape = x.GetShape();
   float count = 0.0;
   for (size_t i = 0; i < shape[0]; i++) {
      for (size_t j = 0; j < shape[1]; j++) {
         EXPECT_EQ(count, x.At({i, j}));
         EXPECT_EQ(count, x.At(i, j));
         EXPECT_EQ(count, x({i, j}));
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
         x.At(i, j) = count;
         x({i, j}) = count;
         x(i, j) = count;
         count++;
      }
   }
   auto data = x.GetData();
   for (size_t i = 0; i < shape.size(); i++)
      EXPECT_EQ((float)i, *(data + i));
}

TEST(RTensor, AdoptMemory)
{
   float data[4] = {0, 0, 0, 0};
   RTensor<float> x(data, {4});
   for (size_t i = 0; i < 4; i++)
      x(i) = (float)i;
   for (size_t i = 0; i < 4; i++)
      EXPECT_EQ((float)i, data[i]);
}

TEST(RTensor, GetShape)
{
   RTensor<int> x({2, 3});
   const auto s = x.GetShape();
   EXPECT_EQ(s.size(), 2u);
   EXPECT_EQ(s[0], 2u);
   EXPECT_EQ(s[1], 3u);
}

TEST(RTensor, Reshape)
{
   RTensor<int> x({2, 3});
   const auto s = x.GetShape();
   EXPECT_EQ(s.size(), 2u);
   EXPECT_EQ(s[0], 2u);
   EXPECT_EQ(s[1], 3u);

   x.Reshape({1, 6});
   const auto s2 = x.GetShape();
   EXPECT_EQ(s2.size(), 2u);
   EXPECT_EQ(s2[0], 1u);
   EXPECT_EQ(s2[1], 6u);

   x.Reshape({6});
   const auto s3 = x.GetShape();
   EXPECT_EQ(s3.size(), 1u);
   EXPECT_EQ(s3[0], 6u);
}

TEST(RTensor, ExpandDims)
{
   RTensor<int> x({2, 3});
   x.ExpandDims(0);
   const auto s = x.GetShape();
   EXPECT_EQ(s.size(), 3u);
   EXPECT_EQ(s[0], 1u);
   EXPECT_EQ(s[1], 2u);
   EXPECT_EQ(s[2], 3u);

   RTensor<int> x1({2, 3});
   x1.ExpandDims(1);
   const auto s1 = x1.GetShape();
   EXPECT_EQ(s1.size(), 3u);
   EXPECT_EQ(s1[0], 2u);
   EXPECT_EQ(s1[1], 1u);
   EXPECT_EQ(s1[2], 3u);

   RTensor<int> x2({2, 3});
   x2.ExpandDims(-1);
   const auto s2 = x2.GetShape();
   EXPECT_EQ(s2.size(), 3u);
   EXPECT_EQ(s2[0], 2u);
   EXPECT_EQ(s2[1], 3u);
   EXPECT_EQ(s2[2], 1u);
}

TEST(RTensor, Squeeze)
{
   RTensor<int> x({1, 2, 3});
   x.Squeeze();
   const auto s = x.GetShape();
   EXPECT_EQ(s.size(), 2u);
   EXPECT_EQ(s[0], 2u);
   EXPECT_EQ(s[1], 3u);

   RTensor<int> x1({1, 2, 1, 3, 1});
   x1.Squeeze();
   const auto s1 = x1.GetShape();
   EXPECT_EQ(s1.size(), 2u);
   EXPECT_EQ(s1[0], 2u);
   EXPECT_EQ(s1[1], 3u);

   RTensor<int> x2({1, 1, 1});
   x2.Squeeze();
   const auto s2 = x2.GetShape();
   EXPECT_EQ(s2.size(), 1u);
   EXPECT_EQ(s2[0], 1u);

   RTensor<int> x3({});
   x3.Squeeze();
   const auto s3 = x3.GetShape();
   EXPECT_EQ(s3.size(), 0u);
}

TEST(RTensor, Transpose)
{
   // Layout (physical):
   // 0, 1, 2, 3, 4, 5
   float data[6] = {0, 1, 2, 3, 4, 5};
   RTensor<float> x(data, {2, 3});

   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   float count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(i, j) = count;
         count++;
      }
   }

   // Layout (logical):
   // 0, 3
   // 1, 4
   // 2, 5
   x.Transpose();
   EXPECT_EQ(x.GetShape().size(), 2u);
   EXPECT_EQ(x.GetShape()[0], 3u);
   EXPECT_EQ(x.GetShape()[1], 2u);
   count = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(j, i) = count;
         count++;
      }
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
   // 0, 1, 2, 3, 4, 5
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
   // 0, 1, 2, 3, 4, 5
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

TEST(RTensor, IteratorInterface)
{
   // Layout (logical):
   // 0, 1, 2
   // 3, 4, 5
   RTensor<float> x({2, 3}, MemoryOrder::RowMajor);
   float c = 0.0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(i, j) = c;
         c++;
      }
   }
   // Layout (physical):
   // 0, 1, 2, 3, 4, 5
   c = 0.0;
   for (auto &v : x) {
      EXPECT_EQ(v, c);
      c++;
   }

   // Layout (physical):
   // 1, 2, 3, 4, 5, 6
   for (auto &v : x) {
      v++;
   }
   // Layout (logical):
   // 1, 2, 3
   // 4, 5, 6
   c = 1.0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         x(i, j) = c;
         c++;
      }
   }

   // Layout (physical):
   // 0, 1, 2, 3, 4, 5
   RTensor<float> x2({2, 3}, MemoryOrder::ColumnMajor);
   c = 0.0;
   for (auto &v : x2) {
      v = c;
      c++;
   }
   // Layout (logical):
   // 0, 2, 4
   // 1, 3, 5
   std::vector<float> c2 = {0, 2, 4, 1, 3, 5};
   size_t k = 0;
   for (size_t i = 0; i < 2; i++) {
      for (size_t j = 0; j < 3; j++) {
         EXPECT_EQ(x2(i, j), c2[k]);
         k++;
      }
   }
}

TEST(RTensor, Slice)
{
   // Data layout:
   // [ [ 0, 1, 2 ], [ 3, 4, 5 ] ]
   RTensor<float> x({2, 3});
   float c = 0.0;
   for (auto &v : x) {
      v = c;
      c++;
   }

   auto s1 = x.Slice({0, -1});
   std::vector<float> ref1 = {0, 1, 2};
   EXPECT_EQ(s1.GetSize(), ref1.size());
   EXPECT_EQ(s1.GetShape().size(), 1u);
   EXPECT_EQ(s1.GetShape()[0], 3u);
   const auto d1 = s1.GetData();
   for (size_t i = 0; i < ref1.size(); i++) {
      EXPECT_EQ(ref1[i], d1[i]);
   }

   auto s2 = x.Slice(-1, 2);
   std::vector<float> ref2 = {2, 5};
   EXPECT_EQ(s2.GetSize(), ref2.size());
   EXPECT_EQ(s2.GetShape().size(), 1u);
   EXPECT_EQ(s2.GetShape()[0], 2u);
   const auto d2 = s2.GetData();
   for (size_t i = 0; i < ref2.size(); i++) {
      EXPECT_EQ(ref2[i], d2[i]);
   }

   auto s3 = x.Slice(-1, -1);
   std::vector<float> ref3 = {0, 1, 2, 3, 4, 5};
   EXPECT_EQ(s3.GetSize(), ref3.size());
   EXPECT_EQ(s3.GetShape().size(), 2u);
   EXPECT_EQ(s3.GetShape()[0], 2u);
   EXPECT_EQ(s3.GetShape()[1], 3u);
   const auto d3 = s3.GetData();
   for (size_t i = 0; i < ref3.size(); i++) {
      EXPECT_EQ(ref3[i], d3[i]);
   }

   auto s4 = x.Slice({1, 1});
   std::vector<float> ref4 = {4};
   EXPECT_EQ(s4.GetSize(), ref4.size());
   EXPECT_EQ(s4.GetShape().size(), 1u);
   EXPECT_EQ(s4.GetShape()[0], 1u);
   const auto d4 = s4.GetData();
   for (size_t i = 0; i < ref4.size(); i++) {
      EXPECT_EQ(ref4[i], d4[i]);
   }

   RTensor<float> x2({2, 2, 2});
   c = 0.0;
   for (auto &v : x2) {
      v = c;
      c++;
   }

   // Data layout:
   // [ [ [ 0, 1 ], [ 2, 3 ] ], [ [ 4, 5 ], [ 6, 7 ] ] ]
   // Selected:
   // [ [ 2, 3 ], [ 6, 7 ] ]
   auto s5 = x2.Slice({-1, 1, -1});
   std::vector<float> ref5 = {2, 3, 6, 7};
   EXPECT_EQ(s5.GetSize(), ref5.size());
   EXPECT_EQ(s5.GetShape().size(), 2u);
   EXPECT_EQ(s5.GetShape()[0], 2u);
   EXPECT_EQ(s5.GetShape()[1], 2u);
   const auto d5 = s5.GetData();
   for (size_t i = 0; i < ref5.size(); i++) {
      EXPECT_EQ(ref5[i], d5[i]);
   }
}
