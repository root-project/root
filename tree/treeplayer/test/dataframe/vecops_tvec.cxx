#include <gtest/gtest.h>
#include <ROOT/TVec.hxx>
#include <vector>

TEST(VecOps, DefaultCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v;
   EXPECT_EQ(v.size(), 0u);
}

TEST(VecOps, InitListCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v{1,2,3};
   EXPECT_EQ(v.size(), 3u);
   EXPECT_EQ(v[0], 1);
   EXPECT_EQ(v[1], 2);
   EXPECT_EQ(v[2], 3);
}

TEST(VecOps, CopyCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v1{1,2,3};
   ROOT::Experimental::VecOps::TVec<int> v2(v1);
   EXPECT_EQ(v1.size(), 3u);
   EXPECT_EQ(v2.size(), 3u);
   EXPECT_EQ(v2[0], 1);
   EXPECT_EQ(v2[1], 2);
   EXPECT_EQ(v2[2], 3);
}

TEST(VecOps, MoveCtor)
{
   ROOT::Experimental::VecOps::TVec<int> v1{1,2,3};
   ROOT::Experimental::VecOps::TVec<int> v2(std::move(v1));
   EXPECT_EQ(v1.size(), 0u);
   EXPECT_EQ(v2.size(), 3u);
}

TEST(VecOps, MathScalar)
{
   using namespace ROOT::Experimental::VecOps;
   TVec<double> ref {1,2,3};
   TVec<double> v (ref);
   int scalar = 3;
   auto plus = v + scalar;
   auto minus = v - scalar;
   auto mult = v * scalar;
   auto div = v / scalar;

   for (unsigned int i = 0; i< v.size(); ++i) {
      EXPECT_EQ(plus[i], ref[i] + scalar);
      EXPECT_EQ(minus[i], ref[i] - scalar);
      EXPECT_EQ(mult[i], ref[i] * scalar);
      EXPECT_EQ(div[i], ref[i] / scalar);
   }

   // The same with views
   ROOT::Detail::VecOps::TVecAllocator<double> alloc(ref.data(), ref.size());
   std::vector<double, ROOT::Detail::VecOps::TVecAllocator<double>> w(ref.size(), 0, alloc);
   plus = w + scalar;
   minus = w - scalar;
   mult = w * scalar;
   div = w / scalar;

   for (unsigned int i = 0; i< v.size(); ++i) {
      EXPECT_EQ(plus[i], ref[i] + scalar);
      EXPECT_EQ(minus[i], ref[i] - scalar);
      EXPECT_EQ(mult[i], ref[i] * scalar);
      EXPECT_EQ(div[i], ref[i] / scalar);
   }
}

TEST(VecOps, MathVector)
{
   using namespace ROOT::Experimental::VecOps;
   TVec<double> ref {1,2,3};
   TVec<double> vec {3,4,5};
   TVec<double> v (ref);
   auto plus = v + vec;
   auto minus = v - vec;
   auto mult = v * vec;
   auto div = v / vec;

   for (unsigned int i = 0; i< v.size(); ++i) {
      EXPECT_EQ(plus[i], ref[i] + vec[i]);
      EXPECT_EQ(minus[i], ref[i] - vec[i]);
      EXPECT_EQ(mult[i], ref[i] * vec[i]);
      EXPECT_EQ(div[i], ref[i] / vec[i]);
   }

   // The same with 1 view
   ROOT::Detail::VecOps::TVecAllocator<double> alloc(ref.data(), ref.size());
   std::vector<double, ROOT::Detail::VecOps::TVecAllocator<double>> w(ref.size(), 0, alloc);
   plus = w + vec;
   minus = w - vec;
   mult = w * vec;
   div = w / vec;

   for (unsigned int i = 0; i< v.size(); ++i) {
      EXPECT_EQ(plus[i], ref[i] + vec[i]);
      EXPECT_EQ(minus[i], ref[i] - vec[i]);
      EXPECT_EQ(mult[i], ref[i] * vec[i]);
      EXPECT_EQ(div[i], ref[i] / vec[i]);
   }

   // The same with 2 views
   ROOT::Detail::VecOps::TVecAllocator<double> alloc2(ref.data(), ref.size());
   std::vector<double, ROOT::Detail::VecOps::TVecAllocator<double>> w2(ref.size(), 0, alloc);
   plus = w + w2;
   minus = w - w2;
   mult = w * w2;
   div = w / w2;

   for (unsigned int i = 0; i< v.size(); ++i) {
      EXPECT_EQ(plus[i], ref[i] + w2[i]);
      EXPECT_EQ(minus[i], ref[i] - w2[i]);
      EXPECT_EQ(mult[i], ref[i] * w2[i]);
      EXPECT_EQ(div[i], ref[i] / w2[i]);
   }
}

template<typename T, typename V>
void CompAndPrintTVec(ROOT::Experimental::VecOps::TVec<T> v, V w)
{
   using namespace ROOT::Experimental::VecOps;
   std::cout << v << " " << w << std::endl;
   std::cout << v + w << std::endl;
   std::cout << v - w << std::endl;
   std::cout << v * w << std::endl;
   std::cout << v / w << std::endl;
   std::cout << (v > w) << std::endl;
   std::cout << (v >= w) << std::endl;
   std::cout << (v == w) << std::endl;
   std::cout << (v <= w) << std::endl;
   std::cout << (v < w) << std::endl;

}

TEST(VecOps, PrintOps)
{
   using namespace ROOT::Experimental::VecOps;
   TVec<int> ref {1,2,3};
   TVec<int> v (ref);
   CompAndPrintTVec(v, 2.);
   CompAndPrintTVec(v, ref+2);

   ROOT::Detail::VecOps::TVecAllocator<int> alloc(ref.data(), ref.size());
   std::vector<int, ROOT::Detail::VecOps::TVecAllocator<int>> w(ref.size(), 0, alloc);

   CompAndPrintTVec(v, 2.);
   CompAndPrintTVec(v, ref+2);

}

TEST(VecOps, MathFuncs)
{
   using namespace ROOT::Experimental::VecOps;
   TVec<double> ref {1,2,3};
   TVec<double> ref2 {11,12,13};
   sqrt(ref);
   log(ref);
   sin(ref);
   cos(ref);
   asin(ref);
   acos(ref);
   tan(ref);
   atan(ref);
   sinh(ref);
   cosh(ref);
   asinh(ref);
   acosh(ref);
   tanh(ref);
   atanh(ref);

}
