#include "ROOT/RRangeCast.hxx"

#include "gtest/gtest.h"

class ClassA {
public:
   virtual ~ClassA() {}
};
class ClassB : public ClassA {
};

TEST(RRangeCast, CStyleArrayStatic)
{

   ClassB b;
   const ClassA *arr[3] = {&b, &b, &b};
   int n = 0;
   for (auto *item : ROOT::RangeStaticCast<const ClassB *>(arr)) {
      static_assert(std::is_same<decltype(item), const ClassB *>::value,
                    "RangeStaticCast didn't convert to the right type");
      EXPECT_EQ(item, static_cast<const ClassB *>(arr[n]));
      ++n;
   }
   EXPECT_EQ(n, 3);
}

TEST(RRangeCast, CStyleArrayDynamic)
{

   ClassA a;
   ClassB b;
   const ClassA *arr[3] = {&b, &a, &b};
   int n = 0;
   for (auto *item : ROOT::RangeDynCast<const ClassB *>(arr)) {
      static_assert(std::is_same<decltype(item), const ClassB *>::value,
                    "RangeDynCast didn't convert to the right type");
      EXPECT_EQ(item, dynamic_cast<const ClassB *>(arr[n]));
      ++n;
   }
   EXPECT_EQ(n, 3);
}
