#if __cplusplus < 201703L && !defined(_MSC_VER)

#include "ROOT/RNotFn.hxx"

#include "gtest/gtest.h"

bool retTrue(){return true;}
bool retFlip(bool b){return !b;}

TEST(NotFn, Simple)
{
   EXPECT_TRUE(true);
   EXPECT_TRUE(std::not_fn([]() { return false; })());

   int a = 1;
   EXPECT_TRUE(std::not_fn([a]() { return a == 0; })());

   int *p = nullptr;
   EXPECT_TRUE(std::not_fn([&p]() { return p; })());

   EXPECT_FALSE(std::not_fn(retTrue)());

   EXPECT_TRUE(std::not_fn(retFlip)(true));
   EXPECT_FALSE(std::not_fn(retFlip)(false));

   EXPECT_TRUE(std::not_fn(std::not_fn(retFlip))(false));
   EXPECT_FALSE(std::not_fn(std::not_fn(retFlip))(true));

}

#endif
