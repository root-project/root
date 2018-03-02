#include "TROOT.h"
#include "ROOT/TFuture.hxx"

#include <future>

#include "gtest/gtest.h"

#ifdef R__USE_IMT

using namespace ROOT::Experimental;

TEST(TFuture, BuildFromSTLFuture)
{
   TFuture<int> f = std::async([]() { return 1; });
   ASSERT_EQ(1, f.get());
}

TEST(TFuture, BuildFromSTLFuture_ref)
{
   int a(0);
   TFuture<int &> f = std::async([&a]() -> int & { return a; });
   auto &r = f.get();
   ASSERT_EQ(&a, &(r));
}

TEST(TFuture, BuildFromSTLFuture_void)
{
   TFuture<void> f = std::async([]() {});
   f.get();
}

TEST(TFuture, BuildFromAsync)
{
   ROOT::EnableImplicitMT();
   auto f = Async([]() { return 1; });
   ASSERT_EQ(1, f.get());
}

TEST(TFuture, BuildFromAsync_ref)
{
   int a(0);
   auto f = Async([&a]() -> int & { return a; });
   auto &&r = f.get();
   ASSERT_EQ(&a, &(r));
}

TEST(TFuture, BuildFromAsync_void)
{
   auto f = Async([]() {});
   f.get();
}

#endif
