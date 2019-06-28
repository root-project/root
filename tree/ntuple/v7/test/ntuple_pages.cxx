#include "gtest/gtest.h"

#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>

TEST(Pages, Allocation)
{
   ROOT::Experimental::Detail::RPageAllocatorHeap allocator;

   auto page = allocator.NewPage(42, 4, 16);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(64U, page.GetCapacity());
   EXPECT_EQ(0U, page.GetNElements());
   EXPECT_EQ(0U, page.GetSize());
   allocator.DeletePage(page);
}

TEST(Pages, Pool)
{
   ROOT::Experimental::Detail::RPagePool pool;

   auto page = pool.GetPage(0, 0);
   EXPECT_TRUE(page.IsNull());
   EXPECT_FALSE(pool.ReturnPage(page));

   ROOT::Experimental::Detail::RPage::RClusterInfo clusterInfo;
   page = ROOT::Experimental::Detail::RPage(1, &page, 10, 1);
   EXPECT_NE(nullptr, page.TryGrow(10));
   page.SetWindow(50, clusterInfo);
   EXPECT_FALSE(page.IsNull());
   pool.RegisterPage(page);

   page = pool.GetPage(0, 0);
   EXPECT_TRUE(page.IsNull());
   page = pool.GetPage(0, 55);
   EXPECT_TRUE(page.IsNull());
   page = pool.GetPage(1, 55);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(50U, page.GetRangeFirst());
   EXPECT_EQ(59U, page.GetRangeLast());

   EXPECT_FALSE(pool.ReturnPage(page));
   EXPECT_TRUE(pool.ReturnPage(page));
   page = pool.GetPage(1, 55);
   EXPECT_TRUE(page.IsNull());
}
