#include "ntuple_test.hxx"

TEST(Pages, Allocation)
{
   RPageAllocatorHeap allocator;

   auto page = allocator.NewPage(42, 4, 16);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(16U, page.GetMaxElements());
   EXPECT_EQ(0U, page.GetNElements());
   EXPECT_EQ(0U, page.GetNBytes());
   allocator.DeletePage(page);
}

TEST(Pages, Pool)
{
   RPageAllocatorHeap allocator;
   RPagePool pool(&allocator);

   auto page = pool.GetPage(0, 0);
   EXPECT_TRUE(page.IsNull());
   pool.ReturnPage(page); // should not crash

   RPage::RClusterInfo clusterInfo(2, 40);
   page = allocator.NewPage(1, 1, 10);
   page.GrowUnchecked(10);
   EXPECT_EQ(page.GetMaxElements(), page.GetNElements());
   page.SetWindow(50, clusterInfo);
   EXPECT_FALSE(page.IsNull());
   pool.RegisterPage(page);

   page = pool.GetPage(0, 0);
   EXPECT_TRUE(page.IsNull());
   page = pool.GetPage(0, 55);
   EXPECT_TRUE(page.IsNull());
   page = pool.GetPage(1, 55);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(50U, page.GetGlobalRangeFirst());
   EXPECT_EQ(59U, page.GetGlobalRangeLast());
   EXPECT_EQ(10U, page.GetClusterRangeFirst());
   EXPECT_EQ(19U, page.GetClusterRangeLast());

   page = pool.GetPage(1, ROOT::Experimental::RClusterIndex(0, 15));
   EXPECT_TRUE(page.IsNull());
   page = pool.GetPage(1, ROOT::Experimental::RClusterIndex(2, 15));
   EXPECT_FALSE(page.IsNull());

   pool.ReturnPage(page);
   pool.ReturnPage(page);
   pool.ReturnPage(page);
   page = pool.GetPage(1, 55);
   EXPECT_TRUE(page.IsNull());
}
