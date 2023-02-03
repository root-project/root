#include "ntuple_test.hxx"

template <>
thread_local RPageAllocatorCache<RPageAllocatorHeap, 2>::RPageCache
   RPageAllocatorCache<RPageAllocatorHeap, 2>::gCache{};

TEST(Pages, Allocation)
{
   RPageAllocatorHeap allocator;
   auto page = allocator.NewPage(42, 4, 16);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(16U, page.GetMaxElements());
   EXPECT_EQ(0U, page.GetNElements());
   EXPECT_EQ(0U, page.GetNBytes());
   allocator.DeletePage(page);

   auto testRPageAllocatorCache = []() {
      RPageAllocatorCache<RPageAllocatorHeap, 2> cache;
      auto page1 = cache.NewPage(42, 4, 16);
      auto page2 = cache.NewPage(42, 4, 32);
      auto page3 = cache.NewPage(42, 4, 64);
      EXPECT_TRUE(!page1.IsNull() && !page2.IsNull() && !page3.IsNull());
      EXPECT_EQ(64U, page1.GetMaxBytes());
      EXPECT_EQ(128U, page2.GetMaxBytes());
      EXPECT_EQ(256U, page3.GetMaxBytes());
      cache.DeletePage(page3);
      cache.DeletePage(page2);
      cache.DeletePage(page1);
      auto page4 = cache.NewPage(42, 4, 16);
      EXPECT_EQ(page4.GetBuffer(), page2.GetBuffer());
      cache.DeletePage(page4);
      auto page5 = cache.NewPage(42, 4, 16);
      EXPECT_EQ(page5.GetBuffer(), page2.GetBuffer());
      cache.DeletePage(page5);
   };
   // RPageAllocatorCache keeps a thread_local queue of allocations to be reused
   std::thread t1(testRPageAllocatorCache);
   std::thread t2(testRPageAllocatorCache);
   t1.join();
   t2.join();
}

TEST(Pages, Pool)
{
   RPagePool pool;

   auto page = pool.GetPage(0, 0);
   EXPECT_TRUE(page.IsNull());
   pool.ReturnPage(page); // should not crash

   RPage::RClusterInfo clusterInfo(2, 40);
   page = RPage(1, &page, 1, 10);
   page.GrowUnchecked(10);
   EXPECT_EQ(page.GetMaxElements(), page.GetNElements());
   page.SetWindow(50, clusterInfo);
   EXPECT_FALSE(page.IsNull());
   unsigned int nCallDeleter = 0;
   pool.RegisterPage(page, RPageDeleter([&nCallDeleter](const RPage & /*page*/, void * /*userData*/) {
      nCallDeleter++;
   }));

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
   EXPECT_EQ(0U, nCallDeleter);
   pool.ReturnPage(page);
   EXPECT_EQ(0U, nCallDeleter);
   pool.ReturnPage(page);
   EXPECT_EQ(1U, nCallDeleter);
   page = pool.GetPage(1, 55);
   EXPECT_TRUE(page.IsNull());
}
