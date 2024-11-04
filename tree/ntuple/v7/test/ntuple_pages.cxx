#include "ntuple_test.hxx"

TEST(Pages, Allocation)
{
   RPageAllocatorHeap allocator;

   auto page = allocator.NewPage(4, 16);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(16U, page.GetMaxElements());
   EXPECT_EQ(0U, page.GetNElements());
   EXPECT_EQ(0U, page.GetNBytes());
}

TEST(Pages, Pool)
{
   RPageAllocatorHeap allocator;
   RPagePool pool;

   {
      auto pageRef = pool.GetPage(RPagePool::RKey{0, std::type_index(typeid(void))}, 0);
      EXPECT_TRUE(pageRef.Get().IsNull());
   } // returning empty page should not crash

   RPage::RClusterInfo clusterInfo(2, 40);
   auto page = allocator.NewPage(1, 10);
   page.GrowUnchecked(10);
   EXPECT_EQ(page.GetMaxElements(), page.GetNElements());
   page.SetWindow(50, clusterInfo);
   EXPECT_FALSE(page.IsNull());

   {
      auto registeredPage = pool.RegisterPage(std::move(page), RPagePool::RKey{1, std::type_index(typeid(void))});

      {
         auto pageRef = pool.GetPage(RPagePool::RKey{0, std::type_index(typeid(void))}, 0);
         EXPECT_TRUE(pageRef.Get().IsNull());
         pageRef = pool.GetPage(RPagePool::RKey{0, std::type_index(typeid(void))}, 55);
         EXPECT_TRUE(pageRef.Get().IsNull());
         pageRef = pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(int))}, 55);
         EXPECT_TRUE(pageRef.Get().IsNull());
         pageRef = pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(void))}, 55);
         EXPECT_FALSE(pageRef.Get().IsNull());
         EXPECT_EQ(50U, pageRef.Get().GetGlobalRangeFirst());
         EXPECT_EQ(59U, pageRef.Get().GetGlobalRangeLast());
         EXPECT_EQ(10U, pageRef.Get().GetClusterRangeFirst());
         EXPECT_EQ(19U, pageRef.Get().GetClusterRangeLast());

         auto pageRef2 =
            pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(void))}, ROOT::Experimental::RClusterIndex(0, 15));
         EXPECT_TRUE(pageRef2.Get().IsNull());
         pageRef2 =
            pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(int))}, ROOT::Experimental::RClusterIndex(2, 15));
         EXPECT_TRUE(pageRef2.Get().IsNull());
         pageRef2 =
            pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(void))}, ROOT::Experimental::RClusterIndex(2, 15));
         EXPECT_FALSE(pageRef2.Get().IsNull());
      }
   }
   auto pageRef = pool.GetPage(RPagePool::RKey{1, std::type_index(typeid(void))}, 55);
   EXPECT_TRUE(pageRef.Get().IsNull());
}
