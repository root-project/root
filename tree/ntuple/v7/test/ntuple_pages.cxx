#include "gtest/gtest.h"

#include <ROOT/RPageAllocator.hxx>

TEST(Pages, Allocation)
{
   ROOT::Experimental::Detail::RPageAllocatorHeap allocator;

   auto page = allocator.NewPage(42, 4, 16);
   EXPECT_FALSE(page.IsNull());
   EXPECT_EQ(64U, page.GetCapacity());
   EXPECT_EQ(0U, page.GetNElements());
   EXPECT_EQ(0U, page.GetSize());

   allocator.DeletePage(page);
   EXPECT_TRUE(page.IsNull());
   EXPECT_EQ(0U, page.GetCapacity());
}
