#include "liburing.h"
#include "liburing/io_uring.h"

#include "gtest/gtest.h"

TEST(Liburing, MakeRing)
{
   struct io_uring ring;
   int ret = io_uring_queue_init(
      8 /* queue depth */,
      &ring,
      0 /* no setup flags */
   );
   EXPECT_EQ(ret, 0); // ring setup succeeded 
}
