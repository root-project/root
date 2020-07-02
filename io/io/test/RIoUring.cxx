#include "ROOT/RIoUring.hxx"
using RIoUring = ROOT::Internal::RIoUring; 

#include "gtest/gtest.h"

TEST(RIoUring, Basics)
{
   // successfully construct a ring with queue depth 4
   RIoUring ring(4);
}

TEST(RIoUring, IsAvailable)
{
   ASSERT_TRUE(RIoUring::IsAvailable());
}

TEST(RawUring, NopRoundTrip)
{
   struct io_uring ring;
   int ret = io_uring_queue_init(
      4 /* queue depth */,
      &ring,
      0 /* no setup flags */
   );

   // ring setup succeeded
   ASSERT_EQ(ret, 0);

   // can make sqes
   struct io_uring_sqe *sqe;
   sqe = io_uring_get_sqe(&ring);
   ASSERT_NE(sqe, (io_uring_sqe*) NULL);

   // can submit sqes to the ring
   io_uring_prep_nop(sqe);
   unsigned long udata = 42;
   io_uring_sqe_set_data(sqe, (void*) udata);
   ret = io_uring_submit(&ring);
   // submitted a single sqe
   ASSERT_EQ(ret, 1);

   // successful wait for cqe
   struct io_uring_cqe *cqe;
   ret = io_uring_wait_cqe(&ring, &cqe);
   ASSERT_EQ(ret, 0);
   // cqe userdata matches sqe userdata
   ASSERT_EQ((unsigned long) io_uring_cqe_get_data(cqe), udata);

   io_uring_cqe_seen(&ring, cqe);
   io_uring_queue_exit(&ring);
}
