#include "io_test.hxx"

#include "ROOT/RIoUring.hxx"
#include "ROOT/RRawFileUnix.hxx"

using RIoUring = ROOT::Internal::RIoUring;
using RRawFileUnix = ROOT::Internal::RRawFileUnix;

TEST(RIoUring, Basics)
{
   // successfully construct a ring with queue depth 4
   RIoUring ring(4);
}

TEST(RIoUring, IsAvailable)
{
   ASSERT_TRUE(RIoUring::IsAvailable());
}

TEST(RRawFileUnix, ReadV)
{
   using RIOVec = RRawFile::RIOVec;
   auto file = "test_uring_readv";
   auto filesize = 2 << 20;
   FileRaii fileGuard(file, std::string(filesize, 'a')); // ~2MB
   auto f = RRawFileUnix::Create(file);

   auto make_iovecs = [&](int num) -> std::vector<RIOVec> {
      std::vector<RIOVec> iovecs;
      for (int i = 0; i < num; ++i) {
         RIOVec io;
         io.fBuffer = malloc(4096 * 9);
         io.fOffset = std::rand() % filesize;
         io.fSize = 4096 * 8;
         iovecs.push_back(io);
      }
      return iovecs;
   };

   auto nReq = 128;

   auto iovecs = make_iovecs(nReq);
   // auto t1 = std::chrono::high_resolution_clock::now();
   f->ReadV(iovecs.data(), nReq);
   // auto t2 = std::chrono::high_resolution_clock::now();
   // std::cout << std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() << " microseconds\n";
   for (auto iovec: iovecs) {
      for (std::size_t i = 0; i < iovec.fOutBytes; ++i) {
         EXPECT_EQ('a', ((unsigned char*)iovec.fBuffer)[i]);
      }
      free(iovec.fBuffer);
   }
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
