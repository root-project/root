/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/Channel.h"

#include "gtest/gtest.h"

#include <numeric>
#include <string>
#include <vector>

#include <csignal>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

std::pair<RooFit::MultiProcess::Channel, RooFit::MultiProcess::Channel> makeChannelPair()
{
   int fds[2];
   if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
      throw std::runtime_error("socketpair failed");
   }
   return {RooFit::MultiProcess::Channel{fds[0]}, RooFit::MultiProcess::Channel{fds[1]}};
}

std::vector<double> testPattern(std::size_t n, double offset)
{
   std::vector<double> values(n);
   std::iota(values.begin(), values.end(), offset);
   return values;
}

} // namespace

TEST(TestMPChannel, SmallFramesRoundTrip)
{
   auto channels = makeChannelPair();

   RooFit::MultiProcess::send_item(channels.first, std::size_t{42}, false);
   RooFit::MultiProcess::send_item(channels.first, 3.14, false);
   RooFit::MultiProcess::send_item(channels.first, std::string("hello"), false);

   EXPECT_EQ(RooFit::MultiProcess::receive_item<std::size_t>(channels.second), 42u);
   EXPECT_EQ(RooFit::MultiProcess::receive_item<double>(channels.second), 3.14);
   EXPECT_EQ(RooFit::MultiProcess::receive_item<std::string>(channels.second), "hello");
}

TEST(TestMPChannel, MultipartMoreFlag)
{
   auto channels = makeChannelPair();

   RooFit::MultiProcess::send_item(channels.first, 1, true);
   RooFit::MultiProcess::send_item(channels.first, 2, true);
   RooFit::MultiProcess::send_item(channels.first, 3, false);

   bool more = false;
   EXPECT_EQ(RooFit::MultiProcess::receive_item<int>(channels.second, &more), 1);
   EXPECT_TRUE(more);
   EXPECT_EQ(RooFit::MultiProcess::receive_item<int>(channels.second, &more), 2);
   EXPECT_TRUE(more);
   EXPECT_EQ(RooFit::MultiProcess::receive_item<int>(channels.second, &more), 3);
   EXPECT_FALSE(more);
}

// A frame far larger than the kernel socket buffer must be retained in the
// channel's pending-output buffer and be flushed while the receiving side
// waits for input. This exercises the partial-write machinery that production
// fits hit with large state-update and result messages.
TEST(TestMPChannel, LargeFrameExceedsSocketBuffer)
{
   auto channels = makeChannelPair();

   // 4 MB payload, well above the default AF_UNIX buffer size
   auto values = testPattern(512 * 1024, 0.);
   RooFit::MultiProcess::Message msg(values.begin(), values.end());
   RooFit::MultiProcess::send_item(channels.first, msg, false);

   // the socket cannot have accepted everything yet
   EXPECT_TRUE(channels.first.has_pending_output());

   // receiving drains the sender's pending output: Channel::wait flushes the
   // pending output of all channels in the process while waiting for input
   auto received = RooFit::MultiProcess::receive_item<RooFit::MultiProcess::Message>(channels.second);
   ASSERT_EQ(received.size(), values.size() * sizeof(double));
   const double *data = received.data<double>();
   for (std::size_t ix = 0; ix < values.size(); ++ix) {
      ASSERT_EQ(data[ix], values[ix]) << "at index " << ix;
   }
   EXPECT_FALSE(channels.first.has_pending_output());
}

// Both directions blocked at the same time: each side first queues a frame
// larger than the socket buffer, then receives the other side's frame. With
// blocking sends this would deadlock two processes; the pending-output
// buffers plus the flush-during-wait must resolve it.
TEST(TestMPChannel, BidirectionalPendingOutput)
{
   auto channels = makeChannelPair();

   auto valuesA = testPattern(512 * 1024, 0.);
   auto valuesB = testPattern(512 * 1024, 1000000.);
   RooFit::MultiProcess::Message msgA(valuesA.begin(), valuesA.end());
   RooFit::MultiProcess::Message msgB(valuesB.begin(), valuesB.end());

   RooFit::MultiProcess::send_item(channels.first, msgA, false);
   RooFit::MultiProcess::send_item(channels.second, msgB, false);
   EXPECT_TRUE(channels.first.has_pending_output());
   EXPECT_TRUE(channels.second.has_pending_output());

   auto receivedB = RooFit::MultiProcess::receive_item<RooFit::MultiProcess::Message>(channels.first);
   auto receivedA = RooFit::MultiProcess::receive_item<RooFit::MultiProcess::Message>(channels.second);

   ASSERT_EQ(receivedA.size(), valuesA.size() * sizeof(double));
   ASSERT_EQ(receivedB.size(), valuesB.size() * sizeof(double));
   EXPECT_EQ(receivedA.data<double>()[valuesA.size() - 1], valuesA.back());
   EXPECT_EQ(receivedB.data<double>()[valuesB.size() - 1], valuesB.back());
   EXPECT_FALSE(channels.first.has_pending_output());
   EXPECT_FALSE(channels.second.has_pending_output());
}

// A blocking receive must survive benign signal interruptions (profilers,
// SIGCHLD, debuggers): Channel::wait retries on EINTR instead of surfacing
// it, which is what keeps multi-frame message sequences from desynchronizing
// the wire protocol in the event loops.
TEST(TestMPChannel, BenignSignalsDoNotInterruptReceive)
{
   int fds[2];
   ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);

   pid_t child_pid = fork();
   ASSERT_NE(child_pid, -1);
   if (child_pid == 0) { // child: wait a moment, then send one frame
      close(fds[0]);
      {
         RooFit::MultiProcess::Channel channel{fds[1]};
         usleep(200000);
         RooFit::MultiProcess::send_item(channel, std::size_t{1234}, false);
      }
      std::_Exit(0);
   }

   // parent: bombard itself with SIGALRM every 10 ms while blocking in receive
   close(fds[1]);
   struct sigaction sa;
   memset(&sa, '\0', sizeof(sa));
   sa.sa_handler = [](int) {};
   ASSERT_EQ(sigaction(SIGALRM, &sa, nullptr), 0);
   itimerval timer{{0, 10000}, {0, 10000}};
   ASSERT_EQ(setitimer(ITIMER_REAL, &timer, nullptr), 0);

   {
      RooFit::MultiProcess::Channel channel{fds[0]};
      EXPECT_EQ(RooFit::MultiProcess::receive_item<std::size_t>(channel), 1234u);
   }

   itimerval stop_timer{{0, 0}, {0, 0}};
   setitimer(ITIMER_REAL, &stop_timer, nullptr);
   sa.sa_handler = SIG_DFL;
   sigaction(SIGALRM, &sa, nullptr);

   int status = -1;
   ASSERT_EQ(waitpid(child_pid, &status, 0), child_pid);
   EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// Multipart frames are only flushed on the last part, and several messages
// queued back-to-back must come out with intact boundaries and "more" flags.
TEST(TestMPChannel, QueuedMultipartMessages)
{
   auto channels = makeChannelPair();

   auto values = testPattern(128 * 1024, 0.);
   for (int repeat = 0; repeat < 3; ++repeat) {
      RooFit::MultiProcess::Message msg(values.begin(), values.end());
      RooFit::MultiProcess::send_item(channels.first, std::size_t(repeat), true);
      RooFit::MultiProcess::send_item(channels.first, msg, false);
   }

   for (int repeat = 0; repeat < 3; ++repeat) {
      bool more = false;
      auto id = RooFit::MultiProcess::receive_item<std::size_t>(channels.second, &more);
      EXPECT_EQ(id, static_cast<std::size_t>(repeat));
      EXPECT_TRUE(more);
      auto msg = RooFit::MultiProcess::receive_item<RooFit::MultiProcess::Message>(channels.second, &more);
      EXPECT_EQ(msg.size(), values.size() * sizeof(double));
      EXPECT_FALSE(more);
   }
}
