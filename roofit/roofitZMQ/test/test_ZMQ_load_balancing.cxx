/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit_ZMQ/ZeroMQSvc.h"
#include "RooFit_ZMQ/ZeroMQPoller.h"

#include <RooFit/Common.h>

#include "gtest/gtest.h"

#include <unistd.h> // fork, usleep

class ZMQPushPullTest : public ::testing::Test {
protected:
   std::size_t N_children = 4;
   std::size_t max_sends = 20;

   void SetUp() override
   {
      for (std::size_t i = 0; i < N_children; ++i) {
         do {
            child_pid = fork();
         } while (child_pid == -1); // retry if fork fails
         if (child_pid == 0) {      // child
            child_id = i;
            break;
         } else {
            child_pids.push_back(child_pid);
         }
      }

      if (child_pid > 0) { // parent
         pusher.reset(zmqSvc().socket_ptr(zmq::socket_type::push));
         pusher->bind("ipc://" + RooFit::tmpPath() + "ZMQ_test_push_pull_P2C.ipc");
      } else { // child
         puller.reset(zmqSvc().socket_ptr(zmq::socket_type::pull));
         puller->connect("ipc://" + RooFit::tmpPath() + "ZMQ_test_push_pull_P2C.ipc");

         poller.register_socket(*puller, zmq::event_flags::pollin);
      }
   }

   void TearDown() override
   {
      if (child_pid > 0) { // parent
         // wait for children
         int status = 0;
         pid_t pid;
         for (pid_t child_pid_i : child_pids) {
            do {
               pid = waitpid(child_pid_i, &status, 0);
            } while (-1 == pid && EINTR == errno); // retry on interrupted system call
            if (0 != status) {
               if (WIFEXITED(status)) {
                  printf("exited, status=%d\n", WEXITSTATUS(status));
               } else if (WIFSIGNALED(status)) {
                  printf("killed by signal %d\n", WTERMSIG(status));
               } else if (WIFSTOPPED(status)) {
                  printf("stopped by signal %d\n", WSTOPSIG(status));
               } else if (WIFCONTINUED(status)) {
                  printf("continued\n");
               }
            }
            if (-1 == pid) {
               throw std::runtime_error(std::string("waitpid, errno ") + std::to_string(errno));
            }
         }
         pusher.reset(nullptr);
         zmqSvc().close_context();
      } else { // child
         puller.reset(nullptr);
         zmqSvc().close_context();
         _Exit(0);
      }
   }

   void run_parent()
   {
      // start test
      usleep(1000); // wait a second so that all pull sockets are connected for round-robin distribution
      // if you don't wait a second above, the push socket will "round-robin" all the messages to just one or two
      // connected sockets
      for (std::size_t ix = 0; ix < max_sends; ++ix) {
         zmqSvc().send<int>(*pusher, 0);
      }
      for (std::size_t ix = 0; ix < N_children; ++ix) {
         // end by sending some 1's to all children, to let them know the sending is over
         zmqSvc().send<int>(*pusher, 1);
      }
   }

   void run_child()
   {
      std::size_t count = 0;
      for (std::size_t ix = 0; ix < max_sends; ++ix) {
         auto r = poller.poll(2000);
         if (r.empty()) {
            printf("poller of child %d timed out after 2 seconds\n", child_id);
            break;
         }
         auto value = zmqSvc().receive<int>(*puller, zmq::recv_flags::dontwait);
         usleep(200); // "do some work"
         printf("value on child %d: %d\n", child_id, value);
         if (value == 1) {
            printf("child %d got value %d, done here\n", child_id, value);
            break;
         }
         ++count;
      }
      printf("child %d got %zu values\n", child_id, count);
   }

   pid_t child_pid{0};
   int child_id = -1;
   std::vector<pid_t> child_pids;
   ZmqLingeringSocketPtr<> pusher, puller;
   ZeroMQPoller poller;
};

/// This test shows how push-pull is unsuited for load balancing; messages are just sent to the first available pull
/// socket without any dynamic load balancing
TEST_F(ZMQPushPullTest, demoRoundRobin)
{
   if (child_pid > 0) {
      run_parent();
   } else {
      run_child();
   }
}

/// This test tries to see whether push-pull can be made to work as a bit of a load balancer, using a low HWM at the
/// receiver
TEST_F(ZMQPushPullTest, demoHWM1LoadBalancing)
{
   if (child_pid > 0) {
      run_parent();
   } else {
      puller->set(zmq::sockopt::rcvhwm, 1);
      run_child();
   }
}
