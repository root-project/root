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
#include <csignal>  // signal blocking
#include <cstring>  // strsignal()

#include <sstream>

static volatile sig_atomic_t terminated = 0;

void handle_sigterm(int signum)
{
   terminated = 1;
   std::cout << "handled signal " << strsignal(signum) << " on PID " << getpid() << std::endl;
}

std::string unique_tmp_ipc_address(const char *filename_template)
{
   assert(strlen(filename_template) < 256);
   char filename_template_mutable[256];
   strcpy(filename_template_mutable, filename_template);
   while (mkstemp(filename_template_mutable) >= 0) {
   }
   std::stringstream ss;
   ss << "ipc://" << RooFit::tmpPath() << filename_template_mutable << ".ipc";
   return ss.str();
}

TEST(Polling, doublePoll)
{
   auto M2C_address = unique_tmp_ipc_address("ZMQ_test_fork_polling_M2C_XXXXXX");
   auto C2M_address = unique_tmp_ipc_address("ZMQ_test_fork_polling_C2M_XXXXXX");
   pid_t child_pid{0};
   do {
      child_pid = fork();
   } while (child_pid == -1); // retry if fork fails

   if (child_pid > 0) { // master
      sigset_t sigmask, sigmask_old;
      sigemptyset(&sigmask);
      sigaddset(&sigmask, SIGCHLD);
      sigprocmask(SIG_BLOCK, &sigmask, &sigmask_old);

      ZmqLingeringSocketPtr<> pusher, puller;
      pusher.reset(zmqSvc().socket_ptr(zmq::socket_type::push));
      pusher->bind(M2C_address);
      puller.reset(zmqSvc().socket_ptr(zmq::socket_type::pull));
      puller->bind(C2M_address);

      ZeroMQPoller poller1, poller2;
      poller1.register_socket(*puller, zmq::event_flags::pollin);
      poller2.register_socket(*puller, zmq::event_flags::pollin);

      // start test
      zmqSvc().send(*pusher, std::string("breaker breaker"));

      auto result1a = poller1.poll(-1);
      auto result1b = poller1.poll(-1);
      auto result2 = poller2.poll(-1);
      EXPECT_EQ(result1a.size(), result1b.size());
      EXPECT_EQ(result1a.size(), result2.size());

      auto receipt = zmqSvc().receive<int>(*puller, zmq::recv_flags::dontwait);

      EXPECT_EQ(receipt, 1212);

      kill(child_pid, SIGTERM);

      pusher.reset(nullptr);
      puller.reset(nullptr);
      zmqSvc().close_context(); // if you don't close context in parent process as well, the next repeat will hang

      // wait for child
      int status = 0;
      pid_t pid;
      do {
         pid = waitpid(child_pid, &status, 0);
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

      sigprocmask(SIG_SETMASK, &sigmask_old, nullptr);
   } else { // child
      sigset_t sigmask, sigmask_old;
      sigemptyset(&sigmask);
      sigaddset(&sigmask, SIGTERM);
      sigprocmask(SIG_BLOCK, &sigmask, &sigmask_old);

      struct sigaction sa;
      memset(&sa, '\0', sizeof(sa));
      sa.sa_handler = handle_sigterm;

      if (sigaction(SIGTERM, &sa, NULL) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }

      ZmqLingeringSocketPtr<> puller, pusher;
      puller.reset(zmqSvc().socket_ptr(zmq::socket_type::pull));
      puller->connect(M2C_address);
      pusher.reset(zmqSvc().socket_ptr(zmq::socket_type::push));
      pusher->connect(C2M_address);

      ZeroMQPoller poller1, poller2;
      poller1.register_socket(*puller, zmq::event_flags::pollin);
      poller2.register_socket(*puller, zmq::event_flags::pollin);

      // start test
      auto result1a = poller1.poll(-1);
      auto result1b = poller1.poll(-1);
      auto result2 = poller2.poll(-1);
      EXPECT_EQ(result1a.size(), result1b.size());
      EXPECT_EQ(result1a.size(), result2.size());

      auto receipt = zmqSvc().receive<std::string>(*puller, zmq::recv_flags::dontwait);
      if (receipt == "breaker breaker") {
         zmqSvc().send(*pusher, 1212);
      }
      // take care, don't just use _exit, it will not cleanly destroy context etc!
      // if you really need to, at least close and destroy everything properly

      sigprocmask(SIG_SETMASK, &sigmask_old, nullptr);

      while (!terminated) {
      }

      puller.reset(nullptr);
      pusher.reset(nullptr);
      zmqSvc().close_context();
      _Exit(0);
   }
}
