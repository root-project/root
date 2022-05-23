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

#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/ProcessManager.h" // ... JobManager::process_manager()

#include "gtest/gtest.h"

void handle_sigchld(int signum)
{
   printf("handled %s on PID %d\n", strsignal(signum), getpid());
}

TEST(TestMPMessenger, Connections)
{
   struct sigaction sa;
   RooFit::MultiProcess::ProcessManager pm(4);
   if (pm.is_master()) {
      // on master, we have to handle SIGCHLD
      memset(&sa, '\0', sizeof(sa));
      sa.sa_handler = handle_sigchld;
      if (sigaction(SIGCHLD, &sa, nullptr) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }
   }
   RooFit::MultiProcess::Messenger messenger(pm);
   if (pm.is_master()) {
      // more SIGCHLD handling
      sigset_t sigmask;
      sigemptyset(&sigmask);
      sigaddset(&sigmask, SIGCHLD);
      int rc = sigprocmask(SIG_BLOCK, &sigmask, &messenger.ppoll_sigmask);
      if (rc < 0) {
         throw std::runtime_error("sigprocmask failed in TestMPMessenger.Connections");
      }
   }
   messenger.test_connections(pm);
   if (pm.is_master()) {
      // clean up signal management modifications
      sigprocmask(SIG_SETMASK, &messenger.ppoll_sigmask, nullptr);
      sa.sa_handler = SIG_DFL;
      if (sigaction(SIGCHLD, &sa, nullptr) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }
   }
}

TEST(TestMPMessenger, ConnectionsManualExit)
{
   // the point of this test is to see whether clean-up of ZeroMQ resources is done properly without calling any
   // destructors (which is what happens when you call _Exit() instead of regularly ending the program by reaching the
   // end of main()).

   struct sigaction sa;

   RooFit::MultiProcess::ProcessManager pm(4);
   if (pm.is_master()) {
      // on master, we have to handle SIGCHLD
      memset(&sa, '\0', sizeof(sa));
      sa.sa_handler = handle_sigchld;
      if (sigaction(SIGCHLD, &sa, nullptr) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }
   }
   RooFit::MultiProcess::Messenger messenger(pm);
   if (pm.is_master()) {
      // more SIGCHLD handling
      sigset_t sigmask;
      sigemptyset(&sigmask);
      sigaddset(&sigmask, SIGCHLD);
      int rc = sigprocmask(SIG_BLOCK, &sigmask, &messenger.ppoll_sigmask);
      if (rc < 0) {
         throw std::runtime_error("sigprocmask failed in TestMPMessenger.Connections");
      }
   }
   messenger.test_connections(pm);
   if (!pm.is_master()) {
      // just wait until we get terminated
      while (!RooFit::MultiProcess::ProcessManager::sigterm_received()) {
      };
      std::_Exit(0);
   } else {
      pm.terminate();
   }
   if (pm.is_master()) {
      // clean up signal management modifications
      sigprocmask(SIG_SETMASK, &messenger.ppoll_sigmask, nullptr);
      sa.sa_handler = SIG_DFL;
      if (sigaction(SIGCHLD, &sa, nullptr) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }
   }
}

#include <sys/wait.h>

TEST(TestMPMessenger, SigStop)
{
   // Originally, we could not get the debugger (lldb) to play nicely with ZeroMQ; every time we paused a process, it
   // said it crashed because of SIGSTOP, but SIGSTOP should just pause it... this test was meant to reproduce this
   // behavior and fix it, which was done by more carefully handling exceptions.
   RooFit::MultiProcess::ProcessManager pm(2);
   RooFit::MultiProcess::Messenger messenger(pm);

   if (pm.is_master()) {
      printf("first send message to queue...\n");
      messenger.send_from_master_to_queue(1);
      printf("sleep for 2 seconds...\n");
      sleep(2);
      printf("SIGSTOPping all children...\n");
      kill(pm.get_queue_pid(), SIGSTOP);
      kill(pm.get_worker_pids()[0], SIGSTOP);
      kill(pm.get_worker_pids()[1], SIGSTOP);
      printf("send another message to queue...\n");
      messenger.send_from_master_to_queue(2);
      printf("sleep for 2 seconds...\n");
      sleep(2);
      printf("SIGCONT queue and worker 1...\n");
      kill(pm.get_queue_pid(), SIGCONT);
      kill(pm.get_worker_pids()[0], SIGCONT);
      printf("sleep for 2 seconds...\n");
      sleep(2);
      printf("SIGCONT worker 2...\n");
      kill(pm.get_worker_pids()[1], SIGCONT);
      EXPECT_EQ(messenger.receive_from_queue_on_master<int>(), 3);
   }

   if (pm.is_queue()) {
      auto number = messenger.receive_from_master_on_queue<int>();
      printf("received %d on queue\n", number);
      number = messenger.receive_from_master_on_queue<int>();
      printf("received %d on queue\n", number);
      messenger.send_from_queue_to_master(3);
   }

   if (!pm.is_master()) {
      while (!pm.sigterm_received()) {
      }
   }
}

TEST(TestMPMessenger, DISABLED_StressSigStop)
{
   // The SIGSTOP test failed spuriously on CI at some point. We suspected this was due to some
   // improbable race condition caused in some place where SIGSTOP/SIGCONT crashes a process.
   // To find this crash, we bombard the processes with signals in this test.
   // We were not able to trigger the crash, so we disabled the test, but leave it in for when
   // the spurious test resurfaces.
   RooFit::MultiProcess::ProcessManager pm(2);
   RooFit::MultiProcess::Messenger messenger(pm);

   std::size_t runs = 100;
   std::size_t sig_repeats = 10000;

   if (pm.is_master()) {
      for (std::size_t ix = 0; ix < runs; ++ix) {
         //      printf("first send message to queue...\n");
         messenger.send_from_master_to_queue(1);
         //      printf("SIGSTOPping all children...\n");
         for (std::size_t jx = 0; jx < sig_repeats; ++jx) {
            kill(pm.get_queue_pid(), SIGCONT);
            kill(pm.get_worker_pids()[0], SIGCONT);
            kill(pm.get_worker_pids()[1], SIGCONT);
            kill(pm.get_queue_pid(), SIGSTOP);
            kill(pm.get_worker_pids()[0], SIGSTOP);
            kill(pm.get_worker_pids()[1], SIGSTOP);
         }
         //      printf("send another message to queue...\n");
         messenger.send_from_master_to_queue(2);

         for (std::size_t jx = 0; jx < sig_repeats; ++jx) {
            kill(pm.get_queue_pid(), SIGSTOP);
            kill(pm.get_worker_pids()[0], SIGSTOP);
            kill(pm.get_worker_pids()[1], SIGSTOP);
            kill(pm.get_queue_pid(), SIGCONT);
            kill(pm.get_worker_pids()[0], SIGCONT);
            kill(pm.get_worker_pids()[1], SIGCONT);
         }
         EXPECT_EQ(messenger.receive_from_queue_on_master<int>(), 3);
      }
   }

   if (pm.is_queue()) {
      for (std::size_t ix = 0; ix < runs; ++ix) {
         /*auto number =*/messenger.receive_from_master_on_queue<int>();
         //      printf("received %d on queue\n", number);
         /*number =*/messenger.receive_from_master_on_queue<int>();
         //      printf("received %d on queue\n", number);
         messenger.send_from_queue_to_master(3);
      }
   }

   if (!pm.is_master()) {
      while (!pm.sigterm_received()) {
      }
   }
}

// This test has been disabled in favor of the pub-sub connection check in the Messenger ctor, which is more robust
TEST(TestMPMessenger, DISABLED_StressPubSub)
{
   std::size_t N_workers = 64;
   RooFit::MultiProcess::ProcessManager pm(N_workers);
   RooFit::MultiProcess::Messenger messenger(pm);

   if (pm.is_master()) {
      messenger.publish_from_master_to_workers("bert");
      std::size_t ernies = 0;
      for (; ernies < N_workers; ++ernies) {
         auto receipt = messenger.receive_from_worker_on_master<std::string>();
         if (receipt != "ernie") {
            printf("whoops, got %s instead of ernie!\n", receipt.c_str());
            FAIL();
         }
      }
      EXPECT_EQ(ernies, N_workers);
   } else if (pm.is_worker()) {
      auto receipt = messenger.receive_from_master_on_worker<std::string>();
      if (receipt == "bert") {
         messenger.send_from_worker_to_master("ernie");
      } else {
         printf("no bert on worker %zu\n", pm.worker_id());
      }
   }
}
