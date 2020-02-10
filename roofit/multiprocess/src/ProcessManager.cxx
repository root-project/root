/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <csignal>  // for sigterm handling on child processes
#include <cstring>  // for strsignal
#include <iostream>
#include <unordered_set>

#include "RooFit/MultiProcess/util.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/ProcessManager.h"

namespace RooFit {
namespace MultiProcess {


ProcessManager::ProcessManager(std::size_t N_workers) : _N_workers(N_workers) {
   // This class defines three types of processes:
   // 1. master: the initial main process. It defines and enqueues tasks
   //    and processes results.
   // 2. workers: a pool of processes that will try to take tasks from the
   //    queue. These are forked from master.
   // 3. queue: communication between the other types (necessarily) goes
   //    through this process. This process runs the queue_loop and
   //    maintains the queue of tasks.

   initialize_processes();
}

ProcessManager::~ProcessManager()
{
   terminate();
}


// static member initialization
volatile sig_atomic_t ProcessManager::_sigterm_received = 0;

// static function
void ProcessManager::handle_sigterm(int signum) {
   // We need this to tell the children to die, because we can't talk
   // to them anymore during JobManager destruction, because that kills
   // the Messenger first. We do that with SIGTERMs. The sigterm_received()
   // should be checked in message loops to stop them when it's true.
   _sigterm_received = 1;
   std::cout << "handled " << strsignal(signum) << " on PID " << getpid() << std::endl;
}

// static function
bool ProcessManager::sigterm_received() {
   if (_sigterm_received > 0) {
      return true;
   } else {
      return false;
   }
}

void ProcessManager::initialize_processes(bool cpu_pinning) {
   // Initialize processes;
   // ... first workers:
   worker_pids.resize(_N_workers);
   pid_t child_pid {};
   for (std::size_t ix = 0; ix < _N_workers; ++ix) {
      child_pid = fork();
      if (!child_pid) {  // we're on the worker
         _worker_id = ix;
         break;
      } else {          // we're on master
         worker_pids[ix] = child_pid;
      }
   }

   // ... then queue:
   if (child_pid) {  // we're on master
      queue_pid = fork();
      if (!queue_pid) { // we're now on queue
         _is_queue = true;
      } else {
         _is_master = true;
      }
   }

   // set the sigterm handler on the child processes
   if (!_is_master) {
      struct sigaction sa;
      memset (&sa, '\0', sizeof(sa));
      sa.sa_handler = ProcessManager::handle_sigterm;

      if (sigaction(SIGTERM, &sa, NULL) < 0) {
         std::perror("sigaction failed");
         std::exit(1);
      }
   }

   if (cpu_pinning) {
#if defined(__APPLE__)
   #ifndef NDEBUG
      static bool affinity_warned = false;
      if (is_master() & !affinity_warned) {
         std::cout << "CPU affinity cannot be set on macOS" << std::endl;
         affinity_warned = true;
      }
   #endif  // NDEBUG
#elif defined(_WIN32)
   #ifndef NDEBUG
      if (is_master()) std::cerr << "WARNING: CPU affinity setting not implemented on Windows, continuing..." << std::endl;
   #endif  // NDEBUG
#else
      cpu_set_t mask;
      // zero all bits in mask
      CPU_ZERO(&mask);
      // set correct bit
      std::size_t set_cpu;
      if (is_master()) {
       set_cpu = N_workers + 1;
      } else if (is_queue()) {
       set_cpu = N_workers;
      } else {
       set_cpu = worker_id;
      }
      CPU_SET(set_cpu, &mask);
   #ifndef NDEBUG
      // sched_setaffinity returns 0 on success
      if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
       std::cerr << "WARNING: Could not set CPU affinity, continuing..." << std::endl;
      } else {
       std::cerr << "CPU affinity set to cpu " << set_cpu << " in process " << getpid() << std::endl;
      }
   #endif  // NDEBUG
#endif
   }

//      identify_processes();

   initialized = true;
}

bool ProcessManager::is_initialized() const {
   return initialized;
}


void ProcessManager::terminate() noexcept {
   try {
      if (is_master()) {
//         JobManager::instance()->messenger().send_from_master_to_queue(M2Q::terminate);
         shutdown_processes();
      }
   } catch (const std::exception& e) {
      std::cerr << "WARNING: something in ProcessManager::terminate threw an exception! Original exception message:\n" << e.what() << std::endl;
   }
}


//void ProcessManager::terminate_workers() {
//   if (is_queue()) {
//      for(std::size_t worker_ix = 0; worker_ix < _N_workers; ++worker_ix) {
//         JobManager::instance()->messenger().send_from_queue_to_worker(worker_ix, Q2W::terminate);
//      }
//      JobManager::instance()->messenger().close_queue_worker_connections();
//   }
//}


int chill_wait() {
   int status = 0;
   pid_t pid;
   do {
      pid = wait(&status);
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

   return pid;
}

void ProcessManager::shutdown_processes() {
   if (is_master()) {
      // terminate all children
      std::unordered_set<pid_t> children;
      children.insert(queue_pid);
      kill(queue_pid, SIGTERM);
      for (auto pid : worker_pids) {
         kill(pid, SIGTERM);
         children.insert(pid);
      }
      // then wait for them to actually die and clean out the zombies
      while (!children.empty()) {
         pid_t pid = chill_wait();
         std::cout << "chillly waited for PID " << pid << std::endl;
         children.erase(pid);
      }
   }

   initialized = false;
   _is_master = false;
}


// Getters

bool ProcessManager::is_master() const {
   return _is_master;
}

bool ProcessManager::is_queue() const {
   return _is_queue;
}

bool ProcessManager::is_worker() const {
   return !(_is_master || _is_queue);
}

std::size_t ProcessManager::worker_id() const {
   return _worker_id;
}

std::size_t ProcessManager::N_workers() const {
   return _N_workers;
}


// Debugging

void ProcessManager::identify_processes() const {
   // identify yourselves (for debugging)
   if (!(_is_master || _is_queue)) {
      std::cout << "I'm a worker, PID " << getpid() << std::endl;
   } else if (_is_master) {
      std::cout << "I'm master, PID " << getpid() << std::endl;
   } else if (_is_queue) {
      std::cout << "I'm queue, PID " << getpid() << std::endl;
   }
}

} // namespace MultiProcess
} // namespace RooFit
