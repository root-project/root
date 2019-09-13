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

#include <iostream>

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
         JobManager::instance()->messenger().send_from_master_to_queue(M2Q::terminate);
         shutdown_processes();
      }
   } catch (const std::exception& e) {
      std::cerr << "WARNING: something in ProcessManager::terminate threw an exception! Original exception message:\n" << e.what() << std::endl;
   }
}


void ProcessManager::terminate_workers() {
   if (is_queue()) {
      for(std::size_t worker_ix = 0; worker_ix < _N_workers; ++worker_ix) {
         JobManager::instance()->messenger().send_from_queue_to_worker(worker_ix, Q2W::terminate);
      }
      JobManager::instance()->messenger().close_queue_worker_connections();
   }
}


void ProcessManager::shutdown_processes() {
   if (is_master()) {
      // first wait for the workers that will be terminated by the queue
      for (auto pid : worker_pids) {
         wait_for_child(pid, true, 5);
      }
      // then wait for the queue
      wait_for_child(queue_pid, true, 5);
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
