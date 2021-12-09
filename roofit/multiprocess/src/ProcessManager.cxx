/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/util.h"

#include <cstring>    // for strsignal
#include <sys/wait.h> // for wait
#include <iostream>
#include <unordered_set>

namespace RooFit {
namespace MultiProcess {

/// \class ProcessManager
/// \brief Fork processes for queue and workers
///
/// This class manages three types of processes:
/// 1. master: the initial main process. It defines and enqueues tasks
///    and processes results.
/// 2. workers: a pool of processes that will try to take tasks from the
///    queue. These are forked from master.
/// 3. queue: This process runs the queue_loop and maintains the queue of
///    tasks. It is also forked from master.
///
/// \param N_workers Number of worker processes to spawn.
ProcessManager::ProcessManager(std::size_t N_workers) : N_workers_(N_workers)
{
   // Note: zmq context is automatically created in the ZeroMQSvc class and maintained as singleton,
   // but we must close any possibly existing state before reusing it. This assumes that our Messenger
   // is the only user of ZeroMQSvc and that there is only one Messenger at a time. Beware that
   // this must be designed more carefully if either of these assumptions change! Note also that this
   // call must be done before the ProcessManager forks new processes, otherwise the master process'
   // context that will be cloned to all forked processes will be closed multiple times, which will
   // hang, because the ZeroMQ context creates threads and these will not be cloned along with the
   // fork. See the ZeroMQ documentation for more details on this. In principle, one could design this
   // in a more finegrained way by keeping the context on the master process and only recreating it
   // on child processes (while avoiding calling the destructor on the child processes!). This
   // approach may offer more flexibility if this is needed in the future.
   zmqSvc().close_context();
   initialize_processes();
}

ProcessManager::~ProcessManager()
{
   if (is_master()) {
      terminate();
   } else {
      wait_for_sigterm_then_exit();
   }
}

// static member initialization
volatile sig_atomic_t ProcessManager::sigterm_received_ = 0;

// static function
/// We need this to tell the children to die, because we can't talk
/// to them anymore during JobManager destruction, because that kills
/// the Messenger first. We do that with SIGTERMs. The sigterm_received()
/// should be checked in message loops to stop them when it's true.
void ProcessManager::handle_sigterm(int /*signum*/)
{
   sigterm_received_ = 1;
}

// static function
bool ProcessManager::sigterm_received()
{
   if (sigterm_received_ > 0) {
      return true;
   } else {
      return false;
   }
}

pid_t fork_and_handle_errors()
{
   pid_t child_pid = fork();
   int retries = 0;
   while (child_pid == -1) {
      if (retries < 3) {
         ++retries;
         printf("fork returned with error number %d, retrying after 1 second...\n", errno);
         sleep(1);
         child_pid = fork();
      } else {
         printf("fork returned with error number %d\n", errno);
         throw std::runtime_error("fork returned with error 3 times, aborting!");
      }
   }
   return child_pid;
}

/// \brief Fork processes and activate CPU pinning
///
/// \param cpu_pinning Activate CPU pinning if true. Effective on Linux only.
void ProcessManager::initialize_processes(bool cpu_pinning)
{
   // Initialize processes;
   // ... first workers:
   worker_pids_.resize(N_workers_);
   pid_t child_pid{};
   for (std::size_t ix = 0; ix < N_workers_; ++ix) {
      child_pid = fork_and_handle_errors();
      if (!child_pid) { // we're on the worker
         is_worker_ = true;
         worker_id_ = ix;
         break;
      } else { // we're on master
         worker_pids_[ix] = child_pid;
      }
   }

   // ... then queue:
   if (child_pid) { // we're on master
      queue_pid_ = fork_and_handle_errors();
      if (!queue_pid_) { // we're now on queue
         is_queue_ = true;
      } else {
         is_master_ = true;
      }
   }

   // set the sigterm handler on the child processes
   if (!is_master_) {
      struct sigaction sa;
      memset(&sa, '\0', sizeof(sa));
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
#endif // NDEBUG
#elif defined(_WIN32)
#ifndef NDEBUG
      if (is_master())
         std::cerr << "WARNING: CPU affinity setting not implemented on Windows, continuing..." << std::endl;
#endif // NDEBUG
#else
      cpu_set_t mask;
      // zero all bits in mask
      CPU_ZERO(&mask);
      // set correct bit
      std::size_t set_cpu;
      if (is_master()) {
         set_cpu = N_workers() + 1;
      } else if (is_queue()) {
         set_cpu = N_workers();
      } else {
         set_cpu = worker_id();
      }
      CPU_SET(set_cpu, &mask);
#ifndef NDEBUG
      // sched_setaffinity returns 0 on success
      if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
         std::cerr << "WARNING: Could not set CPU affinity, continuing..." << std::endl;
      } else {
         std::cerr << "CPU affinity set to cpu " << set_cpu << " in process " << getpid() << std::endl;
      }
#endif // NDEBUG
#endif
   }

#ifndef NDEBUG
   identify_processes();
#endif // NDEBUG

   initialized_ = true;
}

bool ProcessManager::is_initialized() const
{
   return initialized_;
}

/// Shutdown forked processes if on master and if this process manager is initialized
void ProcessManager::terminate() noexcept
{
   try {
      if (is_master() && is_initialized()) {
         shutdown_processes();
      }
   } catch (const std::exception &e) {
      std::cerr << "WARNING: something in ProcessManager::terminate threw an exception! Original exception message:\n"
                << e.what() << std::endl;
   }
}

void ProcessManager::wait_for_sigterm_then_exit()
{
   if (!is_master()) {
      while (!sigterm_received()) {
      }
      std::_Exit(0);
   }
}

int chill_wait()
{
   int status = 0;
   pid_t pid;
   do {
      pid = wait(&status);
   } while (-1 == pid && EINTR == errno); // retry on interrupted system call

   if (0 != status) {
      if (WIFEXITED(status)) {
         printf("exited, status=%d\n", WEXITSTATUS(status));
      } else if (WIFSIGNALED(status)) {
         if (WTERMSIG(status) != SIGTERM) {
            printf("killed by signal %d\n", WTERMSIG(status));
         }
      } else if (WIFSTOPPED(status)) {
         printf("stopped by signal %d\n", WSTOPSIG(status));
      } else if (WIFCONTINUED(status)) {
         printf("continued\n");
      }
   }

   if (-1 == pid) {
      if (errno == ECHILD) {
         printf("chill_wait: no children (got ECHILD error code from wait call), done\n");
      } else {
         throw std::runtime_error(std::string("chill_wait: error in wait call: ") + strerror(errno) +
                                  std::string(", errno ") + std::to_string(errno));
      }
   }

   return pid;
}

/// Shutdown forked processes if on master
void ProcessManager::shutdown_processes()
{
   if (is_master()) {
      // terminate all children
      std::unordered_set<pid_t> children;
      children.insert(queue_pid_);
      kill(queue_pid_, SIGTERM);
      for (auto pid : worker_pids_) {
         kill(pid, SIGTERM);
         children.insert(pid);
      }
      // then wait for them to actually die and clean out the zombies
      while (!children.empty()) {
         pid_t pid = chill_wait();
         children.erase(pid);
      }
   }

   initialized_ = false;
}

// Getters

bool ProcessManager::is_master() const
{
   return is_master_;
}

bool ProcessManager::is_queue() const
{
   return is_queue_;
}

bool ProcessManager::is_worker() const
{
   return is_worker_;
}

std::size_t ProcessManager::worker_id() const
{
   return worker_id_;
}

std::size_t ProcessManager::N_workers() const
{
   return N_workers_;
}

/// Print to stdout which type of process we are on and what its PID is (for debugging)
void ProcessManager::identify_processes() const
{
   if (is_worker_) {
      printf("I'm a worker, PID %d\n", getpid());
   } else if (is_master_) {
      printf("I'm master, PID %d\n", getpid());
   } else if (is_queue_) {
      printf("I'm queue, PID %d\n", getpid());
   } else {
      printf("I'm not master, queue or worker, weird! PID %d\n", getpid());
   }
}

} // namespace MultiProcess
} // namespace RooFit
