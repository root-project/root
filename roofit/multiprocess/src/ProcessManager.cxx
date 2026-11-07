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
#include "RooFit/MultiProcess/ProcessTimer.h"
#include "RooFit/MultiProcess/Config.h"

#include <thread>
#include <cstring>      // for strsignal
#include <fcntl.h>      // for fcntl, O_NONBLOCK
#include <sys/socket.h> // for socketpair
#include <sys/wait.h>   // for wait
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
   // The socketpairs used for interprocess communication must be created
   // before forking, so that all processes inherit the file descriptors of
   // the connected channels.
   create_channel_fds();
   initialize_processes();
}

ProcessManager::~ProcessManager()
{
   if (is_master()) {
      terminate();
   } else {
      wait_for_sigterm_then_exit();
   }
   close_channel_fds();
}

// static member initialization
volatile sig_atomic_t ProcessManager::sigterm_received_ = 0;
int ProcessManager::sigterm_wake_read_fd_ = -1;
int ProcessManager::sigterm_wake_write_fd_ = -1;

// static function
/// We need this to tell the children to die, because we can't talk
/// to them anymore during JobManager destruction, because that kills
/// the Messenger first. We do that with SIGTERMs. The sigterm_received()
/// should be checked in message loops to stop them when it's true.
/// The handler also writes to a self-pipe, so that a poll that is entered
/// after the flag check but before signal delivery still wakes up.
void ProcessManager::handle_sigterm(int /*signum*/)
{
   sigterm_received_ = 1;
   if (sigterm_wake_write_fd_ >= 0) {
      char byte = 't';
      // write is async-signal-safe; a full pipe just means a wake-up is already pending
      ssize_t unused = write(sigterm_wake_write_fd_, &byte, 1);
      (void)unused;
   }
}

// static function
int ProcessManager::sigterm_wake_fd()
{
   return sigterm_wake_read_fd_;
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

namespace {

/// Set FD_CLOEXEC so that the descriptor is not leaked into programs that a
/// process executes (e.g. with gSystem->Exec). Leaked duplicates of the
/// channel descriptors would keep the connections open after the owning
/// process dies, defeating the closed-connection detection in Channel.
void set_close_on_exec(int fd)
{
   int flags = fcntl(fd, F_GETFD, 0);
   if (flags == -1 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
      throw std::runtime_error(std::string("ProcessManager: could not set FD_CLOEXEC: ") + strerror(errno));
   }
}

void make_socketpair(std::array<int, 2> &fds)
{
   if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) != 0) {
      throw std::runtime_error(std::string("ProcessManager: socketpair failed: ") + strerror(errno));
   }
   set_close_on_exec(fds[0]);
   set_close_on_exec(fds[1]);
}

void close_fd_pair(std::array<int, 2> &fds, int keep = -1)
{
   for (int &fd : fds) {
      if (fd >= 0 && fd != keep) {
         close(fd);
         fd = -1;
      }
   }
}

int claim_fd(int &fd)
{
   if (fd < 0) {
      throw std::logic_error("ProcessManager: channel file descriptor already claimed or not owned by this process");
   }
   int result = fd;
   fd = -1;
   return result;
}

} // namespace

/// Create the socketpairs that connect the processes. Must be called before
/// forking; every process then keeps only the ends it needs (see
/// close_unused_channel_fds).
void ProcessManager::create_channel_fds()
{
   make_socketpair(mq_fds_);
   qw_fds_.resize(N_workers_, {{-1, -1}});
   mw_fds_.resize(N_workers_, {{-1, -1}});
   for (std::size_t ix = 0; ix < N_workers_; ++ix) {
      make_socketpair(qw_fds_[ix]);
      make_socketpair(mw_fds_[ix]);
   }
}

/// Close the channel ends that do not belong to the current process type.
void ProcessManager::close_unused_channel_fds()
{
   for (std::size_t ix = 0; ix < N_workers_; ++ix) {
      if (is_master_) {
         close_fd_pair(qw_fds_[ix]);
         close_fd_pair(mw_fds_[ix], mw_fds_[ix][0]);
      } else if (is_queue_) {
         close_fd_pair(qw_fds_[ix], qw_fds_[ix][0]);
         close_fd_pair(mw_fds_[ix]);
      } else { // worker
         close_fd_pair(qw_fds_[ix], ix == worker_id_ ? qw_fds_[ix][1] : -1);
         close_fd_pair(mw_fds_[ix], ix == worker_id_ ? mw_fds_[ix][1] : -1);
      }
   }
   if (is_master_) {
      close_fd_pair(mq_fds_, mq_fds_[0]);
   } else if (is_queue_) {
      close_fd_pair(mq_fds_, mq_fds_[1]);
   } else {
      close_fd_pair(mq_fds_);
   }
}

/// Close all channel ends still owned by this ProcessManager (i.e. not
/// claimed by a Messenger).
void ProcessManager::close_channel_fds()
{
   close_fd_pair(mq_fds_);
   for (auto &fds : qw_fds_) {
      close_fd_pair(fds);
   }
   for (auto &fds : mw_fds_) {
      close_fd_pair(fds);
   }
}

/// Hand over the master-queue channel end for the current process type.
int ProcessManager::claim_mq_fd()
{
   return claim_fd(is_master_ ? mq_fds_[0] : mq_fds_[1]);
}

/// Hand over the queue-worker channel end for the current process type.
int ProcessManager::claim_qw_fd(std::size_t worker_ix)
{
   return claim_fd(is_queue_ ? qw_fds_[worker_ix][0] : qw_fds_[worker_ix][1]);
}

/// Hand over the master-worker channel end for the current process type.
int ProcessManager::claim_mw_fd(std::size_t worker_ix)
{
   return claim_fd(is_master_ ? mw_fds_[worker_ix][0] : mw_fds_[worker_ix][1]);
}

/// \brief Fork processes and activate CPU pinning
///
/// \param cpu_pinning Activate CPU pinning if true. Effective on Linux only.
void ProcessManager::initialize_processes(bool cpu_pinning)
{
   // Initialize processes;
   // ... first workers:

   // Setup process timer master and assign pid_t 999
   if (RooFit::MultiProcess::Config::getTimingAnalysis()) ProcessTimer::setup(999);

   worker_pids_.resize(N_workers_);
   pid_t child_pid{};
   for (std::size_t ix = 0; ix < N_workers_; ++ix) {
      child_pid = fork_and_handle_errors();
      if (!child_pid) { // we're on the worker
         // Setup process timer, do not overwrite begin time, this keeps timing
         // synced between worker and master processes. The forked process keeps
         // the master process' begin time
         if (RooFit::MultiProcess::Config::getTimingAnalysis()) ProcessTimer::setup(ix, false);
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

   close_unused_channel_fds();

   // set the sigterm handler on the child processes
   if (!is_master_) {
      // Create the self-pipe that the handler writes to before installing the
      // handler. The pipe wakes up any poll on the channels, also when the
      // signal arrived just before the poll was entered (see Channel::wait).
      if (sigterm_wake_read_fd_ < 0) {
         int pipe_fds[2];
         if (pipe(pipe_fds) != 0) {
            std::perror("pipe failed");
            std::exit(1);
         }
         for (int fd : pipe_fds) {
            int flags = fcntl(fd, F_GETFL, 0);
            if (flags == -1 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
               std::perror("fcntl failed");
               std::exit(1);
            }
            int fd_flags = fcntl(fd, F_GETFD, 0);
            if (fd_flags == -1 || fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC) == -1) {
               std::perror("fcntl failed");
               std::exit(1);
            }
         }
         sigterm_wake_read_fd_ = pipe_fds[0];
         sigterm_wake_write_fd_ = pipe_fds[1];
      }

      struct sigaction sa;
      memset(&sa, '\0', sizeof(sa));
      sa.sa_handler = ProcessManager::handle_sigterm;

      if (sigaction(SIGTERM, &sa, nullptr) < 0) {
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
      if (RooFit::MultiProcess::Config::getTimingAnalysis()) ProcessTimer::write_file();
      // Give children some time to write to file
      if (RooFit::MultiProcess::Config::getTimingAnalysis()) std::this_thread::sleep_for(std::chrono::seconds(2));
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
