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
#ifndef ROOT_ROOFIT_MultiProcess_ProcessManager
#define ROOT_ROOFIT_MultiProcess_ProcessManager

#include <sys/types.h> // pid_t
#include <array>
#include <csignal> // sig_atomic_t and for sigterm handling on child processes (in ProcessManager.cxx)
#include <vector>

// forward declaration
class Queue;

namespace RooFit {
namespace MultiProcess {

class ProcessManager {
   friend Queue;

public:
   explicit ProcessManager(std::size_t N_workers);
   ~ProcessManager();

   bool is_initialized() const;

   void terminate() noexcept;
   void wait_for_sigterm_then_exit();

   bool is_master() const;
   bool is_queue() const;
   bool is_worker() const;
   std::size_t worker_id() const;
   std::size_t N_workers() const;

   void identify_processes() const;

   static void handle_sigterm(int signum);
   static bool sigterm_received();
   /// Read end of the self-pipe that the SIGTERM handler writes to (or -1 on
   /// the master process, which installs no handler); used by Channel::wait.
   static int sigterm_wake_fd();

   // Interprocess channel file descriptors, created with socketpair() before
   // forking. The Messenger claims the ends belonging to the current process
   // and takes over their ownership; unclaimed descriptors are closed when
   // this ProcessManager is destroyed.
   int claim_mq_fd();
   int claim_qw_fd(std::size_t worker_ix);
   int claim_mw_fd(std::size_t worker_ix);

   // for debugging/testing:
   pid_t get_queue_pid() const { return queue_pid_; }
   std::vector<pid_t> get_worker_pids() { return worker_pids_; }

private:
   void initialize_processes(bool cpu_pinning = true);
   void shutdown_processes();
   void create_channel_fds();
   void close_unused_channel_fds();
   void close_channel_fds();

   bool is_master_ = false;
   bool is_queue_ = false;
   bool is_worker_ = false;
   std::size_t worker_id_;
   std::size_t N_workers_;

   // master must wait for workers after completion, for which it needs their PIDs
   std::vector<pid_t> worker_pids_;
   pid_t queue_pid_;

   bool initialized_ = false;

   // socketpair ends for the interprocess channels; in each array, index 0 is
   // the end used by the process listed first in the member name (m: master,
   // q: queue, w: worker), index 1 the other end
   std::array<int, 2> mq_fds_{{-1, -1}};
   std::vector<std::array<int, 2>> qw_fds_;
   std::vector<std::array<int, 2>> mw_fds_;

   static volatile sig_atomic_t sigterm_received_;
   static int sigterm_wake_read_fd_;
   static int sigterm_wake_write_fd_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_ProcessManager
