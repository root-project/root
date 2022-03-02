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
#include <csignal>     // sig_atomic_t and for sigterm handling on child processes (in ProcessManager.cxx)
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

   // for debugging/testing:
   pid_t get_queue_pid() const { return queue_pid_; }
   std::vector<pid_t> get_worker_pids() { return worker_pids_; }

private:
   void initialize_processes(bool cpu_pinning = true);
   void shutdown_processes();

   bool is_master_ = false;
   bool is_queue_ = false;
   bool is_worker_ = false;
   std::size_t worker_id_;
   std::size_t N_workers_;

   // master must wait for workers after completion, for which it needs their PIDs
   std::vector<pid_t> worker_pids_;
   pid_t queue_pid_;

   bool initialized_ = false;

   static volatile sig_atomic_t sigterm_received_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_ProcessManager
