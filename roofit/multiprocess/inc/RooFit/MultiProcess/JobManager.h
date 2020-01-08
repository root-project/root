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
#ifndef ROOT_ROOFIT_MultiProcess_JobManager
#define ROOT_ROOFIT_MultiProcess_JobManager

#include <memory>  // unique_ptr
#include <map>

#include "RooFit/MultiProcess/types.h"

namespace RooFit {
namespace MultiProcess {

// forward definitions
class ProcessManager;
class Messenger;
class Queue;
class Job;

/*
 * @brief Main point of access for all MultiProcess infrastructure
 *
 * This class mainly serves as the access point to the multi-process infrastructure
 * for Jobs. It is meant to be used as a singleton that holds and connects the other
 * infrastructural classes: the messenger, process manager, worker and queue loops.
 *
 * It is important that the user of this class, particularly the one that calls
 * instance() first, calls activate_loops() soon after, because everything that is
 * done in between instance() and activate_loops() will be executed on all processes.
 * This may be useful in some cases, but in general, one will probably want to always
 * use the JobManager in its full capacity, including the queue and worker loops.
 * This is the way the Job class uses this class, see
 */

class JobManager {
public:
   static JobManager *instance(std::size_t N_workers);
   static JobManager *instance();
   static bool is_instantiated();

   ~JobManager();

   static std::size_t add_job_object(Job *job_object);
   static Job *get_job_object(std::size_t job_object_id);
   static bool remove_job_object(std::size_t job_object_id);

   ProcessManager & process_manager() const;
   Messenger & messenger() const;
   Queue & queue() const;

   void retrieve();
   void results_from_queue_to_master();

   void activate();
   bool is_activated() const;

private:
   explicit JobManager(std::size_t N_workers);

   std::unique_ptr<ProcessManager> process_manager_ptr;
   std::unique_ptr<Messenger> messenger_ptr;
   std::unique_ptr<Queue> queue_ptr;
   bool activated = false;

   static std::map<std::size_t, Job *> job_objects;
   static std::size_t job_counter;
   static std::unique_ptr <JobManager> _instance;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_JobManager
