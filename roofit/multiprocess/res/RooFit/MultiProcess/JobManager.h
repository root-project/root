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
#ifndef ROOT_ROOFIT_MultiProcess_JobManager
#define ROOT_ROOFIT_MultiProcess_JobManager

#include "RooFit/MultiProcess/types.h"

#include <memory> // unique_ptr
#include <map>

namespace RooFit {
namespace MultiProcess {

// forward definitions
class ProcessManager;
class Messenger;
class Queue;
class Job;

class JobManager {
public:
   static JobManager *instance();
   static bool is_instantiated();

   ~JobManager();

   static std::size_t add_job_object(Job *job_object);
   static Job *get_job_object(std::size_t job_object_id);
   static bool remove_job_object(std::size_t job_object_id);

   ProcessManager &process_manager() const;
   Messenger &messenger() const;
   Queue &queue() const;

   void retrieve(std::size_t requesting_job_id);

   void activate();
   bool is_activated() const;

private:
   explicit JobManager(std::size_t N_workers);

   std::unique_ptr<ProcessManager> process_manager_ptr_;
   std::unique_ptr<Messenger> messenger_ptr_;
   std::unique_ptr<Queue> queue_ptr_;
   bool activated_ = false;

   static std::map<std::size_t, Job *> job_objects_;
   static std::size_t job_counter_;
   static std::unique_ptr<JobManager> instance_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_JobManager
