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

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/ProcessManager.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Job.h"
#include "RooFit/MultiProcess/Queue.h" // complete type for JobManager::queue()
#include "RooFit/MultiProcess/worker.h"
#include "RooFit/MultiProcess/util.h"
#include "RooFit/MultiProcess/Config.h"

namespace RooFit {
namespace MultiProcess {

/** \class JobManager
 *
 * \brief Main point of access for all MultiProcess infrastructure
 *
 * This class mainly serves as the access point to the multi-process infrastructure
 * for 'Job's. It is meant to be used as a singleton that holds and connects the other
 * infrastructural classes: the messenger, process manager, worker and queue loops.
 *
 * It is important that the user of this class, particularly the one that calls
 * 'instance()' first, calls 'activate()' soon after, because everything that is
 * done in between 'instance()' and 'activate()' will be executed on all processes.
 * This may be useful in some cases, but in general, one will probably want to always
 * use the 'JobManager' in its full capacity, including the queue and worker loops.
 * This is the way the Job class uses this class, see 'Job::get_manager()'.
 *
 * The default number of processes is set using 'std::thread::hardware_concurrency()'.
 * To change it, use 'Config::setDefaultNWorkers()' to set it to a different value
 * before creation of a new JobManager instance.
 */

// static function
JobManager *JobManager::instance()
{
   if (!JobManager::is_instantiated()) {
      instance_.reset(new JobManager(Config::getDefaultNWorkers())); // can't use make_unique, because ctor is private
      instance_->messenger().test_connections(instance_->process_manager());
      // set send to non blocking on all processes after checking the connections are working:
      instance_->messenger().set_send_flag(zmq::send_flags::dontwait);
   }
   return instance_.get();
}

// static function
bool JobManager::is_instantiated()
{
   return static_cast<bool>(instance_);
}

// (private) constructor
/// Don't construct JobManager objects manually, use the static instance if
/// you need to run multiple jobs.
JobManager::JobManager(std::size_t N_workers)
{
   queue_ptr_ = std::make_unique<Queue>();
   process_manager_ptr_ = std::make_unique<ProcessManager>(N_workers);
   messenger_ptr_ = std::make_unique<Messenger>(*process_manager_ptr_);
}

JobManager::~JobManager()
{
   // The instance typically gets created by some Job. Once all Jobs are gone, the
   // JM will get destroyed. In this case, the job_objects map should have
   // been emptied.
   // The second case is when the program ends, at which time the static instance
   // is destroyed. Jobs may still be present, for instance, the Job subclass
   // RooFit::TestStatistics::LikelihoodGradientJob, will have
   // been put into RooMinimizer::_theFitter->fObjFunction, as the gradient
   // member. Because _theFitter is also a global static member, we cannot
   // guarantee destruction order, and so the JobManager may be destroyed before
   // all Jobs are destroyed. We cannot therefore make sure that the first
   // condition is met. However, the Job objects stuck in _theFitter are not
   // meant to be run again, because the program is ending anyway. So also in this
   // case, we can safely shut down.
   // There used to be an assert statement that checked whether the job_objects
   // map was empty at destruction time, but that neglected the second possibility
   // and led to assertion failures, which left the Messenger and ProcessManager
   // objects intact, leading to the forked processes and their ZeroMQ resources
   // to remain after exiting the main/master/parent process.
   messenger_ptr_.reset(nullptr);
   process_manager_ptr_.reset(nullptr);
   queue_ptr_.reset(nullptr);
}

// static function
/// \return job_id for added job_object
std::size_t JobManager::add_job_object(Job *job_object)
{
   if (JobManager::is_instantiated()) {
      if (instance_->process_manager().is_initialized()) {
         std::stringstream ss;
         ss << "Cannot add Job to JobManager instantiation, forking has already taken place! Instance object at raw "
               "ptr "
            << instance_.get();
         throw std::logic_error("Cannot add Job to JobManager instantiation, forking has already taken place! Call "
                                "terminate() on the instance before adding new Jobs.");
      }
   }
   std::size_t job_id = job_counter_++;
   job_objects_[job_id] = job_object;
   return job_id;
}

// static function
Job *JobManager::get_job_object(std::size_t job_object_id)
{
   return job_objects_[job_object_id];
}

// static function
/// \return Returns 'true' when removed successfully, 'false' otherwise.
bool JobManager::remove_job_object(std::size_t job_object_id)
{
   bool removed_succesfully = job_objects_.erase(job_object_id) == 1;
   if (job_objects_.empty()) {
      instance_.reset(nullptr);
   }
   return removed_succesfully;
}

ProcessManager &JobManager::process_manager() const
{
   return *process_manager_ptr_;
}

Messenger &JobManager::messenger() const
{
   return *messenger_ptr_;
}

Queue &JobManager::queue() const
{
   return *queue_ptr_;
}

/// Retrieve results for a Job
///
/// \param requesting_job_id ID number of the Job in the JobManager's Job list
void JobManager::retrieve(std::size_t requesting_job_id)
{
   if (process_manager().is_master()) {
      bool job_fully_retrieved = false;
      while (not job_fully_retrieved) {
         try {
            auto task_result_message = messenger().receive_from_worker_on_master<zmq::message_t>();
            auto job_object_id = *reinterpret_cast<std::size_t *>(
               task_result_message.data()); // job_id must always be the first element of the result message!
            bool this_job_fully_retrieved =
               JobManager::get_job_object(job_object_id)->receive_task_result_on_master(task_result_message);
            if (requesting_job_id == job_object_id) {
               job_fully_retrieved = this_job_fully_retrieved;
            }
         } catch (ZMQ::ppoll_error_t &e) {
            zmq_ppoll_error_response response;
            try {
               response = handle_zmq_ppoll_error(e);
            } catch (std::logic_error &) {
               printf("JobManager::retrieve got unhandleable ZMQ::ppoll_error_t\n");
               throw;
            }
            if (response == zmq_ppoll_error_response::abort) {
               throw std::logic_error("in JobManager::retrieve: master received a SIGTERM, aborting");
            } else if (response == zmq_ppoll_error_response::unknown_eintr) {
               printf("EINTR in JobManager::retrieve, continuing\n");
               continue;
            } else if (response == zmq_ppoll_error_response::retry) {
               printf("EAGAIN from ppoll in JobManager::retrieve, continuing\n");
               continue;
            }
         } catch (zmq::error_t &e) {
            printf("unhandled zmq::error_t (not a ppoll_error_t) in JobManager::retrieve with errno %d: %s\n", e.num(),
                   e.what());
            throw;
         }
      }
   }
}

/// \brief Start queue and worker loops on child processes
///
/// This function exists purely because activation from the constructor is
/// impossible; the constructor must return a constructed instance, which it
/// can't do if it's stuck in an infinite loop. This means the Job that first
/// creates the JobManager instance must also activate it (or any other user
/// of this class).
/// This should be called soon after creation of instance, because everything
/// between construction and activation gets executed both on the master
/// process and on the slaves.
void JobManager::activate()
{
   activated_ = true;

   if (process_manager().is_queue()) {
      queue().loop();
      std::_Exit(0);
   }

   if (!is_worker_loop_running() && process_manager().is_worker()) {
      RooFit::MultiProcess::worker_loop();
      std::_Exit(0);
   }
}

bool JobManager::is_activated() const
{
   return activated_;
}

// initialize static members
std::map<std::size_t, Job *> JobManager::job_objects_;
std::size_t JobManager::job_counter_ = 0;
std::unique_ptr<JobManager> JobManager::instance_{nullptr};

} // namespace MultiProcess
} // namespace RooFit
