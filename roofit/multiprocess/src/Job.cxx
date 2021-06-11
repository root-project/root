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
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/Job.h"

namespace RooFit {
namespace MultiProcess {

/** @class Job
 *
 * @brief interface class for defining the actual work that must be done
 *
 * Think of "job" as in "employment", e.g. the job of a baker, which
 * involves *tasks* like baking and selling bread. The Job must define the
 * tasks through its execution (evaluate_task), based on a task index argument.
 *
 * Classes inheriting from Job must implement the pure virtual methods:
 * - void evaluate_task(std::size_t task)
 * - void send_back_task_result_from_worker(std::size_t task)
 * - void receive_task_result_on_master(const zmq::message_t & message)
 *
 * An example/reference implementation can be found in test_Job.cxx.
 *
 * Most Jobs will also want to override the virtual update_state() function.
 * This function can be used to send and receive state from master to worker.
 * In the worker loop, when something is received over the ZeroMQ "SUB" socket,
 * update_state() is called to put the received data into the right places,
 * thus updating for instance parameter values on the worker that were updated
 * since the last call on the master side.
 *
 * ## Message protocol
 *
 * One simple rule must be upheld for the messages that the implementer will
 * send with 'send_back_task_result_from_worker' and 'update_state': the first
 * part of the message must always be the 'Job''s ID, stored in 'Job::id'.
 * The rest of the message, i.e. the actual data to be sent, is completely up
 * to the implementation. Note that on the receiving end, i.e. in the
 * implementation of 'receive_task_result_on_master', one will get the whole
 * message, but the 'Job' ID part will already have been identified in the
 * 'JobManager', so one needn't worry about it further inside
 * 'Job::receive_task_result_on_master' (it is already routed to the correct
 * 'Job'). The same goes for the receiving end of 'update_state', except that
 * update_state is routed from the 'worker_loop', not the 'JobManager'.
 *
 * A second rule applies to 'update_state' messages: the second part must be
 * a state identifier. This identifier will also be sent along with tasks to
 * the queue. When a worker then takes a task from the queue, it can check
 * whether it has already updated its state to what is expected to be there
 * for the task at hand. If not, it should wait for the new state to arrive
 * over the state subscription socket. Note: it is the implementer's task to
 * actually update 'Job::state_id_' inside 'Job::update_state()'!
 *
 * ## Implementers notes
 *
 * The type of result from each task is strongly dependent on the Job at hand
 * and so Job does not provide a default results member. It is up to the
 * inheriting class to implement this in the above functions. We would have
 * liked a template parameter task_result_t, so that we could also provide a
 * default "boilerplate" calculate function to show a typical Job use-case of
 * all the above infrastructure. This is not trivial, because the JobManager
 * has to keep a list of Job pointers, so if there would be different template
 * instantiations of Jobs, this would complicate this list.
 *
 * A typical Job implementation will have an evaluation function that is
 * called from the master process, like RooAbsArg::getVal calls evaluate().
 * This function will have three purposes: 1. send updated parameter values
 * to the workers (possibly through update_state() or in a dedicated
 * function), 2. queue tasks and 3. wait for the results to be retrieved.
 * 'Job::gather_worker_results()' is provided for convenience to wait for
 * all tasks to be retrieved for the current Job. Implementers can also
 * choose to have the master process perform other tasks in between any of
 * these three steps, or even skip steps completely.
 *
 * Child classes should refrain from direct access to the JobManager instance
 * (through JobManager::instance), but rather use the here provided
 * Job::get_manager(). This function starts the worker_loop on the worker when
 * first called, meaning that the workers will not be running before they
 * are needed.
 */

Job::Job()
{
   id_ = JobManager::add_job_object(this);
}

Job::Job(const Job &other) : _manager(other._manager)
{
   id_ = JobManager::add_job_object(this);
}

Job::~Job()
{
   JobManager::remove_job_object(id_);
}

/** \brief Get JobManager instance; create and activate if necessary
 *
 * Child classes should refrain from direct access to the JobManager instance
 * (through JobManager::instance), but rather use the here provided
 * Job::get_manager(). This function starts the worker_loop on the worker when
 * first called, meaning that the workers will not be running before they
 * are needed.
 */
JobManager *Job::get_manager()
{
   if (!_manager) {
      _manager = JobManager::instance();
   }

   if (!_manager->is_activated()) {
      _manager->activate();
   }

   return _manager;
}

/// Wait for all tasks to be retrieved for the current Job.
void Job::gather_worker_results()
{
   get_manager()->retrieve(id_);
}

/// \brief Virtual function to update any necessary state on workers
///
/// This function is called from the worker loop when something is received
/// over the ZeroMQ "SUB" socket. The master process sends messages to workers
/// on its "PUB" socket. Thus, we can update, for instance, parameter values
/// on the worker that were updated since the last call on the master side.
/// \note Implementers: make sure to also update the state_id_ member.
void Job::update_state() {}

/// Get the current state identifier
std::size_t Job::get_state_id()
{
   return state_id_;
}

} // namespace MultiProcess
} // namespace RooFit
