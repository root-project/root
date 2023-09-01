/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *   IP, Inti Pelupessy, Netherlands eScience Center, i.pelupessy@esciencecenter.nl
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/JobManager.h"
#include "RooFit/MultiProcess/Messenger.h"
#include "RooFit/MultiProcess/types.h"
#include "PriorityQueue.h"

#include <algorithm> // std::copy

namespace RooFit {
namespace MultiProcess {

/** \class PriorityQueue
 * \brief Queue that orders tasks according to specified task priorities.
 */

/// See Queue::pop.
bool PriorityQueue::pop(JobTask &job_task)
{
   if (queue_.empty()) {
      return false;
   }
   job_task = queue_.top().jobTask;
   queue_.pop();
   return true;
}

/// See Queue::add.
void PriorityQueue::add(JobTask job_task)
{
   auto& jobManager = *JobManager::instance();
   if (jobManager.process_manager().is_master()) {
      jobManager.messenger().send_from_master_to_queue(M2Q::enqueue, job_task.job_id, job_task.state_id,
                                                       job_task.task_id);
   } else if (jobManager.process_manager().is_queue()) {
      std::size_t priority = 0;  // default priority if no task_priority_ vector was set for this Job
      if (task_priority_.find(job_task.job_id) != task_priority_.end()) {
         priority = task_priority_[job_task.job_id][job_task.task_id];
      }
      queue_.emplace(job_task, priority);
   } else {
      throw std::logic_error("calling Communicator::to_master_queue from slave process");
   }
}

/// Set the desired order for executing tasks of a Job.
///
/// See Config::Queue::suggestTaskOrder.
void PriorityQueue::suggestTaskOrder(std::size_t job_id, const std::vector<Task>& task_order)
{
   std::vector<std::size_t> task_priorities(task_order.size());
   for (std::size_t ix = 0; ix < task_order.size(); ++ix) {
      task_priorities[task_order[ix]] = task_order.size() - ix;
   }
   setTaskPriorities(job_id, task_priorities);
}

/// Set the priority for Job tasks.
///
/// See Config::Queue::setTaskPriorities.
void PriorityQueue::setTaskPriorities(std::size_t job_id, const std::vector<std::size_t>& task_priorities)
{
   task_priority_.clear();
   task_priority_.reserve(task_priorities.size());
   std::copy(task_priorities.cbegin(), task_priorities.cend(), std::back_inserter(task_priority_[job_id]));
}

} // namespace MultiProcess
} // namespace RooFit
