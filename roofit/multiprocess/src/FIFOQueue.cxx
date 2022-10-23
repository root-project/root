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
#include "FIFOQueue.h"

namespace RooFit {
namespace MultiProcess {

/** \class FIFOQueue
 * \brief A Queue with simple FIFO behavior
 */

/// See Queue::pop.
bool FIFOQueue::pop(JobTask &job_task)
{
   if (queue_.empty()) {
      return false;
   } else {
      job_task = queue_.front();
      queue_.pop();
      return true;
   }
}

/// See Queue::add.
void FIFOQueue::add(JobTask job_task)
{
   if (JobManager::instance()->process_manager().is_master()) {
      JobManager::instance()->messenger().send_from_master_to_queue(M2Q::enqueue, job_task.job_id, job_task.state_id,
                                                                    job_task.task_id);
   } else if (JobManager::instance()->process_manager().is_queue()) {
      queue_.push(job_task);
   } else {
      throw std::logic_error("calling Communicator::to_master_queue from slave process");
   }
}

} // namespace MultiProcess
} // namespace RooFit
