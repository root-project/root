/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "RooFit/MultiProcess/Config.h"
#include "RooFit/MultiProcess/JobManager.h"
#include "PriorityQueue.h"
#include "RooFit/MultiProcess/ProcessManager.h"

#include <thread> // std::thread::hardware_concurrency()
#include <cstdio>

namespace RooFit {
namespace MultiProcess {

/** \class Config
 *
 * \brief Configuration for MultiProcess infrastructure
 *
 * This class offers user-accessible configuration of the MultiProcess infrastructure.
 * Since the rest of the MultiProcess classes are only accessible at compile time, a
 * separate class is needed to set configuration. Currently, the configurable parts
 * are:
 * 1. the number of workers to be deployed,
 * 2. the number of event-tasks in LikelihoodJobs,
 * 3. and the number of component-tasks in LikelihoodJobs.
 *
 * The default number of workers is set using 'std::thread::hardware_concurrency()'.
 * To change it, use 'Config::setDefaultNWorkers()' to set it to a different value
 * before creation of a new JobManager instance. Note that it cannot be set to zero
 * and also cannot be changed after JobManager has been instantiated.
 *
 * Use Config::getDefaultNWorkers() to access the current value.
 *
 * Under Config::LikelihoodJob, we find two members for the number of tasks to use to
 * calculate the range of events and components in parallel, respectively:
 * defaultNEventTasks and defaultNComponentTasks. Newly created LikelihoodJobs will
 * then use these values at construction time. Note that (like with the number of
 * workers) the number cannot be changed for an individual LikelihoodJob after it has
 * been created.
 *
 * Both event- and component-based tasks by default are set to automatic mode using
 * the automaticNEventTasks and automaticNComponentTasks constants (both under
 * Config::LikelihoodJob as well). These are currently set to zero, but this could
 * change. Automatic mode for events means that the number of tasks is set to the
 * number of workers in the JobManager, with events divided equally over workers. For
 * components, the automatic mode uses just 1 task for all components. These automatic
 * modes may change in the future (for instance, we may switch them around).
 *
 * Under Config::Queue, we can set the desired queue type: FIFO or Priority. This
 * setting is used when a JobManager is spun up, i.e. usually when the first Job
 * starts. At this point, the Queue is also created according to the setting. The
 * default is FIFO. When using a Priority Queue, the priority of tasks in a Job
 * can be set using either setTaskPriorities or suggestTaskOrder. If no priorities
 * are set, the Priority queue simply assumes equal priority for all tasks. The
 * resulting order then depends on the implementation of std::priority_queue.
 */

void Config::setDefaultNWorkers(unsigned int N_workers)
{
   if (JobManager::is_instantiated()) {
      printf("Warning: Config::setDefaultNWorkers cannot set number of workers after JobManager has been instantiated!\n");
   } else if (N_workers == 0) {
      printf("Warning: Config::setDefaultNWorkers cannot set number of workers to zero.\n");
   } else {
      defaultNWorkers_ = N_workers;
   }
}

void Config::setTimingAnalysis(bool timingAnalysis)
{
   if (JobManager::is_instantiated() && JobManager::instance()->process_manager().is_initialized()) {
      printf("Warning: Config::setTimingAnalysis cannot set logging of timings, forking has already taken place!\n");
   } else {
      timingAnalysis_ = timingAnalysis;
   }
}

bool Config::getTimingAnalysis()
{
   return timingAnalysis_;
}

unsigned int Config::getDefaultNWorkers()
{
   return defaultNWorkers_;
}

bool Config::Queue::setQueueType(QueueType queueType)
{
   if (JobManager::is_instantiated()) {
      printf("Warning: cannot set RooFit::MultiProcess queue type after JobManager has been instantiated!\n");
      return false;
   }
   queueType_ = queueType;
   return true;
}

Config::Queue::QueueType Config::Queue::getQueueType()
{
   return queueType_;
}

/// Set the priority for Job tasks in Priority queue mode.
///
/// Only useful in Priority queue mode, in FIFO mode this doesn't do anything.
/// A higher value means a higher priority.
///
/// \param[in] job_id Job ID to set task order for.
/// \param[in] task_priorities Task priority values, where vector index equals task ID.
void Config::Queue::setTaskPriorities(std::size_t job_id, const std::vector<std::size_t>& task_priorities)
{
   if (queueType_ == QueueType::Priority) {
      dynamic_cast<PriorityQueue *>(JobManager::instance()->queue())->setTaskPriorities(job_id, task_priorities);
   }
}

/// Set the desired order for executing tasks of a Job in Priority queue mode.
///
/// Only useful in Priority queue mode, in FIFO mode this doesn't do anything.
///
/// Translates the desired order to priorities. Because workers will start
/// stealing work immediately after it has been queued, the desired order
/// cannot be guaranteed -- hence "suggest" -- because the first queued task
/// will possibly be taken before higher priority tasks have been sent to
/// the queue.
///
/// \param[in] job_id Job ID to set task order for.
/// \param[in] task_order Task IDs in the desired order.
void Config::Queue::suggestTaskOrder(std::size_t job_id, const std::vector<Task>& task_order)
{
   if (queueType_ == QueueType::Priority) {
      dynamic_cast<PriorityQueue *>(JobManager::instance()->queue())->suggestTaskOrder(job_id, task_order);
   }
}

// initialize static members
unsigned int Config::defaultNWorkers_ = std::thread::hardware_concurrency();
std::size_t Config::LikelihoodJob::defaultNEventTasks = Config::LikelihoodJob::automaticNEventTasks;
std::size_t Config::LikelihoodJob::defaultNComponentTasks = Config::LikelihoodJob::automaticNComponentTasks;
Config::Queue::QueueType Config::Queue::queueType_ = Config::Queue::QueueType::FIFO;
bool Config::timingAnalysis_ = false;

} // namespace MultiProcess
} // namespace RooFit
