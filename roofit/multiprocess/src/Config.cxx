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

unsigned int Config::getDefaultNWorkers()
{
   return defaultNWorkers_;
}

// initialize static members
unsigned int Config::defaultNWorkers_ = std::thread::hardware_concurrency();
std::size_t Config::LikelihoodJob::defaultNEventTasks = Config::LikelihoodJob::automaticNEventTasks;
std::size_t Config::LikelihoodJob::defaultNComponentTasks = Config::LikelihoodJob::automaticNComponentTasks;

} // namespace MultiProcess
} // namespace RooFit
