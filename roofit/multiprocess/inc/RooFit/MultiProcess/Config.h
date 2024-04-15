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

#ifndef ROOT_ROOFIT_MultiProcess_Config
#define ROOT_ROOFIT_MultiProcess_Config

#include "RooFit/MultiProcess/types.h"

#include <vector>
#include <cstddef>  // std::size_t

namespace RooFit {
namespace MultiProcess {

class Config {
public:
   static void setDefaultNWorkers(unsigned int N_workers);
   static unsigned int getDefaultNWorkers();

   static void setTimingAnalysis(bool timingAnalysis);
   static bool getTimingAnalysis();

   struct LikelihoodJob {
      // magic values to indicate that the number of tasks will be set automatically
      constexpr static std::size_t automaticNEventTasks = 0;
      constexpr static std::size_t automaticNComponentTasks = 0;

      static std::size_t defaultNEventTasks;
      static std::size_t defaultNComponentTasks;
   };

   struct Queue {
      enum class QueueType {FIFO, Priority};
      static bool setQueueType(QueueType queueType);
      static QueueType getQueueType();
      static void setTaskPriorities(std::size_t job_id, const std::vector<std::size_t>& task_priorities);
      static void suggestTaskOrder(std::size_t job_id, const std::vector<Task>& task_order);
   private:
      static QueueType queueType_;
   };
private:
   static unsigned int defaultNWorkers_;
   static bool timingAnalysis_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_Config
