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
#ifndef ROOT_ROOFIT_MultiProcess_ReorderQueue
#define ROOT_ROOFIT_MultiProcess_ReorderQueue

#include "RooFit/MultiProcess/Queue.h"
#include <queue>

namespace RooFit {
namespace MultiProcess {

struct OrderedJobTask {
   OrderedJobTask(JobTask job_task, std::size_t task_priority): jobTask(job_task), taskPriority(task_priority) {}
   JobTask jobTask;
   std::size_t taskPriority;

   bool operator < (const OrderedJobTask& rhs) const {return taskPriority < rhs.taskPriority;}
};

class PriorityQueue : public Queue {
public:
   bool pop(JobTask &job_task) override;
   void add(JobTask job_task) override;
   void suggestTaskOrder(std::size_t job_id, const std::vector<Task>& task_order);
   void setTaskPriorities(std::size_t job_id, const std::vector<std::size_t>& task_priorities);
private:
   std::priority_queue<OrderedJobTask> queue_;
   // key: job ID, value: task priority; higher value means higher priority
   std::unordered_map<std::size_t, std::vector<std::size_t>> task_priority_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_ReorderQueue
