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
#ifndef ROOT_ROOFIT_MultiProcess_FIFOQueue
#define ROOT_ROOFIT_MultiProcess_FIFOQueue

#include "RooFit/MultiProcess/Queue.h"
#include <queue>

namespace RooFit {
namespace MultiProcess {

class FIFOQueue : public Queue {
public:
   bool pop(JobTask &job_task) override;
   void add(JobTask job_task) override;

private:
   std::queue<JobTask> queue_;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_FIFOQueue
