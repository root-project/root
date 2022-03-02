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
#ifndef ROOT_ROOFIT_MultiProcess_types
#define ROOT_ROOFIT_MultiProcess_types

#include <utility> // pair

namespace RooFit {
namespace MultiProcess {

// some helper types
using Task = std::size_t;
using State = std::size_t;
/// combined job_object, state and task identifier type
struct JobTask {
   std::size_t job_id;
   State state_id;
   Task task_id;
};

} // namespace MultiProcess
} // namespace RooFit

#endif // ROOT_ROOFIT_MultiProcess_types
