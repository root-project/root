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

#ifndef ROOT_ROOFIT_MultiProcess_util
#define ROOT_ROOFIT_MultiProcess_util

#include "RooFit/MultiProcess/Channel.h" // ppoll_error_t
#include "RooFit/MultiProcess/Poller.h"

#include <tuple>
#include <unistd.h> // getpid, pid_t
#include <vector>

namespace RooFit {
namespace MultiProcess {

int wait_for_child(pid_t child_pid, bool may_throw, int retries_before_killing);

std::tuple<std::vector<std::size_t>, bool> careful_poll(Poller &poller);

} // namespace MultiProcess
} // namespace RooFit
#endif // ROOT_ROOFIT_MultiProcess_util
