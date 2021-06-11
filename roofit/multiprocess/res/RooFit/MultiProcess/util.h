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

#include "RooFit_ZMQ/ppoll.h" // for ZMQ::ppoll_error_t
#include "RooFit_ZMQ/ZeroMQPoller.h"

#include <unistd.h> // getpid, pid_t

namespace RooFit {
namespace MultiProcess {

int wait_for_child(pid_t child_pid, bool may_throw, int retries_before_killing);

enum class zmq_ppoll_error_response { abort, unknown_eintr, retry };
zmq_ppoll_error_response handle_zmq_ppoll_error(ZMQ::ppoll_error_t &e);
std::tuple<std::vector<std::pair<size_t, zmq::event_flags>>, bool>
careful_ppoll(ZeroMQPoller &poller, const sigset_t &ppoll_sigmask, std::size_t max_tries = 2);

} // namespace MultiProcess
} // namespace RooFit
#endif // ROOT_ROOFIT_MultiProcess_util
