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

#include "RooFit_ZMQ/ppoll.h"

namespace ZMQ {

/// Wrapper around zmq_ppoll
/// This function can throw, so wrap in try-catch!
int ppoll(zmq_pollitem_t *items_, size_t nitems_, long timeout_, const sigset_t *sigmask_)
{
   int rc = zmq_ppoll(items_, static_cast<int>(nitems_), timeout_, sigmask_);
   if (rc < 0)
      throw ppoll_error_t();
   return rc;
}

/// Wrapper around zmq_ppoll
/// This function can throw, so wrap in try-catch!
int ppoll(std::vector<zmq_pollitem_t> &items, long timeout_, const sigset_t *sigmask_)
{
   return ppoll(items.data(), items.size(), timeout_, sigmask_);
}

} // namespace ZMQ
