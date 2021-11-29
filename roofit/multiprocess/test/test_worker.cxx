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

#include "RooFit/MultiProcess/worker.h"

#include "gtest/gtest.h"

// TODO: implement something along these lines (the following is pseudocode)
// TEST(worker, loop)
//{
//   do_fork();
//   if (on_master()) {
//      wait();
//      terminate_worker();
//   } else if (on_worker()) {
//      worker_loop();
//   }
//}
