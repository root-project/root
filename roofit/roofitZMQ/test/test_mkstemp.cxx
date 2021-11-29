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

#include "RooFit_ZMQ/ZeroMQSvc.h"

#include "gtest/gtest.h"

#include <cstdlib>  // mkstemp
#include <unistd.h> // for "legacy" systems

TEST(BindToTmpFile, mkstemp)
{
   char filename[] = "/tmp/roofitMP_XXXXXX";
   while (mkstemp(filename) < 0) {
   }
   auto socket = zmqSvc().socket(zmq::socket_type::push);
   char address[50];
   sprintf(address, "ipc://%s", filename);
   try {
      socket.bind(address);
   } catch (const zmq::error_t &) {
      printf("caught an exception\n");
   }
}