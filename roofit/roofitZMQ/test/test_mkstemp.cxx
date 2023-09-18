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

#include <TSystem.h>

#include "gtest/gtest.h"

#include <cstdlib>  // mkstemp
#include <unistd.h> // for "legacy" systems

TEST(BindToTmpFile, mkstemp)
{
   std::string tmpPath = gSystem->TempDirectory();

   std::string filename = tmpPath + "/roofit_MP_XXXXXX";
   while (mkstemp(const_cast<char *>(filename.c_str())) < 0) {
   }
   EXPECT_NE(filename, tmpPath + "/roofit_MP_XXXXXX");
   auto socket = zmqSvc().socket(zmq::socket_type::push);
   std::string address = "ipc://" + filename;
   try {
      socket.bind(address);
   } catch (const zmq::error_t &) {
      printf("caught an exception\n");
   }
}
