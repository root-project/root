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

#include "RooFit/MultiProcess/util.h"
#include "RooFit/MultiProcess/ProcessManager.h"

#include <csignal>    // kill, SIGKILL
#include <iostream>   // cerr, and indirectly WNOHANG, EINTR, W* macros
#include <stdexcept>  // runtime_error
#include <sys/wait.h> // waitpid
#include <string>

namespace RooFit {
namespace MultiProcess {

int wait_for_child(pid_t child_pid, bool may_throw, int retries_before_killing)
{
   int status = 0;
   int patience = retries_before_killing;
   pid_t tmp;
   do {
      if (patience-- < 1) {
         ::kill(child_pid, SIGKILL);
      }
      tmp = waitpid(child_pid, &status, WNOHANG);
   } while (tmp == 0                         // child has not yet changed state, try again
            || (-1 == tmp && EINTR == errno) // retry on interrupted system call
   );

   if (patience < 1) {
      std::cout << "Had to send PID " << child_pid << " " << (-patience + 1) << " SIGKILLs";
   }

   if (0 != status) {
      if (WIFEXITED(status)) {
         printf("exited, status=%d\n", WEXITSTATUS(status));
      } else if (WIFSIGNALED(status)) {
         if (WTERMSIG(status) != SIGTERM) {
            printf("killed by signal %d\n", WTERMSIG(status));
         }
      } else if (WIFSTOPPED(status)) {
         printf("stopped by signal %d\n", WSTOPSIG(status));
      } else if (WIFCONTINUED(status)) {
         printf("continued\n");
      }
   }

   if (-1 == tmp && may_throw)
      throw std::runtime_error(std::string("waitpid, errno ") + std::to_string(errno));

   return status;
}

// returns a tuple containing first the poll result and second a boolean flag that tells the caller whether it should
// abort the enclosing loop because a SIGTERM was received
std::tuple<std::vector<std::size_t>, bool> careful_poll(Poller &poller)
{
   // Benign signal interruptions are already retried inside Channel::wait, so
   // an exception here means a termination request.
   std::vector<std::size_t> poll_result;
   bool abort = true;
   try {
      poll_result = poller.poll(-1);
      abort = false;
   } catch (ppoll_error_t &) {
   }
   return std::make_tuple(poll_result, abort);
}

} // namespace MultiProcess
} // namespace RooFit