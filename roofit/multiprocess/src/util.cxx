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

zmq_ppoll_error_response handle_zmq_ppoll_error(ZMQ::ppoll_error_t &e)
{
   if ((e.num() == EINTR) && (ProcessManager::sigterm_received())) {
      // valid EINTR, because we want to exit and kill the processes on SIGTERM
      return zmq_ppoll_error_response::abort;
   } else if (e.num() == EINTR) {
      // on other EINTRs, we retry (mostly this happens in debuggers)
      return zmq_ppoll_error_response::unknown_eintr;
   } else if (e.num() == EAGAIN) {
      // This can happen from recv if ppoll initially gets a read-ready signal for a socket,
      // but the received data does not pass the checksum test, so the socket becomes unreadable
      // again or from non-blocking send if the socket becomes unwritable either due to the HWM
      // being reached or the socket not being connected (anymore). The latter case usually means
      // the connection has been severed from the other side, meaning it has probably been killed
      // and in that case the next ppoll call will probably also receive a SIGTERM, ending the
      // loop. In case something else is wrong, this message will print multiple times, which
      // should be taken as a cue for writing a bug report :)
      return zmq_ppoll_error_response::retry;
   } else {
      char buffer[512];
      snprintf(buffer, 512,
               "handle_zmq_ppoll_error is out of options to handle exception, caught ZMQ::ppoll_error_t had errno %d "
               "and text: %s\n",
               e.num(), e.what());
      throw std::logic_error(buffer);
   }
}

// returns a tuple containing first the poll result and second a boolean flag that tells the caller whether it should
// abort the enclosing loop
std::tuple<std::vector<std::pair<size_t, zmq::event_flags>>, bool>
careful_ppoll(ZeroMQPoller &poller, const sigset_t &ppoll_sigmask, std::size_t max_tries)
{
   std::size_t tries = 0;
   std::vector<std::pair<size_t, zmq::event_flags>> poll_result;
   bool abort = true;
   bool carry_on = true;
   while (carry_on && (tries++ < max_tries)) {
      if (tries > 1) {
         printf("careful_ppoll try %zu\n", tries);
      }
      try { // watch for zmq_error from ppoll caused by SIGTERM from master
         poll_result = poller.ppoll(-1, &ppoll_sigmask);
         abort = false;
         carry_on = false;
      } catch (ZMQ::ppoll_error_t &e) {
         auto response = handle_zmq_ppoll_error(e);
         if (response == zmq_ppoll_error_response::abort) {
            break;
         } else if (response == zmq_ppoll_error_response::unknown_eintr) {
            printf("EINTR in careful_ppoll but no SIGTERM received, try %zu\n", tries);
            continue;
         } else if (response == zmq_ppoll_error_response::retry) {
            printf("EAGAIN in careful_ppoll (from either send or receive), try %zu\n", tries);
            continue;
         }
      }
   }

   if (tries == max_tries) {
      printf("careful_ppoll reached maximum number of tries, %zu, please report as a bug\n", tries);
   }
   return std::make_tuple(poll_result, abort);
}

} // namespace MultiProcess
} // namespace RooFit