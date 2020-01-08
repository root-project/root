/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2016-2019, Netherlands eScience Center
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <csignal> // kill, SIGKILL
#include <iostream> // cerr, and indirectly WNOHANG, EINTR, W* macros
#include <stdexcept> // runtime_error
#include <sys/wait.h>  // waitpid
#include <string>

#include <MultiProcess/util.h>

namespace RooFit {
  namespace MultiProcess {

    int wait_for_child(pid_t child_pid, bool may_throw, int retries_before_killing) {
      int status = 0;
      int patience = retries_before_killing;
      pid_t tmp;
      do {
        if (patience-- < 1) {
          ::kill(child_pid, SIGKILL);
        }
        tmp = waitpid(child_pid, &status, WNOHANG);
      } while (
          tmp == 0 // child has not yet changed state, try again
          || (-1 == tmp && EINTR == errno) // retry on interrupted system call
          );

      if (patience < 1) {
        std::cout << "Had to send PID " << child_pid << " " << (-patience+1) << " SIGKILLs";
      }

      if (0 != status) {
        if (WIFEXITED(status)) {
          printf("exited, status=%d\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
          printf("killed by signal %d\n", WTERMSIG(status));
        } else if (WIFSTOPPED(status)) {
          printf("stopped by signal %d\n", WSTOPSIG(status));
        } else if (WIFCONTINUED(status)) {
          printf("continued\n");
        }
      }

      if (-1 == tmp && may_throw) throw std::runtime_error(std::string("waitpid, errno ") + std::to_string(errno));

      return status;
    }

  }
}