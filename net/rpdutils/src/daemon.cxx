// @(#)root/rpdutils:$Id$
// Author: Fons Rademakers   11/08/97
// Modifified: Gerardo Ganis 8/04/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DaemonStart                                                          //
//                                                                      //
// Detach a daemon process from login session context.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/param.h>
#if defined(__sun) || defined(__sgi)
#  include <fcntl.h>
#endif

#ifdef SIGTSTP
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#endif

#ifndef NOFILE
#   define NOFILE 0
#endif

#if defined(__hpux)
#define USE_SIGCHLD
#endif

#if defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__APPLE__)
#define USE_SIGCHLD
#define	SIGCLD SIGCHLD
#endif

#if defined(linux) || defined(__hpux) || defined(__sun) || defined(__sgi) || \
    defined(_AIX) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__) || defined(__MACH__) || \
    (defined(__CYGWIN__) && defined(__GNUC__))
#define USE_SETSID
#endif


#include "rpdp.h"

extern int gDebug;

namespace ROOT {

extern ErrorHandler_t gErrSys;

#if defined(USE_SIGCHLD)
//______________________________________________________________________________
static void SigChild(int)
{
   int         pid;
#if defined(__hpux) || defined(__FreeBSD__) || defined(__OpenBSD__) || \
    defined(__APPLE__)
   int status;
#else
   union wait  status;
#endif

   while ((pid = wait3(&status, WNOHANG, 0)) > 0)
      ;
}
#endif

//______________________________________________________________________________
void DaemonStart(int ignsigcld, int fdkeep, EService service)
{
   // Detach a daemon process from login session context.

   // If we were started by init (process 1) from the /etc/inittab file
   // there's no need to detach.
   // This test is unreliable due to an unavoidable ambiguity
   // if the process is started by some other process and orphaned
   // (i.e. if the parent process terminates before we are started).

   int fd;

#if !(defined(__CYGWIN__) && defined(__GNUC__))
   if (getppid() == 1) {
      if (service == kROOTD) printf ("ROOTD_PID=%ld\n", (long) getpid());
      goto out;
   }
#endif

   // Ignore the terminal stop signals (BSD).

#ifdef SIGTTOU
   signal(SIGTTOU, SIG_IGN);
#endif
#ifdef SIGTTIN
   signal(SIGTTIN, SIG_IGN);
#endif
#ifdef SIGTSTP
   signal(SIGTSTP, SIG_IGN);
#endif

   // If we were not started in the background, fork and let the parent
   // exit. This also guarantees the first child is not a process
   // group leader.

   int childpid;
   if ((childpid = fork()) < 0) {
      if (service == kROOTD) fprintf(stderr, "DaemonStart: can't fork first child\n");
      Error(gErrSys,kErrFatal, "DaemonStart: can't fork first child");
   } else if (childpid > 0) {
#ifdef SIGTSTP
      if (service == kROOTD) printf("ROOTD_PID=%d\n", childpid);
#endif
      exit(0);    // parent
   } else {
      if (gDebug > 2)
         ErrorInfo("DaemonStart: this is the child thread ... socket is: %d",fdkeep);
   }

   // First child process...

   // Disassociate from controlling terminal and process group.
   // Ensure the process can't reacquire a new controlling terminal.

#ifdef SIGTSTP

#ifdef USE_SETSID
   if (setsid() == -1) {
#else
   if (setpgrp(0, getpid()) == -1) {
#endif
      if (service == kROOTD)
         fprintf(stderr, "DaemonStart: can't change process group\n");
      Error(gErrSys,kErrFatal, "DaemonStart: can't change process group");
   }

   if ((fd = open("/dev/tty", O_RDWR)) >= 0) {
#if !defined(__hpux) && !defined(__sun) && \
    !(defined(__CYGWIN__) && defined(__GNUC__))
      ioctl(fd, TIOCNOTTY, 0);       // loose controlling tty
#endif
      close(fd);
   }

#else

   if (setpgrp() == -1) {
      if (service == kROOTD)
         fprintf(stderr,"DaemonStart: can't change process group\n");
      Error(gErrSys,kErrFatal, "DaemonStart: can't change process group");
   }

   signal(SIGHUP, SIG_IGN);    // immune from pgrp leader death

   if ((childpid = fork()) < 0) {
      if (service == kROOTD)
         fprintf(stderr,     "DaemonStart: can't fork second child\n");
      Error(gErrSys,kErrFatal, "DaemonStart: can't fork second child");
   } else if (childpid > 0) {
      if (service == kROOTD) printf("ROOTD_PID=%d (2nd fork)\n", childpid);
      exit(0);    // first child
   }

#endif
out:
   // Close any open file descriptors
   for (fd = 0; fd < NOFILE; fd++) {
      if ((fd != fdkeep) || (service == kPROOFD)) close(fd);
   }

   ResetErrno();   // probably got set to EBADF from a close

   // Move current directory to root, make sure we aren't on a mounted
   // file system.

   if (chdir("/") == -1)
      fprintf(stderr, "DaemonStart: cannot chdir to /\n");

   // Clear any inherited file mode creation mask

   umask(0);

   // See if the caller isn't interested in the exit status of its
   // children and doesn't want to have them become zombies and
   // clog up the system.
   // With SysV all we need to do is ignore the signal.
   // With BSD, however, we have to catch each signal
   // and execute the wait3() system call.

   if (ignsigcld) {
#ifdef USE_SIGCHLD
      signal(SIGCLD, SigChild);
#else
#if defined(__alpha) && !defined(linux)
      struct sigaction oldsigact, sigact;
      sigact.sa_handler = SIG_IGN;
      sigemptyset(&sigact.sa_mask);
      sigact.sa_flags = SA_NOCLDWAIT;
      sigaction(SIGCHLD, &sigact, &oldsigact);
#elif defined(__sun)
      sigignore(SIGCHLD);
#else
      signal(SIGCLD, SIG_IGN);
#endif
#endif
   }
}

} // namespace ROOT

