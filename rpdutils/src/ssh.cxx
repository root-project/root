// @(#)root/rpdutils:$Name:  $:$Id: ssh.cxx,v 1.2 2003/08/29 17:23:32 rdm Exp $
// Author: Gerardo Ganis    7/4/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Set of utilities for rootd/proofd daemon SSH authentication.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>
#include <sys/types.h>
#include <time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif
#if defined(__MACH__) && !defined(__APPLE__)
#   define R__GLIBC
#endif

#if defined(_AIX) || (defined(__FreeBSD__) && !defined(__alpha__))
#   define USE_SIZE_T
#elif defined(R__GLIBC) || (defined(__FreeBSD__) && defined(__alpha__))
#   define USE_SOCKLEN_T
#endif

#include "rpdp.h"

namespace ROOT {

//--- Globals ------------------------------------------------------------------
extern int gDebug;


//______________________________________________________________________________
int SshToolAllocateSocket(unsigned int Uid, unsigned int Gid, char **pipe)
{
   // Allocates internal UNIX socket for SSH-like authentication.
   // Sets socket ownership to user for later use.
   // On success returns ID of allocated socket and related pipe, -1 otherwise.

   if (gDebug > 2)
      ErrorInfo("SshToolAllocateSocket: enter: Uid:%d Gid:%d", Uid, Gid);

   // Open socket
   int sd;
   if ((sd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      ErrorInfo("SshToolAllocateSocket: error opening socket");
      return -1;
   }
   // Prepare binding ...
   struct sockaddr_un servAddr;
   servAddr.sun_family = AF_UNIX;

   // Determine unique pipe path: try with /tmp/rootdSSH_<random_string>
   char fsun[] = "/tmp/rootdSSH_XXXXXX";
   mktemp(fsun);
   if (gDebug > 2)
      ErrorInfo("SshToolAllocateSocket: unique pipe name is %s", fsun);

   // Save path ...
   strcpy(servAddr.sun_path, fsun);

   // bind to socket
   if (bind(sd, (struct sockaddr *) &servAddr, sizeof(servAddr)) < 0) {
      ErrorInfo("SshToolAllocateSocket: unable to bind to socket %d", sd);
      return -1;
   }
   // Activate listening
   if (listen(sd, 5)) {
      ErrorInfo
          ("SshToolAllocateSocket: can't activate listening (errno: %d)",
           errno);
      return -1;
   }

   // Change ownerships and try to change them if needed
   // This operaton is possible only as root ... but not always is needed
   // so, do not stop in case of failure.
   struct stat sst;

   // The socket ...
   fstat(sd, &sst);
   if ((unsigned int)sst.st_uid != Uid || (unsigned int)sst.st_gid != Gid) {
      if (fchown(sd, Uid, Gid)) {
         if (gDebug > 0) {
            ErrorInfo
                ("SshToolAllocateSocket: fchown: could not change socket %d ownership (errno= %d) ",
                 sd, errno);
            ErrorInfo("SshToolAllocateSocket: socket (uid,gid) are: %d %d",
                      sst.st_uid, sst.st_gid);
            ErrorInfo
                ("SshToolAllocateSocket: may follow authentication problems");
         }
      }
   }
   // The path ...
   stat(fsun, &sst);
   if ((unsigned int)sst.st_uid != Uid || (unsigned int)sst.st_gid != Gid) {
      if (chown(fsun, Uid, Gid)) {
         if (gDebug > 0) {
            ErrorInfo
                ("SshToolAllocateSocket: chown: could not change path '%s' ownership (errno= %d)",
                 fsun, errno);
            ErrorInfo("SshToolAllocateSocket: path (uid,gid) are: %d %d",
                      sst.st_uid, sst.st_gid);
            ErrorInfo
                ("SshToolAllocateSocket: may follow authentication problems");
         }
      }
   }
   // Fill output
   strcpy(*pipe, fsun);

   // return socket fd
   return sd;
}


//______________________________________________________________________________
void SshToolDiscardSocket(const char *pipe, int sockfd)
{
   // Discards socket.

   if (gDebug > 2)
      ErrorInfo
          ("SshToolDiscardSocket: discarding socket: pipe: %s, fd: %d",
           pipe, sockfd);

   // Unlink pipe
   if (unlink(pipe) == -1) {
      if (GetErrno() != ENOENT) {
         ErrorInfo("SshToolDiscardSocket: unable to unlink %s"
                   "(errno: %d, ENOENT= %d)", pipe, GetErrno(), ENOENT);
      }
   }
   // close socket
   close(sockfd);
}

//______________________________________________________________________________
int SshToolNotifyFailure(const char *Pipe)
{
   // Notifies failure of SSH authentication to relevant rootd/proofd process.

   if (gDebug > 2)
      ErrorInfo("SshToolNotifyFailure: notifying failure to pipe %s\n",
                Pipe);

   // Preparing socket connection
   int sd;
   struct sockaddr_un servAddr;
   servAddr.sun_family = AF_UNIX;
   strcpy(servAddr.sun_path, Pipe);
   if ((sd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      ErrorInfo("SshToolNotifyFailure: cannot open socket: exiting ");
      return 1;
   }
   // Connecting to socket
   int rc;
   if ((rc =
        connect(sd, (struct sockaddr *) &servAddr,
                sizeof(servAddr))) < 0) {
      ErrorInfo("SshToolNotifyFailure: cannot connect socket: exiting ");
      return 1;
   }
   // Sending "KO" ...
   char *okbuf = "KO";
   rc = send(sd, okbuf, strlen(okbuf), 0);
   if (rc != 2) {
      ErrorInfo
          ("SshToolNotifyFailure: sending might have been unsuccessful (bytes send: %d)",
           rc);
   }

   return 0;
}

//______________________________________________________________________________
int SshToolGetAuth(int UnixFd)
{
   int Auth = 0;

   if (gDebug > 2)
      ErrorInfo("SshToolGetAuth: accepting connections on socket %d",
                UnixFd);

   // Wait for verdict form sshd (via ssh2rpd ...)
   struct sockaddr SunAddr;
#if defined(USE_SIZE_T)
   size_t SunAddrLen = sizeof(SunAddr);
#elif defined(USE_SOCKLEN_T)
   socklen_t SunAddrLen = sizeof(SunAddr);
#else
   int SunAddrLen = sizeof(SunAddr);
#endif
   int NewUnixFd =
       accept(UnixFd, (struct sockaddr *) &SunAddr, &SunAddrLen);

   char SshAuth[2] = { 0 };
   int nr = NetRecvRaw(NewUnixFd, SshAuth, 2);
   if (nr != 2) {
      ErrorInfo
          ("RootdSshAuth: incorrect reception from ssh2rpd: bytes:%d, buffer:%s ",
           nr, SshAuth);
   }
   // Check authentication and notify to client
   if (strncmp(SshAuth, "OK", 2) != 0) {
      ErrorInfo("RootdSshAuth: user did not authenticate to sshd: %s (%d)",
                SshAuth, strncmp(SshAuth, "OK", 2));
   } else {
      Auth = 1;
   }

   // Close local socket
   close(NewUnixFd);

   return Auth;
}

} // namespace ROOT
