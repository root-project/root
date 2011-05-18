// @(#)root/rpdutils:$Id$
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

#include "RConfig.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
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

#if (defined(R__AIX) && !defined(_AIX43)) || \
    (defined(R__SUNGCC3) && !defined(__arch64__))
#   define USE_SIZE_T
#elif defined(R__GLIBC) || defined(R__FBSD) || \
      (defined(R__SUNGCC3) && defined(__arch64__)) || \
      defined(R__OBSD) || defined(MAC_OS_X_VERSION_10_4) || \
      (defined(R__AIX) && defined(_AIX43))
#   define USE_SOCKLEN_T
#endif

#include "rpdp.h"

extern int gDebug;

namespace ROOT {

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

   int nAtt0 = 0;

tryagain:
   // Determine unique pipe path: try with /tmp/rootdSSH_<random_string>
   char fsun[25] = {0};
   if (access("/tmp",W_OK) == 0) {
      strncpy(fsun, "/tmp/rootdSSH_XXXXXX", 24);
   } else {
      strncpy(fsun, "rootdSSH_XXXXXX", 24);
   }
   mode_t oldumask = umask(0700);
   int itmp = mkstemp(fsun);
   Int_t nAtt = 0;
   while (itmp == -1 && nAtt < kMAXRSATRIES) {
      nAtt++;
      if (gDebug > 0)
         ErrorInfo("SshToolAllocateSocket: mkstemp failure (nAtt: %d, errno: %d)",
                   nAtt,errno);
      itmp = mkstemp(fsun);
   }
   umask(oldumask);
   if (itmp == -1) {
      ErrorInfo("SshToolAllocateSocket: mkstemp failed %d times - return",
                kMAXRSATRIES);
      return -1;
   } else {
      close(itmp);
      unlink(fsun);
   }
   nAtt0++;
   if (gDebug > 2)
      ErrorInfo("SshToolAllocateSocket: unique pipe name is %s (try: %d)",
                 fsun, nAtt0);

   // Save path ...
   strncpy(servAddr.sun_path, fsun, 104);

   // bind to socket
   if (bind(sd, (struct sockaddr *) &servAddr, sizeof(servAddr)) < 0) {
      if (errno == EADDRINUSE && nAtt0 < kMAXRSATRIES) {
         if (gDebug > 2)
            ErrorInfo
                ("SshToolAllocateSocket: address in use: try again (try: %d)");
         goto tryagain;
      } else {
         ErrorInfo
              ("SshToolAllocateSocket: unable to bind to socket %d (errno: %d)",
                sd, errno);
         return -1;
      }
   }
   // Activate listening
   if (listen(sd, 5)) {
      ErrorInfo
          ("SshToolAllocateSocket: can't activate listening (errno: %d)",
           errno);
      return -1;
   }

   // Change ownerships:
   // This operaton is possible only as root ... but not always is needed
   // so, do not stop in case of failure.
   struct stat sst;

   // The socket ...
   fstat(sd, &sst);
   if ((unsigned int)sst.st_uid != Uid || (unsigned int)sst.st_gid != Gid) {
      if (fchown(sd, Uid, Gid)) {
         if (gDebug > 0) {
            ErrorInfo("SshToolAllocateSocket: fchown: could not change socket"
                      " %d ownership (errno= %d) ",sd, errno);
            ErrorInfo("SshToolAllocateSocket: socket (uid,gid) are: %d %d",
                      sst.st_uid, sst.st_gid);
            ErrorInfo("SshToolAllocateSocket: may follow authentication problems");
         }
      }
   }
   // The path ...
   if (chown(fsun, Uid, Gid) != 0) {
      if (gDebug > 0) {
         ErrorInfo("SshToolAllocateSocket: chown: could not change path"
                     " '%s' ownership (errno= %d)",fsun, errno);
         ErrorInfo("SshToolAllocateSocket: path (uid,gid) are: %d %d",
                     sst.st_uid, sst.st_gid);
         ErrorInfo("SshToolAllocateSocket: may follow authentication problems");
      }
      return -1;
   }

   // Change permissions to access pipe to avoid hacking from a different
   // user account.
   // This is essential for security, so stop if you can't do it
   if (chmod(fsun, 0600)) {
      if (gDebug > 0) {
         ErrorInfo("SshToolAllocateSocket: chmod: could not change"
                   " '%s' permission (errno= %d)",fsun, errno);
         ErrorInfo("SshToolAllocateSocket: path (uid,gid) are: %d %d",
                   sst.st_uid, sst.st_gid);
         SshToolDiscardSocket(fsun,sd);
         return -1;
      }
   }

   // Fill output
   *pipe = strdup(fsun);

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
   memcpy((void *) servAddr.sun_path, (void *) Pipe, 108);
   servAddr.sun_path[107] = '\0';
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
   const char *okbuf = "KO";
   rc = send(sd, okbuf, strlen(okbuf), 0);
   if (rc != 2) {
      ErrorInfo
          ("SshToolNotifyFailure: sending might have been unsuccessful (bytes send: %d)",
           rc);
   }

   return 0;
}

//______________________________________________________________________________
int SshToolGetAuth(int UnixFd, const char *User)
{
   int auth = 0;

   if (gDebug > 2)
      ErrorInfo("SshToolGetAuth: accepting connections on socket %d"
                " for user %s",UnixFd,User);

   // Wait for verdict form sshd (via ssh2rpd ...)
   struct sockaddr sunAddr;
#if defined(USE_SIZE_T)
   size_t sunAddrLen = sizeof(sunAddr);
#elif defined(USE_SOCKLEN_T)
   socklen_t sunAddrLen = sizeof(sunAddr);
#else
   int sunAddrLen = sizeof(sunAddr);
#endif
   int newUnixFd =
       accept(UnixFd, (struct sockaddr *) &sunAddr, &sunAddrLen);
   if (newUnixFd < 0) {
      ErrorInfo
         ("SshToolGetAuth: problems while accepting new connection (errno: %d)", (int) errno);
      return auth;
   }

   int lenr[1], nr, len = 0;
   if ((nr = NetRecvRaw(newUnixFd, lenr, sizeof(lenr))) < 0) {
      ErrorInfo
         ("SshToolGetAuth: incorrect recv from ssh2rpd: bytes:%d, buffer:%d",
           nr, lenr[0]);
      return auth;
   }

   // Transform in human readable form
   len = ntohl(lenr[0]) + 1;
   char *sshAuth = 0;
   if (len > 0) {
      sshAuth = new char[len];
      if (sshAuth) {
         if ((nr = NetRecvRaw(newUnixFd, sshAuth, len)) != len) {
            ErrorInfo
             ("SshToolGetAuth: incorrect recv from ssh2rpd: nr:%d, buf:%s",
              nr, sshAuth);
         } else
            sshAuth[len-1] = 0;
         if (gDebug > 2)
            ErrorInfo("SshToolGetAuth: got: %s",sshAuth);

         // Check authentication and notify to client
         if (strncmp(sshAuth, "OK", 2) != 0) {
            ErrorInfo("SshToolGetAuth: user did not authenticate to sshd: %s (%d)",
                       sshAuth, strncmp(sshAuth, "OK", 2));
         } else {
            if (len > 3) {
               if (strcmp((const char *)(sshAuth+3), User) != 0) {
                  ErrorInfo("SshToolGetAuth: authenticated user not the same"
                            " as requested login username: %s (%s)",
                            sshAuth+3, User);
                  auth = -1;
               } else {
                  auth = 1;
               }
            } else
               auth = -1;
         }
         delete[] sshAuth;
      }
   }

   // Close local socket
   close(newUnixFd);

   return auth;
}

} // namespace ROOT
