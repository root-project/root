// @(#)root/rpdutils:$Id$
// Author: Fons Rademakers   06/02/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// netpar                                                               //
//                                                                      //
// Set of parallel network routines for rootd daemon process. To be     //
// used when remote uses TPSocket to connect to rootd.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "RConfig.h"

// avoid warning due to wrong bzero prototype (used by FD_ZERO macro)
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <fcntl.h>
#include <errno.h>
#if defined(_AIX)
#include <strings.h>
#endif

#if (defined(R__AIX) && !defined(_AIX43)) || \
    (defined(R__SUNGCC3) && !defined(__arch64__))
#   define USE_SIZE_T
#elif defined(R__GLIBC) || defined(R__FBSD) || \
      (defined(R__SUNGCC3) && defined(__arch64__)) || \
      defined(R__OBSD) || defined(MAC_OS_X_VERSION_10_4) || \
      (defined(R__AIX) && defined(_AIX43)) || \
      (defined(R__SOLARIS) && defined(_SOCKLEN_T))
#   define USE_SOCKLEN_T
#endif

#include "rpdp.h"

extern int gDebug;

namespace ROOT {

extern ErrorHandler_t gErrSys;

int gParallel = 0;

static int    gMaxFd;
static int   *gPSockFd;
static int   *gWriteBytesLeft;
static int   *gReadBytesLeft;
static char **gWritePtr;
static char **gReadPtr;
static fd_set gFdSet;

//______________________________________________________________________________
static void InitSelect(int nsock)
{
   // Setup select masks.

   FD_ZERO(&gFdSet);
   gMaxFd = -1;
   for (int i = 0; i < nsock; i++) {
      FD_SET(gPSockFd[i], &gFdSet);
      if (gPSockFd[i] > gMaxFd)
         gMaxFd = gPSockFd[i];
   }
}

//______________________________________________________________________________
int NetParSend(const void *buf, int len)
{
   // Send buffer of specified length over the parallel sockets.
   // Returns len in case of success and -1 in case of error.

   int i, alen = len, nsock = gParallel;

   // If data buffer is < 4K use only one socket
   if (len < 4096)
      nsock = 1;

   for (i = 0; i < nsock; i++) {
      gWriteBytesLeft[i] = len/nsock;
      gWritePtr[i] = (char *)buf + (i*gWriteBytesLeft[i]);
   }
   gWriteBytesLeft[i-1] += len%nsock;

   InitSelect(nsock);

   // Send the data on the parallel sockets
   while (len > 0) {

      fd_set writeReady = gFdSet;

      int isel = select(gMaxFd+1, 0, &writeReady, 0, 0);
      if (isel < 0) {
         ErrorInfo("NetParSend: error on select");
         return -1;
      }

      for (i = 0; i < nsock; i++) {
         if (FD_ISSET(gPSockFd[i], &writeReady)) {
            if (gWriteBytesLeft[i] > 0) {
               int ilen;
again:
               ilen = send(gPSockFd[i], gWritePtr[i], gWriteBytesLeft[i], 0);
               if (ilen < 0) {
                  if (GetErrno() == EAGAIN)
                     goto again;
                  ErrorInfo("NetParSend: error sending for socket %d (%d)",
                            i, gPSockFd[i]);
                  return -1;
               }
               gWriteBytesLeft[i] -= ilen;
               gWritePtr[i] += ilen;
               len -= ilen;
            }
         }
      }
   }

   return alen;
}

//______________________________________________________________________________
int NetParRecv(void *buf, int len)
{
   // Receive buffer of specified length over parallel sockets.
   // Returns len in case of success and -1 in case of error.

   int i, alen = len, nsock = gParallel;

   // If data buffer is < 4K use only one socket
   if (len < 4096)
      nsock = 1;

   for (i = 0; i < nsock; i++) {
      gReadBytesLeft[i] = len/nsock;
      gReadPtr[i] = (char *)buf + (i*gReadBytesLeft[i]);
   }
   gReadBytesLeft[i-1] += len%nsock;

   InitSelect(nsock);

   // Recieve the data on the parallel sockets
   while (len > 0) {

      fd_set readReady = gFdSet;

      int isel = select(gMaxFd+1, &readReady, 0, 0, 0);
      if (isel < 0) {
         ErrorInfo("NetParRecv: error on select");
         return -1;
      }

      for (i = 0; i < nsock; i++) {
         if (FD_ISSET(gPSockFd[i], &readReady)) {
            if (gReadBytesLeft[i] > 0) {
               int ilen = recv(gPSockFd[i], gReadPtr[i], gReadBytesLeft[i], 0);
               if (ilen < 0) {
                  ErrorInfo("NetParRecv: error receiving for socket %d (%d)",
                            i, gPSockFd[i]);
                  return -1;
               } else if (ilen == 0) {
                  ErrorInfo("NetParRecv: EOF on socket %d (%d)",
                            i, gPSockFd[i]);
                  return 0;
               }
               gReadBytesLeft[i] -= ilen;
               gReadPtr[i] += ilen;
               len -= ilen;
            }
         }
      }
   }

   return alen;
}

//______________________________________________________________________________
int NetParOpen(int port, int size)
{
   // Open size parallel sockets back to client. Returns 0 in case of error,
   // and number of parallel sockets in case of success.

   struct sockaddr_in remote_addr;
   memset(&remote_addr, 0, sizeof(remote_addr));

#if defined(USE_SIZE_T)
   size_t remlen = sizeof(remote_addr);
#elif defined(USE_SOCKLEN_T)
   socklen_t remlen = sizeof(remote_addr);
#else
   int remlen = sizeof(remote_addr);
#endif

   if (!getpeername(NetGetSockFd(), (struct sockaddr *)&remote_addr, &remlen)) {
      remote_addr.sin_family = AF_INET;
      remote_addr.sin_port   = htons(port);

      gPSockFd = new int[size];

      for (int i = 0; i < size; i++) {
         if ((gPSockFd[i] = socket(AF_INET, SOCK_STREAM, 0)) < 0)
            Error(gErrSys, kErrFatal, "NetParOpen: can't create socket %d (%d)",
                     i, gPSockFd[i]);

         NetSetOptions(kROOTD, gPSockFd[i], 65535);

         if (connect(gPSockFd[i], (struct sockaddr *)&remote_addr, remlen) < 0)
            Error(gErrSys, kErrFatal, "NetParOpen: can't connect socket %d (%d)",
                  i, gPSockFd[i]);

         // Set non-blocking
         int val;
         if ((val = fcntl(gPSockFd[i], F_GETFL, 0)) < 0)
            Error(gErrSys, kErrFatal, "NetParOpen: can't get control flags");
         val |= O_NONBLOCK;
         if (fcntl(gPSockFd[i], F_SETFL, val) < 0)
            Error(gErrSys, kErrFatal, "NetParOpen: can't make socket non blocking");
      }

      gWriteBytesLeft = new int[size];
      gReadBytesLeft  = new int[size];
      gWritePtr       = new char*[size];
      gReadPtr        = new char*[size];

      // Close initial setup socket
      NetClose();

      gParallel = size;

      if (gDebug > 0)
         ErrorInfo("NetParOpen: %d parallel connections established", size);

   } else
      Error(gErrSys, kErrFatal, "NetParOpen: can't get peer name");

   return gParallel;
}

//______________________________________________________________________________
void NetParClose()
{
   // Close parallel sockets.

   for (int i = 0; i < gParallel; i++)
      close(gPSockFd[i]);

   if (gDebug > 0) {
      std::string host;
      NetGetRemoteHost(host);
      ErrorInfo("NetParClose: closing %d-stream connection to host %s",
                gParallel, host.data());
   }

   delete [] gPSockFd;
   delete [] gWriteBytesLeft;
   delete [] gReadBytesLeft;
   delete [] gWritePtr;
   delete [] gReadPtr;

   gParallel = 0;
}

} // namespace ROOT
