// @(#)root/rootd:$Name:$:$Id:$
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
#include <errno.h>

#if defined(linux)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef R__GLIBC
#         define R__GLIBC
#      endif
#   endif
#endif

#include "rootdp.h"

#if defined(_AIX) || (defined(__FreeBSD__) && !defined(__alpha__))
#   define USE_SIZE_T
#elif defined(R__GLIBC) || (defined(__FreeBSD__) && defined(__alpha__))
#   define USE_SOCKLEN_T
#endif

int gParallel = 0;

static int   *gPSockFd;
static int   *gWriteBytesLeft;
static int   *gReadBytesLeft;
static char **gWritePtr;
static char **gReadPtr;

//______________________________________________________________________________
int NetParSend(const void *buf, int len)
{
   return 0;
}

//______________________________________________________________________________
int NetParRecv(void *buf, int len)
{
   return 0;
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

   if (!getpeername(gSockFd, (struct sockaddr *)&remote_addr, &remlen)) {
      remote_addr.sin_family = AF_INET;
      remote_addr.sin_port   = htons(port);

      gPSockFd = new int[size];

      for (int i = 0; i < size; i++) {
         if ((gPSockFd[i] = socket(AF_INET, SOCK_STREAM, 0)) < 0)
            ErrorSys(kErrFatal, "NetParOpen: can't create socket %d (%d)",
                     i, gPSockFd[i]);

         NetSetOptions(gPSockFd[i], 65535);

         if (connect(gPSockFd[i], (struct sockaddr *)&remote_addr, remlen) < 0)
            ErrorSys(kErrFatal, "NetParOpen: can't connect socket %d (%d)",
                     i, gPSockFd[i]);

         // Set non-blocking

      }

      gWriteBytesLeft = new int[size];
      gReadBytesLeft  = new int[size];
      gWritePtr       = new char*[size];
      gReadPtr        = new char*[size];

      gParallel = size;

      // Close initial setup socket
      NetClose();

      if (gDebug > 0)
         ErrorInfo("NetParOpen: %d parallel connections established", size);

   } else
      ErrorSys(kErrFatal, "NetParOpen: can't get peer name");

   return gParallel;
}

//______________________________________________________________________________
void NetParClose()
{
   // Close parallel sockets.

   for (int i = 0; i < gParallel; i++)
      close(gPSockFd[i]);

   delete [] gPSockFd;
   delete [] gWriteBytesLeft;
   delete [] gReadBytesLeft;
   delete [] gWritePtr;
   delete [] gReadPtr;

   gParallel = 0;

   if (gDebug > 0)
      ErrorInfo("NetParClose: file = %s", gFile);
}
