/******************************************************************************/
/*                                                                            */
/*                      X r d N e t C o n n e c t . c c                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdNetConnectCVSID = "$Id$";

#include "errno.h"
#include "fcntl.h"
#ifndef WIN32
#include "poll.h"
#include "unistd.h"
#include <sys/types.h>
#include <sys/socket.h>
#endif

#include "XrdNet/XrdNetConnect.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                               C o n n e c t                                */
/******************************************************************************/
  
int XrdNetConnect::Connect(             int       fd,
                           const struct sockaddr *name, 
                                        int       namelen, 
                                        int       tsec)
{
   int old_flags, new_flags, myRC;
   SOCKLEN_t myRClen = sizeof(myRC);

// If no timeout wanted, do a plain connect() which will timeout after 3
// minutes on most platforms.
//
   if (!tsec)
      {if (connect(fd, name, namelen)) return errno;
       return 0;
      }

// If a timeout is wanted, then we must convert this file descriptor to be
// non-blocking so that if a connection is not made immediately we can get
// control back and poll for completion for the specified amount of time.
// Regardless of outcome we will restore the original fd settings.
//
   old_flags = fcntl(fd, F_GETFL, 0);
   new_flags = old_flags | O_NDELAY | O_NONBLOCK;
   fcntl(fd, F_SETFL, new_flags);
   if (!connect(fd, name, namelen))  myRC = 0;
      else if (EINPROGRESS != net_errno) myRC = net_errno;
              else {struct pollfd polltab = {fd, POLLOUT|POLLWRNORM, 0};
                    do {myRC = poll(&polltab, 1, tsec*1000);} 
                       while(myRC < 0 && errno == EINTR);
                    if (myRC != 1) myRC = ETIMEDOUT;
                       else getsockopt(fd,SOL_SOCKET,SO_ERROR,(Sokdata_t)&myRC,&myRClen);
                   }
   fcntl(fd, F_SETFD, old_flags);
   return myRC;
}
