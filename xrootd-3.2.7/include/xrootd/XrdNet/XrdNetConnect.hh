#ifndef __XRDNETCONNECT__
#define __XRDNETCONNECT__
/******************************************************************************/
/*                                                                            */
/*                      X r d N e t C o n n e c t . h h                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/types.h>
#ifndef WIN32
#include <sys/socket.h>
#else
#include <Winsock2.h>
#endif
  
class XrdNetConnect
{
public:

// Connect() performs the same function as sycall connect() however, it
//           can optionally apply a thread-safe timeout of tsec seconds.
//           It returns 0 upon success or errno upon failure.
//
static int  Connect(             int       fd,      // Open socket descriptor
                    const struct sockaddr *name,    // Address to connect to
                                 int       namelen, // Size of address
                                 int       tsec=-1);// Optional timeout

private:
        // Only this class is allowed to create and delete this object
        //
        XrdNetConnect() {}
       ~XrdNetConnect() {}
};
#endif
