#ifndef __XRDNETMSG_H__
#define __XRDNETMSG_H__
/******************************************************************************/
/*                                                                            */
/*                          X r d N e t M s g . h h                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#include <string.h>
#ifndef WIN32
#include <strings.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#else
#include <Winsock2.h>
#endif

class XrdSysError;

class XrdNetMsg
{
public:

// Send() sends a message to a host. If a destination is not supplied then the
//        default destination at construction time is used.
//        It returns one of three values:
//        <0 -> Message not sent due to error (e.g., iovec data > 4096 bytes)
//        =0 -> Message send (well as defined by UDP)
//        >0 -> Message not sent, timeout occured.
//
int           Send(const char *buff,          // The data to be send
                         int   blen=0,        // Length (strlen(buff) if zero)
                   const char *dest=0,        // Hostname to send UDP datagram
                         int   tmo=-1);       // Timeout in ms (-1 = none)

int           Send(const struct  iovec iov[], // Remaining parms as above
                         int     iovcnt,      // Number of elements in iovec
                   const char   *dest=0,      // Hostname to send UDP datagram
                         int     tmo=-1);     // Timeout in ms (-1 = none)

                XrdNetMsg(XrdSysError *erp, const char *dest=0);
               ~XrdNetMsg() {if (DestHN) free(DestHN);
                             if (DestIP) free(DestIP);
                            }

protected:
int OK2Send(int timeout, const char *dest);
int retErr(int ecode, const char *dest);

XrdSysError       *eDest;
char              *DestHN;
struct sockaddr   *DestIP;
int                DestSZ;
int                FD;
};
#endif
