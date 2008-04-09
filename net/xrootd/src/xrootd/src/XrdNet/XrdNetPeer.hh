#ifndef __XRDNETPEER_H__
#define __XRDNETPEER_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d N e t P e e r . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

#include <stdlib.h>

#include "XrdNet/XrdNetBuffer.hh"

class XrdNetPeer
{
public:

int             fd;       // File descriptor
struct sockaddr InetAddr; // Incomming peer network address
char           *InetName; // Incomming peer host name (must be copied)
XrdNetBuffer   *InetBuff; // Incomming datagram buffer for UDP accepts

                XrdNetPeer() {InetName = 0; InetBuff = 0;}
               ~XrdNetPeer() {if (InetName) free(InetName);
                              if (InetBuff) InetBuff->Recycle();
                             }
};
#endif
