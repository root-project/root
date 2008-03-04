#ifndef __XRDNetWork_H__
#define __XRDNetWork_H__
/******************************************************************************/
/*                                                                            */
/*                         X r d N e t W o r k . h h                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <stdlib.h>
#ifndef WIN32
#include <strings.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#else
#include <Winsock2.h>
#endif

#include "XrdNet/XrdNet.hh"

class XrdSysError;
class XrdNetLink;
class XrdNetSecurity;

class XrdNetWork : public XrdNet
{
public:

// Accept()   processes incomming connections. When a succesful connection is
//            made, it returns an XrdNetLink object suitable for communications.
//            If a timeout occurs, or an XrdNetLink object cannot be allocated,
//            it returns 0. Options are those defined above. A timeout, in
//            seconds, may be specified.
//
XrdNetLink     *Accept(int opts=0,
                       int timeout=-1);

// Connect() Creates a socket and connects to the given host and port. Upon
//           success, it returns an XrdNetLink object suitable for peer
//           communications. Upon failure it returns zero. Options are as above.
//           A second timeout may be specified.
//
XrdNetLink     *Connect(const char *host,  // Destination host or ip address
                        int   port,        // Port number
                        int   opts=0,      // Options
                        int   timeout=-1   // Second timeout
                       );

// Relay() creates a UDP socket and optionally sets things up so that
//         messages will be routed to a particular host:port destination.
//         Upon success it returs the address of a XrdNetLink object that
//         be used to communicate with the dest. Upon failure return zero.
//
XrdNetLink     *Relay(const char  *dest=0, // Optional destination
                      int          opts=0  // Optional options as above
                     );

// When creating this object, you must specify the error routing object.
// Optionally, specify the security object to screen incomming connections.
// (if zero, no screening is done).
//
                XrdNetWork(XrdSysError *erp, XrdNetSecurity *secp=0)
                          : XrdNet(erp, secp) {}
               ~XrdNetWork() {}
};
#endif
