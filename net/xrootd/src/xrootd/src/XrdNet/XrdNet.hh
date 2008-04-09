#ifndef __XRDNET_H__
#define __XRDNET_H__
/******************************************************************************/
/*                                                                            */
/*                             X r d N e t . h h                              */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
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

#include "XrdNet/XrdNetBuffer.hh"
#include "XrdNet/XrdNetOpts.hh"

class XrdNetPeer;
class XrdNetSecurity;
class XrdSysError;

class XrdNet
{
public:

// Accept()   processes incomming connections. When a succesful connection is
//            made, it places the connection informatio in myPeer and returns
//            true (1). If a timeout or permanent error occurs, it returns
//            false (0). The opts are those defined above and timeout is
//            specified as seconds. Use this method to associate specialized
//            versions of XrdNetLink objects with the connection.
//
int             Accept(XrdNetPeer &myPeer,
                       int opts=0,
                       int timeout=-1);

// Bind()     binds this object to a communications medium. This may be TCP or
//            UDP network via the given port number or a Unix named socket
//            specified by path (the second form).
//            Bind() returns 0 upon success or -errno upon failure.
//
int             Bind(      int   port,             // Port number
                     const char *contype="tcp"     // "tcp" or "udp"
                    );
int             Bind(      char *path,             // Unix path < |109|
                     const char *contype="stream"  // stream | datagram
                    );

// Connect() Creates a socket and connects to the given host and port. Upon
//           success, it fills in the peer object describing the connection.
//           and returns true (1). Upon failure it returns zero. Opts are as 
//           above. A timeout, in seconds, may be specified. Use this method to
//           associate specialized versions of XrdNetLink with the connection.
//
int             Connect(XrdNetPeer &myPeer,
                        const char *host,  // Destination host or ip address
                        int   port,        // Port number
                        int   opts=0,      // Options
                        int   timeout=-1   // Second timeout
                       );

// Relay() creates a UDP socket and optionally decomposes a destination
//         of the form host:port. Upon success it fills in the Peer object
//         and return true (1). Upon failure, it returns false (0).
//
int             Relay(XrdNetPeer &Peer,   // Peer object to be initialized
                      const char *dest,   // Optional destination
                      int         opts=0  // Optional options as above
                     );

// Port() returns he port number, if any, bound to this network.
//
int             Port() {return Portnum;}

// Secure() adds the given NetSecurity object to the existing security
//          constraints. The supplied object is ultimately deleted in the
//          process and cannot be referenced.
//
void            Secure(XrdNetSecurity *secp);

// setDefaults() sets the default socket options, and buffer size for UDP
//               sockets (default is 32k) or window size for TCP sockets
//               (defaults to OS default).
//
void            setDefaults(int options, int buffsz=0)
                           {netOpts = options; Windowsz = buffsz;}

// setDomain() is used to indicate what part of the hostname is so common
//             that it may be trimmed of for incomming hostnames. This is
//             usually the domain in which this object resides/
//
void            setDomain(const char *dname)
                         {if (Domain) free(Domain);
                          Domain = strdup(dname);
                          Domlen = strlen(dname);
                         }

// Trim() trims off the domain name in hname (it's modified).
//
void            Trim(char *hname);

// unbind()    Destroys the association between this object and whatever
//             communications medium it was previously bound to.
//
void            unBind();

// WSzize()    Returns the actual RCVBUF window size. A value of zero
//             indicates that an error has occurred.
//
int            WSize();

// When creating this object, you must specify the error routing object.
// Optionally, specify the security object to screen incomming connections.
// (if zero, no screening is done).
//
                XrdNet(XrdSysError *erp, XrdNetSecurity *secp=0);
               ~XrdNet();

protected:

XrdSysError       *eDest;
XrdNetSecurity    *Police;
char              *Domain;
int                Domlen;
int                iofd;
int                Portnum;
int                PortType;
int                Windowsz;
int                netOpts;
int                BuffSize;
XrdNetBufferQ     *BuffQ;

private:

int                do_Accept_TCP(XrdNetPeer &myPeer, int opts);
int                do_Accept_UDP(XrdNetPeer &myPeer, int opts);
};
#endif
