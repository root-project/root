/******************************************************************************/
/*                                                                            */
/*                             X r d N e t . c c                              */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
 
//         $Id$

const char *XrdNetCVSID = "$Id$";

#include <errno.h>
#include <stdio.h>
#include <string.h>
#ifndef WIN32
#include <poll.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#else
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdNet/XrdNetSecurity.hh"
#include "XrdNet/XrdNetSocket.hh"

#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
#define XRDNET_UDPBUFFSZ 32768
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdNet::XrdNet(XrdSysError *erp, XrdNetSecurity *secp)
{
   iofd   = PortType = -1;
   eDest  = erp;
   Police = secp;
   Domlen = Portnum = Windowsz = netOpts = 0;
   Domain = 0;
   BuffQ  = 0;
}
 
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdNet::~XrdNet()
{
   unBind();
   if (Domain) free(Domain);
}

/******************************************************************************/
/*                                A c c e p t                                 */
/******************************************************************************/

int XrdNet::Accept(XrdNetPeer &myPeer, int opts, int timeout)
{
   int retc;

// Make sure we are bound to a port
//
   if (iofd < 0) 
      {eDest->Emsg("Accept", "Network not bound to a port.");
       return 0;
      }

// Setup up the poll structure to wait for new connections
//
  do {if (timeout >= 0)
         {struct pollfd sfd = {iofd,
                               POLLIN|POLLRDNORM|POLLRDBAND|POLLPRI|POLLHUP,0};
          do {retc = poll(&sfd, 1, timeout*1000);}
             while(retc < 0 && (errno == EAGAIN || errno == EINTR));
          if (!retc)
             {if (!(opts & XRDNET_NOEMSG))
                 eDest->Emsg("Accept", "Accept timed out.");
              return 0;
             }
         }
     } while(!(PortType == SOCK_STREAM ? do_Accept_TCP(myPeer, opts)
                                       : do_Accept_UDP(myPeer, opts)));

// Accept completed, trim the host name if a domain has been specified,
//
   if (Domain && !(opts & XRDNET_NODNTRIM)) Trim(myPeer.InetName);
   return 1;
}
  
/******************************************************************************/
/*                                  B i n d                                   */
/******************************************************************************/
  
int XrdNet::Bind(int bindport, const char *contype)
{
    XrdNetSocket mySocket(eDest);
    int opts = XRDNET_SERVER | netOpts;
    int buffsz = Windowsz;

// Close any open socket here
//
   unBind();

// Get correct option settings
//
   if (*contype != 'u') PortType = SOCK_STREAM;
      else {PortType = SOCK_DGRAM;
            opts |= XRDNET_UDPSOCKET;
            if (!buffsz) buffsz = XRDNET_UDPBUFFSZ;
           }

// Try to open and bind to this port
//
   if (mySocket.Open(0, bindport, opts, buffsz) < 0)
      return -mySocket.LastError();

// Success, get the socket number and return
//
   iofd = mySocket.Detach();

// Obtain port number of generic port being used
//
   Portnum = (bindport ? bindport : XrdNetDNS::getPort(iofd));

// For udp sockets, we must allocate a buffer queue object
//
   if (PortType == SOCK_DGRAM)
      {BuffSize = buffsz;
       BuffQ = new XrdNetBufferQ(buffsz);
      }
   return 0;
}
  
/******************************************************************************/

int XrdNet::Bind(char *path, const char *contype)
{
    XrdNetSocket mySocket(eDest);
    int opts = XRDNET_SERVER | netOpts;
    int buffsz = Windowsz;

// Make sure this is a path and not a host name
//
   if (*path != '/')
      {eDest->Emsg("Bind", "Invalid bind path -", path);
       return -EINVAL;
      }

// Close any open socket here
//
   unBind();

// Get correct option settings
//
   if (*contype != 'd') PortType = SOCK_STREAM;
      else {PortType = SOCK_DGRAM;
            opts |= XRDNET_UDPSOCKET;
            if (!buffsz) buffsz = XRDNET_UDPBUFFSZ;
           }

// Try to open and bind to this path
//
   if (mySocket.Open(path, -1, opts, buffsz) < 0) return -mySocket.LastError();

// Success, get the socket number and return
//
   iofd = mySocket.Detach();

// For udp sockets, we must allocate a buffer queue object
//
   if (PortType == SOCK_DGRAM)
      {BuffSize = buffsz;
       BuffQ = new XrdNetBufferQ(buffsz);
      }
   return 0;
}

/******************************************************************************/
/*                               C o n n e c t                                */
/******************************************************************************/

int XrdNet::Connect(XrdNetPeer &myPeer,
                    const char *host, int port, int opts, int tmo)
{
   XrdNetSocket mySocket(opts & XRDNET_NOEMSG ? 0 : eDest);
   struct sockaddr *sap;
   int buffsz = Windowsz;

// Determine appropriate options
//
   if (!opts) opts = netOpts;
   if ((opts & XRDNET_UDPSOCKET) && !buffsz) buffsz = XRDNET_UDPBUFFSZ;
   if (tmo > 0) opts = (opts & ~XRDNET_TOUT) | (tmo > 255 ? 255 : tmo);

// Now perform the connect and return the peer structure if successful
//
   if (mySocket.Open(host, port, opts, buffsz) < 0) return 0;
   if (myPeer.InetName) free(myPeer.InetName);
   if ((opts & XRDNET_UDPSOCKET) || !host) 
      {myPeer.InetName = strdup("n/a");
       memset((void *)&myPeer.InetAddr, 0, sizeof(myPeer.InetAddr));
      } else {
       const char *pn = mySocket.Peername(&sap);
       if (pn) {memcpy((void *)&myPeer.InetAddr, sap, sizeof(myPeer.InetAddr));
                myPeer.InetName = strdup(pn);
                if (Domain && !(opts & XRDNET_NODNTRIM)) Trim(myPeer.InetName);
               } else {
                memset((void *)&myPeer.InetAddr, 0, sizeof(myPeer.InetAddr));
                myPeer.InetName = strdup("unknown");
               }
      }
   myPeer.fd = mySocket.Detach();
   return 1;
}

/******************************************************************************/
/*                                 R e l a y                                  */
/******************************************************************************/
  
int XrdNet::Relay(XrdNetPeer &Peer, const char *dest, int opts)
{
   return Connect(Peer, dest, -1, opts | XRDNET_UDPSOCKET);
}
  
/******************************************************************************/
/*                                S e c u r e                                 */
/******************************************************************************/
  
void XrdNet::Secure(XrdNetSecurity *secp)
{

// If we don't have a Police object then use the one supplied. Otherwise
// merge the supplied object into the existing object.
//
   if (Police) Police->Merge(secp);
      else     Police = secp;
}

/******************************************************************************/
/*                                  T r i m                                   */
/******************************************************************************/
  
void XrdNet::Trim(char *hname)
{
  int k = strlen(hname);
  char *hnp;

  if (Domlen && k > Domlen)
     {hnp = hname + (k - Domlen);
      if (!strcmp(Domain, hnp)) *hnp = '\0';
     }
}

/******************************************************************************/
/*                                u n B i n d                                 */
/******************************************************************************/
  
void XrdNet::unBind()
{
   if (iofd >= 0) {close(iofd); iofd=-1; Portnum=0;}
   if (BuffQ) {delete BuffQ; BuffQ = 0;}
}

/******************************************************************************/
/*                                 W S i z e                                  */
/******************************************************************************/
  
int XrdNet::WSize()
{
  int wsz;

  if (iofd >= 0 && !XrdNetSocket::getWindow(iofd, wsz, eDest)) return wsz;
  return 0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                         d o _ A c c e p t _ T C P                          */
/******************************************************************************/
  
int XrdNet::do_Accept_TCP(XrdNetPeer &myPeer, int opts)
{
  static int noAcpt = 0;
  int        newfd;
  char      *hname;
  struct sockaddr addr;
  SOCKLEN_t  addrlen = sizeof(addr);

// Accept a connection
//
   do {newfd = accept(iofd, &addr, &addrlen);}
      while(newfd < 0 && errno == EINTR);

   if (newfd < 0)
      {if (errno != EMFILE || !(0x1ff & noAcpt++))
          eDest->Emsg("Accept", errno, "perform accept");
       return 0;
      }

// Authorize by ip address or full (slow) hostname format
//
   if (Police)
      {if (!(hname = Police->Authorize(&addr)))
          {eDest->Emsg("Accept", EACCES, "accept TCP connection from",
                      (hname = XrdNetDNS::getHostName(addr)));
           free(hname);
           close(newfd);
           return 0;
          }
      } else hname = (opts & XRDNET_NORLKUP ? XrdNetDNS::getHostID(addr)
                                            : XrdNetDNS::getHostName(addr));

// Set all required fd options are set
//
   XrdNetSocket::setOpts(newfd, (opts ? opts : netOpts));

// Fill out the peer structure and success
//
   myPeer.fd       = newfd;
   memcpy(&(myPeer.InetAddr), &addr, sizeof(myPeer.InetAddr));
   if (myPeer.InetName) free(myPeer.InetName);
   myPeer.InetName = hname;
   return 1;
}

/******************************************************************************/
/*                         d o _ A c c e p t _ U D P                          */
/******************************************************************************/
  
int XrdNet::do_Accept_UDP(XrdNetPeer &myPeer, int opts)
{
  char           *hname = 0;
  int             dlen;
  struct sockaddr addr;
  SOCKLEN_t       addrlen = sizeof(addr);
  XrdNetBuffer   *bp;

// For UDP connections, get a buffer for the message. To be thread-safe, we
// must actually receive the message to maintain the host-datagram pairing.
//
   if (!(bp = BuffQ->Alloc()))
      {eDest->Emsg("Accept", ENOMEM, "accept UDP message");
       return 0;
      }

// Read the message and get the host address
//
   do {dlen = recvfrom(iofd,(Sokdata_t)bp->data,BuffSize-1,0,&addr,&addrlen);
      } while(dlen < 0 && errno == EINTR);

   if (dlen < 0)
      {eDest->Emsg("Receive", errno, "perform UDP recvfrom()");
       BuffQ->Recycle(bp);
       return 0;
      } else bp->data[dlen] = '\0';

// Authorize this connection. We don't accept messages that set the
// loopback address since this can be trivially spoofed in UDP packets.
//
   if (XrdNetDNS::isLoopback(addr)
   || (Police && !(hname = Police->Authorize(&addr))))
      {eDest->Emsg("Accept", -EACCES, "accept connection from",
                     (hname = XrdNetDNS::getHostName(addr)));
       free(hname);
       BuffQ->Recycle(bp);
       return 0;
      } else {
       if (!hname) hname=(opts & XRDNET_NORLKUP ? XrdNetDNS::getHostID(addr)
                                                : XrdNetDNS::getHostName(addr));
      }

// Fill in the peer structure. We use our base FD for outgoing messages.
// Note that XrdNetLink object never closes this FDS for UDP messages.
//
   myPeer.fd = (opts & XRDNET_NEWFD ? dup(iofd) : iofd);
   memcpy(&(myPeer.InetAddr), &addr, sizeof(myPeer.InetAddr));
   if (myPeer.InetName) free(myPeer.InetName);
   myPeer.InetName = hname;
   if (myPeer.InetBuff) myPeer.InetBuff->Recycle();
   myPeer.InetBuff = bp;
   return 1;
}
