// @(#)root/rpdutils:$Id$
// Author: Fons Rademakers   12/08/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// net                                                                  //
//                                                                      //
// Set of network routines for rootd daemon process.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <ROOT/RConfig.hxx>

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

#include "rpdp.h"
#include "rpderr.h"

extern int     gDebug;

namespace ROOT {

extern std::string gServName[3];

extern ErrorHandler_t gErrSys;
extern ErrorHandler_t gErrFatal;

static double  gBytesSent = 0;
static double  gBytesRecv = 0;

static std::string gOpenhost = "????";

static int                gTcpSrvSock;
static struct sockaddr_in gTcpSrvAddr;
static struct sockaddr_in gTcpCliAddr;

static int  gSockFd             = -1;
static SigPipe_t   gSigPipeHook = 0;
extern int  gParallel;

////////////////////////////////////////////////////////////////////////////////
/// return received bytes

double NetGetBytesRecv()
{
   return gBytesRecv;
}

////////////////////////////////////////////////////////////////////////////////
/// return sent bytes

double NetGetBytesSent()
{
   return gBytesSent;
}

////////////////////////////////////////////////////////////////////////////////
/// Return name of connected host

void NetGetRemoteHost(std::string &OpenHost)
{
   OpenHost = gOpenhost;
}

////////////////////////////////////////////////////////////////////////////////
/// return open socket descriptor

int NetGetSockFd()
{
   return gSockFd;
}

////////////////////////////////////////////////////////////////////////////////
/// reset byte counts

void NetResetByteCount()
{
   gBytesRecv = 0;
   gBytesSent = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Set hook for SIGPIPE calls

void NetSetSigPipeHook(SigPipe_t Hook)
{
   gSigPipeHook = Hook;
}

////////////////////////////////////////////////////////////////////////////////
/// Send exactly length bytes from buffer.

static int Sendn(int sock, const void *buffer, int length)
{
   if (sock < 0) return -1;

   int n, nsent = 0;
   const char *buf = (const char *)buffer;

   for (n = 0; n < length; n += nsent) {
      if ((nsent = send(sock, buf+n, length-n, 0)) <= 0) {
         Error(gErrFatal, -1, "Sendn: error (sock: %d): errno: %d",
               sock, GetErrno());
         return nsent;
      }
   }

   gBytesSent += n;

   return n;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive exactly length bytes into buffer. Returns number of bytes
/// received or 0 in case connection is closed. Returns -1 in case of error.

static int Recvn(int sock, void *buffer, int length)
{
   if (sock < 0) return -1;

   int n, nrecv = 0;
   char *buf = (char *)buffer;

   for (n = 0; n < length; n += nrecv) {
      while ((nrecv = recv(sock, buf+n, length-n, 0)) == -1 && GetErrno() == EINTR)
         ResetErrno();   // probably a SIGCLD that was caught
      if (nrecv == 0)
         break;          // EOF
      if (nrecv < 0) {
         Error(gErrFatal,-1,"Recvn: error (sock: %d): errno: %d",sock,GetErrno());
         return nrecv;
      }
   }

   gBytesRecv += n;

   return n;
}

////////////////////////////////////////////////////////////////////////////////
/// Send buffer of len bytes.

int NetSendRaw(const void *buf, int len)
{
   if (gParallel > 0) {

      if (NetParSend(buf, len) != len) {
         Error(gErrFatal,-1,"NetSendRaw: NetParSend error");
      }

   } else {

      if (gSockFd == -1 || !buf) return -1;
      if (Sendn(gSockFd, buf, len) != len) {
         Error(gErrFatal,-1,"NetSendRaw: Sendn error");
      }
   }

   return len;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer of maximum len bytes.

int NetRecvRaw(void *buf, int len)
{
   if (gParallel > 0) {

      if (NetParRecv(buf, len) != len) {
         Error(gErrFatal,-1,"NetRecvRaw: NetParRecv error");
      }

   } else {

      if (gSockFd == -1) return -1;
      if (Recvn(gSockFd, buf, len) < 0) {
         Error(gErrFatal,-1,"NetRecvRaw: Recvn error (gSockFd: %d)",gSockFd);
      }
   }

   return len;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer of maximum len bytes from generic socket sock.

int NetRecvRaw(int sock, void *buf, int len)
{
   if (sock == -1) return -1;

   if (Recvn(sock, buf, len) < 0) {
      Error(gErrFatal,-1,"NetRecvRaw: Recvn error (sock: %d, errno: %d)",sock,GetErrno());
   }

   return len;
}

////////////////////////////////////////////////////////////////////////////////
/// Send buffer of len bytes. Message will be of type "kind".

int NetSend(const void *buf, int len, EMessageTypes kind)
{
   int hdr[2];
   int hlen = sizeof(int) + len;
   hdr[0] = htonl(hlen);
   hdr[1] = htonl(kind);
   if (NetSendRaw(hdr, sizeof(hdr)) < 0)
      return -1;

   return NetSendRaw(buf, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Send integer. Message will be of type "kind".

int NetSend(int code, EMessageTypes kind)
{
   int hdr[3];
   int hlen = sizeof(int) + sizeof(int);
   hdr[0] = htonl(hlen);
   hdr[1] = htonl(kind);
   hdr[2] = htonl(code);
   return NetSendRaw(hdr, sizeof(hdr));
}

////////////////////////////////////////////////////////////////////////////////
/// Send a string. Message will be of type "kind".

int NetSend(const char *msg, EMessageTypes kind)
{
   int len = 0;

   if (msg)
      len = strlen(msg)+1;

   return NetSend(msg, len, kind);
}

////////////////////////////////////////////////////////////////////////////////

int NetSendAck()
{
   return NetSend(0, kROOTD_ACK);
}

////////////////////////////////////////////////////////////////////////////////

int NetSendError(ERootdErrors err)
{
   return NetSend(err, kROOTD_ERR);
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer. Returns the newly allocated buffer, the length
/// of the buffer and message type in kind.

int NetRecvAllocate(void *&buf, int &len, EMessageTypes &kind)
{
   int hdr[2] = { 0, 0 };

   if (NetRecvRaw(hdr, sizeof(hdr)) < 0)
      return -1;

   len = ntohl(hdr[0]) - sizeof(int);
   if (len < 0) len = 0;
   kind = (EMessageTypes) ntohl(hdr[1]);
   if (len) {
      buf = new char* [len];
      return NetRecvRaw(buf, len);
   }
   buf = 0;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a string of maximum len length. Returns message type in kind.
/// Return value is msg length.

int NetRecv(char *msg, int len, EMessageTypes &kind)
{
   int   mlen = 0;
   void *tmpbuf = nullptr;

   int res = NetRecvAllocate(tmpbuf, mlen, kind);
   char *buf = static_cast<char *>(tmpbuf);

   if (res < 0) {
      delete [] buf;
      return -1;
   }

   if (mlen == 0) {
      msg[0] = 0;
      delete [] buf;
      return 0;
   } else if (mlen > len-1) {
      strncpy(msg, buf, len-1);
      msg[len-1] = 0;
      mlen = len;
   } else {
      strncpy(msg, buf, mlen);
      msg[mlen] = 0;
   }

   delete [] buf;

   return mlen - 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Simulate TSocket::Recv(char *str, int max).

int NetRecv(char *msg, int max)
{
   EMessageTypes kind;

   return NetRecv((char *)msg, max, kind);
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the server's end.
/// We are passed a flag that says whether or not we are started
/// by a "master daemon" such as inetd. A master daemon will have
/// already waited for a message to arrive for us and will have
/// already set up the connection to the client. If we weren't
/// started by a master daemon, then we must wait for a client's
/// request to arrive.

int NetOpen(int inetdflag, EService service)
{
   socklen_t clilen = sizeof(gTcpCliAddr);

   if (inetdflag) {

      // When we're fired up by inetd, file decriptors 0, 1 and 2
      // are sockets to the client.

      gSockFd = 0;
      if (!getpeername(gSockFd, (struct sockaddr *)&gTcpCliAddr, &clilen)) {
         struct hostent *hp;
         if ((hp = gethostbyaddr((const char *)&gTcpCliAddr.sin_addr,
                                 sizeof(gTcpCliAddr.sin_addr), AF_INET)))
            gOpenhost = std::string(hp->h_name);
         else {
            struct in_addr *host_addr = (struct in_addr*)&gTcpCliAddr.sin_addr;
            gOpenhost = std::string(inet_ntoa(*host_addr));
         }
      }

      // Notify, if requested ...
      if (gDebug > 1)
         ErrorInfo("NetOpen: fired by inetd: connection from host %s"
                   " via socket %d", gOpenhost.data(),gSockFd);

      // Set several general performance network options
      NetSetOptions(service,gSockFd, 65535);

      return 0;
   }

   // For the concurrent server that's not initiated by inetd,
   // we have to wait for a connection request to arrive, then
   // fork a child to handle the client's request.
   // Beware that the accept() can be interrupted, such as by
   // a previously spawned child process that has terminated
   // (for which we caught the SIGCLD signal).

again:
   int newsock = accept(gTcpSrvSock, (struct sockaddr *)&gTcpCliAddr, &clilen);
   if (newsock < 0) {
      if (GetErrno() == EINTR) {
         ResetErrno();
         goto again;   // probably a SIGCLD that was caught
      }
      Error(gErrSys,kErrFatal, "NetOpen: accept error (errno: %d) ... socket %d",
                    GetErrno(),gTcpSrvSock);
      return 0;
   }

   struct hostent *hp;
   if ((hp = gethostbyaddr((const char *)&gTcpCliAddr.sin_addr,
                           sizeof(gTcpCliAddr.sin_addr), AF_INET)))
      gOpenhost = std::string(hp->h_name);
   else {
      struct in_addr *host_addr = (struct in_addr*)&gTcpCliAddr.sin_addr;
      gOpenhost = std::string(inet_ntoa(*host_addr));
   }

   // Fork a child process to handle the client's request.
   // The parent returns the child pid to the caller, which is
   // probably a concurrent server that'll call us again, to wait
   // for the next client request to this well-known port.

   int childpid;
   if ((childpid = fork()) < 0)
      Error(gErrSys,kErrFatal, "NetOpen: server can't fork");
   else if (childpid > 0) {    // parent
      close(newsock);
      return childpid;
   }

   // Child process continues here.
   // First close the original socket so that the parent
   // can accept any further requests that arrive there.
   // Then set "gSockFd" in our process to be the descriptor
   // that we are going to process.

   close(gTcpSrvSock);

   gSockFd = newsock;

   // Notify, if requested ...
   if (gDebug > 1)
      ErrorInfo("NetOpen: concurrent server: connection from host %s"
                " via socket %d", gOpenhost.data(), gSockFd);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Close the network connection.

void NetClose()
{
   if (gParallel > 0) {

      NetParClose();

   } else {

      close(gSockFd);
      if (gDebug > 0)
         ErrorInfo("NetClose: host = %s, fd = %d",
                   gOpenhost.data(), gSockFd);
      gSockFd = -1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialize the network connection for the server, when it has *not*
/// been invoked by inetd. Used by rootd.

int NetInit(EService servtype, int port1, int port2, int tcpwindowsize)
{
   // We weren't started by a master daemon.
   // We have to create a socket ourselves and bind our well-known
   // address to it.

   std::string service = gServName[servtype];

   if (port1 <= 0) {
      if (service.length()) {
         struct servent *sp = getservbyname(service.data(), "tcp");
         if (!sp) {
            if (servtype == kROOTD) {
               port1 = 1094;
            } else if (servtype == kPROOFD) {
               port1 = 1093;
            } else {
               fprintf(stderr,"NetInit: unknown service: %s/tcp\n", service.data());
               Error(gErrFatal, kErrFatal,
                     "NetInit: unknown service: %s/tcp", service.data());
            }
         } else {
            port1 = ntohs(sp->s_port);
         }
         port2 += port1;   // in this case, port2 is relative to service port
      } else {
         fprintf(stderr, "NetInit: must specify either service or port\n");
         Error(gErrFatal,kErrFatal,
                         "NetInit: must specify either service or port");
      }
   }

   // Create the socket and bind our local address so that any client can
   // send to us.

   if ((gTcpSrvSock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      fprintf(stderr,     "NetInit: can't create socket\n");
      Error(gErrSys,kErrFatal, "NetInit: can't create socket");
      return gTcpSrvSock;
   }

   int val = 1;
   if (setsockopt(gTcpSrvSock, SOL_SOCKET, SO_REUSEADDR, (char*) &val,
                  sizeof(val)) == -1) {
      fprintf(stderr,     "NetInit: can't set SO_REUSEADDR socket option\n");
      Error(gErrSys, kErrFatal, "NetInit: can't set SO_REUSEADDR socket option");
   }

   // Set several general performance network options
   NetSetOptions(kROOTD,gTcpSrvSock, tcpwindowsize);

   memset(&gTcpSrvAddr, 0, sizeof(gTcpSrvAddr));
   gTcpSrvAddr.sin_family      = AF_INET;
   gTcpSrvAddr.sin_addr.s_addr = htonl(INADDR_ANY);

   int port;
   for (port= port1; port <= port2; port++) {
      gTcpSrvAddr.sin_port = htons(port);
      if (bind(gTcpSrvSock, (struct sockaddr *) &gTcpSrvAddr,
               sizeof(gTcpSrvAddr)) == 0) break;
   }

   if (port > port2) {
      fprintf(stderr, "NetInit: can't bind local address to ports %d-%d\n", port1, port2);
      Error(gErrSys, kErrFatal, "NetInit: can't bind local address to ports %d-%d", port1, port2);
   }

   printf("ROOTD_PORT=%d\n", port);
   port1 = port;

   // And set the listen parameter, telling the system that we're
   // ready to accept incoming connection requests.

   //   listen(gTcpSrvSock, 5);
   if (listen(gTcpSrvSock, 5)==-1) {
      ErrorInfo("NetInit: listen: error (errno: %d)",GetErrno());
   }

   if (gDebug > 0)
      ErrorInfo("NetInit: socket %d listening on port %d", gTcpSrvSock,
                ntohs(gTcpSrvAddr.sin_port));

   return gTcpSrvSock;
}

////////////////////////////////////////////////////////////////////////////////
/// Set some options for network socket.

void NetSetOptions(EService serv, int sock, int tcpwindowsize)
{
   int val = 1;

   if (serv == kROOTD) {
      if (!setsockopt(sock,IPPROTO_TCP,TCP_NODELAY,(char *)&val,sizeof(val)))
         if (gDebug > 0)
            ErrorInfo("NetSetOptions: set TCP_NODELAY");
      if (!setsockopt(sock,SOL_SOCKET,SO_KEEPALIVE,(char *)&val,sizeof(val))) {
         if (gDebug > 0)
            ErrorInfo("NetSetOptions: set SO_KEEPALIVE");
         if (gSigPipeHook != 0)
            signal(SIGPIPE, (*gSigPipeHook));   // handle SO_KEEPALIVE failure
      }
   }

   val = tcpwindowsize;
   if (!setsockopt(sock,SOL_SOCKET,SO_SNDBUF,(char *)&val,sizeof(val)))
      if (gDebug > 0)
         ErrorInfo("NetSetOptions: set SO_SNDBUF %d", val);
   if (!setsockopt(sock,SOL_SOCKET,SO_RCVBUF,(char *)&val,sizeof(val)))
      if (gDebug > 0)
         ErrorInfo("NetSetOptions: set SO_RCVBUF %d", val);

   if (gDebug > 0) {
      socklen_t optlen = sizeof(val);
      if (serv == kROOTD) {
         getsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (char*)&val, &optlen);
         ErrorInfo("NetSetOptions: get TCP_NODELAY: %d", val);
         getsockopt(sock, SOL_SOCKET, SO_KEEPALIVE, (char*)&val, &optlen);
         ErrorInfo("NetSetOptions: get SO_KEEPALIVE: %d", val);
      }
      getsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char*)&val, &optlen);
      ErrorInfo("NetSetOptions: get SO_SNDBUF: %d", val);
      getsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char*)&val, &optlen);
      ErrorInfo("NetSetOptions: get SO_RCVBUF: %d", val);
   }
}

} // namespace ROOT
