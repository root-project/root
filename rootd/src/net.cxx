// @(#)root/rootd:$Name:  $:$Id: net.cxx,v 1.7 2000/12/12 18:20:33 rdm Exp $
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


double  gBytesSent = 0;
double  gBytesRecv = 0;

static char openhost[256] = "????";

static int                tcp_srv_sock;
static struct sockaddr_in tcp_srv_addr;
static struct sockaddr_in tcp_cli_addr;


//______________________________________________________________________________
static void SigPipe(int)
{
   // After SO_KEEPALIVE times out we probably get a SIGPIPE.

   ErrorInfo("SigPipe: got a SIGPIPE");
   RootdClose();
   exit(1);
}

//______________________________________________________________________________
static int Sendn(int sock, const void *buffer, int length)
{
   // Send exactly length bytes from buffer.

   if (sock < 0) return -1;

   int n, nsent = 0;
   const char *buf = (const char *)buffer;

   for (n = 0; n < length; n += nsent) {
      if ((nsent = send(sock, buf+n, length-n, 0)) <= 0)
         return nsent;
   }

   gBytesSent += n;

   return n;
}

//______________________________________________________________________________
static int Recvn(int sock, void *buffer, int length)
{
   // Receive exactly length bytes into buffer. Returns number of bytes
   // received. Returns -1 in case of error.

   if (sock < 0) return -1;

   int n, nrecv = 0;
   char *buf = (char *)buffer;

   for (n = 0; n < length; n += nrecv) {
      while ((nrecv = recv(sock, buf+n, length-n, 0)) == -1 && GetErrno() == EINTR)
         ResetErrno();     // probably a SIGCLD that was caught
      if (nrecv < 0)
         return nrecv;
      else if (nrecv == 0)
         break;         // EOF
   }

   gBytesRecv += n;

   return n;
}

//______________________________________________________________________________
void NetInit(const char *service, int port)
{
   // Initialize the network connection for the server, when it has *not*
   // been invoked by inetd.

   // We weren't started by a master daemon.
   // We have to create a socket ourselves and bind our well-known
   // address to it.

   memset(&tcp_srv_addr, 0, sizeof(tcp_srv_addr));
   tcp_srv_addr.sin_family      = AF_INET;
   tcp_srv_addr.sin_addr.s_addr = htonl(INADDR_ANY);

   if (service) {

      if (port > 0)
         tcp_srv_addr.sin_port = htons(port);
      else {
         struct servent *sp;
         if ((sp = getservbyname(service, "tcp")) == 0)
            ErrorFatal(kErrFatal, "NetInit: unknown service: %s/tcp", service);
         tcp_srv_addr.sin_port = sp->s_port;
      }

   } else {

      if (port <= 0)
         ErrorFatal(kErrFatal, "NetInit: must specify either service or port");
      tcp_srv_addr.sin_port = htons(port);

   }

   // Create the socket and bind our local address so that any client can
   // send to us.

   if ((tcp_srv_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
      ErrorSys(kErrFatal, "NetInit: can't create socket");

   if (bind(tcp_srv_sock, (struct sockaddr *) &tcp_srv_addr,
            sizeof(tcp_srv_addr)) < 0)
      ErrorSys(kErrFatal, "NetInit: can't bind local address");

   // And set the listen parameter, telling the system that we're
   // ready to accept incoming connection requests.

   listen(tcp_srv_sock, 5);

   if (gDebug > 0)
      ErrorInfo("NetInit: socket %d listening on port %d", tcp_srv_sock,
                ntohs(tcp_srv_addr.sin_port));
}

//______________________________________________________________________________
int NetOpen(int inetdflag)
{
   // Initialize the server's end.
   // We are passed a flag that says whether or not we are started
   // by a "master daemon" such as inetd. A master daemon will have
   // already waited for a message to arrive for us and will have
   // already set up the connection to the client. If we weren't
   // started by a master daemon, then we must wait for a client's
   // request to arrive.

   if (inetdflag) {

      // When we're fired up by inetd, file decriptors 0, 1 and 2
      // are sockets to the client.

      gSockFd = 0;

      if (gDebug > 0) {
#if defined(USE_SIZE_T)
         size_t clilen = sizeof(tcp_cli_addr);
#elif defined(USE_SOCKLEN_T)
         socklen_t clilen = sizeof(tcp_cli_addr);
#else
         int clilen = sizeof(tcp_cli_addr);
#endif
         if (!getpeername(gSockFd, (struct sockaddr *)&tcp_cli_addr, &clilen)) {
            struct hostent *hp;
            if ((hp = gethostbyaddr((const char *)&tcp_cli_addr.sin_addr,
                                    sizeof(tcp_cli_addr.sin_addr), AF_INET)))
               strcpy(openhost, hp->h_name);
            else {
               struct in_addr *host_addr = (struct in_addr*)&tcp_cli_addr.sin_addr;
               strcpy(openhost, inet_ntoa(*host_addr));
            }
         }
         ErrorInfo("NetOpen: accepted connection from host %s", openhost);

         ErrorInfo("NetOpen: connection established via socket %d", gSockFd);
      }

      return 0;

   }

   // For the concurrent server that's not initiated by inetd,
   // we have to wait for a connection request to arrive, then
   // fork a child to handle the client's request.
   // Beware that the accept() can be interrupted, such as by
   // a previously spawned child process that has terminated
   // (for which we caught the SIGCLD signal).

again:
#if defined(USE_SIZE_T)
   size_t clilen = sizeof(tcp_cli_addr);
#elif defined(USE_SOCKLEN_T)
   socklen_t clilen = sizeof(tcp_cli_addr);
#else
   int clilen = sizeof(tcp_cli_addr);
#endif
   int newsock = accept(tcp_srv_sock, (struct sockaddr *)&tcp_cli_addr, &clilen);
   if (newsock < 0) {
      if (GetErrno() == EINTR) {
         ResetErrno();
         goto again;   // probably a SIGCLD that was caught
      }
      ErrorSys(kErrFatal, "NetOpen: accept error");
   }

   if (gDebug > 0) {
      struct hostent *hp;
      if ((hp = gethostbyaddr((const char *)&tcp_cli_addr.sin_addr,
                              sizeof(tcp_cli_addr.sin_addr), AF_INET)))
         strcpy(openhost, hp->h_name);
      else {
         struct in_addr *host_addr = (struct in_addr*)&tcp_cli_addr.sin_addr;
         strcpy(openhost, inet_ntoa(*host_addr));
      }

      ErrorInfo("NetOpen: accepted connection from host %s", openhost);
   }

   // Fork a child process to handle the client's request.
   // The parent returns the child pid to the caller, which is
   // probably a concurrent server that'll call us again, to wait
   // for the next client request to this well-known port.

   int childpid;
   if ((childpid = fork()) < 0)
      ErrorSys(kErrFatal, "NetOpen: server can't fork");
   else if (childpid > 0) {    // parent
      close(newsock);
      return childpid;
   }

   // Child process continues here.
   // First close the original socket so that the parent
   // can accept any further requests that arrive there.
   // Then set "gSockFd" in our process to be the descriptor
   // that we are going to process.

   close(tcp_srv_sock);

   gSockFd = newsock;

   if (gDebug > 0)
      ErrorInfo("NetOpen: connection established via socket %d", gSockFd);

   return 0;
}

//______________________________________________________________________________
void NetClose()
{
   // Close the network connection.

   close(gSockFd);
   gSockFd = -1;

   if (gDebug > 0)
      ErrorInfo("NetClose: host = %s, fd = %d, file = %s", openhost, gSockFd,
                gFile);
}

//______________________________________________________________________________
void NetSetOptions()
{
   // Set some options for network socket.

   int val = 1;
   if (!setsockopt(gSockFd, IPPROTO_TCP, TCP_NODELAY, (char *)&val, sizeof(val))) {
       if (gDebug > 0) ErrorInfo("NetSetOptions: set TCP_NODELAY");
   }
   if (!setsockopt(gSockFd, SOL_SOCKET,  SO_KEEPALIVE, (char *)&val, sizeof(val))) {
      if (gDebug > 0) ErrorInfo("NetSetOptions: set SO_KEEPALIVE");
      signal(SIGPIPE, SigPipe);   // handle SO_KEEPALIVE failure
   }

   val = 65536;
   if (!setsockopt(gSockFd, SOL_SOCKET,  SO_SNDBUF,    (char *)&val, sizeof(val))) {
      if (gDebug > 0) ErrorInfo("NetSetOptions: set SO_SNDBUF %d", val);
   }
   if (!setsockopt(gSockFd, SOL_SOCKET,  SO_RCVBUF,    (char *)&val, sizeof(val))) {
      if (gDebug > 0) ErrorInfo("NetSetOptions: set SO_RCVBUF %d", val);
   }
}

//______________________________________________________________________________
int NetSendRaw(const void *buf, int len)
{
   // Send buffer of len bytes.

   if (gSockFd == -1) return -1;

   if (Sendn(gSockFd, buf, len) != len) {
      ErrorInfo("NetSendRaw: Sendn error");
      RootdClose();
      exit(1);
   }
   return len;
}

//______________________________________________________________________________
int NetSend(const void *buf, int len, EMessageTypes kind)
{
   // Send buffer of len bytes. Message will be of type "kind".

   int hdr[2];
   int hlen = sizeof(int) + len;
   hdr[0] = htonl(hlen);
   hdr[1] = htonl(kind);
   if (NetSendRaw(hdr, sizeof(hdr)) < 0)
      return -1;

   return NetSendRaw(buf, len);
}

//______________________________________________________________________________
int NetSend(int code, EMessageTypes kind)
{
   // Send integer. Message will be of type "kind".

   int hdr[3];
   int hlen = sizeof(int) + sizeof(int);
   hdr[0] = htonl(hlen);
   hdr[1] = htonl(kind);
   hdr[2] = htonl(code);
   return NetSendRaw(hdr, sizeof(hdr));
}

//______________________________________________________________________________
int NetSend(const char *msg, EMessageTypes kind)
{
   // Send a string. Message will be of type "kind".

   int len = 0;

   if (msg)
      len = strlen(msg)+1;

   return NetSend(msg, len, kind);
}

//______________________________________________________________________________
int NetSendAck()
{
   return NetSend(0, kROOTD_ACK);
}

//______________________________________________________________________________
int NetSendError(ERootdErrors err)
{
   return NetSend(err, kROOTD_ERR);
}

//______________________________________________________________________________
int NetRecvRaw(void *buf, int len)
{
   // Receive a buffer of maximum len bytes.

   if (gSockFd == -1) return -1;

   if (Recvn(gSockFd, buf, len) < 0) {
      ErrorInfo("NetRecvRaw: Recvn error");
      RootdClose();
      exit(1);
   }
   return len;
}

//______________________________________________________________________________
int NetRecv(void *&buf, int &len, EMessageTypes &kind)
{
   // Receive a buffer. Returns the newly allocated buffer, the length
   // of the buffer and message type in kind.

   int hdr[2];

   if (NetRecvRaw(hdr, sizeof(hdr)) < 0)
      return -1;

   len = ntohl(hdr[0]) - sizeof(int);
   kind = (EMessageTypes) ntohl(hdr[1]);
   if (len) {
      buf = new char* [len];
      return NetRecvRaw(buf, len);
   }
   buf = 0;
   return 0;
}

//______________________________________________________________________________
int NetRecv(char *msg, int len, EMessageTypes &kind)
{
   // Receive a string of maximum len length. Returns message type in kind.
   // Return value is msg length.

   int   mlen;
   char *buf;

   if (NetRecv((void *&)buf, mlen, kind) < 0)
      return -1;

   if (mlen == 0) {
      msg[0] = 0;
      return 0;
   } else if (mlen > len) {
      strncpy(msg, buf, len-1);
      msg[len-1] = 0;
      mlen = len;
   } else
      strcpy(msg, buf);

   delete [] buf;

   return mlen - 1;
}
