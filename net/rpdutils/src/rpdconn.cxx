// @(#)root/rpdutils:$Id$
// Author: Gerardo Ganis, March 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdconn                                                              //
//                                                                      //
// This header file contains the definition of some utility classes     //
// used for process communication between xproofd, rootd, proofexecv.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>

#include "rpdconn.h"
#include "RConfig.h"

#define RPD_MAXLEN 8192

//
// Control meaning of the msghdr structure
//
#if defined(__sun)
#define HAVE_MSGHDR_ACCRIGHT
#endif

//
// To make socklen_t portable use SOCKLEN_t
//
#if defined(__solaris__) && !defined(__linux__)
#   if __GNUC__ >= 3 || __GNUC_MINOR__ >= 90
#      define XR__SUNGCC3
#   endif
#endif
#if defined(__linux__)
#   include <features.h>
#   if __GNU_LIBRARY__ == 6
#      ifndef XR__GLIBC
#         define XR__GLIBC
#      endif
#   endif
#endif
#if defined(__MACH__) && defined(__i386__)
#   define R__GLIBC
#endif
#if defined(_AIX) || (defined(XR__SUNGCC3) && !defined(__arch64__))
#   define SOCKLEN_t size_t
#elif defined(XR__GLIBC) || defined(__FreeBSD__) || \
     (defined(XR__SUNGCC3) && defined(__arch64__)) || defined(__APPLE__) || \
     (defined(__sun) && defined(_SOCKLEN_T))
#   ifndef SOCKLEN_t
#      define SOCKLEN_t socklen_t
#   endif
#elif !defined(SOCKLEN_t)
#   define SOCKLEN_t int
#endif

//
// Class describing a basic connection
//

//__________________________________________________________________________
rpdconn::rpdconn(int r, int w) : rdfd(r), wrfd(w)
{
   // Constructor

   int rc = 0;
   pthread_mutexattr_t attr;
   if (!(rc = pthread_mutexattr_init(&attr))) {
      if (!(rc = pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE))) {
         pthread_mutex_init(&wrmtx, &attr);
         pthread_mutex_init(&rdmtx, &attr);
      }
   }
   pthread_mutexattr_destroy(&attr);
}

//__________________________________________________________________________
int rpdconn::send(int i)
{
   // Send an integer
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock

   rpdmtxhelper mh(&wrmtx);
   if (isvalid(0)) {
      if (mh.isok()) {
         // Send the integer
         i = htonl(i);
         if (write(wrfd, &i, sizeof(i)) !=  sizeof(i))
            return -errno;
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::send(int type, const char *msg)
{
   // Send a typed message
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock

   rpdmtxhelper mh(&wrmtx);
   if (isvalid(0)) {
      if (mh.isok()) {
         // Send the type
         type = htonl(type);
         if (write(wrfd, &type, sizeof(type)) !=  sizeof(type))
            return -errno;
         // Send the message, if any
         int len = (msg) ? strlen(msg) : 0;
         int mlen = len;
         // Send the length
         len = htonl(len);
         if (write(wrfd, &len, sizeof(len)) !=  sizeof(len))
            return -errno;
         if (msg && mlen > 0)
            if (write(wrfd, msg, mlen) !=  mlen)
               return -errno;
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::send(const rpdmsg &msg)
{
   // Send message
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock

   return send(msg.type, msg.buf.c_str());
}

//__________________________________________________________________________
int rpdconn::send(const void *buf, int len)
{
   // Send 'len' bytes at 'buf'
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock
   //         -3    if the operation would block
   //         -4    if connection broken
   //         -errno  if any another failure

   rpdmtxhelper mh(&wrmtx);
   if (isvalid(0)) {
      if (mh.isok()) {
         // Send the buffer
         int n, nsnt = 0;
         const char *b = (const char *)buf;
         for (n = 0; n < len; n += nsnt) {
            errno = 0;
            if ((nsnt = ::send(wrfd, b+n, len-n, 0)) <= 0) {
               if (nsnt == 0) break;
               if (errno != EINTR) {
                  if (errno == EPIPE || errno == ECONNRESET)
                     return -4;
                  else if (errno == EWOULDBLOCK)
                     return -3;
                  else
                     return -errno;
               }
            }
         }
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::recv(int &i)
{
   // Receive an integer
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock

   rpdmtxhelper mh(&rdmtx);
   if (isvalid(1)) {
      if (mh.isok()) {
         // Read the integer
         if (read(rdfd, &i, sizeof(i)) != sizeof(i))
            return -errno;
         i = ntohl(i);
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::recv(int &type, std::string &msg)
{
   // Receive a typed message
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock
   //         -3    if failed to receive the full message (the partial message
   //               is stored in 'msg')

   rpdmtxhelper mh(&rdmtx);
   if (isvalid(1)) {
      if (mh.isok()) {
         // Read message type
         if (read(rdfd, &type, sizeof(type)) != sizeof(type))
            return -errno;
         type = ntohl(type);
         // Read message len
         int len = 0;
         if (read(rdfd, &len, sizeof(len)) != sizeof(len))
            return -errno;
         len = ntohl(len);
         int rc = 0;
         if (len > 0) {
            msg = "";
            // Read message
            char buf[RPD_MAXLEN];
            int nr = -1;
            do {
               int wanted = (len > RPD_MAXLEN-1) ? RPD_MAXLEN-1 : len;
               while ((nr = read(rdfd, buf, wanted)) < 0 &&
                     errno == EINTR)
                  errno = 0;
               if (nr < wanted) {
                  if (nr < 0) rc = -3;
                  break;
               } else {
                  buf[nr] = '\0';
                  msg += buf;
               }
               // Update counters
               len -= nr;
            } while (nr > 0 && len > 0);
            
         }
         // Done
         return rc;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::recv(rpdmsg &msg)
{
   // Recv message
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock

   return recv(msg.type, msg.buf);
}

//__________________________________________________________________________
int rpdconn::recv(void *buf, int len)
{
   // Receive 'len' bytes at 'buf'
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock
   //         -3    if the operation would block
   //         -4    if connection broken
   //         -errno  if any another failure

   rpdmtxhelper mh(&rdmtx);
   if (isvalid(1)) {
      if (mh.isok()) {
         int n, nrcv = 0;
         char *b = (char *)buf;
         for (n = 0; n < len; n += nrcv) {
            errno = 0;
            if ((nrcv = ::recv(rdfd, b+n, len-n, 0)) <= 0) {
               if (nrcv == 0) break;        // EOF
               if (errno != EINTR) {
                  if (errno == EPIPE || errno == ECONNRESET)
                     return -4;
                  else if (errno == EWOULDBLOCK)
                     return -3;
                  else
                     return -errno;
               }
            }
         }
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::pollrd(int to)
{
   // Poll over the read descriptor for to secs;
   // Return:
   //           >0     ready to poll
   //            0     timeout
   //           -1     invalid descriptor
   //          <-1     -errno from 'poll'

   if (isvalid(1)) {

      // Read descriptor
      struct pollfd fds_r;
      fds_r.fd = rdfd;
      fds_r.events = POLLIN;

      // We wait for processes to communicate a session status change
      int pollrc = 0;
      int xto = (to > 0) ? to * 1000 : -1;
      while ((pollrc = poll(&fds_r, 1, xto)) < 0 && (errno == EINTR)) {
         errno = 0;
      }
      // Done
      return (pollrc >= 0) ? pollrc : -errno;
   }
   // Invalid pipe
   return -1;
}

//__________________________________________________________________________
int rpdconn::senddesc(int desc)
{
   // Send a descriptor to a the connected peer
   // Return:
   //          0       if OK
   //         -1       if invalid
   //         -2       if failed to acquire mutex lock
   //        <-2       -errno from sendmsg

   rpdmtxhelper mh(&wrmtx);
   if (isvalid(0)) {
      if (mh.isok()) {
         // Create the structure
         struct msghdr msg;
         memset(&msg, 0, sizeof(msg));
         // Set the descriptor pointers
#ifndef HAVE_MSGHDR_ACCRIGHT
         struct cmsghdr *cmsg;
         int    myfds[1] = {desc};         // Contains the file descriptor to pass
#ifdef R__MACOSX
         char   buf[sizeof(struct cmsghdr) + sizeof (myfds)];
#else
         char   buf[CMSG_SPACE(sizeof myfds)];  // ancillary data buffer
#endif
         int   *fdptr = 0;
         msg.msg_control = buf;
         msg.msg_controllen = sizeof buf;
         cmsg = CMSG_FIRSTHDR(&msg);
         cmsg->cmsg_level = SOL_SOCKET;
         cmsg->cmsg_type = SCM_RIGHTS;
         cmsg->cmsg_len = CMSG_LEN(sizeof(int));
         // Initialize the payload:
         fdptr = (int *) CMSG_DATA(cmsg);
         memcpy(fdptr, myfds, sizeof(int));
         // Sum of the length of all control messages in the buffer
         msg.msg_controllen = cmsg->cmsg_len;
#else
         msg.msg_accrights = (caddr_t) &desc;
         msg.msg_accrightslen = sizeof(desc);
#endif
         struct iovec iov[1];
         memset(iov, 0, sizeof(iov));
         // The recvmsg() call will NOT block unless a non-zero length data
         // buffer is specified 
         char c = '\0';
         iov[0].iov_base = &c;
         iov[0].iov_len  = 1;
         msg.msg_iov     = iov;
         msg.msg_iovlen  = 1;
         // Send it over
         if (sendmsg(wrfd, &msg, 0) < 0)
            return -errno;
         // We can close the descriptor in this process
         ::close(desc);
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdconn::recvdesc(int &desc)
{
   // Receive a descriptor from a the connected peer
   // Return:
   //          0       if OK
   //         -1       if invalid
   //         -2       if failed to acquire mutex lock
   //        <-2       -errno from recvmsg

   rpdmtxhelper mh(&rdmtx);
   if (isvalid(1)) {
      if (mh.isok()) {
         // Create the structures and initialize them
         struct msghdr msg;
         memset(&msg, 0, sizeof(msg));
#ifndef HAVE_MSGHDR_ACCRIGHT
         union {
            struct cmsghdr cm;
#ifdef R__MACOSX
            char     control[sizeof(struct cmsghdr) + sizeof (int)];
#else
            char     control[CMSG_SPACE(sizeof (int))];
#endif
         } control_un;
         struct cmsghdr *cmptr;

         msg.msg_control  = control_un.control;
         msg.msg_controllen = sizeof(control_un.control);
#else
         int tmpdesc = 0;
         msg.msg_accrights = (caddr_t) &tmpdesc;
         msg.msg_accrightslen = sizeof(tmpdesc);
#endif
         struct iovec iov[1];
         memset(iov, 0, sizeof(iov));
         // The recvmsg() call will NOT block unless a non-zero length data
         // buffer is specified 
         char c;
         iov[0].iov_base = &c;
         iov[0].iov_len  = 1;
         msg.msg_iov     = iov;
         msg.msg_iovlen  = 1;
         // Receive it
         if (recvmsg(rdfd, &msg, 0) < 0)
            return -errno;
#ifndef HAVE_MSGHDR_ACCRIGHT
         if ((cmptr = CMSG_FIRSTHDR(&msg)) != 0 &&
              cmptr->cmsg_len == CMSG_LEN(sizeof(int))) {
            if (cmptr->cmsg_level != SOL_SOCKET)
               return -errno;
            if (cmptr->cmsg_type != SCM_RIGHTS)
               return -errno;
            memcpy((void *)&desc, CMSG_DATA(cmptr), sizeof(int));
         } else
            desc = -1;           // descriptor was not passed
#else
         if (msg.msg_accrightslen == sizeof(int))
            desc = tmpdesc;
         else
            desc = -1;          // descriptor was not passed
#endif
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//
// Class describing a UNIX connection
//

//__________________________________________________________________________
rpdtcp::rpdtcp(const char *h, int p) : rpdconn(), host(h), port(p), fd(-1)
{
   // Constructor
 
   struct hostent *hent = 0;
   if (!(hent = gethostbyname(h))) {
      fprintf(stderr, "rpdtcp::rpdtcp: ERROR: failure resolving host address (errno: %d)\n", errno);
      return;
   }
   memset(&addr, 0, sizeof(addr));
    
   // The structure   
   struct sockaddr_in server;
   memset(&server, 0, sizeof(server));
   server.sin_family = hent->h_addrtype;
   memcpy((char *) &server.sin_addr.s_addr, hent->h_addr_list[0], hent->h_length);
   server.sin_port   = htons(port);
     
   // Open socket
   if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      fprintf(stderr, "rpdtcp::rpdtcp: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   // Connect
   errno = 0;
   while (connect(fd, (struct sockaddr*) &server, sizeof(server)) == -1) {
      if (errno == EINTR) {
         errno = 0;
      } else {
         fprintf(stderr, "rpdtcp::rpdtcp: ERROR: failure while connecting to '%s:%d' (errno: %d)\n",
                         h, p, errno);
         ::close(fd);
         return;
      }
   }

   // Set descriptors
   setdescriptors(fd, fd);
   
   // Done
   return;
}

//
// Class describing a server TCP connection
//

//__________________________________________________________________________
rpdtcpsrv::rpdtcpsrv(int p, int backlog) : rpdtcp(p)
{
   // Constructor

   // The structure
   struct sockaddr_in inserver;
   memset(&inserver, 0, sizeof(inserver));
   inserver.sin_family = AF_INET;
   inserver.sin_addr.s_addr = htonl(INADDR_ANY);
   inserver.sin_port = htons(p);

   // Open socket
   if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
      fprintf(stderr, "rpdtcpsrv::rpdtcpsrv: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   if (bind(fd, (struct sockaddr*) &inserver, sizeof(inserver))) {
      fprintf(stderr, "rpdtcpsrv::rpdtcpsrv: ERROR: failure binding socket (errno: %d)\n", errno);
      ::close(fd);
      fd = -1;
      return;
   }

   // Start accepting connections
   if (listen(fd, backlog)) {
      fprintf(stderr, "rpdtcpsrv::rpdtcpsrv: ERROR: failure enabling listening on socket (errno: %d)\n", errno);
      ::close(fd);
      fd = -1;
      return;
   }

   // Set descriptors
   setdescriptors(fd, fd);

   // Done
   return;
}

//______________________________________________________________________________
rpdtcp *rpdtcpsrv::accept(int to, int *err)
{
   // Accept a connection on the server socket.
   // If to > 0, wait max to secs (granularity 1 sec). 
   // Return a rpdtcp object describing the open connection (to be destroyed
   // by the caller).
   // On error return a NULL pointer and the errno in *err, if defined;

   int d = -1;
   // Wait for incoming connections in steps of 1 sec
   int tw = 0, rc = 0;
   while (d < 0 && (to <= 0 || tw < to)) {
      struct pollfd sfd = {fd, POLLIN|POLLRDNORM|POLLRDBAND|POLLPRI|POLLHUP, 0};
      do {
         errno = 0;
         rc = poll(&sfd, 1, 1000);
      } while (rc < 0 && (errno == EAGAIN || errno == EINTR));
      if (rc > 0) {
         errno = 0;
         SOCKLEN_t addrlen = sizeof(addr);
         while ((d = ::accept(fd, &addr, &addrlen)) == -1 && errno == EINTR) {
            errno = 0;
         }
      }
      // Count waited time
      tw++;
   }
     
   // Create the socket
   rpdtcp *c = 0;
   if (d < 0 && err) {
      if (rc == 0) {
         *err = ETIME;  // Timed out
      } else if (errno > 0) {
         *err = (int) errno;
      } else {
         *err = -1;
      }
   } else {
      c = new rpdtcp(d);
         printf("rpdtcpsrv::accept: got descriptor %d\n", d);
   }

   // Done
   return c;
}

//
// Class describing a UNIX connection
//

//__________________________________________________________________________
rpdunix::rpdunix(const char *p) : rpdtcp(0), sockpath(p)
{
   // Constructor

   // Need a valid path
   unsigned int plen = 0;
   if (!p || (p && (plen = strlen(p)) <= 0)) {
      fprintf(stderr, "rpdunix::rpdunix: ERROR: path is undefined\n");
      return;
   }

   struct sockaddr_un unserver;
   memset(&unserver, 0, sizeof(unserver));
   unserver.sun_family = AF_UNIX;

   if (plen > sizeof(unserver.sun_path)-1) {
      fprintf(stderr, "rpdunix::rpdunix: ERROR: socket path %s, longer than max allowed length (%u)\n",
                      p, (unsigned int)sizeof(unserver.sun_path)-1);
      return;
   }
   strcpy(unserver.sun_path, p);

   // Open socket
   if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      fprintf(stderr, "rpdunix::rpdunix: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   // Connect
   errno = 0;
   while (connect(fd, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2) == -1) {
      if (errno == EINTR) {
         errno = 0;
      } else {
         fprintf(stderr, "rpdunix::rpdunix: ERROR: failure while connecting over '%s' (errno: %d)\n",
                         p, errno);
         ::close(fd);
         fd = -1;
         return;
      }
   }

   // Set descriptors
   setdescriptors(fd, fd);
   
   // Done
   return;
}

//
// Class describing a server UNIX connection
//

//__________________________________________________________________________
rpdunixsrv::rpdunixsrv(const char *p, int backlog) : rpdunix()
{
   // Constructor

   // Need a valid path
   unsigned int plen = 0;
   if (!p || (p && (plen = strlen(p)) <= 0)) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: path is undefined\n");
      return;
   }

   // Clean the path, if already existing
   struct stat st;
   if (stat(p, &st) != 0) {
      if (errno != ENOENT) {
         fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: cannot operate on (parts of) path '%s' (errno: %d)\n", p, errno);
         return;
      }
   } else {
      // Remove it
      if (unlink(p)) {
         fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: cannot unlink path '%s'\n", p);
         return;
      }
   }

   // Prepare the structure
   struct sockaddr_un unserver;
   memset(&unserver, 0, sizeof(unserver));
   unserver.sun_family = AF_UNIX;

   if (plen > sizeof(unserver.sun_path)-1) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: socket path %s, longer than max allowed length (%u)\n",
                      p, (unsigned int)sizeof(unserver.sun_path)-1);
      return;
   }
   strcpy(unserver.sun_path, p);

   // Open socket
   if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   // Should be able to reuse this
   int val = 1;
   if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (char*)&val, sizeof(val)) == -1) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: failure setting SO_REUSEADDR (errno: %d)\n", errno);
      return;
   }

   // Bind
   if (bind(fd, (struct sockaddr*) &unserver, strlen(unserver.sun_path)+2)) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: failure binding socket (errno: %d)\n", errno);
      ::close(fd);
      fd = -1;
      return;
   }

   // Start accepting connections
   if (listen(fd, backlog)) {
      fprintf(stderr, "rpdunixsrv::rpdunixsrv: ERROR: failure enabling listening on socket (errno: %d)\n", errno);
      ::close(fd);
      fd = -1;
      return;
   }

   // Set descriptors
   setdescriptors(fd, fd);

   // Save the path
   sockpath = p;
   
   // Done
   return;
}

//______________________________________________________________________________
rpdunix *rpdunixsrv::accept(int to, int *err)
{
   // Accept a connection on the server socket.
   // If to > 0, wait max to secs (granularity 1 sec). 
   // Return a rpdunix object describing the open connection (to be destroyed
   // by the caller).
   // On error return a NULL pointer and the errno in *err, if defined;

   int d = -1;

   // Wait for incoming connections in steps of 1 sec
   int tw = 0, rc = 0;
   while (d < 0 && (to <= 0 || tw < to)) {
      struct pollfd sfd = {fd, POLLIN|POLLRDNORM|POLLRDBAND|POLLPRI|POLLHUP, 0};
      do {
         errno = 0;
         rc = poll(&sfd, 1, 1000);
      } while (rc < 0 && (errno == EAGAIN || errno == EINTR));
      if (rc > 0) {
         SOCKLEN_t addrlen = sizeof(addr);
         errno = 0;
         while ((d = ::accept(fd, &addr, &addrlen)) == -1 && errno == EINTR) {
            errno = 0;
         }
      }
      // Count waited time
      tw++;
   }

   // Create the socket
   rpdunix *c = 0;
   if (d < 0 && err) {
      if (rc == 0) {
         *err = ETIME;  // Timed out
      } else if (errno > 0) {
         *err = (int) errno;
      } else {
         *err = -1;
      }
   } else {
      c = new rpdunix(d);
   }

   // Done
   return c;
}

//
// Class describing a UDP connection
//

//__________________________________________________________________________
rpdudp::rpdudp(const char *h, int p) : rpdtcp(h,p)
{
   // Constructor
 
   struct hostent *hent = 0;
   if (!(hent = gethostbyname(h))) {
      fprintf(stderr, "rpdtcp::rpdtcp: ERROR: failure resolving host address (errno: %d)\n", errno);
      return;
   }
   
   // The structure   
   struct sockaddr_in server;
   memset(&server, 0, sizeof(server));
   server.sin_family = hent->h_addrtype;
   memcpy((char *) &server.sin_addr.s_addr, hent->h_addr_list[0], hent->h_length);
   server.sin_port   = htons(port);
    
   // Open socket
   if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
      fprintf(stderr, "rpdudp::rpdudp: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   // Connect
   errno = 0;
   while (connect(fd, (struct sockaddr*) &server, sizeof(server)) == -1) {
      if (errno == EINTR) {
         errno = 0;
      } else {
         fprintf(stderr, "rpdudp::rpdudp: ERROR: failure while connecting to '%s:%d' (errno: %d)\n",
                         h, p, errno);
         ::close(fd);
         return;
      }
   }

   // Set descriptors
   setdescriptors(fd, fd);
   
   // Done
   return;
}

//__________________________________________________________________________
rpdudpsrv::rpdudpsrv(int p) : rpdudp(p)
{
   // Constructor

   // The structure
   struct sockaddr_in inserver;
   memset(&inserver, 0, sizeof(inserver));
   inserver.sin_family = AF_INET;
   inserver.sin_addr.s_addr = htonl(INADDR_ANY);
   inserver.sin_port = htons(p);

   // Open socket
   if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
      fprintf(stderr, "rpdudpsrv::rpdudpsrv: ERROR: failure getting socket descriptor (errno: %d)\n", errno);
      return;
   }

   if (bind(fd, (struct sockaddr*) &inserver, sizeof(inserver))) {
      fprintf(stderr, "rpdudpsrv::rpdudpsrv: ERROR: failure binding socket (errno: %d)\n", errno);
      ::close(fd);
      fd = -1;
      return;
   }

   // Set descriptors
   setdescriptors(fd, fd);

   // Done
   return;
}

//__________________________________________________________________________
int rpdudp::send(const void *buf, int len)
{
   // Send 'len' bytes at 'buf'
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock
   //         -errno  if any another failure

   rpdmtxhelper mh(&wrmtx);
   if (isvalid(0)) {
      if (mh.isok()) {
         // Send the buffer
         int n, nsnt = 0;
         const char *b = (const char *)buf;
         for (n = 0; n < len; n += nsnt) {
            errno = 0;
            if ((nsnt = ::sendto(wrfd, b+n, len-n, 0, 0, 0)) <= 0) {
               if (nsnt == 0) break;
               return -errno;
            }
         }
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//__________________________________________________________________________
int rpdudp::recv(void *buf, int len)
{
   // Receive 'len' bytes at 'buf'
   // Return:
   //          0    if OK
   //         -1    if invalid descriptor
   //         -2    if failed to acquire mutex lock
   //         -errno  if any another failure

   rpdmtxhelper mh(&rdmtx);
   if (isvalid(1)) {
      if (mh.isok()) {
         int n, nrcv = 0;
         char *b = (char *)buf;
         for (n = 0; n < len; n += nrcv) {
            errno = 0;
            SOCKLEN_t addrlen = sizeof(addr);
            if ((nrcv = recvfrom(rdfd, b+n, len-n, 0, &addr, &addrlen)) <= 0) {
               if (nrcv == 0) break;        // EOF
               return -errno;
            }
         }
         // Done
         return 0;
      }
      // Could acquire the mutex lock
      return -2;
   }
   // Invalid descriptor
   return -1;
}

//
// Class describing a basic message
//

//______________________________________________________________________________
void rpdmsg::w_int(int i)
{
   // Add int 'i' to the internal buffer

   char b[64] = {0};
   snprintf(b, 64, "%d", i);
   if (!buf.empty()) buf += " ";
   buf += b;
   if (cur < 0) cur = 0;
}

//______________________________________________________________________________
void rpdmsg::w_double(double d)
{
   // Add double 'd' to the internal buffer

   char b[128] = {0};
   snprintf(b, 128, "%f", d);
   if (!buf.empty()) buf += " ";
   buf += b;
   if (cur < 0) cur = 0;
}

//______________________________________________________________________________
void rpdmsg::w_string(const std::string &s)
{
   // Add string 's' to the internal buffer

   if (!buf.empty()) buf += " ";
   buf += "'";
   buf += s;
   buf += "'";
   if (cur < 0) cur = 0;
}

//______________________________________________________________________________
void rpdmsg::r_int(int &i)
{
   // Retrieve an int from the internal buffer

   if (cur < 0 || cur > (int) buf.length()) return;

   char *p= ((char *)buf.c_str()) + cur;
   while (*p == ' ') p++;
   sscanf(p, "%d", &i);
   if ((p = (char *) strchr(p+1, ' '))) while (*p == ' ') p++;

   // Update pointer
   if (p) {
      cur = (int) (p - (char *)buf.c_str());
   } else {
      cur = (int) buf.length();
   }
}

//______________________________________________________________________________
void rpdmsg::r_double(double &d)
{
   // Retrieve a double from the internal buffer

   if (cur < 0 || cur > (int) buf.length()) return;

   char *p= ((char *)buf.c_str()) + cur;
   while (*p == ' ') p++;
   float f;
   sscanf(p, "%f", &f); 
   d = (double) f;
   if ((p = (char *) strchr(p+1, ' '))) while (*p == ' ') p++;

   // Update pointer
   if (p) {
      cur = (int) (p - (char *)buf.c_str());
   } else {
      cur = (int) buf.length();
   }
}

//______________________________________________________________________________
void rpdmsg::r_string(std::string &s)
{
   // Retrieve a string from the internal buffer

   if (cur < 0 || cur > (int) buf.length()) return;

   s = "";
   int from = cur;
   char *p = ((char *)buf.c_str()) + cur;
   while (*p == ' ') { from++; p++; }
   char *e = strchr(p, ' ');
   int len = buf.length() - from;
   len = (e) ? (int) (e - p) : buf.length() - from;
   if (len > 0) s.assign(buf, from, len);
   // Remove single quotes, if any
   if (s[0] == '\'') s.erase(0,1);
   if (s.length() > 0 && s[s.length() - 1] == '\'') s.erase(s.length() - 1, std::string::npos);
   
   // Update pointer
   if (e) {
      cur = (int) (e - (char *)buf.c_str()) + 1;
   } else {
      cur = (int) buf.length();
   }
}

