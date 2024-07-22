// @(#)root/auth:$Id$
// Author: Gerri Ganis   19/1/2004

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DaemonUtils                                                          //
//                                                                      //
// This file defines wrappers to client utils calls used by server      //
// authentication daemons                                               //
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
#if defined(__MACH__) && !defined(__APPLE__)
#   define R__GLIBC
#endif

#ifdef __sun
#   ifndef _REENTRANT
#      if __SUNPRO_CC > 0x420
#         define GLOBAL_ERRNO
#      endif
#   endif
#endif

#include "Rtypes.h"
#include "Varargs.h"
#include "DaemonUtils.h"
#include "TAuthenticate.h"
#include "TSecContext.h"
#include "TEnv.h"
#include "TROOT.h"

//________________________________________________________________________

// --- Globals --------------------------------------------------------
static TSocket *gSocket;

// This is to be changed whenever something is changed
// in non-backward compatible way
// 0 -> 1: support for SSH authentication via SSH tunnel
static Int_t gSrvProtocol = 1;
static EService gService = kSOCKD;
static Int_t gReuseAllow = 0x1F;

using namespace std;
using namespace ROOT;

extern "C" {
   Int_t SrvAuthenticate(TSocket *socket,
                         const char *confdir, const char *tmpdir,
                         string &user, Int_t &meth, Int_t &type, string &ctkn,
                         TSeqCollection *secctxlist) {
      return SrvAuthImpl(socket, confdir, tmpdir, user, meth, type, ctkn, secctxlist);
   }
}

extern "C" {
   Int_t SrvAuthCleanup(TSeqCollection *sls) {
      return SrvClupImpl(sls);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set relevant environment variables

static Int_t SrvSetVars(string /*confdir*/)
{
   // Executables and conf dirs

   // Make rootbindir available to all the session via env
   string execdir = TROOT::GetBinDir().Data();
   if (execdir.length()) {
      int len = 15 + execdir.length();
      char *tmp = new char[len+1];
      if (tmp) {
         snprintf(tmp,len+1, "ROOTBINDIR=%.*s", len, execdir.c_str());
         putenv(tmp);
      } else
         return -1;
   }

   // Make rootetcdir available to all the session via env
   string etcdir = TROOT::GetEtcDir().Data();
   if (etcdir.length()) {
      int len = 15 + etcdir.length();
      char *tmp = new char[len+1];
      if (tmp) {
         snprintf(tmp, len+1, "ROOTETCDIR=%.*s", len, etcdir.c_str());
         putenv(tmp);
      } else
         return -1;
   }

   // If specified, set the special daemonrc file to be used
   string daemonrc = string(gEnv->GetValue("SrvAuth.DaemonRc",""));
   if (daemonrc.length()) {
      int len = 15 + daemonrc.length();
      char *tmp = new char[len+1];
      if (tmp) {
         snprintf(tmp, len+1, "ROOTDAEMONRC=%.*s", len, daemonrc.c_str());
         putenv(tmp);
      } else
         return -1;
   }

   // If specified, set the special gridmap file to be used
   string gridmap = string(gEnv->GetValue("SrvAuth.GridMap",""));
   if (gridmap.length()) {
      int len = 15 + gridmap.length();
      char *tmp = new char[len+1];
      if (tmp) {
         snprintf(tmp, len+1, "GRIDMAP=%.*s", len, gridmap.c_str());
         putenv(tmp);
      } else
         return -1;
   }

   // If specified, set the special hostcert.conf file to be used
   string hcconf = string(gEnv->GetValue("SrvAuth.HostCert",""));
   if (hcconf.length()) {
      int len = 15 + hcconf.length();
      char *tmp = new char[len+1];
      if (tmp) {
         snprintf(tmp, len+1, "ROOTHOSTCERT=%.*s", len, hcconf.c_str());
         putenv(tmp);
      } else
         return -1;
   }

   return 0;
}


////////////////////////////////////////////////////////////////////////////////

void Err(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   if (level > -1) NetSend(level, kROOTD_ERR);
}

////////////////////////////////////////////////////////////////////////////////

void ErrFatal(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   if (level > -1) NetSend(msg, kMESS_STRING);
}

////////////////////////////////////////////////////////////////////////////////

void ErrSys(int level, const char *msg, int size)
{
   Perror((char *)msg, size);
   ErrFatal(level, msg, size);
}

////////////////////////////////////////////////////////////////////////////////
/// Wrapper to cleanup code

Int_t SrvClupImpl(TSeqCollection *secls)
{
   TIter next(secls);
   TSecContext *nsc ;
   while ((nsc = (TSecContext *)next())) {
      if (!strncmp(nsc->GetID(),"server",6)) {
         int rc = RpdCleanupAuthTab(nsc->GetToken());
         if (gDebug > 0 && rc < 0)
            ErrorInfo("SrvClupImpl: operation unsuccessful (rc: %d, ctkn: %s)",
                      rc, nsc->GetToken());
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Server authentication code.
/// Returns 0 in case authentication failed
///         1 in case of success
/// On success, returns authenticated username in user

Int_t SrvAuthImpl(TSocket *socket, const char *confdir, const char *tmpdir,
                  string &user, Int_t &meth, Int_t &type, string &ctoken,
                  TSeqCollection *secctxlist)
{
   Int_t rc = 0;

   // Check if hosts equivalence is required
   Bool_t hequiv = gEnv->GetValue("SrvAuth.CheckHostsEquivalence",0);

   // Pass file for SRP
   string altSRPpass = string(gEnv->GetValue("SrvAuth.SRPpassfile",""));

   // Port for the SSH daemon
   Int_t sshdport = gEnv->GetValue("SrvAuth.SshdPort",22);

   // Set envs
   if (SrvSetVars(string(confdir)) == -1)
      // Problems setting environment
      return rc;

   // Parent ID
   int parentid = getpid(); // Process identifier

   // default job options
   unsigned int options = kDMN_RQAUTH | kDMN_HOSTEQ;
   if (!hequiv)
      options &= ~kDMN_HOSTEQ;

   // Init error handlers
   RpdSetErrorHandler(Err, ErrSys, ErrFatal);

   // Init daemon code
   RpdInit(gService, parentid, gSrvProtocol, options,
           gReuseAllow, sshdport,
           tmpdir, altSRPpass.c_str());

   // Generate Local RSA keys for the session
   if (RpdGenRSAKeys(0))
      // Problems generating keys
      return rc;

   // Reset check of the available method list
   RpdSetMethInitFlag(0);

   // Trasmit relevant socket details
   SrvSetSocket(socket);

   // Init Session (get protocol, run authentication, ...)
   // type of authentication:
   //    0 (new), 1 (existing), 2 (updated offset)
   int clientprotocol = 0;
   rc = RpdInitSession(gService, user, clientprotocol, meth, type, ctoken);

   TSecContext *seccontext = 0;
   if (rc > 0) {
      string openhost(socket->GetInetAddress().GetHostName());

      if (type == 1) {
         // An existing authentication has been re-used: retrieve
         // the related security context
         TIter next(gROOT->GetListOfSecContexts());
         while ((seccontext = (TSecContext *)next())) {
            if (!(strncmp(seccontext->GetID(),"server",6))) {
               if (seccontext->GetMethod() == meth) {
                  if (!strcmp(openhost.c_str(),seccontext->GetHost())) {
                     if (!strcmp(user.c_str(),seccontext->GetUser()))
                        break;
                  }
               }
            }
         }
      }

      if (!seccontext) {
         // New authentication: Fill a SecContext for cleanup
         // in case of interrupt
         seccontext = new TSecContext(user.c_str(), openhost.c_str(), meth, -1,
                                      "server", ctoken.c_str());
         if (seccontext) {
            // Add to the list
            secctxlist->Add(seccontext);
            // Store SecContext
            socket->SetSecContext(seccontext);
         } else {
            if (gDebug > 0)
               ErrorInfo("SrvAuthImpl: could not create sec context object"
                         ": potential problems in cleaning");
         }
      }
   }


   // Done
   return rc;
}


namespace ROOT {

   static int gSockFd = -1;

////////////////////////////////////////////////////////////////////////////////
/// Fill socket parameters

   void SrvSetSocket(TSocket *Socket)
   {
      gSocket = Socket;
      gSockFd = Socket->GetDescriptor();
   }

////////////////////////////////////////////////////////////////////////////////
/// Receive exactly length bytes into buffer. Returns number of bytes
/// received. Returns -1 in case of error.

   static int Recvn(int sock, void *buffer, int length)
   {
      if (sock < 0) return -1;

      int n, nrecv = 0;
      char *buf = (char *)buffer;

      for (n = 0; n < length; n += nrecv) {
         while ((nrecv = recv(sock, buf+n, length-n, 0)) == -1
                && GetErrno() == EINTR)
            ResetErrno();   // probably a SIGCLD that was caught
         if (nrecv < 0) {
            Error(gErrFatal,-1,
                  "Recvn: error (sock: %d): errno: %d",sock,GetErrno());
            return nrecv;
         } else if (nrecv == 0)
            break;         // EOF
      }

      return n;
   }

////////////////////////////////////////////////////////////////////////////////
/// Empty call, for consistency

   void NetClose()
   {
      return;
   }

////////////////////////////////////////////////////////////////////////////////
/// return open socket descriptor

   int NetGetSockFd()
   {
      return gSockFd;
   }

////////////////////////////////////////////////////////////////////////////////
/// Empty call, for consistency

   int NetParOpen(int port, int size)
   {
      if (port+size)
         return (port+size);
      else
         return 1;
   }

////////////////////////////////////////////////////////////////////////////////
/// Receive a string of maximum length max.

   int NetRecv(char *msg, int max)
   {
      return gSocket->Recv(msg, max);
   }

////////////////////////////////////////////////////////////////////////////////
/// Receive a string of maximum len length. Returns message type in kind.
/// Return value is msg length.

   int NetRecv(char *msg, int len, EMessageTypes &kind)
   {
      Int_t tmpkind;
      Int_t rc = gSocket->Recv(msg, len, tmpkind);
      kind = (EMessageTypes)tmpkind;
      return rc;
   }

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer. Returns the newly allocated buffer, the length
/// of the buffer and message type in kind.

   int NetRecv(void *&buf, int &len, EMessageTypes &kind)
   {
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

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer of maximum len bytes.

   int NetRecvRaw(void *buf, int len)
   {
      return gSocket->RecvRaw(buf,len);
   }

////////////////////////////////////////////////////////////////////////////////
/// Receive a buffer of maximum len bytes from generic socket sock.

   int NetRecvRaw(int sock, void *buf, int len)
   {
      if (sock == -1) return -1;

      if (Recvn(sock, buf, len) < 0) {
         Error(gErrFatal,-1,
               "NetRecvRaw: Recvn error (sock: %d, errno: %d)",sock,GetErrno());
      }

      return len;
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

      return gSocket->SendRaw(hdr, sizeof(hdr));
   }

////////////////////////////////////////////////////////////////////////////////
/// Send a string. Message will be of type "kind".

   int NetSend(const char *msg, EMessageTypes kind)
   {
      return gSocket->Send(msg, kind);
   }

////////////////////////////////////////////////////////////////////////////////
/// Send buffer of len bytes. Message will be of type "kind".

   int NetSend(const void *buf, int len, EMessageTypes kind)
   {
      int hdr[2];
      int hlen = sizeof(int) + len;
      hdr[0] = htonl(hlen);
      hdr[1] = htonl(kind);
      if (gSocket->SendRaw(hdr, sizeof(hdr)) < 0)
         return -1;

      return gSocket->SendRaw(buf, len);
   }

////////////////////////////////////////////////////////////////////////////////
/// Send acknowledge code

   int NetSendAck()
   {
      return NetSend(0, kROOTD_ACK);
   }

////////////////////////////////////////////////////////////////////////////////
/// Send error code

   int NetSendError(ERootdErrors err)
   {
      return NetSend(err, kROOTD_ERR);
   }

////////////////////////////////////////////////////////////////////////////////
/// Send buffer of len bytes.

   int NetSendRaw(const void *buf, int len)
   {
      return gSocket->SendRaw(buf, len);
   }

////////////////////////////////////////////////////////////////////////////////
/// Return name of connected host

   void NetGetRemoteHost(std::string &openhost)
   {
      // Get Host name
      openhost = string(gSocket->GetInetAddress().GetHostName());
   }

////////////////////////////////////////////////////////////////////////////////
/// return errno

   int GetErrno()
   {
#ifdef GLOBAL_ERRNO
      return ::errno;
#else
      return errno;
#endif
   }
////////////////////////////////////////////////////////////////////////////////
/// reset errno

   void ResetErrno()
   {
#ifdef GLOBAL_ERRNO
      ::errno = 0;
#else
      errno = 0;
#endif
   }

////////////////////////////////////////////////////////////////////////////////
/// Return in buf the message belonging to errno.

   void Perror(char *buf, int size)
   {
      int len = strlen(buf);
#if (defined(__sun) && defined (__SVR4)) || defined (__linux) || \
   defined(_AIX) || defined(__MACH__)
      snprintf(buf+len, size, " (%s)", strerror(GetErrno()));
#else
      if (GetErrno() >= 0 && GetErrno() < sys_nerr)
         snprintf(buf+len, size, " (%s)", sys_errlist[GetErrno()]);
#endif
   }

////////////////////////////////////////////////////////////////////////////////
/// Formats a string in a circular formatting buffer and prints the string.
/// Appends a newline.
/// Cut & Paste from Printf in base/src/TString.cxx

   void ErrorInfo(const char *va_(fmt), ...)
   {
      char    buf[1024];
      va_list ap;
      va_start(ap,va_(fmt));
      vsprintf(buf, fmt, ap);
      va_end(ap);
      printf("%s\n", buf);
      fflush(stdout);
   }

////////////////////////////////////////////////////////////////////////////////
/// Write error message and call a handler, if required

   void Error(ErrorHandler_t func,int code,const char *va_(fmt), ...)
   {
      char    buf[1024];
      va_list ap;
      va_start(ap,va_(fmt));
      vsprintf(buf, fmt, ap);
      va_end(ap);
      printf("%s\n", buf);
      fflush(stdout);

      // Actions are defined by the specific error handler (
      // see rootd.cxx and proofd.cxx)
      if (func) (*func)(code,(const char *)buf, sizeof(buf));
   }

} // namespace ROOT
