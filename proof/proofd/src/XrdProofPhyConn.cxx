// @(#)root/proofd:$Id$
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofPhyConn                                                      //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
//  XrdProofConn implementation using a simple physical connection      //
// (Unix or Tcp)                                                        //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofPhyConn.h"
#include "XpdSysDNS.h"

#include "XrdVersion.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdSec/XrdSecInterface.hh"

#ifndef WIN32
#ifndef ROOT_XrdFour
#  include <sys/socket.h>
#endif
#include <sys/types.h>
#include <netdb.h>
#include <pwd.h>
#else
#include <Winsock2.h>
#endif

// Tracing utils
#include "XrdProofdTrace.h"

#define URLTAG "["<<fUrl.Host<<":"<<fUrl.Port<<"]"

////////////////////////////////////////////////////////////////////////////////
/// Constructor. Open a direct connection (Unix or Tcp) to a remote
/// XrdProofd instance. Does not use the connection manager.

XrdProofPhyConn::XrdProofPhyConn(const char *url, int psid, char capver,
                                 XrdClientAbsUnsolMsgHandler *uh, bool tcp, int fd)
   : XrdProofConn(0, 'i', psid, capver, uh)
{
   XPDLOC(ALL, "PhyConn")

   fTcp = tcp;

   // Mutex
   fMutex = new XrdSysRecMutex();

   // Initialization
   if (url && !Init(url, fd)) {
      TRACE(XERR, "severe error occurred while"
                  " opening a connection" << " to server "<<URLTAG);
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Initialization

bool XrdProofPhyConn::Init(const char *url, int fd)
{
   XPDLOC(ALL, "PhyConn::Init")

   // Save url
   fUrl.TakeUrl(XrdOucString(url));

   // Get user
   fUser = fUrl.User.c_str();
   if (fUser.length() <= 0) {
      // Use local username, if not specified
#ifndef WIN32
      struct passwd *pw = getpwuid(getuid());
      fUser = pw ? pw->pw_name : "";
#else
      char  lname[256];
      DWORD length = sizeof (lname);
      ::GetUserName(lname, &length);
      fUser = lname;
#endif
   }

   // Host and Port
   if (!fTcp) {
      char *hn = XrdSysDNS::getHostName(((fUrl.Host.length() > 0) ?
                                       fUrl.Host.c_str() : "localhost"));
      fHost = hn;
      free(hn);
      fPort = -1;
      fUrl.Host = "";
      fUrl.User = "";
   } else {

      fHost = fUrl.Host.c_str();
      fPort = fUrl.Port;
      // Check port
      if (fPort <= 0) {
         struct servent *sent = getservbyname("proofd", "tcp");
         if (!sent) {
            TRACE(XERR, "service 'proofd' not found by getservbyname" <<
                        ": using default IANA assigned tcp port 1093");
            fPort = 1093;
         } else {
            fPort = (int)ntohs(sent->s_port);
            // Update port in url
            fUrl.Port = fPort;
            TRACE(XERR, "getservbyname found tcp port " << fPort <<
                        " for service 'proofd'");
         }
      }
   }

   // Run the connection attempts: the result is stored in fConnected
   Connect(fd);

   // We are done
   return fConnected;
}

////////////////////////////////////////////////////////////////////////////////
/// Run the connection attempts: the result is stored in fConnected

void XrdProofPhyConn::Connect(int fd)
{
   XPDLOC(ALL, "PhyConn::Connect")

   int maxTry = -1, timeWait = -1;
   // Max number of tries and timeout; use current settings, if any
   XrdProofConn::GetRetryParam(maxTry, timeWait);
   maxTry = (maxTry > -1) ? maxTry : EnvGetLong(NAME_FIRSTCONNECTMAXCNT);
   timeWait = (timeWait > -1) ? timeWait : EnvGetLong(NAME_CONNECTTIMEOUT);

   int logid = -1;
   int i = 0;
   for (; (i < maxTry) && (!fConnected); i++) {

      // Try connection
      logid = TryConnect(fd);

      // We are connected to a host. Let's handshake with it.
      if (fConnected) {

         // Now the have the logical Connection ID, that we can use as streamid for
         // communications with the server
         TRACE(DBG, "new logical connection ID: "<<logid);

         // Get access to server
         if (!GetAccessToSrv()) {
            if (fLastErr == kXR_NotAuthorized) {
               // Authentication error: does not make much sense to retry
               Close("P");
               XrdOucString msg = fLastErrMsg;
               msg.erase(msg.rfind(":"));
               TRACE(XERR, "authentication failure: " << msg);
               return;
            } else {
               TRACE(XERR, "access to server failed (" << fLastErrMsg << ")");
            }
            continue;
         } else {

            // Manager call in client: no need to create or attach: just notify
            TRACE(DBG, "access to server granted.");
            break;
         }
      }

      // We force a physical disconnection in this special case
      TRACE(DBG, "disconnecting");
      Close("P");

      // And we wait a bit before retrying
      TRACE(DBG, "connection attempt failed: sleep " << timeWait << " secs");
#ifndef WIN32
      sleep(timeWait);
#else
      Sleep(timeWait * 1000);
#endif

   } //for connect try
}

////////////////////////////////////////////////////////////////////////////////
/// Connect to remote server

int XrdProofPhyConn::TryConnect(int fd)
{
   XPDLOC(ALL, "PhyConn::TryConnect")

   const char *ctype[2] = {"UNIX", "TCP"};

   // Create physical connection
   fPhyConn = new XrdClientPhyConnection(this, 0);

   // Connect
   bool isUnix = (fTcp) ? 0 : 1;
#if ROOTXRDVERS <= ROOT_PhyConnNoReuse
   if (fd > 0) {
      TRACE(XERR, "Reusing an existing connection (descriptor "<<fd<<
                  ") not supported by the xroot client version (requires xrootd >= 3.0.3)");
      fLogConnID = -1;
      fConnected = 0;
      return -1;
   }
   if (!(fPhyConn->Connect(fUrl, isUnix))) {
#else
   if (!(fPhyConn->Connect(fUrl, isUnix, fd))) {
#endif
      TRACE(XERR, "creating "<<ctype[fTcp]<<" connection to "<<URLTAG);
      fLogConnID = -1;
      fConnected = 0;
      return -1;
   }
   TRACE(DBG, ctype[fTcp]<<"-connected to "<<URLTAG);

   // Set some vars
   fLogConnID = 0;
   fStreamid = 1;
   fConnected = 1;

   // Replies are processed asynchronously
   SetAsync(fUnsolMsgHandler);

   // We are done
   return fLogConnID;
}

////////////////////////////////////////////////////////////////////////////////
/// Close the connection.

void XrdProofPhyConn::Close(const char *)
{
   // Make sure we are connected
   if (!fConnected)
      return;

   // Close connection
   if (fPhyConn)
      fPhyConn->Disconnect();

   // Flag this action
   fConnected = 0;

   // We are done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Set handler of unsolicited responses

void XrdProofPhyConn::SetAsync(XrdClientAbsUnsolMsgHandler *uh,
                               XrdProofConnSender_t, void *)
{
   if (fPhyConn)
      fPhyConn->UnsolicitedMsgHandler = uh;
}

////////////////////////////////////////////////////////////////////////////////
/// Pickup message from the queue

XrdClientMessage *XrdProofPhyConn::ReadMsg()
{
   return (fPhyConn ? fPhyConn->ReadMessage(fStreamid) : (XrdClientMessage *)0);
}

////////////////////////////////////////////////////////////////////////////////
/// Gets access to the connected server.
/// The login and authorization steps are performed here.

bool XrdProofPhyConn::GetAccessToSrv(XrdClientPhyConnection *)
{
   XPDLOC(ALL, "PhyConn::GetAccessToSrv")

   // Now we are connected and we ask for the kind of the server
   { XrdClientPhyConnLocker pcl(fPhyConn);
      fServerType = DoHandShake();
   }

   switch (fServerType) {

   case kSTXProofd:
      TRACE(DBG, "found server at "<<URLTAG);

      // Now we can start the reader thread in the physical connection, if needed
      fPhyConn->StartReader();
      fPhyConn->fServerType = kSTBaseXrootd;
      break;

   case kSTError:
      TRACE(XERR, "handshake failed with server "<<URLTAG);
      Close();
      return 0;

   case kSTProofd:
   case kSTNone:
   default:
      TRACE(XERR, "server at "<<URLTAG<< " is unknown : protocol error");
      Close();
      return 0;
   }

   // Execute a login
   if (fPhyConn->IsLogged() != kNo) {
      TRACE(XERR, "client already logged-in at "<<URLTAG<<" (!): protocol error!");
      return 0;
   }

   // Login
   return Login();
}


////////////////////////////////////////////////////////////////////////////////
/// Low level write call

int XrdProofPhyConn::WriteRaw(const void *buf, int len, XrdClientPhyConnection *)
{
   if (fPhyConn)
      return fPhyConn->WriteRaw(buf, len);

   // No connection open
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Low level write call

int XrdProofPhyConn::ReadRaw(void *buf, int len, XrdClientPhyConnection *)
{
   if (fPhyConn)
      return fPhyConn->ReadRaw(buf, len);

   // No connection open
   return -1;
}
