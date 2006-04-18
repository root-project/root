// @(#)root/proofd:$Name:  $:$Id: XrdProofPhyConn.cxx,v 1.4 2006/03/16 09:08:08 rdm Exp $
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
//  XrdProofConn implementation using a simple phycical connection      //
//  (Unix or Tcp)                                                       //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofPhyConn.h"

#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdSec/XrdSecInterface.hh"

#ifndef WIN32
#include <sys/socket.h>
#include <sys/types.h>
#include <pwd.h>
#else
#include <Winsock2.h>
#endif

// Tracing utils
#include "XrdProofdTrace.h"
extern XrdOucTrace *XrdProofdTrace;
static const char *gTraceID = " ";
#define TRACEID gTraceID

#define URLTAG "["<<fUrl.Host<<":"<<fUrl.Port<<"]"

//_____________________________________________________________________________
XrdProofPhyConn::XrdProofPhyConn(const char *url, int psid, char capver,
                                 XrdClientAbsUnsolMsgHandler *uh, bool tcp)
   : XrdProofConn(0, 'i', psid, capver, uh)
{
   // Constructor. Open a direct connection (Unix or Tcp) to a remote
   // XrdProofd instance. Does not use the connection manager.

   fTcp = tcp;

   // Initialization
   if (url && !Init(url)) {
      TRACE(REQ, "XrdProofPhyConn: severe error occurred while"
                 " opening a connection" << " to server "<<URLTAG);
      return;
   }
}

//_____________________________________________________________________________
bool XrdProofPhyConn::Init(const char *url)
{
   // Initialization

   // Save url
   fUrl.TakeUrl(XrdOucString(url));

   if (!fTcp) {
      // Set some variables
#ifndef WIN32
      struct passwd *pw = getpwuid(getuid());
      fUser = (pw) ? pw->pw_name : "";
#else
      char  name[256];
      DWORD length = sizeof (name);
      ::GetUserName(name, &length);
      fUser = name;
#endif
      fHost = "localhost";
      fPort = -1;

   } else {

      // Parse Url
      fUser = fUrl.User.c_str();
      fHost = fUrl.Host.c_str();
      fPort = fUrl.Port;
      // Check port
      if (fPort <= 0) {
         struct servent *sent = getservbyname("rootd", "tcp");
         if (!sent) {
            TRACE(ALL,"XrdProofPhyConn::Init: service 'rootd' not found by getservbyname" <<
                  ": using default IANA assigned tcp port 1094");
            fPort = 1094;
         } else {
            fPort = (int)ntohs(sent->s_port);
            // Update port in url
            fUrl.Port = fPort;
            TRACE(REQ,"XrdProofPhyConn::Init: getservbyname found tcp port " << fPort <<
                  " for service 'rootd'");
         }
      }
   }

   // Max number of tries and timeout
   int maxTry = EnvGetLong(NAME_FIRSTCONNECTMAXCNT);
   int timeOut = EnvGetLong(NAME_CONNECTTIMEOUT);

   int logid = -1;
   int i = 0;
   for (; (i < maxTry) && (!fConnected); i++) {

      // Try connection
      logid = Connect();

      // We are connected to a host. Let's handshake with it.
      if (fConnected) {

         // Now the have the logical Connection ID, that we can use as streamid for
         // communications with the server
         TRACE(REQ,"XrdProofPhyConn::Init: new logical connection ID: "<<logid);

         // Get access to server
         if (!GetAccessToSrv()) {
            if (fLastErr == kXR_NotAuthorized) {
               // Authentication error: does not make much sense to retry
               Close("P");
               XrdOucString msg = fLastErrMsg;
               msg.erase(msg.rfind(":"));
               TRACE(REQ,"XrdProofPhyConn::Init: authentication failure: " << msg);
               return 0;
            } else {
               TRACE(REQ,"XrdProofPhyConn::Init: access to server failed (" <<
                         fLastErrMsg << ")");
            }
            continue;
         } else {

            // Manager call in client: no need to create or attach: just notify
            TRACE(REQ,"XrdProofPhyConn::Init: access to server granted.");
            break;
         }
      }

      // We force a physical disconnection in this special case
      TRACE(REQ,"XrdProofPhyConn::Init: disconnecting.");
      Close("P");

      // And we wait a bit before retrying
      TRACE(REQ,"XrdProofPhyConn::Init: connection attempt failed: sleep " << timeOut << " secs");
#ifndef WIN32
      sleep(timeOut);
#else
      Sleep(timeOut * 1000);
#endif

   } //for connect try

   // We are done
   return fConnected;
}

//_____________________________________________________________________________
int XrdProofPhyConn::Connect()
{
   // Connect to remote server
   char *ctype[2] = {"UNIX", "TCP"};

   // Create physical connection
   fPhyConn = new XrdClientPhyConnection(this);

   // Connect
   if (!(fPhyConn->Connect(fUrl,~fTcp))) {
      TRACE(REQ,"XrdProofPhyConn::Connect: creating "<<ctype[fTcp]<<
                " connection to "<<URLTAG);
      fLogConnID = -1;
      fConnected = 0;
      return -1;
   }
   TRACE(REQ,"XrdProofPhyConn::Connect: "<<ctype[fTcp]<<"-connected to "<<URLTAG);

   // Set some vars
   fLogConnID = 0;
   fStreamid = 1;
   fConnected = 1;

   // Replies are processed asynchronously
   SetAsync(fUnsolMsgHandler);

   // We are done
   return fLogConnID;
}

//_____________________________________________________________________________
void XrdProofPhyConn::Close(const char *)
{
   // Close the connection.

   // Make sure we are connected
   if (!fConnected) {
      TRACE(REQ,"XrdProofPhyConn::Close: not connected: nothing to do");
      return;
   }

   // Close connection
   if (fPhyConn)
      fPhyConn->Disconnect();

   // Flag this action
   fConnected = 0;

   // We are done
   return;
}

//_____________________________________________________________________________
void XrdProofPhyConn::SetAsync(XrdClientAbsUnsolMsgHandler *uh)
{
   // Set handler of unsolicited responses

   if (fPhyConn)
      fPhyConn->UnsolicitedMsgHandler = uh;
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofPhyConn::ReadMsg()
{
   // Pickup message from the queue

   return (fPhyConn ? fPhyConn->ReadMessage(fStreamid) : (XrdClientMessage *)0);
}

//_____________________________________________________________________________
bool XrdProofPhyConn::GetAccessToSrv()
{
   // Gets access to the connected server.
   // The login and authorization steps are performed here.

   // Now we are connected and we ask for the kind of the server
   { XrdClientPhyConnLocker pcl(fPhyConn);
   fServerType = DoHandShake();
   }

   switch (fServerType) {

   case kSTXProofd:
      TRACE(REQ,"XrdProofPhyConn::GetAccessToSrv: found server at "<<URLTAG);

      // Now we can start the reader thread in the physical connection, if needed
      fPhyConn->StartReader();
      fPhyConn->SetTTL(DLBD_TTL);// = DLBD_TTL;
      fPhyConn->fServerType = kBase;
      break;

   case kSTError:
      TRACE(REQ,"XrdProofPhyConn::GetAccessToSrv: handShake failed with server "<<URLTAG);
      Close();
      return 0;

   case kSTProofd:
   case kSTNone:
   default:
      TRACE(REQ,"XrdProofPhyConn::GetAccessToSrv: server at "<<URLTAG<<
            " is unknown : protocol error");
      Close();
      return 0;
   }

   // Execute a login
   if (fPhyConn->IsLogged() != kNo) {
      TRACE(REQ,"XrdProofPhyConn::GetAccessToSrv: client already logged-in"
            " at "<<URLTAG<<" (!): protocol error!");
      return 0;
   }

   // Login
   return Login();
}


//_____________________________________________________________________________
int XrdProofPhyConn::WriteRaw(const void *buf, int len)
{
   // Low level write call

   if (fPhyConn)
      return fPhyConn->WriteRaw(buf, len);

   // No connection open
   return -1;
}

//_____________________________________________________________________________
int XrdProofPhyConn::ReadRaw(void *buf, int len)
{
   // Low level write call

   if (fPhyConn)
      return fPhyConn->ReadRaw(buf, len);

   // No connection open
   return -1;
}
