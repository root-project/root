// @(#)root/proofd:$Name:  $:$Id: XrdProofConn.cxx,v 1.8 2006/04/19 10:52:46 rdm Exp $
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
// XrdProofConn                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Low level handler of connections to xproofd.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdProofConn.h"
#include "XProofProtocol.h"

#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientDebug.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientPhyConnection.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"

// Dynamic libs
// Bypass Solaris ELF madness
#if (defined(SUNCC) || defined(SUN))
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

#ifndef WIN32
#include <dlfcn.h>
#if !defined(__APPLE__)
#include <link.h>
#endif
#endif

// Tracing utils
#include "XrdProofdTrace.h"
extern XrdOucTrace *XrdProofdTrace;
static const char *TraceID = " ";
#define TRACEID TraceID

#ifndef WIN32
#include <sys/socket.h>
#include <sys/types.h>
#include <pwd.h>
#else
#include <process.h>
#include <Winsock2.h>
#endif

// Security handle
typedef XrdSecProtocol *(*XrdSecGetProt_t)(const char *, const struct sockaddr &,
                                           const XrdSecParameters &, XrdOucErrInfo *);

XrdClientConnectionMgr *XrdProofConn::fgConnMgr = 0;

#ifndef SafeDelete
#define SafeDelete(x) { if (x) { delete x; x = 0; } }
#endif
#define URLTAG "["<<fUrl.Host<<":"<<fUrl.Port<<"]"

//_____________________________________________________________________________
XrdProofConn::XrdProofConn(const char *url, char m, int psid, char capver,
                           XrdClientAbsUnsolMsgHandler *uh, const char *logbuf)
   : fMode(m), fConnected(0), fSessionID(psid), fLastErr(kXR_Unsupported),
     fCapVer(capver), fLoginBuffer(logbuf), fPhyConn(0), fUnsolMsgHandler(uh)
{
   // Constructor. Open the connection to a remote XrdProofd instance.
   // The mode 'm' indicates the role of this connection:
   //     'a'      Administrator; used by an XPD to contact the head XPD
   //     'i'      Internal; used by a TXProofServ to call back its creator
   //              (see XrdProofUnixConn)
   //     'M'      Client contacting a top master
   //     'm'      Top master contacting a submaster
   //     's'      Master contacting a slave
   // The buffer 'logbuf' is a null terminated string to be sent over at
   // login. In case of need, internally it is overwritten with a token
   // needed during redirection.

   // Initialization
   if (url && !Init(url)) {
      if (GetServType() != kSTProofd)
         TRACE(REQ, "XrdProofConn: severe error occurred while opening a"
                    " connection" << " to server "<<URLTAG);
      return;
   }
}

//_____________________________________________________________________________
bool XrdProofConn::Init(const char *url)
{
   // Initialization

   // Init connection manager (only once)
   if (!fgConnMgr) {
      if (!(fgConnMgr = new XrdClientConnectionMgr())) {
         TRACE(REQ,"XrdProofConn::Init: error initializing connection manager");
         return 0;
      }
   }

   // Parse Url
   fUrl.TakeUrl(XrdOucString(url));
   fUser = fUrl.User.c_str();
   // Get username from Url
   if (fUser.length() <= 0) {
      // If not specified, use local username
#ifndef WIN32
      struct passwd *pw = getpwuid(getuid());
      fUser = pw ? pw->pw_name : "";
#else
      char  name[256];
      DWORD length = sizeof (name);
      ::GetUserName(name, &length);
      if (strlen(name) > 1)
         fUser = name;
#endif
   }
   fHost = fUrl.Host.c_str();
   fPort = fUrl.Port;

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

         // Set the port used
         fPort = fUrl.Port;

         if (fPhyConn->IsLogged() == kNo) {
            // Now the have the logical Connection ID, that we can use as streamid for
            // communications with the server
            TRACE(REQ,"XrdProofConn::Init: new logical connection ID: "<<logid);

            // Get access to server
            if (!GetAccessToSrv()) {
               fConnected = 0;
               if (GetServType() == kSTProofd)
                  return fConnected;
               if (fLastErr == kXR_NotAuthorized || fLastErr == kXR_InvalidRequest) {
                  // Auth error or iunvalid request: does not make much sense to retry
                  Close("P");
                  XrdOucString msg = fLastErrMsg;
                  msg.erase(msg.rfind(":"));
                  TRACE(REQ,"XrdProofConn::Init: failure: " << msg);
                  return fConnected;
               } else {
                  TRACE(REQ,"XrdProofConn::Init: access to server failed (" << fLastErrMsg << ")");
               }
               continue;
            } else {

               // Manager call in client: no need to create or attach: just notify
               TRACE(REQ,"XrdProofConn::Init: access to server granted.");
               break;
            }
         }

         // Notify
         TRACE(REQ,"XrdProofConn::Init: session create / attached successfully.");
         break;

      }

      // We force a physical disconnection in this special case
      TRACE(REQ,"XrdProofConn::Init: disconnecting.");
      Close("P");

      // And we wait a bit before retrying
      TRACE(REQ,"XrdProofConn::Init: connection attempt failed: sleep " << timeOut << " secs");
      sleep(timeOut);

   } //for connect try

   // We are done
   return fConnected;
}

//_____________________________________________________________________________
XrdProofConn::~XrdProofConn()
{
   // Destructor

   // Disconnect from remote server (the connection manager is
   // responsible of the underlying physical connection, so we do not
   // force its closing)
   Close();
}

//_____________________________________________________________________________
int XrdProofConn::Connect()
{
   // Connect to remote server

   int logid;
   logid = -1;

   // Resolve the DNS information
   char *haddr[10] = {0}, *hname[10] = {0};
   int naddr = XrdNetDNS::getAddrName(fUrl.Host.c_str(), 10, haddr, hname);

   int i = 0;
   for (; i < naddr; i++ ) {
      // Address
      fUrl.HostAddr = (const char *) haddr[i];
      // Name
      fUrl.Host = (const char *) hname[i];
      // Notify
      TRACE(REQ,"XrdProofConn::Connect: found host "<<fUrl.Host<<
                " with addr " << fUrl.HostAddr);
   }

   // Set port: the first time find the default
   static int servdef = -1;
   if (servdef < 0) {
      struct servent *ent = getservbyname("proofd", "tcp");
      servdef = (ent) ? (int)ntohs(ent->s_port) : 1093;
   }
   fUrl.Port = (fUrl.Port <= 0) ? servdef : fUrl.Port;

   // Connect
   if ((logid = fgConnMgr->Connect(fUrl)) < 0) {
      TRACE(REQ,"XrdProofConn::Connect: creating logical connection to " <<URLTAG);
      fLogConnID = logid;
      fConnected = 0;
      return -1;
   }
   TRACE(REQ,"XrdProofConn::Connect: connect to "<<URLTAG<<" returned "<<logid );

   // Set some vars
   fLogConnID = logid;
   fStreamid = fgConnMgr->GetConnection(fLogConnID)->Streamid();
   fPhyConn = fgConnMgr->GetConnection(fLogConnID)->GetPhyConnection();
   fConnected = 1;

   // Handle asynchronous requests
   SetAsync(fUnsolMsgHandler);

   // We are done
   return logid;
}

//_____________________________________________________________________________
void XrdProofConn::Close(const char *opt)
{
   // Close connection. Available options are (case insensitive)
   //   'P'   force closing of the underlying physical connection
   //   'S'   shutdown remote session, is any
   // A session ID can be given using #...# signature, e.g. "#1#".
   // Default is opt = "".

   // Make sure we are connected
   if (!fConnected) {
      TRACE(REQ,"XrdProofConn::Close: not connected: nothing to do");
      return;
   }

   // Parse options
   bool hard = (opt) ? (strchr(opt,'P') || strchr(opt,'p')) : 0;

   // Close connection
   if (fgConnMgr) {
      fgConnMgr->Disconnect(GetLogConnID(), hard);
      if (hard)
         fgConnMgr->GarbageCollect();
   }

   // Flag this action
   fConnected = 0;

   // We are done
   return;
}

//_____________________________________________________________________________
UnsolRespProcResult XrdProofConn::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *,
                                                        XrdClientMessage *)
{
   // We are here if an unsolicited response comes from a logical conn
   // The response comes in the form of an XrdClientMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited
   // responses are asynchronous by nature.

   TRACE(REQ,"XrdProofConn::ProcessUnsolicitedMsg: processing unsolicited response");

   return kUNSOL_KEEP;
}

//_____________________________________________________________________________
void XrdProofConn::SetAsync(XrdClientAbsUnsolMsgHandler *uh)
{
   // Set handler of unsolicited responses

   if (fgConnMgr)
      fgConnMgr->GetConnection(fLogConnID)->UnsolicitedMsgHandler = uh;
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofConn::ReadMsg()
{
   // Pickup message from the queue

   return (fgConnMgr ? fgConnMgr->ReadMsg(fLogConnID) : (XrdClientMessage *)0);
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofConn::SendRecv(XPClientRequest *req, const void *reqData,
                                         void **answData)
{
   // SendRecv sends a command to the server and to get a response.
   // The header of the last response is returned as pointer to a XrdClientMessage.
   // The data, if any, are returned in *answData; if *answData == 0 in input,
   // the buffer is internally allocated and must be freed by the caller.
   // If (*answData != 0) the program assumes that the caller has allocated
   // enough bytes to contain the reply.
   XrdClientMessage *xmsg = 0;

   // We have to unconditionally set the streamid inside the
   // header, because, in case of 'rebouncing here', the Logical Connection
   // ID might have changed, while in the header to write it remained the
   // same as before, not valid anymore
   SetSID(req->header.streamid);

   // Notify what we are going to send
   if (TRACING(TRACE_ALL))
      XPD::smartPrintClientHeader(req);

   // We need the right order
   int reqDataLen = req->header.dlen;
   XPD::clientMarshall(req);
   if (LowWrite(req, reqData, reqDataLen) != kOK) {
      TRACE(REQ, "XrdProofConn::SendRecv: sending request to server "<<URLTAG);
      return xmsg;
   }

   // Check if the client has already allocated the buffer
   bool needalloc = (answData && !(*answData));

   // Read from server the answer
   // Note that the answer can be composed by many reads, in the case that
   // the status field of the responses is kXR_oksofar
   size_t dataRecvSize = 0;
   do {
      //
      // NB: Xmsg contains ALSO the information about the result of
      // the communication at low level.
      kXR_int16 xst = kXR_error;
      if (!(xmsg = ReadMsg()) ||
          xmsg->IsError()) {
         TRACE(REQ, "XrdProofConn::SendRecv: reading msg from connmgr (server "<<URLTAG<<")");
      } else {
         // Dump header, if required
         if (TRACING(TRACE_ALL))
            XPD::smartPrintServerHeader(&(xmsg->fHdr));
         // Get the status
         xst = xmsg->HeaderStatus();
      }

      // We save the result, if the caller wants so. In any case
      // we update the counters
      if ((xst == kXR_ok) || (xst == kXR_oksofar) || (xst == kXR_authmore)) {
         if (answData && xmsg->DataLen() > 0) {
            if (needalloc) {
               *answData = realloc(*answData, dataRecvSize + xmsg->DataLen());
               if (!(*answData)) {
                  // Memory resources exhausted
                  TRACE(REQ, "XrdProofConn::SendRecv: reallocating "<<dataRecvSize<<" bytes");
                  free(*answData);
                  *answData = 0;
                  SafeDelete(xmsg);
                  return xmsg;
               }
            }
            // Now we copy the content of the Xmsg to the buffer where
            // the data are needed
            memcpy(((kXR_char *)(*answData))+dataRecvSize,
                   xmsg->GetData(), xmsg->DataLen());
            //
            // Dump the buffer *answData, if requested
            if (TRACING(TRACE_ALL)) {
               TRACE(REQ, "XrdProofConn::SendRecv: dumping read data ...");
               for (int jj = 0; jj < xmsg->DataLen(); jj++) {
                  printf("0x%.2x ", *(((kXR_char *)xmsg->GetData())+jj));
                  if (!(jj%10)) printf("\n");
               }
            }
         }
         // Update counters
         dataRecvSize += xmsg->DataLen();

      } else if (xst != kXR_error) {
         //
         // Status unknown: protocol error?
         TRACE(REQ, "XrdProofConn::SendRecv: status in reply is unknown ["<<
               XPD::convertRespStatusToChar(xmsg->fHdr.status)<<
               "] (server "<<URLTAG<<") - Abort");
         // We cannot continue
         SafeDelete(xmsg);
         return xmsg;
      }
      // The last message may be empty: not an error
      if (xmsg && (xst == kXR_oksofar) && (xmsg->DataLen() == 0))
         return xmsg;

   } while (xmsg && (xmsg->HeaderStatus() == kXR_oksofar));

   // We might have collected multiple partial response also in a given mem block
   if (xmsg)
      xmsg->fHdr.dlen = dataRecvSize;

   return xmsg;
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofConn::SendReq(XPClientRequest *req, const void *reqData,
                                        void **answData, const char *CmdName)
{
   // SendReq tries to send a single command for a number of times
   XrdClientMessage *answMex = 0;

   int retry = 0;
   bool resp = 0, abortcmd = 0;

   while (!abortcmd && !resp) {

      abortcmd = 0;

      // Send the cmd, dealing automatically with redirections and
      // redirections on error
      TRACE(REQ,"XrdProofConn::SendReq: calling SendRecv");
      answMex = SendRecv(req, reqData, answData);

      // On serious communication error we retry for a number of times,
      // waiting for the server to come back
      retry++;
      if (!answMex || answMex->IsError()) {
         TRACE(REQ,"XrdProofConn::SendReq: communication error detected with "<<URLTAG);
         if (retry > kXR_maxReqRetry) {
            TRACE(REQ,"XrdProofConn::SendReq: max number of retries reached - Abort");
            abortcmd = 1;
         } else
            abortcmd = 0;
      } else {

         // We are here if we got an answer for the command, so
         // the server (original or redirected) is alive
         resp = CheckResp(&(answMex->fHdr), CmdName);

         // If the answer was not (or not totally) positive, we must
         // investigate on the result
         if (!resp)
            abortcmd = CheckErrorStatus(answMex, retry, CmdName);

         if (retry > kXR_maxReqRetry) {
            TRACE(REQ,"XrdProofConn::SendReq: max number of retries reached - Abort");
            abortcmd = 1;
         }
      }
      if (abortcmd)
         // Cleanup if failed
         SafeDelete(answMex);
   }

   // We are done
   return answMex;
}

//_____________________________________________________________________________
bool XrdProofConn::CheckResp(struct ServerResponseHeader *resp, const char *method)
{
   // Checks if the server's response is ours.
   // If the response's status is "OK" returns 1; if the status is "redirect", it
   // means that the max number of redirections has been achieved, so returns 0.

   if (MatchStreamID(resp)) {

      if (resp->status != kXR_ok && resp->status != kXR_authmore &&
          resp->status != kXR_wait) {
         TRACE(REQ,"XrdProofConn::CheckResp: server "<<URLTAG<<
               " did not return OK replying to last request");
         return 0;
      }
      return 1;

   } else {
      TRACE(REQ, method << " return message not belonging to this client"
            " - Protocol error");
      return 0;
   }
}

//_____________________________________________________________________________
bool XrdProofConn::MatchStreamID(struct ServerResponseHeader *ServerResponse)
{
   // Check stream ID matching

   char sid[2];

   memcpy(sid, &fStreamid, sizeof(sid));

   // Matches the streamid contained in the server's response with the ours
   return (memcmp(ServerResponse->streamid, sid, sizeof(sid)) == 0 );
}

//_____________________________________________________________________________
void XrdProofConn::SetSID(kXR_char *sid) {
   // Set our stream id, to match against that one in the server's response.

   memcpy((void *)sid, (const void*)&fStreamid, 2);
}

//_____________________________________________________________________________
XReqErrorType XrdProofConn::LowWrite(XPClientRequest *req, const void* reqData,
                                     int reqDataLen)
{
   // Send request to server
   // (NB: req is marshalled at this point, so we need also the plain reqDataLen)

   // Strong mutual exclusion over the physical channel
   XrdClientPhyConnLocker pcl(fPhyConn);
   int wc = 0;

   //
   // Send header info first
   int len = sizeof(req->header);
   if ((wc = WriteRaw(req, len)) != len) {
      TRACE(REQ,"XrdProofConn::LowWrite: sending header to server "<<URLTAG<<
            " (rc="<<wc<<")");
      return kWRITE;
   }

   //
   // Send data next, if any
   if (reqDataLen > 0) {
      //
      if ((wc = WriteRaw(reqData, reqDataLen)) != reqDataLen) {
         TRACE(REQ,"XrdProofConn::LowWrite: sending data ("<<reqDataLen<<
               " bytes) to server "<<URLTAG<<" (rc="<<wc<<")");
         return kWRITE;
      }
   }

   return kOK;
}

//_____________________________________________________________________________
bool XrdProofConn::CheckErrorStatus(XrdClientMessage *mex, int &Retry,
                                    const char *CmdName)
{
   // Check error status
   TRACE(REQ,"XrdProofConn::CheckErrorStatus: parsing reply from server "<<URLTAG);

   if (mex->HeaderStatus() == kXR_error) {
      //
      // The server declared an error.
      // In this case it's better to exit, unhandled error

      struct ServerResponseBody_Error *body_err;

      body_err = (struct ServerResponseBody_Error *)mex->GetData();

      if (body_err) {
         fLastErr = (XErrorCode)ntohl(body_err->errnum);
         fLastErrMsg = body_err->errmsg;
         // Print out the error information, as received by the server
         TRACE(ALL,"XrdProofConn::CheckErrorStatus: error "<<fLastErr<<": '"<<fLastErrMsg<<"'");
      }
      return 1;
   }

   if (mex->HeaderStatus() == kXR_wait) {
      //
      // We have to wait for a specified number of seconds and then
      // retry the same cmd

      struct ServerResponseBody_Wait *body_wait;

      body_wait = (struct ServerResponseBody_Wait *)mex->GetData();

      if (body_wait) {
         int sleeptime = ntohl(body_wait->seconds);
         if (mex->DataLen() > 4) {
            TRACE(REQ,"XrdProofConn::CheckErrorStatus: wait request ("<<sleeptime<<
                  " secs); message: "<<(const char*)body_wait->infomsg);
         } else {
            TRACE(REQ,"XrdProofConn::CheckErrorStatus: wait request ("<<sleeptime<<" secs)");
         }
         sleep(sleeptime);
      }

      // We don't want kxr_wait to count as an error
      Retry--;
      return 0;
   }

   // We don't understand what the server said. Better investigate on it...
   TRACE(REQ,"XrdProofConn::CheckErrorStatus: after: "<<CmdName<<
         ": server reply not recognized - Protocol error");

   return 1;
}

//_____________________________________________________________________________
bool XrdProofConn::GetAccessToSrv()
{
   // Gets access to the connected server.
   // The login and authorization steps are performed here.

   // Now we are connected and we ask for the kind of the server
   { XrdClientPhyConnLocker pcl(fPhyConn);
   fServerType = DoHandShake();
   }

   switch (fServerType) {

   case kSTXProofd:

      TRACE(REQ,"XrdProofConn::GetAccessToSrv: found server at "<<URLTAG);

      // Now we can start the reader thread in the physical connection, if needed
      fPhyConn->StartReader();
      fPhyConn->SetTTL(DLBD_TTL);// = DLBD_TTL;
      fPhyConn->fServerType = kBase;
      break;

   case kSTProofd:
      TRACE(REQ,"XrdProofConn::GetAccessToSrv: server at "<<URLTAG<<" is a proofd");
      // Close correctly this connection to proofd
      kXR_int32 dum[2];
      dum[0] = (kXR_int32)htonl(0);
      dum[1] = (kXR_int32)htonl(2034);
      WriteRaw(&dum[0], sizeof(dum));
      Close("P");
      return 0;

   case kSTError:
      TRACE(REQ,"XrdProofConn::GetAccessToSrv: handShake failed with server "<<URLTAG);
      Close("P");
      return 0;

   case kSTNone:
      TRACE(REQ,"XrdProofConn::GetAccessToSrv: server at "<<URLTAG<<" is unknown");
      Close("P");
      return 0;
   }

   bool ok = (fPhyConn->IsLogged() == kNo) ? Login() : 1;
   if (!ok) {
      TRACE(REQ,"XrdProofConn::GetAccessToSrv: client could not login at "<<URLTAG);
      return ok;
   }

   // We are done
   return ok;
}

//_____________________________________________________________________________
int XrdProofConn::WriteRaw(const void *buf, int len)
{
   // Low level write call

   if (fgConnMgr)
      return fgConnMgr->WriteRaw(fLogConnID, buf, len);

   // No connection open
   return -1;
}

//_____________________________________________________________________________
int XrdProofConn::ReadRaw(void *buf, int len)
{
   // Low level write call

   if (fgConnMgr)
      return fgConnMgr->ReadRaw(fLogConnID, buf, len);

   // No connection open
   return -1;
}

//_____________________________________________________________________________
XrdProofConn::ESrvType XrdProofConn::DoHandShake()
{
   // Performs initial hand-shake with the server in order to understand which
   // kind of server is there at the other side

   // Nothing to do if already connected
   if (fPhyConn->fServerType == kBase) {

      TRACE(REQ,"XrdProofConn::DoHandShake: already connected to a PROOF server "<<URLTAG);
      return kSTXProofd;
   }

   // Set field in network byte order
   struct ClientInitHandShake initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.third  = (kXR_int32)htonl((int)1);

   // Send to the server the initial hand-shaking message asking for the
   // kind of server
   int len = sizeof(initHS);
   TRACE(REQ,"XrdProofConn::DoHandShake: step 1: sending "<<len<<" bytes to server "<<URLTAG);

   int writeCount = WriteRaw(&initHS, len);
   if (writeCount != len) {
      TRACE(ALL,"XrdProofConn::DoHandShake: sending "<<len<<" bytes to server "<<URLTAG);
      return kSTError;
   }

   // These 8 bytes are need by 'proofd' and discarded by XPD
   kXR_int32 dum[2];
   dum[0] = (kXR_int32)htonl(4);
   dum[1] = (kXR_int32)htonl(2012);
   writeCount = WriteRaw(&dum[0], sizeof(dum));
   if (writeCount != sizeof(dum)) {
      TRACE(ALL,"XrdProofConn::DoHandShake: sending "<<sizeof(dum)<<
                " bytes to server "<<URLTAG);
      return kSTError;
   }

   // Read from server the first 4 bytes
   ServerResponseType type;
   len = sizeof(type);
   TRACE(REQ,"XrdProofConn::DoHandShake: step 2: reading "<<len<<" bytes from server "<<URLTAG);

   // Read returns the return value of TSocket->RecvRaw... that returns the
   // return value of recv (unix low level syscall)
   int readCount = ReadRaw(&type, len); // 4(2+2) bytes
   if (readCount != len) {
      if (readCount == (int)TXSOCK_ERR_TIMEOUT) {
         TRACE(ALL,"XrdProofConn::DoHandShake: -----------------------");
         TRACE(ALL,"XrdProofConn::DoHandShake: TimeOut condition reached reading from remote server.");
         TRACE(ALL,"XrdProofConn::DoHandShake: This may indicate that the server is a 'proofd', version <= 12");
         TRACE(ALL,"XrdProofConn::DoHandShake: Retry commenting the 'Plugin.TSlave' line in system.rootrc or adding");
         TRACE(ALL,"XrdProofConn::DoHandShake: Plugin.TSlave: ^xpd  TSlave Proof \"TSlave(const char *,const char"
               " *,int,const char *, TProof *,ESlaveType,const char *,const char *)\"");
         TRACE(ALL,"XrdProofConn::DoHandShake: to your $HOME/.rootrc .");
         TRACE(ALL,"XrdProofConn::DoHandShake: -----------------------");
      } else {
         TRACE(ALL,"XrdProofConn::DoHandShake: reading "<<len<<" bytes from server "<<URLTAG);
      }
      return kSTError;
   }

   // to host byte order
   type = ntohl(type);

   // Check if the server is the eXtended proofd
   if (type == 0) {

      struct ServerInitHandShake xbody;

      // ok
      len = sizeof(xbody);
      TRACE(REQ,"XrdProofConn::DoHandShake: step 3: reading "<<len<<" bytes from"
            " server "<<URLTAG);

      readCount = ReadRaw(&xbody, len); // 12(4+4+4) bytes
      if (readCount != len) {
         TRACE(ALL,"XrdProofConn::DoHandShake: reading "<<len<<" bytes from server "<<URLTAG);
         return kSTError;
      }

      XPD::ServerInitHandShake2HostFmt(&xbody);

      fRemoteProtocol = xbody.protover;

      return kSTXProofd;

   } else if (type == 8) {
      // Standard proofd
      return kSTProofd;
   } else {
      // We don't know the server type
      TRACE(ALL,"XrdProofConn::DoHandShake: unknown server type ("<<type<<")");
      return kSTNone;
   }
}

//_____________________________________________________________________________
int XrdProofConn::GetLowSocket()
{
   // Return the socket descriptor of the underlying connection

   return (fPhyConn ? fPhyConn->GetSocket() : -1);
}

//_____________________________________________________________________________
bool XrdProofConn::Login()
{
   // This method perform the loggin-in into the server just after the
   // hand-shake. It also calls the Authenticate() method

   XPClientRequest reqhdr;

   // We fill the header struct containing the request for login
   memset( &reqhdr, 0, sizeof(reqhdr));

   reqhdr.login.pid = getpid();

   // Fill login username
   if (fUser.length() >= 0)
      strcpy( (char *)reqhdr.login.username, (char *)(fUser.c_str()) );
   else
      strcpy( (char *)reqhdr.login.username, "????" );

   // This is the place to send a token for fast authentication
   // or id to the server (or any other information)
   const void *buf = (const void *)(fLoginBuffer.c_str());
   reqhdr.header.dlen = fLoginBuffer.length();

   // Set the connection mode (see constructor header)
   reqhdr.login.role[0] = fMode;

   // For normal connections this is the PROOF protocol version run by the client.
   // For internal connections this is the id of the session we want to be
   // connected.
   short int sessID = fSessionID;
   // We use the 2 reserved bytes
   memcpy(&reqhdr.login.reserved[0], &sessID, 2);

   // Send also a capability (protocol) version number
   reqhdr.login.capver[0] = (char)fCapVer;

   // We call SendReq, the function devoted to sending commands.
   TRACE(REQ,"XrdProofConn::Login: logging into server "<<URLTAG<<
         "; pid="<<reqhdr.login.pid<<"; uid=" << reqhdr.login.username);

   // Reset logged state
   fPhyConn->SetLogged(kNo);

   bool notdone = 1;
   bool resp = 1;

   // If positive answer
   XrdSecProtocol *secp = 0;
   while (notdone) {

      // server response header
      char *pltmp = 0;
      SetSID(reqhdr.header.streamid);
      reqhdr.header.requestid = kXP_login;
      XrdClientMessage *xrsp = SendReq(&reqhdr, buf,
                                       (void **)&pltmp, "XrdProofConn::Login");

      // If positive answer
      secp = 0;
      char *plref = pltmp;
      if (xrsp) {
         //
         // Pointer to data
         int len = xrsp->DataLen();
         if (len >= (int)sizeof(kXR_int32)) {
            // The first 4 bytes contain the remote daemon version
            kXR_int32 vers = 0;
            memcpy(&vers, pltmp, sizeof(kXR_int32));
            fRemoteProtocol = ntohl(vers);
            pltmp = (char *)((char *)pltmp + sizeof(kXR_int32));
            len -= sizeof(kXR_int32);
         }
         // Check if we need to authenticate
         if (pltmp && (len > 0)) {
            //
            // Reset the result
            resp = 0;
            //
            // Set some environment variables: debug
            char *s = 0;
            if (EnvGetLong(NAME_DEBUG) > 0) {
               s = new char [strlen("XrdSecDEBUG")+20];
               sprintf(s, "XrdSecDEBUG=%ld", EnvGetLong(NAME_DEBUG));
               putenv(s);
            }
            // user name
            s = new char [strlen("XrdSecUSER")+fUser.length()+2];
            sprintf(s, "XrdSecUSER=%s", fUser.c_str());
            putenv(s);
            // host name
            s = new char [strlen("XrdSecHOST")+fHost.length()+2];
            sprintf(s, "XrdSecHOST=%s", fHost.c_str());
            putenv(s);
            // netrc file
            XrdOucString netrc;
#ifndef WIN32
            struct passwd *pw = getpwuid(getuid());
            if (pw) {
               netrc = pw->pw_dir;
               netrc += "/.rootnetrc";
            }
#endif
            if (netrc.length() > 0) {
               s = new char [strlen("XrdSecNETRC")+netrc.length()+2];
               sprintf(s, "XrdSecNETRC=%s", netrc.c_str());
               putenv(s);
            }
            //
            // Null-terminate server reply
            char *plist = new char[len+1];
            memcpy(plist, pltmp, len);
            plist[len] = 0;
            TRACE(REQ,"XrdProofConn::Login: server requires authentication");

            secp = Authenticate(plist, (int)(len+1));
            resp = (secp != 0);

            if (plist)
               delete[] plist;
         } else {
            // We are successfully done
            resp = 1;
            notdone = 0;
         }
         // Cleanup
         SafeDelete(xrsp);
      } else {
         // We failed but we are done with this attempt
         resp = 0;
         notdone = 0;
      }

      // Cleanup
      if (plref)
         free(plref);

   }

   // Flag success if everything went ok
   if (resp) {
      fPhyConn->SetLogged(kYes);
      fPhyConn->SetSecProtocol(secp);
   }

   // We are done
   return resp;
}

//_____________________________________________________________________________
XrdSecProtocol *XrdProofConn::Authenticate(char *plist, int plsiz)
{
   // Negotiate authentication with the remote server. Tries in turn
   // all available protocols proposed by the server (in plist),
   // starting from the first.

   static XrdSecGetProt_t getp = 0;
   XrdSecProtocol *protocol = (XrdSecProtocol *)0;

   if (!plist || plsiz <= 0)
      return protocol;

   TRACE(REQ,"XrdProofConn::Authenticate: host "<<URLTAG<<
             " sent a list of "<<plsiz<<" bytes");
   //
   // Prepare host/IP information of the remote xrootd. This is required
   // for the authentication.
   struct sockaddr_in netaddr;
   char **hosterrmsg = 0;
   if (XrdNetDNS::getHostAddr((char *)fUrl.HostAddr.c_str(),
                                (struct sockaddr &)netaddr, hosterrmsg) <= 0) {
      TRACE(REQ,"XrdProofConn::Authenticate: getHostAddr: "<< *hosterrmsg);
      return protocol;
   }
   netaddr.sin_port   = fUrl.Port;
   //
   // Variables for negotiation
   XrdSecParameters  *secToken = 0;
   XrdSecCredentials *credentials = 0;

   //
   // Now try in turn the available protocols (first preferred)
   bool resp = FALSE;
   int lp = 0;
   char *pp = strstr(plist+lp,"&P=");
   while (pp && pp <= (plist + plsiz - 3)) {
      //
      // The delimitation id next protocol string or the end ...
      char *pn = pp+3;
      while (pn <= (plist + plsiz - 3)) {
         if ((*pn) == '&')
            if (!strncmp(pn+1,"P=",2)) break;
         pn++;
      }
      pn = (pn > (plist + plsiz - 3)) ? 0 : pn;
      //
      // Token length
      int lpar = (pn) ? ((int)(pn-pp)) : (plsiz - (int)(pp-plist));
      //
      // Prepare the parms object
      char *bpar = (char *)malloc(lpar+1);
      if (bpar)
         memcpy(bpar, pp, lpar);
      bpar[lpar] = 0;
      XrdSecParameters Parms(bpar,lpar+1);

      // We need to load the protocol getter the first time we are here
      if (!getp) {
         // Open the security library
         void *lh = 0;
         if (!(lh = dlopen("libXrdSec.so", RTLD_NOW))) {
            TRACE(REQ,"XrdProofConn::Authenticate: unable to load libXrdSec.so");
            return protocol;
         }

         // Get the client protocol getter
         if (!(getp = (XrdSecGetProt_t) dlsym(lh, "XrdSecGetProtocol"))) {
            TRACE(REQ,"XrdProofConn::Authenticate: unable to load XrdSecGetProtocol()");
            return protocol;
         }
      }
      //
      // Retrieve the security protocol context from the xrootd server
      if (!(protocol = (*getp)((char *)fUrl.Host.c_str(),
                               (const struct sockaddr &)netaddr, Parms, 0))) {
         TRACE(REQ,"XrdProofConn::Authenticate: unable to get protocol object.");
         // Set error, in case of need
         fLastErr = kXR_NotAuthorized;
         fLastErrMsg = "unable to get protocol object.";
         pp = pn;
         continue;
      }

      //
      // Protocol name
      XrdOucString protname = protocol->Entity.prot;
      //
      // Once we have the protocol, get the credentials
      credentials = protocol->getCredentials(&Parms);
      if (!credentials) {
         TRACE(REQ,"XrdProofConn::Authenticate:"
                   " cannot obtain credentials (protocol: "<<protname<<")");
         // Set error, in case of need
         fLastErr = kXR_NotAuthorized;
         fLastErrMsg = "cannot obtain credentials for protocol: ";
         fLastErrMsg += protname;
         pp = pn;
         continue;
      } else {
         TRACE(REQ,"XrdProofConn::Authenticate:"
                   "credentials size: " << credentials->size);
      }
      //
      // We fill the header struct containing the request for login
      XPClientRequest reqhdr;
      memset(reqhdr.auth.reserved, 0, 12);
      memcpy(reqhdr.auth.credtype, protname.c_str(), protname.length());

      int status = kXR_authmore;
      int dlen = 0;
      char *srvans = 0;
      resp = FALSE;
      XrdClientMessage *xrsp = 0;
      while (status == kXR_authmore) {
         //
         // Length of the credentials buffer
         SetSID(reqhdr.header.streamid);
         reqhdr.header.requestid = kXP_auth;
         reqhdr.header.dlen = credentials->size;
         xrsp = SendReq(&reqhdr, credentials->buffer,
                               (void **)&srvans, "XrdProofConn::Authenticate");
         SafeDelete(credentials);
         status = (xrsp) ? xrsp->HeaderStatus() : kXR_error;
         dlen = (xrsp) ? xrsp->DataLen() : 0;
         TRACE(REQ,"XrdProofConn::Authenticate:"
                   "server reply: status: "<<status<<" dlen: "<<dlen);

         if (xrsp && (status == kXR_authmore)) {
            //
            // We are required to send additional information
            // First assign the security token that we have received
            // at the login request
            secToken = new XrdSecParameters(srvans, dlen);
            //
            // then get next part of the credentials
            credentials = protocol->getCredentials(secToken);
            SafeDelete(secToken); // nb: srvans is released here
            srvans = 0;
            if (!credentials) {
               TRACE(REQ,"XrdProofConn::Authenticate: cannot obtain credentials");
               // Set error, in case of need
               fLastErr = kXR_NotAuthorized;
               fLastErrMsg = "cannot obtain credentials";
               break;
            } else {
               TRACE(REQ,"XrdProofConn::Authenticate:"
                         "credentials size " << credentials->size);
            }
         } else if (status == kXR_ok) {
            // Success
            resp = TRUE;
         }

         // Cleanup message
         SafeDelete(xrsp);
      }
      if (!resp)
         // Get next
         pp = pn;
      else
        // We are done
         break;
   }

   // Return the result of the negotiation
   //
   return protocol;
}

//_____________________________________________________________________________
void XrdProofConn::SetInterrupt()
{
   // Interrupt the underlying socket

   if (fPhyConn)
      fPhyConn->SetInterrupt();
}
