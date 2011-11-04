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

// XrdProofConn                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Low level handler of connections to xproofd.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include "XrdProofdXrdVers.h"

#include "XpdSysDNS.h"
#include "XpdSysError.h"
#include "XpdSysPlugin.h"
#include "XpdSysPthread.h"

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
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSys/XrdSysPlatform.hh"

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

// Retry controllers
int XrdProofConn::fgMaxTry = 5;
int XrdProofConn::fgTimeWait = 2;  // seconds

XrdSysPlugin *XrdProofConn::fgSecPlugin = 0;       // Sec library plugin
void         *XrdProofConn::fgSecGetProtocol = 0;  // Sec protocol getter

#ifndef SafeDelete
#define SafeDelete(x) { if (x) { delete x; x = 0; } }
#endif
#define URLTAG "["<<fUrl.Host<<":"<<fUrl.Port<<"]"

//_____________________________________________________________________________
XrdProofConn::XrdProofConn(const char *url, char m, int psid, char capver,
                           XrdClientAbsUnsolMsgHandler *uh, const char *logbuf)
   : fMode(m), fConnected(0), fLogConnID(-1), fStreamid(0), fRemoteProtocol(-1),
     fServerProto(-1), fServerType(kSTNone), fSessionID(psid),
     fLastErr(kXR_Unsupported), fCapVer(capver), fLoginBuffer(logbuf), fMutex(0),
     fConnectInterruptMtx(0), fConnectInterrupt(0),
     fPhyConn(0), fUnsolMsgHandler(uh), fSender(0), fSenderArg(0)
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
   XPDLOC(ALL, "XrdProofConn")

   // Mutex
   fMutex = new XrdSysRecMutex();
   fConnectInterruptMtx = new XrdSysRecMutex();

   // Initialization
   if (url && !Init(url)) {
      if (GetServType() != kSTProofd && !(fLastErr == kXR_NotAuthorized))
         TRACE(XERR, "XrdProofConn: severe error occurred while opening a"
                     " connection" << " to server "<<URLTAG);
   }

   return;
}

//_____________________________________________________________________________
void XrdProofConn::GetRetryParam(int &maxtry, int &timewait)
{
   // Retrieve current values of the retry control parameters, numer of retries
   // and wait time between attempts (in seconds).

   maxtry = fgMaxTry;
   timewait = fgTimeWait;
}

//_____________________________________________________________________________
void XrdProofConn::SetRetryParam(int maxtry, int timewait)
{
   // Change values of the retry control parameters, numer of retries
   // and wait time between attempts (in seconds).

   fgMaxTry = maxtry;
   fgTimeWait = timewait;
}

//_____________________________________________________________________________
bool XrdProofConn::Init(const char *url, int)
{
   // Initialization
   XPDLOC(ALL, "Conn::Init")

   // Init connection manager (only once)
   if (!fgConnMgr) {
      if (!(fgConnMgr = new XrdClientConnectionMgr())) {
         TRACE(XERR,"error initializing connection manager");
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

   // Run the connection attempts: the result is stored in fConnected
   Connect();

   // We are done
   return fConnected;
}

//_____________________________________________________________________________
void XrdProofConn::Connect(int)
{
   // Run the connection attempts: the result is stored in fConnected
   XPDLOC(ALL, "Conn::Connect")

   // Max number of tries and timeout
   int maxTry = (fgMaxTry > -1) ? fgMaxTry : EnvGetLong(NAME_FIRSTCONNECTMAXCNT);
   int timeWait = (fgTimeWait > -1) ? fgTimeWait : EnvGetLong(NAME_CONNECTTIMEOUT);

   fConnected = 0;
   int logid = -1;
   int i = 0;
   for (; (i < maxTry) && (!fConnected); i++) {

      // Try connection
      logid = TryConnect();

      // Check if interrupted
      if (ConnectInterrupt()) {
         TRACE(ALL, "got an interrupt while connecting - aborting attempts");
         break;
      }

      // We are connected to a host. Let's handshake with it.
      if (fConnected) {

         // Set the port used
         fPort = fUrl.Port;

         if (fPhyConn->IsLogged() == kNo) {
            // Now the have the logical Connection ID, that we can use as streamid for
            // communications with the server
            TRACE(DBG, "new logical connection ID: "<<logid);

            // Get access to server
            if (!GetAccessToSrv()) {
               if (GetServType() == kSTProofd) {
                  fConnected = 0;
                  return;
               }
               if (fLastErr == kXR_NotAuthorized || fLastErr == kXR_InvalidRequest) {
                  // Auth error or invalid request: does not make much sense to retry
                  Close("P");
                  if (fLastErr == kXR_InvalidRequest) {
                     XrdOucString msg = fLastErrMsg;
                     msg.erase(msg.rfind(":"));
                     TRACE(XERR, "failure: " << msg);
                  }
                  return;
               } else {
                  TRACE(XERR, "access to server failed (" << fLastErrMsg << ")");
               }
               fConnected = 0;
               continue;
            }
         }

         // Notify
         TRACE(DBG, "connection successfully created");
         break;

      }

      // Reset
      TRACE(REQ, "disconnecting");
      Close();

      // And we wait a bit before retrying
      if (i < maxTry - 1) {
         TRACE(DBG, "connection attempt failed: sleep " << timeWait << " secs");
         if (fUrl.Host == "lite" || fUrl.Host == "pod") {
            const char *cdef = (fUrl.Host == "lite") ? " (or \"\": check 'Proof.LocalDefault')" : "";
            const char *cnow = (fUrl.Host == "lite") ? "now " : "";
            const char *cses = (fUrl.Host == "lite") ? "PROOF-Lite" : "PoD";
            TRACE(ALL, "connection attempt to server \""<<fUrl.Host<<"\" failed. We are going to retry after some sleep,");
            TRACE(ALL, "but if you intended to start a "<<cses<<" session instead, please note that you must");
            TRACE(ALL, cnow<<"use \""<<fUrl.Host<<"://\" as connection string"<<cdef);
         }
         sleep(timeWait);
      }

   } //for connect try

   // Notify failure
   if (!fConnected) {
      TRACE(XERR, "failed to connect to " << fUrl.GetUrl());
   }
}

//_____________________________________________________________________________
XrdProofConn::~XrdProofConn()
{
   // Destructor

   // Disconnect from remote server (the connection manager is
   // responsible of the underlying physical connection, so we do not
   // force its closing)
   if (fRemoteProtocol > 1004) {
      // We may be into a reconnection attempt: interrupt it ...
      SetConnectInterrupt();
      // ... and wait for the OK
      XrdClientPhyConnLocker pcl(fPhyConn);
      // Can close now
      Close();
   } else {
      Close();
   }

   // Cleanup mutex
   SafeDelete(fMutex);
   SafeDelete(fConnectInterruptMtx);
}

//_____________________________________________________________________________
void XrdProofConn::ReConnect()
{
   // Perform a reconnection attempt when a connection is not valid any more
   XPDLOC(ALL, "Conn::ReConnect")

   if (!IsValid()) {
      if (fRemoteProtocol > 1004) {

         // Block any other attempt to use this connection
         XrdClientPhyConnLocker pcl(fPhyConn);

         Close();
         int maxtry, timewait;
         XrdProofConn::GetRetryParam(maxtry, timewait);
         XrdProofConn::SetRetryParam(300, 1);
         Connect();
         XrdProofConn::SetRetryParam();

      } else {
         TRACE(DBG, "server does not support reconnections (protocol: %d" <<
                    fRemoteProtocol << " < 1005)");
      }
   }
}

//_____________________________________________________________________________
int XrdProofConn::TryConnect(int)
{
   // Connect to remote server
   XPDLOC(ALL, "Conn::TryConnect")

   int logid;
   logid = -1;

   // Resolve the DNS information
   char *haddr[10] = {0}, *hname[10] = {0};
   int naddr = XrdSysDNS::getAddrName(fUrl.Host.c_str(), 10, haddr, hname);

   int i = 0;
   for (; i < naddr; i++ ) {
      // Address
      fUrl.HostAddr = (const char *) haddr[i];
      // Name
      fUrl.Host = (const char *) hname[i];
      // Notify
      TRACE(HDBG, "found host "<<fUrl.Host<<" with addr " << fUrl.HostAddr);
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
      TRACE(DBG, "failure creating logical connection to " <<URLTAG);
      fLogConnID = logid;
      fConnected = 0;
      return -1;
   }

   // Set some vars
   fLogConnID = logid;
   fStreamid = fgConnMgr->GetConnection(fLogConnID)->Streamid();
   fPhyConn = fgConnMgr->GetConnection(fLogConnID)->GetPhyConnection();
   fConnected = 1;

   TRACE(DBG, "connect to "<<URLTAG<<" returned {"<<fLogConnID<<", "<< fStreamid<<"}");

   // Fill in the remote protocol: either it was received during handshake
   // or it was saved in the underlying physical connection
   if (fRemoteProtocol < 0)
      fRemoteProtocol = fPhyConn->fServerProto;

   // Handle asynchronous requests
   SetAsync(fUnsolMsgHandler);

   // We are done
   return logid;
}

//_____________________________________________________________________________
void XrdProofConn::Close(const char *opt)
{
   // Close connection.
   XPDLOC(ALL, "Conn::Close")

   // Make sure we are connected
   if (!fConnected)
      return;

   // Close also theunderlying physical connection ?
   bool closephys = (opt[0] == 'P') ? 1 : 0;
   TRACE(DBG, URLTAG <<": closing also physical connection ? "<< closephys);

   // Close connection
   if (fgConnMgr)
      fgConnMgr->Disconnect(GetLogConnID(), closephys);

   // Flag this action
   fConnected = 0;

   // We are done
   return;
}

//_____________________________________________________________________________
UnsolRespProcResult XrdProofConn::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *,
                                                        XrdClientMessage *m)
{
   // We are here if an unsolicited response comes from a logical conn
   // The response comes in the form of an XrdClientMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited
   // responses are asynchronous by nature.
   XPDLOC(ALL, "Conn::ProcessUnsolicitedMsg")

   TRACE(DBG,"processing unsolicited response");

   if (!m || m->IsError()) {
      TRACE(XERR, "Got empty or error unsolicited message");
   } else {
      // Check length
      int len = 0;
      if ((len = m->DataLen()) < (int)sizeof(kXR_int32)) {
         TRACE(XERR, "empty or bad-formed message - ignoring");
         return kUNSOL_KEEP;
      }
      // The first 4 bytes contain the action code
      kXR_int32 acod = 0;
      memcpy(&acod, m->GetData(), sizeof(kXR_int32));
      //
      // Update pointer to data
      void *pdata = (void *)((char *)(m->GetData()) + sizeof(kXR_int32));
      //
      // Only interested in service messages
      if (acod == kXPD_srvmsg) {
         // The next 4 bytes may contain a flag to control the way the message is displayed
         kXR_int32 opt = 0;
         memcpy(&opt, pdata, sizeof(kXR_int32));
         opt = ntohl(opt);
         if (opt == 0 || opt == 1 || opt == 2) {
            // Update pointer to data
            pdata = (void *)((char *)pdata + sizeof(kXR_int32));
            len -= sizeof(kXR_int32);
         } else {
            opt = 1;
         }
         // Send up, if required
         if (fSender) {
            (*fSender)((const char *)pdata, len, fSenderArg);
         }
      }
   }

   return kUNSOL_KEEP;
}

//_____________________________________________________________________________
void XrdProofConn::SetAsync(XrdClientAbsUnsolMsgHandler *uh,
                            XrdProofConnSender_t sender, void *arg)
{
   // Set handler of unsolicited responses

   if (fgConnMgr && (fLogConnID > -1)  && fgConnMgr->GetConnection(fLogConnID))
      fgConnMgr->GetConnection(fLogConnID)->UnsolicitedMsgHandler = uh;

   // Set also the sender method and its argument, if required
   fSender = sender;
   fSenderArg = arg;
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofConn::ReadMsg()
{
   // Pickup message from the queue

   return (fgConnMgr ? fgConnMgr->ReadMsg(fLogConnID) : (XrdClientMessage *)0);
}

//_____________________________________________________________________________
XrdClientMessage *XrdProofConn::SendRecv(XPClientRequest *req, const void *reqData,
                                         char **answData)
{
   // SendRecv sends a command to the server and to get a response.
   // The header of the last response is returned as pointer to a XrdClientMessage.
   // The data, if any, are returned in *answData; if *answData == 0 in input,
   // the buffer is internally allocated and must be freed by the caller.
   // If (*answData != 0) the program assumes that the caller has allocated
   // enough bytes to contain the reply.
   XPDLOC(ALL, "Conn::SendRecv")

   XrdClientMessage *xmsg = 0;

   // We have to unconditionally set the streamid inside the
   // header, because, in case of 'rebouncing here', the Logical Connection
   // ID might have changed, while in the header to write it remained the
   // same as before, not valid anymore
   SetSID(req->header.streamid);

   // Notify what we are going to send
   if (TRACING(HDBG))
      XPD::smartPrintClientHeader(req);

   // We need the right order
   int reqDataLen = req->header.dlen;
   if (XPD::clientMarshall(req) != 0) {
      TRACE(XERR, "problems marshalling "<<URLTAG);
      return xmsg;
   }
   if (LowWrite(req, reqData, reqDataLen) != kOK) {
      TRACE(XERR, "problems sending request to server "<<URLTAG);
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
      if (!(xmsg = ReadMsg()) || xmsg->IsError()) {
         TRACE(XERR, "reading msg from connmgr (server "<<URLTAG<<")");
      } else {
         // Dump header, if required
         if (TRACING(HDBG))
            XPD::smartPrintServerHeader(&(xmsg->fHdr));
         // Get the status
         xst = xmsg->HeaderStatus();
      }

      // We save the result, if the caller wants so. In any case
      // we update the counters
      if ((xst == kXR_ok) || (xst == kXR_oksofar) || (xst == kXR_authmore)) {
         if (answData && xmsg->DataLen() > 0) {
            if (needalloc) {
               *answData = (char *) realloc(*answData, dataRecvSize + xmsg->DataLen());
               if (!(*answData)) {
                  // Memory resources exhausted
                  TRACE(XERR, "reallocating "<<dataRecvSize<<" bytes");
                  free((void *) *answData);
                  *answData = 0;
                  SafeDelete(xmsg);
                  return xmsg;
               }
            }
            // Now we copy the content of the Xmsg to the buffer where
            // the data are needed
            memcpy((*answData)+dataRecvSize,
                   xmsg->GetData(), xmsg->DataLen());
            //
            // Dump the buffer *answData, if requested
            if (TRACING(HDBG)) {
               TRACE(DBG, "dumping read data ...");
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
         TRACE(XERR, "status in reply is unknown ["<<
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
                                        char **answData, const char *CmdName,
                                        bool notifyerr)
{
   // SendReq tries to send a single command for a number of times
   XPDLOC(ALL, "Conn::SendReq")

   XrdClientMessage *answMex = 0;

   TRACE(DBG,"len: "<<req->sendrcv.dlen);

   int retry = 0;
   bool resp = 0, abortcmd = 0;
   int maxTry = (fgMaxTry > -1) ? fgMaxTry : kXR_maxReqRetry;

   // We need the unmarshalled request for retries
   XPClientRequest reqsave;
   memcpy(&reqsave, req, sizeof(XPClientRequest));

   while (!abortcmd && !resp) {

      TRACE(HDBG, this << " locking phyconn: "<<fPhyConn);

      // Ok, now we can try
      abortcmd = 0;

      // Make sure we have the unmarshalled request
      memcpy(req, &reqsave, sizeof(XPClientRequest));

      // Send the cmd, dealing automatically with redirections and
      // redirections on error
      TRACE(DBG,"calling SendRecv");
      answMex = SendRecv(req, reqData, answData);

      // On serious communication error we retry for a number of times,
      // waiting for the server to come back
      retry++;
      if (!answMex || answMex->IsError()) {

         TRACE(DBG, "communication error detected with "<<URLTAG);
         if (retry > maxTry) {
            TRACE(XERR,"max number of retries reached - Abort");
            abortcmd = 1;
         } else {
            if (!IsValid()) {
               // Connection is gone: try to reconnect and if this fails, give up
               ReConnect();
               if (!IsValid()) {
                  TRACE(XERR,"not connected: nothing to do");
                  break;
               }
            }
            abortcmd = 0;
            // Restore the unmarshalled request
            memcpy(req, &reqsave, sizeof(XPClientRequest));
         }
      } else {

         // We are here if we got an answer for the command, so
         // the server (original or redirected) is alive
         resp = CheckResp(&(answMex->fHdr), CmdName, notifyerr);

         // If the answer was not (or not totally) positive, we must
         // investigate on the result
         if (!resp)
            abortcmd = CheckErrorStatus(answMex, retry, CmdName, notifyerr);

         if (retry > maxTry) {
            TRACE(XERR,"max number of retries reached - Abort");
            abortcmd = 1;
         }
      }
      if (abortcmd) {
         // Cleanup if failed
         SafeDelete(answMex);
      } else if (!resp) {
         // Sleep a while before retrying
         int sleeptime = 1;
         TRACE(DBG,"sleep "<<sleeptime<<" secs ...");
         sleep(sleeptime);
      }
   }

   // We are done
   return answMex;
}

//_____________________________________________________________________________
bool XrdProofConn::CheckResp(struct ServerResponseHeader *resp,
                             const char *method, bool notifyerr)
{
   // Checks if the server's response is ours.
   // If the response's status is "OK" returns 1; if the status is "redirect", it
   // means that the max number of redirections has been achieved, so returns 0.
   XPDLOC(ALL, "Conn::CheckResp")

   if (MatchStreamID(resp)) {

      if (resp->status != kXR_ok && resp->status != kXR_authmore &&
          resp->status != kXR_wait) {
         if (notifyerr) {
            TRACE(XERR,"server "<<URLTAG<<
                       " did not return OK replying to last request");
         }
         return 0;
      }
      return 1;

   } else {
      if (notifyerr) {
         TRACE(XERR, method << " return message not belonging to this client"
                               " - protocol error");
      }
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
   XPDLOC(ALL, "Conn::LowWrite")

   // Strong mutual exclusion over the physical channel
   XrdClientPhyConnLocker pcl(fPhyConn);
   int wc = 0;

   //
   // Send header info first
   int len = sizeof(req->header);
   if ((wc = WriteRaw(req, len)) != len) {
      TRACE(XERR, "sending header to server "<<URLTAG<<" (rc="<<wc<<")");
      return kWRITE;
   }

   //
   // Send data next, if any
   if (reqDataLen > 0) {
      //
      if ((wc = WriteRaw(reqData, reqDataLen)) != reqDataLen) {
         TRACE(XERR, "sending data ("<<reqDataLen<<" bytes) to server "<<URLTAG<<
                    " (rc="<<wc<<")");
         return kWRITE;
      }
   }

   return kOK;
}

//_____________________________________________________________________________
bool XrdProofConn::CheckErrorStatus(XrdClientMessage *mex, int &Retry,
                                    const char *CmdName, bool notifyerr)
{
   // Check error status
   XPDLOC(ALL, "Conn::CheckErrorStatus")

   TRACE(DBG, "parsing reply from server "<<URLTAG);

   if (mex->HeaderStatus() == kXR_error) {
      //
      // The server declared an error.
      // In this case it's better to exit, unhandled error

      struct ServerResponseBody_Error *body_err;

      body_err = (struct ServerResponseBody_Error *)mex->GetData();

      if (body_err) {
         fLastErr = (XErrorCode)ntohl(body_err->errnum);
         fLastErrMsg = body_err->errmsg;
         if (notifyerr) {
            // Print out the error information, as received by the server
            if (fLastErr == (XErrorCode)kXP_reconnecting) {
               TRACE(XERR, fLastErrMsg);
            } else {
               TRACE(XERR,"error "<<fLastErr<<": '"<<fLastErrMsg<<"'");
            }
         }
      }
      if (fLastErr == (XErrorCode)kXP_reconnecting)
         return 0;
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
            TRACE(DBG,"wait request ("<<sleeptime<<
                  " secs); message: "<<(const char*)body_wait->infomsg);
         } else {
            TRACE(DBG,"wait request ("<<sleeptime<<" secs)");
         }
         sleep(sleeptime);
      }

      // We don't want kxr_wait to count as an error
      Retry--;
      return 0;
   }

   // We don't understand what the server said. Better investigate on it...
   TRACE(XERR,"after: "<<CmdName<<": server reply not recognized - protocol error");

   return 1;
}

//_____________________________________________________________________________
bool XrdProofConn::GetAccessToSrv(XrdClientPhyConnection *p)
{
   // Gets access to the connected server.
   // The login and authorization steps are performed here.
   XPDLOC(ALL, "Conn::GetAccessToSrv")

   XrdClientPhyConnection *phyconn = (p) ? p : fPhyConn;
   // Now we are connected and we ask for the kind of the server
   {  XrdClientPhyConnLocker pcl(phyconn);
      fServerType = DoHandShake(p);
   }

   switch (fServerType) {

   case kSTXProofd:

      TRACE(DBG,"found server at "<<URLTAG);

      // Now we can start the reader thread in the physical connection, if needed
      if (phyconn == fPhyConn) fPhyConn->StartReader();
      fPhyConn->fServerType = kSTBaseXrootd;
      break;

   case kSTProofd:
      TRACE(DBG,"server at "<<URLTAG<<" is a proofd");
      // Close correctly this connection to proofd
      kXR_int32 dum[2];
      dum[0] = (kXR_int32)htonl(0);
      dum[1] = (kXR_int32)htonl(2034);
      WriteRaw(&dum[0], sizeof(dum), p);
      Close("P");
      return 0;

   case kSTError:
      TRACE(XERR,"handshake failed with server "<<URLTAG);
      Close("P");
      return 0;

   case kSTNone:
      TRACE(XERR,"server at "<<URLTAG<<" is unknown");
      Close("P");
      return 0;
   }

   bool ok = (phyconn == fPhyConn && fPhyConn->IsLogged() == kNo) ? Login() : 1;
   if (!ok) {
      TRACE(XERR,"client could not login at "<<URLTAG);
      return ok;
   }

   // We are done
   return ok;
}

//_____________________________________________________________________________
int XrdProofConn::WriteRaw(const void *buf, int len, XrdClientPhyConnection *phyconn)
{
   // Low level write call

   if (phyconn && phyconn->IsValid()) {
      phyconn->WriteRaw(buf, len, 0);
   } else if (fgConnMgr) {
      return fgConnMgr->WriteRaw(fLogConnID, buf, len, 0);
   }
   
   // No connection open
   return -1;
}

//_____________________________________________________________________________
int XrdProofConn::ReadRaw(void *buf, int len, XrdClientPhyConnection *phyconn)
{
   // Low level receive call

   if (phyconn && phyconn->IsValid()) {
      phyconn->ReadRaw(buf, len);
   } else if (fgConnMgr) {
      return fgConnMgr->ReadRaw(fLogConnID, buf, len);
   }

   // No connection open
   return -1;
}

//_____________________________________________________________________________
XrdProofConn::ESrvType XrdProofConn::DoHandShake(XrdClientPhyConnection *p)
{
   // Performs initial hand-shake with the server in order to understand which
   // kind of server is there at the other side
   XPDLOC(ALL, "Conn::DoHandShake")

   XrdClientPhyConnection *phyconn = (p) ? p : fPhyConn;
   
   // Nothing to do if already connected
   if (phyconn->fServerType == kSTBaseXrootd) {

      TRACE(DBG,"already connected to a PROOF server "<<URLTAG);
      return kSTXProofd;
   }

   // Set field in network byte order
   struct ClientInitHandShake initHS;
   memset(&initHS, 0, sizeof(initHS));
   initHS.third  = (kXR_int32)htonl((int)1);

   // Send to the server the initial hand-shaking message asking for the
   // kind of server
   int len = sizeof(initHS);
   TRACE(HDBG, "step 1: sending "<<len<<" bytes to server "<<URLTAG);

   int writeCount = WriteRaw(&initHS, len, p);
   if (writeCount != len) {
      TRACE(XERR, "sending "<<len<<" bytes to server "<<URLTAG);
      return kSTError;
   }

   // These 8 bytes are need by 'proofd' and discarded by XPD
   kXR_int32 dum[2];
   dum[0] = (kXR_int32)htonl(4);
   dum[1] = (kXR_int32)htonl(2012);
   writeCount = WriteRaw(&dum[0], sizeof(dum), p);
   if (writeCount != sizeof(dum)) {
      TRACE(XERR, "sending "<<sizeof(dum)<<" bytes to server "<<URLTAG);
      return kSTError;
   }

   // Read from server the first 4 bytes
   ServerResponseType type;
   len = sizeof(type);
   TRACE(HDBG, "step 2: reading "<<len<<" bytes from server "<<URLTAG);

   // Read returns the return value of TSocket->RecvRaw... that returns the
   // return value of recv (unix low level syscall)
   int readCount = ReadRaw(&type, len, p); // 4(2+2) bytes
   if (readCount != len) {
      if (readCount == (int)TXSOCK_ERR_TIMEOUT) {
         TRACE(ALL,"-----------------------");
         TRACE(ALL,"TimeOut condition reached reading from remote server.");
         TRACE(ALL,"This may indicate that the server is a 'proofd', version <= 12");
         TRACE(ALL,"Retry commenting the 'Plugin.TSlave' line in system.rootrc or adding");
         TRACE(ALL,"Plugin.TSlave: ^xpd  TSlave Proof \"TSlave(const char *,const char"
               " *,int,const char *, TProof *,ESlaveType,const char *,const char *)\"");
         TRACE(ALL,"to your $HOME/.rootrc .");
         TRACE(ALL,"-----------------------");
      } else {
         TRACE(XERR, "reading "<<len<<" bytes from server "<<URLTAG);
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
      TRACE(HDBG, "step 3: reading "<<len<<" bytes from server "<<URLTAG);

      readCount = ReadRaw(&xbody, len, p); // 12(4+4+4) bytes
      if (readCount != len) {
         TRACE(XERR, "reading "<<len<<" bytes from server "<<URLTAG);
         return kSTError;
      }

      XPD::ServerInitHandShake2HostFmt(&xbody);

      fRemoteProtocol = xbody.protover;
      if (fPhyConn->fServerProto <= 0)
         fPhyConn->fServerProto = fRemoteProtocol;

      return kSTXProofd;

   } else if (type == 8) {
      // Standard proofd
      return kSTProofd;
   } else {
      // We don't know the server type
      TRACE(XERR, "unknown server type ("<<type<<")");
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
   XPDLOC(ALL, "Conn::Login")

   XPClientRequest reqhdr, reqsave;

   // We fill the header struct containing the request for login
   memset( &reqhdr, 0, sizeof(reqhdr));

   reqhdr.login.pid = getpid();

   // User[:group] info (url's password field used for the group)
   XrdOucString ug = fUser;
   if (fUrl.Passwd.length() > 0) {
      ug += ":";
      ug += fUrl.Passwd;
   }

   // Fill login username
   if (ug.length() > 8) {
      // The name must go in the attached buffer because the login structure
      // can accomodate at most 8 chars
      strcpy( (char *)reqhdr.login.username, "?>buf" );
      // Add the name to the login buffer, if not already done during
      // a previous login (for example if we are reconnecting ...)
      if (fLoginBuffer.find("|usr:") == STR_NPOS) {
         fLoginBuffer += "|usr:";
         fLoginBuffer += ug;
      }
   } else if (ug.length() >= 0) {
      memcpy((void *)reqhdr.login.username, (void *)(ug.c_str()), ug.length());
      if (ug.length() < 8) reqhdr.login.username[ug.length()] = '\0';
   } else {
      strcpy((char *)reqhdr.login.username, "????" );
   }

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
   reqhdr.login.capver[0] = fCapVer;

   // We call SendReq, the function devoted to sending commands.
   if (TRACING(DBG)) {
      XrdOucString usr((const char *)&reqhdr.login.username[0], 8);
      TRACE(DBG, "logging into server "<<URLTAG<<"; pid="<<reqhdr.login.pid<<
                 "; uid=" << usr);
   }

   // Finish to fill up and ...
   SetSID(reqhdr.header.streamid);
   reqhdr.header.requestid = kXP_login;
   // ... saved it unmarshalled for retrials, if any
   memcpy(&reqsave, &reqhdr, sizeof(XPClientRequest));

   // Reset logged state
   fPhyConn->SetLogged(kNo);

   bool notdone = 1;
   bool resp = 1;


   // If positive answer
   XrdSecProtocol *secp = 0;
   while (notdone) {

      // server response header
      char *pltmp = 0;

      // Make sure we have the unmarshalled version
      memcpy(&reqhdr, &reqsave, sizeof(XPClientRequest));

      XrdClientMessage *xrsp = SendReq(&reqhdr, buf,
                                       &pltmp, "XrdProofConn::Login", 0);
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
            TRACE(DBG, "server requires authentication");

            secp = Authenticate(plist, (int)(len+1));
            resp = (secp != 0) ? 1 : 0;

            if (!resp)
               // We failed the aythentication attempt: cannot continue
               notdone = 0;

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
         // Print error msg, if any
         if (GetLastErr())
            XPDPRT(fHost << ": "<< GetLastErr());
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
   XPDLOC(ALL, "Conn::Authenticate")

   XrdSecProtocol *protocol = (XrdSecProtocol *)0;

   if (!plist || plsiz <= 0)
      return protocol;

   TRACE(DBG, "host "<<URLTAG<< " sent a list of "<<plsiz<<" bytes");
   //
   // Prepare host/IP information of the remote xrootd. This is required
   // for the authentication.
   struct sockaddr_in netaddr;
   char **hosterrmsg = 0;
   if (XrdSysDNS::getHostAddr((char *)fUrl.HostAddr.c_str(),
                                (struct sockaddr &)netaddr, hosterrmsg) <= 0) {
      TRACE(XERR, "getHostAddr: "<< *hosterrmsg);
      return protocol;
   }
   netaddr.sin_port   = fUrl.Port;
   //
   // Variables for negotiation
   XrdSecParameters  *secToken = 0;
   XrdSecCredentials *credentials = 0;

   //
   // Prepare the parms object
   char *bpar = (char *)malloc(plsiz + 1);
   if (bpar)
      memcpy(bpar, plist, plsiz);
   bpar[plsiz] = 0;
   XrdSecParameters Parms(bpar, plsiz + 1);

   // We need to load the protocol getter the first time we are here
   if (!fgSecGetProtocol) {
      static XrdSysLogger log;
      static XrdSysError err(&log, "XrdProofConn_");
      // Initialize the security library plugin, if needed
      XrdOucString libsec;
      if (!fgSecPlugin) { 
#if ROOTXRDVERS >= ROOT_XrdUtils
         libsec = "libXrdSec";
         libsec += LT_MODULE_EXT;
#else
         libsec = "libXrdSec.so";
#endif
         fgSecPlugin = new XrdSysPlugin(&err, libsec.c_str());
      }

      // Get the client protocol getter
      if (!(fgSecGetProtocol = fgSecPlugin->getPlugin("XrdSecGetProtocol"))) {
         TRACE(XERR, "unable to load XrdSecGetProtocol()");
         return protocol;
      }
   }
   //
   // Cycle through the security protocols accepted by the server
   while ((protocol = (*((XrdSecGetProt_t)fgSecGetProtocol))((char *)fUrl.Host.c_str(),
                                          (const struct sockaddr &)netaddr, Parms, 0))) {
      //
      // Protocol name
      XrdOucString protname = protocol->Entity.prot;
      //
      // Once we have the protocol, get the credentials
      XrdOucErrInfo ei;
      credentials = protocol->getCredentials(0, &ei);
      if (!credentials) {
         TRACE(XERR, "cannot obtain credentials (protocol: "<<protname<<")");
         // Set error, in case of need
         fLastErr = kXR_NotAuthorized;
         if (fLastErrMsg.length() > 0) fLastErrMsg += ":";
         fLastErrMsg += "cannot obtain credentials for protocol: ";
         fLastErrMsg += ei.getErrText();
         protocol->Delete();
         protocol = 0;
         continue;
      } else {
         TRACE(HDBG, "credentials size: " << credentials->size);
      }
      //
      // We fill the header struct containing the request for login
      XPClientRequest reqhdr;
      memset(reqhdr.auth.reserved, 0, 12);
      memset(reqhdr.auth.credtype, 0, 4);
      memcpy(reqhdr.auth.credtype, protname.c_str(), protname.length());

      bool failed = 0;
      int status = kXR_authmore;
      int dlen = 0;
      char *srvans = 0;
      XrdClientMessage *xrsp = 0;
      while (status == kXR_authmore) {
         //
         // Length of the credentials buffer
         SetSID(reqhdr.header.streamid);
         reqhdr.header.requestid = kXP_auth;
         reqhdr.header.dlen = (credentials) ? credentials->size : 0;
         char *credbuf = (credentials) ? credentials->buffer : 0;
         xrsp = SendReq(&reqhdr, credbuf, &srvans, "XrdProofConn::Authenticate");
         SafeDelete(credentials);
         status = (xrsp) ? xrsp->HeaderStatus() : kXR_error;
         dlen = (xrsp) ? xrsp->DataLen() : 0;
         TRACE(HDBG, "server reply: status: "<<status<<" dlen: "<<dlen);

         if (xrsp && (status == kXR_authmore)) {
            //
            // We are required to send additional information
            // First assign the security token that we have received
            // at the login request
            secToken = new XrdSecParameters(srvans, dlen);
            //
            // then get next part of the credentials
            credentials = protocol->getCredentials(secToken, &ei);
            SafeDelete(secToken); // nb: srvans is released here
            srvans = 0;
            if (!credentials) {
               TRACE(XERR, "cannot obtain credentials");
               // Set error, in case of need
               fLastErr = kXR_NotAuthorized;
               if (fLastErrMsg.length() > 0) fLastErrMsg += ":";
               fLastErrMsg += "cannot obtain credentials: ";
               fLastErrMsg += ei.getErrText();
               protocol->Delete();
               protocol = 0;
               // Server does not implement yet full cycling, so we are
               // allowed to try the handshake only for one protocol; we
               // cleanup the message and fail;
               SafeDelete(xrsp);
               failed = 1;
               break;
            } else {
               TRACE(HDBG, "credentials size " << credentials->size);
            }
         } else if (status != kXR_ok) {
            // Unexpected reply; print error msg, if any
            if (GetLastErr())
               TRACE(XERR, fHost << ": "<< GetLastErr());
            if (protocol) {
               protocol->Delete();
               protocol = 0;
            }
         }
         // Cleanup message
         SafeDelete(xrsp);
      }

      // If we are done
      if (protocol) {
         fLastErr = kXR_Unsupported;
         fLastErrMsg = "";
         break;
      }
      // Server does not implement yet full cycling, so we are
      // allowed to try the handshake only for one protocol; we
      if (failed) break;
   }
   if (!protocol) {
      TRACE(XERR, "unable to get protocol object.");
      // Set error, in case of need
      fLastErr = kXR_NotAuthorized;
      if (fLastErrMsg.length() > 0) fLastErrMsg += ":";
      fLastErrMsg += "unable to get protocol object.";
      TRACE(XERR, fLastErrMsg.c_str());
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

//_____________________________________________________________________________
void XrdProofConn::SetConnectInterrupt()
{
   // Interrupt connection attempts

   XrdSysMutexHelper mhp(fConnectInterruptMtx);
   fConnectInterrupt = 1;
}

//_____________________________________________________________________________
bool XrdProofConn::ConnectInterrupt()
{
   // Check if interrupted during connect

   bool rc = 0;
   {  XrdSysMutexHelper mhp(fConnectInterruptMtx);
      rc = fConnectInterrupt;
      // Reset the interrupt
      fConnectInterrupt = 0;
   }
   // Done
   return rc;
}

//_____________________________________________________________________________
bool XrdProofConn::IsValid() const
{
   // Test validity of this connection

   if (fConnected)
      if (fPhyConn && fPhyConn->IsValid())
         return 1;
   // Invalid
   return 0;
}

