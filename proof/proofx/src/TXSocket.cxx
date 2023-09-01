// @(#)root/proofx:$Id$
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TXSocket
\ingroup proofx

High level handler of connections to XProofD.
See TSocket for details.

*/

#include "MessageTypes.h"
#include "TEnv.h"
#include "TError.h"
#include "TException.h"
#include "TMonitor.h"
#include "TObjString.h"
#include "TProof.h"
#include "TSlave.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TUrl.h"
#include "TXHandler.h"
#include "TXSocket.h"
#include "XProofProtocol.h"

#include "XrdProofConn.h"

#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientMessage.hh"

#ifndef WIN32
#include <sys/socket.h>
#else
#include <Winsock2.h>
#endif


#include "XpdSysError.h"
#include "XpdSysLogger.h"

// ---- Tracing utils ----------------------------------------------------------
#include "XrdProofdTrace.h"
XrdOucTrace *XrdProofdTrace = 0;
static XrdSysLogger eLogger;
static XrdSysError eDest(0, "Proofx");

#ifdef WIN32
ULong64_t TSocket::fgBytesSent;
ULong64_t TSocket::fgBytesRecv;
#endif

//______________________________________________________________________________

//---- error handling ----------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////
/// Interface to ErrorHandler (protected).

void TXSocket::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   ::ErrorHandler(level, Form("TXSocket::%s", location), fmt, va);
}

//----- Ping handler -----------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////

class TXSocketPingHandler : public TFileHandler {
   TXSocket  *fSocket;
public:
   TXSocketPingHandler(TXSocket *s, Int_t fd)
      : TFileHandler(fd, 1) { fSocket = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

////////////////////////////////////////////////////////////////////////////////
/// Ping the socket

Bool_t TXSocketPingHandler::Notify()
{
   fSocket->Ping("ping handler");

   return kTRUE;
}

// Env variables init flag
Bool_t TXSocket::fgInitDone = kFALSE;

// Static variables for input notification
TXSockPipe   TXSocket::fgPipe;               // Pipe for input monitoring
TString      TXSocket::fgLoc = "undef";      // Location string

// Static buffer manager
std::mutex   TXSocket::fgSMtx;               // To protect spare list
std::list<TXSockBuf *> TXSocket::fgSQue;     // list of spare buffers
Long64_t     TXSockBuf::fgBuffMem = 0;       // Total allocated memory
Long64_t     TXSockBuf::fgMemMax = 10485760; // Max allowed allocated memory [10 MB]

////////////////////////////////////////////////////////////////////////////////
/// Constructor
/// Open the connection to a remote XrdProofd instance and start a PROOF
/// session.
/// The mode 'm' indicates the role of this connection:
///     'a'      Administrator; used by an XPD to contact the head XPD
///     'i'      Internal; used by a TXProofServ to call back its creator
///              (see XrdProofUnixConn)
///     'C'      PROOF manager: open connection only (do not start a session)
///     'M'      Client creating a top master
///     'A'      Client attaching to top master
///     'm'      Top master creating a submaster
///     's'      Master creating a slave
/// The buffer 'logbuf' is a null terminated string to be sent over at
/// login.

TXSocket::TXSocket(const char *url, Char_t m, Int_t psid, Char_t capver,
                   const char *logbuf, Int_t loglevel, TXHandler *handler)
         : TSocket(), fMode(m), fLogLevel(loglevel),
           fBuffer(logbuf), fConn(0), fASem(0), fAsynProc(1),
           fDontTimeout(kFALSE), fRDInterrupt(kFALSE), fXrdProofdVersion(-1)
{
   fUrl = url;
   // Enable tracing in the XrdProof client. if not done already
   eDest.logger(&eLogger);
   if (!XrdProofdTrace)
      XrdProofdTrace = new XrdOucTrace(&eDest);

   // Init envs the first time
   if (!fgInitDone)
      InitEnvs();

   // Async queue related stuff
   fAQue.clear();

   // Interrupts queue related stuff
   fILev = -1;
   fIForward = kFALSE;

   // Init some variables
   fByteLeft = 0;
   fByteCur = 0;
   fBufCur = 0;
   fServType = kPROOFD; // for consistency
   fTcpWindowSize = -1;
   fRemoteProtocol = -1;
   // By default forward directly to end-point
   fSendOpt = (fMode == 'i') ? (kXPD_internal | kXPD_async) : kXPD_async;
   fSessionID = (fMode == 'C') ? -1 : psid;
   fSocket = -1;

   // This is used by external code to create a link between this object
   // and another one
   fReference = 0;

   // The global pipe
   if (!fgPipe.IsValid()) {
      Error("TXSocket", "internal pipe is invalid");
      return;
   }

   // Some initial values
   TUrl u(url);
   fAddress = gSystem->GetHostByName(u.GetHost());
   u.SetProtocol("proof", kTRUE);
   fAddress.fPort = (u.GetPort() > 0) ? u.GetPort() : 1093;

   // Set the asynchronous handler
   fHandler = handler;

   if (url) {

      // Create connection (for managers the type of the connection is the same
      // as for top masters)
      char md = (fMode !='A' && fMode !='C') ? fMode : 'M';
      fConn = new XrdProofConn(url, md, psid, capver, this, fBuffer.Data());
      if (!fConn || !(fConn->IsValid())) {
         if (fConn->GetServType() != XrdProofConn::kSTProofd)
            if (gDebug > 0)
               Error("TXSocket", "fatal error occurred while opening a connection"
                                 " to server [%s]: %s", url, fConn->GetLastErr());
         return;
      }

      // Fill some info
      fUser = fConn->fUser.c_str();
      fHost = fConn->fHost.c_str();
      fPort = fConn->fPort;

      // Create new proofserv if not client manager or administrator or internal mode
      if (fMode == 'm' || fMode == 's' || fMode == 'M' || fMode == 'A'|| fMode == 'L') {
         // We attach or create
         if (!Create()) {
            // Failure
            Error("TXSocket", "create or attach failed (%s)",
                  ((fConn->fLastErrMsg.length() > 0) ? fConn->fLastErrMsg.c_str() : "-"));
            Close();
            return;
         }
      }

      // Fill some other info available if Create is successful
      if (fMode == 'C') {
         fXrdProofdVersion = fConn->fRemoteProtocol;
         fRemoteProtocol = fConn->fRemoteProtocol;
      }

      // Also in the base class
      fUrl = fConn->fUrl.GetUrl().c_str();
      fAddress = gSystem->GetHostByName(fConn->fUrl.Host.c_str());
      fAddress.fPort = fPort;

      // This is needed for the reader thread to signal an interrupt
      fPid = gSystem->GetPid();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TXSocket::~TXSocket()
{
   // Disconnect from remote server (the connection manager is
   // responsible of the underlying physical connection, so we do not
   // force its closing)
   Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Set location string

void TXSocket::SetLocation(const char *loc)
{
   if (loc) {
      fgLoc = loc;
      fgPipe.SetLoc(loc);
   } else {
      fgLoc = "";
      fgPipe.SetLoc("");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set session ID to 'id'. If id < 0, disable also the asynchronous handler.

void TXSocket::SetSessionID(Int_t id)
{
   if (id < 0 && fConn)
      fConn->SetAsync(0);
   fSessionID = id;
}

////////////////////////////////////////////////////////////////////////////////
/// Disconnect a session. Use opt= "S" or "s" to
/// shutdown remote session.
/// Default is opt = "".

void TXSocket::DisconnectSession(Int_t id, Option_t *opt)
{
   // Make sure we are connected
   if (!IsValid()) {
      if (gDebug > 0)
         Info("DisconnectSession","not connected: nothing to do");
      return;
   }

   Bool_t shutdown = opt && (strchr(opt,'S') || strchr(opt,'s'));
   Bool_t all = opt && (strchr(opt,'A') || strchr(opt,'a'));

   if (id > -1 || all) {
      // Prepare request
      XPClientRequest Request;
      memset(&Request, 0, sizeof(Request) );
      fConn->SetSID(Request.header.streamid);
      if (shutdown)
         Request.proof.requestid = kXP_destroy;
      else
         Request.proof.requestid = kXP_detach;
      Request.proof.sid = id;

      // Send request
      XrdClientMessage *xrsp =
         fConn->SendReq(&Request, (const void *)0, 0, "DisconnectSession");

      // Print error msg, if any
      if (!xrsp && fConn->GetLastErr())
         Printf("%s: %s", fHost.Data(), fConn->GetLastErr());

      // Cleanup
      SafeDelete(xrsp);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close connection. Available options are (case insensitive)
///   'P'   force closing of the underlying physical connection
///   'S'   shutdown remote session, is any
/// A session ID can be given using #...# signature, e.g. "#1#".
/// Default is opt = "".

void TXSocket::Close(Option_t *opt)
{
   Int_t to = gEnv->GetValue("XProof.AsynProcSemTimeout", 60);
   if (fAsynProc.Wait(to*1000) != 0)
      Warning("Close", "could not hold semaphore for async messages after %d sec: closing anyhow (may give error messages)", to);

   // Remove any reference in the global pipe and ready-sock queue
   TXSocket::fgPipe.Flush(this);

   // Make sure we have a connection
   if (!fConn) {
      if (gDebug > 0)
         Info("Close","no connection: nothing to do");
      fAsynProc.Post();
      return;
   }

   // Disconnect the asynchronous requests handler
   fConn->SetAsync(0);

   // If we are connected we disconnect
   if (IsValid()) {

      // Parse options
      TString o(opt);
      Int_t sessID = fSessionID;
      if (o.Index("#") != kNPOS) {
         o.Remove(0,o.Index("#")+1);
         if (o.Index("#") != kNPOS) {
            o.Remove(o.Index("#"));
            sessID = o.IsDigit() ? o.Atoi() : sessID;
         }
      }

      if (sessID > -1) {
         // Warn the remote session, if any (after destroy the session is gone)
         DisconnectSession(sessID, opt);
      } else {
         // We are the manager: close underlying connection
         fConn->Close(opt);
      }
   }

   // Delete the connection module
   SafeDelete(fConn);

   // Post semaphore
   fAsynProc.Post();
}

////////////////////////////////////////////////////////////////////////////////
/// We are here if an unsolicited response comes from a logical conn
/// The response comes in the form of an XrdClientMessage *, that must NOT be
/// destroyed after processing. It is destroyed by the first sender.
/// Remember that we are in a separate thread, since unsolicited
/// responses are asynchronous by nature.

UnsolRespProcResult TXSocket::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *,
                                                    XrdClientMessage *m)
{
   UnsolRespProcResult rc = kUNSOL_KEEP;

   // If we are closing we will not do anything
   TXSemaphoreGuard semg(&fAsynProc);
   if (!semg.IsValid()) {
      Error("ProcessUnsolicitedMsg", "%p: async semaphore taken by Close()! Should not be here!", this);
      return kUNSOL_CONTINUE;
   }

   if (!m) {
      if (gDebug > 2)
         Info("ProcessUnsolicitedMsg", "%p: got empty message: skipping", this);
      // Some one is perhaps interested in empty messages
      return kUNSOL_CONTINUE;
   } else {
      if (gDebug > 2)
         Info("ProcessUnsolicitedMsg", "%p: got message with status: %d, len: %d bytes (ID: %d)",
              this, m->GetStatusCode(), m->DataLen(), m->HeaderSID());
   }

   // Error notification
   if (m->IsError()) {
      if (m->GetStatusCode() != XrdClientMessage::kXrdMSC_timeout) {
         if (gDebug > 0)
            Info("ProcessUnsolicitedMsg","%p: got error from underlying connection", this);
         XHandleErr_t herr = {1, 0};
         if (!fHandler || fHandler->HandleError((const void *)&herr)) {
            if (gDebug > 0)
               Info("ProcessUnsolicitedMsg","%p: handler undefined or recovery failed", this);
            // Avoid to contact the server any more
            fSessionID = -1;
         } else {
            // Connection still usable: update usage timestamp
            Touch();
         }
      } else {
         // Time out
         if (gDebug > 2)
            Info("ProcessUnsolicitedMsg", "%p: underlying connection timed out", this);
      }
      // Propagate the message to other possible handlers
      return kUNSOL_CONTINUE;
   }

   // From now on make sure is for us (but only if not during setup, i.e. fConn == 0; otherwise
   // we may miss some important server message)
   if (fConn && !m->MatchStreamid(fConn->fStreamid)) {
      if (gDebug > 1)
         Info("ProcessUnsolicitedMsg", "%p: IDs do not match: {%d, %d}", this, fConn->fStreamid, m->HeaderSID());
      return kUNSOL_CONTINUE;
   }

   // Local processing ...
   Int_t len = 0;
   if ((len = m->DataLen()) < (int)sizeof(kXR_int32)) {
      Error("ProcessUnsolicitedMsg", "empty or bad-formed message - disabling");
      PostMsg(kPROOF_STOP);
      return rc;
   }

   // Activity on the line: update usage timestamp
   Touch();

   // The first 4 bytes contain the action code
   kXR_int32 acod = 0;
   memcpy(&acod, m->GetData(), sizeof(kXR_int32));
   if (acod > 10000)
         Info("ProcessUnsolicitedMsg", "%p: got acod %d (%x): message has status: %d, len: %d bytes (ID: %d)",
              this, acod, acod, m->GetStatusCode(), m->DataLen(), m->HeaderSID());
   //
   // Update pointer to data
   void *pdata = (void *)((char *)(m->GetData()) + sizeof(kXR_int32));
   len -= sizeof(kXR_int32);
   if (gDebug > 1)
      Info("ProcessUnsolicitedMsg", "%p: got action: %d (%d bytes) (ID: %d)",
           this, acod, len, m->HeaderSID());

   if (gDebug > 3)
      fgPipe.DumpReadySock();

   // Case by case
   kXR_int32 ilev = -1;
   const char *lab = 0;

   switch (acod) {
      case kXPD_ping:
         //
         // Special interrupt
         ilev = TProof::kPing;
         lab = "kXPD_ping";
      case kXPD_interrupt:
         //
         // Interrupt
         lab = !lab ? "kXPD_interrupt" : lab;
         {  std::lock_guard<std::recursive_mutex> lock(fIMtx);
            if (acod == kXPD_interrupt) {
               memcpy(&ilev, pdata, sizeof(kXR_int32));
               ilev = net2host(ilev);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // The next 4 bytes contain the forwarding option
            kXR_int32 ifw = 0;
            if (len > 0) {
               memcpy(&ifw, pdata, sizeof(kXR_int32));
               ifw = net2host(ifw);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","%s: forwarding option: %d", lab, ifw);
            }
            //
            // Save the interrupt
            fILev = ilev;
            fIForward = (ifw == 1) ? kTRUE : kFALSE;

            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, 0, 0, 0};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_timer:
         //
         // Set shutdown timer
         {
            kXR_int32 opt = 1;
            kXR_int32 delay = 0;
            // The next 4 bytes contain the shutdown option
            if (len > 0) {
               memcpy(&opt, pdata, sizeof(kXR_int32));
               opt = net2host(opt);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_timer: found opt: %d", opt);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // The next 4 bytes contain the delay
            if (len > 0) {
               memcpy(&delay, pdata, sizeof(kXR_int32));
               delay = net2host(delay);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_timer: found delay: %d", delay);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }

            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, opt, delay, 0};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_inflate:
         //
         // Set inflate factor
         {
            kXR_int32 inflate = 1000;
            if (len > 0) {
               memcpy(&inflate, pdata, sizeof(kXR_int32));
               inflate = net2host(inflate);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_inflate: factor: %d", inflate);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, inflate, 0, 0};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_priority:
         //
         // Broadcast group priority
         {
            kXR_int32 priority = -1;
            if (len > 0) {
               memcpy(&priority, pdata, sizeof(kXR_int32));
               priority = net2host(priority);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_priority: priority: %d", priority);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, priority, 0, 0};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_flush:
         //
         // Flush request
         {
            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, 0, 0, 0};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_urgent:
         //
         // Set shutdown timer
         {
            // The next 4 bytes contain the urgent msg type
            kXR_int32 type = -1;
            if (len > 0) {
               memcpy(&type, pdata, sizeof(kXR_int32));
               type = net2host(type);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_urgent: found type: %d", type);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // The next 4 bytes contain the first info container
            kXR_int32 int1 = -1;
            if (len > 0) {
               memcpy(&int1, pdata, sizeof(kXR_int32));
               int1 = net2host(int1);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_urgent: found int1: %d", int1);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            // The next 4 bytes contain the second info container
            kXR_int32 int2 = -1;
            if (len > 0) {
               memcpy(&int2, pdata, sizeof(kXR_int32));
               int2 = net2host(int2);
               if (gDebug > 1)
                  Info("ProcessUnsolicitedMsg","kXPD_urgent: found int2: %d", int2);
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }

            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, type, int1, int2};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
      case kXPD_msg:
         //
         // Data message
         {  std::lock_guard<std::recursive_mutex> lock(fAMtx);

            // Get a spare buffer
            TXSockBuf *b = PopUpSpare(len);
            if (!b) {
               Error("ProcessUnsolicitedMsg","could allocate spare buffer");
               return rc;
            }
            memcpy(b->fBuf, pdata, len);
            b->fLen = len;

            // Update counters
            fBytesRecv += len;

            // Produce the message
            fAQue.push_back(b);

            // Post the global pipe
            fgPipe.Post(this);

            // Signal it and release the mutex
            if (gDebug > 2)
               Info("ProcessUnsolicitedMsg","%p: %s: posting semaphore: %p (%d bytes)",
                                            this, GetTitle(), &fASem, len);
            fASem.Post();
         }

         break;
      case kXPD_feedback:
         Info("ProcessUnsolicitedMsg",
              "kXPD_feedback treatment not yet implemented");
         break;
      case kXPD_srvmsg:
         //
         // Service message
         {
            // The next 4 bytes may contain a flag to control the way the message is displayed
            kXR_int32 opt = 0;
            memcpy(&opt, pdata, sizeof(kXR_int32));
            opt = net2host(opt);
            if (opt >= 0 && opt <= 4) {
               // Update pointer to data
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            } else {
               opt = 1;
            }

            if (opt == 0) {
               // One line
               Printf("| %.*s", len, (char *)pdata);
            } else if (opt == 2) {
               // Raw displaying
               Printf("%.*s", len, (char *)pdata);
            } else if (opt == 3) {
               // Incremental displaying
               fprintf(stderr, "%.*s", len, (char *)pdata);
            } else if (opt == 4) {
               // Rewind
               fprintf(stderr, "%.*s\r", len, (char *)pdata);
            } else {
               // A small header
               Printf(" ");
               Printf("| Message from server:");
               Printf("| %.*s", len, (char *)pdata);
            }
         }
         break;
      case kXPD_errmsg:
         //
         // Error condition with message
         Printf("\n\n");
         Printf("| Error condition occured: message from server:");
         Printf("|    %.*s", len, (char *)pdata);
         Printf("\n");
         // Handle error
         if (fHandler)
            fHandler->HandleError();
         else
            Error("ProcessUnsolicitedMsg","handler undefined");
         break;
      case kXPD_msgsid:
         //
         // Data message
         { std::lock_guard<std::recursive_mutex> lock(fAMtx);

            // The next 4 bytes contain the sessiond id
            kXR_int32 cid = 0;
            memcpy(&cid, pdata, sizeof(kXR_int32));
            cid = net2host(cid);

            if (gDebug > 1)
               Info("ProcessUnsolicitedMsg","found cid: %d", cid);

            // Update pointer to data
            pdata = (void *)((char *)pdata + sizeof(kXR_int32));
            len -= sizeof(kXR_int32);

            // Get a spare buffer
            TXSockBuf *b = PopUpSpare(len);
            if (!b) {
               Error("ProcessUnsolicitedMsg","could allocate spare buffer");
               return rc;
            }
            memcpy(b->fBuf, pdata, len);
            b->fLen = len;

            // Set the sid
            b->fCid = cid;

            // Update counters
            fBytesRecv += len;

            // Produce the message
            fAQue.push_back(b);

            // Post the global pipe
            fgPipe.Post(this);

            // Signal it and release the mutex
            if (gDebug > 2)
               Info("ProcessUnsolicitedMsg","%p: cid: %d, posting semaphore: %p (%d bytes)",
                    this, cid, &fASem, len);
            fASem.Post();
         }

         break;
      case kXPD_wrkmortem:
         //
         // A worker died
         {  TString what = TString::Format("%.*s", len, (char *)pdata);
            if (what.BeginsWith("idle-timeout")) {
               // Notify the idle timeout
               PostMsg(kPROOF_FATAL, kPROOF_WorkerIdleTO);
            } else {
               Printf(" ");
               Printf("| %s", what.Data());
               // Handle error
               if (fHandler)
                  fHandler->HandleError();
               else
                  Error("ProcessUnsolicitedMsg","handler undefined");
            }
         }
         break;

      case kXPD_touch:
         //
         // Request for remote touch: post a message to do that
         PostMsg(kPROOF_TOUCH);
         break;
      case kXPD_resume:
         //
         // process the next query (in the TXProofServ)
         PostMsg(kPROOF_STARTPROCESS);
         break;
      case kXPD_clusterinfo:
         //
         // Broadcast cluster information
         {
            kXR_int32 nsess = -1, nacti = -1, neffs = -1;
            if (len > 0) {
               // Total sessions
               memcpy(&nsess, pdata, sizeof(kXR_int32));
               nsess = net2host(nsess);
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
               // Active sessions
               memcpy(&nacti, pdata, sizeof(kXR_int32));
               nacti = net2host(nacti);
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
               // Effective sessions
               memcpy(&neffs, pdata, sizeof(kXR_int32));
               neffs = net2host(neffs);
               pdata = (void *)((char *)pdata + sizeof(kXR_int32));
               len -= sizeof(kXR_int32);
            }
            if (gDebug > 1)
               Info("ProcessUnsolicitedMsg","kXPD_clusterinfo: # sessions: %d,"
                    " # active: %d, # effective: %f", nsess, nacti, neffs/1000.);
            // Handle this input in this thread to avoid queuing on the
            // main thread
            XHandleIn_t hin = {acod, nsess, nacti, neffs};
            if (fHandler)
               fHandler->HandleInput((const void *)&hin);
            else
               Error("ProcessUnsolicitedMsg","handler undefined");
         }
         break;
     default:
         Error("ProcessUnsolicitedMsg","%p: unknown action code: %d received from '%s' - disabling",
                                       this, acod, GetTitle());
         PostMsg(kPROOF_STOP);
         break;
   }

   // We are done
   return rc;
}

////////////////////////////////////////////////////////////////////////////////
/// Post a message of type 'type' into the read messages queue.
/// If 'msg' is defined it is also added as TString.
/// This is used, for example, with kPROOF_FATAL to force the main thread
/// to mark this socket as bad, avoiding race condition when a worker
/// dies while in processing state.

void TXSocket::PostMsg(Int_t type, const char *msg)
{
   // Create the message
   TMessage m(type);

   // Add the string if any
   if (msg && strlen(msg) > 0)
      m << TString(msg);

   // Write length in first word of buffer
   m.SetLength();

   // Get pointer to the message buffer
   char *mbuf = m.Buffer();
   Int_t mlen = m.Length();
   if (m.CompBuffer()) {
      mbuf = m.CompBuffer();
      mlen = m.CompLength();
   }

   //
   // Data message
   std::lock_guard<std::recursive_mutex> lock(fAMtx);

   // Get a spare buffer
   TXSockBuf *b = PopUpSpare(mlen);
   if (!b) {
      Error("PostMsg", "could allocate spare buffer");
      return;
   }

   // Fill the pipe buffer
   memcpy(b->fBuf, mbuf, mlen);
   b->fLen = mlen;

   // Update counters
   fBytesRecv += mlen;

   // Produce the message
   fAQue.push_back(b);

   // Post the global pipe
   fgPipe.Post(this);

   // Signal it and release the mutex
   if (gDebug > 0)
      Info("PostMsg", "%p: posting type %d to semaphore: %p (%d bytes)",
                          this, type, &fASem, mlen);
   fASem.Post();

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Wake up all threads waiting for at the semaphore (used by TXSlave)

void TXSocket::PostSemAll()
{
   std::lock_guard<std::recursive_mutex> lock(fAMtx);

   // Post semaphore to wake up anybody waiting; send as many posts as needed
   while (fASem.TryWait() != 1)
      fASem.Post();
  
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Getter for logical connection ID

Int_t TXSocket::GetLogConnID() const
{
   return (fConn ? fConn->GetLogConnID() : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Getter for last error

Int_t TXSocket::GetOpenError() const
{
   return (fConn ? fConn->GetOpenError() : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Getter for server type

Int_t TXSocket::GetServType() const
{
   return (fConn ? fConn->GetServType() : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Getter for session ID

Int_t TXSocket::GetSessionID() const
{
   return (fConn ? fConn->GetSessionID() : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Getter for validity status

Bool_t TXSocket::IsValid() const
{
   return (fConn ? (fConn->IsValid()) : kFALSE);
}

////////////////////////////////////////////////////////////////////////////////
/// Return kTRUE if the remote server is a 'proofd'

Bool_t TXSocket::IsServProofd()
{
   if (fConn && (fConn->GetServType() == XrdProofConn::kSTProofd))
      return kTRUE;

   // Failure
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Get latest interrupt level and reset it; if the interrupt has to be
/// propagated to lower stages forward will be kTRUE after the call

Int_t TXSocket::GetInterrupt(Bool_t &forward)
{
   if (gDebug > 2)
      Info("GetInterrupt","%p: waiting to lock mutex", this);

   std::lock_guard<std::recursive_mutex> lock(fIMtx);

   // Reset values
   Int_t ilev = -1;
   forward = kFALSE;

   // Check if filled
   if (fILev == -1)
      Error("GetInterrupt", "value is unset (%d) - protocol error",fILev);

   // Fill output
   ilev = fILev;
   forward = fIForward;

   // Reset values (we process it only once)
   fILev = -1;
   fIForward = kFALSE;

   // Return what we got
   return ilev;
}

////////////////////////////////////////////////////////////////////////////////
/// Flush the asynchronous queue.
/// Typically called when a kHardInterrupt is received.
/// Returns number of bytes in flushed buffers.

Int_t TXSocket::Flush()
{
   Int_t nf = 0;
   list<TXSockBuf *> splist;
   list<TXSockBuf *>::iterator i;

   {  std::lock_guard<std::recursive_mutex> lock(fAMtx);

      // Must have something to flush
      if (fAQue.size() > 0) {

         // Save size for later semaphore cleanup
         Int_t sz = fAQue.size();
         // get the highest interrupt level
         for (i = fAQue.begin(); i != fAQue.end();) {
            if (*i) {
               splist.push_back(*i);
               nf += (*i)->fLen;
               i = fAQue.erase(i);
            }
         }

         // Reset the asynchronous queue
         while (sz--) {
            if (fASem.TryWait() == 1)
               Printf("Warning in TXSocket::Flush: semaphore counter already 0 (sz: %d)", sz);
         }
         fAQue.clear();
      }
   }

   // Move spares to the spare queue
   {  std::lock_guard<std::mutex> lock(fgSMtx);
      if (splist.size() > 0) {
         for (i = splist.begin(); i != splist.end();) {
            fgSQue.push_back(*i);
            i = splist.erase(i);
         }
      }
   }

   // We are done
   return nf;
}

////////////////////////////////////////////////////////////////////////////////
/// This method sends a request for creation of (or attachment to) a remote
/// server application.

Bool_t TXSocket::Create(Bool_t attach)
{
   // Make sure we are connected
   if (!IsValid()) {
      if (gDebug > 0)
         Info("Create","not connected: nothing to do");
      return kFALSE;
   }

   Int_t retriesleft = gEnv->GetValue("XProof.CreationRetries", 4);

   while (retriesleft--) {

      XPClientRequest reqhdr;

      // We fill the header struct containing the request for login
      memset( &reqhdr, 0, sizeof(reqhdr));
      fConn->SetSID(reqhdr.header.streamid);

      // This will be a kXP_attach or kXP_create request
      if (fMode == 'A' || attach) {
         reqhdr.header.requestid = kXP_attach;
         reqhdr.proof.sid = fSessionID;
      } else {
         reqhdr.header.requestid = kXP_create;
      }

      // Send log level
      reqhdr.proof.int1 = fLogLevel;

      // Send also the chosen alias
      const void *buf = (const void *)(fBuffer.Data());
      reqhdr.header.dlen = fBuffer.Length();
      if (gDebug >= 2)
         Info("Create", "sending %d bytes to server", reqhdr.header.dlen);

      // We call SendReq, the function devoted to sending commands.
      if (gDebug > 1)
         Info("Create", "creating session of server %s", fUrl.Data());

      // server response header
      char *answData = 0;
      XrdClientMessage *xrsp = fConn->SendReq(&reqhdr, buf,
                                              &answData, "TXSocket::Create", 0);
      struct ServerResponseBody_Protocol *srvresp = (struct ServerResponseBody_Protocol *)answData;

      // If any, the URL the data pool entry point will be stored here
      fBuffer = "";
      if (xrsp) {

         //
         // Pointer to data
         void *pdata = (void *)(xrsp->GetData());
         Int_t len = xrsp->DataLen();

         if (len >= (Int_t)sizeof(kXR_int32)) {
            // The first 4 bytes contain the session ID
            kXR_int32 psid = 0;
            memcpy(&psid, pdata, sizeof(kXR_int32));
            fSessionID = net2host(psid);
            pdata = (void *)((char *)pdata + sizeof(kXR_int32));
            len -= sizeof(kXR_int32);
         } else {
            Error("Create","session ID is undefined!");
            fSessionID = -1;
            if (srvresp) free(srvresp);
            return kFALSE;
         }

         if (len >= (Int_t)sizeof(kXR_int16)) {
            // The second 2 bytes contain the remote PROOF protocol version
            kXR_int16 dver = 0;
            memcpy(&dver, pdata, sizeof(kXR_int16));
            fRemoteProtocol = net2host(dver);
            pdata = (void *)((char *)pdata + sizeof(kXR_int16));
            len -= sizeof(kXR_int16);
         } else {
            Warning("Create","protocol version of the remote PROOF undefined!");
         }

         if (fRemoteProtocol == 0) {
            // We are dealing with an older server: the PROOF protocol is on 4 bytes
            len += sizeof(kXR_int16);
            kXR_int32 dver = 0;
            memcpy(&dver, pdata, sizeof(kXR_int32));
            fRemoteProtocol = net2host(dver);
            pdata = (void *)((char *)pdata + sizeof(kXR_int32));
            len -= sizeof(kXR_int32);
         } else {
            if (len >= (Int_t)sizeof(kXR_int16)) {
               // The third 2 bytes contain the remote XrdProofdProtocol version
               kXR_int16 dver = 0;
               memcpy(&dver, pdata, sizeof(kXR_int16));
               fXrdProofdVersion = net2host(dver);
               pdata = (void *)((char *)pdata + sizeof(kXR_int16));
               len -= sizeof(kXR_int16);
            } else {
               Warning("Create","version of the remote XrdProofdProtocol undefined!");
            }
         }

         if (len > 0) {
            // From top masters, the url of the data pool
            char *url = new char[len+1];
            memcpy(url, pdata, len);
            url[len] = 0;
            fBuffer = url;
            delete[] url;
         }

         // Cleanup
         SafeDelete(xrsp);
         if (srvresp) free(srvresp);

         // Notify
         return kTRUE;
      } else {
         // Extract log file path, if any
         Ssiz_t ilog = kNPOS;
         if (retriesleft <= 0 && fConn->GetLastErr()) {
            fBuffer = fConn->GetLastErr();
            if ((ilog = fBuffer.Index("|log:")) != kNPOS) fBuffer.Remove(0, ilog);
         }
         // If not free resources now, just give up
         if (fConn->GetOpenError() == kXP_TooManySess) {
            // Avoid to contact the server any more
            fSessionID = -1;
            if (srvresp) free(srvresp);
            return kFALSE;
         } else {
            // Print error msg, if any
            if ((retriesleft <= 0 || gDebug > 0) && fConn->GetLastErr()) {
               TString emsg(fConn->GetLastErr());
               if ((ilog = emsg.Index("|log:")) != kNPOS) emsg.Remove(ilog);
               Printf("%s: %s", fHost.Data(), emsg.Data());
            }
         }
      }

      if (gDebug > 0)
         Info("Create", "creation/attachment attempt failed: %d attempts left", retriesleft);
      if (retriesleft <= 0)
         Error("Create", "%d creation/attachment attempts failed: no attempts left",
                         gEnv->GetValue("XProof.CreationRetries", 4));

      if (srvresp) free(srvresp);
   } // Creation retries

   // The session is invalid: reset the sessionID to invalid state (it was our protocol
   // number during creation
   fSessionID = -1;

   // Notify failure
   Error("Create:",
         "problems creating or attaching to a remote server (%s)",
         ((fConn->fLastErrMsg.length() > 0) ? fConn->fLastErrMsg.c_str() : "-"));
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Send a raw buffer of specified length.
/// Use opt = kDontBlock to ask xproofd to push the message into the proofsrv.
/// (by default is appended to a queue waiting for a request from proofsrv).
/// Returns the number of bytes sent or -1 in case of error.

Int_t TXSocket::SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt)
{
   TSystem::ResetErrno();

   // Options and request ID
   fSendOpt = (opt == kDontBlock) ? (kXPD_async | fSendOpt)
                                  : (~kXPD_async & fSendOpt) ;

   // Prepare request
   XPClientRequest Request;
   memset( &Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   Request.sendrcv.requestid = kXP_sendmsg;
   Request.sendrcv.sid = fSessionID;
   Request.sendrcv.opt = fSendOpt;
   Request.sendrcv.cid = GetClientID();
   Request.sendrcv.dlen = length;
   if (gDebug >= 2)
      Info("SendRaw", "sending %d bytes to server", Request.sendrcv.dlen);

   // Send request
   XrdClientMessage *xrsp = fConn->SendReq(&Request, buffer, 0, "SendRaw");

   if (xrsp) {
      // Prepare return info
      Int_t nsent = length;

      // Update counters
      fBytesSent += length;

      // Cleanup
      SafeDelete(xrsp);

      // Success: update usage timestamp
      Touch();

      // ok
      return nsent;
   } else {
      // Print error message, if any
      if (fConn->GetLastErr())
         Printf("%s: %s", fHost.Data(), fConn->GetLastErr());
      else
         Printf("%s: error occured but no message from server", fHost.Data());
   }

   // Failure notification (avoid using the handler: we may be exiting)
   Error("SendRaw", "%s: problems sending %d bytes to server",
                    fHost.Data(), length);
   return -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Ping functionality: contact the server to check its vitality.
/// If external, the server waits for a reply from the server
/// Returns kTRUE if OK or kFALSE in case of error.

Bool_t TXSocket::Ping(const char *ord)
{
   TSystem::ResetErrno();

   if (gDebug > 0)
      Info("Ping","%p: %s: sid: %d", this, ord ? ord : "int", fSessionID);

   // Make sure we are connected
   if (!IsValid()) {
      Error("Ping","not connected: nothing to do");
      return kFALSE;
   }

   // Options
   kXR_int32 options = (fMode == 'i') ? kXPD_internal : 0;

   // Prepare request
   XPClientRequest Request;
   memset( &Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   Request.sendrcv.requestid = kXP_ping;
   Request.sendrcv.sid = fSessionID;
   Request.sendrcv.opt = options;
   Request.sendrcv.dlen = 0;

   // Send request
   Bool_t res = kFALSE;
   if (fMode != 'i') {
      char *pans = 0;
      XrdClientMessage *xrsp =
         fConn->SendReq(&Request, (const void *)0, &pans, "Ping");
      kXR_int32 *pres = (kXR_int32 *) pans;

      // Get the result
      if (xrsp && xrsp->HeaderStatus() == kXR_ok) {
         *pres = net2host(*pres);
         res = (*pres == 1) ? kTRUE : kFALSE;
         // Success: update usage timestamp
         Touch();
      } else {
         // Print error msg, if any
         if (fConn->GetLastErr())
            Printf("%s: %s", fHost.Data(), fConn->GetLastErr());
      }

      // Cleanup
      SafeDelete(xrsp);
      if (pans) free(pans);

   } else {
      if (XPD::clientMarshall(&Request) == 0) {
         XReqErrorType e = fConn->LowWrite(&Request, 0, 0);
         res = (e == kOK) ? kTRUE : kFALSE;
      } else {
         Error("Ping", "%p: int: problems marshalling request", this);
      }
   }

   // Failure notification (avoid using the handler: we may be exiting)
   if (!res) {
      Error("Ping", "%p: %s: problems sending ping to server", this, ord ? ord : "int");
   } else if (gDebug > 0) {
      Info("Ping","%p: %s: sid: %d OK", this, ord ? ord : "int", fSessionID);
   }

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Remote touch functionality: contact the server to proof our vitality.
/// No reply from server is expected.

void TXSocket::RemoteTouch()
{
   TSystem::ResetErrno();

   if (gDebug > 0)
      Info("RemoteTouch","%p: sending touch request to %s", this, GetName());

   // Make sure we are connected
   if (!IsValid()) {
      Error("RemoteTouch","not connected: nothing to do");
      return;
   }

   // Prepare request
   XPClientRequest Request;
   memset( &Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   Request.sendrcv.requestid = kXP_touch;
   Request.sendrcv.sid = fSessionID;
   Request.sendrcv.opt = 0;
   Request.sendrcv.dlen = 0;

   // We need the right order
   if (XPD::clientMarshall(&Request) != 0) {
      Error("Touch", "%p: problems marshalling request ", this);
      return;
   }
   if (fConn->LowWrite(&Request, 0, 0) != kOK)
      Error("Touch", "%p: problems sending touch request to server", this);

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Interrupt the remote protocol instance. Used to propagate Ctrl-C.
/// No reply from server is expected.

void TXSocket::CtrlC()
{
   TSystem::ResetErrno();

   if (gDebug > 0)
      Info("CtrlC","%p: sending ctrl-c request to %s", this, GetName());

   // Make sure we are connected
   if (!IsValid()) {
      Error("CtrlC","not connected: nothing to do");
      return;
   }

   // Prepare request
   XPClientRequest Request;
   memset( &Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   Request.proof.requestid = kXP_ctrlc;
   Request.proof.sid = 0;
   Request.proof.dlen = 0;

   // We need the right order
   if (XPD::clientMarshall(&Request) != 0) {
      Error("CtrlC", "%p: problems marshalling request ", this);
      return;
   }
   if (fConn->LowWrite(&Request, 0, 0) != kOK)
      Error("CtrlC", "%p: problems sending ctrl-c request to server", this);

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Wait and pick-up next buffer from the asynchronous queue

Int_t TXSocket::PickUpReady()
{
   fBufCur = 0;
   fByteLeft = 0;
   fByteCur = 0;
   if (gDebug > 2)
      Info("PickUpReady", "%p: %s: going to sleep", this, GetTitle());

   // User can choose whether to wait forever or for a fixed amount of time
   if (!fDontTimeout) {
      static Int_t timeout = gEnv->GetValue("XProof.ReadTimeout", 300) * 1000;
      static Int_t dt = 2000;
      Int_t to = timeout;
      SetInterrupt(kFALSE);
      while (to && !IsInterrupt()) {
         SetAWait(kTRUE);
         if (fASem.Wait(dt) != 0) {
            to -= dt;
            if (to <= 0) {
               Error("PickUpReady","error waiting at semaphore");
               return -1;
            } else {
               if (gDebug > 0)
                  Info("PickUpReady", "%p: %s: got timeout: retring (%d secs)",
                                      this, GetTitle(), to/1000);
            }
         } else
            break;
         SetAWait(kFALSE);
      }
      // We wait forever
      if (IsInterrupt()) {
         if (gDebug > 2)
            Info("PickUpReady","interrupted");
         SetInterrupt(kFALSE);
         SetAWait(kFALSE);
         return -1;
      }
   } else {
      // We wait forever
      SetAWait(kTRUE);
      if (fASem.Wait() != 0) {
         Error("PickUpReady","error waiting at semaphore");
         SetAWait(kFALSE);
         return -1;
      }
      SetAWait(kFALSE);
   }
   if (gDebug > 2)
      Info("PickUpReady", "%p: %s: waken up", this, GetTitle());

   std::lock_guard<std::recursive_mutex> lock(fAMtx);

   // Get message, if any
   if (fAQue.size() <= 0) {
      Error("PickUpReady","queue is empty - protocol error ?");
      return -1;
   }
   if (!(fBufCur = fAQue.front())) {
      Error("PickUpReady","got invalid buffer - protocol error ?");
      return -1;
   }
   // Remove message from the queue
   fAQue.pop_front();

   // Set number of available bytes
   fByteLeft = fBufCur->fLen;

   if (gDebug > 2)
      Info("PickUpReady", "%p: %s: got message (%d bytes)",
                          this, GetTitle(), (Int_t)(fBufCur ? fBufCur->fLen : 0));

   // Update counters
   fBytesRecv += fBufCur->fLen;

   // Set session ID
   if (fBufCur->fCid > -1 && fBufCur->fCid != GetClientID())
      SetClientID(fBufCur->fCid);

   // Clean entry in the underlying pipe
   fgPipe.Clean(this);

   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Pop-up a buffer of at least size bytes from the spare list
/// If none is found either one is reallocated or a new one
/// created

TXSockBuf *TXSocket::PopUpSpare(Int_t size)
{
   TXSockBuf *buf = 0;
   static Int_t nBuf = 0;

   std::lock_guard<std::mutex> lock(fgSMtx);

   Int_t maxsz = 0;
   if (fgSQue.size() > 0) {
      list<TXSockBuf *>::iterator i;
      for (i = fgSQue.begin(); i != fgSQue.end(); ++i) {
         maxsz = ((*i)->fSiz > maxsz) ? (*i)->fSiz : maxsz;
         if ((*i) && (*i)->fSiz >= size) {
            buf = *i;
            if (gDebug > 2)
               Info("PopUpSpare","asked: %d, spare: %d/%d, REUSE buf %p, sz: %d",
                                 size, (int) fgSQue.size(), nBuf, buf, buf->fSiz);
            // Drop from this list
            fgSQue.erase(i);
            return buf;
         }
      }
      // All buffers are too small: enlarge the first one
      buf = fgSQue.front();
      buf->Resize(size);
      if (gDebug > 2)
         Info("PopUpSpare","asked: %d, spare: %d/%d, maxsz: %d, RESIZE buf %p, sz: %d",
                           size, (int) fgSQue.size(), nBuf, maxsz, buf, buf->fSiz);
      // Drop from this list
      fgSQue.pop_front();
      return buf;
   }

   // Create a new buffer
   buf = new TXSockBuf((char *)malloc(size), size);
   nBuf++;

   if (gDebug > 2)
      Info("PopUpSpare","asked: %d, spare: %d/%d, maxsz: %d, NEW buf %p, sz: %d",
                        size, (int) fgSQue.size(), nBuf, maxsz, buf, buf->fSiz);

   // We are done
   return buf;
}

////////////////////////////////////////////////////////////////////////////////
/// Release read buffer giving back to the spare list

void TXSocket::PushBackSpare()
{
   std::lock_guard<std::mutex> lock(fgSMtx);

   if (gDebug > 2)
      Info("PushBackSpare","release buf %p, sz: %d (BuffMem: %lld)",
                           fBufCur, fBufCur->fSiz, TXSockBuf::BuffMem());

   if (TXSockBuf::BuffMem() < TXSockBuf::GetMemMax()) {
      fgSQue.push_back(fBufCur);
   } else {
      delete fBufCur;
   }
   fBufCur = 0;
   fByteCur = 0;
   fByteLeft = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a raw buffer of specified length bytes.

Int_t TXSocket::RecvRaw(void *buffer, Int_t length, ESendRecvOptions)
{
   // Inputs must make sense
   if (!buffer || (length <= 0))
      return -1;

   // Wait and pick-up a read buffer if we do not have one
   if (!fBufCur && (PickUpReady() != 0))
      return -1;

   // Use it
   if (fByteLeft >= length) {
      memcpy(buffer, fBufCur->fBuf + fByteCur, length);
      fByteCur += length;
      if ((fByteLeft -= length) <= 0)
         // All used: give back
         PushBackSpare();
      // Success: update usage timestamp
      Touch();
      return length;
   } else {
      // Copy the first part
      memcpy(buffer, fBufCur->fBuf + fByteCur, fByteLeft);
      Int_t at = fByteLeft;
      Int_t tobecopied = length - fByteLeft;
      PushBackSpare();
      while (tobecopied > 0) {
         // Pick-up next buffer (it may wait inside)
         if (PickUpReady() != 0)
            return -1;
         // Copy the fresh meat
         Int_t ncpy = (fByteLeft > tobecopied) ? tobecopied : fByteLeft;
         memcpy((void *)((Char_t *)buffer+at), fBufCur->fBuf, ncpy);
         fByteCur = ncpy;
         if ((fByteLeft -= ncpy) <= 0)
            // All used: give back
            PushBackSpare();
         // Recalculate
         tobecopied -= ncpy;
         at += ncpy;
      }
   }

   // Update counters
   fBytesRecv  += length;
   fgBytesRecv += length;

   // Success: update usage timestamp
   Touch();

   return length;
}

////////////////////////////////////////////////////////////////////////////////
/// Send urgent message (interrupt) to remote server
/// Returns 0 or -1 in case of error.

Int_t TXSocket::SendInterrupt(Int_t type)
{
   TSystem::ResetErrno();

   // Prepare request
   XPClientRequest Request;
   memset(&Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   if (type == (Int_t) TProof::kShutdownInterrupt)
      Request.interrupt.requestid = kXP_destroy;
   else
      Request.interrupt.requestid = kXP_interrupt;
   Request.interrupt.sid = fSessionID;
   Request.interrupt.type = type;    // type of interrupt (see TProof::EUrgent)
   Request.interrupt.dlen = 0;

   // Send request
   XrdClientMessage *xrsp =
      fConn->SendReq(&Request, (const void *)0, 0, "SendInterrupt");
   if (xrsp) {
      // Success: update usage timestamp
      Touch();
      // Cleanup
      SafeDelete(xrsp);
      // ok
      return 0;
   } else {
      // Print error msg, if any
      if (fConn->GetLastErr())
         Printf("%s: %s", fHost.Data(), fConn->GetLastErr());
   }

   // Failure notification (avoid using the handler: we may be exiting)
   Error("SendInterrupt", "problems sending interrupt to server");
   return -1;
}

////////////////////////////////////////////////////////////////////////////////

void TXSocket::SetInterrupt(Bool_t i)
{
   std::lock_guard<std::recursive_mutex> lock(fAMtx);
   fRDInterrupt = i;
   if (i && fConn) fConn->SetInterrupt();
   if (i && fAWait) fASem.Post();
}

////////////////////////////////////////////////////////////////////////////////
/// Send a TMessage object. Returns the number of bytes in the TMessage
/// that were sent and -1 in case of error.

Int_t TXSocket::Send(const TMessage &mess)
{
   TSystem::ResetErrno();

   if (mess.IsReading()) {
      Error("Send", "cannot send a message used for reading");
      return -1;
   }

   // send streamer infos in case schema evolution is enabled in the TMessage
   SendStreamerInfos(mess);

   // send the process id's so TRefs work
   SendProcessIDs(mess);

   mess.SetLength();   //write length in first word of buffer

   if (GetCompressionLevel() > 0 && mess.GetCompressionLevel() == 0)
      const_cast<TMessage&>(mess).SetCompressionSettings(fCompress);

   if (mess.GetCompressionLevel() > 0)
      const_cast<TMessage&>(mess).Compress();

   char *mbuf = mess.Buffer();
   Int_t mlen = mess.Length();
   if (mess.CompBuffer()) {
      mbuf = mess.CompBuffer();
      mlen = mess.CompLength();
   }

   // Parse message type to choose sending options
   kXR_int32 fSendOptDefault = fSendOpt;
   switch (mess.What()) {
      case kPROOF_PROCESS:
         fSendOpt |= kXPD_process;
         break;
      case kPROOF_PROGRESS:
      case kPROOF_FEEDBACK:
         fSendOpt |= kXPD_fb_prog;
         break;
      case kPROOF_QUERYSUBMITTED:
         fSendOpt |= kXPD_querynum;
         fSendOpt |= kXPD_fb_prog;
         break;
      case kPROOF_STARTPROCESS:
         fSendOpt |= kXPD_startprocess;
         fSendOpt |= kXPD_fb_prog;
         break;
      case kPROOF_STOPPROCESS:
         fSendOpt |= kXPD_fb_prog;
         break;
      case kPROOF_SETIDLE:
         fSendOpt |= kXPD_setidle;
         fSendOpt |= kXPD_fb_prog;
         break;
      case kPROOF_LOGFILE:
      case kPROOF_LOGDONE:
         if (GetClientIDSize() <= 1)
            fSendOpt |= kXPD_logmsg;
         break;
      default:
         break;
   }

   if (gDebug > 2)
      Info("Send", "sending type %d (%d bytes) to '%s'", mess.What(), mlen, GetTitle());

   Int_t nsent = SendRaw(mbuf, mlen);
   fSendOpt = fSendOptDefault;

   if (nsent <= 0)
      return nsent;

   fBytesSent  += nsent;
   fgBytesSent += nsent;

   return nsent - sizeof(UInt_t);  //length - length header
}

////////////////////////////////////////////////////////////////////////////////
/// Receive a TMessage object. The user must delete the TMessage object.
/// Returns length of message in bytes (can be 0 if other side of connection
/// is closed) or -1 in case of error or -5 if pipe broken (connection invalid).
/// In those case mess == 0.

Int_t TXSocket::Recv(TMessage *&mess)
{
   TSystem::ResetErrno();

   if (!IsValid()) {
      mess = 0;
      return -5;
   }

oncemore:
   Int_t  n;
   UInt_t len;
   if ((n = RecvRaw(&len, sizeof(UInt_t))) <= 0) {
      mess = 0;
      return n;
   }
   len = net2host(len);  //from network to host byte order

   char *buf = new char[len+sizeof(UInt_t)];
   if ((n = RecvRaw(buf+sizeof(UInt_t), len)) <= 0) {
      delete [] buf;
      mess = 0;
      return n;
   }

   fBytesRecv  += n + sizeof(UInt_t);
   fgBytesRecv += n + sizeof(UInt_t);

   mess = new TMessage(buf, len+sizeof(UInt_t));

   // receive any streamer infos
   if (RecvStreamerInfos(mess))
      goto oncemore;

   // receive any process ids
   if (RecvProcessIDs(mess))
      goto oncemore;

   if (mess->What() & kMESS_ACK) {
      // Acknowledgement embedded: ignore ...
      mess->SetWhat(mess->What() & ~kMESS_ACK);
   }

   return n;
}

////////////////////////////////////////////////////////////////////////////////
/// Send message to intermediate coordinator.
/// If any output is due, this is returned as an obj string to be
/// deleted by the caller

TObjString *TXSocket::SendCoordinator(Int_t kind, const char *msg, Int_t int2,
                                      Long64_t l64, Int_t int3, const char *)
{
   TObjString *sout = 0;

   // We fill the header struct containing the request
   XPClientRequest reqhdr;
   const void *buf = 0;
   char *bout = 0;
   char **vout = 0;
   memset(&reqhdr, 0, sizeof(reqhdr));
   fConn->SetSID(reqhdr.header.streamid);
   reqhdr.header.requestid = kXP_admin;
   reqhdr.proof.int1 = kind;
   reqhdr.proof.int2 = int2;
   switch (kind) {
      case kQueryMssUrl:
      case kQueryROOTVersions:
      case kQuerySessions:
      case kQueryWorkers:
         reqhdr.proof.sid = 0;
         reqhdr.header.dlen = 0;
         vout = (char **)&bout;
         break;
      case kCleanupSessions:
         reqhdr.proof.int2 = (int2 == 1) ? (kXR_int32) kXPD_AnyServer
                                         : (kXR_int32) kXPD_TopMaster;
         reqhdr.proof.int3 = int2;
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         buf = (msg) ? (const void *)msg : buf;
         break;
      case kCpFile:
      case kGetFile:
      case kPutFile:
      case kExec:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         buf = (msg) ? (const void *)msg : buf;
         vout = (char **)&bout;
         break;
      case kQueryLogPaths:
         vout = (char **)&bout;
         reqhdr.proof.int3 = int3;
      case kReleaseWorker:
      case kSendMsgToUser:
      case kGroupProperties:
      case kSessionTag:
      case kSessionAlias:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         buf = (msg) ? (const void *)msg : buf;
         break;
      case kROOTVersion:
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         buf = (msg) ? (const void *)msg : buf;
         break;
      case kGetWorkers:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         if (msg)
            buf = (const void *)msg;
         vout = (char **)&bout;
         break;
      case kReadBuffer:
         reqhdr.header.requestid = kXP_readbuf;
         reqhdr.readbuf.ofs = l64;
         reqhdr.readbuf.len = int2;
         if (int3 > 0 && fXrdProofdVersion < 1003) {
            Info("SendCoordinator", "kReadBuffer: old server (ver %d < 1003):"
                 " grep functionality not supported", fXrdProofdVersion);
            return sout;
         }
         reqhdr.readbuf.int1 = int3;
         if (!msg || strlen(msg) <= 0) {
            Info("SendCoordinator", "kReadBuffer: file path undefined");
            return sout;
         }
         reqhdr.header.dlen = strlen(msg);
         buf = (const void *)msg;
         vout = (char **)&bout;
         break;
      default:
         Info("SendCoordinator", "unknown message kind: %d", kind);
         return sout;
   }

   // server response header
   Bool_t noterr = (gDebug > 0) ? kTRUE : kFALSE;
   XrdClientMessage *xrsp =
      fConn->SendReq(&reqhdr, buf, vout, "TXSocket::SendCoordinator", noterr);

   // If positive answer
   if (xrsp) {
      // Check if we need to create an output string
      if (bout && (xrsp->DataLen() > 0))
         sout = new TObjString(TString(bout,xrsp->DataLen()));
      if (bout)
         free(bout);
      // Success: update usage timestamp
      Touch();
      SafeDelete(xrsp);
   } else {
      // Print error msg, if any
      if (fConn->GetLastErr())
         Printf("%s: %s", fHost.Data(), fConn->GetLastErr());
   }

   // Failure notification (avoid using the handler: we may be exiting)
   return sout;
}

////////////////////////////////////////////////////////////////////////////////
/// Send urgent message to counterpart; 'type' specifies the type of
/// the message (see TXSocket::EUrgentMsgType), and 'int1', 'int2'
/// two containers for additional information.

void TXSocket::SendUrgent(Int_t type, Int_t int1, Int_t int2)
{
   TSystem::ResetErrno();

   // Prepare request
   XPClientRequest Request;
   memset(&Request, 0, sizeof(Request) );
   fConn->SetSID(Request.header.streamid);
   Request.proof.requestid = kXP_urgent;
   Request.proof.sid = fSessionID;
   Request.proof.int1 = type;    // type of urgent msg (see TXSocket::EUrgentMsgType)
   Request.proof.int2 = int1;    // 4-byte container info 1
   Request.proof.int3 = int2;    // 4-byte container info 2
   Request.proof.dlen = 0;

   // Send request
   XrdClientMessage *xrsp =
      fConn->SendReq(&Request, (const void *)0, 0, "SendUrgent");
   if (xrsp) {
      // Success: update usage timestamp
      Touch();
      // Cleanup
      SafeDelete(xrsp);
   } else {
      // Print error msg, if any
      if (fConn->GetLastErr())
         Printf("%s: %s", fHost.Data(), fConn->GetLastErr());
   }

   // Done
   return;
}

////////////////////////////////////////////////////////////////////////////////

Int_t TXSocket::GetLowSocket() const {
   return (fConn ? fConn->GetLowSocket() : -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Init environment variables for XrdClient

void TXSocket::InitEnvs()
{
   // Set debug level
   Int_t deb = gEnv->GetValue("XProof.Debug", -1);
   EnvPutInt(NAME_DEBUG, deb);
   if (deb > 0) {
      XrdProofdTrace->What |= TRACE_REQ;
      if (deb > 1) {
         XrdProofdTrace->What |= TRACE_DBG;
         if (deb > 2)
            XrdProofdTrace->What |= TRACE_ALL;
      }
   }
   const char *cenv = 0;

   // List of domains where connection is allowed
   TString allowCO = gEnv->GetValue("XProof.ConnectDomainAllowRE", "");
   if (allowCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINALLOW_RE, allowCO.Data());

   // List of domains where connection is denied
   TString denyCO  = gEnv->GetValue("XProof.ConnectDomainDenyRE", "");
   if (denyCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINDENY_RE, denyCO.Data());

   // Max number of retries on first connect and related timeout
   XrdProofConn::SetRetryParam(-1, -1);
   Int_t maxRetries = gEnv->GetValue("XProof.FirstConnectMaxCnt",5);
   EnvPutInt(NAME_FIRSTCONNECTMAXCNT, maxRetries);
   Int_t connTO = gEnv->GetValue("XProof.ConnectTimeout", 2);
   EnvPutInt(NAME_CONNECTTIMEOUT, connTO);

   // Reconnect Wait
   Int_t recoTO = gEnv->GetValue("XProof.ReconnectWait",
                                  DFLT_RECONNECTWAIT);
   if (recoTO == DFLT_RECONNECTWAIT) {
      // Check also the old variable name
      recoTO = gEnv->GetValue("XProof.ReconnectTimeout",
                                  DFLT_RECONNECTWAIT);
   }
   EnvPutInt(NAME_RECONNECTWAIT, recoTO);

   // Request Timeout
   Int_t requTO = gEnv->GetValue("XProof.RequestTimeout", 150);
   EnvPutInt(NAME_REQUESTTIMEOUT, requTO);

   // No automatic proofd backward-compatibility
   EnvPutInt(NAME_KEEPSOCKOPENIFNOTXRD, 0);

   // Dynamic forwarding (SOCKS4)
   TString socks4Host = gEnv->GetValue("XNet.SOCKS4Host","");
   Int_t socks4Port = gEnv->GetValue("XNet.SOCKS4Port",-1);
   if (socks4Port > 0) {
      if (socks4Host.IsNull())
         // Default
         socks4Host = "127.0.0.1";
      EnvPutString(NAME_SOCKS4HOST, socks4Host.Data());
      EnvPutInt(NAME_SOCKS4PORT, socks4Port);
   }

   // For password-based authentication
   TString autolog = gEnv->GetValue("XSec.Pwd.AutoLogin","1");
   if (autolog.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecPWDAUTOLOG")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecPWDAUTOLOG",autolog.Data());

   // For password-based authentication
   TString netrc;
   netrc.Form("%s/.rootnetrc",gSystem->HomeDirectory());
   gSystem->Setenv("XrdSecNETRC", netrc.Data());

   TString alogfile = gEnv->GetValue("XSec.Pwd.ALogFile","");
   if (alogfile.Length() > 0)
      gSystem->Setenv("XrdSecPWDALOGFILE",alogfile.Data());

   TString verisrv = gEnv->GetValue("XSec.Pwd.VerifySrv","1");
   if (verisrv.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecPWDVERIFYSRV")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecPWDVERIFYSRV",verisrv.Data());

   TString srvpuk = gEnv->GetValue("XSec.Pwd.ServerPuk","");
   if (srvpuk.Length() > 0)
      gSystem->Setenv("XrdSecPWDSRVPUK",srvpuk.Data());

   // For GSI authentication
   TString cadir = gEnv->GetValue("XSec.GSI.CAdir","");
   if (cadir.Length() > 0)
      gSystem->Setenv("XrdSecGSICADIR",cadir.Data());

   TString crldir = gEnv->GetValue("XSec.GSI.CRLdir","");
   if (crldir.Length() > 0)
      gSystem->Setenv("XrdSecGSICRLDIR",crldir.Data());

   TString crlext = gEnv->GetValue("XSec.GSI.CRLextension","");
   if (crlext.Length() > 0)
      gSystem->Setenv("XrdSecGSICRLEXT",crlext.Data());

   TString ucert = gEnv->GetValue("XSec.GSI.UserCert","");
   if (ucert.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERCERT",ucert.Data());

   TString ukey = gEnv->GetValue("XSec.GSI.UserKey","");
   if (ukey.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERKEY",ukey.Data());

   TString upxy = gEnv->GetValue("XSec.GSI.UserProxy","");
   if (upxy.Length() > 0)
      gSystem->Setenv("XrdSecGSIUSERPROXY",upxy.Data());

   TString valid = gEnv->GetValue("XSec.GSI.ProxyValid","");
   if (valid.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYVALID",valid.Data());

   TString deplen = gEnv->GetValue("XSec.GSI.ProxyForward","0");
   if (deplen.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSIPROXYDEPLEN")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSIPROXYDEPLEN",deplen.Data());

   TString pxybits = gEnv->GetValue("XSec.GSI.ProxyKeyBits","");
   if (pxybits.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYKEYBITS",pxybits.Data());

   TString crlcheck = gEnv->GetValue("XSec.GSI.CheckCRL","1");
   if (crlcheck.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSICRLCHECK")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSICRLCHECK",crlcheck.Data());

   TString delegpxy = gEnv->GetValue("XSec.GSI.DelegProxy","0");
   if (delegpxy.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSIDELEGPROXY")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSIDELEGPROXY",delegpxy.Data());

   TString signpxy = gEnv->GetValue("XSec.GSI.SignProxy","1");
   if (signpxy.Length() > 0 &&
      (!(cenv = gSystem->Getenv("XrdSecGSISIGNPROXY")) || strlen(cenv) <= 0))
      gSystem->Setenv("XrdSecGSISIGNPROXY",signpxy.Data());

   // Print the tag, if required (only once)
   if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
      ::Info("TXSocket","(C) 2005 CERN TXSocket (XPROOF client) %s",
            gROOT->GetVersion());

   // Only once
   fgInitDone = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Try reconnection after failure

Int_t TXSocket::Reconnect()
{
   if (gDebug > 0) {
      Info("Reconnect", "%p (c:%p, v:%d): trying to reconnect to %s (logid: %d)",
                        this, fConn, (fConn ? fConn->IsValid() : 0),
                        fUrl.Data(), (fConn ? fConn->GetLogConnID() : -1));
   }

   Int_t tryreconnect = gEnv->GetValue("TXSocket.Reconnect", 0);
   if (tryreconnect == 0 || fXrdProofdVersion < 1005) {
      if (tryreconnect == 0)
         Info("Reconnect","%p: reconnection attempts explicitly disabled!", this);
      else
         Info("Reconnect","%p: server does not support reconnections (protocol: %d < 1005)",
                          this, fXrdProofdVersion);
      return -1;
   }

   if (fConn) {
      if (gDebug > 0)
         Info("Reconnect", "%p: locking phyconn: %p", this, fConn->fPhyConn);
      fConn->ReConnect();
      if (fConn->IsValid()) {
         // Create new proofserv if not client manager or administrator or internal mode
         if (fMode == 'm' || fMode == 's' || fMode == 'M' || fMode == 'A') {
            // We attach or create
            if (!Create(kTRUE)) {
               // Failure
               Error("TXSocket", "create or attach failed (%s)",
                     ((fConn->fLastErrMsg.length() > 0) ? fConn->fLastErrMsg.c_str() : "-"));
               Close();
               return -1;
            }
         }
      }
   }

   if (gDebug > 0) {
      if (fConn) {
         Info("Reconnect", "%p (c:%p): attempt %s (logid: %d)", this, fConn,
                           (fConn->IsValid() ? "succeeded!" : "failed"),
                           fConn->GetLogConnID() );
      } else {
         Info("Reconnect", "%p (c:0x0): attempt failed", this);
      }
   }

   // Done
   return ((fConn && fConn->IsValid()) ? 0 : -1);
}

////////////////////////////////////////////////////////////////////////////////
///constructor

TXSockBuf::TXSockBuf(Char_t *bp, Int_t sz, Bool_t own)
{
   fBuf = fMem = bp;
   fSiz = fLen = sz;
   fOwn = own;
   fCid = -1;
   fgBuffMem += sz;
}

////////////////////////////////////////////////////////////////////////////////
///destructor

TXSockBuf::~TXSockBuf()
{
   if (fOwn && fMem) {
      free(fMem);
      fgBuffMem -= fSiz;
   }
}

////////////////////////////////////////////////////////////////////////////////
///resize socket buffer

void TXSockBuf::Resize(Int_t sz)
{
   if (sz > fSiz) {
      if ((fMem = (Char_t *)realloc(fMem, sz))) {
         fgBuffMem += (sz - fSiz);
         fBuf = fMem;
         fSiz = sz;
         fLen = 0;
      }
   }
}

//_____________________________________________________________________________
//
// TXSockBuf static methods
//

////////////////////////////////////////////////////////////////////////////////
/// Return the currently allocated memory

Long64_t TXSockBuf::BuffMem()
{
   return fgBuffMem;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the max allocated memory allowed

Long64_t TXSockBuf::GetMemMax()
{
   return fgMemMax;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the max allocated memory allowed

void TXSockBuf::SetMemMax(Long64_t memmax)
{
   fgMemMax = memmax > 0 ? memmax : fgMemMax;
}

//_____________________________________________________________________________
//
// TXSockPipe
//

////////////////////////////////////////////////////////////////////////////////
/// Constructor

TXSockPipe::TXSockPipe(const char *loc) : fLoc(loc)
{
   // Create the pipe
   if (pipe(fPipe) != 0) {
      Printf("TXSockPipe: problem initializing pipe for socket inputs");
      fPipe[0] = -1;
      fPipe[1] = -1;
      return;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

TXSockPipe::~TXSockPipe()
{
   if (fPipe[0] >= 0) close(fPipe[0]);
   if (fPipe[1] >= 0) close(fPipe[1]);
}


////////////////////////////////////////////////////////////////////////////////
/// Write a byte to the global pipe to signal new availibility of
/// new messages

Int_t TXSockPipe::Post(TSocket *s)
{
   if (!IsValid() || !s) return -1;

   // This must be an atomic action
   Int_t sz = 0;
   {  std::lock_guard<std::recursive_mutex> lock(fMutex);
      // Add this one
      fReadySock.Add(s);

      // Only one char
      Char_t c = 1;
      if (write(fPipe[1],(const void *)&c, sizeof(Char_t)) < 1) {
         Printf("TXSockPipe::Post: %s: can't notify pipe", fLoc.Data());
         return -1;
      }
      if (gDebug > 2) sz = fReadySock.GetSize();
   }

   if (gDebug > 2)
      Printf("TXSockPipe::Post: %s: %p: pipe posted (pending %d) (descriptor: %d)",
                               fLoc.Data(), s, sz, fPipe[1]);
   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read a byte to the global pipe to synchronize message pickup

Int_t TXSockPipe::Clean(TSocket *s)
{
   // Pipe must have been created
   if (!IsValid() || !s) return -1;

   // Only one char
   Int_t sz = 0;
   Char_t c = 0;
   {  std::lock_guard<std::recursive_mutex> lock(fMutex);
      if (read(fPipe[0],(void *)&c, sizeof(Char_t)) < 1) {
         Printf("TXSockPipe::Clean: %s: can't read from pipe", fLoc.Data());
         return -1;
      }
      // Remove this one
      fReadySock.Remove(s);

      if (gDebug > 2) sz = fReadySock.GetSize();
   }

   if (gDebug > 2)
      Printf("TXSockPipe::Clean: %s: %p: pipe cleaned (pending %d) (descriptor: %d)",
                               fLoc.Data(), s, sz, fPipe[0]);

   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove any reference to socket 's' from the global pipe and
/// ready-socket queue

Int_t TXSockPipe::Flush(TSocket *s)
{
   // Pipe must have been created
   if (!IsValid() || !s) return -1;

   TObject *o = 0;
   // This must be an atomic action
   {  std::lock_guard<std::recursive_mutex> lock(fMutex);
      o = fReadySock.FindObject(s);

      while (o) {
         // Remove from the list
         fReadySock.Remove(s);
         o = fReadySock.FindObject(s);
         // Remove one notification from the pipe
         Char_t c = 0;
         if (read(fPipe[0],(void *)&c, sizeof(Char_t)) < 1)
            Printf("TXSockPipe::Flush: %s: can't read from pipe", fLoc.Data());
      }
   }
   // Flush also the socket
   ((TXSocket *)s)->Flush();

   // Notify
   if (gDebug > 0)
      Printf("TXSockPipe::Flush: %s: %p: pipe flushed", fLoc.Data(), s);

   // We are done
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Dump content of the ready socket list

void TXSockPipe::DumpReadySock()
{
   std::lock_guard<std::recursive_mutex> lock(fMutex);

   TString buf = Form("%d |", fReadySock.GetSize());
   TIter nxs(&fReadySock);
   TObject *o = 0;
   while ((o = nxs()))
      buf += Form(" %p",o);
   Printf("TXSockPipe::DumpReadySock: %s: list content: %s", fLoc.Data(), buf.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Return last ready socket

TXSocket *TXSockPipe::GetLastReady()
{
   std::lock_guard<std::recursive_mutex> lock(fMutex);

   return (TXSocket *) fReadySock.Last();
}
