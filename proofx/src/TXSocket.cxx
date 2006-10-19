// @(#)root/proofx:$Name:  $:$Id: TXSocket.cxx,v 1.20 2006/10/06 09:14:58 rdm Exp $
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
// TXSocket                                                             //
//                                                                      //
// High level handler of connections to xproofd.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

// ---- Tracing utils ----------------------------------------------------------
#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucLogger.hh"
#include "XrdProofdTrace.h"
XrdOucTrace *XrdProofdTrace = 0;
static XrdOucLogger eLogger;
static XrdOucError eDest(0, "Proofx");

#ifdef WIN32
ULong64_t TSocket::fgBytesSent;
ULong64_t TSocket::fgBytesRecv;
#endif

//______________________________________________________________________________

//---- error handling ----------------------------------------------------------

//______________________________________________________________________________
void TXSocket::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   // Interface to ErrorHandler (protected).

   ::ErrorHandler(level, Form("TXSocket::%s", location), fmt, va);
}

//----- Ping handler -----------------------------------------------------------
//______________________________________________________________________________
class TXSocketPingHandler : public TFileHandler {
   TXSocket  *fSocket;
public:
   TXSocketPingHandler(TXSocket *s, Int_t fd)
      : TFileHandler(fd, 1) { fSocket = s; }
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
};

//______________________________________________________________________________
Bool_t TXSocketPingHandler::Notify()
{
   // Ping the socket
   fSocket->Ping(kTRUE);

   return kTRUE;
}

// Env variables init flag
Bool_t TXSocket::fgInitDone = kFALSE;

// Static variables for input notification
TList        TXSocket::fgReadySock;          // Static list of sockets ready to be read
TMutex       TXSocket::fgReadyMtx(kTRUE);    // Protect access to the sockets-ready list
Int_t        TXSocket::fgPipe[2] = {-1,-1};  // Pipe for input monitoring
TString      TXSocket::fgLoc = "undef";      // Location string

// Static buffer manager
TMutex       TXSocket::fgSMtx;               // To protect spare list
std::list<TXSockBuf *> TXSocket::fgSQue;     // list of spare buffers
Long64_t     TXSockBuf::fgBuffMem = 0;       // Total allocated memory
Long64_t     TXSockBuf::fgMemMax = 10485760; // Max allowed allocated memory [10 MB]

//_____________________________________________________________________________
TXSocket::TXSocket(const char *url, Char_t m, Int_t psid,
                   Char_t capver, const char *logbuf, Int_t loglevel)
         : TSocket(), fMode(m), fLogLevel(loglevel),
           fBuffer(logbuf), fASem(0), fDontTimeout(kFALSE)
{
   // Constructor
   // Open the connection to a remote XrdProofd instance and start a PROOF
   // session.
   // The mode 'm' indicates the role of this connection:
   //     'a'      Administrator; used by an XPD to contact the head XPD
   //     'i'      Internal; used by a TXProofServ to call back its creator
   //              (see XrdProofUnixConn)
   //     'C'      PROOF manager: open connection only (do not start a session)
   //     'M'      Client creating a top master
   //     'A'      Client attaching to top master
   //     'm'      Top master creating a submaster
   //     's'      Master creating a slave
   // The buffer 'logbuf' is a null terminated string to be sent over at
   // login.

   // Enable tracing in the XrdProof client. if not done already
   eDest.logger(&eLogger);
   if (!XrdProofdTrace)
      XrdProofdTrace = new XrdOucTrace(&eDest);

   // Init envs the first time
   if (!fgInitDone)
      InitEnvs();

   // Async queue related stuff
   if (!(fAMtx = new TMutex(kTRUE))) {
      Error("TXSocket", "problems initializing mutex for async queue");
      return;
   }
   fAQue.clear();

   // Interrupts queue related stuff
   if (!(fIMtx = new TMutex(kTRUE))) {
      Error("TXSocket", "problems initializing mutex for interrupts");
      return;
   }
   fILev = -1;

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
   fHandler = 0;

   // The global pipe
   if (fgPipe[0] == -1) {
      if (pipe(fgPipe) != 0) {
          Error("TXSocket", "problem initializing global pipe for socket inputs");
          return;
      }
   }

   if (url) {

      // Create connection (for managers the type of the connection is the same
      // as for top masters)
      char md = (m != 'A' && m != 'C') ? m : 'M';
      fConn = new XrdProofConn(url, md, psid, capver, this, fBuffer.Data());
      if (!fConn || !(fConn->IsValid())) {
         if (fConn->GetServType() != XrdProofConn::kSTProofd)
             Error("TXSocket", "severe error occurred while opening a connection"
                   " to server [%s]: %s", url, fConn->GetLastErr());
         return;
      }

      // Create new proofserv if not client manager or administrator or internal mode
      if (m == 'm' || m == 's' || m == 'M' || m == 'A') {
         // We attach or create
         if (!Create()) {
            // Failure
            Error("TXSocket", "create or attach failed (%s)",
                  ((fConn->fLastErrMsg.length() > 0) ? fConn->fLastErrMsg.c_str() : "-"));
            Close();
            return;
         }
      }

      // Fill some info
      fUser = fConn->fUser.c_str();
      fHost = fConn->fHost.c_str();
      fPort = fConn->fPort;

      // Also in the base class
      fUrl = fConn->fUrl.GetUrl().c_str();
      fAddress.fPort = fPort;

      // This is needed for the reader thread to signal an interrupt
      fPid = gSystem->GetPid();
   }
}

//______________________________________________________________________________
TXSocket::TXSocket(const TXSocket &s) : TSocket(s),XrdClientAbsUnsolMsgHandler(s)
{
   // TXSocket copy ctor.
}

//______________________________________________________________________________
TXSocket& TXSocket::operator=(const TXSocket&)
{
   // TXSocket assignment operator.
   return *this;
}

//_____________________________________________________________________________
TXSocket::~TXSocket()
{
   // Destructor

   // Disconnect from remote server (the connection manager is
   // responsible of the underlying physical connection, so we do not
   // force its closing)
   Close();

   // Delete mutexes
   SafeDelete(fAMtx);
   SafeDelete(fIMtx);
}

//_____________________________________________________________________________
void TXSocket::DisconnectSession(Int_t id, Option_t *opt)
{
   // Disconnect a session. Use opt= "S" or "s" to
   // shutdown remote session.
   // Default is opt = "".

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

      // Cleanup
      SafeDelete(xrsp);
   }
}

//_____________________________________________________________________________
void TXSocket::Close(Option_t *opt)
{
   // Close connection. Available options are (case insensitive)
   //   'P'   force closing of the underlying physical connection
   //   'S'   shutdown remote session, is any
   // A session ID can be given using #...# signature, e.g. "#1#".
   // Default is opt = "".

   // Remove any reference in the glovbal pipe and ready-sock queue
   TXSocket::FlushPipe(this);

   // Make sure we are connected
   if (!IsValid()) {
      if (gDebug > 0)
         Info("Close","not connected: nothing to do");
      return;
   }

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

   // Delete the connection module
   SafeDelete(fConn);
}

//_____________________________________________________________________________
UnsolRespProcResult TXSocket::ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *,
                                          XrdClientMessage *m)
{
   // We are here if an unsolicited response comes from a logical conn
   // The response comes in the form of an XrdClientMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited
   // responses are asynchronous by nature.
   UnsolRespProcResult rc = kUNSOL_KEEP;

   if (gDebug > 2)
      Info("ProcessUnsolicitedMsg", "Processing unsolicited msg: %p", m);
   if (!m) {
      // Some one is perhaps interested in empty messages
      return kUNSOL_CONTINUE;
   } else {
      if (gDebug > 2)
         Info("ProcessUnsolicitedMsg", "status: %d, len: %d bytes",
              m->GetStatusCode(), m->DataLen());
   }

   // Error notification
   if (m->IsError()) {
      if (m->GetStatusCode() != XrdClientMessage::kXrdMSC_timeout) {
         if (gDebug > 0)
            Info("ProcessUnsolicitedMsg","got error from underlying connection");
         if (fHandler)
            fHandler->HandleError();
         else
            Error("ProcessUnsolicitedMsg","handler undefined");
         // Avoid to contact the server any more
         fSessionID = -1;
      } else {
         // Time out
         if (gDebug > 2)
            Info("ProcessUnsolicitedMsg", "underlying connection timed out");
      }
      // Propagate the message to other possible handlers
      return kUNSOL_CONTINUE;
   }

   // From now on make sure is for us
   if (!fConn || !m->MatchStreamid(fConn->fStreamid))
      return kUNSOL_CONTINUE;


   // Local processing ...
   if (!m) {
      Error("ProcessUnsolicitedMsg","undefined message");
      return rc;
   }

   Int_t len = 0;
   if ((len = m->DataLen()) < (int)sizeof(kXR_int32)) {
      Error("ProcessUnsolicitedMsg","empty or bad-formed message");
      return rc;
   }

   // The first 4 bytes contain the action code
   kXR_int32 acod = 0;
   memcpy(&acod, m->GetData(), sizeof(kXR_int32));
   //
   // Update pointer to data
   void *pdata = (void *)((char *)(m->GetData()) + sizeof(kXR_int32));
   len -= sizeof(kXR_int32);
   if (gDebug > 1)
      Info("ProcessUnsolicitedMsg", "%p: got action: %d (%d bytes) (ID: %d)",
           this, acod, len, m->HeaderSID());

   if (gDebug > 3)
      DumpReadySock();

   // Case by case
   kXR_int32 ilev = -1;

   switch (acod) {
      case kXPD_ping:
         //
         // Special interrupt
         ilev = TProof::kPing;
      case kXPD_interrupt:
         //
         // Interrupt
         { R__LOCKGUARD(fIMtx);
            if (acod == kXPD_interrupt) {
               memcpy(&ilev, pdata, sizeof(kXR_int32));
               ilev = net2host(ilev);
            }
            //
            // Save the interrupt
            fILev = ilev;

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
         { R__LOCKGUARD(fAMtx);

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
            PostPipe(this);

            // Signal it and release the mutex
            if (gDebug > 2)
               Info("ProcessUnsolicitedMsg","%p: posting semaphore: %p (%d bytes)",
                    this,&fASem,len);
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
         Printf(" ");
         Printf("| Message from server:");
         Printf("| %.*s", len, (char *)pdata);
         break;
      case kXPD_errmsg:
         //
         // Error condition with message
         Printf(" ");
         Printf("| Error condition occured: message from server:");
         Printf("| %.*s", len, (char *)pdata);
         // Handle error
         if (fHandler)
            fHandler->HandleError();
         else
            Error("ProcessUnsolicitedMsg","handler undefined");
         break;
      case kXPD_msgsid:
         //
         // Data message
         { R__LOCKGUARD(fAMtx);

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
            PostPipe(this);

            // Signal it and release the mutex
            if (gDebug > 2)
               Info("ProcessUnsolicitedMsg","%p: cid: %d, posting semaphore: %p (%d bytes)",
                    this, cid, &fASem, len);
            fASem.Post();
         }

         break;
     default:
         Error("ProcessUnsolicitedMsg","unknown action code: %d", acod);
   }

   // We are done
   return rc;
}

//_______________________________________________________________________
Int_t TXSocket::GetPipeRead()
{
   // Get read descriptor of the global pipe used for monitoring of the
   // XPD sockets

   // Create the pipe, if not already done
   if (fgPipe[0] == -1) {
      if (pipe(fgPipe) != 0) {
         fgPipe[0] = -1;
         ::Error("TXSocket::GetPipeRead", "error: errno: %d", errno);
      }
   }
   return fgPipe[0];
}

//____________________________________________________________________________
Int_t TXSocket::PostPipe(TSocket *s)
{
   // Write a byte to the global pipe to signal new availibility of
   // new messages

   // This must be an atomic action
   { R__LOCKGUARD(&TXSocket::fgReadyMtx);

     // Add this one
     TXSocket::fgReadySock.Add(s);
   }

   // Pipe must have been created
   if (fgPipe[1] < 0)
      return -1;

   // Only one char
   Char_t c = 1;
   if (write(fgPipe[1],(const void *)&c, sizeof(Char_t)) < 1) {
      ::Error("TXSocket::PostPipe", "can't notify pipe");
      return -1;
   }

   if (gDebug > 2)
      ::Info("TXSocket::PostPipe", "%s: %p: pipe posted", fgLoc.Data(), s);

   // We are done
   return 0;
}

//____________________________________________________________________________
Int_t TXSocket::CleanPipe(TSocket *s)
{
   // Read a byte to the global pipe to synchronize message pickup

   // Pipe must have been created
   if (fgPipe[0] < 0)
      return -1;

   // Only one char
   Char_t c = 0;
   if (read(fgPipe[0],(void *)&c, sizeof(Char_t)) < 1) {
      ::Error("TXSocket::CleanPipe", "%s: can't read from pipe", fgLoc.Data());
      return -1;
   }

   // This must be an atomic action
   R__LOCKGUARD(&TXSocket::fgReadyMtx);

   // Remove this one
   TXSocket::fgReadySock.Remove(s);

   if (gDebug > 2)
      ::Info("TXSocket::CleanPipe", "%s: %p: pipe cleaned", fgLoc.Data(), s);

   // We are done
   return 0;
}

//____________________________________________________________________________
Int_t TXSocket::FlushPipe(TSocket *s)
{
   // Remove any reference to socket 's' from the global pipe and
   // ready-socket queue

   // Pipe must have been created
   if (fgPipe[0] < 0)
      return -1;

   // This must be an atomic action
   R__LOCKGUARD(&TXSocket::fgReadyMtx);

   while (TXSocket::fgReadySock.FindObject(s)) {
      // Remove from the list
      TXSocket::fgReadySock.Remove(s);
      // Remove one notification from the pipe
      Char_t c = 0;
      if (read(fgPipe[0],(void *)&c, sizeof(Char_t)) < 1)
         ::Warning("TXSocket::FlushPipe", "%s: can't read from pipe", fgLoc.Data());
   }

   // Notify
   if (gDebug > 0)
      ::Info("TXSocket::ResetPipe", "%s: %p: pipe flushed", fgLoc.Data(), s);

   // We are done
   return 0;
}

//____________________________________________________________________________
Bool_t TXSocket::IsServProofd()
{
   // Return kTRUE if the remote server is a 'proofd'

   if (fConn && (fConn->GetServType() == XrdProofConn::kSTProofd))
      return kTRUE;

   // Failure
   return kFALSE;
}

//_____________________________________________________________________________
Int_t TXSocket::GetInterrupt()
{
   // Get highest interrupt level in the queue

   if (gDebug > 2)
      Info("GetInterrupt","%p: waiting to lock mutex %p", fIMtx);

   R__LOCKGUARD(fIMtx);

   if (fILev == -1)
      Error("GetInterrupt","value is unset (%d) - protocol error",fILev);

   // Return what we got
   return fILev;
}

//_____________________________________________________________________________
Int_t TXSocket::Flush()
{
   // Flush the asynchronous queue.
   // Typically called when a kHardInterrupt is received.
   // Returns number of bytes in flushed buffers.

   R__LOCKGUARD(fAMtx);

   // Must have something to flush
   Int_t nf = 0;
   if (fAQue.size() > 0) {

      // Save size for later semaphore cleanup
      Int_t sz = fAQue.size();
      // get the highest interrupt level
      list<TXSockBuf *>::iterator i;
      for (i = fAQue.begin(); i != fAQue.end(); i++) {
         if (*i) {
            {  R__LOCKGUARD(&fgSMtx);
               fgSQue.push_back(*i);
            }
            fAQue.erase(i);
            nf += (*i)->fLen;
         }
      }

      // Reset the asynchronous queue
      while (sz--)
         fASem.TryWait();
      fAQue.clear();
   }

   // We are done
   return nf;
}

//_____________________________________________________________________________
Bool_t TXSocket::Create()
{
   // This method sends a request for creation of (or attachment to) a remote
   // server application.

   // Make sure we are connected
   if (!IsValid()) {
      if (gDebug > 0)
         Info("Create","not connected: nothing to do");
      return kFALSE;
   }

   XPClientRequest reqhdr;

   // We fill the header struct containing the request for login
   memset( &reqhdr, 0, sizeof(reqhdr));
   fConn->SetSID(reqhdr.header.streamid);

   // This will be a kXP_attach or kXP_create request
   if (fMode == 'A') {
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
   struct ServerResponseBody_Protocol *srvresp = 0;
   XrdClientMessage *xrsp = fConn->SendReq(&reqhdr, buf,
                                          (void **)&srvresp, "TXSocket::Create");

   // In any, the URL the data pool entry point will be stored here
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
      }

      if (len >= (Int_t)sizeof(kXR_int32)) {
         // The second 4 bytes contain the remote protocol version
         kXR_int32 dver = 0;
         memcpy(&dver, pdata, sizeof(kXR_int32));
         fRemoteProtocol = net2host(dver);
         pdata = (void *)((char *)pdata + sizeof(kXR_int32));
         len -= sizeof(kXR_int32);
      } else {
         Warning("Create","protocol version of the remote daemon undefined!");
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
      if (srvresp)
         free(srvresp);

      // Notify
      return kTRUE;

   }

   // Notify failure
   Error("Create:",
         "problems creating or attaching to a remote server (%s)",
         ((fConn->fLastErrMsg.length() > 0) ? fConn->fLastErrMsg.c_str() : "-"));
   return kFALSE;
}

//______________________________________________________________________________
Int_t TXSocket::SendRaw(const void *buffer, Int_t length, ESendRecvOptions opt)
{
   // Send a raw buffer of specified length.
   // Use opt = kDontBlock to ask xproofd to push the message into the proofsrv.
   // (by default is appended to a queue waiting for a request from proofsrv).
   // Returns the number of bytes sent or -1 in case of error.

   TSystem::ResetErrno();

   // Make sure we are connected
   if (!IsValid()) {
      Error("SendRaw","not connected: nothing to do");
      return -1;
   }

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
      // ok
      return nsent;
   }

   // Failure notification (avoid using the handler: we may be exiting)
   Error("SendRaw", "problems sending data to server");

   return -1;
}

//______________________________________________________________________________
Bool_t TXSocket::Ping(Bool_t)
{
   // Ping functionality: contact the server and get an acknowledgement.
   // If external, the server waits for a reply from the server
   // Use opt = kDontBlock to ask xproofd to push the message into the proofsrv.
   // (by default is appended to a queue waiting for a request from proofsrv).
   // Returns the number of bytes sent or -1 in case of error.

   TSystem::ResetErrno();

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
   kXR_int32 *pres = 0;
   XrdClientMessage *xrsp =
      fConn->SendReq(&Request, (const void *)0, (void **)&pres, "Ping");

   // Get the result
   Bool_t res = kFALSE;
   if (xrsp && xrsp->HeaderStatus() == kXR_ok) {
      *pres = net2host(*pres);
      res = (*pres == 1);
   }

   // Cleanup
   SafeDelete(xrsp);

   // Failure notification (avoid using the handler: we may be exiting)
   Error("Ping", "problems sending ping to server");

   return res;
}

//______________________________________________________________________________
Int_t TXSocket::PickUpReady()
{
   // Wait and pick-up next buffer from the asynchronous queue

   fBufCur = 0;
   fByteLeft = 0;
   fByteCur = 0;
   if (gDebug > 2)
      Info("PickUpReady","%p: going to sleep", this);

   // User can choose whether to wait forever or for a fixed amount of time
   if (!fDontTimeout) {
      static Int_t timeout = gEnv->GetValue("XProof.ReadTimeout", 60) * 1000;
      static Int_t dt = 2000;
      Int_t to = timeout;
      while (to) {
         if (fASem.Wait(dt) != 0) {
            to -= dt;
            if (to <= 0) {
               Error("PickUpReady","error waiting at semaphore");
               return -1;
            } else {
               if (gDebug > 0)
                  Info("PickUpReady","%p: got timeout: retring (%d secs)", this, to/1000);
            }
         } else
            break;
      }
   } else {
      // We wait forever
      if (fASem.Wait() != 0) {
         Error("PickUpReady","error waiting at semaphore");
         return -1;
      }
   }
   if (gDebug > 2)
      Info("PickUpReady","%p: waken up", this);

   R__LOCKGUARD(fAMtx);

   // Get message, if any
   if (fAQue.size() <= 0) {
      Error("PickUpReady","queue is empty - protocol error ?");
      return -1;
   }
   fBufCur = fAQue.front();
   // Remove message from the queue
   fAQue.pop_front();
   // Set number of available bytes
   if (fBufCur)
      fByteLeft = fBufCur->fLen;

   if (gDebug > 2)
      Info("PickUpReady","%p: got message (%d bytes)", this, (Int_t)(fBufCur ? fBufCur->fLen : 0));

   // Update counters
   fBytesRecv += fBufCur->fLen;

   // Set session ID
   if (fBufCur->fCid > -1 && fBufCur->fCid != GetClientID())
      SetClientID(fBufCur->fCid);

   // Clean entry in the underlying pipe
   CleanPipe(this);

   // We are done
   return 0;
}

//______________________________________________________________________________
TXSockBuf *TXSocket::PopUpSpare(Int_t size)
{
   // Pop-up a buffer of at least size bytes from the spare list
   // If none is found either one is reallocated or a new one
   // created
   TXSockBuf *buf = 0;
   static Int_t nBuf = 0;


   R__LOCKGUARD(&fgSMtx);


   Int_t maxsz = 0;
   if (fgSQue.size() > 0) {
      list<TXSockBuf *>::iterator i;
      for (i = fgSQue.begin(); i != fgSQue.end(); i++) {
         maxsz = ((*i)->fSiz > maxsz) ? (*i)->fSiz : maxsz;
         if ((*i) && (*i)->fSiz >= size) {
            buf = *i;
            if (gDebug > 2)
               Info("PopUpSpare","asked: %d, spare: %d/%d, REUSE buf %p, sz: %d",
                                 size, fgSQue.size(), nBuf, buf, buf->fSiz);
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
                           size, fgSQue.size(), nBuf, maxsz, buf, buf->fSiz);
      // Drop from this list
      fgSQue.pop_front();
      return buf;
   }

   // Create a new buffer
   char *b = (char *)malloc(size);
   if (b)
      buf = new TXSockBuf(b, size);
   nBuf++;
   if (gDebug > 2)
      Info("PopUpSpare","asked: %d, spare: %d/%d, maxsz: %d, NEW buf %p, sz: %d",
                        size, fgSQue.size(), nBuf, maxsz, buf, buf->fSiz);

   // We are done
   return buf;
}

//______________________________________________________________________________
void TXSocket::PushBackSpare()
{
   // Release read buffer giving back to the spare list

   R__LOCKGUARD(&fgSMtx);

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

//______________________________________________________________________________
Int_t TXSocket::RecvRaw(void *buffer, Int_t length, ESendRecvOptions)
{
   // Receive a raw buffer of specified length bytes.

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

   return length;
}

//______________________________________________________________________________
Int_t TXSocket::SendInterrupt(Int_t type)
{
   // Send urgent message (interrupt) to remote server
   // Returns 0 or -1 in case of error.

   TSystem::ResetErrno();

   // Make sure we are connected
   if (!IsValid()) {
      Error("SendInterrupt","not connected: nothing to do");
      return -1;
   }

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
      // Cleanup
      SafeDelete(xrsp);
      // ok
      return 0;
   }

   // Failure notification (avoid using the handler: we may be exiting)
   Error("SendInterrupt", "problems sending interrupt to server");
   return -1;
}

//______________________________________________________________________________
Int_t TXSocket::Send(const TMessage &mess)
{
   // Send a TMessage object. Returns the number of bytes in the TMessage
   // that were sent and -1 in case of error.

   TSystem::ResetErrno();

   if (!IsValid()) {
      Error("Send","not connected: nothing to do");
      return -1;
   }

   if (mess.IsReading()) {
      Error("Send", "cannot send a message used for reading");
      return -1;
   }

   mess.SetLength();   //write length in first word of buffer

   if (fCompress > 0 && mess.GetCompressionLevel() == 0)
      const_cast<TMessage&>(mess).SetCompressionLevel(fCompress);

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

   Int_t nsent = SendRaw(mbuf, mlen);
   fSendOpt = fSendOptDefault;

   if (nsent <= 0)
      return nsent;

   fBytesSent  += nsent;
   fgBytesSent += nsent;

   return nsent - sizeof(UInt_t);  //length - length header
}

//______________________________________________________________________________
Int_t TXSocket::Recv(TMessage *&mess)
{
   // Receive a TMessage object. The user must delete the TMessage object.
   // Returns length of message in bytes (can be 0 if other side of connection
   // is closed) or -1 in case of error. In those case mess == 0.

   TSystem::ResetErrno();

   if (!IsValid()) {
      mess = 0;
      return -1;
   }

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

   return n;
}

//______________________________________________________________________________
TObjString *TXSocket::SendCoordinator(Int_t kind, const char *msg)
{
   // Send message to intermediate coordinator.
   // If any output is due, this is returned as an obj string to be
   // deleted by the caller

   TObjString *sout = 0;

   // We fill the header struct containing the request
   XPClientRequest reqhdr;
   const void *buf = 0;
   char *bout = 0;
   void **vout = 0;
   memset(&reqhdr, 0, sizeof(reqhdr));
   fConn->SetSID(reqhdr.header.streamid);
   reqhdr.header.requestid = kXP_admin;
   reqhdr.proof.int1 = kind;
   switch (kind) {
      case kQuerySessions:
         reqhdr.proof.sid = 0;
         reqhdr.header.dlen = 0;
         vout = (void **)&bout;
         break;
      case kCleanupSessions:
         reqhdr.proof.int2 = (kXR_int32) kXPD_TopMaster;
      case kSessionTag:
      case kSessionAlias:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = (msg) ? strlen(msg) : 0;
         buf = (msg) ? (const void *)msg : buf;
         break;
      case kGetWorkers:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = 0;
         vout = (void **)&bout;
         break;
      case kQueryWorkers:
         reqhdr.proof.sid = 0;
         reqhdr.header.dlen = 0;
         vout = (void **)&bout;
         break;
      default:
         Info("SendCoordinator", "unknown message kind: %d", kind);
         return sout;
   }

   // server response header
   XrdClientMessage *xrsp =
      fConn->SendReq(&reqhdr, buf, vout, "TXSocket::SendCoordinator");

   // If positive answer
   if (xrsp) {

      // Check if we need to create an output string
      if (bout && (xrsp->DataLen() > 0))
         sout = new TObjString(TString(bout,xrsp->DataLen()));

      if (bout)
         free(bout);
      SafeDelete(xrsp);
   }


   // Failure notification (avoid using the handler: we may be exiting)
   return sout;
}

//______________________________________________________________________________
void TXSocket::SendUrgent(Int_t type, Int_t int1, Int_t int2)
{
   // Send urgent message to counterpart; 'type' specifies the type of
   // the message (see TXSocket::EUrgentMsgType), and 'int1', 'int2'
   // two containers for additional information.

   TSystem::ResetErrno();

   // Make sure we are connected
   if (!IsValid()) {
      Error("SendUrgent","not connected: nothing to do");
      return;
   }

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
   if (xrsp)
      // Cleanup
      SafeDelete(xrsp);

   // Done
   return;
}

//______________________________________________________________________________
void TXSocket::DumpReadySock()
{
   // Dump content of the ready socket list

   R__LOCKGUARD(&fgReadyMtx);

   TString buf = Form("%d |", fgReadySock.GetSize());
   TIter nxs(&fgReadySock);
   TObject *o = 0;
   while ((o = nxs()))
      buf += Form(" %p",o);
   ::Info("TXSocket::DumpReadySock", "%s: list content: %s", fgLoc.Data(), buf.Data());

}

//_____________________________________________________________________________
void TXSocket::InitEnvs()
{
   // Init environment variables for XrdClient

   // Set debug level
   EnvPutInt(NAME_DEBUG, gEnv->GetValue("XProof.Debug", 0));
   if (gEnv->GetValue("XProof.Debug", 0) > 0)
      XrdProofdTrace->What = TRACE_REQ;
      if (gEnv->GetValue("XProof.Debug", 0) > 1)
         XrdProofdTrace->What = TRACE_ALL;

   // List of domains where connection is allowed
   TString allowCO = gEnv->GetValue("XProof.ConnectDomainAllowRE", "");
   if (allowCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINALLOW_RE, allowCO.Data());

   // List of domains where connection is denied
   TString denyCO  = gEnv->GetValue("XProof.ConnectDomainDenyRE", "");
   if (denyCO.Length() > 0)
      EnvPutString(NAME_CONNECTDOMAINDENY_RE, denyCO.Data());

   // Connect Timeout
   Int_t connTO = gEnv->GetValue("XProof.ConnectTimeout", 2);
   EnvPutInt(NAME_CONNECTTIMEOUT, connTO);

   // Reconnect Timeout
   Int_t recoTO = gEnv->GetValue("XProof.ReconnectTimeout",
                                  DFLT_RECONNECTTIMEOUT);
   EnvPutInt(NAME_RECONNECTTIMEOUT, recoTO);

   // Request Timeout
   Int_t requTO = gEnv->GetValue("XProof.RequestTimeout", DFLT_REQUESTTIMEOUT);
   EnvPutInt(NAME_REQUESTTIMEOUT, requTO);

   // Whether to use a separate thread for garbage collection
   Int_t garbCollTh = gEnv->GetValue("XProof.StartGarbageCollectorThread",
                                      DFLT_STARTGARBAGECOLLECTORTHREAD);
   EnvPutInt(NAME_STARTGARBAGECOLLECTORTHREAD, garbCollTh);

   // Max number of retries on first connect
   Int_t maxRetries = gEnv->GetValue("XProof.FirstConnectMaxCnt",5);
   EnvPutInt(NAME_FIRSTCONNECTMAXCNT, maxRetries);

   // No automatic proofd backward-compatibility
   EnvPutInt(NAME_KEEPSOCKOPENIFNOTXRD, 0);

   // For password-based authentication
   TString autolog = gEnv->GetValue("XSec.Pwd.AutoLogin","1");
   if (autolog.Length() > 0)
      gSystem->Setenv("XrdSecPWDAUTOLOG",autolog.Data());

   // For password-based authentication
   TString netrc;
   netrc.Form("%s/.rootnetrc",gSystem->HomeDirectory());
   gSystem->Setenv("XrdSecNETRC", netrc.Data());

   TString alogfile = gEnv->GetValue("XSec.Pwd.ALogFile","");
   if (alogfile.Length() > 0)
      gSystem->Setenv("XrdSecPWDALOGFILE",alogfile.Data());

   TString verisrv = gEnv->GetValue("XSec.Pwd.VerifySrv","1");
   if (verisrv.Length() > 0)
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
   if (deplen.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYDEPLEN",deplen.Data());

   TString pxybits = gEnv->GetValue("XSec.GSI.ProxyKeyBits","");
   if (pxybits.Length() > 0)
      gSystem->Setenv("XrdSecGSIPROXYKEYBITS",pxybits.Data());

   TString crlcheck = gEnv->GetValue("XSec.GSI.CheckCRL","2");
   if (crlcheck.Length() > 0)
      gSystem->Setenv("XrdSecGSICRLCHECK",crlcheck.Data());

   TString delegpxy = gEnv->GetValue("XSec.GSI.DelegProxy","0");
   if (delegpxy.Length() > 0)
      gSystem->Setenv("XrdSecGSIDELEGPROXY",delegpxy.Data());

   TString signpxy = gEnv->GetValue("XSec.GSI.SignProxy","1");
   if (signpxy.Length() > 0)
      gSystem->Setenv("XrdSecGSISIGNPROXY",signpxy.Data());

   // Print the tag, if required (only once)
   if (gEnv->GetValue("XNet.PrintTAG",0) == 1)
      ::Info("TXSocket","(C) 2005 CERN TXSocket (XPROOF client) %s",
            gROOT->GetVersion());

   // Only once
   fgInitDone = kTRUE;
}

//
// TXSockBuf static methods
//

//_____________________________________________________________________________
Long64_t TXSockBuf::BuffMem()
{
   // Return the currently allocated memory

   return fgBuffMem;
}

//_____________________________________________________________________________
Long64_t TXSockBuf::GetMemMax()
{
   // Return the max allocated memory allowed

   return fgMemMax;
}

//_____________________________________________________________________________
void TXSockBuf::SetMemMax(Long64_t memmax)
{
   // Return the max allocated memory allowed

   fgMemMax = memmax > 0 ? memmax : fgMemMax;
}


