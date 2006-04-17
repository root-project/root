// @(#)root/proofx:$Name:  $:$Id: TXSocket.cxx,v 1.6 2006/04/07 09:26:19 rdm Exp $
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
#include "TObjString.h"
#include "TProof.h"
#include "TSlave.h"
#include "TRegexp.h"
#include "TROOT.h"
#include "TUrl.h"
#include "TXSocket.h"
#include "XProofProtocol.h"

#include "XrdProofConn.h"

#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientMessage.hh"

#include <sys/socket.h>

// ---- Tracing utils ----------------------------------------------------------
#include "XrdOuc/XrdOucError.hh"
#include "XrdOuc/XrdOucLogger.hh"
#include "XrdProofdTrace.h"
XrdOucTrace *XrdProofdTrace = 0;
static XrdOucLogger eLogger;
static XrdOucError eDest(0, "Proofx");

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
TList        TXSocket::fgReadySock;         // Static list of sockets ready to be read
TMutex       TXSocket::fgReadyMtx;          // Protect access to the sockets-ready list
Int_t        TXSocket::fgPipe[2] = {-1,-1}; // Pipe for input monitoring
TString      TXSocket::fgLoc = "undef";     // Location string

//_____________________________________________________________________________
TXSocket::TXSocket(const char *url,
                   Char_t m, Int_t psid, Char_t capver, const char *alias)
         : TSocket(), fMode(m), fSessionID(psid), fAlias(alias), fASem(0), fISem(0)
{
   // Constructor
   // Open the connection to a remote XrdProofd instance and start a PROOF
   // session.
   // The mode 'm' indicates the role of this connection:
   //     'a'      Administrator; used by an XPD to contact the head XPD
   //     'i'      Internal; used by a TXProofServ to call back its creator
   //              (see XrdProofUnixConn)
   //     'C'      PROOF manager: open connection only (do not start a session)
   //     'M'      Client contacting a top master
   //     'm'      Top master contacting a submaster
   //     's'      Master contacting a slave

   // Enable tracing in the XrdProof client. if not done already
   eDest.logger(&eLogger);
   if (!XrdProofdTrace)
      XrdProofdTrace = new XrdOucTrace(&eDest);

   // Init envs the first time
   if (!fgInitDone)
      InitEnvs();

   // Async queue related stuff
   if (!(fAMtx = new TMutex())) {
      Error("TXSocket", "problems initializing mutex for async queue");
      return;
   }
   fAQue.clear();

   // Queue for spare buffers
   if (!(fSMtx = new TMutex())) {
      Error("TXSocket", "problems initializing mutex for spare queue");
      return;
   }
   fSQue.clear();

   // Interrupts queue related stuff
   if (!(fIMtx = new TMutex())) {
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
   fSocket = -1;

   // This is used by external code to create a link between this object
   // and another one
   fReference = 0;

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
      char md = (m != 'C') ? m : 'M';
      fConn = new XrdProofConn(url, md, psid, capver, this);
      if (!fConn || !(fConn->IsValid())) {
         if (fConn->GetServType() != XrdProofConn::kSTProofd)
             Error("TXSocket", "severe error occurred while opening a connection"
                   " to server [%s]", fUrl.Data());
         return;
      }

      // Create new proofserv if not client manager or administrator or internal mode
      if (m == 'm' || m == 's' || m == 'M') {
         // We attach or create
         if (!Create()) {
            // Failure
            Error("TXSocket", "create or attach failed (%s)",fConn->fLastErrMsg.c_str());
            Close();
            return;
         }
      }

      // Fill some info
      fHost = fConn->fHost.c_str();
      fPort = fConn->fPort;

      // Also in the base class
      fUrl = fConn->fUrl.GetUrl().c_str();
      fAddress.fPort = fPort;

      // This is needed for the reader thread to signal an interrupt
      fPid = gSystem->GetPid();
   }
}

//_____________________________________________________________________________
TXSocket::~TXSocket()
{
   // Destructor

   // Disconnect from remote server (the connection manager is
   // responsible of the underlying physical connection, so we do not
   // force its closing)
   Close();

   // Cleanup buffers
   list<TXSockBuf *>::iterator i;
   for (i = fSQue.begin(); i != fSQue.end(); ++i) {
      if (*i)
         delete (*i);
   }
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

   // Warn the remote session, if any (after destroy the session is gone)
   if (sessID > -1)
      DisconnectSession(sessID, opt);

   // Close underlying connection
   fConn->Close(opt);

   // Delete the connection module
   SafeDelete(fConn);

   SafeDelete(fAMtx);
   SafeDelete(fSMtx);
   SafeDelete(fIMtx);
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

   // From now on make sure is for us
   if (!m->MatchStreamid(fConn->fStreamid))
      return kUNSOL_CONTINUE;

   if (gDebug > 2)
      Info("TXSocket::ProcessUnsolicitedMsg", "Processing unsolicited response");

   // Local processing ....
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
            //
            // Post it
            if (gDebug > 3)
               Info("ProcessUnsolicitedMsg","%p: posting interrupt: %d",this,ilev);
            fISem.Post();
            //
            // Signal it
            if (gDebug > 3)
               Info("ProcessUnsolicitedMsg","%p: sending SIGURG: %d",this,ilev);
            if (kill(fPid, SIGURG) != 0) {
               Error("ProcessUnsolicitedMsg","%d: problems sending kSigUrgent", this);
               return rc;
            }
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
      Printf("TXSocket::PostPipe: %s: %p: pipe posted", fgLoc.Data(), s);

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
      ::Error("TXSocket::CleanPipe", "can't read from pipe");
      return -1;
   }

   // This must be an atomic action
   R__LOCKGUARD(&TXSocket::fgReadyMtx);

   // Remove this one
   TXSocket::fgReadySock.Remove(s);

   if (gDebug > 2)
      Printf("TXSocket::CleanPipe: %s: %p: pipe cleaned", fgLoc.Data(), s);

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
Int_t TXSocket::GetInterrupt(Int_t to)
{
   // Get highest interrupt level in the queue

   if (gDebug > 2)
      Info("GetInterrupt","%p: waiting at interrupt semaphore: %p", this, &fISem);
   if (fISem.Wait(to) != 0) {
      Error("GetInterrupt","error waiting at semaphore");
      return -1;
   }

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
            R__LOCKGUARD(fSMtx);
            fSQue.push_back(*i);
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
   if (fSessionID > -1) {
      reqhdr.header.requestid = kXP_attach;
      reqhdr.proof.sid = fSessionID;
   } else {
      reqhdr.header.requestid = kXP_create;
   }

   // Send also the chosen alias
   const void *buf = (const void *)(fAlias.Data());
   reqhdr.header.dlen = fAlias.Length();
   if (gDebug >= 2)
      Info("Create", "sending %d bytes to server", reqhdr.header.dlen);

   // We call SendReq, the function devoted to sending commands.
   if (gDebug > 1)
      Info("Create", "creating session of server %s", fUrl.Data());

   // server response header
   struct ServerResponseBody_Protocol *srvresp = 0;
   XrdClientMessage *xrsp = fConn->SendReq(&reqhdr, buf,
                                          (void **)&srvresp, "TXSocket::Create");
   if (xrsp) {

      //
      // Pointer to data
      void *pdata = (void *)(xrsp->GetData());
      Int_t len = xrsp->DataLen();

      if (len > (Int_t)sizeof(kXR_int32)) {
         // The first 4 bytes contain the session ID
         kXR_int32 psid = 0;
         memcpy(&psid, pdata, sizeof(kXR_int32));
         fSessionID = net2host(psid);
         pdata = (void *)((char *)pdata + sizeof(kXR_int32));
         len -= sizeof(kXR_int32);
      } else {
         Error("Create","session ID is undefined!");
      }

      if (len > (Int_t)sizeof(kXR_int32)) {
         // The second 4 bytes contain the remote daemon version
         kXR_int32 dver = 0;
         memcpy(&dver, pdata, sizeof(kXR_int32));
         fRemoteProtocol = net2host(dver);
         pdata = (void *)((char *)pdata + sizeof(kXR_int32));
         len -= sizeof(kXR_int32);
      } else {
         Warning("Create","protocol version of the remote daemon undefined!");
      }

      if (len > 0) {
         // The login username
         char *u = new char[len+1];
         if (u) {
            memcpy(u, pdata, len);
            u[len] = 0;
            fUser = u;
            delete[] u;
         }
      }

      // Cleanup
      SafeDelete(xrsp);
      if (srvresp)
         free(srvresp);

      // Notify
      return kTRUE;

   }

   // Notify failure
   Error("Create",
         "creating or attaching to a remote server (%s)", fConn->fLastErrMsg.c_str());
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

   // Failure
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

   // Failure
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
      Info("RecvRaw","%p: going to sleep", this);
   if (fASem.Wait() != 0) {
      Error("RecvRaw","error waiting at semaphore");
      return -1;
   }
   if (gDebug > 2)
      Info("RecvRaw","%p: waken up", this);

   R__LOCKGUARD(fAMtx);

   // Get message, if any
   if (fAQue.size() <= 0) {
      Error("RecvRaw","queue is empty - protocol error ?");
      return -1;
   }
   fBufCur = fAQue.front();
   // Remove message from the queue
   fAQue.pop_front();
   // Set number of available bytes
   if (fBufCur)
      fByteLeft = fBufCur->fLen;

   if (gDebug > 2)
      Info("RecvRaw","%p: got message (%d bytes)", this, (Int_t)(fBufCur ? fBufCur->fLen : 0));

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

   R__LOCKGUARD(fSMtx);

   if (fSQue.size() > 0) {
      list<TXSockBuf *>::iterator i;
      for (i = fSQue.begin(); i != fSQue.end(); i++) {
         if ((*i) && (*i)->fSiz >= size) {
            buf = *i;
            // Drop from this list
            fSQue.erase(i);
            return buf;
         }
      }
      // All buffers are too small: enlarge the first one
      buf = fSQue.front();
      buf->Resize(size);
      // Drop from this list
      fSQue.pop_front();
   }

   // Create a new buffer
   char *b = (char *)malloc(size);
   if (b)
      buf = new TXSockBuf(b, size);

   // We are done
   return buf;
}

//______________________________________________________________________________
void TXSocket::PushBackSpare()
{
   // Release read buffer giving back to the spare list

   R__LOCKGUARD(fSMtx);

   fSQue.push_back(fBufCur);
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
      if (fByteLeft == length)
         // All used: give back
         PushBackSpare();
      fByteLeft -= length;
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
         fByteLeft -= ncpy;
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

   // Failure
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
      case kSessionTag:
      case kSessionAlias:
         reqhdr.proof.sid = fSessionID;
         reqhdr.header.dlen = strlen(msg);
         buf = (const void *)msg;
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

   return sout;
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
   Printf("TXSocket::DumpReadySock: %s: list content: %s", fgLoc.Data(), buf.Data());

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
   Int_t requTO = gEnv->GetValue("XProof.RequestTimeout",
                                  DFLT_REQUESTTIMEOUT);
   EnvPutInt(NAME_REQUESTTIMEOUT, requTO);

   // Whether to use a separate thread for garbage collection
   Int_t garbCollTh = gEnv->GetValue("XProof.StartGarbageCollectorThread",
                                      DFLT_STARTGARBAGECOLLECTORTHREAD);
   EnvPutInt(NAME_STARTGARBAGECOLLECTORTHREAD, garbCollTh);

   // PROOF needs a separate thread for reading
   EnvPutInt(NAME_GOASYNC, 1);

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
