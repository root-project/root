// @(#)root/proofx:$Id$
// Author: G. Ganis Oct 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXSocket
#define ROOT_TXSocket

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXSocket                                                             //
//                                                                      //
// High level handler of connections to xproofd.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#define DFLT_CONNECTMAXTRY           10

#ifndef ROOT_TMutex
#include "TMutex.h"
#endif
#ifndef ROOT_TSemaphore
#include "TSemaphore.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TMessage
#include "TMessage.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TSocket
#include "TSocket.h"
#endif
#ifndef ROOT_XrdProofConn
#include "XrdProofConn.h"
#endif
#ifndef XRC_UNSOLMSG_H
#include "XrdClient/XrdClientUnsolMsg.hh"
#endif

#include <list>

class TObjString;
class TXSockBuf;
class TXSockPipe;
class TXHandler;
class TXSocketHandler;
class XrdClientMessage;

// To transmit info to Handlers
typedef struct {
   Int_t   fInt1;
   Int_t   fInt2;
   Int_t   fInt3;
   Int_t   fInt4;
} XHandleIn_t;
typedef struct {
   Int_t   fOpt;
   const char *fMsg;
} XHandleErr_t;

class TXSocket  : public TSocket, public XrdClientAbsUnsolMsgHandler {

friend class TXProofMgr;
friend class TXProofServ;
friend class TXSlave;
friend class TXSocketHandler;
friend class TXSockPipe;
friend class TXUnixSocket;

private:
   char                fMode;          // 'e' (def) or 'i' (internal - proofsrv)
   kXR_int32           fSendOpt;       // Options for sending messages
   Short_t             fSessionID;     // proofsrv: remote ID of connected session
   TString             fUser;          // Username used for login
   TString             fHost;          // Remote host
   Int_t               fPort;          // Remote port

   Int_t               fLogLevel;      // Log level to be transmitted to servers

   TString             fBuffer;        // Container for exchanging information
   TObject            *fReference;     // Generic object reference of this socket
   TXHandler          *fHandler;       // Handler of asynchronous events (input, error)

   XrdProofConn       *fConn;          // instance of the underlying connection module

   // Asynchronous messages
   TSemaphore          fASem;          // Control access to conn async msg queue
   TMutex             *fAMtx;          // To protect async msg queue
   Bool_t              fAWait;         // kTRUE if waiting at the async msg queue
   std::list<TXSockBuf *> fAQue;          // list of asynchronous messages
   Int_t               fByteLeft;      // bytes left in the first buffer
   Int_t               fByteCur;       // current position in the first buffer
   TXSockBuf          *fBufCur;        // current read buffer

   TSemaphore          fAsynProc;      // Control actions while processing async messages

   // Interrupts
   TMutex             *fIMtx;          // To protect interrupt queue
   kXR_int32           fILev;          // Highest received interrupt
   Bool_t              fIForward;      // Whether the interrupt should be propagated

   // Process ID of the instatiating process (to signal interrupts)
   Int_t               fPid;

   // Whether to timeout or not
   Bool_t              fDontTimeout;   // If true wait forever for incoming messages
   Bool_t              fRDInterrupt;   // To interrupt waiting for messages

   // Version of the remote XrdProofdProtocol
   Int_t               fXrdProofdVersion;

   // Static area for input handling
   static TXSockPipe   fgPipe;         //  Pipe for input monitoring
   static TString      fgLoc;          // Location string
   static Bool_t       fgInitDone;     // Avoid initializing more than once

   // List of spare buffers
   static TMutex       fgSMtx;          // To protect spare list
   static std::list<TXSockBuf *> fgSQue; // list of spare buffers

   // Manage asynchronous message
   Int_t               PickUpReady();
   TXSockBuf          *PopUpSpare(Int_t sz);
   void                PushBackSpare();

   // Post a message into the queue for asynchronous processing
   void                PostMsg(Int_t type, const char *msg = 0);

   // Auxilliary
   Int_t               GetLowSocket() const { return (fConn ? fConn->GetLowSocket() : -1); }

   static void         SetLocation(const char *loc = ""); // Set location string

   static void         InitEnvs(); // Initialize environment variables

public:
   // Should be the same as in proofd/src/XrdProofdProtocol::Urgent
   enum EUrgentMsgType { kStopProcess = 2000 };

   TXSocket(const char *url, Char_t mode = 'M', Int_t psid = -1, Char_t ver = -1,
            const char *logbuf = 0, Int_t loglevel = -1, TXHandler *handler = 0);
#if 0
   TXSocket(const TXSocket &xs);
   TXSocket& operator=(const TXSocket& xs);
#endif
   virtual ~TXSocket();

   virtual void        Close(Option_t *opt = "");
   Bool_t              Create(Bool_t attach = kFALSE);
   void                DisconnectSession(Int_t id, Option_t *opt = "");

   void                DoError(int level,
                               const char *location, const char *fmt, va_list va) const;

   virtual UnsolRespProcResult ProcessUnsolicitedMsg(XrdClientUnsolMsgSender *s,
                                                     XrdClientMessage *msg);

   virtual Int_t       GetClientID() const { return -1; }
   virtual Int_t       GetClientIDSize() const { return 1; }
   Int_t               GetLogConnID() const { return (fConn ? fConn->GetLogConnID() : -1); }
   Int_t               GetOpenError() const { return (fConn ? fConn->GetOpenError() : -1); }
   Int_t               GetServType() const { return (fConn ? fConn->GetServType() : -1); }
   Int_t               GetSessionID() const { return (fConn ? fConn->GetSessionID() : -1); }
   Int_t               GetXrdProofdVersion() const { return fXrdProofdVersion; }

   Bool_t              IsValid() const { return (fConn ? (fConn->IsValid()) : kFALSE); }
   Bool_t              IsServProofd();
   virtual void        RemoveClientID() { }
   virtual void        SetClientID(Int_t) { }
   void                SetSendOpt(ESendRecvOptions o) { fSendOpt = o; }
   void                SetSessionID(Int_t id);

   // Send interfaces
   Int_t               Send(const TMessage &mess);
   Int_t               Send(Int_t kind) { return TSocket::Send(kind); }
   Int_t               Send(Int_t status, Int_t kind)
                                        { return TSocket::Send(status, kind); }
   Int_t               Send(const char *mess, Int_t kind = kMESS_STRING)
                                        { return TSocket::Send(mess, kind); }
   Int_t               SendRaw(const void *buf, Int_t len,
                               ESendRecvOptions opt = kDontBlock);

   TObjString         *SendCoordinator(Int_t kind, const char *msg = 0, Int_t int2 = 0,
                                       Long64_t l64 = 0, Int_t int3 = 0, const char *opt = 0);

   // Recv interfaces
   Int_t               Recv(TMessage *&mess);
   Int_t               Recv(Int_t &status, Int_t &kind)
                                        { return TSocket::Recv(status, kind); }
   Int_t               Recv(char *mess, Int_t max)
                                        { return TSocket::Recv(mess, max); }
   Int_t               Recv(char *mess, Int_t max, Int_t &kind)
                                        { return TSocket::Recv(mess, max, kind); }
   Int_t               RecvRaw(void *buf, Int_t len,
                               ESendRecvOptions opt = kDefault);

   // Interrupts
   Int_t               SendInterrupt(Int_t type);
   Int_t               GetInterrupt(Bool_t &forward);

   // Urgent message
   void                SendUrgent(Int_t type, Int_t int1, Int_t int2);

   // Interrupt the low level socket
   inline void         SetInterrupt(Bool_t i = kTRUE) { R__LOCKGUARD(fAMtx);
                                        fRDInterrupt = i;
                                        if (i && fConn) fConn->SetInterrupt();
                                        if (i && fAWait) fASem.Post(); }
   inline Bool_t       IsInterrupt()  { R__LOCKGUARD(fAMtx); return fRDInterrupt; }
   // Set / Check async msg queue waiting status
   inline void         SetAWait(Bool_t w = kTRUE) { R__LOCKGUARD(fAMtx); fAWait = w; }
   inline Bool_t       IsAWait()  { R__LOCKGUARD(fAMtx); return fAWait; }

   // Flush the asynchronous queue
   Int_t               Flush();

   // Ping the counterpart
   Bool_t              Ping(const char *ord = 0);

   // Request remote touch of the admin file associated with this connection
   void                RemoteTouch();
   // Propagate a Ctrl-C
   void                CtrlC();

   // Standard options cannot be set
   Int_t               SetOption(ESockOptions, Int_t) { return 0; }

   // Disable / Enable read timeout
   void                DisableTimeout() { fDontTimeout = kTRUE; }
   void                EnableTimeout() { fDontTimeout = kFALSE; }

   // Try reconnection after error
   virtual Int_t       Reconnect();

   ClassDef(TXSocket, 0) //A high level connection class for PROOF
};


//
// The following structure is used to store buffers received asynchronously
//
class TXSockBuf {
public:
   Int_t   fSiz;
   Int_t   fLen;
   Char_t *fBuf;
   Bool_t  fOwn;
   Int_t   fCid;

   TXSockBuf(Char_t *bp=0, Int_t sz=0, Bool_t own=1);
  ~TXSockBuf();

   void Resize(Int_t sz);

   static Long64_t BuffMem();
   static Long64_t GetMemMax();
   static void     SetMemMax(Long64_t memmax);

private:
   Char_t *fMem;
   static Long64_t fgBuffMem; // Total allocated memory
   static Long64_t fgMemMax;  // Max allocated memory allowed
};

//
// The following class describes internal pipes
//
class TXSockPipe {
public:

   TXSockPipe(const char *loc = "");
   virtual ~TXSockPipe();

   Bool_t       IsValid() const { return ((fPipe[0] >= 0 && fPipe[1] >= 0) ? kTRUE : kFALSE); }

   TXSocket    *GetLastReady();

   Int_t        GetRead() const { return fPipe[0]; }
   Int_t        Post(TSocket *s);  // Notify socket ready via global pipe
   Int_t        Clean(TSocket *s); // Clean previous pipe notification
   Int_t        Flush(TSocket *s); // Remove any instance of 's' from the pipe
   void         DumpReadySock();

   void         SetLoc(const char *loc = "") { fLoc = loc; }

private:
   TMutex       fMutex;     // Protect access to the sockets-ready list
   Int_t        fPipe[2];   // Pipe for input monitoring
   TString      fLoc;       // Location string
   TList        fReadySock;    // List of sockets ready to be read
};

//
// Guard for a semaphore
//
class TXSemaphoreGuard {
public:

   TXSemaphoreGuard(TSemaphore *sem) : fSem(sem), fValid(kTRUE) { if (!fSem || fSem->TryWait()) fValid = kFALSE; }
   virtual ~TXSemaphoreGuard() { if (fValid && fSem) fSem->Post(); }

   Bool_t       IsValid() const { return fValid; }

private:
   TSemaphore  *fSem;
   Bool_t       fValid;
};

#endif
