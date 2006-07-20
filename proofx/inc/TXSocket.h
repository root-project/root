// @(#)root/proofx:$Name:  $:$Id: TXSocket.h,v 1.7 2006/06/21 16:18:26 rdm Exp $
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

class TXSocket  : public TSocket, public XrdClientAbsUnsolMsgHandler {

friend class TXProofMgr;
friend class TXProofServ;
friend class TXSlave;
friend class TXSocketHandler;
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
   std::list<TXSockBuf *> fAQue;          // list of asynchronous messages
   Int_t               fByteLeft;      // bytes left in the first buffer
   Int_t               fByteCur;       // current position in the first buffer
   TXSockBuf          *fBufCur;        // current read buffer

   // Interrupts
   TMutex             *fIMtx;          // To protect interrupt queue
   kXR_int32           fILev;          // Highest received interrupt

   // Process ID of the instatiating process (to signal interrupts)
   Int_t               fPid;

   // Whether to timeout or not
   Bool_t              fDontTimeout;   // If true wait forever for incoming messages

   // Static area for input handling
   static TList        fgReadySock;    // Static list of sockets ready to be read
   static TMutex       fgReadyMtx;     // Protect access to the sockets-ready list
   static Int_t        fgPipe[2];      // Pipe for input monitoring
   static TString      fgLoc;          // Location string
   static Bool_t       fgInitDone;     // Avoid initializing more than once

   // List of spare buffers
   static TMutex       fgSMtx;          // To protect spare list
   static std::list<TXSockBuf *> fgSQue; // list of spare buffers

   // Manage asynchronous message
   Int_t               PickUpReady();
   TXSockBuf          *PopUpSpare(Int_t sz);
   void                PushBackSpare();

   // Auxilliary
   Int_t               GetLowSocket() const { return (fConn ? fConn->GetLowSocket() : -1); }

   static Int_t        GetPipeRead(); // Return the read-descriptor of the global pipe
   static Int_t        PostPipe(TSocket *s=0);  // Notify socket ready via global pipe
   static Int_t        CleanPipe(TSocket *s=0); // Clean previous pipe notification
   static Int_t        FlushPipe(TSocket *s=0); // Remove any instance of 's' from the pipe

   static void         InitEnvs(); // Initialize environment variables

   static void         DumpReadySock(); // Dump content of the ready-socket list

public:
   // Should be the same as in proofd/src/XrdProofdProtocol.cxx (local definitions)
   enum ECoordMsgType { kQuerySessions = 1000,
                        kSessionTag, kSessionAlias, kGetWorkers, kQueryWorkers,
                        kCleanupSessions };
   // Should be the same as in proofd/src/XrdProofdProtocol::Urgent
   enum EUrgentMsgType { kStopProcess = 2000 };

   TXSocket(const char *url, Char_t mode = 'M',
            Int_t psid = -1, Char_t ver = -1, const char *logbuf = 0, Int_t loglevel = -1);
   TXSocket(const TXSocket &xs);
   TXSocket& operator=(const TXSocket& xs);
   virtual ~TXSocket();

   virtual void        Close(Option_t *opt = "");
   Bool_t              Create();
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

   Bool_t              IsValid() const { return (fConn ? (fConn->IsValid()) : kFALSE); }
   Bool_t              IsServProofd();
   virtual void        RemoveClientID() { }
   virtual void        SetClientID(Int_t) { }
   void                SetSendOpt(ESendRecvOptions o) { fSendOpt = o; }
   void                SetSessionID(Int_t id) { fSessionID = id; }

   // Send interfaces
   Int_t               Send(const TMessage &mess);
   Int_t               Send(Int_t kind) { return TSocket::Send(kind); }
   Int_t               Send(Int_t status, Int_t kind)
                                        { return TSocket::Send(status, kind); }
   Int_t               Send(const char *mess, Int_t kind = kMESS_STRING)
                                        { return TSocket::Send(mess, kind); }
   Int_t               SendRaw(const void *buf, Int_t len,
                               ESendRecvOptions opt = kDontBlock);

   TObjString         *SendCoordinator(Int_t kind, const char *msg = 0);

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
   Int_t               GetInterrupt();

   // Urgent message
   void                SendUrgent(Int_t type, Int_t int1, Int_t int2);

   // Interrupt the low level socket
   void                SetInterrupt() { if (fConn) fConn->SetInterrupt(); }

   // Flush the asynchronous queue
   Int_t               Flush();

   // Ping the counterpart
   Bool_t              Ping(Bool_t cleanpipe = kFALSE);

   // Standard options cannot be set
   Int_t               SetOption(ESockOptions, Int_t) { return 0; }

   // Disable / Enable read timeout
   void                DisableTimeout() { fDontTimeout = kTRUE; }
   void                EnableTimeout() { fDontTimeout = kFALSE; }

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

   TXSockBuf(Char_t *bp=0, Int_t sz=0, Bool_t own=1)
             { fBuf = fMem = bp; fSiz = fLen = sz; fOwn = own; fCid = -1; fgBuffMem += sz; }
  ~TXSockBuf() {if (fOwn && fMem) { free(fMem); fgBuffMem -= fSiz; }}

   void Resize(Int_t sz) { if (sz > fSiz)
                              if ((fMem = (Char_t *)realloc(fMem, sz))) { fgBuffMem += (sz - fSiz);
                                 fBuf = fMem; fSiz = sz; fLen = 0;}}

   static Long64_t BuffMem() { return fgBuffMem; }

private:
   Char_t *fMem;
   static Long64_t fgBuffMem;  // Total allocated memory
};

#endif
