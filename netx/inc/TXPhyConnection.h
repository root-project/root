// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXPhyConnection
#define ROOT_TXPhyConnection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXPhyConnection                                                      //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Class handling physical connections to xrootd servers                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXSocket
#include "TXSocket.h"
#endif
#ifndef ROOT_TXMessage
#include "TXMessage.h"
#endif
#ifndef ROOT_TXUnsolicitedMsg
#include "TXUnsolicitedMsg.h"
#endif
#ifndef ROOT_TXInputBuffer
#include "TXInputBuffer.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TMutex
#include "TMutex.h"
#endif

#include <time.h> // for time_t data type
#include <pthread.h>

enum ELoginState {
   kNo      = 0,
   kYes     = 1, 
   kPending = 2
};
enum ERemoteServer {
   kBase    = 100, 
   kData    = 101, 
   kUnknown = 102
};

class TXPhyConnection: public TXUnsolicitedMsgSender, public TObject {
private:
   time_t              fLastUseTimestamp;
   enum ELoginState    fLogged;       // only 1 login/auth is needed for physical  
   TXInputBuffer       fMsgQ;         // The queue used to hold incoming messages
   Int_t               fRequestTimeout;
  
   TMutex              *fRwMutex;     // Lock before using the physical channel 
                                      // (for reading and/or writing)

   TThread             *fReaderthreadhandler; // The thread which is going to pump
                                             // out the data from the socket
                                             // in the async mode
   Bool_t              fReaderthreadrunning;

/* #ifndef __CINT__ */

/*    pthread_t           fReaderthreadhandler; // The thread which is going to pump */
/*                                              // out the data from the socket */
/*                                              // in the async operations */
/* #endif */

   TString             fRemoteAddress;
   Int_t               fRemotePort;
   TXSocket           *fSocket;

   void HandleUnsolicited(TXMessage *m);

public:
   ERemoteServer       fServer;
   Long_t              fTTLsec;

   TXPhyConnection(TXAbsUnsolicitedMsgHandler *h);
   ~TXPhyConnection();

   TXMessage     *BuildXMessage(ESendRecvOptions opt, Bool_t IgnoreTimeouts,
                                Bool_t Enqueue);
   Bool_t         Connect(TString TcpAddress, Int_t TcpPort, Int_t TcpWindowSize);
   void           Disconnect();
   Bool_t         ExpiredTTL();
   UInt_t         GetBytesRecv();
   UInt_t         GetBytesSent();
   UInt_t         GetSocketBytesRecv();
   UInt_t         GetSocketBytesSent();
   Long_t         GetTTL() const { return fTTLsec; }
   Bool_t         IsAddress(TString &addr) { return (fRemoteAddress == addr);}
   ELoginState    IsLogged() const { return fLogged; }
   Bool_t         IsPort(Int_t port) const { return (fRemotePort == port); };
   Bool_t         IsValid() const { return (fSocket && fSocket->IsValid());}
   void           LockChannel();
   Int_t          ReadRaw(void *buffer, Int_t BufferLength, 
                          ESendRecvOptions opt = kDefault);
   TXMessage     *ReadXMessage(Int_t streamid);
   Bool_t         ReConnect(TString TcpAddress, Int_t TcpPort, Int_t TcpWindowSize);
   void           SetLogged(ELoginState status) { fLogged = status; }
   inline void    SetTTL(Long_t ttl) { fTTLsec = ttl; }
   void           StartReader();
   void           Touch();
   void           UnlockChannel();
   Int_t          WriteRaw(const void *buffer, Int_t BufferLength, 
                           ESendRecvOptions opt = kDefault);

   ClassDef(TXPhyConnection, 1);
};

#endif
