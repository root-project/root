// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXLogConnection
#define ROOT_TXLogConnection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXLogConnection                                                      //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Class implementing logical connections                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TXUnsolicitedMsg
#include "TXUnsolicitedMsg.h"
#endif
#ifndef ROOT_TXPhyConnection
#include "TXPhyConnection.h"
#endif

class TXLogConnection: public TXAbsUnsolicitedMsgHandler, 
   TXUnsolicitedMsgSender /*, public TObject*/ {
private:
   TXPhyConnection *fPhyConnection;
   Int_t            fLogLastBytesSent;
   Int_t            fLogBytesSent;
   Int_t            fLogLastBytesRecv;
   Int_t            fLogBytesRecv;

public:
   TXLogConnection();
   virtual ~TXLogConnection();

   inline UInt_t GetBytesSent() const { return fLogBytesSent; }
   inline UInt_t GetBytesRecv() const { return fLogBytesRecv; }

   UInt_t        GetPhyBytesSent();
   UInt_t        GetPhyBytesRecv();
   inline TXPhyConnection *GetPhyConnection() { return fPhyConnection; }
   Int_t         LastBytesRecv(void) { return fLogLastBytesRecv; }
   Int_t         LastBytesSent(void) { return fLogLastBytesSent; }
   Bool_t        ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                       TXMessage *unsolmsg);
   Int_t         ReadRaw(void *buffer, Int_t BufferLength, 
                         ESendRecvOptions opt = kDefault);
   inline void   SetPhyConnection(TXPhyConnection *PhyConn) 
                 { fPhyConnection = PhyConn; }
   Int_t         WriteRaw(const void *buffer, Int_t BufferLength, 
                          ESendRecvOptions opt = kDefault);

   ClassDef(TXLogConnection, 1); // The logical connection of the client
};

#endif
