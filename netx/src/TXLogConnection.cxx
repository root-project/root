// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TXLogConnection.h"
#include "TXPhyConnection.h"
#include "TXDebug.h"
#include "TError.h"
#include <stdlib.h>

ClassImp(TXLogConnection);

//_____________________________________________________________________________
TXLogConnection::TXLogConnection() : fLogLastBytesSent(0), fLogBytesSent(0),
                                     fLogLastBytesRecv(0), fLogBytesRecv(0)
{
   // Constructor
}

//_____________________________________________________________________________
TXLogConnection::~TXLogConnection()
{
   // Destructor
}

//_____________________________________________________________________________
UInt_t TXLogConnection::GetPhyBytesSent()
{ 
   // Return number of bytes sent

   return fPhyConnection->GetBytesSent(); 
}

//_____________________________________________________________________________
UInt_t TXLogConnection::GetPhyBytesRecv() 
{ 
   // Return number of bytes received

   return fPhyConnection->GetBytesRecv();
}

//_____________________________________________________________________________
Int_t TXLogConnection::WriteRaw(const void *buffer, Int_t bufferlength, 
                                ESendRecvOptions opt)
{
   // Send over the open physical connection 'bufferlength' bytes located
   // at buffer.
   // Return number of bytes sent.

   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("WriteRaw", "Writing %d bytes to physical connection",
           bufferlength);
  
   Int_t nwrite = fPhyConnection->WriteRaw(buffer, bufferlength, opt);
   fLogLastBytesSent = nwrite;

   if (fLogLastBytesSent > 0)
      fLogBytesSent += fLogLastBytesSent;

   return fLogLastBytesSent;
}

//_____________________________________________________________________________
Int_t TXLogConnection::ReadRaw(void *buffer, Int_t bufferlength, 
                               ESendRecvOptions opt)
{
   // Receive from the open physical connection 'bufferlength' bytes and 
   // save in buffer.
   // Return number of bytes received.

   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("ReadRaw", "Reading %d bytes from physical connection",
           bufferlength);

   Int_t nread = fPhyConnection->ReadRaw(buffer, bufferlength, opt);
  
   fLogLastBytesRecv = nread;
   if (fLogLastBytesRecv > 0)
      fLogBytesRecv += fLogLastBytesRecv;
  
   return fLogLastBytesRecv;
}

//_____________________________________________________________________________
Bool_t TXLogConnection::ProcessUnsolicitedMsg(TXUnsolicitedMsgSender *sender,
                                              TXMessage *unsolmsg)
{
   // We are here if an unsolicited response comes from the connmgr
   // The response comes in the form of an TXMessage *, that must NOT be
   // destroyed after processing. It is destroyed by the first sender.
   // Remember that we are in a separate thread, since unsolicited responses
   // are asynchronous by nature.

   Info("ProcessUnsolicitedMsg", "Processing unsolicited response");

   // Local processing ....

   // We propagate the event to the obj which registered itself here
   SendUnsolicitedMsg(sender, unsolmsg);
   return kTRUE;
}
