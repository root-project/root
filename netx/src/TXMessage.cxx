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
// TXMessage                                                            //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// Class to handle messages to/from xrootd                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXMessage.h"
#include "TXPhyConnection.h"
#include "TXProtocol.h"

#include "TError.h"
#include "TXDebug.h"
#include <stdlib.h> // for malloc
#include <string.h> // for memcpy

ClassImp(TXMessage);

//__________________________________________________________________________
TXMessage::TXMessage(struct ServerResponseHeader header)
{
   // Constructor

   fStatusCode = kXMSC_ok;
   memcpy((void *)&fHdr, (const void*)&header, sizeof(ServerResponseHeader));
   fData = 0;
   fMarshalled = false;
   if (!CreateData()) {
      Error("TXMessage", 
            "Error allocating %d bytes for this TXMessage", fHdr.dlen);
      fAllocated = false;
   } else 
      fAllocated = true;
}

//__________________________________________________________________________
TXMessage::TXMessage()
{
   // Default constructor

   memset(&fHdr, 0, sizeof(fHdr));
   fStatusCode = kXMSC_ok;
   fData = 0;
   fMarshalled = false;
   fAllocated = false;
}

//__________________________________________________________________________
TXMessage::~TXMessage()
{
   // Destructor

   if (fData)
      free(fData);
}

//__________________________________________________________________________
void *TXMessage::DonateData()
{
   // Unlink the owned data in order to pass them elsewhere

   void *res = fData;
   fData = 0;
   fAllocated = false;
  
   return (res);
}

//__________________________________________________________________________
Bool_t TXMessage::CreateData()
{
   // Allocate data

   if (!fAllocated) {
      if (fHdr.dlen) {
         fData = malloc(fHdr.dlen+1);
         if (!fData) {
            Error("TXMessage::CreateData","Fatal ERROR *** malloc failed."
                  " Probable system resources exhausted.");
            gSystem->Abort();
         }
         char *tmpPtr = (char *)fData;
         memset((void*)(tmpPtr+fHdr.dlen), 0, 1);
      }
      if (!fData)
         return kFALSE;
      else
         return kTRUE;
   } else
      return kTRUE;
}

//__________________________________________________________________________
void TXMessage::Marshall()
{
   // Marshall, i.e. put in network byte order

   if (!fMarshalled) {
      ROOT::ServerResponseHeader2NetFmt(&fHdr);
      fMarshalled = kTRUE;
   }
}

//__________________________________________________________________________
void TXMessage::Unmarshall()
{
   // Unmarshall, i.e. from network byte to normal order

   if (fMarshalled) {
      ROOT::clientUnmarshall(&fHdr);
      fMarshalled = kFALSE;
   }
}

//__________________________________________________________________________
Int_t TXMessage::ReadRaw(TXPhyConnection *phy, ESendRecvOptions opt)
{
   // Given a physical connection, we completely build the content
   // of the message, reading it from the socket of a phyconn

   Int_t bytesread;
   int readLen = sizeof(ServerResponseHeader);

   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      Info("TXMessage::ReadRaw", "Reading header (%d bytes) from socket.",
 	   readLen);
  
   bytesread = phy->ReadRaw((void *)&fHdr, readLen, opt);
   if (bytesread < readLen) {
      if (bytesread == TXSOCK_ERR_TIMEOUT)
         SetStatusCode(kXMSC_timeout);
      else {
         Error("TXMessage::ReadRaw", "Error reading %d bytes.", readLen);
         SetStatusCode(kXMSC_readerr);
      }
      memset(&fHdr, 0, sizeof(fHdr));
   }

   // the data arrive marshalled from the server (i.e. network byte order)
   SetMarshalled(kTRUE);
   Unmarshall();

   if (fHdr.dlen) {
      if (DebugLevel() >= TXDebug::kDUMPDEBUG)
         Info("TXMessage::ReadRaw", "Reading data (%d bytes) from socket.",
              fHdr.dlen);
      CreateData();
      if (phy->ReadRaw(fData, fHdr.dlen, opt) < fHdr.dlen) {
         Error("TXMessage::ReadRaw", "Error reading %d bytes.", fHdr.dlen);
         SetStatusCode(kXMSC_readerr);
      }
   }
   return 1;
}

//___________________________________________________________________________
void TXMessage::Int2CharStreamid(kXR_char *charstreamid, Short_t intstreamid)
{
   // Converts a streamid given as an integer to its representation
   // suitable for the streamid inside the messages (i.e. ascii)

   memcpy(charstreamid, &intstreamid, sizeof(intstreamid));
}

//___________________________________________________________________________
Short_t TXMessage::CharStreamid2Int(kXR_char *charstreamid)
{
   // Converts a streamid given as an integer to its representation
   // suitable for the streamid inside the messages (i.e. ascii)

   Int_t res = *((Short_t *)charstreamid);

   return res;
}
