// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXMessage
#define ROOT_TXMessage


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

#ifndef __XPROTOCOL_H
#include "XProtocol/XProtocol.hh"
#endif
#ifndef ROOT_TXSocket
#include "TXSocket.h"
#endif

class TXPhyConnection;

class TXMessage {

private:
   Bool_t           fAllocated;
   void*            fData;   //!
   Bool_t           fMarshalled;
   kXR_int16        fStatusCode;

   Short_t     CharStreamid2Int(kXR_char *charstreamid);
   void        Int2CharStreamid(kXR_char *charstreamid, Short_t intstreamid);

public:

   enum EXMSCStatus {             // Some status codes useful
      kXMSC_ok       = 0,
      kXMSC_readerr  = 1,
      kXMSC_writeerr = 2,
      kXMSC_timeout  = 3
   };
   ServerResponseHeader fHdr;

   TXMessage(ServerResponseHeader header);
   TXMessage();
   virtual ~TXMessage();

   Bool_t             CreateData();
   inline kXR_int32   DataLen() { return fHdr.dlen; }
   void              *DonateData();
   inline void       *GetData() {return fData;}
   inline kXR_int16   GetStatusCode() { return fStatusCode;}
   inline kXR_int16   HeaderStatus() { return fHdr.status; }
   inline Short_t     HeaderSID() { return CharStreamid2Int(fHdr.streamid); }
   Bool_t             IsAttn() { return (fHdr.status == kXR_attn); }
   inline bool        IsError() { return (fStatusCode != kXMSC_ok); };
   inline bool        IsMarshalled() { return fMarshalled; }
   void               Marshall();
   inline Bool_t      MatchStreamid(Short_t sid) { return (HeaderSID() == sid);}
   Int_t              ReadRaw(TXPhyConnection *phy, ESendRecvOptions opt);
   inline void        SetHeaderStatus(kXR_int16 sts) { fHdr.status = sts; }
   inline void        SetMarshalled(Bool_t m) { fMarshalled = m; }
   inline void        SetStatusCode(kXR_int16 status) { fStatusCode = status; }
   void               Unmarshall();

   ClassDef(TXMessage,0);
};

#endif
