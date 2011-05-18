// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBonjourResolver
#define ROOT_TBonjourResolver


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBonjourResolver                                                     //
//                                                                      //
// This class consists of one main member function,                     //
// ResolveBonjourRecord(), that resolves the service to an actual       //
// IP address and port number. The rest of the class wraps the various  //
// bits of Bonjour service resolver. The static callback function       //
// is marked with the DNSSD_API macro to make sure that the callback    //
// has the correct calling convention on Windows.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TInetAddress
#include "TInetAddress.h"
#endif

#if !defined(__CINT__)
#include <dns_sd.h>
#else
typedef ULong_t DNSServiceRef;
typedef UInt_t  DNSServiceFlags;
typedef Int_t   DNSServiceErrorType;
#endif

class TFileHandler;
class TBonjourRecord;


class TBonjourResolver : public TObject, public TQObject {

private:
   DNSServiceRef    fDNSRef;
   TFileHandler    *fBonjourSocketHandler;
   TInetAddress     fHostAddress;
   Int_t            fPort;
   TString          fTXTRecord;

   void *GetSender() { return this; }  // used to get gTQSender

#if !defined(__CINT__)
   static void DNSSD_API BonjourResolveReply(DNSServiceRef, DNSServiceFlags, UInt_t,
                                             DNSServiceErrorType,
                                             const char *, const char *,
                                             UShort_t, UShort_t, const char *, void *);
#else
   static void BonjourResolveReply(DNSServiceRef, DNSServiceFlags, UInt_t,
                                   DNSServiceErrorType,
                                   const char *, const char *,
                                   UShort_t, UShort_t, const char *, void *);
#endif

public:
   TBonjourResolver();
   virtual ~TBonjourResolver();

   TInetAddress GetInetAddress() const { return fHostAddress; }
   Int_t GetPort() const { return fPort; }
   const char * GetTXTRecord() const { return fTXTRecord; }

   Int_t ResolveBonjourRecord(const TBonjourRecord &record);

   void RecordResolved(const TInetAddress *hostInfo, Int_t port);  //*SIGNAL*

   void BonjourSocketReadyRead();  // private slot

   ClassDef(TBonjourResolver,0)  // Resolve Bonjour to actual IP address and port
};

#endif
