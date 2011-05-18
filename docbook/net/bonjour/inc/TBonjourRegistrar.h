// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBonjourRegistrar
#define ROOT_TBonjourRegistrar


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBonjourRegistrar                                                    //
//                                                                      //
// This class consists of one main member function, RegisterService(),  //
// that registers the service. As long as the object is alive, the      //
// service stays registered. The rest of the class wraps the various    //
// bits of Bonjour service registration. The static callback function   //
// is marked with the DNSSD_API macro to make sure that the callback    //
// has the correct calling convention on Windows.                       //
//                                                                      //
// Bonjour works out-of-the-box on MacOS X. On Linux you have to        //
// install the Avahi package and run the avahi-daemon. To compile       //
// these classes and run Avahi on Linux you need to install the:        //
//    avahi                                                             //
//    avahi-compat-libdns_sd-devel                                      //
//    nss-mdns                                                          //
// packages. After installation make sure the avahi-daemon is started.  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif
#ifndef ROOT_TBonjourRecord
#include "TBonjourRecord.h"
#endif

#if !defined(__CINT__)
#include <dns_sd.h>
#else
typedef ULong_t DNSServiceRef;
typedef UInt_t  DNSServiceFlags;
typedef Int_t   DNSServiceErrorType;
#endif

class TFileHandler;


class TBonjourRegistrar : public TObject, public TQObject {

private:
   DNSServiceRef    fDNSRef;
   TFileHandler    *fBonjourSocketHandler;
   TBonjourRecord   fFinalRecord;

   void *GetSender() { return this; }  // used to get gTQSender

#if !defined(__CINT__)
   static void DNSSD_API BonjourRegisterService(DNSServiceRef, DNSServiceFlags, DNSServiceErrorType,
                                                const char *, const char *, const char *, void *);
#else
   static void BonjourRegisterService(DNSServiceRef, DNSServiceFlags, DNSServiceErrorType,
                                      const char *, const char *, const char *, void *);
#endif

public:
   TBonjourRegistrar();
   virtual ~TBonjourRegistrar();

   Int_t RegisterService(const TBonjourRecord &record, UShort_t servicePort);
   TBonjourRecord RegisteredRecord() const { return fFinalRecord; }

   void ServiceRegistered(TBonjourRecord *record);  //*SIGNAL*

   void BonjourSocketReadyRead();  // private slot

   ClassDef(TBonjourRegistrar,0)  // Register Bonjour service
};

#endif
