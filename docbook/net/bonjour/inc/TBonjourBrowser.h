// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBonjourBrowser
#define ROOT_TBonjourBrowser


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBonjourBrowser                                                      //
//                                                                      //
// This class consists of one main member function,                     //
// BrowseForServiceType(), that looks for the service.                  //
// The rest of the class wraps the various bits of Bonjour service      //
// browser. The static callback function is marked with the DNSSD_API   //
// macro to make sure that the callback has the correct calling         //
// convention on Windows.                                               //
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
#ifndef ROOT_TString
#include "TString.h"
#endif

#if !defined(__CINT__)
#include <dns_sd.h>
#else
typedef ULong_t DNSServiceRef;
typedef UInt_t  DNSServiceFlags;
typedef Int_t   DNSServiceErrorType;
#endif

class TFileHandler;
class TList;


class TBonjourBrowser : public TObject, public TQObject {

private:
   DNSServiceRef    fDNSRef;
   TFileHandler    *fBonjourSocketHandler;
   TList           *fBonjourRecords;
   TString          fBrowsingType;

   void *GetSender() { return this; }  // used to get gTQSender

#if !defined(__CINT__)
   static void DNSSD_API BonjourBrowseReply(DNSServiceRef,
                                            DNSServiceFlags, UInt_t, DNSServiceErrorType,
                                            const char *, const char *, const char *, void *);
#else
   static void BonjourBrowseReply(DNSServiceRef,
                                  DNSServiceFlags, Int_t, DNSServiceErrorType,
                                  const char *, const char *, const char *, void *);
#endif

public:
   TBonjourBrowser();
   virtual ~TBonjourBrowser();

   Int_t       BrowseForServiceType(const char *serviceType);
   TList      *CurrentRecords() const { return fBonjourRecords; }
   const char *ServiceType() const { return fBrowsingType; }

   void CurrentBonjourRecordsChanged(TList *bonjourRecords);  //*SIGNAL*

   void BonjourSocketReadyRead();  // private slot

   ClassDef(TBonjourBrowser,0)  // Browse hosts for specific bonjour service type
};

#endif
