// @(#)root/bonjour:$Id$
// Author: Fons Rademakers   29/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TBonjourBrowser.h"
#include "TBonjourRecord.h"
#include "TSysEvtHandler.h"
#include "TList.h"
#include "TError.h"
#include "TSystem.h"


ClassImp(TBonjourBrowser)

//______________________________________________________________________________
TBonjourBrowser::TBonjourBrowser() : fDNSRef(0), fBonjourSocketHandler(0)
{
   // Default ctor.

   fBonjourRecords = new TList;
   fBonjourRecords->SetOwner();

   // silence Avahi about using Bonjour compat layer
   gSystem->Setenv("AVAHI_COMPAT_NOWARN", "1");
}

//______________________________________________________________________________
TBonjourBrowser::~TBonjourBrowser()
{
   // Cleanup.

   delete fBonjourRecords;
   delete fBonjourSocketHandler;

   if (fDNSRef) {
      DNSServiceRefDeallocate(fDNSRef);
      fDNSRef = 0;
   }
}

//______________________________________________________________________________
Int_t TBonjourBrowser::BrowseForServiceType(const char *serviceType)
{
   // Tell Bonjour to start browsing for a specific type of service.
   // Returns -1 in case of error, 0 otherwise.

   DNSServiceErrorType err = DNSServiceBrowse(&fDNSRef, 0,
                                              0, serviceType, 0,
                                              (DNSServiceBrowseReply)BonjourBrowseReply,
                                              this);
   if (err != kDNSServiceErr_NoError) {
      Error("BrowseForServiceType", "error in DNSServiceBrowse (%d)", err);
      return -1;
   }

   Int_t sockfd = DNSServiceRefSockFD(fDNSRef);
   if (sockfd == -1) {
      Error("BrowseForServiceType", "invalid sockfd");
      return -1;
   }

   fBonjourSocketHandler = new TFileHandler(sockfd, TFileHandler::kRead);
   fBonjourSocketHandler->Connect("Notified()", "TBonjourBrowser", this, "BonjourSocketReadyRead()");
   fBonjourSocketHandler->Add();

   return 0;
}

//______________________________________________________________________________
void TBonjourBrowser::CurrentBonjourRecordsChanged(TList *bonjourRecords)
{
   // Emit CurrentBonjourRecordsChanged signal.

   Emit("CurrentBonjourRecordsChanged(TList*)", (Long_t)bonjourRecords);
}

//______________________________________________________________________________
void TBonjourBrowser::BonjourSocketReadyRead()
{
   // The Bonjour socket is ready for reading. Tell Bonjour to process the
   // information on the socket, this will invoke the BonjourBrowseReply
   // callback. This is a private slot, used in BrowseForServiceType.

   // in case the browser has already been deleted
   if (!fDNSRef) return;

   DNSServiceErrorType err = DNSServiceProcessResult(fDNSRef);
   if (err != kDNSServiceErr_NoError)
      Error("BonjourSocketReadyRead", "error in DNSServiceProcessResult");
}

//______________________________________________________________________________
void TBonjourBrowser::BonjourBrowseReply(DNSServiceRef,
                                         DNSServiceFlags flags, UInt_t,
                                         DNSServiceErrorType errorCode,
                                         const char *serviceName, const char *regType,
                                         const char *replyDomain, void *context)
{
   // Static Bonjour browser callback function.

   TBonjourBrowser *browser = static_cast<TBonjourBrowser*>(context);
   if (errorCode != kDNSServiceErr_NoError) {
      ::Error("TBonjourBrowser::BonjourBrowseReply", "error in BonjourBrowseReply");
      //browser->Error(errorCode);
   } else {
      TBonjourRecord *record = new TBonjourRecord(serviceName, regType, replyDomain);
      if (flags & kDNSServiceFlagsAdd) {
         if (!browser->fBonjourRecords->FindObject(record))
            browser->fBonjourRecords->Add(record);
         else
            delete record;
      } else {
         TBonjourRecord *r = (TBonjourRecord*)browser->fBonjourRecords->Remove(record);
         delete r;
         delete record;
      }
      if (!(flags & kDNSServiceFlagsMoreComing)) {
         browser->CurrentBonjourRecordsChanged(browser->fBonjourRecords);
      }
   }
}
