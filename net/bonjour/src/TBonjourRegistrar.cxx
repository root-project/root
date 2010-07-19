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

#include "TBonjourRegistrar.h"
#include "TSysEvtHandler.h"
#include "TError.h"
#include "TSystem.h"

#include <arpa/inet.h>


ClassImp(TBonjourRegistrar)

//______________________________________________________________________________
TBonjourRegistrar::TBonjourRegistrar() : fDNSRef(0), fBonjourSocketHandler(0)
{
   // Default ctor.

   // silence Avahi about using Bonjour compat layer
   gSystem->Setenv("AVAHI_COMPAT_NOWARN", "1");
}

//______________________________________________________________________________
TBonjourRegistrar::~TBonjourRegistrar()
{
   // Cleanup.

   delete fBonjourSocketHandler;

   if (fDNSRef) {
      DNSServiceRefDeallocate(fDNSRef);
      fDNSRef = 0;
   }
}

//______________________________________________________________________________
Int_t TBonjourRegistrar::RegisterService(const TBonjourRecord &record, UShort_t servicePort)
{
   // Register Bonjour service.
   // Return -1 in case or error, 0 otherwise.

   if (fDNSRef) {
      Warning("RegisterService", "already registered a service");
      return 0;
   }

   UShort_t sport = htons(servicePort);

   // register our service and callback
   DNSServiceErrorType err = DNSServiceRegister(&fDNSRef, 0, kDNSServiceInterfaceIndexAny,
                                                !strlen(record.GetServiceName()) ? 0
                                                : record.GetServiceName(),
                                                record.GetRegisteredType(),
                                                !strlen(record.GetReplyDomain()) ? 0
                                                : record.GetReplyDomain(),
                                                0, sport,
                                                record.GetTXTRecordsLength(),
                                                !strlen(record.GetTXTRecords()) ? 0
                                                : record.GetTXTRecords(),
                                                (DNSServiceRegisterReply)BonjourRegisterService,
                                                this);
   if (err != kDNSServiceErr_NoError) {
      Error("RegisterService", "error in DNSServiceRegister (%d)", err);
      return -1;
   }

   Int_t sockfd = DNSServiceRefSockFD(fDNSRef);
   if (sockfd == -1) {
      Error("RegisterService", "invalid sockfd");
      return -1;
   }

   fBonjourSocketHandler = new TFileHandler(sockfd, TFileHandler::kRead);
   fBonjourSocketHandler->Connect("Notified()", "TBonjourRegistrar", this, "BonjourSocketReadyRead()");
   fBonjourSocketHandler->Add();

   return 0;
}

//______________________________________________________________________________
void TBonjourRegistrar::ServiceRegistered(TBonjourRecord *record)
{
   // Emit ServiceRegistered signal.

   Emit("ServiceRegistered(TBonjourRecord*)", (Long_t)record);
}

//______________________________________________________________________________
void TBonjourRegistrar::BonjourSocketReadyRead()
{
   // The Bonjour socket is ready for reading. Tell Bonjour to process the
   // information on the socket, this will invoke the BonjourRegisterService
   // callback. This is a private slot, used in RegisterService.

   DNSServiceErrorType err = DNSServiceProcessResult(fDNSRef);
   if (err != kDNSServiceErr_NoError)
      Error("BonjourSocketReadyRead", "error in DNSServiceProcessResult");
}

//______________________________________________________________________________
void TBonjourRegistrar::BonjourRegisterService(DNSServiceRef, DNSServiceFlags,
                                               DNSServiceErrorType errCode,
                                               const char *name, const char *regType,
                                               const char *domain, void *context)
{
   // Static Bonjour register callback function.

   TBonjourRegistrar *registrar = static_cast<TBonjourRegistrar*>(context);
   if (errCode != kDNSServiceErr_NoError) {
      ::Error("TBonjourRegistrar::BonjourRegisterService", "error in BonjourRegisterService");
      //registrar->Error(errorCode);
   } else {
      registrar->fFinalRecord = TBonjourRecord(name, regType, domain);
      registrar->ServiceRegistered(&registrar->fFinalRecord);
   }
}
