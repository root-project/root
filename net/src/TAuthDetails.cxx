// @(#)root/net:$Name:  $:$Id: TAuthDetails.cxx,v 1.1 2003/08/29 10:38:19 rdm Exp $
// Author: G. Ganis   19/03/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAuthDetails                                                         //
//                                                                      //
// Contains details about successful authentications                    //
// Used by THostAuth                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef HAVE_CONFIG
#include "config.h"
#endif

#include <stdlib.h>

#include "TAuthDetails.h"

ClassImp(TAuthDetails)

//______________________________________________________________________________
TAuthDetails::TAuthDetails(const char *host, Int_t meth, Int_t offset,
                           Bool_t reuse, const char *details, const char *token,
                           Int_t key, const char *login)
{
   // Create authdetails object.

   fHost         = host;  // contains also info about remote port and service
   fPort         = -2;
   fService      = (EService) 0;
   fMethod       = meth;
   fRemoteOffSet = offset;
   fRemoteLogin  = login;
   fDetails      = details;
   fReUse        = reuse;
   fToken        = token;
   fRSAKey       = key;
}

//______________________________________________________________________________
const char *TAuthDetails::GetHost() const
{
  // Return remote host name.

   if (fRealHost == "") {
      const_cast<TAuthDetails*>(this)->fRealHost = fHost;
      if (fRealHost.Index(":") != kNPOS)
         const_cast<TAuthDetails*>(this)->fRealHost.Remove(fRealHost.Index(":"));
   }

   return fRealHost;
}

//______________________________________________________________________________
Int_t TAuthDetails::GetPort() const
{
  // Return remote port. Returns -1 if port not found.

   if (fPort == -2) {
      Int_t f = fHost.First(':');
      Int_t l = fHost.Last(':');
      if (l == kNPOS || f == kNPOS || f == l) {
         const_cast<TAuthDetails*>(this)->fPort = -1;
         return fPort;
      }
      f++;
      TString port = fHost(f, l-f);
      const_cast<TAuthDetails*>(this)->fPort = atoi(port.Data());
   }
   return fPort;
}

//______________________________________________________________________________
Int_t TAuthDetails::GetService() const
{
  // Return remote service flag, either kROOTD, kPROOFD or kUNKNOWN.

   if (fService == 0) {
      Int_t f = fHost.First(':');
      Int_t l = fHost.Last(':');
      if (l == kNPOS || f == kNPOS || f == l) {
         const_cast<TAuthDetails*>(this)->fService = (EService)-1;
         return fService;
      }
      l++;
      TString service = fHost(l, fHost.Length()-1);
      const_cast<TAuthDetails*>(this)->fService = (EService) atoi(service.Data());
   }
   return fService;
}

//______________________________________________________________________________
void TAuthDetails::Print(Option_t *opt) const
{
   // Print object content. If option is "e" print "established details.

   // Method names
   const char *Service[3]= {" ","rootd","proofd"};

   Int_t srv = (GetService() > 0 && GetService() < 3) ? GetService() : 0;

   if (opt[0] == 'e') {
      Info("PrintEstblshd","+ Method:%d (%s) OffSet:%d Login:%s ReUse:%d Port:%d Service:%s",
           fMethod,TAuthenticate::GetAuthMethod(fMethod),fRemoteOffSet,fRemoteLogin.Data(),
           fReUse,GetPort(),Service[srv]);
      Info("PrintEstblshd","+   Details:%s",fDetails.Data());
   } else {
      Info("Print","+ Host:%s Port:%d Service:%s Method:%d (%s) OffSet:%d Login:%s ReUse:%d Details:%s",
           GetHost(),GetPort(),Service[srv],fMethod,TAuthenticate::GetAuthMethod(fMethod),
           fRemoteOffSet,fRemoteLogin.Data(),fReUse,fDetails.Data());
   }
}
