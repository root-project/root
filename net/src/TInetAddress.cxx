// @(#)root/net:$Name:  $:$Id: TInetAddress.cxx,v 1.4 2004/07/08 17:55:41 rdm Exp $
// Author: Fons Rademakers   16/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInetAddress                                                         //
//                                                                      //
// This class represents an Internet Protocol (IP) address.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInetAddress.h"

ClassImp(TInetAddress)

//______________________________________________________________________________
TInetAddress::TInetAddress()
{
   // Default ctor. Used in case of unknown host. Not a valid address.

   fHostname  = "UnknownHost";
   fAddress   = 0;
   fFamily    = -1;
   fPort      = -1;
}

//______________________________________________________________________________
TInetAddress::TInetAddress(const char *host, UInt_t addr, Int_t family, Int_t port)
{
   // Create TInetAddress. Private ctor. TInetAddress objects can only
   // be created via the friend classes TSystem, TServerSocket and TSocket.
   // Use the IsValid() method to check the validity of a TInetAddress.

   fAddress = addr;
   if (!strcmp(host, "????"))
      fHostname = GetHostAddress();
   else
      fHostname = host;
   fFamily    = family;
   fPort      = port;
}

//______________________________________________________________________________
TInetAddress::TInetAddress(const TInetAddress &adr) : TObject(adr)
{
   // TInetAddress copy ctor.

   fHostname  = adr.fHostname;
   fAddress   = adr.fAddress;
   fFamily    = adr.fFamily;
   fPort      = adr.fPort;
   fAddresses = adr.fAddresses;
   fAliases   = adr.fAliases;
}

//______________________________________________________________________________
TInetAddress& TInetAddress::operator=(const TInetAddress &rhs)
{
   // TInetAddress assignment operator.

   if (this != &rhs) {
      TObject::operator=(rhs);
      fHostname  = rhs.fHostname;
      fAddress   = rhs.fAddress;
      fFamily    = rhs.fFamily;
      fPort      = rhs.fPort;
      fAddresses = rhs.fAddresses;
      fAliases   = rhs.fAliases;
   }
   return *this;
}

//______________________________________________________________________________
UChar_t *TInetAddress::GetAddressBytes() const
{
   // Returns the raw IP address in host byte order. The highest
   // order byte position is in addr[0]. To be prepared for 64-bit
   // IP addresses an array of bytes is returned.

   static UChar_t addr[4];

   addr[0] = (UChar_t) ((fAddress >> 24) & 0xFF);
   addr[1] = (UChar_t) ((fAddress >> 16) & 0xFF);
   addr[2] = (UChar_t) ((fAddress >> 8) & 0xFF);
   addr[3] = (UChar_t) (fAddress & 0xFF);

   return addr;
}

//______________________________________________________________________________
const char *TInetAddress::GetHostAddress(UInt_t addr)
{
   // Returns the IP address string "%d.%d.%d.%d", use it to convert
   // alternative addresses obtained via GetAddresses().
   // Copy string immediately, it will be reused. Static function.

   return Form("%d.%d.%d.%d", (addr >> 24) & 0xFF,
                              (addr >> 16) & 0xFF,
                              (addr >>  8) & 0xFF,
                               addr & 0xFF);
}

//______________________________________________________________________________
const char *TInetAddress::GetHostAddress() const
{
   // Returns the IP address string "%d.%d.%d.%d".
   // Copy string immediately, it will be reused.

   return GetHostAddress(fAddress);
}

//______________________________________________________________________________
void TInetAddress::Print(Option_t *) const
{
   // Print internet address as string.

   if (fPort == -1)
      Printf("%s/%s (not connected)", GetHostName(), GetHostAddress());
   else
      Printf("%s/%s (port %d)", GetHostName(), GetHostAddress(), fPort);

   int i = 0;
   AddressList_t::const_iterator ai;
   for (ai = fAddresses.begin(); ai != fAddresses.end(); ai++) {
      if (!i) printf("Alternative addresses:");
      printf(" %s", GetHostAddress(*ai));
      i++;
   }
   if (i) printf("\n");

   i = 0;
   AliasList_t::const_iterator ali;
   for (ali = fAliases.begin(); ali != fAliases.end(); ali++) {
      if (!i) printf("Aliases:");
      printf(" %s", ali->Data());
      i++;
   }
   if (i) printf("\n");
}

//______________________________________________________________________________
void TInetAddress::AddAddress(UInt_t addr)
{
   // Add alternative address to list of addresses.

   fAddresses.push_back(addr);
}

//______________________________________________________________________________
void TInetAddress::AddAlias(const char *alias)
{
   // Add alias to list of aliases.

   fAliases.push_back(TString(alias));
}

//______________________________________________________________________________
void TInetAddress::Streamer(TBuffer &R__b)
{
   // Stream an object of class TInetAddress.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      TObject::Streamer(R__b);
      fHostname.Streamer(R__b);
      R__b >> fAddress;
      R__b >> fFamily;
      R__b >> fPort;
      if (R__v > 1) {
         TInetAddress::AddressList_t &R__stl1 =  fAddresses;
         R__stl1.clear();
         int R__i, R__n;
         R__b >> R__n;
         R__stl1.reserve(R__n);
         for (R__i = 0; R__i < R__n; R__i++) {
            unsigned int R__t1;
            R__b >> R__t1;
            R__stl1.push_back(R__t1);
         }
         TInetAddress::AliasList_t &R__stl2 =  fAliases;
         R__stl2.clear();
         R__b >> R__n;
         R__stl2.reserve(R__n);
         for (R__i = 0; R__i < R__n; R__i++) {
            TString R__t2;
            R__t2.Streamer(R__b);
            R__stl2.push_back(R__t2);
         }
      }
      R__b.CheckByteCount(R__s, R__c, TInetAddress::IsA());
   } else {
      R__c = R__b.WriteVersion(TInetAddress::IsA(), kTRUE);
      TObject::Streamer(R__b);
      fHostname.Streamer(R__b);
      R__b << fAddress;
      R__b << fFamily;
      R__b << fPort;
      {
         TInetAddress::AddressList_t &R__stl =  fAddresses;
         int R__n=(&R__stl) ? int(R__stl.size()) : 0;
         R__b << R__n;
         if(R__n) {
            TInetAddress::AddressList_t::iterator R__k;
            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
            R__b << (*R__k);
            }
         }
      }
      {
         TInetAddress::AliasList_t &R__stl =  fAliases;
         int R__n=(&R__stl) ? int(R__stl.size()) : 0;
         R__b << R__n;
         if(R__n) {
            TInetAddress::AliasList_t::iterator R__k;
            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {
            ((TString&)(*R__k)).Streamer(R__b);
            }
         }
      }
      R__b.SetByteCount(R__c, kTRUE);
   }
}
