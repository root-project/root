// @(#)root/net:$Name:  $:$Id: TInetAddress.cxx,v 1.3 2002/05/18 08:43:30 brun Exp $
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
