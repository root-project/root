// @(#)root/net:$Name:  $:$Id: TInetAddress.h,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
// Author: Fons Rademakers   16/12/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TInetAddress
#define ROOT_TInetAddress


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInetAddress                                                         //
//                                                                      //
// This class represents an Internet Protocol (IP) address.             //
// Objects of this class can not be created directly, but only via      //
// the TSystem GetHostByName(), GetSockName(), and GetPeerName()        //
// members and via members of the TServerSocket and TSocket classes.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TInetAddress : public TObject {

friend class TSystem;
friend class TUnixSystem;
friend class TWinNTSystem;
friend class TVmsSystem;
friend class TMacSystem;
friend class TSocket;
friend class TServerSocket;

private:
   TString fHostname;    // full qualified hostname
   UInt_t  fAddress;     // IP address in host byte order
   Int_t   fFamily;      // address family
   Int_t   fPort;        // port through which we are connected

   TInetAddress(const char *host, UInt_t addr, Int_t family, Int_t port = -1);

public:
   TInetAddress();
   TInetAddress(const TInetAddress &adr);
   TInetAddress &operator=(const TInetAddress &rhs);
   virtual ~TInetAddress() { }

   UInt_t      GetAddress() const { return fAddress; }
   UChar_t    *GetAddressBytes() const;
   const char *GetHostAddress() const;
   const char *GetHostName() const { return (const char *) fHostname; }
   Int_t       GetFamily() const { return fFamily; }
   Int_t       GetPort() const { return fPort; }
   Bool_t      IsValid() const { return fFamily == -1 ? kFALSE : kTRUE; }
   void        Print(Option_t *option="") const;

   ClassDef(TInetAddress,1)  //Represents an Internet Protocol (IP) address
};

#endif
