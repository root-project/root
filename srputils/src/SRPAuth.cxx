// @(#)root/srputils:$Name$:$Id$
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <stdio.h>
extern "C" {
#include <t_pwd.h>
#include <t_client.h>
}

#include "TSocket.h"
#include "TNetFile.h"

   
Int_t SRPAuthenticate(TNetFile *);

class SRPAuthInit {
public:
   SRPAuthInit() { TNetFile::SetSecureAuthHook(&SRPAuthenticate); }
};
static SRPAuthInit srpauth_init;


//______________________________________________________________________________
Int_t SRPAuthenticate(TNetFile *nf)
{
   // Authenticate to remote rootd server using the SRP (secure remote
   // password) protocol. Returns 0 if authentication failed, 1 if
   // authentication succeeded and 2 if SRP is not available and standard
   // authentication should be tried

   Int_t  result = 0;
   char  *passwd = 0;
   Int_t  stat, ikind;
   EMessageTypes kind;
   
   // check rootd protocol version (we need at least protocol version 2)
   /* Redundant since kROOTD_PROTOCOL is only known since version 2
   nf->fSocket->Send(kROOTD_PROTOCOL);
   nf->Recv(stat, kind);
   if (kind == kROOTD_PROTOCOL && stat < 2)
      return 2;
   */

   // send user name
   nf->fSocket->Send(nf->fUser, kROOTD_SRPUSER);

   nf->Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      nf->PrintError("SRPAuthenticate", stat);
      return result;
   }
   // stat == 2 when no SRP support compiled in remote rootd
   if (kind == kROOTD_AUTH && stat == 2)
      return 2;

   struct t_num     n, g, s, B, *A;
   struct t_client *tc;
   char    hexbuf[MAXHEXPARAMLEN];
   UChar_t buf1[MAXPARAMLEN], buf2[MAXPARAMLEN], buf3[MAXSALTLEN];

   // receive n from server
   nf->fSocket->Recv(hexbuf, MAXHEXPARAMLEN, ikind);
   if ((EMessageTypes)ikind != kROOTD_SRPN) {
      nf->Error("SRPAuthenticate", "expected kROOTD_SRPN message");
      goto out;
   }
   n.data = buf1;
   n.len  = t_fromb64((char*)n.data, hexbuf);

   // receive g from server
   nf->fSocket->Recv(hexbuf, MAXHEXPARAMLEN, ikind);
   if ((EMessageTypes)ikind != kROOTD_SRPG) {
      nf->Error("SRPAuthenticate", "expected kROOTD_SRPG message");
      goto out;
   }
   g.data = buf2;
   g.len  = t_fromb64((char*)g.data, hexbuf);

   // receive salt from server
   nf->fSocket->Recv(hexbuf, MAXHEXPARAMLEN, ikind);
   if ((EMessageTypes)ikind != kROOTD_SRPSALT) {
      nf->Error("SRPAuthenticate", "expected kROOTD_SRPSALT message");
      goto out;
   }
   s.data = buf3;
   s.len  = t_fromb64((char*)s.data, hexbuf);

   tc = t_clientopen(nf->fUser, &n, &g, &s);

   A = t_clientgenexp(tc);

   // send A to server
   nf->fSocket->Send(t_tob64(hexbuf, (char*)A->data, A->len), kROOTD_SRPA);

   if (!passwd) {
      passwd = nf->GetPasswd("Secure password: ");
      if (!passwd)
         nf->Error("SRPAuthenticate", "password not set");
   }

   t_clientpasswd(tc, passwd);

   // receive B from server
   nf->fSocket->Recv(hexbuf, MAXHEXPARAMLEN, ikind);
   if ((EMessageTypes)ikind != kROOTD_SRPB) {
      nf->Error("SRPAuthenticate", "expected kROOTD_SRPB message");
      goto out;
   }
   B.data = buf1;
   B.len  = t_fromb64((char*)B.data, hexbuf);

   t_clientgetkey(tc, &B);

   // send response to server
   nf->fSocket->Send(t_tohex(hexbuf, (char*)t_clientresponse(tc), RESPONSE_LEN),
                     kROOTD_SRPRESPONSE);

   t_clientclose(tc);

   nf->Recv(stat, kind);
   if (kind == kROOTD_ERR)
      nf->PrintError("SRPAuthenticate", stat);
   if (kind == kROOTD_AUTH && stat == 1)
      result = 1;

out:
   delete [] passwd;

   return result;
}
