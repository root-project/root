// @(#)root/srputils:$Name:  $:$Id: SRPAuth.cxx,v 1.2 2000/11/27 10:50:17 rdm Exp $
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
#include "TAuthenticate.h"
#include "TError.h"


Int_t SRPAuthenticate(TSocket *, const char *user, const char *passwd,
                      const char *remote);

class SRPAuthInit {
public:
   SRPAuthInit() { TAuthenticate::SetSecureAuthHook(&SRPAuthenticate); }
};
static SRPAuthInit srpauth_init;


//______________________________________________________________________________
Int_t SRPAuthenticate(TSocket *sock, const char *user, const char *passwd,
                      const char *remote)
{
   // Authenticate to remote rootd server using the SRP (secure remote
   // password) protocol. Returns 0 if authentication failed, 1 if
   // authentication succeeded and 2 if SRP is not available and standard
   // authentication should be tried

   Int_t  result = 0;
   char  *usr = 0;
   char  *psswd = 0;
   Int_t  stat, kind;

   // check rootd protocol version (we need at least protocol version 2)
   /* Redundant since kROOTD_PROTOCOL is only known since version 2
   sock->Send(kROOTD_PROTOCOL);
   sock->Recv(stat, kind);
   if (kind == kROOTD_PROTOCOL && stat < 2)
      return 2;
   */

   // send user name
   if (user && user[0])
      usr = StrDup(user);
   else
      usr = TAuthenticate::PromptUser(remote);

   sock->Send(usr, kROOTD_SRPUSER);

   sock->Recv(stat, kind);

   if (kind == kROOTD_ERR) {
      TAuthenticate::AuthError("SRPAuthenticate", stat);
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
   sock->Recv(hexbuf, MAXHEXPARAMLEN, kind);
   if (kind != kROOTD_SRPN) {
      ::Error("SRPAuthenticate", "expected kROOTD_SRPN message");
      goto out;
   }
   n.data = buf1;
   n.len  = t_fromb64((char*)n.data, hexbuf);

   // receive g from server
   sock->Recv(hexbuf, MAXHEXPARAMLEN, kind);
   if (kind != kROOTD_SRPG) {
      ::Error("SRPAuthenticate", "expected kROOTD_SRPG message");
      goto out;
   }
   g.data = buf2;
   g.len  = t_fromb64((char*)g.data, hexbuf);

   // receive salt from server
   sock->Recv(hexbuf, MAXHEXPARAMLEN, kind);
   if (kind != kROOTD_SRPSALT) {
      ::Error("SRPAuthenticate", "expected kROOTD_SRPSALT message");
      goto out;
   }
   s.data = buf3;
   s.len  = t_fromb64((char*)s.data, hexbuf);

   tc = t_clientopen(usr, &n, &g, &s);

   A = t_clientgenexp(tc);

   // send A to server
   sock->Send(t_tob64(hexbuf, (char*)A->data, A->len), kROOTD_SRPA);

   if (passwd && passwd[0])
      psswd = StrDup(passwd);
   else {
      psswd = TAuthenticate::PromptPasswd("Secure password: ");
      if (!psswd)
         ::Error("SRPAuthenticate", "password not set");
   }

   t_clientpasswd(tc, psswd);

   // receive B from server
   sock->Recv(hexbuf, MAXHEXPARAMLEN, kind);
   if (kind != kROOTD_SRPB) {
      ::Error("SRPAuthenticate", "expected kROOTD_SRPB message");
      goto out;
   }
   B.data = buf1;
   B.len  = t_fromb64((char*)B.data, hexbuf);

   t_clientgetkey(tc, &B);

   // send response to server
   sock->Send(t_tohex(hexbuf, (char*)t_clientresponse(tc), RESPONSE_LEN),
              kROOTD_SRPRESPONSE);

   t_clientclose(tc);

   sock->Recv(stat, kind);
   if (kind == kROOTD_ERR)
      TAuthenticate::AuthError("SRPAuthenticate", stat);
   if (kind == kROOTD_AUTH && stat == 1)
      result = 1;

out:
   delete [] usr;
   delete [] psswd;

   return result;
}
