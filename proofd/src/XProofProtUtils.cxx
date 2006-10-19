// @(#)root/proofd:$Name:  $:$Id: XProofProtUtils.cxx,v 1.5 2006/06/21 16:18:26 rdm Exp $
// Author: Gerardo Ganis  12/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XProofProtUtils.cxx                                                  //
//                                                                      //
// Authors: G. Ganis, CERN 2005                                         //
//                                                                      //
// Utility functions prototypes for client-to-server                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XProofProtocol.h"
#include "XProofProtUtils.h"
#include "Bytes.h"


namespace XPD {

//___________________________________________________________________________
int clientMarshall(XPClientRequest* str)
{
   // This function applies the network byte order on those
   // parts of the 16-bytes buffer, only if it is composed
   // by some binary part
   // Return 0 if OK, -1 in case the ID is unknown

   switch(str->header.requestid) {

   case kXP_login:
      str->login.pid = host2net(str->login.pid);
      break;
   case kXP_auth:
      // no swap on ASCII fields
      break;
   case kXP_create:
      // no swap on ASCII fields
      str->proof.int1 = host2net(str->proof.int1);
      break;
   case kXP_destroy:
      str->proof.sid = host2net(str->proof.sid);
      break;
   case kXP_attach:
      str->proof.sid = host2net(str->proof.sid);
      break;
   case kXP_detach:
      str->proof.sid = host2net(str->proof.sid);
      break;
   case kXP_cleanup:
      str->proof.sid = host2net(str->proof.sid);
      str->proof.int1 = host2net(str->proof.int1);
      str->proof.int2 = host2net(str->proof.int2);
      break;
   case kXP_sendmsg:
      str->sendrcv.sid = host2net(str->sendrcv.sid);
      str->sendrcv.opt = host2net(str->sendrcv.opt);
      str->sendrcv.cid = host2net(str->sendrcv.cid);
      break;
   case kXP_admin:
      str->proof.sid = host2net(str->proof.sid);
      str->proof.int1 = host2net(str->proof.int1);
      str->proof.int2 = host2net(str->proof.int2);
      str->proof.int3 = host2net(str->proof.int3);
      break;
   case kXP_interrupt:
      str->interrupt.sid = host2net(str->interrupt.sid);
      str->interrupt.type = host2net(str->interrupt.type);
      break;
   case kXP_ping:
      str->sendrcv.sid = host2net(str->sendrcv.sid);
      str->sendrcv.opt = host2net(str->sendrcv.opt);
      break;
   case kXP_urgent:
      str->proof.sid = host2net(str->proof.sid);
      str->proof.int1 = host2net(str->proof.int1);
      str->proof.int2 = host2net(str->proof.int2);
      str->proof.int3 = host2net(str->proof.int3);
      break;
   default:
      fprintf(stderr,"clientMarshall: unknown req ID: %d (0x%x)\n",
                      str->header.requestid, str->header.requestid);
      return -1;
      break;
   }

   str->header.requestid = host2net(str->header.requestid);
   str->header.dlen      = host2net(str->header.dlen);

   return 0;
}

//_________________________________________________________________________
void clientUnmarshall(struct ServerResponseHeader* str)
{
   str->status = net2host(str->status);
   str->dlen   = net2host(str->dlen);
}

//_________________________________________________________________________
void ServerResponseHeader2NetFmt(struct ServerResponseHeader *srh)
{
   srh->status = host2net(srh->status);
   srh->dlen   = host2net(srh->dlen);
}

//_________________________________________________________________________
void ServerInitHandShake2HostFmt(struct ServerInitHandShake *srh)
{
   srh->msglen  = net2host(srh->msglen);
   srh->protover = net2host(srh->protover);
   srh->msgval  = net2host(srh->msgval);
}

//_________________________________________________________________________
char *convertRequestIdToChar(kXR_int16 requestid)
{
   // This procedure convert the request code id (an integer defined in
   // XProtocol.hh) in the ascii label (human readable)

   switch(requestid) {

   case kXP_login:
      return (char *)"kXP_login";
   case kXP_auth:
      return (char *)"kXP_auth";
   case kXP_create:
      return (char *)"kXP_create";
   case kXP_destroy:
      return (char *)"kXP_destroy";
   case kXP_attach:
      return (char *)"kXP_attach";
   case kXP_detach:
      return (char *)"kXP_detach";
   case kXP_sendmsg:
      return (char *)"kXP_sendmsg";
   case kXP_admin:
      return (char *)"kXP_admin";
   case kXP_interrupt:
      return (char *)"kXP_interrupt";
   case kXP_ping:
      return (char *)"kXP_ping";
   case kXP_cleanup:
      return (char *)"kXP_cleanup";
   case kXP_urgent:
      return (char *)"kXP_urgent";
   default:
#if 0
      int dbg = 1;
      while (dbg)
         ;
#endif
      return (char *)"kXP_UNKNOWN";
   }
}

//___________________________________________________________________________
char *convertRespStatusToChar(kXR_int16 status)
{
   switch( status) {
   case kXP_ok:
      return (char *)"kXP_ok";
      break;
   case kXP_oksofar:
      return (char *)"kXP_oksofar";
      break;
   case kXP_attn:
      return (char *)"kXP_attn";
      break;
   case kXP_authmore:
      return (char *)"kXP_authmore";
      break;
   case kXP_error:
      return (char *)"kXP_error";
      break;
   case kXP_wait:
      return (char *)"kXP_wait";
      break;
   default:
      return (char *)"kXP_UNKNOWN";
      break;
   }
}

//___________________________________________________________________________
void smartPrintClientHeader(XPClientRequest* hdr)
{
   printf("\n\n================= DUMPING CLIENT REQUEST HEADER =================\n");

   printf("%40s0x%.2x 0x%.2x\n", "ClientHeader.streamid = ",
          hdr->header.streamid[0],
          hdr->header.streamid[1]);

   printf("%40s%s (%d)\n",
          "ClientHeader.requestid = ",
          convertRequestIdToChar(hdr->header.requestid), hdr->header.requestid);

   switch(hdr->header.requestid) {

   case kXP_login:
      printf("%40s%d \n",
             "ClientHeader.login.pid = ",
             hdr->login.pid);

      printf("%40s%s\n",
             "ClientHeader.login_body.username = ",
             hdr->login.username);

      printf("%40s0 repeated %d times\n",
             "ClientHeader.login.reserved = ",
             *((kXR_int16 *)&(hdr->login.reserved[0])));

      printf("%40s%d\n",
             "ClientHeader.login.role = ",
             (kXR_int32)hdr->login.role[0]);
      break;

   case kXP_auth:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.auth.reserved = ",
             (kXR_int32)sizeof(hdr->auth.reserved));

      printf("  ClientHeader.auth.credtype= 0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             hdr->auth.credtype[0],
             hdr->auth.credtype[1],
             hdr->auth.credtype[2],
             hdr->auth.credtype[3]);
      break;

   case kXP_create:
      break;
   case kXP_destroy:
      printf("%40s%d \n",
             "ClientHeader.proof.sid = ", hdr->proof.sid);
      break;
   case kXP_attach:
      printf("%40s%d \n",
             "ClientHeader.proof.sid = ", hdr->proof.sid);
      break;
   case kXP_detach:
      printf("%40s%d \n",
             "ClientHeader.proof.sid = ", hdr->proof.sid);
      break;
   case kXP_cleanup:
      printf("%40s%d \n",
             "ClientHeader.proof.sid = ", hdr->proof.sid);
      printf("%40s%d \n",
             "ClientHeader.proof.int1 = ", hdr->proof.int1);
      printf("%40s%d \n",
             "ClientHeader.proof.int2 = ", hdr->proof.int2);
      break;
   case kXP_sendmsg:
      printf("%40s%d \n",
             "ClientHeader.sendrcv.sid = ", hdr->sendrcv.sid);
      printf("%40s%d \n",
             "ClientHeader.sendrcv.opt = ", hdr->sendrcv.opt);
      printf("%40s%d \n",
             "ClientHeader.sendrcv.cid = ", hdr->sendrcv.cid);
      break;
   case kXP_interrupt:
      printf("%40s%d \n",
             "ClientHeader.interrupt.sid = ", hdr->interrupt.sid);
      printf("%40s%d \n",
             "ClientHeader.interrupt.type = ", hdr->interrupt.type);
      break;
   case kXP_ping:
      printf("%40s%d \n",
             "ClientHeader.sendrcv.sid = ", hdr->sendrcv.sid);
      printf("%40s%d \n",
             "ClientHeader.sendrcv.opt = ", hdr->sendrcv.opt);
      break;
   case kXP_admin:
   case kXP_urgent:
      printf("%40s%d \n",
             "ClientHeader.proof.sid = ", hdr->proof.sid);
      printf("%40s%d \n",
             "ClientHeader.proof.int1 = ", hdr->proof.int1);
      printf("%40s%d \n",
             "ClientHeader.proof.int2 = ", hdr->proof.int2);
      printf("%40s%d \n",
             "ClientHeader.proof.int3 = ", hdr->proof.int3);
      break;
   }

   printf("%40s%d",
          "ClientHeader.header.dlen = ",
          hdr->header.dlen);
   printf("\n=================== END CLIENT HEADER DUMPING ===================\n\n");
}

//___________________________________________________________________________
void smartPrintServerHeader(struct ServerResponseHeader* hdr)
{
   printf("\n\n======== DUMPING SERVER RESPONSE HEADER ========\n");
   printf("%30s0x%.2x 0x%.2x\n",
          "ServerHeader.streamid = ",
          hdr->streamid[0],
          hdr->streamid[1]);
   switch(hdr->status) {
   case kXP_ok:
      printf("%30skXP_ok",
             "ServerHeader.status = ");
      break;
   case kXP_attn:
      printf("%30skXP_attn",
             "ServerHeader.status = ");
      break;
   case kXP_authmore:
      printf("%30skXP_authmore",
             "ServerHeader.status = ");
      break;
   case kXP_error:
      printf("%30skXP_error",
             "ServerHeader.status = ");
      break;
   case kXP_oksofar:
      printf("%30skXP_oksofar",
             "ServerHeader.status = ");
      break;
   case kXP_wait:
      printf("%30skXP_wait",
             "ServerHeader.status = ");
      break;
   }
   printf(" (%d)\n", hdr->status);
   printf("%30s%d",
          "ServerHeader.dlen = ", hdr->dlen);
   printf("\n========== END DUMPING SERVER HEADER ===========\n\n");
}

} // namespace ROOT
