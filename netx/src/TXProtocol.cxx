// @(#)root/netx:$Name:  $:$Id: TXProtocol.cxx,v 1.2 2004/08/20 22:16:33 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXProtocol.cxx                                                       //
//                                                                      //
// Utility functions for client-to-server.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXProtocol.h"
#include "Bytes.h"


namespace ROOT {

//___________________________________________________________________________
void clientMarshall(ClientRequest* str)
{
   // This function applies the network byte order on those
   // parts of the 16-bytes buffer, only if it is composed
   // by some binary part

   switch(str->header.requestid) {
   case kXR_auth:
      // no swap on ASCII fields
      break;
   case kXR_chmod:
      str->chmod.mode = host2net(str->chmod.mode);
      break;
   case kXR_close:
      // no swap on ASCII fields
      break;
   case kXR_dirlist:
      // no swap on ASCII fields
      break;
   case kXR_getfile:
      str->getfile.options = host2net(str->getfile.options);
      str->getfile.buffsz  = host2net(str->getfile.buffsz);
      break;
   case kXR_login:
      str->login.pid = host2net(str->login.pid);
      break;
   case kXR_mkdir:
      // no swap on ASCII fields
      break;
   case kXR_mv:
      // no swap on ASCII fields
      break;
   case kXR_open:
      str->open.mode    = host2net(str->open.mode);
      str->open.options = host2net(str->open.options);
      break;
   case kXR_ping:
      // no swap on ASCII fields
      break;
   case kXR_protocol:
      // no swap on ASCII fields
      break;
   case kXR_putfile:
      str->putfile.options = host2net(str->putfile.options);
      str->putfile.buffsz  = host2net(str->putfile.buffsz);
      break;
   case kXR_query:
      // no swap on ASCII fields
      break;
   case kXR_read:
      str->read.offset = host2net(str->read.offset);
      str->read.rlen   = host2net(str->read.rlen);
      break;
   case kXR_rm:
      // no swap on ASCII fields
      break;
   case kXR_rmdir:
      // no swap on ASCII fields
      break;
   case kXR_set:
      // no swap on ASCII fields
      break;
   case kXR_stat:
      // no swap on ASCII fields
      break;
   case kXR_sync:
      // no swap on ASCII fields
      break;
   case kXR_write:
      str->write.offset = host2net(str->write.offset);
      break;
   }

   str->header.requestid = host2net(str->header.requestid);
   str->header.dlen      = host2net(str->header.dlen);
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
   srh->msgtype = net2host(srh->msgtype);
   srh->msgval  = net2host(srh->msgval);
}

//_________________________________________________________________________
bool isRedir(struct ServerResponseHeader *ServerResponse)
{
   // Recognizes if the response contains a redirection

   return ( (ServerResponse->status == kXR_redirect) ? true : false);
}

//_________________________________________________________________________
char *convertRequestIdToChar(kXR_int16 requestid)
{
   // This procedure convert the request code id (an integer defined in
   // XProtocol.hh) in the ascii label (human readable)

   switch(requestid) {
   case kXR_auth:
      return (char *)"kXR_auth";
      break;
   case kXR_chmod:
      return (char *)"kXR_chmod";
      break;
   case kXR_close:
      return (char *)"kXR_close";
      break;
   case kXR_dirlist:
      return (char *)"kXR_dirlist";
      break;
   case kXR_getfile:
      return (char *)"kXR_getfile";
      break;
   case kXR_login:
      return (char *)"kXR_login";
      break;
   case kXR_mkdir:
      return (char *)"kXR_mkdir";
      break;
   case kXR_mv:
      return (char *)"kXR_mv";
      break;
   case kXR_open:
      return (char *)"kXR_open";
      break;
   case kXR_ping:
      return (char *)"kXR_ping";
      break;
   case kXR_protocol:
      return (char *)"kXR_protocol";
      break;
   case kXR_putfile:
      return (char *)"kXR_putfile";
      break;
   case kXR_query:
      return (char *)"kXR_query";
      break;
   case kXR_read:
      return (char *)"kXR_read";
      break;
   case kXR_rm:
      return (char *)"kXR_rm";
      break;
   case kXR_rmdir:
      return (char *)"kXR_rmdir";
      break;
   case kXR_set:
      return (char *)"kXR_set";
      break;
   case kXR_stat:
      return (char *)"kXR_stat";
      break;
   case kXR_sync:
      return (char *)"kXR_sync";
      break;
   case kXR_write:
      return (char *)"kXR_write";
      break;
   case kXR_prepare:
      return (char *)"kXR_prepare";
      break;
   case kXR_admin:
      return (char *)"kXR_admin";
      break;
   case kXR_statx:
      return (char *)"kXR_statx";
      break;
   default:
      return (char *)"kXR_UNKNOWN";
      break;
   }
}

//___________________________________________________________________________
void PutFilehandleInRequest(ClientRequest* str, char *fHandle)
{
   // this function inserts a filehandle in a generic request header
   // already composed

   switch(str->header.requestid) {
   case kXR_close:
      memcpy( str->close.fhandle, fHandle, sizeof(str->close.fhandle) );
      break;
   case kXR_read:
      memcpy( str->read.fhandle, fHandle, sizeof(str->read.fhandle) );
      break;
   case kXR_sync:
      memcpy( str->sync.fhandle, fHandle, sizeof(str->sync.fhandle) );
      break;
   case kXR_write:
      memcpy( str->write.fhandle, fHandle, sizeof(str->write.fhandle) );
      break;
   }
}

//___________________________________________________________________________
char *convertRespStatusToChar(kXR_int16 status)
{
   switch( status) {
   case kXR_ok:
      return (char *)"kXR_ok";
      break;
   case kXR_oksofar:
      return (char *)"kXR_oksofar";
      break;
   case kXR_attn:
      return (char *)"kXR_attn";
      break;
   case kXR_authmore:
      return (char *)"kXR_authmore";
      break;
   case kXR_error:
      return (char *)"kXR_error";
      break;
   case kXR_redirect:
      return (char *)"kXR_redirect";
      break;
   case kXR_wait:
      return (char *)"kXR_wait";
      break;
   default:
      return (char *)"kXR_UNKNOWN";
      break;
   }
}


//___________________________________________________________________________
void smartPrintClientHeader(ClientRequest* hdr)
{
   printf("\n\n================= DUMPING CLIENT REQUEST HEADER =================\n");

   printf("%40s0x%.2x 0x%.2x\n", "ClientHeader.streamid = ",
          hdr->header.streamid[0],
          hdr->header.streamid[1]);

   printf("%40s%s (%d)\n",
          "ClientHeader.requestid = ",
          convertRequestIdToChar(hdr->header.requestid), hdr->header.requestid);

   switch(hdr->header.requestid) {
   case kXR_admin:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.admin.reserved = ",
             (kXR_int32)sizeof(hdr->admin.reserved));
      break;

   case kXR_auth:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.auth.reserved = ",
             (kXR_int32)sizeof(hdr->auth.reserved));

      printf("  ClientHeader.auth.credtype= 0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             hdr->auth.credtype[0],
             hdr->auth.credtype[1],
             hdr->auth.credtype[2],
             hdr->auth.credtype[3]);
      break;

   case kXR_chmod:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.chmod.reserved = ",
             (kXR_int32)sizeof(hdr->chmod.reserved));

      printf("  ClientHeader.chmod.mode= 0x%.2x 0x%.2x \n",
             *((kXR_char *)&hdr->chmod.mode),
             *(((kXR_char *)&hdr->chmod.mode)+1)
         );
      break;

   case kXR_close:
      printf("%40s0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             "ClientHeader.close.fhandle = ",
             hdr->close.fhandle[0],
             hdr->close.fhandle[1],
             hdr->close.fhandle[2],
             hdr->close.fhandle[3]);

      printf("%40s0 repeated %d times\n",
             "ClientHeader.close.reserved = ",
             (kXR_int32)sizeof(hdr->close.reserved));
      break;

   case kXR_dirlist:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.dirlist.reserved = ",
             (kXR_int32)sizeof(hdr->dirlist.reserved));
      break;

   case kXR_login:
      printf("%40s%d \n",
             "ClientHeader.login.pid = ",
             hdr->login.pid);

      printf("%40s%s\n",
             "ClientHeader.login_body.username = ",
             hdr->login.username);

      printf("%40s0 repeated %d times\n",
             "ClientHeader.login.reserved = ",
             (kXR_int32)sizeof(hdr->login.reserved));

      printf("%40s%d\n",
             "ClientHeader.login.role = ",
             (kXR_int32)hdr->login.role);
      break;

   case kXR_mkdir:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.mkdir.reserved = ",
             (kXR_int32)sizeof(hdr->mkdir.reserved));

      printf("%40s0x%.2x 0x%.2x\n",
             "ClientHeader.mkdir.mode = ",
             *((kXR_char*)&hdr->mkdir.mode),
             *(((kXR_char*)&hdr->mkdir.mode)+1)
         );
      break;

   case kXR_mv:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.mv.reserved = ",
             (kXR_int32)sizeof(hdr->mv.reserved));
      break;

   case kXR_open:
      printf("%40s0x%.2x 0x%.2x\n",
             "ClientHeader.open.mode = ",
             *((kXR_char*)&hdr->open.mode),
             *(((kXR_char*)&hdr->open.mode)+1)
         );

      printf("%40s0x%.2x 0x%.2x\n",
             "ClientHeader.open.options = ",
             *((kXR_char*)&hdr->open.options),
             *(((kXR_char*)&hdr->open.options)+1));

      printf("%40s0 repeated %d times\n",
             "ClientHeader.open.reserved = ",
             (kXR_int32)sizeof(hdr->open.reserved));
      break;

   case kXR_ping:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.ping.reserved = ",
             (kXR_int32)sizeof(hdr->ping.reserved));
      break;

   case kXR_protocol:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.protocol.reserved = ",
             (kXR_int32)sizeof(hdr->protocol.reserved));
      break;

   case kXR_prepare:
      printf("%40s0x%c\n",
             "ClientHeader.prepare.options = ",
             hdr->prepare.options);
      printf("%40s0x%c\n",
             "ClientHeader.prepare.prty = ",
             hdr->prepare.prty);
      printf("%40s0 repeated %d times\n",
             "ClientHeader.prepare.reserved = ",
             (kXR_int32)sizeof(hdr->prepare.reserved));
      break;

   case kXR_read:
      printf("%40s0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             "ClientHeader.read.fhandle = ",
             hdr->read.fhandle[0],
             hdr->read.fhandle[1],
             hdr->read.fhandle[2],
             hdr->read.fhandle[3]);

      printf("%40s%lld\n",
             "ClientHeader.read.offset = ",
             hdr->read.offset);

      printf("%40s%d\n",
             "ClientHeader.read.rlen = ",
             hdr->read.rlen);
      break;

   case kXR_rm:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.rm.reserved = ",
             (kXR_int32)sizeof(hdr->rm.reserved));

      break;

   case kXR_rmdir:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.rmdir.reserved = ",
             (kXR_int32)sizeof(hdr->rmdir.reserved));
      break;

   case kXR_set:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.set.reserved = ",
             (kXR_int32)sizeof(hdr->set.reserved));
      break;

   case kXR_stat:
      printf("%40s0 repeated %d times\n",
             "ClientHeader.stat.reserved = ",
             (kXR_int32)sizeof(hdr->stat.reserved));
      break;

   case kXR_sync:
      printf("%40s0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             "ClientHeader.sync.fhandle = ",
             hdr->sync.fhandle[0],
             hdr->sync.fhandle[1],
             hdr->sync.fhandle[2],
             hdr->sync.fhandle[3]);

      printf("%40s0 repeated %d times\n",
             "ClientHeader.sync.reserved = ",
             (kXR_int32)sizeof(hdr->sync.reserved));
      break;

   case kXR_write:
      printf("%40s0x%.2x 0x%.2x 0x%.2x 0x%.2x \n",
             "ClientHeader.write.fhandle = ",
             hdr->write.fhandle[0],
             hdr->write.fhandle[1],
             hdr->write.fhandle[2],
             hdr->write.fhandle[3]);

      printf("%40s%lld\n",
             "ClientHeader.write.offset = ",
             hdr->write.offset);

      printf("%40s0 repeated %d times\n",
             "ClientHeader.write.reserved = ",
             (kXR_int32)sizeof(hdr->write.reserved));
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
   case kXR_ok:
      printf("%30skXR_ok",
             "ServerHeader.status = ");
      break;
   case kXR_attn:
      printf("%30skXR_attn",
             "ServerHeader.status = ");
      break;
   case kXR_authmore:
      printf("%30skXR_authmore",
             "ServerHeader.status = ");
      break;
   case kXR_error:
      printf("%30skXR_error",
             "ServerHeader.status = ");
      break;
   case kXR_oksofar:
      printf("%30skXR_oksofar",
             "ServerHeader.status = ");
      break;
   case kXR_redirect:
      printf("%30skXR_redirect",
             "ServerHeader.status = ");
      break;
   case kXR_wait:
      printf("%30skXR_wait",
             "ServerHeader.status = ");
      break;
   }
   printf(" (%d)\n", hdr->status);
   printf("%30s%d",
          "ServerHeader.dlen = ", hdr->dlen);
   printf("\n========== END DUMPING SERVER HEADER ===========\n\n");
}

} // namespace ROOT
