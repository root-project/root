// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXNetConn                                                            //
//                                                                      //
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
// High level handler of connections to xrootd.                         //
// Instantiated by TXNetFile.                                           // 
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TError.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TRegexp.h"
#include "TObjString.h"

#include "TXDebug.h"
#include "TXConnectionMgr.h"
#include "TXNetConn.h"
#include "TXPhyConnection.h"
#include "TXProtocol.h"

#include "XrdSec/XrdSecInterface.hh"
#include "TUrl.h"

#include <stdio.h>      // needed by printf
#include <stdlib.h>     // needed by getenv()
#include <pwd.h>        // needed by getpwuid()
#include <sys/types.h>  // needed by getpid()
#include <unistd.h>     // needed by getpid() and getuid()
#include <string.h>     // needed by memcpy() and strcspn()
#include <ctype.h>
//#include <netinet/in.h> // needed by htonl()
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

extern TEnv *gEnv;

//
// Implementation of TXPhyConnLocker
//
//_____________________________________________________________________________
TXPhyConnLocker::TXPhyConnLocker(TXPhyConnection *phyc)
{
   // Constructor
   phyconn = phyc;
   phyconn->LockChannel();
}

//_____________________________________________________________________________
TXPhyConnLocker::~TXPhyConnLocker()
{
   // Destructor. 
   phyconn->UnlockChannel();
}

//_____________________________________________________________________________
void ParseRedir(TXMessage* xmsg, Int_t &port, TString &host, TString &token)
{
   // Small utility function... we want to parse the content
   // of a redir response from the server.

   Ssiz_t pos;

   // Remember... an instance of TXMessage automatically 0-terminates the
   // data if present
   struct ServerResponseBody_Redirect* redirdata =
      (struct ServerResponseBody_Redirect*)xmsg->GetData();

   port = 0;

   if (redirdata) {

      host = redirdata->host;
      token = "";
      if ( (pos = host.First('?')) != kNPOS ) {
         token = host[pos+1];
         host.Remove(pos);
      }
      port = net2host(redirdata->port);
   }
}

//_____________________________________________________________________________
TXNetConn::TXNetConn(): fOpenError((XErrorCode)0), fConnected(kFALSE), 
                        fLastNetopt(0), fLBSUrl(0), fUrl("")
{
   // Constructor

   fClientHostDomain = GetDomainToMatch(gSystem->HostName());

   if (strlen(fClientHostDomain.Data()) == 0)
      Warning("TXNetConn", "Error resolving this host's (%s) domain name.",
                           gSystem->HostName());
  
   fRedirHandler = 0;

   // Init the redirection counter parameters
   fGlobalRedirLastUpdateTimestamp = time(0);
   fGlobalRedirCnt = 0;
   fMaxGlobalRedirCnt = gEnv->GetValue("XNet.MaxRedirectCount",
                                       DFLT_MAXREDIRECTCOUNT);
}

//_____________________________________________________________________________
TXNetConn::~TXNetConn()
{
   // Destructor
}

//_____________________________________________________________________________
Short_t TXNetConn::Connect(TString Host, const Int_t Port, Int_t netopt)
{
   // Connect method (called the first time when TXNetFile is first created, 
   // and used for each redirection). The global static connection manager 
   // object is firstly created here. If another TXNetFile object is created
   // inside the same application this connection manager will be used and
   // no new one will be created.
   // No login/authentication are performed at this stage.

   fLastNetopt = netopt;

   // We try to connect to the host. What we get is the logical conn id
   Short_t logid;
   logid = -1;

   logid = ConnectionManager->Connect(Host.Data(), Port, netopt);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("Connect", "Connect(%s, %d, %d) returned %d",
                      Host.Data(), Port, netopt, logid );

   if (logid < 0) {
      Error("TXNetFile", "Error creating logical connection with [%s:%d]", 
                         Host.Data(), Port );
      SetLogConnID(logid);
      fConnected = kFALSE;
      return -1;
   }

   fConnected = kTRUE;

   SetLogConnID(logid);
   return logid;
}

//_____________________________________________________________________________
void TXNetConn::Disconnect(Bool_t ForcePhysicalDisc)
{
   // Disconnect

   ConnectionManager->Disconnect(GetLogConnID(), ForcePhysicalDisc);
   fConnected = kFALSE;
}

//_____________________________________________________________________________
TXMessage *TXNetConn::ClientServerCmd(ClientRequest *req, const void *reqMoreData,
                                      void **answMoreDataAllocated,
                                      void *answMoreData, Bool_t HasToAlloc) 
{
   // ClientServerCmd tries to send a command to the server and to get a response.
   // Here the kXR_redirect is handled, as well as other things.
   //
   // If the calling function requests the memory allocation (HasToAlloc is true) 
   // then:  
   //  o answMoreDataAllocated is filled with a pointer to the new block.
   //  o The caller MUST free it when it's no longer used if 
   //    answMoreDataAllocated is 0 
   //    then the caller is not interested in getting the data.
   //  o We must anyway read it from the stream and throw it away.
   //
   // If the calling function does NOT request the memory allocation 
   // (HasToAlloc is false) then: 
   //  o answMoreData is filled with the data read
   //  o the caller MUST be sure that the arriving data will fit into the
   //  o passed memory block
   //
   // We need to do this here because the calling func *may* not know the size 
   // to allocate for the request to be submitted. For instance, for the kXR_read
   // cmd the size is known, while for the kXR_getfile cmd is not.

   Int_t len;
   ClientRequest reqtmp;

   size_t TotalBlkSize = 0;

   void *tmpMoreData;
   XReqErrorType errorType = kOK;

   TXMessage *xmsg = 0;

   // In the case of an abort due to errors, better to return
   // a blank struct. Also checks the validity of the pointer.
   // memset(answhdr, 0, sizeof(answhdr));

   // Cycle for redirections...
   do {

      // Send to the server the request
      len = sizeof(ClientRequest);

      // We have to unconditionally set the streamid inside the
      // header, because, in case of 'rebouncing here', the Logical Connection 
      // ID might have changed, while in the header to write it remained the 
      // same as before, not valid anymore
      SetSID(req->header.streamid);

      reqtmp = *req;

      if (DebugLevel() >= TXDebug::kDUMPDEBUG) 
         ROOT::smartPrintClientHeader(&reqtmp);

      ROOT::clientMarshall(&reqtmp);

      errorType = WriteToServer(&reqtmp, req, reqMoreData, fLogConnID);
      
      // Read from server the answer
      // Note that the answer can be composed by many reads, in the case that
      // the status field of the responses is kXR_oksofar

      TotalBlkSize = 0;

      // A temp pointer to the mem block growing across the multiple kXR_oksofar
      tmpMoreData = 0;
      if ((answMoreData != 0) && !HasToAlloc)
         tmpMoreData = answMoreData;
      
      // Cycle for the kXR_oksofar i.e. partial answers to be collected
      do {

         TXNetConn::EThreeStateReadHandler whatToDo;
         xmsg = ReadPartialAnswer(errorType, TotalBlkSize, req, HasToAlloc,
                                  &tmpMoreData, whatToDo);

         if (whatToDo == kTSRHReturnNullMex)
            return 0;
         if (whatToDo == kTSRHReturnMex)
            return xmsg;
	
         if (xmsg && (xmsg->HeaderStatus() == kXR_oksofar) && 
                     (xmsg->DataLen() == 0))
            return xmsg;
	
      } while (xmsg && (xmsg->HeaderStatus() == kXR_oksofar));

   } while ((fGlobalRedirCnt < fMaxGlobalRedirCnt) &&
            xmsg && (xmsg->HeaderStatus() == kXR_redirect)); 

   // We collected all the partial responses into a single memory block.
   // If the block has been allocated here then we must pass its address
   if (HasToAlloc && (answMoreDataAllocated)) {
      *answMoreDataAllocated = tmpMoreData;
   }

   // We might have collected multiple partial response also in a given mem block
   if (xmsg)
      xmsg->fHdr.dlen = TotalBlkSize;

   return xmsg;
}

//_____________________________________________________________________________
Bool_t TXNetConn::SendGenCommand(ClientRequest *req, const void *reqMoreData,
				 void **answMoreDataAllocated, 
                                 void *answMoreData, Bool_t HasToAlloc,
                                 char *CmdName,
                                 struct ServerResponseHeader *srh)
{
   // SendGenCommand tries to send a single command for a number of times 

   Short_t retry = 0;
   Bool_t resp = kFALSE, abortcmd = kFALSE;

   // if we're going to open a file for the 2nd time we should reset fOpenError, 
   // just in case...
   if (req->header.requestid == kXR_open)
      fOpenError = (XErrorCode)0;

   while (!abortcmd && !resp) {
      abortcmd = kFALSE;

      // Send the cmd, dealing automatically with redirections and
      // redirections on error
      if (DebugLevel() >= TXDebug::kHIDEBUG) 
         Info("SendGenCommand","Calling ClientServerCmd...");

      TXMessage *cmdrespMex = ClientServerCmd(req, reqMoreData,
                                              answMoreDataAllocated, 
                                              answMoreData, HasToAlloc);

      // Save server response header if requested
      if (srh)
         memcpy(srh,&cmdrespMex->fHdr,sizeof(struct ServerResponseHeader));

      // Check for the redir count limit
      if (fGlobalRedirCnt >= fMaxGlobalRedirCnt) {
         Error("SendGenCommand",
               "Too many redirections for request [%s]. Aborting command.", 
               ROOT::convertRequestIdToChar(req->header.requestid));
         abortcmd = kTRUE;
      }
      else {

         // On serious communication error we retry for a number of times,
         // waiting for the server to come back
         if (!cmdrespMex || cmdrespMex->IsError()) {
            if (DebugLevel() >= TXDebug::kHIDEBUG)
               Info("SendGenCommand", "Communication error detected with [%s:%d].", 
                     fUrl.GetHost(), fUrl.GetPort());

            // For the kxr_open request we don't rely on the count limit of other
            // reqs. The open request is bounded only by the redir count limit
            if (req->header.requestid != kXR_open) 
               retry++;

            if (retry > kXR_maxReqRetry) {
               Error("SendGenCommand",
                     "Too many errors communication errors with server"
                     " [%s:%d] for request [%s]. Aborting command.", 
                     fUrl.GetHost(), fUrl.GetPort(), CmdName);
               abortcmd = kTRUE;
            } else
               abortcmd = kFALSE;
         } else {

	    // We are here if we got an answer for the command, so
	    // the server (original or redirected) is alive
	    resp = CheckResp(&cmdrespMex->fHdr, CmdName);
	    retry++;
	    
	    // If the answer was not (or not totally) positive, we must 
            // investigate on the result
	    if (!resp)
               abortcmd = CheckErrorStatus(cmdrespMex, retry, CmdName);
	    
	    if (retry > kXR_maxReqRetry) {
               Error("SendGenCommand",
                     "Too many errors messages from server [%s:%d] for"
                     " request [%s]. Aborting command.", 
                     fUrl.GetHost(), fUrl.GetPort(), CmdName);
               abortcmd = kTRUE;
	    }
         } // else... the case of a correct server response but declaring an error
      }

      SafeDelete(cmdrespMex);
   } // while

   return (!abortcmd);
}

//_____________________________________________________________________________
Bool_t TXNetConn::CheckHostDomain(TString hostToCheck, TString allow, TString deny)
{
   // Checks domain matching

   TString domain;
   TRegexp reAllow(""), reDeny("");

   // Get the domain for the url to check
   domain = GetDomainToMatch(hostToCheck);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("CheckHostDomain", "Resolved [%s]'s domain name into [%s]",
	   hostToCheck.Data(), domain.Data());

   // If we are unable to get the domain for the url to check --> access denied to it
   if (!domain.Length()) {
      Error("CheckHostDomain",
            "Error resolving domain name for %s. Denying access.",
	    hostToCheck.Data());
      return kFALSE;
   }

   // Given a list of |-separated regexps for the hosts to DENY, 
   // match every entry with domain. If any match is found, deny access.
   Short_t pos;

   deny += "|";
   while(deny.Length() > 0) {
      pos = deny.First('|');
      if (pos != kNPOS) {
	 TString tmp(deny);

	 tmp.Remove(pos);
	 deny.Remove(0, pos+1);

	 reDeny = tmp;

	 // if the domain matches any regexp for the domains to deny --> access denied
	 if (domain.Index(reDeny) != kNPOS) {
	    Error("CheckHostDomain",
		  "Access denied to the domain of [%s] (expr: [%s]).",
                  hostToCheck.Data(),tmp.Data());
	    return kFALSE;
	 }
      }
   }

   // Given a list of |-separated regexps for the hosts to ALLOW, 
   // match every entry with domain. If any match is found, grant access.

   allow += "|";
   while(allow.Length() > 0) {
      pos = allow.First('|');
      if (pos != kNPOS) {
	 TString tmp(allow);

	 tmp.Remove(pos);
	 allow.Remove(0, pos+1);

	 reAllow = tmp;

	 // if the domain matches any regexp for the domains to allow --> access granted
	 if (domain.Index(reAllow) != kNPOS) {
	    if (DebugLevel() >= TXDebug::kHIDEBUG)
	       Info("CheckHostDomain",
                    "Access granted to the domain of [%s] (expr: [%s]).",
		    hostToCheck.Data(), tmp.Data());
	    return kTRUE;
	 }
      }
   }

   Error("CheckHostDomain",
	 "Access to domain %s is not allowed nor denied. Not Allowed.",
          domain.Data());

   return kFALSE;
}

//_____________________________________________________________________________
Bool_t TXNetConn::CheckResp(struct ServerResponseHeader *resp, const char *method)
{
   // Checks if the server's response is the ours.
   // If the response's status is "OK" returns kTRUE; if the status is "redirect", it 
   // means that the max number of redirections has been achieved, so returns kFALSE.

   if (MatchStreamid(resp)) {

      // ok the response belongs to me
      if (resp->status == kXR_redirect) {
         // too many redirections. Exit!
         Error(method, "Too many redirections. System error.");
         return kFALSE;
      }

      if (resp->status != kXR_ok) {
         if (resp->status != kXR_wait)
            Error(method, "Server [%s:%d] did not return OK message for"
                          " last request.", fUrl.GetHost(), fUrl.GetPort());
         return kFALSE;
      }
      return kTRUE;

   } else {
      Error(method, "The return message doesn't belong to this client."
                    " Program exits.");
      return kFALSE;
   }
}

//_____________________________________________________________________________
Bool_t TXNetConn::MatchStreamid(struct ServerResponseHeader *ServerResponse)
{
   // Check stream ID matching

   char sid[2];

   memcpy(sid, &fLogConnID, sizeof(sid));

   // Matches the streamid contained in the server's response with the ours
   return (memcmp(ServerResponse->streamid, sid, sizeof(sid)) == 0 );
}

//_____________________________________________________________________________
void TXNetConn::SetSID(kXR_char *sid) {
   // Set our stream id, to match against that one in the server's response.

   memcpy((void *)sid, (const void*)&fLogConnID, 2);
}

//_____________________________________________________________________________
void TXNetConn::Streamer(TBuffer &R__b)
{
   // Dummy Streamer; rootcint chokes trying to generate one, but ROOT wants
   // one to load the shared library. 
   // Stream an object of class TXNetFile.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      R__b.CheckByteCount(R__s, R__c, TXNetConn::IsA());
   } else {
      R__c = R__b.WriteVersion(TXNetConn::IsA(), kTRUE);
      R__b.SetByteCount(R__c, kTRUE);
   }
}

//_____________________________________________________________________________
XReqErrorType TXNetConn::WriteToServer(ClientRequest *reqtmp, ClientRequest *req, 
				       const void* reqMoreData, Short_t LogConnID) 
{
   // Send message to server

   // Strong mutual exclusion over the physical channel
   // Note that we consider this the beginning of an atomic transaction
   // Also note that the lock is removed at the end of the block (Stroustroup 
   // page 365)
   {
      TXPhyConnLocker pcl(ConnectionManager->GetConnection(fLogConnID)
                                           ->GetPhyConnection());

      // Now we write the request to the logical connection through the
      // connection manager

      Short_t len = sizeof(req->header);

      Int_t writeCount = ConnectionManager->WriteRaw(LogConnID, reqtmp, len,
                                                     kDefault);
      fLastDataBytesSent = req->header.dlen;
  
      // A complete communication failure has to be handled later, but we
      // don't have to abort what we are doing
      if ((writeCount < 0) || (writeCount < len)) {
         Error("WriteToServer","Error sending %d bytes in the header part"
               " to server [%s:%d]. writeCount=%d.", 
               len, fUrl.GetHost(), fUrl.GetPort(), writeCount);
         return kWRITE;
      }

      // Send to the server the data.
      // If we got an error we can safely skip this... no need to get more
      if (req->header.dlen > 0) {

         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("WriteToServer",
                 "Sending %d bytes of DATA to the server [%s:%d].", 
                 req->header.dlen, fUrl.GetHost(), fUrl.GetPort());
    
         // Now we write the data associated to the request. Through the
         //  connection manager
         writeCount = ConnectionManager->WriteRaw(LogConnID, reqMoreData,
                                                  req->header.dlen, kDefault);
    
         // A complete communication failure has to be handled later, but we
         //  don't have to abort what we are doing
         if ((writeCount < 0) || (writeCount != req->header.dlen)) {
            Error("WriteToServer", "Error sending %d bytes in the data part"
                                   " to server [%s:%d]. writeCount=%d.", 
                  req->header.dlen, fUrl.GetHost(), fUrl.GetPort(), writeCount);
            return kWRITE;
         }
      }
      fLastDataBytesSent = req->header.dlen;
      return kOK;
   }
}

//_____________________________________________________________________________
Bool_t TXNetConn::CheckErrorStatus(TXMessage *mex, Short_t &Retry, char *CmdName)
{
   // Check error status

   if (mex->HeaderStatus() == kXR_redirect) {
      // Too many redirections
      Error("SendGenCommand","Max redirection count reached for request"
                             " [%s]. Aborting command.", CmdName);
      return kTRUE;
   }
  
   if (mex->HeaderStatus() == kXR_error) {
      // The server declared an error. 
      // In this case it's better to exit, unhandled error

      struct ServerResponseBody_Error *body_err;

      body_err = (struct ServerResponseBody_Error *)mex->GetData();

      if (body_err) {
         // Print out the error information, as received by the server
         Error("SendGenCommand", "Server declared error %d: '%s'", 
               net2host(body_err->errnum), (const char*)body_err->errmsg);
         fOpenError = (XErrorCode)net2host(body_err->errnum);
      }
      return kTRUE;
   }
    
   if (mex->HeaderStatus() == kXR_wait) {
      // We have to wait for a specified number of seconds and then
      // retry the same cmd

      struct ServerResponseBody_Wait *body_wait;

      body_wait = (struct ServerResponseBody_Wait *)mex->GetData();
    
      if (body_wait) {
         if (DebugLevel() >= TXDebug::kUSERDEBUG) {
            if (mex->DataLen() > 4) 
               Info("SendGenCommand", "Server [%s:%d] requested %d seconds"
                    " of wait. Server message is %s", 
                    fUrl.GetHost(), fUrl.GetPort(), net2host(body_wait->seconds),
                    (const char*)body_wait->infomsg);
            else
               Info("SendGenCommand", "Server [%s:%d] requested %d seconds"
                    " of wait.", 
                    fUrl.GetHost(), fUrl.GetPort(), net2host(body_wait->seconds));
         }
         gSystem->Sleep(1000 * net2host(body_wait->seconds));
      }

      // We don't want kxr_wait to count as an error
      Retry--;
      return kFALSE;
   }
    
   // We don't understand what the server said. Better investigate on it...
   Warning("SendGenCommand", 
           "Answer from server [%s:%d] not recognized after executing %s.", 
           fUrl.GetHost(), fUrl.GetPort(), CmdName);

   return kTRUE;
}

//_____________________________________________________________________________
TXMessage *TXNetConn::ReadPartialAnswer(XReqErrorType &errorType,
                                        size_t &TotalBlkSize, 
                                        ClientRequest *req,  
                                        Bool_t HasToAlloc, void** tmpMoreData,
                                        EThreeStateReadHandler &what_to_do)
{
   // Read server answer

   Int_t len;
   TXMessage *Xmsg = 0;
   void *tmp2MoreData;

   // No need to actually read if we are in error...
   if (errorType == kOK) {
    
      len = sizeof(ServerResponseHeader);
      if(DebugLevel() >= TXDebug::kHIDEBUG)
         Info("ReadPartialAnswer",
              "Reading a TXMessage from the server [%s:%d]...",
              fUrl.GetHost(), fUrl.GetPort());
    
      // A complete communication failure has to be handled later, but we
      //  don't have to abort what we are doing
    
      // Beware! Now Xmsg contains ALSO the information about the esit of
      // the communication at low level.
      Xmsg = ConnectionManager->ReadMsg(fLogConnID, kDefault);

      if(Xmsg)
         fLastDataBytesRecv = Xmsg->DataLen();
      else 
         fLastDataBytesRecv = 0;

      if ( !Xmsg || (Xmsg->IsError()) ) {
         Error("ReadPartialAnswer", "Error reading msg from connmgr"
               " (server [%s:%d]).", fUrl.GetHost(), fUrl.GetPort());
         if (HasToAlloc) {
            if (*tmpMoreData)
               free(*tmpMoreData);
            *tmpMoreData = 0;
         }
         errorType = kREAD;
      }
      else
         // is not necessary because the Connection Manager unmarshall the mex
         Xmsg->Unmarshall(); 
   }

   if (DebugLevel() >= TXDebug::kDUMPDEBUG)
      if (Xmsg != 0)
         ROOT::smartPrintServerHeader(&Xmsg->fHdr);

   // Now we have all the data. We must copy it back to the buffer where
   // they are needed, only if we are not in troubles with errorType
   if ((errorType == kOK) && (Xmsg->DataLen() > 0)) {
    
      // If this is a redirection answer, its data MUST NOT overwrite 
      // the given buffer
      if ( (Xmsg->HeaderStatus() == kXR_ok) ||
           (Xmsg->HeaderStatus() == kXR_oksofar) ||
           (Xmsg->HeaderStatus() == kXR_authmore) ) 
      {
         // Now we allocate a sufficient memory block, if needed
         // If the calling function passed a null pointer, then we 
         // fill it with the new pointer, otherwise the func knew
         // about the size of the expected answer, and we use
         // the given pointer.
         // We need to do this here because the calling func *may* not 
         // know the size to allocate
         // For instance, for the ReadBuffer cmd the size is known, while 
         // for the ReadFile cmd is not
         if (HasToAlloc) {
            tmp2MoreData = realloc(*tmpMoreData, TotalBlkSize + Xmsg->DataLen());
            if (!tmp2MoreData) {
               Error("ReadPartialAnswer", "Error reallocating %d bytes", 
                     TotalBlkSize);
               free(*tmpMoreData);
               *tmpMoreData = 0;
               what_to_do = kTSRHReturnNullMex;
               return 0;
            }
            *tmpMoreData = tmp2MoreData;
         }
	
         // Now we copy the content of the Xmsg to the buffer where
         // the data are needed
         if (*tmpMoreData)
            memcpy(((kXR_char *)(*tmpMoreData)) + TotalBlkSize,
                     Xmsg->GetData(), Xmsg->DataLen());
	
         // Dump the buffer tmpMoreData
         if (DebugLevel() >= TXDebug::kDUMPDEBUG) {
            Info ("ReadPartialAnswer","Dumping read data...");
            for(Int_t jj = 0; jj < Xmsg->DataLen(); jj++) {
               printf("0x%.2x ", *( ((kXR_char *)Xmsg->GetData()) + jj ) );
               if ( !(jj % 10) ) printf("\n");
            }
         }
         TotalBlkSize += Xmsg->DataLen();
	
      } else {
         if (DebugLevel() >= TXDebug::kHIDEBUG) {
             Info("ReadPartialAnswer", 
                 "Server [%s:%d] did not answer OK. Resp status is [%s]", 
                 fUrl.GetHost(), fUrl.GetPort(),
                 ROOT::convertRespStatusToChar(Xmsg->fHdr.status));
         }
      }
   } // End of DATA reading
  
   // Now answhdr contains the server response. We pass it as is to the
   // calling function.
   // The only exception is that we must deal here with redirections.
   // If the server redirects us, then we
   //   add 1 to redircnt
   //   close the logical connection
   //   try to connect to the new destination.
   //   login/auth to the new destination (this can generate other calls
   //       to this method if it has been invoked by DoLogin!)
   //   Reopen the file if the current fhandle value is not null (this 
   //     can generate other calls to this method, not for the dologin 
   //     phase)
   //   resend the command
   //
   // Also a READ/WRITE error requires a redirection
   // 
   if ( (errorType == kREAD) || 
        (errorType == kWRITE) || 
        ROOT::isRedir(&Xmsg->fHdr) ) 
   {
      // this procedure can decide if return to caller or
      // continue with processing
      
      ESrvErrorHandlerRetval Return = HandleServerError(errorType, Xmsg, req);
      
      if (Return == kSEHRReturnMsgToCaller) {
         // The caller is allowed to continue its processing
         //  with the current Xmsg
         // Note that this can be a way to stop retrying
         //  e.g. if the resp in Xmsg is kxr_redirect, it means
         //  that the redir limit has been reached
         if (HasToAlloc) { 
            free(*tmpMoreData);
            *tmpMoreData = 0;
         }
	
         // Return the message to the client (SendGenCommand)
         what_to_do = kTSRHReturnMex;
         return Xmsg;
      }
      
      if (Return == kSEHRReturnNoMsgToCaller) {
         // There was no Xmsg to return, or the previous one
         //  has no meaning anymore
	
         // The caller will retry the cmd for some times,
         // If we are connected the attempt will go OK,
         //  otherwise the next retry will fail, causing a
         //  redir to the lb or a rebounce.
         if (HasToAlloc) { 
            free(*tmpMoreData);
            *tmpMoreData = 0;
         }
	
         SafeDelete(Xmsg);
         Xmsg = 0;

         what_to_do = kTSRHReturnMex;
         return Xmsg;
      }
   }

   what_to_do = kTSRHContinue;
   return Xmsg;
}

//_____________________________________________________________________________
Int_t TXNetConn::LastBytesSent(void)
{
   // Return number of bytes sent during last transaction

   return ConnectionManager->GetConnection(fLogConnID)->LastBytesSent();
}

//_____________________________________________________________________________
Int_t TXNetConn::LastBytesRecv(void)
{
   // Return number of bytes received during last transaction

   return ConnectionManager->GetConnection(fLogConnID)->LastBytesRecv();
}

//_____________________________________________________________________________
Int_t TXNetConn::LastDataBytesSent(void)
{
   // Return number of data bytes sent during last transaction

   return fLastDataBytesSent;
}

//_____________________________________________________________________________
Int_t TXNetConn::LastDataBytesRecv(void)
{
   // Return number of data bytes received during last transaction

   return fLastDataBytesRecv;
}

//_____________________________________________________________________________
Bool_t TXNetConn::GetAccessToSrv()
{
   // Gets access to the connected server. The login and authorization steps
   // are performed here (calling method DoLogin() that performs loggin-in
   // and calls DoAuthentication() ).
   // If the server redirects us, this is gently handled by the general
   // functions devoted to the handling of the server's responses.
   // Nothing is visible here, and nothing is visible from the other high
   // level functions.

   TXLogConnection *logconn = 0;

   // Now we are connected and we ask for the kind of the server
   ConnectionManager->GetConnection(fLogConnID)->GetPhyConnection()->LockChannel();
   SetServerType(DoHandShake(fLogConnID));
   ConnectionManager->GetConnection(fLogConnID)->GetPhyConnection()->UnlockChannel();

   // Now we can start the reader thread in the phyconn, if needed
   ConnectionManager->GetConnection(fLogConnID)->GetPhyConnection()->StartReader();

   switch (GetServerType()) {
   case kSTError:
      Info("GetAccessToSrv", "HandShake failed with server [%s:%d].",
                             fUrl.GetHost(), fUrl.GetPort());
      ConnectionManager->Disconnect(fLogConnID, kTRUE);
      return kFALSE;

   case TXNetConn::kSTNone: 
      Info("GetAccessToSrv", "The server on [%s:%d] is unknown", 
                             fUrl.GetHost(), fUrl.GetPort());
      ConnectionManager->Disconnect(fLogConnID, kTRUE);
      return kFALSE;

   case TXNetConn::kSTRootd: 
      if (DebugLevel() >= TXDebug::kHIDEBUG) {
         Info("GetAccessToSrv","Ok: the server on [%s:%d] is an old rootd."
                               " Turning ON back compatibility mode.",
                                fUrl.GetHost(), fUrl.GetPort());
      }
      break;

   case TXNetConn::kSTBaseXrootd: 
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("GetAccessToSrv", 
              "Ok: the server on [%s:%d] is a xrootd load balancer.",
              fUrl.GetHost(), fUrl.GetPort());

      logconn = ConnectionManager->GetConnection(fLogConnID);
      logconn->GetPhyConnection()->SetTTL(DLBD_TTL);// = DLBD_TTL;
      logconn->GetPhyConnection()->fServer = kBase;
      break;

   case TXNetConn::kSTDataXrootd: 
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("GetAccessToSrv", 
              "Ok, the server on [%s:%d] is a xrootd data server.",
              fUrl.GetHost(), fUrl.GetPort());
      logconn = ConnectionManager->GetConnection(fLogConnID);
      logconn->GetPhyConnection()->SetTTL(DATA_TTL);        // = DATA_TTL;
      logconn->GetPhyConnection()->fServer = kData;
      break;
   }

   // Execute a login if connected to a xrootd server. For an old rootd, 
   // TNetFile takes care of the login phase
   if (GetServerType() != TXNetConn::kSTRootd) {
      if (logconn->GetPhyConnection()->IsLogged() == kNo)
         return DoLogin();
      else {
         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("GetAccessToSrv", "Client already logged-in using this"
                 " physical channel (server [%s:%d]).",
                 fUrl.GetHost(), fUrl.GetPort());
         return kTRUE;
      }
   }
   else
      return kTRUE;
}

//_____________________________________________________________________________
TXNetConn::ServerType TXNetConn::DoHandShake(short int log)
{
   // Performs initial hand-shake with the server in order to understand which 
   // kind of server is there at the other side and to make the server know who 
   // we are (TXNetFile instead of an old TNetFile)
   struct ClientInitHandShake initHS;
   struct ServerInitHandShake xbody;
   ServerResponseType type;

   Int_t writeCount, readCount, len;
  
   // Set field in network byte order
   memset(&initHS, 0, sizeof(initHS));
   initHS.fourth = (kXR_int32)host2net((UInt_t)4);
   initHS.fifth  = (kXR_int32)host2net((UInt_t)2012);

   if (ConnectionManager->GetConnection(log)->GetPhyConnection()->fServer == kBase) {

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("DoHandShake",
              "The physical channel is already bound to a load balancer"
              " server [%s:%d]. No handshake is needed.",
              fUrl.GetHost(), fUrl.GetPort());

      if (!fLBSUrl || !strlen(fLBSUrl->GetUrl())) {
         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("DoHandShake", "Setting Load Balancer Server Url = \"%s\".",
                                fUrl.GetUrl() );

         // Save the url of load balancer server for future uses...
         fLBSUrl = new TUrl(fUrl.GetUrl());
         if(!fLBSUrl) {
            Error("DoHandShake","Fatal ERROR *** Object creation with new"
                  " failed ! Probable system resources exhausted.");
            gSystem->Abort();
         }
      }
      return kSTBaseXrootd;
   }
   if (ConnectionManager->GetConnection(log)->GetPhyConnection()->fServer == kData) {

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("DoHandShake",
              "The physical channel is already bound to the data server"
              " [%s:%d]. No handshake is needed.",
              fUrl.GetHost(), fUrl.GetPort());
      return kSTDataXrootd;
   }

   // Send to the server the initial hand-shaking message asking for the 
   // kind of server
   len = sizeof(initHS);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("DoHandShake",
           "HandShake step 1: Sending %d bytes to the server [%s:%d].",
           len, fUrl.GetHost(), fUrl.GetPort());

   writeCount = ConnectionManager->WriteRaw(log, &initHS, len, kDefault);

   if (writeCount != len) {
      Error("DoHandShake", "Error sending %d bytes to the server [%s:%d]", 
	    len, fUrl.GetHost(), fUrl.GetPort());
      return kSTError;
   }

   // Read from server the first 4 bytes
   len = sizeof(type);

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("DoHandShake",
           "HandShake step 2: Reading %d bytes from the server [%s:%d]." ,
           len, fUrl.GetHost(), fUrl.GetPort());
 
   //
   // Read returns the return value of TSocket->RecvRaw... that returns the 
   // return value of recv (unix low level syscall)
   //
   readCount = ConnectionManager->ReadRaw(log, &type, 
                                          len, kDefault); // Reads 4(2+2) bytes
               
   if (readCount != len) {
      Error("DoHandShake", "Error reading %d bytes from the server [%s:%d].", 
	    len, fUrl.GetHost(), fUrl.GetPort());
      return kSTError;
   }

   // to host byte order
   type = net2host(type);

   // Check if the server is the eXtended rootd or not, checking the value 
   // of type
   if (type == 0) { // ok, eXtended!
      len = sizeof(xbody);

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("DoHandShake", "HandShake step 3: Reading %d bytes from"
              " XRootd on server [%s:%d].", len, fUrl.GetHost(), fUrl.GetPort());

      readCount = ConnectionManager->ReadRaw(log, &xbody, 
                                             len, kDefault); // Read 12(4+4+4) bytes
      if (readCount != len) {
         Error("DoHandShake", "Error reading %d bytes from server [%s:%d]", 
               len, fUrl.GetHost(), fUrl.GetPort());
         return kSTError;
      }

      ROOT::ServerInitHandShake2HostFmt(&xbody);

      fServerProto = xbody.msgtype;
    
      // check if the eXtended rootd is a data server
      switch (xbody.msgval) {
      case kXR_DataServer:
         // This is a data server
         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("DoHandShake",
                 "Data server found on [%s:%d].", fUrl.GetHost(), fUrl.GetPort());
         return kSTDataXrootd;

      case kXR_LBalServer:
         // This is a load balancing server
         if (DebugLevel() >= TXDebug::kHIDEBUG)
            Info("DoHandShake", "Load balancer server found.");

         if (!fLBSUrl || !strlen(fLBSUrl->GetUrl())) {

            if (DebugLevel() >= TXDebug::kHIDEBUG)
               Info("DoHandShake", "Setting Load Balancer Server Url = \"%s\".",
                    fUrl.GetUrl() );

            // Save the url of load balancer server for future uses...
            fLBSUrl = new TUrl(fUrl.GetUrl());
            if (!fLBSUrl) {
               Error("DoHandShake","Fatal ERROR *** Object creation with"
                     " new failed ! Probable system resources exhausted.");
               gSystem->Abort();
            }
         }
         return TXNetConn::kSTBaseXrootd;

      default:
         // Unknown server type
         return kSTNone;
      }
   } else {
      // We are here if it wasn't an XRootd
      // and we need to complete the reading
      if (type == 8)
         return kSTRootd;
      else 
         // We dunno the server type
         return kSTNone;
   }
}

//_____________________________________________________________________________
Bool_t TXNetConn::DoLogin() 
{
   // This method perform the loggin-in into the server just after the
   // hand-shake. It also calls the DoAuthentication() method

   ClientRequest reqhdr;
   Bool_t resp;
  
   // We fill the header struct containing the request for login
   memset( &reqhdr, 0, sizeof(reqhdr));

   SetSID(reqhdr.header.streamid);
   reqhdr.header.requestid = kXR_login;
   reqhdr.login.pid = getpid();

   // Get username from Url
   TString User = fUrl.GetUser();
   if (!User.Length()) {
      // Use local username, if not specified
      UserGroup_t *u = gSystem->GetUserInfo();
      if (u)
         User = u->fUser;
      delete u;
   }
   if (User.Length())
      strcpy( (char *)reqhdr.login.username, (char *)(User.Data()) );
   else
      strcpy( (char *)reqhdr.login.username, "????" );

   // set the token with the value provided by a previous 
   // redirection (if any)
   reqhdr.header.dlen = fRedirInternalToken.Length(); 
  
   // We call SendGenCommand, the function devoted to sending commands. 
   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("DoLogin", "Logging into the server [%s:%d]. pid=%d uid=%s",
            fUrl.GetHost(), fUrl.GetPort(),
            reqhdr.login.pid, reqhdr.login.username);

   ConnectionManager->GetConnection(fLogConnID)->GetPhyConnection()
                                               ->SetLogged(kNo);

   // server response header
   struct ServerResponseHeader reshdr;
   char *plist = 0;
   resp = SendGenCommand(&reqhdr, fRedirInternalToken.Data(), 
                         (void **)&plist, 0, 
                         kTRUE, (char *)"XTNetconn::doLogin",&reshdr);

   // Check if we need to authenticate 
   if (reshdr.dlen && plist) {

      // Terminate server reply
      plist[reshdr.dlen]=0;

      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("DoLogin","server requires authentication");
      resp = DoAuthentication(User, plist);
   }

   // Flag success if everything went ok
   if (resp) 
      ConnectionManager->GetConnection(fLogConnID)->GetPhyConnection()
         ->SetLogged(kYes);
   if (plist)
      delete[] plist;

   return resp;

}

//_____________________________________________________________________________
Bool_t TXNetConn::DoAuthentication(const char *username, char *plist)
{
  // Negotiate authentication with the remote server. Tries in turn
  // all available protocols proposed by the server (in plist), 
  // starting from the first.

  if (!plist || !strlen(plist))
     return kTRUE;

  if (DebugLevel() >= TXDebug::kHIDEBUG)
     Info("DoAuthentication", "remote host: %s,"
          " list of available protocols: %s (%d)",
          fUrl.GetHost(), plist, strlen(plist));
 
  // Prepare host/IP information of the remote xrootd. This is required
  // for the authentication.
  //
  struct sockaddr_in netaddr;
  netaddr.sin_family = AF_INET;
  netaddr.sin_port   = 0;
  netaddr.sin_addr.s_addr = INADDR_ANY;

  UInt_t addr = htonl(gSystem->GetHostByName(fUrl.GetHost()).GetAddress());
  memcpy((void *)&netaddr.sin_addr.s_addr, &addr,
	   sizeof(netaddr.sin_addr.s_addr));
      
  // Variables for negotiation
  XrdSecParameters   secToken;
  XrdSecProtocol    *protocol;
  XrdSecCredentials *credentials;

  // Prepare tokenization
  //
  TString inPString((const char *)plist);
  TObjArray *inPList = inPString.Tokenize(" "); 

  // Make sure that we got at least one token
  //
  if (inPList->GetEntries() < 1) { 
     if (DebugLevel() >= TXDebug::kHIDEBUG)
        Warning("DoAuthentication", "Protocol list empty");
     return kFALSE;
  }

  // Now try in turn the available methods (first preferred)
  //
  Bool_t resp = kFALSE;
  TIter nxtoken(inPList);
  TObjString *tkn = 0;
  while (!resp && (tkn = (TObjString *)nxtoken())) {

     TString token = tkn->GetString();

     // Assign the security token that we have received at the login request
     //
     secToken.buffer = (char *)(token.Data());   
     secToken.size   = token.Length();
     
     // Retrieve the security protocol context from the xrootd server
     //
     protocol = XrdSecGetProtocol((const struct sockaddr &)netaddr,secToken, 0);
     if (!protocol) { 
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("DoAuthentication", 
                "Unable to get protocol object (token: %s)",token.Data());
        continue;
     }

     // Extract the protocol name (identifier)
     TString protname = "";
     if (!token.BeginsWith("&P=")) {
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Warning("DoAuthentication",
                   "Unable to get protocol name (token: %s)",token.Data());
     } else {
        protname = token;
        protname.ReplaceAll("&P=","");
        protname.Resize(protname.Index(","));
     }
     
     // Now we add the username and the hostname to the token, because
     // they may be needed to get the credentials
     secToken.size   = token.Length()+strlen(username)+strlen(fUrl.GetHost())+2;
     char *etoken    = new char[secToken.size+5];
     snprintf(etoken,secToken.size+5,"%s,%s,%s",token.Data(),username,fUrl.GetHost());
     secToken.buffer = etoken;
     
     // Once we have the protocol, get the credentials
     //
     credentials = protocol->getCredentials(&secToken);
     if (!credentials) {
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("DoAuthentication", 
                "Cannot obtain credentials (token: %s)",etoken);
        if (etoken) 
           delete[] etoken;
        continue;
     } else
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("DoAuthentication", "cred= %s size=%d",
                                    credentials->buffer, credentials->size);
     if (etoken)
        delete[] etoken;
     
     // We fill the header struct containing the request for login
     ClientRequest reqhdr;
     SetSID(reqhdr.header.streamid);
     reqhdr.header.requestid = kXR_auth;
     memset(reqhdr.auth.reserved, 0, 12);
     memcpy(reqhdr.auth.credtype, protname.Data(), protname.Length());
     
     struct ServerResponseHeader reshdr;
     reshdr.status = kXR_authmore;
     char *srvans = 0;
     
     resp = kFALSE;
     while (reshdr.status == kXR_authmore) {
        
        // Length of the credentials buffer
        reqhdr.header.dlen = credentials->size;
        
        resp = SendGenCommand(&reqhdr, credentials->buffer, 
                              (void **)&srvans, 0, kTRUE, 
                              (char *)"XTNetconn::DoAuthentication",&reshdr);
        if (DebugLevel() >= TXDebug::kHIDEBUG)
           Info("DoAuthenticate", "Server reply: status: %d dlen: %d",
                                  reshdr.status,reshdr.dlen);
     
        if (reshdr.status == kXR_authmore) {
           // We are required to send additional information
           // First assign the security token that we have received
           // at the login request
           //
           secToken.buffer = srvans;   
           secToken.size   = strlen(srvans);
     
           // then get next part of the credentials
           //
           credentials = protocol->getCredentials(&secToken);
           if (!credentials) {
              if (DebugLevel() >= TXDebug::kUSERDEBUG)
                 Info("DoAuthentication", 
                      "Cannot obtain credentials (token: %s)", srvans);
              break;
           } else {
              if (DebugLevel() >= TXDebug::kHIDEBUG)
                 Info("DoAuthentication", "cred= %s size=%d",
                      credentials->buffer, credentials->size);
           }
        }
        // Release buffer allocated for the server reply
        if (srvans)
           delete[] srvans;
     }
  }

  // Return the result of the negotiation
  //
  return resp;
}

//_____________________________________________________________________________
TXNetConn::ESrvErrorHandlerRetval
TXNetConn::HandleServerError(XReqErrorType &errorType, TXMessage *xmsg,
                             ClientRequest *req)
{
   // Handle errors from server

   Int_t newport; 	
   TString newhost; 	
   TString token;
  
   // Close the log connection at this point the fLogConnID is no longer valid.
   // On read/write error the physical channel may be not OK, so it's a good
   // idea to shutdown it.
   // If there are other logical conns pointing to it, they will get an error,
   // which will be handled
   if ((errorType == kREAD) || (errorType == kWRITE))
      ConnectionManager->Disconnect(fLogConnID, kTRUE);
   else
      ConnectionManager->Disconnect(fLogConnID, kFALSE);
  
   // We cycle repeatedly trying to ask the dlb for a working redir destination
   do {
    
      // Consider the timeout for the count of the redirections
      // this instance got in the last period of time
      if ( (time(0) - fGlobalRedirLastUpdateTimestamp) >  REDIRCNTTIMEOUT) {
         // REDIRCNTTIMEOUT is defined in XProtocol.hh
         fGlobalRedirCnt = 0;
         fGlobalRedirLastUpdateTimestamp = time(0);
      }

      // Anyway, let's update the counter, we have just been redirected
      fGlobalRedirCnt++;
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("HandleServerError","Redir count=%d", fGlobalRedirCnt);

      if ( fGlobalRedirCnt >= fMaxGlobalRedirCnt ) 
         return kSEHRContinue;
    
      newhost = "";
      newport = 0;
      token = "";
    
      if ((errorType == kREAD) || 
          (errorType == kWRITE) || 
          (errorType == kREDIRCONNECT)) {
         // We got some errors in the communication phase
         // the physical connection has been closed;
         // then we must go back to the load balancer
         // if there is any
         if ((0 != fLBSUrl) && (0 != strlen(fLBSUrl->GetUrl()))) {
            newhost = fLBSUrl->GetHost();
            newport = fLBSUrl->GetPort();
         }
         else {
            Error("HandleServerError",
                  "No Load Balancer to contact after a communication error"
                  " with server [%s:%d]. Rebouncing here.",
                  fUrl.GetHost(), fUrl.GetPort());
            newhost = fUrl.GetHost();
            newport = fUrl.GetPort();
         }
      
      } else if (ROOT::isRedir(&xmsg->fHdr)) {
         // No comm errors, but we got an explicit redir message      
         // If we did not meet a dlb before, we consider this as a dlb
         // to return to after an error
         if ((0 == fLBSUrl) || (0 == strlen(fLBSUrl->GetUrl()))) {
            if (DebugLevel() >= TXDebug::kHIDEBUG)
               Info("HandleServerError", 
                    "Setting Load Balancer Server Url = \"%s\".", fUrl.GetUrl() );

            // Save the url of load balancer server for future uses...
            fLBSUrl = new TUrl(fUrl.GetUrl());
            if (!fLBSUrl) {
               Error("HandleServerError",
                     "Fatal ERROR *** Object creation with new failed !"
                     " Probable system resources exhausted.");
               gSystem->Abort();
            }
         }
      
         // Extract the info (new host:port) from the response
         newhost = "";
         token   = "";
         newport = 0;
         ParseRedir(xmsg, newport, newhost, token);
      }
    
      // Now we should have the parameters needed for the redir
      // a member class 'internalToken' is needed because the host that 
      // answers with a kXR_redirect
      // message also provides a token that must be passed to the new host...
      fRedirInternalToken = token;

      CheckPort(newport);

      if ((newhost.Length() > 0) && newport) {

         if (DebugLevel() >= TXDebug::kUSERDEBUG)
            Info("HandleServerError",
                 "Received redirection to [%s:%d]. Token=[%s].", 
                 newhost.Data(), newport, fRedirInternalToken.Data());

         errorType = kOK;
      
	 TString quotedDomain = fClientHostDomain.Data();
	 
	 if ( !CheckHostDomain( newhost,
                                gEnv->GetValue("XNet.RedirDomainAllowRE",
                                               fClientHostDomain.Data()),
                                gEnv->GetValue("XNet.RedirDomainDenyRE",
                                               "<unknown>") ) ) {
	    Error("HandleServerError",
		  "Redirection to a server out-of-domain disallowed. Abort.");
	    gSystem->Abort();
	 }
         
         errorType = GoToAnotherServer(newhost, newport, fLastNetopt);
      }
      else {
         // Host or port are not valid or empty
         Error("HandleServerError", 
               "Received redirection to [%s:%d]. Token=[%s]. No server to go...",
               newhost.Data(), newport, fRedirInternalToken.Data());
         errorType = kREDIRCONNECT;
      }
    
      // We don't want to flood servers...
      if (errorType == kREDIRCONNECT)
         gSystem->Sleep(1000 * gEnv->GetValue("XNet.ReconnectTimeout",
                                              DFLT_RECONNECTTIMEOUT) );

      // We keep trying the connection to the same host (we have only one)
      //  until we are connected, or the max count for
      //  redirections is reached

   } while (errorType == kREDIRCONNECT);


   // We are here if correctly connected and handshaked and logged
   if (!IsConnected()) {
      Error("HandleServerError", 
            "Not connected. Internal error. Abort.");
      gSystem->Abort();
   }

   // If the former request was a kxr_open,
   //  there is no need to reissue it, since it will be the next attempt
   //  to rerun the cmd.
   // We simply return to the caller, which will retry
   // The same applies to kxr_login. No need to reopen a file if we are
   // just logging into another server.
   // The open request will surely follow if needed.
   if ((req->header.requestid == kXR_open) ||
       (req->header.requestid == kXR_login))  return kSEHRReturnNoMsgToCaller;

   // Here we are. If we had a filehandle then we must
   //  request a new one.
   char localfhandle[4];
   Bool_t wasopen;

   if (fRedirHandler &&
      (fRedirHandler->OpenFileWhenRedirected(localfhandle, wasopen) && wasopen)) {
      // We are here if the file has been opened succesfully
      // or if it was not open
      // Tricky thing: now we have a new filehandle, perhaps in
      // a different server. Then we must correct the filehandle in
      // the msg we were sending and that we must repeat...
      ROOT::PutFilehandleInRequest(req, localfhandle);
    
      // Everything should be ok here.
      // If we have been redirected,then we are connected, logged and reopened
      // the file. If we had a r/w error (xmsg==0 or xmsg->IsError) we are
      // OK too. Since we may come from a comm error, then xmsg can be null.
      if (xmsg && !xmsg->IsError())
         return kSEHRContinue; // the case of explicit redir
      else
         return kSEHRReturnNoMsgToCaller; // the case of recovered error
   }

   // We are here if we had no fRedirHandler or the reopen failed
   // If we have no fRedirHandler then treat it like an OK
   if (!fRedirHandler) {
      // Since we may come from a comm error, then xmsg can be null.
      //if (xmsg) xmsg->SetHeaderStatus( kXR_ok );
      if (xmsg && !xmsg->IsError())
         return kSEHRContinue; // the case of explicit redir
      else
         return kSEHRReturnNoMsgToCaller; // the case of recovered error
   }

   // We are here if we have been unable to connect somewhere to handle the
   //  troubled situation
   return kSEHRContinue;
}

//_____________________________________________________________________________
XReqErrorType TXNetConn::GoToAnotherServer(TString Newhost, 
                                           Int_t Newport, Int_t opt)
{
   // Re-directs to another server


   if ((fLogConnID = Connect( Newhost, Newport, opt)) == -1) {
    
      // Note: if Connect is unable to work then we are in trouble.
      // It seems that we have been redirected to a non working server
      Error("GoToAnotherServer", "Error connecting to [%s:%d]", 
            Newhost.Data(), Newport);
    
      // If no conn is possible then we return to the load balancer
      return kREDIRCONNECT;
   }
  
   //
   // Set fUrl to the new data/lb server if the 
   // connection has been succesfull
   //
   TString hostForNewUrl = "root://";
   hostForNewUrl += Newhost;
   hostForNewUrl += ":";
   hostForNewUrl += Newport;
   hostForNewUrl += fUrl.GetFile();
   fUrl = TUrl(hostForNewUrl.Data());

   if (IsConnected() && !GetAccessToSrv()) {
      Error("GoToAnotherServer", "Error handshaking to [%s:%d]", 
            Newhost.Data(), Newport);
      return kREDIRCONNECT;
   }
   return kOK;
}

//_____________________________________________________________________________
TString TXNetConn::GetDomainToMatch(TString hostname) {
   // Return net-domain of host hostname in 's'.
   // If the host is unknown in the DNS world but it's a
   //  valid inet address, then that address is returned, in order
   //  to be matched later for access granting

   TInetAddress addr;
   TString res, hnm;

   // Let's look up the hostname
   // It may also be a w.x.y.z type address.
   addr = gSystem->GetHostByName(hostname.Data());
   
   if (addr.IsValid()) {
      // The looked up address is valid
      // The hostname domain can still be unknown
     
      if (DebugLevel() >= TXDebug::kHIDEBUG)
	 Info("GetDomainToMatch", "GetHostByName(%s) returned name='%s' addr='%s'"
	      " port=%d.", hostname.Data(), addr.GetHostName(),
	      addr.GetHostAddress(), addr.GetPort());

      hnm = addr.GetHostName();

      res = ParseDomainFromHostname(hnm);

      if (hnm == "UnNamedHost")
	 res = addr.GetHostAddress();

   } else {
      if (DebugLevel() >= TXDebug::kHIDEBUG)
         Info("GetDomainToMatch", "GetHostByName(%s) returned a non valid address.",
              hostname.Data());
      res = "";
   }

   if (DebugLevel() >= TXDebug::kHIDEBUG)
      Info("GetDomainToMatch", "GetDomain(%s) --> '%s'.", hostname.Data(), res.Data());

   return res;
}

//_____________________________________________________________________________
TString TXNetConn::ParseDomainFromHostname(TString hostname) {

   TString res;
   Int_t pos;

   res = hostname;

   // Isolate domain
   pos = res.First('.');

   if (pos == kNPOS)
      res = "";
   else
      res.Remove(0, pos+1);

   return res;
}


//_____________________________________________________________________________
void TXNetConn::CheckPort(Int_t &port) {

   if(port <= 0) {

      if(DebugLevel() >= TXDebug::kHIDEBUG)
	 Warning("checkPort", 
		 "TCP port not specified. Trying to get it from /etc/services...");

      struct servent *S = getservbyname("rootd", "tcp");
      if(!S) {
	 if(DebugLevel() >= TXDebug::kHIDEBUG)
	    Warning("checkPort", "Service %s not specified in /etc/services. "
		    "Using default IANA tcp port 1094", "rootd");
	 port = 1094;
      } else {
	 Info("checkPort", "Found tcp port %d in /etc/service", ntohs(S->s_port));
	 port = (Int_t)ntohs(S->s_port);
      }

   }
}
