// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXNetConn
#define ROOT_TXNetConn

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

#define DFLT_MAXREDIRECTCOUNT        255
#define DFLT_DEBUG                   0
#define DFLT_RECONNECTTIMEOUT        10
#define DFLT_REQUESTTIMEOUT          60
#define REDIRCNTTIMEOUT		     3600

#include "time.h"
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TXAbsNetCommon
#include "TXAbsNetCommon.h"
#endif
#ifndef ROOT_TXMessage
#include "TXMessage.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

class TXNetConn : public TObject {

public:
   enum ServerType {
      kSTError      = -1,  // Some error occurred: server type undetermined 
      kSTNone       = 0,   // Remote server type un-recognized
      kSTRootd      = 1,   // Remote server type: old rootd server
      kSTBaseXrootd = 2,   // Remote server type: xrootd dynamic load balancer
      kSTDataXrootd = 3    // Remote server type: xrootd data server
   }; 
   enum ESrvErrorHandlerRetval {
      kSEHRReturnMsgToCaller   = 0,
      kSEHRBreakLoop           = 1,
      kSEHRContinue            = 2,
      kSEHRReturnNoMsgToCaller = 3,
      kSEHRRedirLimitReached   = 4
   };
   enum EThreeStateReadHandler {
      kTSRHReturnMex     = 0,
      kTSRHReturnNullMex = 1,
      kTSRHContinue      = 2
   };

   Int_t             fLastDataBytesRecv;
   Int_t             fLastDataBytesSent;
   XErrorCode        fOpenError;	

  
   TXNetConn();
   ~TXNetConn();

   Bool_t           CheckHostDomain(TString hostToCheck, TString allow, 
                                                         TString deny);
   Short_t          Connect(TString newHost, Int_t newPort, Int_t netopt);
   void             Disconnect(Bool_t ForcePhysicalDisc);
   Bool_t           GetAccessToSrv();
   TString          GetClientHostDomain() const { return fClientHostDomain; }
   Int_t            GetLogConnID() const { return fLogConnID; }
   TUrl            *GetLBSUrl() const { return fLBSUrl; }
   XErrorCode       GetOpenError() const { return fOpenError; }
   XReqErrorType    GoToAnotherServer(TString, Int_t, Int_t);
   Bool_t           IsConnected() const { return fConnected; }
   Int_t            LastBytesRecv();
   Int_t            LastBytesSent();
   Int_t            LastDataBytesRecv();
   Int_t            LastDataBytesSent();
   Bool_t           SendGenCommand(ClientRequest *req, 
                                   const void *reqMoreData,       
                                   void **answMoreDataAllocated,
                                   void *answMoreData, Bool_t HasToAlloc,
                                   char *CmdName,
                                   struct ServerResponseHeader *srh = 0);
   ServerType       GetServerType() const { return fServerType; }
   void             SetClientHostDomain(const char *src) 
                                                { fClientHostDomain = src; }
   void             SetConnected(Bool_t conn) { fConnected = conn; }
   void             SetLogConnID(Short_t logconnid) { fLogConnID = logconnid;}
   void             SetOpenError(XErrorCode err) { fOpenError = err; }
   void             SetRedirHandler(TXAbsNetCommon *rh) { fRedirHandler = rh; }
   void             SetServerType(ServerType type) { fServerType = type; }
   void             SetSID(kXR_char *sid);
   inline void      SetUrl(TUrl thisUrl) { fUrl = thisUrl; }

private:

   TString             fClientHostDomain; // Save the client's domain name
   Bool_t              fConnected;
   Short_t             fGlobalRedirCnt;    // Number of redirections
   time_t              fGlobalRedirLastUpdateTimestamp; // Timestamp of last redirection
   Int_t               fLastNetopt;
   TUrl               *fLBSUrl;            // Needed to save the load balancer url
   Short_t             fLogConnID;        // Logical connection ID of the current
                                          // TXNetFile object
   Short_t             fMaxGlobalRedirCnt;
   TXAbsNetCommon     *fRedirHandler;     // Pointer to a class inheriting from
                                         // TXAbsNetCommon providing methods
                                         // to handle the redir at higher level
   TString             fRedirInternalToken; // Token returned by the server when
                                           // redirecting
   Long_t              fServerProto;      // The server protocol
   ServerType          fServerType;       // Server type as returned by doHandShake() 
                                         // (see enum ServerType)
   TUrl                fUrl;


   Bool_t              CheckErrorStatus(TXMessage *, Short_t &, char *);
   void                CheckPort(Int_t &port);
   Bool_t              CheckResp(struct ServerResponseHeader *resp, const char *method);
   TXMessage          *ClientServerCmd(ClientRequest *req, const void *reqMoreData,
                                   void **answMoreDataAllocated, void *answMoreData,
                                   Bool_t HasToAlloc);
   Bool_t              DoAuthentication(const char *usr, char *list);
   ServerType          DoHandShake(Short_t log);
   Bool_t              DoLogin();

   TString             GetDomainToMatch(TString hostname);

   ESrvErrorHandlerRetval HandleServerError(XReqErrorType &, TXMessage *,
                                            ClientRequest *);
   Bool_t              MatchStreamid(struct ServerResponseHeader *ServerResponse);

   TString             ParseDomainFromHostname(TString hostname);

   TXMessage          *ReadPartialAnswer(XReqErrorType &, size_t &, 
                                         ClientRequest *, Bool_t, void**,
                                         EThreeStateReadHandler &);
   XReqErrorType       WriteToServer(ClientRequest *, ClientRequest *, 
                                         const void*, Short_t);
  
   ClassDef(TXNetConn, 1); //A high level connection class for TXNetAdmin.
};


//
// Class implementing a trick to automatically unlock a TXPhyConnection.
// Used only by TXNetConn.
//
class TXPhyConnection;

class TXPhyConnLocker {
private:
   TXPhyConnection *phyconn;

public:
   TXPhyConnLocker(TXPhyConnection *phyc);
   ~TXPhyConnLocker();
};

#endif
