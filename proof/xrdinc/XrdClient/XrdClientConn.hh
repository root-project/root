#ifndef XRD_CONN_H
#define XRD_CONN_H
/******************************************************************************/
/*                                                                            */
/*                    X r d C l i e n t C o n n . h h                         */
/*                                                                            */
/* Author: Fabrizio Furano (INFN Padova, 2004)                                */
/* Adapted from TXNetFile (root.cern.ch) originally done by                   */
/*  Alvise Dorigo, Fabrizio Furano                                            */
/*          INFN Padova, 2003                                                 */
/*                                                                            */
/* This file is part of the XRootD software suite.                            */
/*                                                                            */
/* XRootD is free software: you can redistribute it and/or modify it under    */
/* the terms of the GNU Lesser General Public License as published by the     */
/* Free Software Foundation, either version 3 of the License, or (at your     */
/* option) any later version.                                                 */
/*                                                                            */
/* XRootD is distributed in the hope that it will be useful, but WITHOUT      */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or      */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public       */
/* License for more details.                                                  */
/*                                                                            */
/* You should have received a copy of the GNU Lesser General Public License   */
/* along with XRootD in a file called COPYING.LESSER (LGPL license) and file  */
/* COPYING (GPL license).  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                            */
/* The copyright holder's institutional names and contributor's names may not */
/* be used to endorse or promote products derived from this software without  */
/* specific prior written permission of the institution or contributor.       */
/******************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// High level handler of connections to xrootd.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "XrdClient/XrdClientConst.hh"

#include "time.h"
#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientMessage.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClientReadCache.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

#define ConnectionManager XrdClientConn::GetConnectionMgr()

class XrdClientAbs;
class XrdSecProtocol;

class XrdClientConn {

public:

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

    // To keep info about an open session
    struct                     SessionIDInfo {
	char id[16];
    };

    int                        fLastDataBytesRecv;
    int                        fLastDataBytesSent;
    XErrorCode                 fOpenError;	

    XrdOucString               fRedirOpaque;        // Opaque info returned by the server when

    // redirecting. To be used in the next opens
    XrdClientConn();
    virtual ~XrdClientConn();

    inline bool                CacheWillFit(long long bytes) {
	if (!fMainReadCache)
	    return FALSE;
	return fMainReadCache->WillFit(bytes);
    }

    bool                       CheckHostDomain(XrdOucString hostToCheck);
    short                      Connect(XrdClientUrlInfo Host2Conn,
				       XrdClientAbsUnsolMsgHandler *unsolhandler);
    void                       Disconnect(bool ForcePhysicalDisc);
    virtual bool               GetAccessToSrv();
    XReqErrorType              GoBackToRedirector();

    XrdOucString               GetClientHostDomain() { return fgClientHostDomain; }


    static XrdClientPhyConnection     *GetPhyConn(int LogConnID);


    // --------- Cache related stuff

    long                       GetDataFromCache(const void *buffer,
						long long begin_offs,
						long long end_offs,
						bool PerfCalc,
						XrdClientIntvList &missingblks,
						long &outstandingblks );

    bool                       SubmitDataToCache(XrdClientMessage *xmsg,
						 long long begin_offs,
						 long long end_offs);

    bool                       SubmitRawDataToCache(const void *buffer,
						 long long begin_offs,
						 long long end_offs);

    void                       SubmitPlaceholderToCache(long long begin_offs,
							long long end_offs) {
	if (fMainReadCache)
	    fMainReadCache->PutPlaceholder(begin_offs, end_offs);
    }

  
    void                       RemoveAllDataFromCache(bool keepwriteblocks=true) {
        if (fMainReadCache)
            fMainReadCache->RemoveItems(keepwriteblocks);
    }

    void                       RemoveDataFromCache(long long begin_offs,
                                                   long long end_offs, bool remove_overlapped = false) {
        if (fMainReadCache)
	  fMainReadCache->RemoveItems(begin_offs, end_offs, remove_overlapped);
    }

    void                       RemovePlaceholdersFromCache() {
        if (fMainReadCache)
            fMainReadCache->RemovePlaceholders();
    }

    void                       PrintCache() {
        if (fMainReadCache)
            fMainReadCache->PrintCache();
    }


  bool                       GetCacheInfo(
					  // The actual cache size
					  int &size,

					  // The number of bytes submitted since the beginning
					  long long &bytessubmitted,

					  // The number of bytes found in the cache (estimate)
					  long long &byteshit,

					  // The number of reads which did not find their data
                                          // (estimate)
					  long long &misscount,

					  // miss/totalreads ratio (estimate)
					  float &missrate,

					  // number of read requests towards the cache
					  long long &readreqcnt,

					  // ratio between bytes found / bytes submitted
					  float &bytesusefulness
					  ) {
      if (!fMainReadCache) return false;

      fMainReadCache->GetInfo(size,
			      bytessubmitted,
			      byteshit,
			      misscount,
			      missrate,
			      readreqcnt,
			      bytesusefulness);
      return true;
    }  
					  

    void                       SetCacheSize(int CacheSize) {
        if (!fMainReadCache && CacheSize)
	  fMainReadCache = new XrdClientReadCache();

        if (fMainReadCache)
	   fMainReadCache->SetSize(CacheSize);
    }

    void                       SetCacheRmPolicy(int RmPolicy) {
        if (fMainReadCache)
	   fMainReadCache->SetBlkRemovalPolicy(RmPolicy);
    }

    void                       UnPinCacheBlk(long long begin_offs, long long end_offs) {
        fMainReadCache->UnPinCacheBlk(begin_offs, end_offs);
	// Also use this to signal the possibility to proceed for a hard checkpoint
	fWriteWaitAck->Broadcast();
    }


    // -------------------


    int                        GetLogConnID() const { return fLogConnID; }

    ERemoteServerType          GetServerType() const { return fServerType; }

    kXR_unt16                  GetStreamID() const { return fPrimaryStreamid; }

    inline XrdClientUrlInfo    *GetLBSUrl() { return fLBSUrl; }
    inline XrdClientUrlInfo    *GetMetaUrl() { return fMetaUrl; }
    inline XrdClientUrlInfo    GetCurrentUrl() { return fUrl; }
    inline XrdClientUrlInfo    GetRedirUrl() { return fREQUrl; }

    XErrorCode                 GetOpenError() const { return fOpenError; }
    virtual XReqErrorType      GoToAnotherServer(XrdClientUrlInfo &newdest);
    virtual XReqErrorType      GoToMetaManager();
    bool                       IsConnected() const { return fConnected; }
    bool                       IsPhyConnConnected();

    struct ServerResponseHeader
                               LastServerResp;

    struct ServerResponseBody_Error
                               LastServerError;

    void                       ClearLastServerError() {
                                   memset(&LastServerError, 0, sizeof(LastServerError));
                                   LastServerError.errnum = kXR_noErrorYet;
                               }

    UnsolRespProcResult        ProcessAsynResp(XrdClientMessage *unsolmsg);

    virtual bool               SendGenCommand(ClientRequest *req, 
					      const void *reqMoreData,       
					      void **answMoreDataAllocated,
					      void *answMoreData, bool HasToAlloc,
					      char *CmdName, int substreamid = 0);

    int                        GetOpenSockFD() const { return fOpenSockFD; }

    void                       SetClientHostDomain(const char *src) { fgClientHostDomain = src; }
    void                       SetConnected(bool conn) { fConnected = conn; }

    void                       SetOpenError(XErrorCode err) { fOpenError = err; }

    // Gets a parallel stream id to use to set the return path for a re
    int                        GetParallelStreamToUse(int reqsperstream);
    int                        GetParallelStreamCount();     // Returns the total number of connected streams

    void                       SetRedirHandler(XrdClientAbs *rh) { fRedirHandler = rh; }

    void                       SetRequestedDestHost(char *newh, kXR_int32 port) {
	fREQUrl = fUrl;
	fREQUrl.Host = newh;
	fREQUrl.Port = port;
	fREQUrl.SetAddrFromHost();
    }

    // Puts this instance in pause state for wsec seconds.
    // A value <= 0 revokes immediately the pause state
    void                       SetREQPauseState(kXR_int32 wsec) {
	// Lock mutex
	fREQWait->Lock();

	if (wsec > 0)
	    fREQWaitTimeLimit = time(0) + wsec;
	else {
	    fREQWaitTimeLimit = 0;
	    fREQWait->Broadcast();
	}

	// UnLock mutex
	fREQWait->UnLock();
    }

    // Puts this instance in connect-pause state for wsec seconds.
    // Any future connection attempt will not happen before wsec
    //  and the first one will be towards the given host
    void                       SetREQDelayedConnectState(kXR_int32 wsec) {
	// Lock mutex
	fREQConnectWait->Lock();

	if (wsec > 0)
	    fREQConnectWaitTimeLimit = time(0) + wsec;
	else {
	    fREQConnectWaitTimeLimit = 0;
	    fREQConnectWait->Broadcast();
	}

	// UnLock mutex
	fREQConnectWait->UnLock();
    }

    void                       SetSID(kXR_char *sid);
    inline void                SetUrl(XrdClientUrlInfo thisUrl) { fUrl = thisUrl; }

    // Sends the request to the server, through logconn with ID LogConnID
    // The request is sent with a streamid 'child' of the current one, then marked as pending
    // Its answer will be caught asynchronously
    XReqErrorType              WriteToServer_Async(ClientRequest *req, 
						   const void* reqMoreData,
						   int substreamid = 0);

    static XrdClientConnectionMgr *GetConnectionMgr()
    { return fgConnectionMgr;} //Instance of the conn manager

    static void DelSessionIDRepo()         {fSessionIDRMutex.Lock();
                                            fSessionIDRepo.Purge();
                                            fSessionIDRMutex.UnLock();
                                           }

    void GetSessionID(SessionIDInfo &sess) {sess = mySessionID;}

    long                       GetServerProtocol() { return fServerProto; }

    short                      GetMaxRedirCnt() const { return fMaxGlobalRedirCnt; }
    void                       SetMaxRedirCnt(short mx) {fMaxGlobalRedirCnt = mx; }
    short                      GetRedirCnt() const { return fGlobalRedirCnt; }

    bool                       DoWriteSoftCheckPoint();
    bool                       DoWriteHardCheckPoint();
    void                       UnPinCacheBlk();


    // To give a max number of seconds for an operation to complete, no matter what happens inside
    // e.g. redirections, sleeps, failed connection attempts etc.
    void                       SetOpTimeLimit(int delta_secs);
    bool                       IsOpTimeLimitElapsed(time_t timenow);


protected:
    void                       SetLogConnID(int cid) { fLogConnID = cid; }
    void                       SetStreamID(kXR_unt16 sid) { fPrimaryStreamid = sid; }



    // The handler which first tried to connect somewhere
    XrdClientAbsUnsolMsgHandler *fUnsolMsgHandler;

    XrdClientUrlInfo           fUrl;                // The current URL
    XrdClientUrlInfo           *fLBSUrl;            // Needed to save the load balancer url
    XrdClientUrlInfo           fREQUrl;             // For explicitly requested redirs

    short                      fGlobalRedirCnt;    // Number of redirections

private:

    static XrdOucString        fgClientHostDomain; // Save the client's domain name
    bool                       fConnected;
    bool                       fGettingAccessToSrv; // To avoid recursion in desperate situations
    time_t                     fGlobalRedirLastUpdateTimestamp; // Timestamp of last redirection

    int                        fLogConnID;        // Logical connection ID used
    kXR_unt16                  fPrimaryStreamid;  // Streamid used for normal communication
    // NB it's a copy of the one contained in
    // the logconn

    short                      fMaxGlobalRedirCnt;
    XrdClientReadCache         *fMainReadCache;

    // The time limit for a transaction
    time_t                     fOpTimeLimit;

    XrdClientAbs               *fRedirHandler;      // Pointer to a class inheriting from
    // XrdClientAbs providing methods
    // to handle a redir at higher level

    XrdOucString               fRedirInternalToken; // Token returned by the server when
    // redirecting. To be used in the next logins

    XrdSysCondVar              *fREQWaitResp;           // For explicitly requested delayed async responses
    ServerResponseBody_Attn_asynresp *
                               fREQWaitRespData;        // For explicitly requested delayed async responses

    time_t                     fREQWaitTimeLimit;   // For explicitly requested pause state
    XrdSysCondVar              *fREQWait;           // For explicitly requested pause state
    time_t                     fREQConnectWaitTimeLimit;   // For explicitly requested delayed reconnect
    XrdSysCondVar              *fREQConnectWait;           // For explicitly requested delayed reconnect

    long                       fServerProto;        // The server protocol
    ERemoteServerType          fServerType;         // Server type as returned by doHandShake() 
    SessionIDInfo              mySessionID;         // Login session ID


    static XrdSysMutex         fSessionIDRMutex;    // Mutex for the Repo
    static XrdOucHash<SessionIDInfo>
    fSessionIDRepo;      // The repository of session IDs, shared.
    // Association between
    // <hostname>:<port>.<user> and a SessionIDInfo struct

    int                        fOpenSockFD;         // Descriptor of the underlying socket
    static XrdClientConnectionMgr *fgConnectionMgr; //Instance of the Connection Manager

    XrdSysCondVar              *fWriteWaitAck;
    XrdClientVector<ClientRequest> fWriteReqsToRetry; // To store the write reqs to retry in case of a disconnection

    bool                       CheckErrorStatus(XrdClientMessage *, short &, char *);
    void                       CheckPort(int &port);
    void                       CheckREQPauseState();
    void                       CheckREQConnectWaitState();
    bool                       CheckResp(struct ServerResponseHeader *resp, const char *method);
    XrdClientMessage           *ClientServerCmd(ClientRequest *req,
						const void *reqMoreData,
						void **answMoreDataAllocated,
						void *answMoreData,
						bool HasToAlloc,
						int substreamid = 0);
    XrdSecProtocol            *DoAuthentication(char *plist, int plsiz);

    ERemoteServerType          DoHandShake(short log);

    bool                       DoLogin();
    bool                       DomainMatcher(XrdOucString dom, XrdOucString domlist);

    XrdOucString               GetDomainToMatch(XrdOucString hostname);

    ESrvErrorHandlerRetval     HandleServerError(XReqErrorType &, XrdClientMessage *,
						 ClientRequest *);
    bool                       MatchStreamid(struct ServerResponseHeader *ServerResponse);

    // Sends a close request, without waiting for an answer
    // useful (?) to be sent just before closing a badly working stream
    bool                       PanicClose();

    XrdOucString               ParseDomainFromHostname(XrdOucString hostname);

    XrdClientMessage           *ReadPartialAnswer(XReqErrorType &, size_t &, 
						  ClientRequest *, bool, void**,
						  EThreeStateReadHandler &);

//  void                       ClearSessionID();

    XReqErrorType              WriteToServer(ClientRequest *req, 
					     const void* reqMoreData,
					     short LogConnID,
					     int substreamid = 0);

    bool                       WaitResp(int secsmax);

    XrdClientUrlInfo           *fMetaUrl;           // Meta manager url
    bool                        fLBSIsMeta;         // Is current redirector a meta manager?

public:
    XrdOucString               fRedirCGI;           // Same as fRedirOpaque but persistent

};
#endif
