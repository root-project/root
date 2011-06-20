//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientConn                                                        // 
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
// Adapted from TXNetFile (root.cern.ch) originally done by             //
//  Alvise Dorigo, Fabrizio Furano                                      //
//          INFN Padova, 2003                                           //
//                                                                      //
// High level handler of connections to xrootd.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientConnCVSID = "$Id$";

#include "XrdClient/XrdClientDebug.hh"

#include "XrdClient/XrdClientConnMgr.hh"
#include "XrdClient/XrdClientConn.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientPhyConnection.hh"
#include "XrdClient/XrdClientProtocol.hh"

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdClient/XrdClientUrlInfo.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientAbs.hh"

#include "XrdClient/XrdClientSid.hh"

#include "XrdSys/XrdSysPriv.hh"

// Dynamic libs
// Bypass Solaris ELF madness
//
#if defined(__solaris__)
#include <sys/isa_defs.h>
#if defined(_ILP32) && (_FILE_OFFSET_BITS != 32)
#undef  _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 32
#undef  _LARGEFILE_SOURCE
#endif
#endif

#ifndef WIN32
#include <dlfcn.h>
#ifndef __macos__
#include <link.h>
#endif
#endif

#include <stdio.h>      // needed by printf
#include <stdlib.h>     // needed by getenv()
#ifndef WIN32
#include <pwd.h>        // needed by getpwuid()
#include <sys/types.h>  // needed by getpid()
#include <unistd.h>     // needed by getpid() and getuid()
#else
#include <process.h>
#include "XrdSys/XrdWin32.hh"
#endif
#include <string.h>     // needed by memcpy() and strcspn()
#include <ctype.h>

#define  SafeDelete(x) { if (x) { delete x; x = 0; } }


// Security handle
typedef XrdSecProtocol *(*XrdSecGetProt_t)(const char *, const struct sockaddr &,
                                           const XrdSecParameters &, XrdOucErrInfo *);

XrdOucHash<XrdClientConn::SessionIDInfo> XrdClientConn::fSessionIDRepo;

// Instance of the Connection Manager
XrdClientConnectionMgr *XrdClientConn::fgConnectionMgr = 0;
XrdOucString XrdClientConn::fgClientHostDomain;

//_____________________________________________________________________________
void ParseRedirHost(XrdOucString &host, XrdOucString &opaque, XrdOucString &token)
{
    // Small utility function... we want to parse a hostname which
    // can contain opaque or token info
  
    int pos;

    token = "";
    opaque = "";

    if ( (pos = host.find('?')) != STR_NPOS ) {
      opaque.assign(host,pos+1);
      host.erasefromend(host.length()-pos);

      if ( (pos = opaque.find('?')) != STR_NPOS ) {
	token.assign(host,pos+1);
	opaque.erasefromend(opaque.length()-pos);
      }

    }

}

//_____________________________________________________________________________
void ParseRedir(XrdClientMessage* xmsg, int &port, XrdOucString &host, XrdOucString &opaque, XrdOucString &token)
{
    // Small utility function... we want to parse the content
    // of a redir response from the server.

    // Remember... an instance of XrdClientMessage automatically 0-terminates the
    // data if present
    struct ServerResponseBody_Redirect* redirdata =
	(struct ServerResponseBody_Redirect*)xmsg->GetData();

    port = 0;

    if (redirdata) {
      XrdOucString h(redirdata->host);
      ParseRedirHost(h, opaque, token);
      host = h;
      port = ntohl(redirdata->port);
    }
}




//_____________________________________________________________________________
XrdClientConn::XrdClientConn(): fOpenError((XErrorCode)0), fUrl(""),
				fLBSUrl(0), 
                                fConnected(false), 
				fGettingAccessToSrv(false),
				fMainReadCache(0),
				fREQWaitRespData(0),
				fREQWaitTimeLimit(0),
				fREQConnectWaitTimeLimit(0) {
    // Constructor
    ClearLastServerError();
    memset(&LastServerResp, 0, sizeof(LastServerResp));
    LastServerResp.status = kXR_noResponsesYet;
 
    fREQUrl.Clear();
    fREQWait = new XrdSysCondVar(0);
    fREQConnectWait = new XrdSysCondVar(0);
    fREQWaitResp = new XrdSysCondVar(0);
    fWriteWaitAck = new XrdSysCondVar(0);

    fRedirHandler = 0;
    fUnsolMsgHandler = 0;

    // Init the redirection counter parameters
    fGlobalRedirLastUpdateTimestamp = time(0);
    fGlobalRedirCnt = 0;
    fMaxGlobalRedirCnt = EnvGetLong(NAME_MAXREDIRECTCOUNT);

    fOpenSockFD = -1;

    // Init connection manager (only once)
    if (!fgConnectionMgr) {
	if (!(fgConnectionMgr = new XrdClientConnectionMgr())) {
	    Error("XrdClientConn::XrdClientConn", "initializing connection manager");
	}

      char buf[255];
      gethostname(buf, sizeof(buf));
      fgClientHostDomain = GetDomainToMatch(buf);

      if (fgClientHostDomain == "")
	  Error("XrdClientConn",
	        "Error resolving this host's domain name." );

      XrdOucString goodDomainsRE = fgClientHostDomain;
      goodDomainsRE += "|*";

      if (EnvGetString(NAME_REDIRDOMAINALLOW_RE) == 0)
	 EnvPutString(NAME_REDIRDOMAINALLOW_RE, goodDomainsRE.c_str());
      if (EnvGetString(NAME_REDIRDOMAINDENY_RE) == 0)
	 EnvPutString(NAME_REDIRDOMAINDENY_RE, "<unknown>");
      if (EnvGetString(NAME_CONNECTDOMAINALLOW_RE) == 0)
	 EnvPutString(NAME_CONNECTDOMAINALLOW_RE, goodDomainsRE.c_str());
      if (EnvGetString(NAME_CONNECTDOMAINDENY_RE) == 0)
	 EnvPutString(NAME_CONNECTDOMAINDENY_RE, "<unknown>");
    }

    // Server type unknown at initialization
    fServerType = kSTNone;
}

//_____________________________________________________________________________
XrdClientConn::~XrdClientConn()
{


  // Disconnect underlying logical connection
  Disconnect(FALSE);

  // Destructor
  if (fMainReadCache && (DebugLevel() >= XrdClientDebug::kUSERDEBUG))
    fMainReadCache->PrintPerfCounters();

  if (fLBSUrl) delete fLBSUrl;

  if (fMainReadCache) delete fMainReadCache;
  fMainReadCache = 0;

  delete fREQWait;
  fREQWait = 0;

  delete fREQConnectWait;
  fREQConnectWait = 0;

  delete fREQWaitResp;
  fREQWaitResp = 0;

  delete fWriteWaitAck;
  fWriteWaitAck = 0;

}

//_____________________________________________________________________________
short XrdClientConn::Connect(XrdClientUrlInfo Host2Conn,
			     XrdClientAbsUnsolMsgHandler *unsolhandler)
{
    // Connect method (called the first time when XrdClient is first created, 
    // and used for each redirection). The global static connection manager 
    // object is firstly created here. If another XrdClient object is created
    // inside the same application this connection manager will be used and
    // no new one will be created.
    // No login/authentication are performed at this stage.

    // We try to connect to the host. What we get is the logical conn id
    short logid;
    logid = -1;
    fPrimaryStreamid = 0;
    fLogConnID = 0;

    CheckREQConnectWaitState();

    Info(XrdClientDebug::kHIDEBUG,
	 "XrdClientConn", "Trying to connect to " <<
	 Host2Conn.HostAddr << ":" << Host2Conn.Port);

    logid = ConnectionManager->Connect(Host2Conn);

    Info(XrdClientDebug::kHIDEBUG,
	 "Connect", "Connect(" << Host2Conn.Host << ", " <<
	 Host2Conn.Port << ") returned " <<
	 logid );

    if (logid < 0) {
	Error("XrdNetFile",
	      "Error creating logical connection to " << 
	      Host2Conn.Host << ":" << Host2Conn.Port );

	fLogConnID = logid;
	fConnected = FALSE;
	return -1;
    }

    fConnected = TRUE;

    fLogConnID = logid;
    fPrimaryStreamid = ConnectionManager->GetConnection(fLogConnID)->Streamid();

    ConnectionManager->GetConnection(fLogConnID)->UnsolicitedMsgHandler = unsolhandler;
    fUnsolMsgHandler = unsolhandler;

    return logid;
}

//_____________________________________________________________________________
void XrdClientConn::Disconnect(bool ForcePhysicalDisc)
{
    // Disconnect... is it so difficult? Yes!
    if( ConnectionManager->SidManager() )
      ConnectionManager->SidManager()->GetAllOutstandingWriteRequests(fPrimaryStreamid, fWriteReqsToRetry);

    if (fMainReadCache && (DebugLevel() >= XrdClientDebug::kDUMPDEBUG) ) fMainReadCache->PrintCache();

    if (fConnected)
       ConnectionManager->Disconnect(fLogConnID, ForcePhysicalDisc);

    fConnected = FALSE;
}

//_____________________________________________________________________________
XrdClientMessage *XrdClientConn::ClientServerCmd(ClientRequest *req, const void *reqMoreData,
						 void **answMoreDataAllocated,
						 void *answMoreData, bool HasToAlloc,
						 int substreamid) 
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

    int len;
   

    size_t TotalBlkSize = 0;

    void *tmpMoreData;
    XReqErrorType errorType = kOK;

    XrdClientMessage *xmsg = 0;

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

	errorType = WriteToServer(req, reqMoreData, fLogConnID, substreamid);
      
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

	    XrdClientConn::EThreeStateReadHandler whatToDo;

	    delete xmsg;

	    xmsg = ReadPartialAnswer(errorType, TotalBlkSize, req, HasToAlloc,
				     &tmpMoreData, whatToDo);

	    // If the cmd went ok and was a read request, we use it to populate
	    // the cache
	    if (xmsg && fMainReadCache && (req->header.requestid == kXR_read) &&
		((xmsg->HeaderStatus() == kXR_oksofar) || 
		 (xmsg->HeaderStatus() == kXR_ok)))
		// To compute the end offset of the block we have to take 1 from the size!
		fMainReadCache->SubmitXMessage(xmsg, req->read.offset + TotalBlkSize - xmsg->fHdr.dlen,
					       req->read.offset + TotalBlkSize - 1);

	    if (whatToDo == kTSRHReturnNullMex) {
		delete xmsg;
		return 0;
	    }

	    if (whatToDo == kTSRHReturnMex)
		return xmsg;
	
	    if (xmsg && (xmsg->HeaderStatus() == kXR_oksofar) && 
		(xmsg->DataLen() == 0))
		return xmsg;
	
	} while (xmsg && (xmsg->HeaderStatus() == kXR_oksofar));

    } while ((fGlobalRedirCnt < fMaxGlobalRedirCnt) &&
             !IsOpTimeLimitElapsed(time(0)) &&
	     xmsg && (xmsg->HeaderStatus() == kXR_redirect)); 

    // We collected all the partial responses into a single memory block.
    // If the block has been allocated here then we must pass its address
    if (HasToAlloc && (answMoreDataAllocated)) {
	*answMoreDataAllocated = tmpMoreData;
    }

    // We might have collected multiple partial response also in a given mem block
    if (xmsg && (xmsg->HeaderStatus() == kXR_ok) && TotalBlkSize)
	xmsg->fHdr.dlen = TotalBlkSize;

    return xmsg;
}

//_____________________________________________________________________________
bool XrdClientConn::SendGenCommand(ClientRequest *req, const void *reqMoreData,
				   void **answMoreDataAllocated, 
				   void *answMoreData, bool HasToAlloc,
				   char *CmdName,
				   int substreamid) {

    // SendGenCommand tries to send a single command for a number of times 

    short retry = 0;
    bool resp = FALSE, abortcmd = FALSE;

    string orig_fname,new_fname;
    if (req->header.requestid == kXR_open && reqMoreData)
      orig_fname.assign((const char *)reqMoreData);

    // if we're going to open a file for the 2nd time we should reset fOpenError, 
    // just in case...
    if (req->header.requestid == kXR_open)
	fOpenError = (XErrorCode)0;





    while (!abortcmd && !resp) {
	abortcmd = FALSE;

	// This client might have been paused
	CheckREQPauseState();

	// Send the cmd, dealing automatically with redirections and
	// redirections on error
	Info(XrdClientDebug::kHIDEBUG,
	     "SendGenCommand","Sending command " << CmdName);

        // Note: some older server versions expose a bug associated to kXR_retstat
        if ( (req->header.requestid == kXR_open) &&
             (GetServerProtocol() < 0x00000270) ) {
           if (req->open.options & kXR_retstat)
	     req->open.options ^= kXR_retstat;

           Info(XrdClientDebug::kHIDEBUG, "SendGenCommand",
            "Old server proto version(" << GetServerProtocol() <<
	    ". kXR_retstat is now disabled. Current open options: " << req->open.options);
        }

 
        kXR_int32 oldlen = 0;
        if (req->header.requestid == kXR_open && reqMoreData) {
          oldlen = req->open.dlen;
          new_fname = orig_fname;
          if (fRedirOpaque.length()) {
            new_fname += "?";
            new_fname += string(fRedirOpaque.c_str());
          }
          reqMoreData = new_fname.c_str();
          req->open.dlen = new_fname.length();
        }

        XrdClientMessage *cmdrespMex = ClientServerCmd(req, reqMoreData,
                                                       answMoreDataAllocated,
                                                       answMoreData, HasToAlloc,
                                                       substreamid);

        if (req->header.requestid == kXR_open && reqMoreData) {
          req->open.dlen = oldlen;
        }


	// Save server response header if requested
	if (cmdrespMex)
	    memcpy(&LastServerResp, &cmdrespMex->fHdr,sizeof(struct ServerResponseHeader));

        // Check for the max time allowed for this request
        if (IsOpTimeLimitElapsed(time(0))) {
           Error("SendGenCommand",
                 "Max time limit elapsed for request  " <<
                 convertRequestIdToChar(req->header.requestid) <<
                 ". Aborting command.");
           abortcmd = TRUE;
           
        } else
           // Check for the redir count limit
           if (fGlobalRedirCnt >= fMaxGlobalRedirCnt) {
              Error("SendGenCommand",
                    "Too many redirections for request  " <<
                    convertRequestIdToChar(req->header.requestid) <<
                    ". Aborting command.");
              
              abortcmd = TRUE;
           }
           else {
              
            // On serious communication error we retry for a number of times,
	    // waiting for the server to come back
	    if (!cmdrespMex || cmdrespMex->IsError()) {

		Info(XrdClientDebug::kHIDEBUG,
		     "SendGenCommand", "Got (and maybe recovered) an error from " <<
		     fUrl.Host << ":" << fUrl.Port);


		// For the kxr_open request we don't rely on the count limit of other
		// reqs. The open request is bounded only by the redir count limit
		if (req->header.requestid != kXR_open) 
		    retry++;

		if (retry > kXR_maxReqRetry) {
		    Error("SendGenCommand",
			  "Too many errors communication errors with server"
			  ". Aborting command.");

		    abortcmd = TRUE;
		} 

		else
		    if (req->header.requestid == kXR_bind) {
			Info(XrdClientDebug::kHIDEBUG,
			     "SendGenCommand", "Parallel stream bind failure. Aborting request." <<
			     fUrl.Host << ":" << fUrl.Port);

			abortcmd = TRUE;
		    }


		    else {

			// Here we are connected, but we could not have a filehandle for
			// various reasons. The server may have denied it to us.
			// So, if we were requesting things that needed a filehandle,
			// and the file seems not open, then abort the request
			if ( (LastServerResp.status != kXR_ok) && 
			     (  (req->header.requestid == kXR_read) ||
				(req->header.requestid == kXR_write) ||
				(req->header.requestid == kXR_sync) ||
				(req->header.requestid == kXR_close) ) ) {
			
			    Info(XrdClientDebug::kHIDEBUG,
				 "SendGenCommand", "Recovery failure detected. Aborting request." <<
				 fUrl.Host << ":" << fUrl.Port);
			
			    abortcmd = TRUE;
			
			}
			else
			    abortcmd = FALSE;

		    }
	    } else {

		// We are here if we got an answer for the command, so
		// the server (original or redirected) is alive
		resp = CheckResp(&cmdrespMex->fHdr, CmdName);
		retry++;
	    

		// If the answer was not (or not totally) positive, we must 
		// investigate on the result
		if (!resp) {
	      
                            
		    // this could be a delayed response. A Strange hybrid. Not a quark.
		    if (cmdrespMex->fHdr.status == kXR_waitresp) {
			// Let's sleep!
			kXR_int32 *maxwait = (kXR_int32 *)cmdrespMex->GetData();
			kXR_int32 mw;

			if (maxwait)
			  mw = ntohl(*maxwait);
			else mw = 30;

			if (!WaitResp(mw)) {
			    // we did not time out, so the response is here
		  
			    memcpy(&LastServerResp, &fREQWaitRespData->resphdr,
				   sizeof(struct ServerResponseHeader));
		  		   
			    // Let's fake a regular answer

			    // Note: kXR_wait can be a fake response used to make the client retry!
			    if (fREQWaitRespData->resphdr.status == kXR_wait) {
				cmdrespMex->fHdr.status = kXR_wait;
				if (fREQWaitRespData->resphdr.dlen) 
				  memcpy(cmdrespMex->GetData(), fREQWaitRespData->respdata, sizeof(kXR_int32));
				else memset(cmdrespMex->GetData(), 0, sizeof(kXR_int32));

				CheckErrorStatus(cmdrespMex, retry, CmdName);
				// This causes a retry
				resp = false;
			    }
			    else {

				if (HasToAlloc) {
				    *answMoreDataAllocated = malloc(LastServerResp.dlen);
				    memcpy(*answMoreDataAllocated,
					   &fREQWaitRespData->respdata,
					   LastServerResp.dlen);
				}
				else {
			    
				    memcpy(answMoreData,
					   &fREQWaitRespData->respdata,
					   LastServerResp.dlen);
			    
				}
				
				// This makes the request exit with the new answer
				resp = true;
			    }

			    free( fREQWaitRespData);
			    fREQWaitRespData = 0;

			    abortcmd = false;

			}


		    }
		    else {

			abortcmd = CheckErrorStatus(cmdrespMex, retry, CmdName);

			// An open request which fails for an application reason like kxr_wait
			// must have its kXR_Refresh bit cleared.
			if (req->header.requestid == kXR_open)
			    req->open.options &= ((kXR_unt16)~kXR_refresh);
		    }
		}

		if (retry > kXR_maxReqRetry) {
		    Error("SendGenCommand",
			  "Too many errors messages from server."
			  " Aborting command.");

		    abortcmd = TRUE;
		}
	    } // else... the case of a correct server response but declaring an error
	}

	delete cmdrespMex;
    } // while

    return (!abortcmd);
}

//_____________________________________________________________________________
bool XrdClientConn::CheckHostDomain(XrdOucString hostToCheck)
{
    // Checks domain matching

    static XrdOucHash<int> knownHosts;
    static XrdOucString alloweddomains = EnvGetString(NAME_REDIRDOMAINALLOW_RE);
    static XrdOucString denieddomains = EnvGetString(NAME_REDIRDOMAINDENY_RE);
    static XrdSysMutex knownHostsMutex;

    XrdSysMutexHelper scopedLock(knownHostsMutex);

    // Check cached info
    int *he = knownHosts.Find(hostToCheck.c_str());
    if (he)
       return (*he == 1) ? TRUE : FALSE;

    // Get the domain for the url to check
    XrdOucString domain = GetDomainToMatch(hostToCheck);

    // If we are unable to get the domain for the url to check --> access denied to it
    if (domain.length() <= 0) {
       Error("CheckHostDomain", "Error resolving domain name for " <<
                                hostToCheck << ". Denying access.");
       return FALSE;
    }
    Info(XrdClientDebug::kHIDEBUG, "CheckHostDomain", "Resolved [" << hostToCheck <<
                                   "]'s domain name into [" << domain << "]" );

    // Given a list of |-separated regexps for the hosts to DENY,
    // match every entry with domain. If any match is found, deny access.
    if (DomainMatcher(domain, denieddomains) ) {
       knownHosts.Add(hostToCheck.c_str(), new int(0));
       Error("CheckHostDomain", "Access denied to the domain of [" << hostToCheck << "].");
       return FALSE;
    }

    // Given a list of |-separated regexps for the hosts to ALLOW,
    // match every entry with domain. If any match is found, grant access.
    if (DomainMatcher(domain, alloweddomains) ) {
       knownHosts.Add(hostToCheck.c_str(), new int(1));
       Info(XrdClientDebug::kHIDEBUG, "CheckHostDomain",
            "Access granted to the domain of [" << hostToCheck << "].");
       return TRUE;
    }

    Error("CheckHostDomain", "Access to domain " << domain <<
          " is not allowed nor denied: deny.");

    return FALSE;

}

//___________________________________________________________________________
bool XrdClientConn::DomainMatcher(XrdOucString dom, XrdOucString domlist)
{
    // Check matching of domain 'dom' in list 'domlist'.
    // Items in list are separated by '|' and can contain the wild
    // cards '*', e.g.
    //
    //   domlist.c_str() = "cern.ch|*.stanford.edu|slac.*.edu"
    //
    // The domain to match is a FQDN host or domain name, e.g.
    //
    //   dom.c_str() = "flora02.slac.stanford.edu"
    //

    Info(XrdClientDebug::kHIDEBUG, "DomainMatcher",
	 "search for '"<<dom<<"' in '"<<domlist<<"'");
    //
    // Parse domlist
    if (domlist.length() > 0) {
	XrdOucString domain;
	int nm = 0, from = 0;
	while ((from = domlist.tokenize(domain, from, '|')) != STR_NPOS) {
	    Info(XrdClientDebug::kDUMPDEBUG, "DomainMatcher",
		 "checking domain: "<<domain);
	    nm = dom.matches(domain.c_str());
	    if (nm > 0) {
		Info(XrdClientDebug::kHIDEBUG, "DomainMatcher",
		     "domain: "<<domain<<" matches '"<<dom
		     <<"' (matching chars: "<<nm<<")");
		return TRUE;
	    }
	}
    }
    Info(XrdClientDebug::kHIDEBUG, "DomainMatcher",
	 "no domain matching '"<<dom<<"' found in '"<<domlist<<"'");

    return FALSE;
}

//_____________________________________________________________________________
bool XrdClientConn::CheckResp(struct ServerResponseHeader *resp, const char *method)
{
    // Checks if the server's response is the ours.
    // If the response's status is "OK" returns TRUE; if the status is "redirect", it 
    // means that the max number of redirections has been achieved, so returns FALSE.

    if (MatchStreamid(resp)) {

	// ok the response belongs to me
	if (resp->status == kXR_redirect) {
	    // too many redirections. Exit!
	    Error(method, "Error in handling a redirection.");
	    return FALSE;
	}

	if ((resp->status != kXR_ok) && (resp->status != kXR_authmore))
	    // Error message is notified in CheckErrorStatus
	    return FALSE;

	return TRUE;

    } else {
	Error(method, "The return message doesn't belong to this client.");
	return FALSE;
    }
}

//_____________________________________________________________________________
bool XrdClientConn::MatchStreamid(struct ServerResponseHeader *ServerResponse)
{
    // Check stream ID matching between the given response and
    // the one contained in the current logical conn

    
    return ( memcmp(ServerResponse->streamid,
		    &fPrimaryStreamid,
		    sizeof(ServerResponse->streamid)) == 0 );
}

//_____________________________________________________________________________
void XrdClientConn::SetSID(kXR_char *sid) {
    // Set our stream id, to match against that one in the server's response.

    memcpy((void *)sid, (const void*)&fPrimaryStreamid, 2);
}


//_____________________________________________________________________________
XReqErrorType XrdClientConn::WriteToServer(ClientRequest *req, 
					   const void* reqMoreData,
					   short LogConnID,
					   int substreamid) {

    // Send message to server
    ClientRequest req_netfmt = *req;
    XrdClientLogConnection *lgc = 0;
    XrdClientPhyConnection *phyc = 0;

    if (DebugLevel() >= XrdClientDebug::kDUMPDEBUG)
	smartPrintClientHeader(req);

    lgc = ConnectionManager->GetConnection(LogConnID);
    if (!lgc) {
	Error("WriteToServer",
	      "Unknown logical conn " << LogConnID);
       
	return kWRITE;
    }

    phyc = lgc->GetPhyConnection();
    if (!phyc) {
	Error("WriteToServer",
	      "Cannot find physical conn for logid " << LogConnID);
       
	return kWRITE;
    }

    clientMarshall(&req_netfmt);

    // Strong mutual exclusion over the physical channel
    {
	XrdClientPhyConnLocker pcl(phyc);

	// Now we write the request to the logical connection through the
	// connection manager

	short len = sizeof(req->header);

	// A request header is always sent through the main stream, except for kxr_bind!
	int writeres;
	if ( req->header.requestid == kXR_bind )
	  writeres = ConnectionManager->WriteRaw(LogConnID, &req_netfmt, len, substreamid);
	else
	  writeres = ConnectionManager->WriteRaw(LogConnID, &req_netfmt, len, 0);

	fLastDataBytesSent = req->header.dlen;
  
	// A complete communication failure has to be handled later, but we
	// don't have to abort what we are doing
	if (writeres < 0) {
	    Error("WriteToServer",
		  "Error sending " << len << " bytes in the header part"
		  " to server [" <<
		  fUrl.Host << ":" << fUrl.Port << "].");

	    return kWRITE;
	}

	// Send to the server the data.
	// If we got an error we can safely skip this... no need to get more
	if (req->header.dlen > 0) {

	    // Now we write the data associated to the request. Through the
	    //  connection manager
	    // the data chunk can be sent through a parallel stream
	    writeres = ConnectionManager->WriteRaw(LogConnID, reqMoreData,
						   req->header.dlen, substreamid);
    
	    // A complete communication failure has to be handled later, but we
	    //  don't have to abort what we are doing
	    if (writeres < 0) {
		Error("WriteToServer", 
		      "Error sending " << req->header.dlen << " bytes in the data part"
		      " to server [" <<
		      fUrl.Host << ":" << fUrl.Port << "].");

		return kWRITE;
	    }
	}

	fLastDataBytesSent = req->header.dlen;
	return kOK;
    }
}

//_____________________________________________________________________________
bool XrdClientConn::CheckErrorStatus(XrdClientMessage *mex, short &Retry, char *CmdName)
{
    // Check error status, returns true if the retrials have to be aborted

    if (mex->HeaderStatus() == kXR_redirect) {
	// Too many redirections
	Error("CheckErrorStatus",
	      "Error while being redirected for request " << CmdName );
	return TRUE;
    }
 
    if (mex->HeaderStatus() == kXR_error) {
	// The server declared an error. 
	// In this case it's better to exit, unhandled error

	struct ServerResponseBody_Error *body_err;

	body_err = (struct ServerResponseBody_Error *)(mex->GetData());

	if (body_err) {
	    // Print out the error information, as received by the server

	    fOpenError = (XErrorCode)ntohl(body_err->errnum);
 
	    Info(XrdClientDebug::kNODEBUG, "CheckErrorStatus", "Server [" << GetCurrentUrl().HostWPort <<
                 "] declared: " <<
		 (const char*)body_err->errmsg << "(error code: " << fOpenError << ")");

	    // Save the last error received
	    memset(&LastServerError, 0, sizeof(LastServerError));
	    memcpy(&LastServerError, body_err, mex->DataLen());
	    LastServerError.errnum = fOpenError;

	}
	return TRUE;
    }
    
    if (mex->HeaderStatus() == kXR_wait) {
	// We have to wait for a specified number of seconds and then
	// retry the same cmd

	struct ServerResponseBody_Wait *body_wait;

	body_wait = (struct ServerResponseBody_Wait *)mex->GetData();
    
	if (body_wait) {

            if (mex->DataLen() > 4) 
		Info(XrdClientDebug::kUSERDEBUG, "CheckErrorStatus", "Server [" << 
		     fUrl.Host << ":" << fUrl.Port <<
		     "] requested " << ntohl(body_wait->seconds) << " seconds"
		     " of wait. Server message is " << body_wait->infomsg)
		else
		    Info(XrdClientDebug::kUSERDEBUG, "CheckErrorStatus", "Server [" << 
			 fUrl.Host << ":" << fUrl.Port <<
			 "] requested " << ntohl(body_wait->seconds) << " seconds"
			 " of wait")

            // Check if we have to sleep
            int cmw = (getenv("XRDCLIENTMAXWAIT")) ? atoi(getenv("XRDCLIENTMAXWAIT")) : -1;
            int bws = (int)ntohl(body_wait->seconds);
            if ((cmw > -1) && cmw < bws) {
               Error("CheckErrorStatus", "XROOTD MaxWait forced - file is offline"
                                       ". Aborting command. " << cmw << " : " << bws);
               Retry= kXR_maxReqRetry;
               return TRUE;
            }


	    // Look for too stupid a delay. In this case, set a reasonable value.
	    int newbws = bws;
	    if (bws <= 0) newbws = 1;
	    if (bws > 1800) newbws = 10;
	    if (bws != newbws)
	      Error("CheckErrorStatus", "Sleep time fixed from " << bws << " to " << newbws);

            // Sleep now, and hope that the sandman does not enter here.
            sleep(newbws);
	}

	// We don't want kxr_wait to count as an error
	Retry--;
	return FALSE;
    }
    
    // We don't understand what the server said. Better investigate on it...
    Error("CheckErrorStatus", 
	  "Answer from server [" << 
	  fUrl.Host << ":" << fUrl.Port <<
	  "]  not recognized after executing " << CmdName);

    return TRUE;
}

//_____________________________________________________________________________
XrdClientMessage *XrdClientConn::ReadPartialAnswer(XReqErrorType &errorType,
						   size_t &TotalBlkSize, 
						   ClientRequest *req,  
						   bool HasToAlloc, void** tmpMoreData,
						   EThreeStateReadHandler &what_to_do)
{
    // Read server answer

    int len;
    XrdClientMessage *Xmsg = 0;
    void *tmp2MoreData;

    // No need to actually read if we are in error...
    if (errorType == kOK) {
    
	len = sizeof(ServerResponseHeader);

	Info(XrdClientDebug::kHIDEBUG, "ReadPartialAnswer",
	     "Reading a XrdClientMessage from the server [" << 
	     fUrl.Host << ":" << fUrl.Port << "]...");
    
	// A complete communication failure has to be handled later, but we
	//  don't have to abort what we are doing
    
	// Beware! Now Xmsg contains ALSO the information about the esit of
	// the communication at low level.
	Xmsg = ConnectionManager->ReadMsg(fLogConnID);

	fLastDataBytesRecv = Xmsg ? Xmsg->DataLen() : 0;

	if ( !Xmsg || (Xmsg->IsError()) ) {
	    Info(XrdClientDebug::kNODEBUG, "ReadPartialAnswer", "Failed to read msg from connmgr"
		  " (server [" << fUrl.Host << ":" << fUrl.Port << "]). Retrying ...");

	    if (HasToAlloc) {
		if (*tmpMoreData)
		    free(*tmpMoreData);
		*tmpMoreData = 0;
	    }
	    errorType = kREAD;
	}
	else
	    // is not necessary because the Connection Manager unmarshalls the mex
	    Xmsg->Unmarshall(); 
    }

    if (Xmsg != 0)
	if (DebugLevel() >= XrdClientDebug::kDUMPDEBUG)
	    smartPrintServerHeader(&Xmsg->fHdr);

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

			Error("ReadPartialAnswer", "Error reallocating " << 
			      TotalBlkSize << " bytes.");

			free(*tmpMoreData);
			*tmpMoreData = 0;
			what_to_do = kTSRHReturnNullMex;

			delete Xmsg;

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
// 		if (DebugLevel() >= XrdClientDebug::kDUMPDEBUG) {

// 		    Info (XrdClientDebug::kDUMPDEBUG, "ReadPartialAnswer","Dumping read data...");
// 		    for(int jj = 0; jj < Xmsg->DataLen(); jj++) {
// 			printf("0x%.2x ", *( ((kXR_char *)Xmsg->GetData()) + jj ) );
// 			if ( !((jj+1) % 10) ) printf("\n");
// 		    }
// 		    printf("\n\n");
// 		}

		TotalBlkSize += Xmsg->DataLen();
	
	    } else {

	    Info(XrdClientDebug::kHIDEBUG, "ReadPartialAnswer", 
		 "Server [" <<
		 fUrl.Host << ":" << fUrl.Port << "] answered [" <<
		 convertRespStatusToChar(Xmsg->fHdr.status) <<
		 "] (" << Xmsg->fHdr.status << ")");
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
	 isRedir(&Xmsg->fHdr) ) 
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
	
		delete Xmsg;
		Xmsg = 0;

		what_to_do = kTSRHReturnMex;
		return Xmsg;
	    }
	}

    what_to_do = kTSRHContinue;
    return Xmsg;
}


//_____________________________________________________________________________
bool XrdClientConn::GetAccessToSrv()
{
    // Gets access to the connected server. The login and authorization steps
    // are performed here (calling method DoLogin() that performs logging-in
    // and calls DoAuthentication() ).
    // If the server redirects us, this is gently handled by the general
    // functions devoted to the handling of the server's responses.
    // Nothing is visible here, and nothing is visible from the other high
    // level functions.

    XrdClientLogConnection *logconn = ConnectionManager->GetConnection(fLogConnID);

    // This is to prevent recursion in this delicate phase
    if (fGettingAccessToSrv) {
      logconn->GetPhyConnection()->StartReader();
      return true;
    }

    fGettingAccessToSrv = true;

    switch ((fServerType = DoHandShake(fLogConnID))) {
    case kSTError:
	Info(XrdClientDebug::kNODEBUG,
	     "GetAccessToSrv",
	     "HandShake failed with server [" <<
	     fUrl.Host << ":" << fUrl.Port << "]");

	Disconnect(TRUE);

	fGettingAccessToSrv = false;
	return FALSE;

    case kSTNone: 
	Info(XrdClientDebug::kNODEBUG,
	     "GetAccessToSrv", "The server on [" <<
	     fUrl.Host << ":" << fUrl.Port << "] is unknown");

	Disconnect(TRUE);

	fGettingAccessToSrv = false;
	return FALSE;

    case kSTRootd: 

	if (EnvGetLong(NAME_KEEPSOCKOPENIFNOTXRD) == 1) {
	    Info(XrdClientDebug::kHIDEBUG,
		 "GetAccessToSrv","Ok: the server on [" <<
		 fUrl.Host << ":" << fUrl.Port <<
		 "] is a rootd. Saving socket for later use.");
	    // Get socket descriptor
	    fOpenSockFD = logconn->GetPhyConnection()->SaveSocket();
	    Disconnect(TRUE);
	    ConnectionManager->GarbageCollect();
	    break;

	} else {

	    Info(XrdClientDebug::kHIDEBUG,
		 "GetAccessToSrv","Ok: the server on [" <<
		 fUrl.Host << ":" << fUrl.Port << "] is a rootd."
		 " Not supported.");

	    Disconnect(TRUE);

	    fGettingAccessToSrv = false;
	    return FALSE;
	}

    case kSTBaseXrootd: 

	Info(XrdClientDebug::kHIDEBUG,
	     "GetAccessToSrv", 
	     "Ok: the server on [" <<
	     fUrl.Host << ":" << fUrl.Port << "] is an xrootd redirector.");
      
	logconn->GetPhyConnection()->SetTTL(EnvGetLong(NAME_LBSERVERCONN_TTL));

	break;

    case kSTDataXrootd: 

	Info( XrdClientDebug::kHIDEBUG,
	      "GetAccessToSrv", 
	      "Ok, the server on [" <<
	      fUrl.Host << ":" << fUrl.Port << "] is an xrootd data server.");

	logconn->GetPhyConnection()->SetTTL(EnvGetLong(NAME_DATASERVERCONN_TTL));

	break;
    }

    bool retval = false;


    XrdClientPhyConnection *phyc = logconn->GetPhyConnection();
    if (!phyc) {
       fGettingAccessToSrv = false;
       return false;
    }

    XrdClientPhyConnLocker pl(phyc);

    // Execute a login if connected to a xrootd server
    if (fServerType != kSTRootd) {

        phyc = logconn->GetPhyConnection();
        if (!phyc || !phyc->IsValid()) {
           Error( "GetAccessToSrv", "Physical connection disappeared.");
           fGettingAccessToSrv = false;
           return false;
        }

	// Start the reader thread in the phyconn, if needed
	phyc->StartReader();

	if (phyc->IsLogged() == kNo)
	    retval = DoLogin();
	else {

	    Info( XrdClientDebug::kHIDEBUG,
		  "GetAccessToSrv", "Reusing physical connection to server [" <<
		  fUrl.Host << ":" << fUrl.Port << "]).");

	    retval = true;
	}
    }
    else
	retval = true;

    fGettingAccessToSrv = false;
    return retval;
}

//_____________________________________________________________________________
ERemoteServerType XrdClientConn::DoHandShake(short int log) {

    struct ServerInitHandShake xbody;
    ERemoteServerType type;

    // Get the physical connection
    XrdClientLogConnection *lcn = ConnectionManager->GetConnection(log);

    if (!lcn) return kSTError;

    XrdClientPhyConnection *phyconn = lcn->GetPhyConnection();

    if (!phyconn || !phyconn->IsValid()) return kSTError;


    {
      XrdClientPhyConnLocker pl(phyconn);

      if (phyconn->fServerType == kSTBaseXrootd) {

	Info(XrdClientDebug::kUSERDEBUG,
	     "DoHandShake",
	     "The physical channel is already bound to a load balancer"
	     " server [" <<
	     fUrl.Host << ":" << fUrl.Port << "]. No handshake is needed.");

	fServerProto = phyconn->fServerProto;

	if (!fLBSUrl || (fLBSUrl->Host == "")) {

	  Info(XrdClientDebug::kHIDEBUG,
	       "DoHandShake", "Setting Load Balancer Server Url = " <<
	       fUrl.GetUrl() );

	  // Save the url of load balancer server for future uses...
	  fLBSUrl = new XrdClientUrlInfo(fUrl.GetUrl());
	  if(!fLBSUrl) {
	    Error("DoHandShake","Object creation "
		  " failed. Probable system resources exhausted.");
	    abort();
	  }
	}
	return kSTBaseXrootd;
      }


      if (phyconn->fServerType == kSTDataXrootd) {

	if (DebugLevel() >= XrdClientDebug::kHIDEBUG)
	  Info(XrdClientDebug::kHIDEBUG,
	       "DoHandShake",
	       "The physical channel is already bound to the data server"
	       " [" << fUrl.Host << ":" << fUrl.Port << "]. No handshake is needed.");

	fServerProto = phyconn->fServerProto;

	return kSTDataXrootd;
      }


      type = phyconn->DoHandShake(xbody);
      if (type == kSTError) return type;


      // Check if the server is the eXtended rootd or not, checking the value 
      // of type
      fServerProto = xbody.protover;

      // This is useful for other streams trying to use the same phyconn
      // they will be able to get the proto version
      phyconn->fServerProto = fServerProto;

      if (type == kSTBaseXrootd) {
	// This is a load balancing server
	if (!fLBSUrl || (fLBSUrl->Host == "")) {

	  Info(XrdClientDebug::kHIDEBUG, "DoHandShake", "Setting Load Balancer Server Url = " <<
	       fUrl.GetUrl() );

	  // Save the url of load balancer server for future uses...
	  fLBSUrl = new XrdClientUrlInfo(fUrl.GetUrl());
	  if (!fLBSUrl) {
	    Error("DoHandShake","Object creation failed.");
	    abort();
	  }
	}
       
      }

      return type;


    }
}

//_____________________________________________________________________________
bool XrdClientConn::DoLogin() 
{
    // This method perform the loggin-in into the server just after the
    // hand-shake. It also calls the DoAuthentication() method

    ClientRequest reqhdr;
    bool resp;
  
    // We fill the header struct containing the request for login
    memset( &reqhdr, 0, sizeof(reqhdr));

    SetSID(reqhdr.header.streamid);
    reqhdr.header.requestid = kXR_login;
    reqhdr.login.capver[0] = XRD_CLIENT_CAPVER;
    reqhdr.login.pid = getpid();

    // Get username from Url
    XrdOucString User = fUrl.User;
    if (User.length() <= 0) {
	// Use local username, if not specified
#ifndef WIN32
	struct passwd *u = getpwuid(getuid());
	if (u >= 0)
	    User = u->pw_name;
#else
	char  name[256];
	DWORD length = sizeof (name);
	GetUserName(name, &length);
	User = name;
#endif
    }
    if (User.length() > 0)
      strncpy( (char *)reqhdr.login.username, User.c_str(), 8 );
    else
	strcpy( (char *)reqhdr.login.username, "????" );

    // If we run with root as effective user we need to temporary change
    // effective ID to User
    XrdOucString effUser = User;
#ifndef WIN32
    if (!getuid()) {
      if (getenv("XrdClientEUSER")) effUser = getenv("XrdClientEUSER");
    }
    XrdSysPrivGuard guard(effUser.c_str());
    if (!guard.Valid() && !getuid()) {
      // Set error, in case of need
      fOpenError = kXR_NotAuthorized;
      LastServerError.errnum = fOpenError;
      XrdOucString emsg("Cannot set effective uid for user: ");
      emsg += effUser;
      strcpy(LastServerError.errmsg, emsg.c_str());
      Error("DoLogin", emsg << ". Exiting.");
      return false;
    }
#endif

    // set the token with the value provided by a previous 
    // redirection (if any)
    reqhdr.header.dlen = fRedirInternalToken.length(); 
  
    // We call SendGenCommand, the function devoted to sending commands. 
    Info(XrdClientDebug::kHIDEBUG,
	 "DoLogin",
	 "Logging into the server [" << fUrl.Host << ":" << fUrl.Port <<
	 "]. pid=" << reqhdr.login.pid << " uid=" << reqhdr.login.username);

    {
       XrdClientLogConnection *l = ConnectionManager->GetConnection(fLogConnID);
       XrdClientPhyConnection *p = 0;
       if (l) p = l->GetPhyConnection();
       if (p) p->SetLogged(kNo);
       else {
          Error("DoLogin",
                "Logical connection disappeared before request?!? Srv: [" << fUrl.Host << ":" << fUrl.Port <<
                "]. Exiting.");
          return false;
       }
    }


    char *plist = 0;
    resp = SendGenCommand(&reqhdr, fRedirInternalToken.c_str(), 
			  (void **)&plist, 0, 
			  TRUE, (char *)"XrdClientConn::DoLogin");

    // plist is the plain response from the server. We need a way to 0-term it.
    XrdSecProtocol *secp = 0;
    SessionIDInfo *prevsessid = 0;
    XrdOucString sessname;
    XrdOucString sessdump;
    if (resp && LastServerResp.dlen && plist) {

	plist = (char *)realloc(plist, LastServerResp.dlen+1);
	// Terminate server reply
	plist[LastServerResp.dlen]=0;

	char *pauth = 0;
	int lenauth = 0; 
	if ((fServerProto >= 0x240) && (LastServerResp.dlen >= 16)) {

           if (XrdClientDebug::kHIDEBUG <= DebugLevel()) {
	      char b[20];
	      for (unsigned int i = 0; i < 16; i++) {
		  snprintf(b, 20, "%.2x", plist[i]);
		  sessdump += b;
	      }
	      Info(XrdClientDebug::kHIDEBUG,
		  "DoLogin","Got session ID: " << sessdump);
            }

	    // Get the previous session id, in order to kill it later

	    char buf[20];
	    snprintf(buf, 20, "%d", fUrl.Port);

	    sessname = fUrl.HostAddr;
	    if (sessname.length() <= 0)
		sessname = fUrl.Host;

	    sessname += ":";
	    sessname += buf;

	    prevsessid = fSessionIDRepo.Find(sessname.c_str());

	    // Check if we need to authenticate 
	    if (LastServerResp.dlen > 16) {
		Info(XrdClientDebug::kHIDEBUG, "DoLogin","server requires authentication");
		pauth = plist+16;
		lenauth = LastServerResp.dlen-15; 
	    }


	} else {
	    // We need to authenticate 
	    Info(XrdClientDebug::kHIDEBUG, "DoLogin","server requires authentication");
	    pauth = plist;
	    lenauth = LastServerResp.dlen+1; 
	}

	// Run authentication, if needed
	if (pauth) {

	    char *cenv = 0;
	    //
	    // Set trace level
	    if (EnvGetLong(NAME_DEBUG) > 0) {
		cenv = new char[18];
		sprintf(cenv, "XrdSecDEBUG=%ld",EnvGetLong(NAME_DEBUG));
		putenv(cenv);
	    }
	    //
	    // Set username
	    cenv = new char[User.length()+12];
	    sprintf(cenv, "XrdSecUSER=%s",User.c_str());
	    putenv(cenv);
	    //
	    // Set remote hostname
	    cenv = new char[fUrl.Host.length()+12];
	    sprintf(cenv, "XrdSecHOST=%s",fUrl.Host.c_str());
	    putenv(cenv);

	    secp = DoAuthentication(pauth, lenauth);
	    resp = (secp != 0) ? 1 : 0;
	}


	if (prevsessid) {
	    //
	    // We have to kill the previous session, if any
	    // By sending a kXR_endsess

           if (XrdClientDebug::kHIDEBUG <= DebugLevel()) {
	      XrdOucString sessdump;
	      char b[20];
	      for (unsigned int i = 0; i < sizeof(prevsessid->id); i++) {
		  snprintf(b, 20, "%.2x", prevsessid->id[i]);
		  sessdump += b;
	      }
	      Info(XrdClientDebug::kHIDEBUG,
		   "DoLogin","Found prev session info for " << sessname <<
		   ": " << sessdump);
            }

	    memset( &reqhdr, 0, sizeof(reqhdr));
	    SetSID(reqhdr.header.streamid);
	    reqhdr.header.requestid = kXR_endsess;

	    memcpy(reqhdr.endsess.sessid, prevsessid->id, sizeof(prevsessid->id));

	    // terminate session
	    Info(XrdClientDebug::kHIDEBUG,
		 "DoLogin","Trying to terminate previous session.");

	    SendGenCommand(&reqhdr, 0, 0, 0, 
			   FALSE, (char *)"XrdClientConn::Endsess");

	    // Now overwrite the previous session info with the new one
	    for (unsigned int i=0; i < 16; i++)
		prevsessid->id[i] = plist[i];



	} else {
	    Info(XrdClientDebug::kHIDEBUG,
		 "DoLogin","No prev session info for " << sessname);

	    // No session info? Let's create one.
	    SessionIDInfo *newsessid = new SessionIDInfo;

	    for (int i=0; i < int(sizeof(newsessid->id)); i++)
		newsessid->id[i] = plist[i];

	    fSessionIDRepo.Rep(sessname.c_str(), newsessid);
	}

    }

    // Flag success if everything went ok
    {
       XrdClientLogConnection *l = ConnectionManager->GetConnection(fLogConnID);
       XrdClientPhyConnection *p = 0;
       if (l) p = l->GetPhyConnection();
       if (!p) {
          Error("DoLogin",
                "Logical connection disappeared after request?!? Srv: [" << fUrl.Host << ":" << fUrl.Port <<
                "]. Exiting.");
          return false;
       }

       if (resp) {
          p->SetLogged(kYes);
          p->SetSecProtocol(secp);
       }
       else Disconnect(true);
    }

    if (plist)
	free(plist);

    return resp;

}

//_____________________________________________________________________________
XrdSecProtocol *XrdClientConn::DoAuthentication(char *plist, int plsiz)
{
   // Negotiate authentication with the remote server. Tries in turn
   // all available protocols proposed by the server (in plist),
   // starting from the first.
   static XrdSecGetProt_t getp = 0;
   XrdSecProtocol *protocol = (XrdSecProtocol *)0;

   if (!plist || plsiz <= 0)
      return protocol;

   Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
         "host " << fUrl.Host << " sent a list of " << plsiz << " bytes");
   //
   // Prepare host/IP information of the remote xrootd. This is required
   // for the authentication.
   struct sockaddr_in netaddr;
   char **hosterrmsg = 0;
   if (XrdNetDNS::getHostAddr((char *)fUrl.HostAddr.c_str(),
                              (struct sockaddr &)netaddr, hosterrmsg) <= 0) {
      Info(XrdClientDebug::kUSERDEBUG, "DoAuthentication",
                                       "getHostAddr said '" << *hosterrmsg << "'");
      return protocol;
   }
   netaddr.sin_port   = fUrl.Port;

   //
   // Variables for negotiation
   XrdSecParameters  *secToken = 0;
   XrdSecCredentials *credentials = 0;

   //
   // Prepare the parms object
   char *bpar = (char *)malloc(plsiz + 1);
   if (bpar) memcpy(bpar, plist, plsiz);
   bpar[plsiz] = 0;
   XrdSecParameters Parms(bpar, plsiz + 1);

   // We need to load the protocol getter the first time we are here
   if (!getp) {
      // Open the security library
      void *lh = 0;
      if (!(lh = dlopen("libXrdSec.so", RTLD_NOW))) {
         Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                       "unable to load libXrdSec.so");
         // Set error, in case of need
         fOpenError = kXR_NotAuthorized;
         LastServerError.errnum = fOpenError;
         strcpy(LastServerError.errmsg, "unable to load libXrdSec.so");
         return protocol;
      }

      // Get the client protocol getter
      if (!(getp = (XrdSecGetProt_t) dlsym(lh, "XrdSecGetProtocol"))) {
         Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                       "unable to load XrdSecGetProtocol()");
         // Set error, in case of need
         fOpenError = kXR_NotAuthorized;
         LastServerError.errnum = fOpenError;
         strcpy(LastServerError.errmsg, "unable to load XrdSecGetProtocol()");
         return protocol;
      }
   }
   //
   // Get a instance of XrdSecProtocol; the order of preference is the one
   // specified by the server; the env XRDSECPROTOCOL can be used to force
   // the choice.
   while ((protocol = (*getp)((char *)fUrl.Host.c_str(),
                      (const struct sockaddr &)netaddr, Parms, 0))) {
      //
      // Protocol name
      XrdOucString protname = protocol->Entity.prot;
      //
      // Once we have the protocol, get the credentials
      XrdOucErrInfo ei;
      credentials = protocol->getCredentials(0, &ei);
      if (!credentials) {
         Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                        "cannot obtain credentials (protocol: "<<
                                        protname<<")");
         // Set error, in case of need
         fOpenError = kXR_NotAuthorized;
         LastServerError.errnum = fOpenError;
         strcpy(LastServerError.errmsg, "cannot obtain credentials for protocol: ");
         strcat(LastServerError.errmsg, ei.getErrText());
         protocol->Delete();
         protocol = 0;
         continue;
      } else {
         Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                        "credentials size: "<< credentials->size);
      }
      //
      // We fill the header struct containing the request for login
      ClientRequest reqhdr;
      memset(reqhdr.auth.reserved, 0, 12);
      memset(reqhdr.auth.credtype, 0, 4 );
      memcpy(reqhdr.auth.credtype, protname.c_str(), protname.length() > 4 ? 4 : protname.length() );

      LastServerResp.status = kXR_authmore;
      char *srvans = 0;
      while (LastServerResp.status == kXR_authmore) {
         bool resp = false;

         //
         // Length of the credentials buffer
         reqhdr.header.dlen = credentials->size;
         SetSID(reqhdr.header.streamid);
         reqhdr.header.requestid = kXR_auth;
         resp = SendGenCommand(&reqhdr, credentials->buffer, (void **)&srvans, 0, TRUE,
                                (char *)"XrdClientConn::DoAuthentication");
         SafeDelete(credentials);
         Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                        "server reply: status: "<<
                                         LastServerResp.status <<
                                        " dlen: "<< LastServerResp.dlen);
         if (resp && (LastServerResp.status == kXR_authmore)) {
            //
            // We are required to send additional information
            // First assign the security token that we have received
            // at the login request
            secToken = new XrdSecParameters(srvans,LastServerResp.dlen);
            //
            // then get next part of the credentials
            credentials = protocol->getCredentials(secToken, &ei);
            SafeDelete(secToken); // nb: srvans is released here
            srvans = 0;
            if (!credentials) {
               Info(XrdClientDebug::kUSERDEBUG, "DoAuthentication",
                                                "cannot obtain credentials");
               // Set error, in case of need
               fOpenError = kXR_NotAuthorized;
               LastServerError.errnum = fOpenError;
               strcpy(LastServerError.errmsg, "cannot obtain credentials: ");
               strcat(LastServerError.errmsg, ei.getErrText());
               protocol->Delete();
               protocol = 0;
               break;
            } else {
               Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                             "credentials size " << credentials->size);
            }
         } else {
            // Something happened, it could be an error or a good thing as well

            if (LastServerResp.status == kXR_error) {
               // Unexpected reply: stop handshake and print error msg, if any

               if (LastServerError.errmsg)
                  Error("DoAuthentication", LastServerError.errmsg);

               protocol->Delete();
               protocol = 0;
               // This is a fatal auth error
               break;
            }

            if (!resp) {
               // Communication error

               protocol->Delete();
               protocol = 0;
               // This is a fatal auth error
               break;
            }

         }
      }
   
      // If we are done
      if (protocol) break;
   }

   if (!protocol) {
      Info(XrdClientDebug::kHIDEBUG, "DoAuthentication",
                                    "unable to get protocol object.");
      // Set error, in case of need
      fOpenError = kXR_NotAuthorized;
      LastServerError.errnum = fOpenError;
      strcpy(LastServerError.errmsg, "unable to get protocol object.");
   }

   // Return the result of the negotiation
   //
   return protocol;
}

//_____________________________________________________________________________
XrdClientConn::ESrvErrorHandlerRetval
XrdClientConn::HandleServerError(XReqErrorType &errorType, XrdClientMessage *xmsg,
				 ClientRequest *req)
{
    // Handle errors from server

    int newport; 	
    XrdOucString newhost; 	
  
    bool noRedirError = (fMaxGlobalRedirCnt == 1 && xmsg && isRedir(&xmsg->fHdr));

    // Close the log connection at this point the fLogConnID is no longer valid.
    // On read/write error the physical channel may be not OK, so it's a good
    // idea to shutdown it.
    // If there are other logical conns pointing to it, they will get an error,
    // which will be handled
    if (!noRedirError) {
       if ((errorType == kREAD) || (errorType == kWRITE)) {
	  Disconnect(TRUE);

	  if (fMainReadCache)
	     fMainReadCache->RemovePlaceholders();
       } else
	  Disconnect(FALSE);
    }
  
    // We cycle repeatedly trying to ask the dlb for a working redir destination
    do {
    
	// Anyway, let's update the counter, we have just been redirected
	fGlobalRedirCnt++;

	Info(XrdClientDebug::kHIDEBUG,
	     "HandleServerError",
	     "Redir count=" << fGlobalRedirCnt);

	if ( fGlobalRedirCnt >= fMaxGlobalRedirCnt ) {
           if (noRedirError) {
              // The caller just wants the redir info:
              // extract it (new host:port) from the response and return
              newhost = "";
              newport = 0;
              // An explicit redir overwrites token and opaque info
              ParseRedir(xmsg, newport, newhost, fRedirOpaque, fRedirInternalToken);

              // Save it in fREQUrl
              // fREQUrl = fUrl;
              // fREQUrl.Host = newhost;
              // fREQUrl.Port = newport;

              // Reset counter
              fGlobalRedirCnt = 0;
              return kSEHRReturnMsgToCaller;
           } else {
              return kSEHRContinue;
           }
        }

        // If the time limit has expired... exit
        if (IsOpTimeLimitElapsed(time(0))) return kSEHRContinue;

	newhost = "";
	newport = 0;
    
	if ((errorType == kREAD) || 
	    (errorType == kWRITE) || 
	    (errorType == kREDIRCONNECT)) {
	 
	    bool cangoaway = ( fRedirHandler &&
			       fRedirHandler->CanRedirOnError() );

	    // We got some errors in the communication phase
	    // the physical connection has been closed;
	    // then we must go back to the load balancer
	    // if there is any

	    // The exception here is that if the file was open in
	    // write mode, we must rebounce and not go away to other hosts
	    // To state this, we just asked to our redir handler

	    if ( (fREQUrl.Host.length() > 0) ) {
		// If this client was explicitly told to redirect somewhere...
		//ClearSessyionID();

	        // Note, this could contain opaque information
		newhost = fREQUrl.Host;
		newport = fREQUrl.Port;

		ParseRedirHost(newhost, fRedirOpaque, fRedirInternalToken);

		// An unsuccessful connection to the dest host will make the
		//  client go to the LB
		fREQUrl.Clear();
	    }
	    else
		if ( cangoaway && fLBSUrl && (fLBSUrl->GetUrl().length() > 0) ) {
		    // "Normal" error... we go to the LB if any
		    // Clear the current session info. Rather simplicistic.
		    //ClearSessionID();
	       
		    newhost = fLBSUrl->Host;
		    newport = fLBSUrl->Port;
		}
		else {
	       
		    Error("HandleServerError",
			  "Communication error"
			  " with server [" << fUrl.Host << ":" << fUrl.Port <<
			  "]. Rebouncing here.");
	       
		    if (fUrl.Host.length()) newhost = fUrl.Host;
		    else
			newhost = fUrl.HostAddr;
	       
		    newport = fUrl.Port;
		}
	 
	} else if (isRedir(&xmsg->fHdr)) {
      
	    // Extract the info (new host:port) from the response
	    newhost = "";
	    newport = 0;

	    // An explicit redir overwrites token and opaque info
	    ParseRedir(xmsg, newport, newhost, fRedirOpaque, fRedirInternalToken);

	    // Clear the current session info. Rather simplicistic.
	    //ClearSessionID();
	}
    
	// Now we should have the parameters needed for the redir

	CheckPort(newport);

	if ((newhost.length() > 0) && newport) {
	    XrdClientUrlInfo NewUrl(fUrl.GetUrl());

	    if (DebugLevel() >= XrdClientDebug::kUSERDEBUG)
		Info(XrdClientDebug::kUSERDEBUG,
		     "HandleServerError",
		     "Received redirection to [" << newhost << ":" << newport <<
		     "]. Token=[" << fRedirInternalToken << "]" <<
		     "]. Opaque=[" << fRedirOpaque << "].");

	    errorType = kOK;

	    NewUrl.Host = NewUrl.HostAddr = newhost;
	    NewUrl.Port = newport;
	    NewUrl.SetAddrFromHost();


	    if ( !CheckHostDomain(newhost) ) {
		Error("HandleServerError",
		      "Redirection to a server out-of-domain disallowed. Abort.");
		abort();
	    }

	    errorType = GoToAnotherServer(NewUrl);
	}
	else {
	    // Host or port are not valid or empty
	    Error("HandleServerError", 
		  "Received redirection to [" << newhost << ":" << newport <<
		  "]. Token=[" << fRedirInternalToken << "]" <<
		  "]. Opaque=[" << fRedirOpaque << "]. No server to go...");

	    errorType = kREDIRCONNECT;
	}
    
	// We don't want to flood servers...
	if (errorType == kREDIRCONNECT) {
           if (LastServerError.errnum == kXR_NotAuthorized)
              return kSEHRReturnMsgToCaller;

           sleep(EnvGetLong(NAME_RECONNECTWAIT));
        }

	// We keep trying the connection to the same host (we have only one)
	//  until we are connected, or the max count for
	//  redirections is reached

        // The attempts must be stopped if we are not authorized
    } while ((errorType == kREDIRCONNECT) && (LastServerError.errnum != kXR_NotAuthorized));


    // We are here if correctly connected and handshaked and logged
    if (!IsConnected()) {
	Error("HandleServerError", 
	      "Not connected. Internal error. Abort.");
	abort();
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
    bool wasopen, newopenok;

    if (fRedirHandler) {
	newopenok = fRedirHandler->OpenFileWhenRedirected(localfhandle, wasopen);
	if (newopenok && wasopen) {
	    // We are here if the file has been opened succesfully
	    // or if it was not open
	    // Tricky thing: now we have a new filehandle, perhaps in
	    // a different server. Then we must correct the filehandle in
	    // the msg we were sending and that we must repeat...
	    PutFilehandleInRequest(req, localfhandle);
         
	    // Everything should be ok here.
	    // If we have been redirected,then we are connected, logged and reopened
	    // the file. If we had a r/w error (xmsg==0 or xmsg->IsError) we are
	    // OK too. Since we may come from a comm error, then xmsg can be null.
	    if (xmsg && !xmsg->IsError())
		return kSEHRContinue; // the case of explicit redir
	    else
		return kSEHRReturnNoMsgToCaller; // the case of recovered error
	}

	if (!newopenok) return kSEHRContinue; // the case of explicit redir
         
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
XReqErrorType XrdClientConn::GoToAnotherServer(XrdClientUrlInfo &newdest)
{
    // Re-directs to another server
   
    fGettingAccessToSrv = false; 

    if (!newdest.Port) newdest.Port = 1094;
    if (newdest.HostAddr == "") newdest.HostAddr = newdest.Host;

    if ( (fLogConnID = Connect( newdest, fUnsolMsgHandler)) == -1) {
	  
	// Note: if Connect is unable to work then we are in trouble.
	// It seems that we have been redirected to a non working server
	Error("GoToAnotherServer", "Error connecting to [" <<  
	      newdest.Host << ":" <<  newdest.Port);
      
	// If no conn is possible then we return to the load balancer
	return kREDIRCONNECT;
    }
   
    //
    // Set fUrl to the new data/lb server if the 
    // connection has been succesfull
    //
    fUrl = newdest;

    if (IsConnected() && !GetAccessToSrv()) {
	Error("GoToAnotherServer", "Error handshaking to [" << 
	      newdest.Host.c_str() << ":" <<  newdest.Port << "]");
	return kREDIRCONNECT;
    }

    fPrimaryStreamid = ConnectionManager->GetConnection(fLogConnID)->Streamid();

    return kOK;
}

//_____________________________________________________________________________
XReqErrorType XrdClientConn::GoBackToRedirector() {
  // This is a primitive used to force a client to consider again
  // the root node as the default connection, even after requests that involve
  // redirections. Used typically for stat and similar functions
  Disconnect(false);
  if (fGlobalRedirCnt) fGlobalRedirCnt--;
  return (fLBSUrl ? GoToAnotherServer(*fLBSUrl) : kOK);
}

//_____________________________________________________________________________
XrdOucString XrdClientConn::GetDomainToMatch(XrdOucString hostname) {
    // Return net-domain of host hostname in 's'.
    // If the host is unknown in the DNS world but it's a
    //  valid inet address, then that address is returned, in order
    //  to be matched later for access granting

    char *fullname, *err;

    // The name may be already a FQDN: try extracting the domain
    XrdOucString res = ParseDomainFromHostname(hostname);
    if (res.length() > 0)
       return res;

    // Let's look up the hostname
    // It may also be a w.x.y.z type address.
    err = 
	fullname = XrdNetDNS::getHostName((char *)hostname.c_str(), &err);
   
    if ( strcmp(fullname, (char *)"0.0.0.0") ) {
	// The looked up name seems valid
	// The hostname domain can still be unknown
     
	Info(XrdClientDebug::kHIDEBUG,
	     "GetDomainToMatch", "GetHostName(" << hostname <<
	     ") returned name=" << fullname);

	res = ParseDomainFromHostname(fullname);

	if (res == "") {
	    Info(XrdClientDebug::kHIDEBUG,
		 "GetDomainToMatch", "No domain contained in " << fullname);

	    res = ParseDomainFromHostname(hostname);
	}
	if (res == "") {
	    Info(XrdClientDebug::kHIDEBUG,
		 "GetDomainToMatch", "No domain contained in " << hostname);

	    res = hostname;
	}

    } else {

	Info(XrdClientDebug::kHIDEBUG,
	     "GetDomainToMatch", "GetHostName(" << hostname << ") returned a non valid address. errtxt=" << err);

	res = ParseDomainFromHostname(hostname);
    }

    Info(XrdClientDebug::kHIDEBUG,
	 "GetDomainToMatch", "GetDomain(" << hostname << ") --> " << res);
   

    if (fullname) free(fullname);

    return res;
}

//_____________________________________________________________________________
XrdOucString XrdClientConn::ParseDomainFromHostname(XrdOucString hostname)
{
   // Extract the domain

    XrdOucString res;
    int idot = hostname.find('.');
    if (idot != STR_NPOS)
       res.assign(hostname, idot+1);
    // Done
    return res;
}

//_____________________________________________________________________________
void XrdClientConn::CheckPort(int &port) {

    if(port <= 0) {

	Info(XrdClientDebug::kHIDEBUG,
	     "checkPort", 
	     "TCP port not specified. Trying to get it from /etc/services...");

	struct servent *S = getservbyname("rootd", "tcp");
	if(!S) {

	    Info(XrdClientDebug::kHIDEBUG,
		 "checkPort", "Service rootd not specified in /etc/services. "
		 "Using default IANA tcp port 1094");
	    port = 1094;
	} else {
	    Info(XrdClientDebug::kNODEBUG,
		 "checkPort", "Found tcp port " << ntohs(S->s_port) <<
		 " in /etc/service");

	    port = (int)ntohs(S->s_port);
	}

    }
}


//___________________________________________________________________________
long XrdClientConn::GetDataFromCache(const void *buffer, long long begin_offs,
				     long long end_offs, bool PerfCalc,
				     XrdClientIntvList &missingblks, long &outstandingblks) {

    // Copies the requested data from the cache. Returns the number of bytes got
    // Perfcalc = kFALSE forces the call not to impact the perf counters

    if (!fMainReadCache)
	return FALSE;

    return ( fMainReadCache->GetDataIfPresent(buffer,
					      begin_offs,
					      end_offs,
					      PerfCalc,
					      missingblks, outstandingblks) );
}

//___________________________________________________________________________
bool XrdClientConn::SubmitDataToCache(XrdClientMessage *xmsg, long long begin_offs,
				      long long end_offs) {
    // Inserts the data part of this message into the cache
    if (xmsg && fMainReadCache &&
	((xmsg->HeaderStatus() == kXR_oksofar) || 
	 (xmsg->HeaderStatus() == kXR_ok)))

	fMainReadCache->SubmitXMessage(xmsg, begin_offs, end_offs);

    return true;
}

//___________________________________________________________________________
bool XrdClientConn::SubmitRawDataToCache(const void *buffer,
					 long long begin_offs,
					 long long end_offs) {


    if (fMainReadCache) {
      if (!fMainReadCache->SubmitRawData(buffer, begin_offs, end_offs))
         free(const_cast<void *>(buffer));
    }
      
    return true;

}


//___________________________________________________________________________
XReqErrorType XrdClientConn::WriteToServer_Async(ClientRequest *req,
						 const void* reqMoreData,
						 int substreamid) {


    // We allocate a new child streamid, linked to this req
    // Note that the content of the req will be copied. This allows us
    //  to send N times the same req without destroying it
    //  if an answer comes before we finish
    // req is automatically updated with the new streamid
    if (!ConnectionManager->SidManager()->GetNewSid(fPrimaryStreamid, req))
	return kNOMORESTREAMS;

    // If this is a write request, its buffer has to be inserted into the cache
    // This will be used for reference if the request has to be retried later
    // or to give coherency to the read/write semantic
    // From this point on, we consider the request as outstanding
    // Note that his kind of blocks has to be pinned inside the cache until the write is successful
    if (fMainReadCache && (req->header.requestid == kXR_write)) {
      // We have to dup the mem blk
      // It will be destroyed when purged by the cache, only after it has been
      // acknowledged and pinned
      void *locbuf = malloc(req->header.dlen);
      if (!locbuf) { 
	Error("WriteToServer_Async", "Error allocating " << 
	      req->header.dlen << " bytes.");
	return kGENERICERR;
      }

      memcpy(locbuf, reqMoreData, req->header.dlen);

      if (!fMainReadCache->SubmitRawData(locbuf, req->write.offset, req->write.offset+req->header.dlen-1, true))
	free(locbuf);
    }
    // Send the req to the server
    return WriteToServer(req, reqMoreData, fLogConnID, substreamid);

}


//_____________________________________________________________________________
bool XrdClientConn::PanicClose() {
    ClientRequest closeFileRequest;
  
    memset(&closeFileRequest, 0, sizeof(closeFileRequest) );

    SetSID(closeFileRequest.header.streamid);

    closeFileRequest.close.requestid = kXR_close;

    //memcpy(closeFileRequest.close.fhandle, fHandle, sizeof(fHandle) );

    closeFileRequest.close.dlen = 0;
  
    WriteToServer(&closeFileRequest, 0, fLogConnID);

    return TRUE;
}



void XrdClientConn::CheckREQPauseState() {
    // This client might have been paused. In this case the calling thread
    // is put to sleep into a condvar until the desired time arrives.
    // The caller can be awakened by signalling the condvar. But, if the
    // requested wake up time did not elapse, the caller has to sleep again.
   
    time_t timenow;
   
    // Lock mutex
    fREQWait->Lock();
   
    // Check condition
    while (1) {
	timenow = time(0);
      
	if ((timenow < fREQWaitTimeLimit) && !IsOpTimeLimitElapsed(timenow)) {
           // If still to wait... wait in relatively small steps
           time_t tt = xrdmin(fREQWaitTimeLimit - timenow, 10);
           
           fREQWait->Wait(tt);
        }
	else break;
    }
   
    // Unlock mutex
    fREQWait->UnLock();
}


void XrdClientConn::CheckREQConnectWaitState() {
    // This client might have been paused. In this case the calling thread
    // is put to sleep into a condvar until the desired time arrives.
    // The caller can be awakened by signalling the condvar. But, if the
    // requested wake up time did not elapse, the caller has to sleep again.
   
    time_t timenow;
   
    // Lock mutex
    fREQConnectWait->Lock();
   
    // Check condition
    while (1) {
	timenow = time(0);
      
	if ((timenow < fREQConnectWaitTimeLimit) && !IsOpTimeLimitElapsed(timenow)) {
           // If still to wait... wait in relatively small steps
           time_t tt = xrdmin(fREQWaitTimeLimit - timenow, 10);
           // If still to wait... wait
           fREQConnectWait->Wait(tt);
        }
	else break;
    }
   
    // Unlock mutex
    fREQConnectWait->UnLock();
}


bool XrdClientConn::WaitResp(int secsmax) {
    // This client might have been paused to wait for a delayed response.
    // In this case the calling thread
    // is put to sleep into a condvar until the timeout or the response arrives.

    // Returns true on timeout, false if a signal was caught

   int rc = false;

   Info(XrdClientDebug::kHIDEBUG,
        "WaitResp", "Waiting response for " << secsmax << " secs." );

   // Lock condvar
   fREQWaitResp->Lock();

   time_t timelimit = time(0)+secsmax;

   while (!fREQWaitRespData) {
      rc = true;
      time_t timenow = time(0);
          
      if ((timenow < timelimit) && !IsOpTimeLimitElapsed(timenow)) {
         // If still to wait... wait in relatively small steps
         time_t tt = xrdmin(timelimit - timenow, 10);
         fREQWaitResp->Wait(tt);

         // Let's see if there's something
         // If not.. continue waiting
         if (fREQWaitRespData) {
            rc = false;
            break;
         }

      }
      else break;
      

   }
    
   // Unlock condvar
   fREQWaitResp->UnLock();
       
   if (rc) {
      Info(XrdClientDebug::kHIDEBUG,
           "WaitResp", "Timeout elapsed.");
   }
   else {
      Info(XrdClientDebug::kHIDEBUG,
           "WaitResp", "Got an unsolicited response. Data=" << fREQWaitRespData);
   }
    
   return rc;
}



UnsolRespProcResult XrdClientConn::ProcessAsynResp(XrdClientMessage *unsolmsg) {
  // A client on the current physical conn might be in a "wait for response" state
  // Here we process a potential response


  // If this is a comm error, let's awake the sleeping thread and continue
  if (unsolmsg->GetStatusCode() != XrdClientMessage::kXrdMSC_ok) {
    fREQWaitResp->Lock();

    // We also have to fake a regular answer. kxr_wait is ok!
    fREQWaitRespData = (ServerResponseBody_Attn_asynresp *)malloc( sizeof(struct ServerResponseBody_Attn_asynresp) );
    memset( fREQWaitRespData, 0, sizeof(struct ServerResponseBody_Attn_asynresp) );

    fREQWaitRespData->resphdr.status = kXR_wait;
    fREQWaitRespData->resphdr.dlen = sizeof(kXR_int32);

    kXR_int32 i = htonl(1);
    memcpy(&fREQWaitRespData->respdata, &i, sizeof(i));

    fREQWaitResp->Signal();
    fREQWaitResp->UnLock();
    return kUNSOL_CONTINUE;
  }


  ServerResponseBody_Attn_asynresp *ar;
  ar = (ServerResponseBody_Attn_asynresp *)unsolmsg->GetData();

    

  // If the msg streamid matched ours then continue
  if ( !MatchStreamid(&ar->resphdr) ) return kUNSOL_CONTINUE;

  Info(XrdClientDebug::kHIDEBUG,
       "ProcessAsynResp", "Streamid matched." );

  fREQWaitResp->Lock(); 

  // Strip the data from the message and save it. It's the response we are waiting for.
  // Note that it will contain also the data!
  fREQWaitRespData = ar;
   

  clientUnmarshall(&fREQWaitRespData->resphdr);

  if (DebugLevel() >= XrdClientDebug::kDUMPDEBUG)
    smartPrintServerHeader(&fREQWaitRespData->resphdr);

  // After all, this is the last resp we received
  memcpy(&LastServerResp, &fREQWaitRespData->resphdr, sizeof(struct ServerResponseHeader));


  switch (fREQWaitRespData->resphdr.status) {
  case kXR_error: {
    // The server declared an error. 
    // We want to save its content
      
    struct ServerResponseBody_Error *body_err;
      
    body_err = (struct ServerResponseBody_Error *)(&fREQWaitRespData->respdata);
      
    if (body_err) {
      // Print out the error information, as received by the server
	
      kXR_int32 fErr = (XErrorCode)ntohl(body_err->errnum);
	
      Info(XrdClientDebug::kNODEBUG, "ProcessAsynResp", "Server declared: " <<
	   (const char*)body_err->errmsg << "(error code: " << fErr << ")");
	
      // Save the last error received
      memset(&LastServerError, 0, sizeof(LastServerError));
      memcpy(&LastServerError, body_err, xrdmin(fREQWaitRespData->resphdr.dlen, (kXR_int32)(sizeof(LastServerError)-1) ));
      LastServerError.errnum = fErr;
	
    }

    break;
  }

  case kXR_redirect: {

    // Hybrid case. A sync redirect request which comes out the async way.
    // We handle it by simulating an async one

    // Get the encapsulated data
    struct ServerResponseBody_Redirect *rd;
    rd = (struct ServerResponseBody_Redirect *)fREQWaitRespData->respdata;

    // Explicit redirection request
    if (rd && (strlen(rd->host) > 0)) {
      Info(XrdClientDebug::kUSERDEBUG,
	   "ProcessAsynResp", "Requested sync redir (via async response) to " << rd->host <<
	   ":" << ntohl(rd->port));
	
      SetRequestedDestHost(rd->host, ntohl(rd->port));

      // And then we disconnect only this logical conn
      // The subsequent retry will bounce to the requested host
      Disconnect(FALSE);
    }


    // We also have to fake a regular answer. kxr_wait is ok to make the thing retry!
    fREQWaitRespData = (ServerResponseBody_Attn_asynresp *)malloc( sizeof(struct ServerResponseBody_Attn_asynresp) );
    memset( fREQWaitRespData, 0, sizeof(struct ServerResponseBody_Attn_asynresp) );
      
    fREQWaitRespData->resphdr.status = kXR_wait;
    fREQWaitRespData->resphdr.dlen = sizeof(kXR_int32);
      
    kXR_int32 i = htonl(1);
    memcpy(&fREQWaitRespData->respdata, &i, sizeof(i));

    free(unsolmsg->DonateData());
    break;
  }
  }

  unsolmsg->DonateData(); // The data blk is released from the orig message

  // Signal the waiting condvar. Waiting is no more needed
  // Note: the message's data will be freed by the waiting process!
  fREQWaitResp->Signal();

  fREQWaitResp->UnLock();

  // The message is to be destroyed, its data has been saved
  return kUNSOL_DISPOSE;
}






//_____________________________________________________________________________
int XrdClientConn::GetParallelStreamToUse(int reqsperstream) {
    // Gets a parallel stream id to use to set the return path for a req
    XrdClientLogConnection *lgc = 0;
    XrdClientPhyConnection *phyc = 0;

    lgc = ConnectionManager->GetConnection(fLogConnID);
    if (!lgc) {
	Error("GetParallelStreamToUse",
	      "Unknown logical conn " << fLogConnID);
       
	return kWRITE;
    }

    phyc = lgc->GetPhyConnection();
    if (!phyc) {
	Error("GetParallelStreamToUse",
	      "Cannot find physical conn for logid " << fLogConnID);
       
	return kWRITE;
    }

    return phyc->GetSockIdHint(reqsperstream);
}

//_____________________________________________________________________________
bool XrdClientConn::IsPhyConnConnected() {
  // Tells if this instance seems correctly connected to a server

  XrdClientLogConnection *lgc = 0;
  XrdClientPhyConnection *phyc = 0;

  lgc = ConnectionManager->GetConnection(fLogConnID);
  if (!lgc) return false;

  phyc = lgc->GetPhyConnection();
  if (!phyc) return false;

  return phyc->IsValid();
}
//_____________________________________________________________________________
int XrdClientConn:: GetParallelStreamCount() {

    XrdClientLogConnection *lgc = 0;
    XrdClientPhyConnection *phyc = 0;

    lgc = ConnectionManager->GetConnection(fLogConnID);
    if (!lgc) {
        Error("GetParallelStreamCount",
              "Unknown logical conn " << fLogConnID);
       
        return 0;
    }

    phyc = lgc->GetPhyConnection();
    if (!phyc) {
        Error("GetParallelStreamCount",
              "Cannot find physical conn for logid " << fLogConnID);
       
        return 0;
    }

    return phyc->GetSockIdCount();

}


//_____________________________________________________________________________
XrdClientPhyConnection *XrdClientConn::GetPhyConn(int LogConnID) {
  // Protected way to get the underlying physical connection

  XrdClientLogConnection *log;

  log = ConnectionManager->GetConnection(LogConnID);
  if (log) return log->GetPhyConnection();
  return 0;

}


bool XrdClientConn::DoWriteSoftCheckPoint() {
  // Cycle trough the outstanding write requests,
  // If some of them are expired, cancel them
  // and retry in the sync way, one by one
  // If some of them got a negative response of some kind... the same

  // Exit at the first sync error

  // Get the failed write reqs, and put them in a safe place.
  // This call has to be done also just before disconnecting a logical conn,
  // in order to collect all the outstanding write requests before they are forgotten
  ConnectionManager->SidManager()->GetFailedOutstandingWriteRequests(fPrimaryStreamid, fWriteReqsToRetry);

  for (int it = 0; it < fWriteReqsToRetry.GetSize(); it++) {

    ClientRequest req;
    req = fWriteReqsToRetry[it];

    // Get the mem blk to write, directly from the cache, where it should be
    // a unique blk. If it's not there, then this is an internal error.
    void *data = fMainReadCache->FindBlk(req.write.offset, req.write.offset+req.write.dlen-1);

    // Now we have the req and the data, we let the things go almost normally
    // No big troubles, this func is called by the main requesting thread
    if (data) {
      // The recoveries go always through the main stream
      req.write.pathid = 0;
      bool ok = SendGenCommand(&req, data, 0, 0,
					    false, (char *)"Write_checkpoint");

      UnPinCacheBlk(req.write.offset, req.write.offset+req.write.dlen-1);
      // A total sync failure means that there is no more hope and that the destination file is
      // surely incomplete.
      if (!ok) return false;

    }
    else {
      Error("DoWriteSoftCheckPoint", "Checkpoint data disappeared.");
      return false;
    }

  }

  // If we are here, all the requests were successful
  fWriteReqsToRetry.Clear();
  return true;
}

bool XrdClientConn::DoWriteHardCheckPoint() {
  // Do almost the same as the soft checkpoint,
  // But don't exit if there are still pending write reqs
  // This has to guarantee that either a fatal error or a full success has happened
  // Acts like a client-and-network-side flush
  
  while(1) {
    if (ConnectionManager->SidManager()->GetOutstandingWriteRequestCnt(fPrimaryStreamid) == 0) return true;      
    if (!DoWriteSoftCheckPoint()) return false;
    if (ConnectionManager->SidManager()->GetOutstandingWriteRequestCnt(fPrimaryStreamid) == 0) return true;

    //ConnectionManager->SidManager()->PrintoutOutstandingRequests();
    fWriteWaitAck->Wait(1);
  }

}


//_____________________________________________________________________________

void XrdClientConn::SetOpTimeLimit(int delta_secs) {
   fOpTimeLimit = time(0)+delta_secs;
}

bool XrdClientConn::IsOpTimeLimitElapsed(time_t timenow) {
   return (timenow > fOpTimeLimit);
}
