//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientMStream                                                     //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2006)                          //
//                                                                      //
// Helper code for XrdClient to handle multistream behavior             //
// Functionalities dealing with                                         //
//  mstream creation on init                                            //
//  decisions to add/remove one                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//         $Id$

const char *XrdClientMStreamCVSID = "$Id$";


#include "XrdClient/XrdClientMStream.hh"
#include "XrdClient/XrdClientLogConnection.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientDebug.hh"

// This has to be a socket id pool which the server will never assign by itself
// Moreover, socketids are local to an instance of XrdClientPSock
#define XRDCLI_PSOCKTEMP -1000

struct ParStreamOpenerArgs {
   XrdClientThread *thr;
   XrdClientConn *cliconn;
   int wan_port, wan_window;
   int tmpid;
};

//_____________________________________________________________________________
void *ParStreamOpenerThread(void *arg, XrdClientThread *thr)
{
   // This one just opens a new stream

   // Mask all allowed signals
   if (thr->MaskSignal(0) != 0)
      Error("ParStreamOpenerThread", "Warning: problems masking signals");

   ParStreamOpenerArgs *parms = (ParStreamOpenerArgs *)arg;

   XrdClientMStream::AddParallelStream(parms->cliconn, parms->wan_port, parms->wan_window, parms->tmpid);

   return 0;
}


int XrdClientMStream::EstablishParallelStreams(XrdClientConn *cliconn) {
    int mx = EnvGetLong(NAME_MULTISTREAMCNT);
    int i, res;
    int wan_port = 0, wan_window = 0;

    if (mx <= 1) return 1;
    if (cliconn->GetServerType() == kSTBaseXrootd) return 1;

    // Get the XrdClientPhyconn to be used
    XrdClientPhyConnection *phyconn = XrdClientConn::GetPhyConn(cliconn->GetLogConnID());
    if (!phyconn) return 0;
    
    // For a given phyconn we allow only one single attempt to establish multiple streams
    // Any other thread or subsequent attempt will exit
    if (phyconn->TestAndSetMStreamsGoing()) return 1;

    // Query the server config, for the WAN port and the windowsize
    char *qryitems = (char *)"wan_port wan_window";
    ClientRequest qryRequest;
    char qryResp[1024];
    memset( &qryRequest, 0, sizeof(qryRequest) );
    memset( qryResp, 0, 1024 );

    cliconn->SetSID(qryRequest.header.streamid);
    qryRequest.header.requestid = kXR_query;
    qryRequest.query.infotype = kXR_Qconfig;
    qryRequest.header.dlen = strlen(qryitems);

    res =  cliconn->SendGenCommand(&qryRequest, qryitems, 0, qryResp,
				   false, (char *)"QueryConfig");

    if (res && (cliconn->LastServerResp.status == kXR_ok) &&
	cliconn->LastServerResp.dlen) {

      sscanf(qryResp, "%d\n%d",
	     &wan_port,
	     &wan_window);

      Info(XrdClientDebug::kUSERDEBUG,
	   "XrdClientMStream::EstablishParallelStreams", "Server WAN parameters: port=" << wan_port << " windowsize=" << wan_window );
    }

    // Start the whole bunch of asynchronous connection requests
    // By starting one thread for each, calling AddParallelStream once
    // If no more threads are available, wait and retry

    ParStreamOpenerArgs paropeners[16];
    for (i = 0; i < mx; i++) {
       paropeners[i].thr = 0;
       paropeners[i].cliconn = cliconn;
       paropeners[i].wan_port = wan_port;
       paropeners[i].wan_window = wan_window;
       paropeners[i].tmpid = 0;
    }

    for (i = 0; i < mx; i++) {
	Info(XrdClientDebug::kHIDEBUG,
	     "XrdClientMStream::EstablishParallelStreams", "Trying to establish " << i+1 << "th substream." );

        paropeners[i].thr = new XrdClientThread(ParStreamOpenerThread);
        if (paropeners[i].thr) {
           paropeners[i].tmpid = XRDCLI_PSOCKTEMP - i;
           if (paropeners[i].thr->Run(&paropeners[i])) {
              Error("XrdClientMStream::EstablishParallelStreams", "Error establishing " << i+1 << "th substream. Thread start failed.");
              delete paropeners[i].thr;
              paropeners[i].thr = 0;
              break; 
           }
        }

    }

    for (i = 0; i < mx; i++)
       if (paropeners[i].thr) {
          Info(XrdClientDebug::kHIDEBUG,
             "XrdClientMStream::EstablishParallelStreams", "Waiting for substream " << i+1 << "." );
          paropeners[i].thr->Join(0);
          delete paropeners[i].thr;
       }

	// If something goes wrong, stop adding new streams
	//if (AddParallelStream(cliconn, wan_port, wan_window, XRDCLI_PSOCKTEMP - i))
	//    break;


    Info(XrdClientDebug::kHIDEBUG,
         "XrdClientMStream::EstablishParallelStreams", "Parallel streams establishment finished." );

    return i;
}

// Add a parallel stream to the pool used by the given client inst
// Returns 0 if ok
int XrdClientMStream::AddParallelStream(XrdClientConn *cliconn, int port, int windowsz, int tempid) {
    // Get the XrdClientPhyconn to be used
    XrdClientPhyConnection *phyconn = XrdClientConn::GetPhyConn(cliconn->GetLogConnID());


    // If the phyconn already has all the needed streams... exit
    if (phyconn->GetSockIdCount() > EnvGetLong(NAME_MULTISTREAMCNT)) return 0;

    // Connect a new connection, set the temp socket id and get the descriptor
    // Temporary means that we need one to communicate, but its final id
    // will be given by the server
    int sockdescr = phyconn->TryConnectParallelStream(port, windowsz, tempid);

    if (sockdescr < 0) return -1;

    // The connection now is here but has not yet to be considered by the reader threads
    // before having handshaked it, and this has to be sync man
    // Do the handshake
    ServerInitHandShake xbody;
    if (phyconn->DoHandShake(xbody, tempid) == kSTError) return -1;

    // Send the kxr_bind req to get a new substream id, going to be the final one
    int newid = -1;
    int res = -1;
    if (BindPendingStream(cliconn, tempid, newid) &&
	phyconn->IsValid() ) {
      
	// Everything ok, Establish the new connection with the new id
        res = phyconn->EstablishPendingParallelStream(tempid, newid);
    
	if (res) {
	    // If the establish failed we have to remove the pending stream
	    RemoveParallelStream(cliconn, tempid);
	    return res;
	}

        // After everything make the reader thread aware of the new stream
        phyconn->UnBanSockDescr(sockdescr);
        phyconn->ReinitFDTable();
    
    }
    else {
	// If the bind failed we have to remove the pending stream
	RemoveParallelStream(cliconn, tempid);
	return -1;
    }

    Info(XrdClientDebug::kHIDEBUG,
	 "XrdClientMStream::EstablishParallelStreams", "Substream added." );
    return 0;

}

// Remove a parallel stream to the pool used by the given client inst
int XrdClientMStream::RemoveParallelStream(XrdClientConn *cliconn, int substream) {

  // Get the XrdClientPhyconn to be used
  XrdClientLogConnection *log = ConnectionManager->GetConnection(cliconn->GetLogConnID());
  if (!log) return 0;

  XrdClientPhyConnection *phyconn = log->GetPhyConnection();
    
  if (phyconn) 
    phyconn->RemoveParallelStream(substream);
    
  return 0;
    
}



// Binds the pending temporary parallel stream to the current session
// Returns the substreamid assigned by the server into newid
bool XrdClientMStream::BindPendingStream(XrdClientConn *cliconn, int substreamid, int &newid) {
    bool res = false;

    // Prepare request
    ClientRequest bindFileRequest;
    XrdClientConn::SessionIDInfo sess;
    ServerResponseBody_Bind bndresp;

    // Get the XrdClientPhyconn to be used
    XrdClientPhyConnection *phyconn =
	ConnectionManager->GetConnection(cliconn->GetLogConnID())->GetPhyConnection();

    cliconn->GetSessionID(sess);

    memset( &bindFileRequest, 0, sizeof(bindFileRequest) );
    cliconn->SetSID(bindFileRequest.header.streamid);
    bindFileRequest.bind.requestid = kXR_bind;
    memcpy( bindFileRequest.bind.sessid, sess.id, sizeof(sess.id) );

    // The request has to be sent through the stream which has to be bound!
    clientMarshall(&bindFileRequest);
    res = phyconn->WriteRaw(&bindFileRequest, sizeof(bindFileRequest), substreamid);

    if (!res) return false;

    ServerResponseHeader hdr;
    int rdres = 0;

    // Now wait for the header, on the same substream
    rdres = phyconn->ReadRaw(&hdr, sizeof(ServerResponseHeader), substreamid);

    if (rdres < (int)sizeof(ServerResponseHeader)) {
       Error("BindPendingStream", "Error reading bind response header for substream " << substreamid << ".");
       return false;
    }

    clientUnmarshall(&hdr);

    // Now wait for the response data, if any
    // This code is specialized.
    // If the answer is not what we were expecting, just return false,
    //  expecting that this connection will be shut down
    if (hdr.status != kXR_ok) {
       Error("BindPendingStream", "Server denied binding for substream " << substreamid << ".");
       return false;
    }

    if (hdr.dlen != sizeof(bndresp)) {
       Error("BindPendingStream", "Unrecognized response datalen binding substream " << substreamid << ".");
       return false;
    }

    rdres = phyconn->ReadRaw(&bndresp, sizeof(bndresp), substreamid);
    if (rdres != sizeof(bndresp)) {
       Error("BindPendingStream", "Error reading response binding substream " << substreamid << ".");
       return false;
    }

    newid = bndresp.substreamid;

    return res;

}


void XrdClientMStream::GetGoodSplitParameters(XrdClientConn *cliconn,
					      int &spltsize, int &reqsperstream,
					      kXR_int32 len) {
  spltsize = DFLT_MULTISTREAMSPLITSIZE;
  reqsperstream = 4;


  // Let's try to distribute the load into maximum sized chunks
  if (cliconn->GetParallelStreamCount() > 1) {
    
    // We start seeing which length we get trying to fill all the
    // available slots ( per stream)
    int candlen = xrdmax(DFLT_MULTISTREAMSPLITSIZE,
			 len / (reqsperstream * (cliconn->GetParallelStreamCount()-1)) + 1 );
    
    // We don't want blocks smaller than a min value
    // If this is the case we consider only one slot per stream
    if (candlen < DFLT_MULTISTREAMSPLITSIZE) {
      spltsize = xrdmax(DFLT_MULTISTREAMSPLITSIZE,
			len / (cliconn->GetParallelStreamCount()-1) + 1 );
      reqsperstream = 1;
    }
    else spltsize = candlen;
    
  }
  else spltsize = len;

  //cout << "parstreams: " << cliconn->GetParallelStreamCount() <<
  // " len: " << len << " splitsize: " << spltsize << " reqsperstream: " <<
  // reqsperstream << endl << endl;
}


// This splits a long requests into many smaller requests, to be sent in parallel
//  through multiple streams
// Returns false if the chunk is not worth splitting
bool XrdClientMStream::SplitReadRequest(XrdClientConn *cliconn, kXR_int64 offset, kXR_int32 len,
			     XrdClientVector<XrdClientMStream::ReadChunk> &reqlists) {

    int spltsize = 0;
    int reqsperstream = 0;

    GetGoodSplitParameters(cliconn, spltsize, reqsperstream, len);
    for (kXR_int32 pp = 0; pp < len; pp += spltsize) {
      ReadChunk ck;

      ck.offset = pp+offset;
      ck.len = xrdmin(len - pp, spltsize);
      ck.streamtosend = cliconn->GetParallelStreamToUse(reqsperstream);

      reqlists.Push_back(ck);

    }

    return true;
}
