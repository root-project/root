//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientMessage                                                     //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2005)                          //
//                                                                      //
// Utility classes to handle the mapping between xrootd streamids.      //
//  A single streamid can have multiple "parallel" streamids.           //
//  Their use is typically to support the client to submit multiple     //
// parallel requests (belonging to the same LogConnectionID),           //
// which are to be processed asynchronously when the answers arrive.    //
//                                                                      //
////////////////////////////////////////////////////////////////////////// 


//       $Id$

const char *XrdClientSidCVSID = "$Id$";


#include "XrdClient/XrdClientSid.hh"
#include "XrdClient/XrdClientEnv.hh"
#include "XrdClient/XrdClientConst.hh"

XrdClientSid::XrdClientSid() {

   freesids.Resize(65536);

   // We populate the free sids queue
   for (kXR_unt16 i = 65535; i >= 1; i--)
      freesids.Push_back(i);
}

XrdClientSid::~XrdClientSid() {
   freesids.Clear();
   childsidnfo.Purge();

}


// Gets an available sid
// From now on it will be no more available.
// A retval of 0 means that there are no more available sids
kXR_unt16 XrdClientSid::GetNewSid() {
   XrdSysMutexHelper l(fMutex);

   if (!freesids.GetSize()) return 0;
      
   return (freesids.Pop_back());

};

// Gets an available sid for a request which is to be outstanding
// This means that this sid will be inserted into the Rash
// The request gets inserted the new sid in the right place
// Also the one passed as parameter gets the new sid, as should be expected
kXR_unt16 XrdClientSid::GetNewSid(kXR_unt16 sid, ClientRequest *req) {
  XrdSysMutexHelper l(fMutex);

  if (!freesids.GetSize()) return 0;
      
  kXR_unt16 nsid = freesids.Pop_back();
  
  if (nsid) {
    struct SidInfo si;
    
    memcpy(req->header.streamid, &nsid, sizeof(req->header.streamid));

    si.fathersid = sid;
    si.outstandingreq = *req;
    si.reqbyteprogress = 0;
    si.sendtime = time(0);

    si.rspstatuscode = 0;
    si.rsperrno = kXR_noErrorYet;
    si.rsperrmsg = 0;

    childsidnfo.Add(nsid, si);
  }
  
  
  
  return nsid;
  
};

// Report the response for an outstanding request
// Typically this is used to keep track of the received errors, expecially
// for async writes
void XrdClientSid::ReportSidResp(kXR_unt16 sid, kXR_unt16 statuscode, kXR_unt32 errcode, char *errmsg) {
  XrdSysMutexHelper l(fMutex);
  struct SidInfo *si = childsidnfo.Find(sid);
  
  if (si) {
     si->rspstatuscode = statuscode;
     si->rsperrno = errcode;
     if (si->rsperrmsg) free(si->rsperrmsg);

     if (errmsg) si->rsperrmsg = strdup(errmsg);
     else si->rsperrmsg = 0;
  }
    
};

// Releases a sid.
// It is re-inserted into the available set
// Its info is rmeoved from the tree
void XrdClientSid::ReleaseSid(kXR_unt16 sid) {
   XrdSysMutexHelper l(fMutex);

   childsidnfo.Del(sid);
   freesids.Push_back(sid);
};



//_____________________________________________________________________________
struct ReleaseSidTreeItem_data {
  kXR_unt16 fathersid;
  XrdClientVector<kXR_unt16> *freesids;
};
int ReleaseSidTreeItem(kXR_unt16 key,
		       struct SidInfo si, void *arg) {

  ReleaseSidTreeItem_data *data = (ReleaseSidTreeItem_data *)arg;

  // If the sid we have is a son of the given father then delete it
  if (si.fathersid == data->fathersid) {
     free(si.rsperrmsg);
     data->freesids->Push_back(key);
     return -1;
  }

  return 0;
}

// Releases a sid and all its childs
void XrdClientSid::ReleaseSidTree(kXR_unt16 fathersid) {
   XrdSysMutexHelper l(fMutex);

   ReleaseSidTreeItem_data data;
   data.fathersid = fathersid;
   data.freesids = &freesids;

   childsidnfo.Apply(ReleaseSidTreeItem, static_cast<void *>(&data));
   freesids.Push_back(fathersid);
}





static int printoutreq(kXR_unt16,
                       struct SidInfo p, void *) {

  smartPrintClientHeader(&p.outstandingreq);
  return 0;
}

void XrdClientSid::PrintoutOutstandingRequests() {
  cerr << "-------------------------------------------------- start outstanding reqs dump. freesids: " << freesids.GetSize() << endl;

  childsidnfo.Apply(printoutreq, this);
  cerr << "++++++++++++++++++++++++++++++++++++++++++++++++++++ end  outstanding reqs dump." << endl;
}


struct sniffOutstandingFailedWriteReq_data {
  XrdClientVector<ClientRequest> *reqs;
  kXR_unt16 fathersid;
  XrdClientVector<kXR_unt16> *freesids;
};
static int sniffOutstandingFailedWriteReq(kXR_unt16 sid,
				    struct SidInfo p, void *d) {

  sniffOutstandingFailedWriteReq_data *data = (sniffOutstandingFailedWriteReq_data *)d;
  if ((p.fathersid == data->fathersid) &&
      (p.outstandingreq.header.requestid == kXR_write)) {

    // If it went into timeout or got a negative response
    // we add this req to the vector
    if ( (time(0) - p.sendtime > EnvGetLong(NAME_REQUESTTIMEOUT)) ||
	 (p.rspstatuscode != kXR_ok) ) {
      data->reqs->Push_back(p.outstandingreq);

      // And we release the failed sid
      free(p.rsperrmsg);
      data->freesids->Push_back(sid);
      return -1;
    }

  }

  //  smartPrintClientHeader(&p.outstandingreq);
  return 0;
}

static int sniffOutstandingAllWriteReq(kXR_unt16 sid,
				    struct SidInfo p, void *d) {

  sniffOutstandingFailedWriteReq_data *data = (sniffOutstandingFailedWriteReq_data *)d;
  if ((p.fathersid == data->fathersid) &&
      (p.outstandingreq.header.requestid == kXR_write)) {

      // we add this req to the vector
      data->reqs->Push_back(p.outstandingreq);

      // And we release the failed sid
      free(p.rsperrmsg);
      data->freesids->Push_back(sid);
      return -1;
  }

  //  smartPrintClientHeader(&p.outstandingreq);
  return 0;
}

struct countOutstandingWriteReq_data {
  int cnt;
  kXR_unt16 fathersid;
};
static int countOutstandingWriteReq(kXR_unt16 sid,
				    struct SidInfo p, void *c) {

  countOutstandingWriteReq_data *data = (countOutstandingWriteReq_data *)c;

  if ((p.fathersid == data->fathersid) && (p.outstandingreq.header.requestid == kXR_write))
    data->cnt++;

  //  smartPrintClientHeader(&p.outstandingreq);
  return 0;
}

int XrdClientSid::GetFailedOutstandingWriteRequests(kXR_unt16 fathersid, XrdClientVector<ClientRequest> &reqvect) {
  sniffOutstandingFailedWriteReq_data data;
  data.reqs = &reqvect;
  data.fathersid = fathersid;
  data.freesids = &freesids;

  childsidnfo.Apply(sniffOutstandingFailedWriteReq, (void *)&data);
  return reqvect.GetSize();
}

int XrdClientSid::GetOutstandingWriteRequestCnt(kXR_unt16 fathersid) {
  countOutstandingWriteReq_data data;
  data.fathersid = fathersid;
  data.cnt = 0;
  childsidnfo.Apply(countOutstandingWriteReq, (void *)&data);
  return data.cnt;
}



int XrdClientSid::GetAllOutstandingWriteRequests(kXR_unt16 fathersid, XrdClientVector<ClientRequest> &reqvect) {
  sniffOutstandingFailedWriteReq_data data;
  data.reqs = &reqvect;
  data.fathersid = fathersid;
  data.freesids = &freesids;

  childsidnfo.Apply(sniffOutstandingAllWriteReq, (void *)&data);
  return reqvect.GetSize();
}
