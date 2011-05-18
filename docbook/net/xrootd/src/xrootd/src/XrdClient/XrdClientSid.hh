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
// whose answers are to be processed asynchronously when they arrive.   //
//                                                                      //
////////////////////////////////////////////////////////////////////////// 


//       $Id$



#ifndef XRC_SID_H
#define XRC_SID_H

#include "XrdOuc/XrdOucRash.hh"
#include "XProtocol/XProtocol.hh"
#include "XrdClient/XrdClientProtocol.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdSys/XrdSysPthread.hh"

struct  SidInfo {
   kXR_unt16 fathersid;
   ClientRequest outstandingreq;
   long long reqbyteprogress;
   time_t sendtime;

   kXR_unt16 rspstatuscode;
   kXR_unt32 rsperrno;
   char *rsperrmsg;
};

class XrdClientSid {

 private:
   // Used to quickly get info about a sid being used
   //  as a child of another sid. A child sid is used to parallely
   //  interact with a server for the same logical connection using its father sid
   // Only child sids are inserted here. If a sid is not here but is not free,
   //  then it's a father sid, i.e. normally a sid used for the non async xrootd traffic
   // Remember: for any child sid, it's mandatory to keep the request
   //  which is outstanding for that stream. This struct can be used to
   //  read data, but also for preparing many files in advance.
   XrdOucRash<kXR_unt16, struct SidInfo> childsidnfo;

   // To quickly get a sid which is not being used
   // This one has constant time operations if the ops
   // are performed at the back of the vector
   // Remember: 0 is NOT a valid sid
   XrdClientVector<kXR_unt16> freesids;

   XrdSysMutex                fMutex;
   
 public:
   XrdClientSid();
   virtual ~XrdClientSid();

   // Gets an available sid
   // From now on it will be no more available.
   // A retval of 0 means that there are no more available sids
   kXR_unt16 GetNewSid();

   // Gets an available sid for a request which is to be outstanding
   // This means that this sid will be inserted into the Rash
   // The request gets inserted the new sid in the right place
   // Also the one passed as parameter gets the new sid, as should be expected
   kXR_unt16 GetNewSid(kXR_unt16 sid, ClientRequest *req);

      
   // Releases a sid.
   // It is re-inserted into the available set
   // Its info is rmeoved from the tree
   void ReleaseSid(kXR_unt16 sid);

   // Releases a sid and all its childs
   void ReleaseSidTree(kXR_unt16 fathersid);

   // Report the response for an outstanding request
   // Typically this is used to keep track of the received errors, expecially
   // for async writes
   void ReportSidResp(kXR_unt16 sid, kXR_unt16 statuscode, kXR_unt32 errcode, char *errmsg);

   int GetFailedOutstandingWriteRequests(kXR_unt16 fathersid, XrdClientVector<ClientRequest> &reqvect);
   int GetAllOutstandingWriteRequests(kXR_unt16 fathersid, XrdClientVector<ClientRequest> &reqvect);
   int GetOutstandingWriteRequestCnt(kXR_unt16 fathersid);
 
   // 0 if non existent as a child sid
   inline struct SidInfo *GetSidInfo(kXR_unt16 sid) {
      XrdSysMutexHelper l(fMutex);
      return (childsidnfo.Find(sid));
   };

   inline bool JoinedSids(kXR_unt16 father, kXR_unt16 child) {
      XrdSysMutexHelper l(fMutex);

      struct SidInfo *si = childsidnfo.Find(child);

      if (!si) return false;
      return (si->fathersid == father);
   }


  // Useful for debugging
  void PrintoutOutstandingRequests();
};








#endif
