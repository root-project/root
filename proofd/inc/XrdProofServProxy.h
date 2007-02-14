// @(#)root/proofd:$Name:  $:$Id: XrdProofServProxy.h,v 1.8 2006/11/27 14:19:58 rdm Exp $
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofServProxy
#define ROOT_XrdProofServProxy

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>

#include <list>
#include <map>
#include <vector>

#include "Xrd/XrdLink.hh"
#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucSemWait.hh"

#include "XrdProofdResponse.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdSrvBuffer                                                         //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// The following structure is used to store buffers to be sent or       //
// received from clients                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class XrdSrvBuffer {
public:
   int   fSize;
   char *fBuff;

   XrdSrvBuffer(char *bp=0, int sz=0, bool dup=0) {
      if (dup && bp && sz > 0) {
         fMembuf = (char *)malloc(sz);
         if (fMembuf) {
            memcpy(fMembuf, bp, sz);
            fBuff = fMembuf;
            fSize = sz;
         }
      } else {
         fBuff = fMembuf = bp;
         fSize = sz;
      }}
   ~XrdSrvBuffer() {if (fMembuf) free(fMembuf);}

private:
   char *fMembuf;
};


class XrdProofdProtocol;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientID                                                          //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// Mapping of clients and stream IDs                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class XrdClientID {
public:
   XrdProofdProtocol *fP;
   unsigned short     fSid;

   XrdClientID(XrdProofdProtocol *pt = 0, unsigned short id = 0)
            { fP = pt; fSid = id; }
   ~XrdClientID() { }

   bool   IsValid() const { return (fP != 0); }
   void   Reset() { fP = 0; fSid = 0; }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofServProxy                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// This class represent an instance of TProofServ                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#define kXPROOFSRVTAGMAX   64
#define kXPROOFSRVALIASMAX 256

class XrdProofWorker;
class XrdNet;

class XrdProofServProxy
{

// friend class XrdProofdProtocol;

public:
   XrdProofServProxy();
   ~XrdProofServProxy();

   inline const char  *Alias() const { XrdOucMutexHelper mhp(fMutex); return fAlias; }
   inline const char  *Client() const { XrdOucMutexHelper mhp(fMutex); return fClient; }
   inline const char  *Fileout() const { XrdOucMutexHelper mhp(fMutex); return fFileout; }
   inline short int    ID() const { XrdOucMutexHelper mhp(fMutex); return fID; }
   inline bool         IsParent(XrdProofdProtocol *p) const
                                 { XrdOucMutexHelper mhp(fMutex); return (fParent && fParent->fP == p); }
   inline XrdLink     *Link() const { XrdOucMutexHelper mhp(fMutex); return fLink; }
   inline bool         Match(short int id) const { XrdOucMutexHelper mhp(fMutex); return (id == fID); }
   inline XrdOucMutex *Mutex() { return fMutex; }
   inline XrdProofdResponse *ProofSrv() const
                      { XrdOucMutexHelper mhp(fMutex); return (XrdProofdResponse *)&fProofSrv;}
   inline XrdOucSemWait *PingSem() const { XrdOucMutexHelper mhp(fMutex); return fPingSem; }
   inline const char  *Ordinal() const { XrdOucMutexHelper mhp(fMutex); return (const char *)fOrdinal; }
   inline XrdSrvBuffer *QueryNum() const { XrdOucMutexHelper mhp(fMutex); return fQueryNum; }
   inline int          SrvID() const { XrdOucMutexHelper mhp(fMutex); return fSrvID; }
   inline int          SrvType() const { XrdOucMutexHelper mhp(fMutex); return fSrvType; }
   inline void         SetLink(XrdLink *lnk) { XrdOucMutexHelper mhp(fMutex); fLink = lnk;}
   inline void         SetID(short int id) { XrdOucMutexHelper mhp(fMutex); fID = id;}
   inline void         SetParent(XrdClientID *cid) { XrdOucMutexHelper mhp(fMutex); fParent = cid; }
   inline void         SetProtVer(int pv) { XrdOucMutexHelper mhp(fMutex); fProtVer = pv; }
   inline void         SetQueryNum(XrdSrvBuffer *qn) { XrdOucMutexHelper mhp(fMutex); fQueryNum = qn; }
   inline void         SetSrv(int id) { XrdOucMutexHelper mhp(fMutex); fSrvID = id; }
   inline void         SetSrvType(int id) { XrdOucMutexHelper mhp(fMutex); fSrvType = id; }
   inline void         SetStartMsg(XrdSrvBuffer *sm) { XrdOucMutexHelper mhp(fMutex); fStartMsg = sm; }
   inline void         SetStatus(int st) { XrdOucMutexHelper mhp(fMutex); fStatus = st; }
   inline void         SetShutdown(bool sd = 1) { XrdOucMutexHelper mhp(fMutex); fIsShutdown = sd; }
   inline void         SetValid(bool valid = 1) { XrdOucMutexHelper mhp(fMutex); fIsValid = valid; }
   inline XrdSrvBuffer *StartMsg() const { XrdOucMutexHelper mhp(fMutex); return fStartMsg; }
   inline int          Status() const { XrdOucMutexHelper mhp(fMutex); return fStatus;}
   inline const char  *Tag() const { XrdOucMutexHelper mhp(fMutex); return fTag; }
   inline const char  *UserEnvs() const { XrdOucMutexHelper mhp(fMutex); return fUserEnvs; }

   void                CreatePingSem()
                       { XrdOucMutexHelper mhp(fMutex); fPingSem = new XrdOucSemWait(0);}
   void                DeletePingSem()
                       { XrdOucMutexHelper mhp(fMutex); if (fPingSem) delete fPingSem; fPingSem = 0;}

   void                DeleteQueryNum()
                       { XrdOucMutexHelper mhp(fMutex); if (fQueryNum) delete fQueryNum; fQueryNum = 0;}
   void                DeleteStartMsg()
                       { XrdOucMutexHelper mhp(fMutex); if (fStartMsg) delete fStartMsg; fStartMsg = 0;}

   XrdClientID        *GetClientID(int cid);
   int                 GetFreeID();
   int                 GetNClients();

   inline XrdClientID        *Parent() const { XrdOucMutexHelper mhp(fMutex); return fParent; }
   inline std::vector<XrdClientID *> *Clients() const
                      { XrdOucMutexHelper mhp(fMutex); return (std::vector<XrdClientID *> *)&fClients; }
   inline std::list<XrdProofWorker *> *Workers() const
                      { XrdOucMutexHelper mhp(fMutex); return (std::list<XrdProofWorker *> *)&fWorkers; }

   int                 GetNWorkers() { XrdOucMutexHelper mhp(fMutex); return (int) fWorkers.size(); }
   void                AddWorker(XrdProofWorker *w) { XrdOucMutexHelper mhp(fMutex); fWorkers.push_back(w); }
   void                RemoveWorker(XrdProofWorker *w) { XrdOucMutexHelper mhp(fMutex); fWorkers.remove(w); }

   void                SetAlias(const char *a, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fAlias, a, l); }
   void                SetClient(const char *c, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fClient, c, l); }
   void                SetFileout(const char *f, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fFileout, f, l); }
   void                SetOrdinal(const char *o, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fOrdinal, o, l); }
   void                SetTag(const char *t, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fTag, t, l); }
   void                SetUserEnvs(const char *t, int l = 0)
                          { XrdOucMutexHelper mhp(fMutex); XrdProofServProxy::SetCharValue(&fUserEnvs, t, l); }

   bool                IsShutdown() const { XrdOucMutexHelper mhp(fMutex); return fIsShutdown; }
   bool                IsValid() const { XrdOucMutexHelper mhp(fMutex); return fIsValid; }
   const char         *StatusAsString() const;

   void                Reset();

 private:

   XrdOucRecMutex           *fMutex;
   XrdLink                  *fLink;      // Link to proofsrv
   XrdProofdResponse         fProofSrv;  // Utility to talk to proofsrv

   XrdClientID              *fParent;    // Parent creating this session
   std::vector<XrdClientID *> fClients;  // Attached clients stream ids
   std::list<XrdProofWorker *> fWorkers; // Workers assigned to the session

   XrdOucSemWait            *fPingSem;   // To sychronize ping requests

   XrdSrvBuffer             *fQueryNum;  // Msg with sequential number of currebt query
   XrdSrvBuffer             *fStartMsg;  // Msg with start processing info

   int                       fStatus;
   int                       fSrvID;     // Srv process ID
   int                       fSrvType;
   short int                 fID;
   char                      fProtVer;
   char                     *fFileout;

   bool                      fIsValid;   // Validity flag
   bool                      fIsShutdown; // Whether asked to shutdown

   char                     *fAlias;     // Session alias
   char                     *fClient;    // Client name
   char                     *fTag;       // Session unique tag
   char                     *fOrdinal;   // Session ordinal number
   char                     *fUserEnvs;  // List of envs received from the user

   void                      ClearWorkers();

   static void               SetCharValue(char **carr, const char *v, int len = 0);
};
#endif
