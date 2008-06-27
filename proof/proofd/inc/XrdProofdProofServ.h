// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdProofServ
#define ROOT_XrdProofdProofServ

#include <string.h>
#include <unistd.h>
#include <sys/uio.h>
#if !defined(__FreeBSD__) && !defined(__OpenBSD__) && !defined(__APPLE__)
#include <sched.h>
#endif

#include <list>
#include <map>
#include <vector>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#  include "XrdOuc/XrdOucSemWait.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#  include "XrdSys/XrdSysSemWait.hh"
#endif

#include "Xrd/XrdLink.hh"

#include "XrdProofdProtocol.h"
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


class XrdROOT;

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
private:
   XrdProofdProtocol *fP;
   unsigned short     fSid;

public:
   XrdClientID(XrdProofdProtocol *pt = 0, unsigned short id = 0)
            { fP = pt; fSid = id; }
   ~XrdClientID() { }

   XrdProofdClient   *C() const { return fP->Client(); }
   bool               IsValid() const { return (fP != 0); }
   XrdProofdProtocol *P() const { return fP; }
   void               Reset() { fP = 0; fSid = 0; }
   void               SetP(XrdProofdProtocol *p) { fP = p; }
   void               SetSid(unsigned short sid) { fSid = sid; }
   unsigned short     Sid() const { return fSid; }
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProofServ                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// This class represent an instance of TProofServ                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#define kXPROOFSRVTAGMAX   64
#define kXPROOFSRVALIASMAX 256

class XrdProofGroup;
class XrdProofWorker;
class XrdNet;
class XrdSysSemWait;

class XrdProofdProofServ
{

// friend class XrdProofdProtocol;

public:
   XrdProofdProofServ();
   ~XrdProofdProofServ();

   inline const char  *Alias() const { XrdSysMutexHelper mhp(fMutex); return fAlias; }
   inline const char  *Client() const { XrdSysMutexHelper mhp(fMutex); return fClient; }
   inline const char  *Fileout() const { XrdSysMutexHelper mhp(fMutex); return fFileout; }
   inline float        FracEff() const { XrdSysMutexHelper mhp(fMutex); return fFracEff; }
   inline XrdProofGroup *Group() const { XrdSysMutexHelper mhp(fMutex); return fGroup; }
   inline short int    ID() const { XrdSysMutexHelper mhp(fMutex); return fID; }
   inline bool         IsParent(XrdProofdProtocol *p) const
                                 { XrdSysMutexHelper mhp(fMutex); return (fParent && fParent->P() == p); }
   inline XrdLink     *Link() const { XrdSysMutexHelper mhp(fMutex); return fLink; }
   inline bool         Match(short int id) const { XrdSysMutexHelper mhp(fMutex); return (id == fID); }
   inline XrdSysMutex *Mutex() { return fMutex; }
   inline XrdProofdResponse *ProofSrv() const
                      { XrdSysMutexHelper mhp(fMutex); return (XrdProofdResponse *)&fProofSrv;}
   inline XrdSysSemWait *PingSem() const { XrdSysMutexHelper mhp(fMutex); return fPingSem; }
   inline const char  *Ordinal() const { XrdSysMutexHelper mhp(fMutex); return (const char *)fOrdinal; }
   inline XrdSrvBuffer *QueryNum() const { XrdSysMutexHelper mhp(fMutex); return fQueryNum; }
   inline XrdSrvBuffer *Requirements() const { XrdSysMutexHelper mhp(fMutex); return fRequirements; }
   inline XrdROOT     *ROOT() const { XrdSysMutexHelper mhp(fMutex); return fROOT; }
   inline int          SrvID() const { XrdSysMutexHelper mhp(fMutex); return fSrvID; }
   inline int          SrvType() const { XrdSysMutexHelper mhp(fMutex); return fSrvType; }
   inline void         SetFracEff(float ef) { XrdSysMutexHelper mhp(fMutex); fFracEff = ef; }
   inline void         SetGroup(XrdProofGroup *g) { XrdSysMutexHelper mhp(fMutex); fGroup = g; }
   inline void         SetID(short int id) { XrdSysMutexHelper mhp(fMutex); fID = id;}
   inline void         SetLink(XrdLink *lnk) { XrdSysMutexHelper mhp(fMutex); fLink = lnk;}
   inline void         SetParent(XrdClientID *cid) { XrdSysMutexHelper mhp(fMutex); fParent = cid; }
   inline void         SetProtVer(int pv) { XrdSysMutexHelper mhp(fMutex); fProtVer = pv; }
   inline void         SetQueryNum(XrdSrvBuffer *qn) { XrdSysMutexHelper mhp(fMutex); fQueryNum = qn; }
   inline void         SetRequirements(XrdSrvBuffer *rq)
                          { XrdSysMutexHelper mhp(fMutex); fRequirements = rq; }
   inline void         SetROOT(XrdROOT *r) { XrdSysMutexHelper mhp(fMutex); fROOT = r; }
   inline void         SetSrvType(int id) { XrdSysMutexHelper mhp(fMutex); fSrvType = id; }
   inline void         SetStartMsg(XrdSrvBuffer *sm) { XrdSysMutexHelper mhp(fMutex); fStartMsg = sm; }
   inline void         SetStatus(int st) { XrdSysMutexHelper mhp(fMutex); fStatus = st; }
   inline void         SetShutdown(bool sd = 1) { XrdSysMutexHelper mhp(fMutex); fIsShutdown = sd; }
   inline void         SetValid(bool valid = 1) { XrdSysMutexHelper mhp(fMutex); fIsValid = valid; }
   inline XrdSrvBuffer *StartMsg() const { XrdSysMutexHelper mhp(fMutex); return fStartMsg; }
   inline int          Status() const { XrdSysMutexHelper mhp(fMutex); return fStatus;}
   inline const char  *Tag() const { XrdSysMutexHelper mhp(fMutex); return fTag; }
   inline const char  *UserEnvs() const { XrdSysMutexHelper mhp(fMutex); return fUserEnvs; }

   void                CreatePingSem()
                       { XrdSysMutexHelper mhp(fMutex); fPingSem = new XrdSysSemWait(0);}
   void                DeletePingSem()
                       { XrdSysMutexHelper mhp(fMutex); if (fPingSem) delete fPingSem; fPingSem = 0;}

   void                DeleteQueryNum()
                       { XrdSysMutexHelper mhp(fMutex); if (fQueryNum) delete fQueryNum; fQueryNum = 0;}
   void                DeleteStartMsg()
                       { XrdSysMutexHelper mhp(fMutex); if (fStartMsg) delete fStartMsg; fStartMsg = 0;}

   XrdClientID        *GetClientID(int cid);
   int                 GetFreeID();
   int                 GetNClients();

   inline XrdClientID        *Parent() const { XrdSysMutexHelper mhp(fMutex); return fParent; }
   inline std::vector<XrdClientID *> *Clients() const
                      { XrdSysMutexHelper mhp(fMutex); return (std::vector<XrdClientID *> *)&fClients; }
   inline std::list<XrdProofWorker *> *Workers() const
                      { XrdSysMutexHelper mhp(fMutex); return (std::list<XrdProofWorker *> *)&fWorkers; }

   int                 GetNWorkers() { XrdSysMutexHelper mhp(fMutex); return (int) fWorkers.size(); }
   void                AddWorker(XrdProofWorker *w) { XrdSysMutexHelper mhp(fMutex); fWorkers.push_back(w); }
   void                RemoveWorker(XrdProofWorker *w) { XrdSysMutexHelper mhp(fMutex); fWorkers.remove(w); }

   void                SetAlias(const char *a, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fAlias, a, l); }
   void                SetClient(const char *c, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fClient, c, l); }
   void                SetFileout(const char *f, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fFileout, f, l); }
   void                SetOrdinal(const char *o, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fOrdinal, o, l); }
   void                SetTag(const char *t, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fTag, t, l); }
   void                SetUserEnvs(const char *t, int l = 0)
                          { XrdSysMutexHelper mhp(fMutex); SetCharValue(&fUserEnvs, t, l); }

   bool                IsShutdown() const { XrdSysMutexHelper mhp(fMutex); return fIsShutdown; }
   bool                IsValid() const { XrdSysMutexHelper mhp(fMutex); return fIsValid; }
   const char         *StatusAsString() const;

   int                 GetDefaultProcessPriority();
   int                 SetProcessPriority(int priority = -99999);
   int                 SetShutdownTimer(int opt, int delay, bool on = 1);
   int                 TerminateProofServ();
   int                 VerifyProofServ(int timeout);

   int                 BroadcastPriority(int priority);
   int                 SetInflate(int inflate, bool sendover);
   int                 SetSchedRoundRobin(bool on = 1);
   void                SetSrv(int id);

   void                Reset();

 private:

   XrdSysRecMutex           *fMutex;
   XrdLink                  *fLink;      // Link to proofsrv
   XrdProofdResponse         fProofSrv;  // Utility to talk to proofsrv

   XrdClientID              *fParent;    // Parent creating this session
   std::vector<XrdClientID *> fClients;  // Attached clients stream ids
   std::list<XrdProofWorker *> fWorkers; // Workers assigned to the session

   XrdSysSemWait            *fPingSem;   // To sychronize ping requests

   XrdSrvBuffer             *fQueryNum;  // Msg with sequential number of currebt query
   XrdSrvBuffer             *fStartMsg;  // Msg with start processing info

   XrdSrvBuffer             *fRequirements;  // Buffer with session requirements

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

   XrdROOT                  *fROOT;      // ROOT version run by this session

   XrdProofGroup            *fGroup;     // Group, if any, to which the owner belongs

   int                       fInflate;   // Inflate factor in 1/1000
   int                       fSched;     // Current scheduler policy 
   int                       fDefSched;  // Default scheduler policy 
   struct sched_param        fDefSchedParam;    // Default scheduling param
   int                       fDefSchedPriority; // Default scheduling priority
   float                     fFracEff;   // Effective resource fraction

   void                      ClearWorkers();

   static void               SetCharValue(char **carr, const char *v, int len = 0);
};
#endif
