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

#include "XProofProtocol.h"
#include "XrdProofdClient.h"

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
// XrdProofdProofServ                                                   //
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
class XrdSysSemWait;

class XrdProofdProofServ
{

public:
   XrdProofdProofServ();
   ~XrdProofdProofServ();

   void                AddWorker(XrdProofWorker *w) { XrdSysMutexHelper mhp(fMutex); fWorkers.push_back(w); }
   inline const char  *AdminPath() const { XrdSysMutexHelper mhp(fMutex); return fAdminPath.c_str(); }
   inline const char  *Alias() const { XrdSysMutexHelper mhp(fMutex); return fAlias.c_str(); }
   void                Broadcast(const char *msg, int type = kXPD_srvmsg);
   int                 BroadcastPriority(int priority);
   inline const char  *Client() const { XrdSysMutexHelper mhp(fMutex); return fClient.c_str(); }
   void                DeleteStartMsg()
                       { XrdSysMutexHelper mhp(fMutex); if (fStartMsg) delete fStartMsg; fStartMsg = 0;}
   int                 DisconnectTime();
   void                ExportBuf(XrdOucString &buf);
   inline const char  *Fileout() const { XrdSysMutexHelper mhp(fMutex); return fFileout.c_str(); }
   int                 FreeClientID(int pid);
   XrdClientID        *GetClientID(int cid);
   int                 GetNClients(bool check);
   inline const char  *Group() const { XrdSysMutexHelper mhp(fMutex); return fGroup.c_str(); }
   int                 IdleTime();
   inline short int    ID() const { XrdSysMutexHelper mhp(fMutex); return fID; }
   inline bool         IsShutdown() const { XrdSysMutexHelper mhp(fMutex); return fIsShutdown; }
   bool                IsValid();
   inline bool         Match(short int id) const { XrdSysMutexHelper mhp(fMutex); return (id == fID); }
   inline XrdSysRecMutex *Mutex() const { return fMutex; }
   inline const char  *Ordinal() const { XrdSysMutexHelper mhp(fMutex); return fOrdinal.c_str(); }
   inline XrdClientID *Parent() const { XrdSysMutexHelper mhp(fMutex); return fParent; }
   inline void         PingSem() const { XrdSysMutexHelper mhp(fMutex); if (fPingSem) fPingSem->Post(); }
   inline XrdProofdProtocol *Protocol() const { XrdSysMutexHelper mhp(fMutex); return fProtocol; }
   inline XrdSrvBuffer *QueryNum() const { XrdSysMutexHelper mhp(fMutex); return fQueryNum; }

   void                Reset();
   inline XrdROOT     *ROOT() const { XrdSysMutexHelper mhp(fMutex); return fROOT; }
   inline XrdProofdResponse *Response() const { XrdSysMutexHelper mhp(fMutex); return fResponse; }
   int                 SendData(int cid, void *buff, int len);
   int                 SendDataN(void *buff, int len);
   int                 SetAdminPath(const char *a);
   void                SetAlias(const char *a) { XrdSysMutexHelper mhp(fMutex); fAlias = a; }
   void                SetClient(const char *c) { XrdSysMutexHelper mhp(fMutex); fClient = c; }
   inline void         SetConnection(XrdProofdResponse *r) { XrdSysMutexHelper mhp(fMutex); fResponse = r;}

   void                SetFileout(const char *f) { XrdSysMutexHelper mhp(fMutex); fFileout = f; }
   inline void         SetGroup(const char *g) { XrdSysMutexHelper mhp(fMutex); fGroup = g; }
   void                SetIdle();
   inline void         SetID(short int id) { XrdSysMutexHelper mhp(fMutex); fID = id;}
   void                SetOrdinal(const char *o) { XrdSysMutexHelper mhp(fMutex); fOrdinal = o; }
   inline void         SetParent(XrdClientID *cid) { XrdSysMutexHelper mhp(fMutex); fParent = cid; }
   inline void         SetProtocol(XrdProofdProtocol *p) { XrdSysMutexHelper mhp(fMutex); fProtocol = p; }
   inline void         SetProtVer(int pv) { XrdSysMutexHelper mhp(fMutex); fProtVer = pv; }
   inline void         SetROOT(XrdROOT *r) { XrdSysMutexHelper mhp(fMutex); fROOT = r; }
   void                SetRunning();
   inline void         SetShutdown() { XrdSysMutexHelper mhp(fMutex); fIsShutdown = true; }
   inline void         SetSkipCheck() { XrdSysMutexHelper mhp(fMutex); fSkipCheck = true; }
   void                SetSrvPID(int pid) { XrdSysMutexHelper mhp(fMutex); fSrvPID = pid; }
   inline void         SetSrvType(int id) { XrdSysMutexHelper mhp(fMutex); fSrvType = id; }
   inline void         SetStartMsg(XrdSrvBuffer *sm) { XrdSysMutexHelper mhp(fMutex); fStartMsg = sm; }
   inline void         SetStatus(int st) { XrdSysMutexHelper mhp(fMutex); fStatus = st; }
   void                SetTag(const char *t) { XrdSysMutexHelper mhp(fMutex); fTag = t; }
   void                SetUNIXSockPath(const char *s) { XrdSysMutexHelper mhp(fMutex); fUNIXSockPath = s; };
   void                SetUserEnvs(const char *t) { XrdSysMutexHelper mhp(fMutex); fUserEnvs = t; }
   inline void         SetValid(bool valid = 1) { XrdSysMutexHelper mhp(fMutex); fIsValid = valid; }
   bool                SkipCheck();
   inline int          SrvPID() const { XrdSysMutexHelper mhp(fMutex); return fSrvPID; }
   inline int          SrvType() const { XrdSysMutexHelper mhp(fMutex); return fSrvType; }
   inline XrdSrvBuffer *StartMsg() const { XrdSysMutexHelper mhp(fMutex); return fStartMsg; }
   inline int          Status() const { XrdSysMutexHelper mhp(fMutex); return fStatus;}
   inline const char  *Tag() const { XrdSysMutexHelper mhp(fMutex); return fTag.c_str(); }
   int                 TerminateProofServ(bool changeown);
   inline const char  *UserEnvs() const { XrdSysMutexHelper mhp(fMutex); return fUserEnvs.c_str(); }
   int                 VerifyProofServ(bool fw);
   inline std::list<XrdProofWorker *> *Workers() const
                      { XrdSysMutexHelper mhp(fMutex); return (std::list<XrdProofWorker *> *)&fWorkers; }

   int                 CreateUNIXSock(XrdSysError *edest);
   XrdNet             *UNIXSock() const { return fUNIXSock; }
   const char         *UNIXSockPath() const { return fUNIXSockPath.c_str(); }

 private:

   XrdSysRecMutex           *fMutex;
   XrdProofdProtocol        *fProtocol;  // Protocol instance attached to this session
   XrdProofdResponse        *fResponse;  // Response instance attached to this session

   XrdClientID              *fParent;    // Parent creating this session
   int                       fNClients;   // Number of attached clients
   std::vector<XrdClientID *> fClients;  // Attached clients stream ids
   std::list<XrdProofWorker *> fWorkers; // Workers assigned to the session

   XrdSysSemWait            *fPingSem;   // To sychronize ping requests

   XrdSrvBuffer             *fQueryNum;  // Msg with sequential number of currebt query
   XrdSrvBuffer             *fStartMsg;  // Msg with start processing info

   time_t                    fDisconnectTime; // Time at which all clients disconnected
   time_t                    fSetIdleTime; // Time at which the session went idle

   int                       fStatus;
   int                       fSrvPID;     // Srv process ID
   int                       fSrvType;
   short int                 fID;
   char                      fProtVer;
   XrdOucString              fFileout;

   XrdNet                   *fUNIXSock;     // UNIX server socket for internal connections
   XrdOucString              fUNIXSockPath; // UNIX server socket path

   bool                      fIsShutdown; // Whether asked to shutdown
   bool                      fIsValid;    // Validity flag
   bool                      fSkipCheck;  // Skip next validity check

   XrdOucString              fAlias;     // Session alias
   XrdOucString              fClient;    // Client name
   XrdOucString              fTag;       // Session unique tag
   XrdOucString              fOrdinal;   // Session ordinal number
   XrdOucString              fUserEnvs;  // List of envs received from the user
   XrdOucString              fAdminPath; // Admin file in the form "<active-sessions>/<usr>.<grp>.<pid>" 

   XrdROOT                  *fROOT;      // ROOT version run by this session

   XrdOucString              fGroup;     // Group, if any, to which the owner belongs

   void                      ClearWorkers();

   void                      CreatePingSem()
                             { XrdSysMutexHelper mhp(fMutex); fPingSem = new XrdSysSemWait(0);}
   void                      DeletePingSem()
                             { XrdSysMutexHelper mhp(fMutex); if (fPingSem) delete fPingSem; fPingSem = 0;}
};
#endif
