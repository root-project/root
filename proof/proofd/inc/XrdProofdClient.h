// @(#)root/proofd:$Id$
// Author: G. Ganis June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdClient
#define ROOT_XrdProofdClient

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdClient                                                      //
//                                                                      //
// Author: G. Ganis, CERN, 2007                                         //
//                                                                      //
// Auxiliary class describing a PROOF client.                           //
// Used by XrdProofdProtocol.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>
#include <vector>

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucPthread.hh"
#else
#  include "XrdSys/XrdSysPthread.hh"
#endif
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdAux.h"
#include "XrdProofdProtocol.h"
#include "XrdProofdResponse.h"
#include "XrdProofdSandbox.h"

#define XPC_DEFMAXOLDLOGS 10

class XrdNet;
class XrdClientID;
class XrdProofdLauncher;
class XrdROOT;

class XrdProofdClient {

 public:
   XrdProofdClient(XrdProofUI ui,
                   bool master, bool changeown, XrdSysError *edest, const char *tmp);

   virtual ~XrdProofdClient();

   inline const char      *Group() const { return fUI.fGroup.c_str(); }
   inline const char      *User() const { return fUI.fUser.c_str(); }
   inline bool             IsValid() const { return fIsValid; }
   bool                    Match(const char *usr, const char *grp = 0);
   inline XrdSysRecMutex  *Mutex() const { return (XrdSysRecMutex *)&fMutex; }
   inline XrdROOT         *ROOT() const { return fROOT; }
   inline XrdProofdSandbox *Sandbox() const { return (XrdProofdSandbox *)&fSandbox; }
   inline XrdProofUI       UI() const { return fUI; }

   XrdProofdProofServ     *GetServer(int psid);
   XrdProofdProofServ     *GetServer(XrdProofdProtocol *p);
   void                    EraseServer(int psid);
   int                     GetTopServers();

   int                     ResetClientSlot(int ic);
   XrdProofdProtocol      *GetProtocol(int ic);

   int                     GetClientID(XrdProofdProtocol *p);
   int                     ReserveClientID(int cid);
   int                     SetClientID(int cid, XrdProofdProtocol *p);
   XrdProofdProofServ     *GetFreeServObj();
   XrdProofdProofServ     *GetServObj(int id);

   void                    Broadcast(const char *msg);

   XrdOucString            ExportSessions(XrdOucString &emsg, XrdProofdResponse *r = 0);
   void                    SkipSessionsCheck(std::list<XrdProofdProofServ *> *active,
                                             XrdOucString &emsg, XrdProofdResponse *r = 0);
   void                    TerminateSessions(int srvtype, XrdProofdProofServ *ref,
                                             const char *msg, XrdProofdPipe *pipe, bool changeown);
   bool                    VerifySession(XrdProofdProofServ *xps, XrdProofdResponse *r = 0);

   void                    ResetSessions();

   void                    SetGroup(const char *g) { fUI.fGroup = g; }
   void                    SetROOT(XrdROOT *r) { fROOT = r; }

   XrdProofdLauncher      *Launcher() { return fLauncher; }

   void                    SetValid(bool valid = 1) { fIsValid = valid; }

   int                     Size() const { return fClients.size(); }

   int                     Touch(bool reset = 0);

   int                     TrimSessionDirs() { return fSandbox.TrimSessionDirs(); }

   const char             *AdminPath() const { return fAdminPath.c_str(); }

 private:

   XrdSysRecMutex          fMutex; // Local mutex

   bool                    fChangeOwn; // TRUE if ownership must be changed where relevant
   bool                    fIsValid; // TRUE if the instance is complete
   bool                    fAskedToTouch; // TRUE if a touch request has already been sent for this client

   XrdProofUI              fUI;         // user info
   XrdROOT                *fROOT;        // ROOT vers instance to be used for proofserv

   XrdProofdLauncher      *fLauncher; // Session creator configured for this client

   XrdProofdSandbox        fSandbox;     // Clients sandbox

   XrdOucString            fAdminPath;    // Admin path for this client

   std::vector<XrdProofdProofServ *> fProofServs; // Allocated ProofServ sessions
   std::vector<XrdClientID *> fClients;    // Attached Client sessions
};

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
   XrdProofdResponse *fR;
   unsigned short     fSid;

   void               SetR() { fR = (fP && fSid > 0) ? fP->Response(fSid) : 0;}
public:
   XrdClientID(XrdProofdProtocol *pt = 0, unsigned short id = 0)
            { fP = pt; fSid = id; SetR();}
   ~XrdClientID() { }

   XrdProofdClient   *C() const { return fP->Client(); }
   bool               IsValid() const { return (fP != 0); }
   XrdProofdProtocol *P() const { return fP; }
   XrdProofdResponse *R() const { return fR; }
   void               Reset() { fP = 0; fSid = 0; SetR(); }
   void               SetP(XrdProofdProtocol *p) { fP = p; SetR();}
   void               SetSid(unsigned short sid) { fSid = sid; SetR();}
   unsigned short     Sid() const { return fSid; }
};

#endif
