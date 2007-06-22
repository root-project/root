// @(#)root/proofd:$Name:  $:$Id: XrdProofdClient.h,v 1.2 2007/06/21 07:41:05 ganis Exp $
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

#include "XrdOuc/XrdOucPthread.hh"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdAux.h"

#define XPC_DEFMAXOLDLOGS 10

class XrdNet;
class XrdProofdProtocol;
class XrdProofGroup;
class XrdProofServProxy;
class XrdROOT;

class XrdProofdClient {

 public:
   XrdProofdClient(const char *cid,
                   short int clientvers, XrdProofUI ui);

   virtual ~XrdProofdClient();

   inline int              MasterProofServ() const { return fMasterProofServ; }
   inline int              WorkerProofServ() const { return fWorkerProofServ; }
   void                    CountSession(int n = 1, bool worker =1);
   inline XrdProofGroup   *Group() const { return fGroup; }
   inline const char      *ID() const
                              { return (const char *)fClientID; }
   inline bool             IsValid() const { return fIsValid; }
   bool                    Match(const char *id, const char *grp = 0);
   inline XrdOucRecMutex  *Mutex() const { return (XrdOucRecMutex *)&fMutex; }
   inline unsigned short   RefSid() const { return fRefSid; }
   inline XrdROOT         *ROOT() const { return fROOT; }
   inline short            Version() const { return fClientVers; }
   inline const char      *Workdir() const { return fUI.fWorkDir.c_str(); }
   inline XrdProofUI       UI() const { return fUI; }
   inline std::vector<XrdProofServProxy *> *ProofServs()
                           { return (std::vector<XrdProofServProxy *> *)&fProofServs; }
   inline std::vector<XrdProofdProtocol *> *Clients()
                           { return (std::vector<XrdProofdProtocol *> *)&fClients; }
   void                    ResetClient(int i) { fClients[i] = 0; }

   void                    EraseServer(int psid);
   int                     GetClientID(XrdProofdProtocol *p);
   int                     GetFreeServID();

   void                    SetClientVers(short int cv) { fClientVers = cv; }

   void                    SetGroup(XrdProofGroup *g) { fGroup = g; }
   void                    SetROOT(XrdROOT *r) { fROOT = r; }

   void                    SetRefSid(unsigned short sid) { fRefSid = sid; }
   void                    SetValid(bool valid = 1) { fIsValid = valid; }
   void                    SetWorkdir(const char *wrk) { fUI.fWorkDir = wrk; }

   int                     CreateUNIXSock(XrdOucError *edest, char *tmpdir);
   XrdNet                 *UNIXSock() const { return fUNIXSock; }
   char                   *UNIXSockPath() const { return fUNIXSockPath; }
   void                    SaveUNIXPath(); // Save path in the sandbox
   void                    SetUNIXSockSaved() { fUNIXSockSaved = 1;}

   int                     AddNewSession(const char *tag);
   int                     GetSessionDirs(int opt, std::list<XrdOucString *> *sdirs,
                                          XrdOucString *tag = 0);
   int                     GuessTag(XrdOucString &tag, int ridx = 1);
   int                     MvOldSession(const char *tag);

   static void             SetMaxOldLogs(int mx) { fgMaxOldLogs = mx; }

 private:

   XrdOucRecMutex          fMutex; // Local mutex

   bool                    fIsValid; // TRUE if the instance is complete

   char                   *fClientID;   // String identifying this client
   short int               fClientVers; // PROOF version run by client
   unsigned short          fRefSid;     // Reference stream ID for this client
   XrdProofUI              fUI;         // user info

   XrdNet                 *fUNIXSock;     // UNIX server socket for internal connections
   char                   *fUNIXSockPath; // UNIX server socket path
   bool                    fUNIXSockSaved; // TRUE if the socket path has been saved

   XrdROOT                *fROOT;        // ROOT vers instance to be used for proofserv

   XrdProofGroup          *fGroup;       // Group of the client, if any

   std::vector<XrdProofServProxy *> fProofServs; // Allocated ProofServ sessions
   std::vector<XrdProofdProtocol *> fClients;    // Attached Client sessions

   int                     fWorkerProofServ; // Number of active (non idle) ProofServ worker sessions
   int                     fMasterProofServ; // Number of active (non idle) ProofServ master sessions

   static int              fgMaxOldLogs; // max number of old sessions workdirs per client
};

#endif
