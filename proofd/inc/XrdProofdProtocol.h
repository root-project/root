// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdProofdProtocol
#define ROOT_XrdProofdProtocol

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2005                                        //
//                                                                      //
// XrdProtocol implementation to coordinate 'proofserv' applications.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef OLDXRDOUC
#  include "XrdSysToOuc.h"
#  include "XrdOuc/XrdOucError.hh"
#  include "XrdOuc/XrdOucPthread.hh"
#  include "XrdOuc/XrdOucSemWait.hh"
#else
#  include "XrdSys/XrdSysError.hh"
#  include "XrdSys/XrdSysPthread.hh"
#  include "XrdSys/XrdSysSemWait.hh"
#endif

#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdSec/XrdSecInterface.hh"

#include "Xrd/XrdProtocol.hh"
#include "Xrd/XrdObject.hh"

#include "XProofProtocol.h"
#include "XrdProofdManager.h"
#include "XrdProofdResponse.h"
#include "XrdProofGroup.h"
#include "XrdProofdAux.h"

#include <list>
#include <vector>

// Version index: start from 1001 (0x3E9) to distinguish from 'proofd'
// To be increment when non-backward compatible changes are introduced
//  1001 (0x3E9) -> 1002 (0x3EA) : support for flexible env setting
//  1002 (0x3EA) -> 1003 (0x3EB) : many new features
//  1003 (0x3EB) -> 1004 (0x3EC) : restructuring
#define XPROOFD_VERSBIN 0x000003EC
#define XPROOFD_VERSION "0.4"

#define XPD_LOGGEDIN       1
#define XPD_NEED_AUTH      2
#define XPD_ADMINUSER      4
#define XPD_NEED_MAP       8

class XrdBuffer;
class XrdClientMessage;
class XrdLink;
class XrdSysError;
class XrdOucTrace;
class XrdProofdClient;
class XrdProofdPInfo;
class XrdProofdPriority;
class XrdProofServProxy;
class XrdProofWorker;
class XrdScheduler;
class XrdSrvBuffer;

class XrdProofdProtocol : XrdProtocol {

friend class XrdProofdClient;
friend class XrdROOT;

public:
   XrdProofdProtocol();
   virtual ~XrdProofdProtocol() {} // Never gets destroyed

   static int    Configure(char *parms, XrdProtocol_Config *pi);
   void          DoIt() {}
   XrdProtocol  *Match(XrdLink *lp);
   int           Process(XrdLink *lp);
   void          Recycle(XrdLink *lp, int x, const char *y);
   int           Stats(char *buff, int blen, int do_sync);

   const char   *GetID() const { return (const char *)fClientID; }
   const char   *GetGroupID() const { return (const char *)fGroupID; }
   XrdProofdClient *Client() const { return fPClient; }

   static void   Reconfig();
   static int    ProcessDirective(XrdProofdDirective *d,
                                  char *val, XrdOucStream *cfg, bool rcf);

   static XrdProofdManager *Mgr() { return &fgMgr; }

 private:

   int           Admin();
   int           Attach();
   int           Auth();
   int           Create();
   int           Destroy();
   int           Detach();
   int           GetBuff(int quantum);
   int           GetData(const char *dtype, char *buff, int blen);
   XrdProofServProxy *GetServer(int psid);
   int           Interrupt();
   int           Login();
   int           MapClient(bool all = 1);
   int           Ping();
   int           Process2();
   int           ReadBuffer();
   char         *ReadBufferLocal(const char *file, kXR_int64 ofs, int &len);
   char         *ReadBufferLocal(const char *file, const char *pat, int &len, int opt);
   char         *ReadBufferRemote(const char *url, const char *file,
                                  kXR_int64 ofs, int &len, int grep);
   char         *ReadLogPaths(const char *url, const char *stag, int isess);
   void          Reset();
   int           SendData(XrdProofdResponse *resp, kXR_int32 sid = -1, XrdSrvBuffer **buf = 0);
   int           SendDataN(XrdProofServProxy *xps, XrdSrvBuffer **buf = 0);
   int           SendMsg();
   int           SetUserEnvironment();
   int           Urgent();

   int           CleanupProofServ(bool all = 0, const char *usr = 0);
   int           KillProofServ(int pid, bool forcekill = 0);
   int           SetProofServEnv(int psid = -1, int loglevel = -1, const char *cfg = 0);
   int           SetProofServEnvOld(int psid = -1, int loglevel = -1, const char *cfg = 0);
   //
   // Local area
   //
   XrdObject<XrdProofdProtocol>  fProtLink;
   XrdLink                      *fLink;
   XrdBuffer                    *fArgp;
   char                          fStatus;
   char                         *fClientID;    // login username
   char                         *fGroupID;     // login group name
   XrdProofUI                    fUI;           // user info
   unsigned char                 fCapVer;
   kXR_int32                     fSrvType;      // Master or Worker
   bool                          fTopClient;    // External client (not ProofServ)
   bool                          fSuperUser;    // TRUE for privileged clients (admins)
   //
   XrdProofdClient              *fPClient;    // Our reference XrdProofdClient
   kXR_int32                     fCID;        // Reference ID of this client
   //
   XrdSecEntity                 *fClient;
   XrdSecProtocol               *fAuthProt;
   XrdSecEntity                  fEntity;
   //
   char                         *fBuff;
   int                           fBlen;
   int                           fBlast;
   //
   int                           fhcPrev;
   int                           fhcMax;
   int                           fhcNext;
   int                           fhcNow;
   int                           fhalfBSize;
   //
   XPClientRequest               fRequest;  // handle client requests
   XrdProofdResponse             fResponse; // Response to incoming request
   XrdSysRecMutex                fMutex;    // Local mutex

   //
   // Static area: general protocol managing section
   //
   static XrdSysRecMutex         fgXPDMutex;  // Mutex for static area
   static int                    fgCount;
   static XrdObjectQ<XrdProofdProtocol> fgProtStack;
   static XrdBuffManager        *fgBPool;     // Buffer manager
   static int                    fgMaxBuffsz;    // Maximum buffer size we can have

   static XrdScheduler          *fgSched;     // System scheduler
   static XrdSysError            fgEDest;     // Error message handler
   static XrdSysLogger           fgMainLogger; // Error logger

   //
   // Static area: protocol configuration section
   //
   static XrdProofdFile          fgCfgFile;    // Main config file
   static bool                   fgConfigDone; // Whether configure has been run
   //
   static XrdSysSemWait          fgForkSem;   // To serialize fork requests
   //
   static int                    fgReadWait;
   static int                    fgInternalWait; // Timeout on replies from proofsrv
   //
   static XrdOucString           fgProofServEnvs; // Additional envs to be exported before proofserv
   static XrdOucString           fgProofServRCs; // Additional rcs to be passed to proofserv
   //
   static XrdProofdManager       fgMgr;       // Cluster manager

   static XrdOucHash<XrdProofdDirective> fgConfigDirectives; // Config directives
   static XrdOucHash<XrdProofdDirective> fgReConfigDirectives; // Re-configurable directives

   //
   // Static area: methods
   //
   static int    SetProofServEnv(XrdROOT *r);
   static int    SaveAFSkey(XrdSecCredentials *c, const char *fn);

   static int    Config(const char *fn, bool rcf = 0);
   static int    TraceConfig(const char *cfn);

   static void   RegisterConfigDirectives();
   static void   RegisterReConfigDirectives();
   static int    DoDirectivePutEnv(char *, XrdOucStream *, bool);
   static int    DoDirectivePutRc(char *, XrdOucStream *, bool);
};

#endif
