// @(#)root/proofd:$Id$
// Author: G. Ganis  June 2007

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_XrdROOT
#define ROOT_XrdROOT

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdProofdProtocol                                                    //
//                                                                      //
// Authors: G. Ganis, CERN, 2007                                        //
//                                                                      //
// Class describing a ROOT version                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <list>

#include "Xrd/XrdProtocol.hh"
#include "XProofProtocol.h"
#include "XrdOuc/XrdOucString.hh"

#include "XrdProofdConfig.h"

class XrdProofdManager;
class XrdSysLogger;

class XrdROOT {
friend class XrdROOTMgr;
private:
   int          fStatus;
   XrdOucString fDir;
   XrdOucString fBinDir;
   XrdOucString fDataDir;
   XrdOucString fIncDir;
   XrdOucString fLibDir;
   XrdOucString fTag;
   XrdOucString fExport;
   XrdOucString fPrgmSrv;
   kXR_int16    fSrvProtVers;

   XrdOucString fRelease;
   XrdOucString fGitCommit;
   int          fVersionCode;

   int          fVrsMajor;
   int          fVrsMinor;
   int          fVrsPatch;

   int          CheckDir(const char *dir);
   int          ParseROOTVersionInfo();

public:
   XrdROOT(const char *dir, const char *tag, const char *bindir = 0,
           const char *incdir = 0, const char *libdir = 0, const char *datadir = 0);
   ~XrdROOT() { }

   const char *Dir() const { return fDir.c_str(); }
   const char *BinDir() const { return fBinDir.c_str(); }
   const char *DataDir() const { return fDataDir.c_str(); }
   const char *IncDir() const { return fIncDir.c_str(); }
   const char *LibDir() const { return fLibDir.c_str(); }
   const char *Export() const { return fExport.c_str(); }
   const char *GitCommit() const { return fGitCommit.c_str(); }
   bool        IsParked() const { return ((fStatus == 2) ? 1: 0); }
   bool        IsValid() const { return ((fStatus == 1) ? 1: 0); }
   bool        IsInvalid() const { return ((fStatus == -1) ? 1: 0); }
   bool        Match(const char *dir, const char *tag)
                          { return ((fTag == tag && fDir == dir) ? 1 : 0); }
   bool        MatchTag(const char *tag) { return ((fTag == tag) ? 1 : 0); }
   void        Park() { fStatus = 2; }
   const char *PrgmSrv() const { return fPrgmSrv.c_str(); }
   const char *Release() const { return fRelease.c_str(); }
   void        SetValid(kXR_int16 vers = -1);
   kXR_int16   SrvProtVers() const { return fSrvProtVers; }
   const char *Tag() const { return fTag.c_str(); }
   int         VersionCode() const { return fVersionCode; }
   int         VrsMajor() const { return fVrsMajor; }
   int         VrsMinor() const { return fVrsMinor; }
   int         VrsPatch() const { return fVrsPatch; }

   static int  GetVersionCode(const char *release);
   static int  GetVersionCode(int maj, int min, int patch);
   static int  ParseReleaseString(const char *release, int &maj, int &min, int &patch);
};

//
// Manage XrdROOT instances

class XrdROOTMgr : public XrdProofdConfig {

   XrdProofdManager  *fMgr;
   XrdSysLogger      *fLogger;    // Error logger
   XrdOucString      fLogDir;     // Path to dir with validation logs

   std::list<XrdROOT *> fROOT;    // ROOT versions; the first is the default

   int               Validate(XrdROOT *r, XrdScheduler *sched);

   void              RegisterDirectives();
   int               DoDirectiveRootSys(char *, XrdOucStream *, bool);

public:
   XrdROOTMgr(XrdProofdManager *mgr, XrdProtocol_Config *pi, XrdSysError *e);
   virtual ~XrdROOTMgr() { }

   int               Config(bool rcf = 0);
   int               DoDirective(XrdProofdDirective *d,
                                 char *val, XrdOucStream *cfg, bool rcf);

   XrdROOT          *DefaultVersion() const { return fROOT.front(); }
   XrdOucString      ExportVersions(XrdROOT *def);
   XrdROOT          *GetVersion(const char *tag);
   void              SetLogDir(const char *d);
};


#endif
