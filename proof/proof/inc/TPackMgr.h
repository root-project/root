// @(#)root/proof:$Id$
// Author: G. Ganis, Oct 2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPackMgr
#define ROOT_TPackMgr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPackMgr                                                            //
//                                                                      //
// The PROOF manager interacts with the PROOF server coordinator to     //
// create or destroy a PROOF session, attach to or detach from          //
// existing one, and to monitor any client activity on the cluster.     //
// At most one manager instance per server is allowed.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLockPath.h"
#include "TNamed.h"
#include "TMD5.h"
#include "TList.h"
#include "TString.h"

typedef void (*TPackMgrLog_t)(const char *);

class TList;
class THashList;
class TPackMgr : public TNamed {
public:
   enum ECheckVersionOpt { kDontCheck   = 0, kCheckROOT    = 1, kCheckGIT    = 2};

private:
   TPackMgrLog_t     fLogger;    // Logger
   TString           fName;      // Key identifying this package manager
   TString           fDir;       // Directory with packages
   TLockPath         fLock;      // Locker
   TString           fPfx;       // Prefix for notifications, if any
   TList            *fEnabledPackages; // List of packages enabled

   static THashList *fgGlobalPackMgrList; // list of package managers for global packages

private:
   TPackMgr(const TPackMgr&); // Not implemented
   TPackMgr& operator=(const TPackMgr&); // Not implemented

   void              Log(const char *msg);

public:
   TPackMgr(const char *dir, const char *key = "L0");
   virtual ~TPackMgr();

   const char       *GetName() const { return fName.Data(); }
   void              SetName(const char *name) { fName = name; }
   const char       *GetTitle() const { return GetDir(); }

   TLockPath        *GetLock() { return &fLock; }

   void              SetLogger(TPackMgrLog_t logger) { fLogger = logger; }
   void              SetPrefix(const char *pfx) { fPfx = pfx; }

   Int_t             Build(const char *pack, Int_t opt = TPackMgr::kCheckROOT);
   Int_t             Load(const char *pack, TList *optls = 0);
   Int_t             Load(const char *pack, const char *opts);
   Int_t             Unload(const char *pack);

   Bool_t            Has(const char *pack);
   Bool_t            IsInDir(const char *path);
   const char       *GetDir() const { return fDir.Data(); }
   Int_t             GetPackDir(const char *pack, TString &pdir);
   Int_t             GetParPath(const char *pack, TString &path);
   Int_t             GetDownloadDir(TString &dldir);
   void              GetEnabledPackages(TString &packlist);
   Bool_t            IsPackageEnabled(const char *pack) {
                               return (fEnabledPackages &&
                                       fEnabledPackages->FindObject(pack) ? kTRUE : kFALSE); }

   void              Show(const char *title = 0);
   Int_t             Clean(const char *pack);
   Int_t             Remove(const char *pack = 0, Bool_t dolock = kTRUE);
   TList            *GetList() const;

   void              ShowEnabled(const char *title = 0);
   TList*            GetListOfEnabled() const;

   TMD5             *GetMD5(const char *pack);
   TMD5             *ReadMD5(const char *pack);

   Int_t             Install(const char *par, Bool_t rmold = kFALSE);
   Int_t             Unpack(const char *pack, TMD5 *sum = 0);

   // Static methods
   static TPackMgr  *GetPackMgr(const char *pack, TPackMgr *packmgr = nullptr);
   static Int_t      RegisterGlobalPath(const char *paths);
   static Int_t      FindParPath(TPackMgr *packmgr, const char *pack, TString &par);
   static Bool_t     IsEnabled(const char *pack, TPackMgr *packmgr = nullptr);

   ClassDef(TPackMgr,0)  // Package manager interface
};

#endif
