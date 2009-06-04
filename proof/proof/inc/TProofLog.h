// @(#)root/proof:$Id$
// Author: G. Ganis   31/08/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofLog
#define ROOT_TProofLog

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofLog                                                            //
//                                                                      //
// Implementation of the PROOF session log handler                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TQObject
#include "TQObject.h"
#endif

class TMacro;
class TProofLogElem;
class TProofMgr;


class TProofLog : public TNamed, public TQObject {

friend class TProofLogElem;
friend class TProofMgrLite;
friend class TXProofMgr;

private:
   TProofMgr  *fMgr;   // parent TProofMgr
   void       *fFILE;  // pointer logging file, if any
   TList      *fElem;  // list of TProofLogElem objects
   TDatime     fStartTime; // Time at which this session started

   TProofLogElem *Add(const char *ord, const char *url);

public:
   // Screen or GUI box logging
   enum ELogLocationBit {
      kLogToBox = BIT(16)
   };
   enum ERetrieveOpt   { kLeading = 0x1, kTrailing = 0x2,
                         kAll = 0x3, kGrep = 0x4 };

   TProofLog(const char *stag, const char *url, TProofMgr *mgr);
   virtual ~TProofLog();

   void   Display(const char *ord = "*", Int_t from = -10, Int_t to = -1);
   TList *GetListOfLogs() const { return fElem; }
   Int_t  Grep(const char *txt, Int_t from = 0);
   void   Print(Option_t *opt = 0) const;
   void   Prt(const char *what);
   Int_t  Retrieve(const char *ord = "*",
                  TProofLog::ERetrieveOpt opt = TProofLog::kTrailing,
                  const char *fname = 0, const char *pattern = 0);
   Int_t  Save(const char *ord = "*", const char *fname = 0, Option_t *opt="w");

   TDatime StartTime() { return fStartTime; }

   // Where to log
   void SetLogToBox(Bool_t lgbox = kFALSE) { SetBit(kLogToBox, lgbox); }
   Bool_t LogToBox() { return (TestBit(kLogToBox)) ? kTRUE : kFALSE; }

   static void SetMaxTransferSize(Long64_t maxsz);

   ClassDef(TProofLog,0)  // PROOF session log handler
};


class TProofLogElem : public TNamed {

private:
   TProofLog *fLogger;  // parent TProofLog
   TMacro    *fMacro;   // container for the log lines
   Long64_t   fSize;    // best knowledge of the log file size
   Long64_t   fFrom;    // starting offset of the current content
   Long64_t   fTo;      // end offset of the current content
   TString    fRole;    // role (master-submaster-worker)

   static Long64_t fgMaxTransferSize;

   //the name of TProofLogElem is the ordinal number of the corresp. worker
   //the title is the url

public:
   TProofLogElem(const char *ord, const char *url,
                 TProofLog *logger);
   virtual ~TProofLogElem();

   void    Display(Int_t from = 0, Int_t to = -1);
   TMacro *GetMacro() const { return fMacro; }
   const char *    GetRole() { return fRole.Data(); }
   Int_t   Grep(const char *txt, TString &res, Int_t from = 0);
   Bool_t  IsMaster() const { return (fRole == "master") ? kTRUE : kFALSE; }
   Bool_t  IsSubMaster() const { return (fRole == "submaster") ? kTRUE : kFALSE; }
   Bool_t  IsWorker() const { return (fRole == "worker") ? kTRUE : kFALSE; }
   void    Print(Option_t *opt = 0) const;
   void    Prt(const char *what);
   Int_t   Retrieve(TProofLog::ERetrieveOpt opt = TProofLog::kTrailing,
                    const char *pattern = 0);

   static Long64_t GetMaxTransferSize();
   static void     SetMaxTransferSize(Long64_t maxsz);

   ClassDef(TProofLogElem,0)  // PROOF session log element
};

#endif
