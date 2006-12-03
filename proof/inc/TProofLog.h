// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.97 2006/11/28 12:10:52 rdm Exp $
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

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TMacro;
class TProofLogElem;
class TProofMgr;


class TProofLog : public TNamed {

friend class TProofLogElem;
friend class TXProofMgr;

private:
   TProofMgr  *fMgr;   // parent TProofMgr
   void       *fFILE;  // pointer logging file, if any
   TList      *fElem;  // list of TProofLogElem objects

   TProofLogElem *Add(const char *ord, const char *url);

public:
   // Screen or GUI box logging
   enum ELogLocationBit {
      kLogToBox = BIT(16)
   };
   enum ERetrieveOpt   { kLeading = 0x1, kTrailing = 0x2, kAll = 0x3 };

   TProofLog(const char *stag, const char *url, TProofMgr *mgr);
   virtual ~TProofLog();

   void Display(const char *ord = "*", Int_t from = -10, Int_t to = -1);
   void Print(Option_t *opt = 0) const;
   void Prt(const char *what);
   Int_t Retrieve(const char *ord = "*",
                  TProofLog::ERetrieveOpt opt = TProofLog::kTrailing,
                  const char *fname = 0);
   Int_t Save(const char *ord = "*", const char *fname = 0);
   Int_t Grep(const char *txt, Int_t from = 0);

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

   static Long64_t fgMaxTransferSize;

public:
   TProofLogElem(const char *ord, const char *url,
                 TProofLog *logger);
   virtual ~TProofLogElem();

   void Display(Int_t from = 0, Int_t to = -1);
   void Print(Option_t *opt = 0) const;
   void Prt(const char *what);
   Int_t Retrieve(TProofLog::ERetrieveOpt opt = TProofLog::kTrailing);
   Int_t Grep(const char *txt, TString &res, Int_t from = 0);

   static Long64_t GetMaxTransferSize();
   static void     SetMaxTransferSize(Long64_t maxsz);

   ClassDef(TProofLogElem,0)  // PROOF session log element
};

#endif
