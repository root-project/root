// @(#)root/proof:$Name:  $:$Id: TProof.h,v 1.61 2005/08/15 15:57:18 rdm Exp $
// Author: G Ganis Aug 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofQuery
#define ROOT_TProofQuery


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofQuery                                                          //
//                                                                      //
// A classes describing PROOF query                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TDatime
#include "TDatime.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TDSet;
class TProofServ;


class TProofQuery : public TObject {

   friend class TProofServ;

public:
   enum EQueryStatus { kAborted = 0, kRunning, kStopped, kCompleted, kArchived };

private:
   Int_t           fSeqNum;       //query unique sequential number
   EQueryStatus    fStatus;       //query status
   TDatime         fStart;        //time when processing started
   TDatime         fEnd;          //time when processing ended
   Int_t           fStartLog;     //logfile offset of first log
   Int_t           fEndLog;       //logfile offset of last log
   Int_t           fNFiles;       //number of files
   Long64_t        fEntries;      //number of entries processed
   Long64_t        fFirst;        //first entry processed
   Long64_t        fTotalEntries; //total number of entries in files
   TString         fSelecName;    //name of the selector
   TString         fParList;      //colon-separated list of PAR loaded at fStart
   TString         fResultFile;   //URL of the file with output list

   TProofQuery(Int_t seqnum, Int_t startlog, Long64_t entries, Long64_t first,
               TDSet *dset, const char *selec, const char *par, const char *file = 0);
   TProofQuery(const char *file) { FetchFromFile(file); }

   Int_t BackupToFile(const char *file);
   Int_t FetchFromFile(const char *file);

   void  SetArchived(const char *resfile, const char *file = 0);
   void  SetEntries(Long64_t ent) { fEntries = ent; }
   void  SetDone(EQueryStatus status, Int_t endlog,
                 const char *tmpfile = 0, const char *file = 0);

 public:
   TProofQuery() : fSeqNum(-1), fStatus(kAborted), fStartLog(-1), fEndLog(-1),
                   fEntries(-1), fFirst(-1), fNFiles(0), fTotalEntries(0),
                   fSelecName("-"), fParList("-"), fResultFile("-") { }
   virtual ~TProofQuery() { }

   Int_t        GetSeqNum() const { return fSeqNum; }
   EQueryStatus GetStatus() const { return fStatus; }
   TDatime      GetStartTime() const { return fStart; }
   TDatime      GetEndTime() const { return fEnd; }
   Int_t        GetStartLog() const { return fStartLog; }
   Int_t        GetEndLog() const { return fEndLog; }
   Long64_t     GetEntries() const { return fEntries; }
   Long64_t     GetFirst() const { return fFirst; }
   Int_t        GetNFiles() const { return fNFiles; }
   Long64_t     GetTotalEntries() const { return fTotalEntries; }
   const char  *GetSelectorName() const { return fSelecName; }
   const char  *GetParList() const { return fParList; }
   const char  *GetResultFile() const { return fResultFile; }

   Bool_t       IsAborted() const { return (fStatus == kAborted); }
   Bool_t       IsRunning() const { return (fStatus == kRunning); }
   Bool_t       IsDone() const { return (fStatus > kRunning); }
   Bool_t       IsArchived() const { return (fStatus == kArchived); }

   void         Print(Option_t *opt = "") const;

   ClassDef(TProofQuery,1) //Class describing a PROOF query
};

#endif
