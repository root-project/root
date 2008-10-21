// @(#)root/proof:$Id$
// Author: G. Ganis Mar 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TQueryResultManager
#define ROOT_TQueryResultManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQueryResultManager                                                  //
//                                                                      //
// This class manages the query-result area.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TList;
class TProof;
class TProofLockPath;
class TProofQueryResult;
class TQueryResult;
class TVirtualProofPlayer;

class TQueryResultManager : public TObject {

private:
   TString       fQueryDir;         //directory containing query results and status
   TString       fSessionTag;       //tag for the session
   TString       fSessionDir;       //directory containing session dependent files
   Int_t         fSeqNum;           //sequential number of last processed query
   Int_t         fDrawQueries;      //number of draw queries processed
   Int_t         fKeptQueries;      //number of queries fully in memory and in dir
   TList        *fQueries;          //list of TProofQueryResult objects
   TList        *fPreviousQueries;  //list of TProofQueryResult objects from previous sections
   TProofLockPath *fLock;           //dir locker
   FILE         *fLogFile;          //log file
   TStopwatch    fCompute;          //measures time spend processing a query on the master

   void          AddLogFile(TProofQueryResult *pq);

public:
   TQueryResultManager(const char *qdir, const char *stag, const char *sdir,
                       TProofLockPath *lck, FILE *logfile = 0);
   virtual ~TQueryResultManager();

   const char   *QueryDir() const { return fQueryDir.Data(); }
   Int_t         SeqNum() const { return fSeqNum; }
   Int_t         DrawQueries() const { return fDrawQueries; }
   Int_t         KeptQueries() const { return fKeptQueries; }
   TList        *Queries() const { return fQueries; }
   TList        *PreviousQueries() const { return fPreviousQueries; }

   void          IncrementSeqNum() { fSeqNum++; }
   void          IncrementDrawQueries() { fDrawQueries++; }

   Int_t         ApplyMaxQueries(Int_t mxq);
   Int_t         CleanupQueriesDir();
   Bool_t        FinalizeQuery(TProofQueryResult *pq,
                               TProof *proof, TVirtualProofPlayer *player);
   Float_t       GetCpuTime() { return fCompute.CpuTime(); }
   Float_t       GetRealTime() { return fCompute.RealTime(); }
   TProofQueryResult *LocateQuery(TString queryref, Int_t &qry, TString &qdir);
   void          RemoveQuery(TQueryResult *qr, Bool_t soft = kFALSE);
   void          RemoveQuery(const char *queryref, TList *otherlist = 0);
   void          ResetTime() { fCompute.Start(); }
   void          SaveQuery(TProofQueryResult *qr, const char *fout = 0);
   void          SaveQuery(TProofQueryResult *qr, Int_t mxq);

   Int_t         LockSession(const char *sessiontag, TProofLockPath **lck);
   Int_t         CleanupSession(const char *sessiontag);
   void          ScanPreviousQueries(const char *dir);

   ClassDef(TQueryResultManager,0)  //PROOF query result manager
};

#endif

