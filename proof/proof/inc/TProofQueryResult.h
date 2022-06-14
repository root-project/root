// @(#)root/proof:$Id$
// Author: G Ganis Aug 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofQueryResult
#define ROOT_TProofQueryResult


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofQueryResult                                                    //
//                                                                      //
// TQueryResult version adapted to PROOF neeeds.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TQueryResult.h"

class TDSet;

class TProofQueryResult : public TQueryResult {

friend class TProofLite;
friend class TProofServ;
friend class TQueryResultManager;

private:
   Int_t    fStartLog;     //log file offset at start

   TProofQueryResult(Int_t seqnum, const char *opt, TList *inlist,
                     Long64_t entries, Long64_t first, TDSet *dset,
                     const char *selec, TObject *elist = 0);

   void  RecordEnd(EQueryStatus status, TList *outlist = 0)
         { TQueryResult::RecordEnd(status, outlist); }

   void  SetFinalized() { TQueryResult::SetFinalized(); }
   void  SetResultFile(const char *rf) { fResultFile = rf; }
   void  SetRunning(Int_t startlog, const char *par, Int_t nwrks);

public:
   TProofQueryResult() : TQueryResult(), fStartLog(-1) { }
   virtual ~TProofQueryResult() { }

   ClassDef(TProofQueryResult,1)  //Class describing a PROOF query
};

#endif
