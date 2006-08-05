// @(#)root/proof:$Name:  $:$Id: TProofQueryResult.h,v 1.1 2005/09/16 08:48:38 rdm Exp $
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

#ifndef ROOT_TQueryResult
#include "TQueryResult.h"
#endif

class TProofServ;


class TProofQueryResult : public TQueryResult {

friend class TProofServ;

private:
   Int_t    fStartLog;     //log file offset at start

   TProofQueryResult(Int_t seqnum, const char *opt, TList *inlist,
                     Long64_t entries, Long64_t first, TDSet *dset,
                     const char *selec, TEventList *elist = 0);

   void  SetFinalized() { TQueryResult::SetFinalized(); }
   void  SetRunning(Int_t startlog, const char *par);

public:
   TProofQueryResult() : TQueryResult(), fStartLog(-1) { }
   virtual ~TProofQueryResult() { }

   ClassDef(TProofQueryResult,1)  //Class describing a PROOF query
};

#endif
