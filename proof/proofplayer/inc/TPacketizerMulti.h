// $Id$
// Author: G. Ganis  Jan 2010

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPacketizerMulti
#define ROOT_TPacketizerMulti

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerMulti                                                     //
//                                                                      //
// This class allows to do multiple runs in the same query; each run    //
// can be a, for example, different dataset or the same dataset with    //
// entry list.                                                          //
// The multiple packetizer conatins a list of packetizers which are     //
// processed in turn.                                                   //
// The bit TSelector::kNewRun is set in the TSelector object when a new //
// packetizer is used.                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPacketizer
#include "TVirtualPacketizer.h"
#endif


class TIter;
class TList;
class TMap;
class TMessage;
class TProofProgressStatus;
class TSlave;

class TPacketizerMulti : public TVirtualPacketizer {

private:
   TList              *fPacketizers;     // Packetizers to be processed
   TIter              *fPacketizersIter; // Iterator on fPacketizers
   TVirtualPacketizer *fCurrent;         // Packetizer being currently processed
   TMap               *fAssignedPack;    // Map {worker,packetizer} of lat assignement

   TPacketizerMulti();
   TPacketizerMulti(const TPacketizerMulti&);     // no implementation, will generate
   void operator=(const TPacketizerMulti&);  // error on accidental usage

   TVirtualPacketizer *CreatePacketizer(TDSet *dset, TList *wrks, Long64_t first, Long64_t num,
                                        TList *input, TProofProgressStatus *st);

public:
   TPacketizerMulti(TDSet *dset, TList *slaves, Long64_t first, Long64_t num,
                    TList *input, TProofProgressStatus *st);
   virtual ~TPacketizerMulti();

   TDSetElement *GetNextPacket(TSlave *wrk, TMessage *r);

   Int_t    GetEstEntriesProcessed(Float_t f, Long64_t &ent, Long64_t &bytes, Long64_t &calls)
                    { if (fCurrent) return fCurrent->GetEstEntriesProcessed(f,ent,bytes,calls);
                      return 1; }
   Float_t  GetCurrentRate(Bool_t &all) { all = kTRUE;
                                          return (fCurrent? fCurrent->GetCurrentRate(all) : 0.); }
   void     StopProcess(Bool_t abort, Bool_t stoptimer = kFALSE) {
                                        if (fCurrent) fCurrent->StopProcess(abort, stoptimer);
                                        TVirtualPacketizer::StopProcess(abort, stoptimer); }
   void     MarkBad(TSlave *wrk, TProofProgressStatus *st, TList **missing)
                    { if (fCurrent) fCurrent->MarkBad(wrk, st, missing); return; }
   Int_t    AddProcessed(TSlave *wrk, TProofProgressStatus *st, Double_t lat, TList **missing)
                    { if (fCurrent) return fCurrent->AddProcessed(wrk, st, lat, missing);
                      return -1; }

   Int_t    GetActiveWorkers() { if (fCurrent) return fCurrent->GetActiveWorkers(); return 0; }

   ClassDef(TPacketizerMulti,0)  //Generate work packets for parallel processing
};

#endif
