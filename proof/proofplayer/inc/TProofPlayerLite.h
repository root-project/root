// @(#)root/proofplayer:$Id$
// Author: G. Ganis Mar 2008

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofPlayerLite
#define ROOT_TProofPlayerLite


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayerLite                                                     //
//                                                                      //
// This version of TProofPlayerRemote merges the functionality needed   //
// by clients and masters. It is used in optmized local sessions.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofPlayer
#include "TProofPlayer.h"
#endif


class TProofPlayerLite : public TProofPlayerRemote {

protected:
   Bool_t  HandleTimer(TTimer *timer);

   Int_t   MakeSelector(const char *selfile);
   void    SetupFeedback();

public:
   TProofPlayerLite(TProof *proof = 0) : TProofPlayerRemote(proof) { }

   virtual ~TProofPlayerLite() { }   // Owns the fOutput list

   Long64_t       Process(TDSet *set, const char *selector,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0);
   Long64_t       Process(TDSet *set, TSelector *selector,
                          Option_t *option = "", Long64_t nentries = -1,
                          Long64_t firstentry = 0);
   Long64_t       Finalize(Bool_t force = kFALSE, Bool_t sync = kFALSE);
   Long64_t       Finalize(TQueryResult *qr)
                            { return TProofPlayerRemote::Finalize(qr); }

   void           StoreFeedback(TObject *slave, TList *out); // Adopts the list

   ClassDef(TProofPlayerLite,0)  // PROOF player running in PROOF-Lite
};

#endif
