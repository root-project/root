// @(#)root/proof:$Id$
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofSuperMaster
#define ROOT_TProofSuperMaster


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofSuperMaster                                                    //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProof.h"
#include "TString.h"

class TVirtualProofPlayer;
class TDSet;

class TProofSuperMaster : public TProof {

friend class TProofPlayerSuperMaster;

protected:
   Bool_t    StartSlaves(Bool_t);
   void      ValidateDSet(TDSet *dset);
   virtual   TVirtualProofPlayer *MakePlayer(const char *player = 0, TSocket *s = 0);

public:
   TProofSuperMaster(const char *masterurl, const char *conffile = kPROOF_ConfFile,
                     const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0,
                     const char *alias = 0, TProofMgr *mgr = 0);
   virtual ~TProofSuperMaster() { }

   Long64_t Process(TDSet *set, const char *selector,
                    Option_t *option = "", Long64_t nentries = -1,
                    Long64_t firstentry = 0);
   Long64_t Process(TFileCollection *fc, const char *sel, Option_t *o = "",
                    Long64_t nent = -1, Long64_t fst = 0)
                    { return TProof::Process(fc, sel, o, nent, fst); }
   Long64_t Process(const char *dsname, const char *sel,
                    Option_t *o = "", Long64_t nent = -1,
                    Long64_t fst = 0, TObject *enl = 0)
                    { return TProof::Process(dsname, sel, o, nent, fst, enl); }
   Long64_t Process(const char *sel, Long64_t nent, Option_t *o = "")
                    { return TProof::Process(sel, nent, o); }
   // Process via TSelector
   Long64_t Process(TDSet *set, TSelector *selector,
                    Option_t *option = "", Long64_t nentries = -1,
                    Long64_t firstentry = 0)
                    { return TProof::Process(set, selector, option, nentries, firstentry); }
   Long64_t Process(TFileCollection *fc, TSelector *sel, Option_t *o = "",
                    Long64_t nent = -1, Long64_t fst = 0)
                    { return TProof::Process(fc, sel, o, nent, fst); }
   Long64_t Process(const char *dsname, TSelector *sel,
                    Option_t *o = "", Long64_t nent = -1,
                    Long64_t fst = 0, TObject *enl = 0)
                    { return TProof::Process(dsname, sel, o, nent, fst, enl); }
   Long64_t Process(TSelector *sel, Long64_t nent, Option_t *o = "")
                    { return TProof::Process(sel, nent, o); }

   ClassDef(TProofSuperMaster,0) //PROOF control class for making submasters
};

#endif
