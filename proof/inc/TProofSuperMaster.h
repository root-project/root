// @(#)root/proof:$Name:  $:$Id: TProofSuperMaster.h,v 1.1 2005/06/22 20:25:28 brun Exp $
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

#ifndef ROOT_TProof
#include "TProof.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TProofPlayer;
class TDSet;

class TProofSuperMaster : public TProof {

friend class TProofPlayerSuperMaster;

protected:
   Bool_t  StartSlaves(Bool_t);
   Int_t   Process(TDSet *set, const Char_t *selector,
                   Option_t *option = "", Long64_t nentries = -1,
                   Long64_t firstentry = 0, TEventList *evl = 0);
   void    ValidateDSet(TDSet *dset);
   virtual TProofPlayer *MakePlayer();

public:
   TProofSuperMaster(const Char_t *masterurl, const Char_t *conffile = kPROOF_ConfFile,
                     const Char_t *confdir = kPROOF_ConfDir, Int_t loglevel = 0);
   virtual ~TProofSuperMaster() { }

   ClassDef(TProofSuperMaster,0) //PROOF control class for making submasters
};

#endif
