// @(#)root/proof:$Id$
// Author: Fons Rademakers   13/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofCondor
#define ROOT_TProofCondor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofCondor                                                         //
//                                                                      //
// This class controls a Parallel ROOT Facility, PROOF, cluster.        //
// It fires the slave servers, it keeps track of how many slaves are    //
// running, it keeps track of the slaves running status, it broadcasts  //
// messages to all slaves, it collects results, etc.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProof.h"
#include "TString.h"

class TCondor;
class TTimer;

class TProofCondor : public TProof {

friend class TCondor;

private:
   TCondor *fCondor; //proxy for our Condor pool
   TTimer  *fTimer;  //timer for delayed Condor COD suspend

protected:
   Bool_t   StartSlaves(Bool_t);
   TString  GetJobAd();

public:
   TProofCondor(const char *masterurl, const char *conffile = kPROOF_ConfFile,
                const char *confdir = kPROOF_ConfDir, Int_t loglevel = 0,
                const char *alias = 0, TProofMgr *mgr = 0);
   virtual ~TProofCondor();
   virtual void SetActive() { TProof::SetActive(); }
   virtual void SetActive(Bool_t active);

   ClassDef(TProofCondor,0) //PROOF control class for slaves allocated by condor
};

#endif
