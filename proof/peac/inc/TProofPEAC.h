// @(#)root/peac:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofPEAC
#define ROOT_TProofPEAC

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPEAC                                                           //
//                                                                      //
// This class implements a PROOF session which uses PEAC                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProof
#include "TProof.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TLM;
class TCondor;
class TTimer;

class TProofPEAC : public TProof {

friend class TCondor;

private:
   TString         fSession;        //PEAC session identifier
   TLM            *fLM;             //connection to PEAC local manager
   Int_t           fHBPeriod;       //requested heartbeat period in seconds
   TCondor        *fCondor;         //condor instance for condor slaves
   TTimer         *fTimer;          //timer for delayed Condor COD suspend
   TTimer         *fHeartbeatTimer; //timer for sending heartbeat to local manager

protected:
   virtual Bool_t  StartSlaves(Bool_t parallel=kTRUE,Bool_t attach=kFALSE);
   TString         GetJobAd();

public:
   TProofPEAC(const char *masterurl, const char *sessionid = 0,
              const char *confdir = 0, Int_t loglevel = 0,
              const char *alias = 0, TProofMgr *mgr = 0);
   virtual ~TProofPEAC();

   virtual Bool_t IsDataReady(Long64_t &totalbytes, Long64_t &bytesready);
   virtual void   SendHeartbeat();
   virtual void   SetActive(Bool_t active = kTRUE);
   virtual void   Close(Option_t *option="");

   ClassDef(TProofPEAC,0)  // PROOF using PEAC
};

#endif
