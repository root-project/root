// @(#)root/proof:$Id$
// Author: Maarten Ballintijn   06/12/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCondor
#define ROOT_TCondor

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCondor                                                              //
//                                                                      //
// Interface to the Condor system. TCondor provides a (partial) API for //
// querying and controlling the Condor system, including experimental   //
// extensions like COD (computing on demand)                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TList;


//------------------------------------------------------------------------

class TCondorSlave : public TObject {
public:
   TString  fHostname;
   Int_t    fPort;
   Int_t    fPerfIdx;
   TString  fImage;
   TString  fClaimID;
   TString  fOrdinal;
   TString  fWorkDir;

   void        Print(Option_t *option="") const;

   ClassDef(TCondorSlave,0)  // Describes a claimed slave
};


//------------------------------------------------------------------------

class TCondor : public TObject {
public:
   enum EState { kFree, kSuspended, kActive };

private:

   Bool_t   fValid;     //access to Condor
   TString  fPool;      //the condor pool to be accessed
   EState   fState;     //our claim state
   TList   *fClaims;    //list of claims we manage

protected:
   TCondorSlave  *ClaimVM(const char *vm, const char *cmd);

public:
   TCondor(const char *pool = "");
   virtual ~TCondor();


   void           Print(Option_t *option="") const;
   Bool_t         IsValid() const { return fValid; }

   TList         *GetVirtualMachines() const;

   TList         *Claim(Int_t n, const char *cmd);
   TCondorSlave  *Claim(const char *vmname, const char *cmd);
   Bool_t         SetState(EState state);
   EState         GetState() const {return fState;}
   Bool_t         Suspend();
   Bool_t         Resume();
   Bool_t         Release();

   Bool_t         GetVmInfo(const char *vm, TString &image, Int_t &perfidx) const;
   TString        GetImage(const char *host) const;


   ClassDef(TCondor,0)  // Interface to the Condor System
};

#endif
