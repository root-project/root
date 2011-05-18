// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofBenchDataSet
#define ROOT_TProofBenchDataSet

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchDataSet                                                   //
//                                                                      //
// Handle operations on datasets used by ProofBench                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TProof;

class TProofBenchDataSet: public TObject {

private:
   
protected:
   TProof* fProof;   //proof

   Int_t Handle(const char *dset, TObject *type);

public:

   TProofBenchDataSet(TProof *proof = 0);
   virtual ~TProofBenchDataSet() { }

   Bool_t IsProof(TProof *p) { return (p == fProof) ? kTRUE : kFALSE; } 

   Int_t CopyFiles(const char *dset, const char *destdir);
   Int_t ReleaseCache(const char *dset);
   Int_t RemoveFiles(const char *dset);

   ClassDef(TProofBenchDataSet,0)   //Handle operations on datasets
};

#endif
