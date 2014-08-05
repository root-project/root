// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofBenchRun                                                       //
//                                                                      //
// Abstract base class for PROOF benchmark run.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofBenchRun.h"
#include "TList.h"
#include "TProof.h"

ClassImp(TProofBenchRun)

//______________________________________________________________________________
TProofBenchRun::TProofBenchRun(TProof *proof, const char *sel) : fSelName(sel)
{
   // Constructor: check PROOF and load selectors PAR

   fProof = proof;
   if (!fProof){
      fProof = gProof;
   }
   ResetBit(kInvalidObject);
   if (!fProof || (fProof && !fProof->IsValid())) {
      SetBit(kInvalidObject);
      return;
   }
}

//______________________________________________________________________________
TProofBenchRun::~TProofBenchRun()
{
   //destructor
}
