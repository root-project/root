// @(#)root/eg:$Name:$:$Id:$
// Author: P.Murat   15/02/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//-----------------------------------------------------------------------------
//  Feb 16 2001 P.Murat: description of the decay channel
//-----------------------------------------------------------------------------
#include "TDecayChannel.h"

ClassImp(TDecayChannel)

//_____________________________________________________________________________
TDecayChannel::TDecayChannel()
{
  fNumber            = 0;
  fMatrixElementCode = 0;
  fBranchingRatio    = 0;
}

//_____________________________________________________________________________
TDecayChannel::TDecayChannel(Int_t    Number,
			     Int_t    MatrixElementType,
			     Double_t BranchingRatio,
			     Int_t    NDaughters,
			     Int_t*   DaughterPdgCode)
{
  fNumber            = Number;
  fMatrixElementCode = MatrixElementType;
  fBranchingRatio    = BranchingRatio;
  fDaughters.Set(NDaughters,DaughterPdgCode);
}

//_____________________________________________________________________________
TDecayChannel::~TDecayChannel() {
}

