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

