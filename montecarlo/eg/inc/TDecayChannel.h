// @(#)root/eg:$Id$
// Author: P.Murat   15/02/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//-----------------------------------------------------------------------------
//  Feb 15 2001 P.Murat: description of the decay channel
//  --------------------
//  - matrix element for the decay is not defined yet
//-----------------------------------------------------------------------------
#ifndef ROOT_TDecayChannel
#define ROOT_TDecayChannel

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TArrayI
#include "TArrayI.h"
#endif


class TDecayChannel: public TObject{
protected:
   Int_t     fNumber;                        // channel number
   Int_t     fMatrixElementCode;             // matrix element for this decay mode
   Double_t  fBranchingRatio;                // branching ratio ( < 1)
   TArrayI   fDaughters;                     // PDG codes of the daughters
public:
   // ****** constructors and destructor
   TDecayChannel();
   TDecayChannel(Int_t     Number,
                 Int_t     MatrixElementCode,
                 Double_t  BranchingRatio,
                 Int_t     NDaughters,
                 Int_t*    DaughterPdgCode);

   virtual ~TDecayChannel();
   // ****** accessors

   Int_t     Number                () { return fNumber; }
   Int_t     MatrixElementCode     () { return fMatrixElementCode;  }
   Int_t     NDaughters            () { return fDaughters.fN;    }
   Double_t  BranchingRatio        () { return fBranchingRatio; }
   Int_t     DaughterPdgCode(Int_t i) { return fDaughters.fArray[i]; }

   ClassDef(TDecayChannel,1)   // Class describing a particle decay channel
};

#endif
