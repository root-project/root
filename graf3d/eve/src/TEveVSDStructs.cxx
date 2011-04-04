// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveVSDStructs.h"


//______________________________________________________________________________
//
// Not documented.
//

ClassImp(TEveMCTrack);
ClassImp(TEveHit);
ClassImp(TEveCluster);
ClassImp(TEveRecTrackT<Float_t>);
ClassImp(TEveRecTrackT<Double_t>);
ClassImp(TEveRecKink);
ClassImp(TEveRecV0);
ClassImp(TEveRecCascade);
ClassImp(TEveMCRecCrossRef);

template class TEveRecTrackT<Float_t>;
template class TEveRecTrackT<Double_t>;
