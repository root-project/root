// @(#)root/eve:$Id$
// Author: Matevz Tadel 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEvePathMark.h"

//==============================================================================
// TEvePathMark
//==============================================================================

//______________________________________________________________________________
//
// Special-point on track:
//  kDaughter  - daughter creation; fP is momentum of the daughter, it is subtracted from
//               momentum of the track
//  kReference - position/momentum reference
//  kDecay     - decay point, fP not used
//  kCluster2D - measurement with large error in one direction (like strip detectors):
//               fP - normal to detector plane,
//               fE - large error direction, must be normalized.
//               Track is propagated to plane and correction in fE direction is discarded.

ClassImp(TEvePathMarkT<Float_t>);
ClassImp(TEvePathMarkT<Double_t>);

//______________________________________________________________________________
template<typename TT> const char* TEvePathMarkT<TT>::TypeName()
{
   // Return the name of path-mark type.

   switch (fType)
   {
      case kDaughter:  return "Daughter";
      case kReference: return "Reference";
      case kDecay:     return "Decay";
      case kCluster2D: return "Cluster2D";
      default:         return "Unknown";
   }
}

template class TEvePathMarkT<Float_t>;
template class TEvePathMarkT<Double_t>;
