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
// TEveVector
//
// Float three-vector; a inimal Float_t copy of TVector3 used to
// represent points and momenta (also used in VSD).

ClassImp(TEveVector)

//______________________________________________________________________________
Float_t TEveVector::Eta() const
{
   Float_t cosTheta = CosTheta();
   if (cosTheta*cosTheta < 1) return -0.5* TMath::Log( (1.0-cosTheta)/(1.0+cosTheta) );
   Warning("Eta","transverse momentum = 0! return +/- 10e10");
   return (z >= 0) ? 10e10 : -10e10;
}

//______________________________________________________________________________
TEveVector TEveVector::operator + (const TEveVector & b)
{
   return TEveVector(x + b.x, y + b.y, z + b.z);
}

//______________________________________________________________________________
TEveVector TEveVector::operator - (const TEveVector & b)
{
   return TEveVector(x - b.x, y - b.y, z - b.z);
}

//______________________________________________________________________________
TEveVector TEveVector::operator * (Float_t a)
{
   return TEveVector(a*x, a*y, a*z);
}


//______________________________________________________________________________
// TEvePathMark
//
// Special-point on track: position/momentum reference, daughter
// creation or decay (also used in VSD).

ClassImp(TEvePathMark)

//______________________________________________________________________________
const char* TEvePathMark::type_name()
{
   switch (type)
   {
      case Daughter:  return "Daughter";
      case Reference: return "Reference";
      case Decay:     return "Decay";
      default:        return "Unknown";
   }
}


//______________________________________________________________________________
//
// Not documented.
//

ClassImp(TEveMCTrack)
ClassImp(TEveHit)
ClassImp(TEveCluster)
ClassImp(TEveRecTrack)
ClassImp(TEveRecKink)
ClassImp(TEveRecV0)
ClassImp(TEveMCRecCrossRef)
