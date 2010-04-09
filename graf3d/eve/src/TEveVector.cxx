// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveVector.h"
#include "TVector3.h"

//==============================================================================
// TEveVector
//==============================================================================

//______________________________________________________________________________
//
// Float three-vector; a inimal Float_t copy of TVector3 used to
// represent points and momenta (also used in VSD).

ClassImp(TEveVectorT<Float_t>);
ClassImp(TEveVectorT<Double_t>);

//______________________________________________________________________________
template<typename TT> void TEveVectorT<TT>::Dump() const
{
   // Dump to stdout as "(x, y, z)\n".

   printf("(%f, %f, %f)\n", fX, fY, fZ);
}

//______________________________________________________________________________
template<typename TT> void TEveVectorT<TT>::Set(const TVector3& v)
{
   // Set from TVector3.

   fX = v.x(); fY = v.y(); fZ = v.z();
}

//______________________________________________________________________________
template<typename TT> TT TEveVectorT<TT>::Eta() const
{
   // Calculate eta of the point, pretending it's a momentum vector.

   TT cosTheta = CosTheta();
   if (cosTheta*cosTheta < 1) return -0.5* TMath::Log( (1.0-cosTheta)/(1.0+cosTheta) );
   Warning("Eta","transverse momentum = 0, returning +/- 1e10");
   return (fZ >= 0) ? 1e10 : -1e10;
}

//______________________________________________________________________________
template<typename TT> TT TEveVectorT<TT>::Normalize(TT length)
{
   // Normalize the vector to length if current length is non-zero.
   // Returns the old magnitude.

   TT m = Mag();
   if (m != 0)
   {
      length /= m;
      fX *= length; fY *= length; fZ *= length;
   }
   return m;
}

//______________________________________________________________________________
template<typename TT> TEveVectorT<TT> TEveVectorT<TT>::Orthogonal() const
{
   // Returns an orthogonal vector (not normalized).

   Float_t xx = fX < 0 ? -fX : fX;
   Float_t yy = fY < 0 ? -fY : fY;
   Float_t zz = fZ < 0 ? -fZ : fZ;
   if (xx < yy) {
      return xx < zz ? TEveVectorT<TT>(0,fZ,-fY) : TEveVectorT<TT>(fY,-fX,0);
   } else {
      return yy < zz ? TEveVectorT<TT>(-fZ,0,fX) : TEveVectorT<TT>(fY,-fX,0);
   }
}

//______________________________________________________________________________
template<typename TT> void TEveVectorT<TT>::OrthoNormBase(TEveVectorT<TT>& a, TEveVectorT<TT>& b) const
{
   // Set vectors a and b to be normal to this and among themselves,
   // both of length 1.

   a = Orthogonal();
   TMath::Cross(this->Arr(), a.Arr(), b.Arr());
   a.Normalize();
   b.Normalize();
}

template class TEveVectorT<Float_t>;
template class TEveVectorT<Double_t>;


//==============================================================================
// TEveVector4
//==============================================================================

//______________________________________________________________________________
//
// Float four-vector.

ClassImp(TEveVector4T<Float_t>);
ClassImp(TEveVector4T<Double_t>);

//______________________________________________________________________________
template<typename TT> void TEveVector4T<TT>::Dump() const
{
   // Dump to stdout as "(x, y, z; t)\n".

   printf("(%f, %f, %f; %f)\n", TP::fX, TP::fY, TP::fZ, fT);
}

template class TEveVector4T<Float_t>;
template class TEveVector4T<Double_t>;


//==============================================================================
// TEvePoint
//==============================================================================

//______________________________________________________________________________
//
// Float two-vector.

ClassImp(TEvePoint);

//______________________________________________________________________________
void TEvePoint::Normalize(Float_t length)
{
   // Normalize the vector to length if current length is non-zero.

   Float_t m = Mag();
   if (m != 0)
   {
      m = length / m;
      fX *= m; fY *= m;
   }
}

//______________________________________________________________________________
TEvePoint TEvePoint::operator + (const TEvePoint & b) const
{
   // Vector addition.

   return TEvePoint(fX + b.fX, fY + b.fY);
}

//______________________________________________________________________________
TEvePoint TEvePoint::operator - (const TEvePoint & b) const
{
   // Vector subtraction.

   return TEvePoint(fX - b.fX, fY - b.fY);
}

//______________________________________________________________________________
TEvePoint TEvePoint::operator * (Float_t a) const
{
   // Multiplication with scalar.

   return TEvePoint(a*fX, a*fY);
}

//______________________________________________________________________________
void TEvePoint::Dump() const
{
   // Dump to stdout as "(x, y)\n".

   printf("(%f, %f)\n", fX, fY);
}


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
