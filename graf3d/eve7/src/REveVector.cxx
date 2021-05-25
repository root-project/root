// @(#)root/eve7:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveVector.hxx>
#include "TVector3.h"

namespace ROOT {
namespace Experimental {

/** \class REveVectorT
\ingroup REve
Minimal, templated three-vector.
No TObject inheritance and virtual functions.
Also used in VSD.
*/

////////////////////////////////////////////////////////////////////////////////
/// Dump to stdout as "(x, y, z)\n".

template<typename TT> void REveVectorT<TT>::Dump() const
{
   printf("(%f, %f, %f)\n", fX, fY, fZ);
}

////////////////////////////////////////////////////////////////////////////////
/// Set from TVector3.

template<typename TT> void REveVectorT<TT>::Set(const TVector3& v)
{
   fX = v.x(); fY = v.y(); fZ = v.z();
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate eta of the point, pretending it's a momentum vector.

template<typename TT> TT REveVectorT<TT>::Eta() const
{
   TT cosTheta = CosTheta();
   if (cosTheta*cosTheta < 1) return -0.5* TMath::Log( (1.0-cosTheta)/(1.0+cosTheta) );
   Warning("Eta","transverse momentum = 0, returning +/- 1e10");
   return (fZ >= 0) ? 1e10 : -1e10;
}

////////////////////////////////////////////////////////////////////////////////
/// Normalize the vector to length if current length is non-zero.
/// Returns the old magnitude.

template<typename TT> TT REveVectorT<TT>::Normalize(TT length)
{
   TT m = Mag();
   if (m != 0)
   {
      length /= m;
      fX *= length; fY *= length; fZ *= length;
   }
   return m;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns an orthogonal vector (not normalized).

template<typename TT> REveVectorT<TT> REveVectorT<TT>::Orthogonal() const
{
   Float_t xx = fX < 0 ? -fX : fX;
   Float_t yy = fY < 0 ? -fY : fY;
   Float_t zz = fZ < 0 ? -fZ : fZ;
   if (xx < yy) {
      return xx < zz ? REveVectorT<TT>(0,fZ,-fY) : REveVectorT<TT>(fY,-fX,0);
   } else {
      return yy < zz ? REveVectorT<TT>(-fZ,0,fX) : REveVectorT<TT>(fY,-fX,0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set vectors a and b to be normal to this and among themselves,
/// both of length 1.

template<typename TT> void REveVectorT<TT>::OrthoNormBase(REveVectorT<TT>& a, REveVectorT<TT>& b) const
{
   REveVectorT<TT> v(*this);
   v.Normalize();
   a = v.Orthogonal();
   a.Normalize();
   b = v.Cross(a);
   b.Normalize();
}

template class REveVectorT<Float_t>;
template class REveVectorT<Double_t>;

/** \class REveVector4T
\ingroup REve
Minimal, templated four-vector.
No TObject inheritance and virtual functions.
Also used in VSD.
*/

////////////////////////////////////////////////////////////////////////////////
/// Dump to stdout as "(x, y, z; t)\n".

template<typename TT> void REveVector4T<TT>::Dump() const
{
   printf("(%f, %f, %f; %f)\n", TP::fX, TP::fY, TP::fZ, fT);
}

template class REveVector4T<Float_t>;
template class REveVector4T<Double_t>;

/** \class REveVector2T
\ingroup REve
Minimal, templated two-vector.
No TObject inheritance and virtual functions.
Also used in VSD.
*/


////////////////////////////////////////////////////////////////////////////////
/// Normalize the vector to length if current length is non-zero.

template<typename TT> void REveVector2T<TT>::Normalize(TT length)
{
   Float_t m = Mag();
   if (m != 0)
   {
      m = length / m;
      fX *= m; fY *= m;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Dump to stdout as "(x, y)\n".

template<typename TT> void REveVector2T<TT>::Dump() const
{
   printf("(%f, %f)\n", fX, fY);
}

template class REveVector2T<Float_t>;
template class REveVector2T<Double_t>;

} // namespace Experimental
} // namespace ROOT
