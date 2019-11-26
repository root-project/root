// @(#)root/eve7:$Id$
// Author: Matevz Tadel 2010

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REvePathMark
#define ROOT7_REvePathMark

#include <ROOT/REveVector.hxx>

namespace ROOT {
namespace Experimental {

//==============================================================================
// REvePathMark
//==============================================================================

template <typename TT>
class REvePathMarkT {
public:
   enum EType_e { kReference, kDaughter, kDecay, kCluster2D, kLineSegment };

   EType_e fType;      // Mark-type.
   REveVectorT<TT> fV; // Vertex.
   REveVectorT<TT> fP; // Momentum.
   REveVectorT<TT> fE; // Extra, meaning depends on fType.
   TT fTime;           // Time.

   REvePathMarkT(EType_e type = kReference) : fType(type), fV(), fP(), fE(), fTime(0) {}

   REvePathMarkT(EType_e type, const REveVectorT<TT> &v, TT time = 0) : fType(type), fV(v), fP(), fE(), fTime(time) {}

   REvePathMarkT(EType_e type, const REveVectorT<TT> &v, const REveVectorT<TT> &p, TT time = 0)
      : fType(type), fV(v), fP(p), fE(), fTime(time)
   {
   }

   REvePathMarkT(EType_e type, const REveVectorT<TT> &v, const REveVectorT<TT> &p, const REveVectorT<TT> &e,
                 TT time = 0)
      : fType(type), fV(v), fP(p), fE(e), fTime(time)
   {
   }

   template <typename OO>
   REvePathMarkT(const REvePathMarkT<OO> &pm)
      : fType((EType_e)pm.fType), fV(pm.fV), fP(pm.fP), fE(pm.fE), fTime(pm.fTime)
   {
   }

   const char *TypeName();
};

typedef REvePathMarkT<Float_t> REvePathMark;
typedef REvePathMarkT<Float_t> REvePathMarkF;
typedef REvePathMarkT<Double_t> REvePathMarkD;

} // namespace Experimental
} // namespace ROOT

#endif
