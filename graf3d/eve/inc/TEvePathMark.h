// @(#)root/eve:$Id$
// Author: Matevz Tadel 2010

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePathMark
#define ROOT_TEvePathMark

#include <TEveVector.h>

//==============================================================================
// TEvePathMark
//==============================================================================

template <typename TT>
class TEvePathMarkT
{
public:
   enum EType_e { kReference, kDaughter, kDecay, kCluster2D };

   EType_e         fType; // Mark-type.
   TEveVectorT<TT> fV;    // Vertex.
   TEveVectorT<TT> fP;    // Momentum.
   TEveVectorT<TT> fE;    // Extra, meaning depends on fType.
   TT              fTime; // Time.

   TEvePathMarkT(EType_e type=kReference) :
      fType(type), fV(), fP(), fE(), fTime(0) {}

   TEvePathMarkT(EType_e type, const TEveVectorT<TT>& v, TT time=0) :
      fType(type), fV(v), fP(), fE(), fTime(time) {}

   TEvePathMarkT(EType_e type, const TEveVectorT<TT>& v, const TEveVectorT<TT>& p, TT time=0) :
      fType(type), fV(v), fP(p), fE(), fTime(time) {}

   TEvePathMarkT(EType_e type, const TEveVectorT<TT>& v, const TEveVectorT<TT>& p, const TEveVectorT<TT>& e, TT time=0) :
      fType(type), fV(v), fP(p), fE(e), fTime(time) {}

   const char* TypeName();

   ClassDefNV(TEvePathMarkT, 1); // Template for a special point on a track: position/momentum reference, daughter creation or decay.
};

typedef TEvePathMarkT<Float_t>  TEvePathMark;
typedef TEvePathMarkT<Float_t>  TEvePathMarkF;
typedef TEvePathMarkT<Double_t> TEvePathMarkD;

#endif
