// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef Reflex_FuncHandler
#define Reflex_FuncHandler


// Include files
#include "Reflex/Kernel.h"
#include <string>


namespace Reflex {
class FuncHandler {
public:
   template <class R, class C>
   static const std::string DemangleFunRetType(R (C::*)());

   template <class R, class C, class T0>
   static const std::string DemangleFunRetType(R (C::*)(T0));

   template <class R, class C, class T0, class T1>
   static const std::string DemangleFunRetType(R (C::*)(T0, T1));

   template <class R, class C, class T0, class T1, class T2>
   static const std::string DemangleFunRetType(R (C::*)(T0, T1, T2));

   template <class R, class C, class T0, class T1, class T2, class T3>
   static const std::string DemangleFunRetType(R (C::*)(T0, T1, T2, T3));

   template <class R, class C, class T0, class T1, class T2, class T3, class T4>
   static const std::string DemangleFunRetType(R (C::*)(T0, T1, T2, T3, T4));


   template <class R, class C>
   static const std::string DemangleFunParTypes(R (C::*)());

   template <class R, class C, class T0>
   static const std::string DemangleFunParTypes(R (C::*)(T0));

   template <class R, class C, class T0, class T1>
   static const std::string DemangleFunParTypes(R (C::*)(T0, T1));

   template <class R, class C, class T0, class T1, class T2>
   static const std::string DemangleFunParTypes(R (C::*)(T0, T1, T2));

   template <class R, class C, class T0, class T1, class T2, class T3>
   static const std::string DemangleFunParTypes(R (C::*)(T0, T1, T2, T3));

   template <class R, class C, class T0, class T1, class T2, class T3, class T4>
   static const std::string DemangleFunParTypes(R (C::*)(T0, T1, T2, T3, T4));


   template <class R>
   static const std::string DemangleFunRetType(R(*) ());

   template <class R, class T0>
   static const std::string DemangleFunRetType(R(*) (T0));

   template <class R, class T0, class T1>
   static const std::string DemangleFunRetType(R(*) (T0, T1));

   template <class R, class T0, class T1, class T2>
   static const std::string DemangleFunRetType(R(*) (T0, T1, T2));

   template <class R, class T0, class T1, class T2, class T3>
   static const std::string DemangleFunRetType(R(*) (T0, T1, T2, T3));

   template <class R, class T0, class T1, class T2, class T3, class T4>
   static const std::string DemangleFunRetType(R(*) (T0, T1, T2, T3, T4));


   template <class R>
   static const std::string DemangleFunParTypes(R(*) ());

   template <class R, class T0>
   static const std::string DemangleFunParTypes(R(*) (T0));

   template <class R, class T0, class T1>
   static const std::string DemangleFunParTypes(R(*) (T0, T1));

   template <class R, class T0, class T1, class T2>
   static const std::string DemangleFunParTypes(R(*) (T0, T1, T2));

   template <class R, class T0, class T1, class T2, class T3>
   static const std::string DemangleFunParTypes(R(*) (T0, T1, T2, T3));

   template <class R, class T0, class T1, class T2, class T3, class T4>
   static const std::string DemangleFunParTypes(R(*) (T0, T1, T2, T3, T4));

};    // class FuncHandler
} // namespace Reflex

#endif // Reflex_FuncHandler
