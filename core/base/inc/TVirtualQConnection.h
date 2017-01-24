// @(#)root/base:$Id$
// Author: Vassil Vassilev 23/01/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualQConnection
#define ROOT_TVirtualQConnection

#include "TList.h"

class TVirtualQConnection : public TList {
protected:
   virtual void SetArg(Long_t) = 0;
   virtual void SetArg(ULong_t) = 0;
   virtual void SetArg(Float_t) = 0;
   virtual void SetArg(Double_t) = 0;
   virtual void SetArg(Long64_t) = 0;
   virtual void SetArg(ULong64_t) = 0;

   // Note: sets argument list (for potentially more than one arg).
   virtual void SetArg(const Long_t *, Int_t = -1) = 0;
   virtual void SetArg(const char *) = 0;

   // We should 'widen' all types to one of the SetArg overloads.
   template <class T, class = typename std::enable_if<std::is_scalar<T>::value>::type>
   void SetArg(const T& val)
   {
      if (std::is_signed<T>::value)
         SetArg((Long_t)val);
      else
         SetArg((ULong_t)val);
   }

   // We pass arrays as Lont_t*. In the template instance we can deduce their
   // size, too.
   template <class T, class = typename std::enable_if<std::is_array<T>::value>::type>
   void SetArg(const T* val)
   {
      constexpr size_t size = sizeof(val)/sizeof(val[0]);
      static_assert(size > 0, "The array must have at least one element!");
      SetArg((Long_t*)val, size);
   }

   void SetArgsImpl() {} // SetArgsImpl terminator
   template <typename T, typename... Ts> void SetArgsImpl(const T& arg, const Ts&... tail)
   {
      SetArg(arg);
      SetArgsImpl(tail...);
   }
public:
   virtual void SendSignal() = 0;
   template <typename... T> void SetArgs(const T&... args)
   {
      SetArgsImpl(args...);
   }
};

#endif
