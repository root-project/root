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
#include "TInterpreter.h"

/// Mediates the link between the signal and the slot. It decouples the setting of
/// arguments and sending a signal.
///
/// There are three different modes in argument setting required by TQObject's Emit/EmitVA:
/// setting integral types, setting array types and setting const char*.
class TVirtualQConnection : public TList {
protected:
   virtual CallFunc_t *GetSlotCallFunc() const = 0;
   virtual void SetArg(Long_t) = 0;
   virtual void SetArg(ULong_t) = 0;
   virtual void SetArg(Float_t) = 0;
   virtual void SetArg(Double_t) = 0;
   virtual void SetArg(Long64_t) = 0;
   virtual void SetArg(ULong64_t) = 0;

   // Note: sets argument list (for potentially more than one arg).
   virtual void SetArg(const Longptr_t *, Int_t = -1) = 0;
   virtual void SetArg(const char *) = 0;
   void SetArg(const void *ptr) { SetArg((Longptr_t)ptr); };

   // We should 'widen' all types to one of the SetArg overloads.
   template <class T, class = typename std::enable_if<std::is_integral<T>::value>::type>
   void SetArg(const T& val)
   {
      if (std::is_signed<T>::value)
         SetArg((Longptr_t)val);
      else
         SetArg((ULongptr_t)val);
   }

   // We pass arrays as Lont_t*. In the template instance we can deduce their
   // size, too.
   template <class T, class = typename std::enable_if<std::is_array<T>::value>::type>
   void SetArg(const T* val)
   {
      constexpr size_t size = sizeof(val)/sizeof(val[0]);
      static_assert(size > 0, "The array must have at least one element!");
      SetArg((Longptr_t*)val, size);
   }

   void SetArgsImpl() {} // SetArgsImpl terminator
   template <typename T, typename... Ts> void SetArgsImpl(const T& arg, const Ts&... tail)
   {
      SetArg(arg);
      SetArgsImpl(tail...);
   }
public:
   virtual void SendSignal() = 0;

   /// Unpacks the template parameter type and sets arguments of integral and array (scalar) type.
   ///
   template <typename... T> void SetArgs(const T&... args)
   {
      // Do not reset the arguments if we have no arguments to reset with.
      // This is essential in order to support cases like:
      // void f(int) {}; TQObject::Connect (... "f(int=12");
      // The implementation will see we create a slot which has a 'default'
      // argument and create a CallFunc with preset argument values to later call.
      if (!sizeof...(args)) return;
      gInterpreter->CallFunc_ResetArg(GetSlotCallFunc());
      SetArgsImpl(args...);
   }

   /// Sets an array of arguments passed as a pointer type and size. If nargs is not specified
   /// the number of arguments expected by the slot is used.
   ///
   void SetArgs(const Longptr_t* argArray, Int_t nargs = -1)
   {
      SetArg(argArray, nargs);
   }
};

#endif
