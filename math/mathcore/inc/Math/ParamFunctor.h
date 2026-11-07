// @(#)root/mathcore:$Id$
// Author: L. Moneta Mon Nov 13 15:58:13 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for Functor classes.

#ifndef ROOT_Math_ParamFunctor
#define ROOT_Math_ParamFunctor

#include "RtypesCore.h"

#include <functional>
#include <type_traits>
#include <utility>

namespace ROOT {

namespace Math {

/**
   Param Functor class for Multidimensional functions.
   It is used to wrap in a very simple and convenient way
   any other C++ callable object (implementation double operator( const double *, const double * ) )
   or a member function with the correct signature,
   like Foo::EvalPar(const double *, const double *)

   @ingroup  ParamFunc

 */

template <class T>
class ParamFunctorTempl {

public:
   using EvalType = T;

   /// The signature every wrapped callable is normalized to.
   using Signature = T(const T *, const double *);

   ParamFunctorTempl() = default;

   /// Construct from a pointer to a class object and a pointer to one of its member
   /// functions, like `Foo::EvalPar(const double *x, const double *p)`.
   template <class Obj, typename MemFn>
   ParamFunctorTempl(Obj *p, MemFn memFn)
      : fFunc{[p, memFn](const T *x, const double *par) {
           return (p->*memFn)(const_cast<T *>(x), const_cast<double *>(par));
        }}
   {
   }

   /// Construct from any callable object, or from a pointer to one.
   ///
   /// The callable is normalized to the `T (const T *, const double *)` signature: anything
   /// `std::function` accepts as-is is handed to it directly, a pointer to a callable object is
   /// called through without taking ownership of it, and callables that insist on non-const
   /// pointers (the classic `T (T *x, double *p)` signature) get their arguments cast for them.
   ///
   /// The two exclusions keep this greedy forwarding reference away from overloads that a
   /// non-const lvalue would otherwise bind to it by exact match: the implicit copy constructor,
   /// which would end up wrapping a functor in itself, and the `std::function` conversion below,
   /// which is implicit for the Python interface and would become ill-formed via this `explicit` one.
   template <typename Func, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Func>, ParamFunctorTempl> &&
                                                        !std::is_same_v<std::decay_t<Func>, std::function<Signature>>>>
   explicit ParamFunctorTempl(Func &&f)
   {
      using F = std::decay_t<Func>;
      if constexpr (std::is_constructible_v<std::function<Signature>, Func>) {
         fFunc = std::function<Signature>{std::forward<Func>(f)};
      } else if constexpr (std::is_pointer_v<F> && std::is_class_v<std::remove_pointer_t<F>>) {
         fFunc = [f](const T *x, const double *p) { return (*f)(const_cast<T *>(x), const_cast<double *>(p)); };
      } else {
         fFunc = [f = std::forward<Func>(f)](const T *x, const double *p) mutable {
            return f(const_cast<T *>(x), const_cast<double *>(p));
         };
      }
   }

   /// Implicit conversion, relied on by the Python interface when passing a callable to TF1.
   ParamFunctorTempl(std::function<Signature> f) : fFunc{std::move(f)} {}

   T operator()(const T *x, const double *p) const { return fFunc(x, p); }

   bool Empty() const { return !fFunc; }

private:
   std::function<Signature> fFunc;
};

using ParamFunctor = ParamFunctorTempl<double>;

} // end namespace Math

} // end namespace ROOT

#endif /* ROOT_Math_ParamFunctor */
