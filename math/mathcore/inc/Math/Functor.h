// @(#)root/mathcore:$Id$
// Author: L. Moneta Mon Nov 13 15:58:13 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for Functor classes.
// designed is inspired by the Loki Functor

#ifndef ROOT_Math_Functor
#define ROOT_Math_Functor

#include "Math/IFunction.h"

// #ifndef Root_Math_StaticCheck
// #include "Math/StaticCheck.h"
// #endif

#include <algorithm>
#include <memory>
#include <functional>
#include <type_traits>
#include <vector>

namespace ROOT {

namespace Math {

/**
   Documentation for class Functor class.
   It is used to wrap in a very simple and convenient way multi-dimensional function objects.
   It can wrap all the following types:
   <ul>
   <li> any C++ callable object implementation double operator()( const double *  )
   <li> a free C function of type double ()(const double * )
   <li> an std::function of type std::function<double (double const *)>
   <li> a member function with the correct signature like Foo::Eval(const double * ).
       In this case one pass the object pointer and a pointer to the member function (&Foo::Eval)
   </ul>
   The function dimension is required when constructing the functor.

   @ingroup  GenFunc

 */
class Functor : public IBaseFunctionMultiDim  {

public:

   /// Default constructor.
   Functor ()  {}

   /// Construct from a pointer to member function (multi-dim type).
   template <class PtrObj, typename MemFn>
   Functor(const PtrObj& p, MemFn memFn, unsigned int dim )
      : fDim{dim}, fFunc{std::bind(memFn, p, std::placeholders::_1)}
   {}

   /// Construct from a callable object of multi-dimension
   /// with the right signature (implementing `double operator()(const double *x)`).
   Functor(std::function<double(double const *)> const& f, unsigned int dim ) : fDim{dim}, fFunc{f} {}

   // clone of the function handler (use copy-ctor)
   Functor * Clone() const override { return new Functor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const override { return fDim; }

private :

   inline double DoEval (const double * x) const override {
      return fFunc(x);
   }

   unsigned int fDim;
   std::function<double(double const *)> fFunc;
};

/**
   Functor1D class for one-dimensional functions.
   It is used to wrap in a very simple and convenient way:
   <ul>
   <li> any C++ callable object implementation double operator()( double  )
   <li> a free C function of type double ()(double )
   <li> a member function with the correct signature like Foo::Eval(double ).
       In this case one pass the object pointer and a pointer to the member function (&Foo::Eval)
   </ul>


   @ingroup  GenFunc

 */

class Functor1D : public IBaseFunctionOneDim  {

public:

   /// Default constructor.
   Functor1D() = default;

   /// Construct from a callable object with the right signature
   /// implementing `double operator() (double x)`.
   Functor1D(std::function<double(double)> const& f) : fFunc{f} {}

   // Construct from a pointer to member function (1D type).
   template <class PtrObj, typename MemFn>
   Functor1D(const PtrObj& p, MemFn memFn) : fFunc{std::bind(memFn, p, std::placeholders::_1)} {}

   // Clone of the function handler (use copy-ctor).
   Functor1D * Clone() const override { return new Functor1D(*this); }

private :

   inline double DoEval (double x) const override {
      return fFunc(x);
   }

   std::function<double(double)> fFunc;
};

/**
   GradFunctor class for Multidimensional gradient functions.
   It is used to wrap in a very C++ callable object to make gradient functions.
   It can be constructed in three different way:
   <ol>
   <li> from an object implementing both
        double operator()( const double * ) for the function evaluation  and
        double Derivative(const double *, int icoord) for the partial derivatives
    <li>from an object implementing any member function like Foo::XXX(const double *) for the function evaluation
        and any member function like Foo::XXX(const double *, int icoord) for the partial derivatives
    <li>from two function objects implementing
        double operator()( const double * ) for the function evaluation and another function object implementing
        double operator() (const double *, int icoord) for the partial derivatives
    <li>from two function objects
   </ol>
   The function dimension is required when constructing the functor.

   @ingroup  GenFunc

 */
class GradFunctor : public IGradientFunctionMultiDim  {


public:

   /// Default constructor.
   GradFunctor() = default;

   /**
      construct from a callable object of multi-dimension
      implementing operator()(const double *x) and
      Derivative(const double * x,icoord)
    */
   template <typename Func>
   GradFunctor( const Func & f, unsigned int dim ) :
      fDim{dim}, fFunc{f}, fDerivFunc{std::bind(&Func::Derivative, f, std::placeholders::_1, std::placeholders::_2)}
   {}

   /// Construct from a pointer to member function and member function types for function and derivative evaluations.
   template <class PtrObj, typename MemFn, typename DerivMemFn,
             std::enable_if_t<std::is_floating_point<decltype((std::declval<std::remove_pointer_t<PtrObj>>().*
                                                               std::declval<DerivMemFn>())(
                                 std::declval<const double *>(), std::declval<int>()))>::value,
                              bool> = true>
   GradFunctor(const PtrObj &p, MemFn memFn, DerivMemFn gradFn, unsigned int dim)
      : fDim{dim},
        fFunc{std::bind(memFn, p, std::placeholders::_1)},
        fDerivFunc{std::bind(gradFn, p, std::placeholders::_1, std::placeholders::_2)}
   {}

   /// Construct from a pointer to member function and member function, types for function and full derivative
   /// evaluations.
   template <
      class PtrObj, typename MemFn, typename GradMemFn,
      std::enable_if_t<std::is_void<decltype((std::declval<std::remove_pointer_t<PtrObj>>().*std::declval<GradMemFn>())(
                          std::declval<const double *>(), std::declval<double *>()))>::value,
                       bool> = true>
   GradFunctor(const PtrObj &p, MemFn memFn, GradMemFn gradFn, unsigned int dim)
      : fDim{dim},
        fFunc{std::bind(memFn, p, std::placeholders::_1)},
        fGradFunc{std::bind(gradFn, p, std::placeholders::_1, std::placeholders::_2)}
   {
   }

   /// Construct for Gradient Functions of multi-dimension Func gives the
   /// function evaluation, GradFunc the partial derivatives The function
   /// dimension is required.
   GradFunctor(std::function<double(double const *)> const& f,
               std::function<double(double const *, unsigned int)> const& g, unsigned int dim)
      : fDim{dim}, fFunc{f}, fDerivFunc{g}
   {}

   /**
    * @brief Construct a new GradFunctor object using 2 std::function,
    *        one for the function evaluation and one for the Gradient
    *        Note the difference with the constructor above where partial derivative function
    *        is used as input
    *
    * @param f  : function object computing the function value
    * @param dim : number of function dimension
    * @param g   : function object computing the function gradient
    */
   GradFunctor(std::function<double(double const *)> const&f, unsigned int dim,
               std::function<void(double const *, double *)> const& g)
      : fDim{dim}, fFunc{f}, fGradFunc{g}
   {}

   // Clone of the function handler (use copy-ctor).
   GradFunctor * Clone() const override { return new GradFunctor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const override { return fDim; }

   void Gradient(const double *x, double *g) const override {
      // Fall back to base implementation if no gradient function is provided
      // (it will fill the gradient calling DoDerivative() for each component).
      if(!fGradFunc) {
         IGradientFunctionMultiDim::Gradient(x, g);
         return;
      }
      fGradFunc(x, g);
   }

private :

   inline double DoEval (const double * x) const override {
      return fFunc(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord  ) const override {
      if(fDerivFunc) {
         return fDerivFunc(x, icoord);
      }
      // Get the component from the gradient if not derivative function is
      // specified.
      std::vector<double> gradBuffer(fDim);
      std::fill(gradBuffer.begin(), gradBuffer.end(), 0.0);
      fGradFunc(x, gradBuffer.data());
      return gradBuffer[icoord];
   }

   unsigned int fDim;
   std::function<double(const double *)> fFunc;
   std::function<double(double const *, unsigned int)> fDerivFunc;
   std::function<void(const double *, double*)> fGradFunc;
};


//_______________________________________________________________________________________________
/**
   GradFunctor1D class for one-dimensional gradient functions.
   It is used to wrap in a very C++ callable object to make a 1D gradient functions.
   It can be constructed in three different way:
   <ol>
   <li> from an object implementing both
        double operator()( double  ) for the function evaluation  and
        double Derivative(double ) for the partial derivatives
    <li>from an object implementing any member function like Foo::XXX(double ) for the function evaluation
        and any other member function like Foo::YYY(double ) for the derivative.
    <li>from an 2 function objects implementing
        double operator()( double ) . One object provides the function evaluation, the other the derivative.
   </ol>

   @ingroup  GenFunc

 */

class GradFunctor1D : public IGradientFunctionOneDim  {

public:

   /// Default constructor.
   GradFunctor1D() = default;

   /// Construct from an object with the right signature,
   /// implementing both `operator() (double x)` and `Derivative(double x)`.
   template <typename Func>
   GradFunctor1D(const Func & f) : fFunc{f}, fDerivFunc{std::bind(&Func::Derivative, f, std::placeholders::_1)} {}

   /**
       construct from a pointer to class and two pointers to member functions, one for
       the function evaluation and the other for the derivative.
       The member functions must take a double as argument and return a double
    */
   template <class PtrObj, typename MemFn, typename GradMemFn>
   GradFunctor1D(const PtrObj& p, MemFn memFn, GradMemFn gradFn)
      : fFunc{std::bind(memFn, p, std::placeholders::_1)}, fDerivFunc{std::bind(gradFn, p, std::placeholders::_1)}
   {}


   /// Specialized constructor from 2 function objects implementing double
   /// operator()(double x). The first one for the function evaluation and the
   /// second one implementing the function derivative.
   GradFunctor1D(std::function<double(double)> const& f, std::function<double(double)> const& g)
      : fFunc{f}, fDerivFunc{g}
   {}

   // clone of the function handler (use copy-ctor)
   GradFunctor1D * Clone() const override { return new GradFunctor1D(*this); }

private :

   inline double DoEval (double x) const override { return fFunc(x); }
   inline double DoDerivative (double x) const override { return fDerivFunc(x); }

   std::function<double(double)> fFunc;
   std::function<double(double)> fDerivFunc;
};


} // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Functor */
