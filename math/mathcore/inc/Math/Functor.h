// @(#)root/mathcore:$Id$
// Author: L. Moneta Mon Nov 13 15:58:13 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Heaer file for Functor classes.
// designed is inspired by the Loki Functor

#ifndef ROOT_Math_Functor
#define ROOT_Math_Functor

#include "Math/IFunction.h"

// #ifndef Root_Math_StaticCheck
// #include "Math/StaticCheck.h"
// #endif

#include <memory>
#include <functional>
#include <vector>

namespace ROOT {

namespace Math {

/**
   @defgroup Functor_int Internal Functor Classes
   Internal classes for implementing Functor and Functor1D classes
   @ingroup GenFunc
 */

/**
   FunctorImpl is a base class for the functor
   handler implementation class.
   It defines the Copy operator used to clone the functor objects
*/

template<class IBaseFunc>
class FunctorImpl : public IBaseFunc {

public:

   typedef IBaseFunc BaseFunc;


   FunctorImpl() : IBaseFunc() { }

   virtual ~FunctorImpl() {}

   virtual FunctorImpl* Copy() const = 0;

};


/**
   Functor Handler class for gradient functions where both callable objects are provided for the function
   evaluation (type Func) and for the derivative (type DerivFunc) .
   It can be created from any function implementing the correct signature
   corresponding to the requested type
   In the case of one dimension the function evaluation object and the derivative function object must implement
   double operator() (double x).
   In the case of multi-dimension the function evaluation object must implement
   double operator() (const double * x) and the derivative function object must implement
   double operator() (const double * x, int icoord)

   @ingroup  Functor_int
*/
template<class ParentFunctor>
class FunctorDerivHandler : public ParentFunctor::Impl {

   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;

public:

   // constructor for multi-dimensional functions
   FunctorDerivHandler(unsigned int dim, const std::function<double(double const *)>& fun, const  std::function<double(double const *, unsigned int)>& gfun) :
      fDim(dim),
      fFunc(fun),
      fDerivFunc( gfun )
   {}

   virtual ~FunctorDerivHandler() {}

   // clone of the function handler (use copy-ctor)
   ImplFunc * Copy() const { return new FunctorDerivHandler(*this); }

   // clone of the function handler (use copy-ctor)
#ifdef _MSC_VER
   // FIXME: this is a work-around for a a problem with how the compiler
   // generates the covariant virtual function "Clone". To address the
   // issue just use the original return type of the virtual base class.
   // Try to remove this #ifdef when updating Visual Studio
   typename ParentFunctor::ImplBase* Clone() const { return Copy(); }
#else
   BaseFunc * Clone() const { return Copy(); }
#endif

   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

private :

   inline double DoEval (const double * x) const {
      return fFunc(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord ) const {
      return fDerivFunc(x, icoord);
   }


   unsigned int fDim;
   mutable std::function<double(double const *)> fFunc;
   mutable std::function<double(double const *, unsigned int)> fDerivFunc;

};

/**
   Functor Handler class for gradient functions where both callable objects are provided for the function
   evaluation (type Func) and for the gradient (type GradFunc) .
   It can be created from any function implementing the correct signature
   corresponding to the requested type
   The function evaluation (Func)
   In the case of multi-dimension the function evaluation object must implement
   double operator() (const double * x) and the gradient function object must implement
   double operator() (const double * x, int icoord)

   @ingroup  Functor_int
*/
template<class ParentFunctor >
class FunctorGradHandler : public ParentFunctor::Impl {

   // we don't need to template this class and just use std::function

   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;

public:

// constructor for multi-dimensional functions
   FunctorGradHandler(unsigned int dim, std::function<double(const double *)>  fun, std::function<void(const double *, double*)> gfun) :
      fDim(dim),
      fFunc(fun),
      fGradFunc( gfun )
   {}

   virtual ~FunctorGradHandler() {}

   // clone of the function handler (use copy-ctor)
   ImplFunc * Copy() const { return new FunctorGradHandler(*this); }

   // clone of the function handler (use copy-ctor)
#ifdef _MSC_VER
   // FIXME: this is a work-around for a a problem with how the compiler
   // generates the covariant virtual function "Clone". To address the
   // issue just use the original return type of the virtual base class.
   // Try to remove this #ifdef when updating Visual Studio
   typename ParentFunctor::ImplBase* Clone() const { return Copy(); }
#else
   BaseFunc * Clone() const { return Copy(); }
#endif

   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

   void Gradient(const double *x, double *g) const {
         fGradFunc(x,g);
   }

private :

   inline double DoEval (const double * x) const {
      return fFunc(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord ) const {
      std::vector<double> g(fDim);
      fGradFunc(x, g.data());
      return g[icoord];
   }

private:

   unsigned int fDim;
   mutable std::function<double(const double *)> fFunc;
   mutable std::function<void(const double *, double*)> fGradFunc;
};


//****************************
// LM 7/2/2014:  no needed this : make template ctor of Functor1D and GradFunctor1D not
// available to CINT s
//***************************************
//#if defined(__MAKECINT__) || defined(G__DICTIONARY)
// needed since CINT initialize it with TRootIOCtor
//class TRootIOCtor;

// template<class ParentFunctor>
// class FunctorHandler<ParentFunctor,TRootIOCtor *> : public ParentFunctor::Impl
// {
// public:
//    typedef typename ParentFunctor::Impl ImplFunc;
//    typedef typename ImplFunc::BaseFunc BaseFunc;

//    FunctorHandler(TRootIOCtor  *) {}
//    // function required by interface
//    virtual ~FunctorHandler() {}
//    double DoEval (double ) const  { return 0; }
//    double DoDerivative (double ) const  { return 0; }
//    ImplFunc  * Copy() const {  return 0;  }
//    BaseFunc  * Clone() const {  return 0;  }

// };
// #endif


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

   typedef FunctorImpl<IBaseFunctionMultiDim> Impl;
   typedef IBaseFunctionMultiDim::BaseFunc ImplBase;

   /**
      Default constructor
   */
   Functor ()  {}


   /**
       construct from a pointer to member function (multi-dim type)
    */
   template <class PtrObj, typename MemFn>
   Functor(const PtrObj& p, MemFn memFn, unsigned int dim )
      : fDim{dim}, fFunc{std::bind(memFn, p, std::placeholders::_1)}
   {}


   /**
      construct from a callable object of multi-dimension
      with the right signature (implementing operator()(const double *x)
    */
   template <typename Func>
   Functor( const Func & f, unsigned int dim ) : fDim{dim}, fFunc{f} {}

   /**
        specialized constructor from a std::function of multi-dimension
        with the right signature (double operator()(double const *x)
        This specialized constructor is introduced in order to use the Functor class in
        Python passing Python user defined functions
      */
   //template <typename Func>
   Functor(const std::function<double(double const *)> &f, unsigned int dim)
      : fDim{dim}, fFunc{f}
   {}

   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const override { return new Functor(*this); }

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

   typedef FunctorImpl<IBaseFunctionOneDim>          Impl;
   typedef IBaseFunctionOneDim::BaseFunc ImplBase;

   /**
      Default constructor
   */
   Functor1D ()   {}

   /**
      construct from a callable object with the right signature
      implementing operator() (double x)
    */
   template <typename Func>
   Functor1D(const Func & f) :
      fFunc{f}
   {}


   /**
       construct from a pointer to member function (1D type)
    */
   template <class PtrObj, typename MemFn>
   Functor1D(const PtrObj& p, MemFn memFn)
      : fFunc{std::bind(memFn, p, std::placeholders::_1)}
   {}

   /**
      specialized constructor from a std::function implementing the function evaluation.
      This specialized constructor is introduced in order to use the Functor class in
      Python passing Python user defined functions
   */
   Functor1D(const std::function<double(double)> &f) : fFunc{f} {}


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const override { return new Functor1D(*this); }

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

   typedef FunctorImpl<IGradientFunctionMultiDim> Impl;
   typedef IGradientFunctionMultiDim::BaseFunc ImplBase;


   /**
      Default constructor
   */
   GradFunctor ()   {}

   /**
      construct from a callable object of multi-dimension
      implementing operator()(const double *x) and
      Derivative(const double * x,icoord)
    */
   template <typename Func>
   GradFunctor( const Func & f, unsigned int dim ) :
      fImpl(new FunctorDerivHandler<GradFunctor>(dim, f, std::bind(&Func::Derivative, f, std::placeholders::_1, std::placeholders::_2)) )
   {}

   /**
       construct from a pointer to member function and member function types for function and derivative evaluations
    */
   template <class PtrObj, typename MemFn, typename GradMemFn>
   GradFunctor(const PtrObj& p, MemFn memFn, GradMemFn gradFn, unsigned int dim )
      : fImpl(new FunctorDerivHandler<GradFunctor>(dim, std::bind(memFn, p, std::placeholders::_1), std::bind(gradFn, p, std::placeholders::_1, std::placeholders::_2)) )
   {}

   /**
      construct for Gradient Functions of multi-dimension
      Func gives the function evaluation, GradFunc the partial derivatives
      The function dimension is  required
    */
   template <typename Func, typename GradFunc>
   GradFunctor(const Func & f, const GradFunc & g, int dim  ) :
      fImpl(new FunctorDerivHandler<GradFunctor>(dim, f, g) )
   { }

   /**
      specialized constructor from 2 std::functions
      with the right signature (the first one implementing double operator()(double const *x)
      for the function evaluation and the second one implementing double operator()(double const *x, unsigned int icoord)
      for the function partial derivatives.
      This specialized constructor is introduced in order to use the Functor class in
      Python passing Python user defined functions
    */
   // template <typename Func>
   GradFunctor(const std::function<double(double const *)> &f,
               const std::function<double(double const *, unsigned int)> &g, unsigned int dim)
      : fImpl(new FunctorDerivHandler<GradFunctor>(dim, f, g))
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
   GradFunctor(const std::function<double(double const *)> &f, int dim,
               const std::function<void(double const *, double *)> &g)
      : fImpl(new FunctorGradHandler<GradFunctor>(dim, f, g))
   {}

   /**
      Destructor (no operations)
   */
   virtual ~GradFunctor ()  {}


   /**
      Copy constructor for functor based on ROOT::Math::IMultiGradFunction
   */
   GradFunctor(const GradFunctor & rhs) :
      ImplBase()
   {
      if (rhs.fImpl)
         fImpl = std::unique_ptr<Impl>(rhs.fImpl->Copy());
   }

   /**
      Assignment operator
   */
   GradFunctor & operator = (const GradFunctor & rhs)  {
      GradFunctor copy(rhs);
      fImpl.swap(copy.fImpl);
      return *this;
   }


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const { return new GradFunctor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const { return fImpl->NDim(); }

   void Gradient(const double *x, double *g) const {
      fImpl->Gradient(x,g);
   }

private :


   inline double DoEval (const double * x) const {
      return (*fImpl)(x);
   }


   inline double DoDerivative (const double * x, unsigned int icoord  ) const {
      return fImpl->Derivative(x,icoord);
   }

   std::unique_ptr<Impl> fImpl;    // pointer to base grad functor handler


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
   /**
      Default constructor
   */
   GradFunctor1D ()   {}


   /**
      construct from an object with the right signature
      implementing both operator() (double x) and Derivative(double x)
    */
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


   /**
      construct from two 1D function objects
    */
   template <typename Func, typename GradFunc>
   GradFunctor1D(const Func & f, const GradFunc & g ) : fFunc{f}, fDerivFunc{g} {}

   /**
     specialized constructor from 2 std::function objects
     implementing double operator()(double x). The first one for the function evaluation
     and the second one implementing the function derivative.
     This specialized constructor is introduced in order to use the class in
     Python passing Python user defined functions
   */
   GradFunctor1D(const std::function<double(double)> &f, const std::function<double(double)> &g )
      : fFunc{f}, fDerivFunc{g}
   {}

   // clone of the function handler (use copy-ctor)
   GradFunctor1D * Clone() const { return new GradFunctor1D(*this); }


private :

   inline double DoEval (double x) const { return fFunc(x); }
   inline double DoDerivative (double x) const { return fDerivFunc(x); }

   std::function<double(double)> fFunc;
   std::function<double(double)> fDerivFunc;
};



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Functor */
