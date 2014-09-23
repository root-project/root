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

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

// #ifndef Root_Math_StaticCheck
// #include "Math/StaticCheck.h"
// #endif

#include <memory>


namespace ROOT {

namespace Math {

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
   Functor Handler class is responsible for wrapping any other functor and pointer to
   free C functions.
   It can be created from any function implementing the correct signature
   corresponding to the requested type
   In the case of one dimension the function evaluation object must implement
   double operator() (double x). If it implements a method:  double Derivative(double x)
   can be used to create a Gradient function type.

   In the case of multi-dimension the function evaluation object must implement
   double operator()(const double *x). If it implements a method:
   double Derivative(const double *x, int icoord)
   can be used to create a Gradient function type.

   @ingroup  Functor_int

*/
template<class ParentFunctor, class Func >
class FunctorHandler : public ParentFunctor::Impl {

   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;
   //typedef typename ParentFunctor::Dim Dim;


public:

   // constructor for 1d functions
   FunctorHandler(const Func & fun) : fDim(1), fFunc(fun) {}


   // constructor for multi-dimensional functions w/0 NDim()
   FunctorHandler(unsigned int dim, const Func & fun ) :
      fDim(dim),
      fFunc(fun)
   {}

   virtual ~FunctorHandler() {}

   // copy of the function handler (use copy-ctor)
   ImplFunc * Copy() const {
     return new FunctorHandler(*this);
   }

   // clone of the function handler (use copy-ctor)
   BaseFunc * Clone() const {
      return Copy();
   }


   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

private :

   inline double DoEval (double x) const {
      return fFunc(x);
   }

   inline double DoEval (const double * x) const {
      return fFunc(x);
   }

   inline double DoDerivative (double x) const {
      return fFunc.Derivative(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord ) const {
      return fFunc.Derivative(x,icoord);
   }


   unsigned int fDim;
   mutable Func fFunc;  // should here be a reference and pass a non-const ref in ctor

};


/**
   Functor Handler class for gradient functions where both callable objects are provided for the function
   evaluation (type Func) and for the gradient (type GradFunc) .
   It can be created from any function implementing the correct signature
   corresponding to the requested type
   In the case of one dimension the function evaluation object and the derivative function object must implement
   double operator() (double x).
   In the case of multi-dimension the function evaluation object must implement
   double operator() (const double * x) and the gradient function object must implement
   double operator() (const double * x, int icoord)

   @ingroup  Functor_int
*/
template<class ParentFunctor, class Func, class GradFunc  >
class FunctorGradHandler : public ParentFunctor::Impl {

   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;
   //typedef typename ParentFunctor::Dim Dim;

public:

   // constructor for 1d functions
   FunctorGradHandler(const Func & fun, const GradFunc & gfun) :
      fDim(1),
      fFunc(fun),
      fGradFunc(gfun)
   {}


   // constructor for multi-dimensional functions
   FunctorGradHandler(unsigned int dim, const Func & fun, const GradFunc & gfun) :
      fDim(dim),
      fFunc(fun),
      fGradFunc( gfun )
   {}

   virtual ~FunctorGradHandler() {}

   // clone of the function handler (use copy-ctor)
   ImplFunc * Copy() const { return new FunctorGradHandler(*this); }

   // clone of the function handler (use copy-ctor)
   BaseFunc * Clone() const { return Copy(); }

   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

private :

   inline double DoEval (double x) const {
      return fFunc(x);
   }

   inline double DoEval (const double * x) const {
      return fFunc(x);
   }

   inline double DoDerivative (double x) const {
      return fGradFunc(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord ) const {
      return fGradFunc(x, icoord);
   }


   unsigned int fDim;
   mutable Func fFunc;
   mutable GradFunc fGradFunc;

};


/**
   Functor Handler to Wrap pointers to member functions
   The member function type must be (XXX means any name is allowed) :
   double XXX ( double x) for 1D functions
   and
   double XXXX (const double *x) for multi-dimensional functions

   @ingroup  Functor_int
*/
template <class ParentFunctor, typename PointerToObj,
          typename PointerToMemFn>
class MemFunHandler : public ParentFunctor::Impl
{
   //typedef typename ParentFunctor::Dim Dim;
   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;

public:

   /// constructor from a pointer to the class and a pointer to the function
   MemFunHandler(const PointerToObj& pObj, PointerToMemFn pMemFn)
      : fDim(1), fObj(pObj), fMemFn(pMemFn)        // should pass pointer by value ??
   {}

   /// constructor from a pointer to the class and a pointer to the function
   MemFunHandler(unsigned int dim, const PointerToObj& pObj, PointerToMemFn pMemFn)
      : fDim(dim), fObj(pObj), fMemFn(pMemFn)
   {}

   virtual ~MemFunHandler() {}

   // clone of the function handler (use copy-ctor)
   ImplFunc * Copy() const { return new MemFunHandler(*this); }

   // clone of the function handler (use copy-ctor)
   BaseFunc * Clone() const { return new MemFunHandler(*this); }

   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

private :

   inline double DoEval (double x) const {
      return ((*fObj).*fMemFn)(x);
   }

   inline double DoEval (const double * x) const {
      return ((*fObj).*fMemFn)(x);
   }

   unsigned int fDim;
   mutable PointerToObj fObj;
   PointerToMemFn fMemFn;

};

/**
   Functor Handler to Wrap pointers to member functions for the evaluation of the function
   and the gradient.
   The member function type must be (XXX means any name is allowed) :
   double XXX ( double x) for 1D function and derivative evaluation
   double XXX (const double *x) for multi-dimensional function evaluation and
   double XXX (cost double *x, int icoord) for partial derivatives evaluation

   @ingroup  Functor_int

*/
template <class ParentFunctor, typename PointerToObj,
          typename PointerToMemFn, typename PointerToGradMemFn>
class MemGradFunHandler : public ParentFunctor::Impl
{
   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;
   //typedef typename ParentFunctor::Dim Dim;

public:

   /// constructor from a pointer to the class and a pointer to the function
   MemGradFunHandler(const PointerToObj& pObj, PointerToMemFn pMemFn, PointerToGradMemFn pGradMemFn)
      : fDim(1),
        fObj(pObj),
        fMemFn(pMemFn),
        fGradMemFn(pGradMemFn)
   {}

   /// constructor from a pointer to the class and a pointer to the function
   MemGradFunHandler(unsigned int dim,
                 const PointerToObj& pObj,
                 PointerToMemFn pMemFn,
                 PointerToGradMemFn pGradMemFn )
      : fDim(dim),
        fObj(pObj),
        fMemFn(pMemFn),
        fGradMemFn(pGradMemFn)
   {}

   virtual ~MemGradFunHandler() {}

   // clone of the function handler (use copy-ctor)
   ImplFunc * Copy() const { return new MemGradFunHandler(*this); }

   // clone of the function handler (use copy-ctor)
   BaseFunc * Clone() const { return new MemGradFunHandler(*this); }

   // constructor for multi-dimensional functions
   unsigned int NDim() const {
      return fDim;
   }

private :

   inline double DoEval (double x) const {
      return ((*fObj).*fMemFn)(x);
   }

   inline double DoEval (const double * x) const {
      return ((*fObj).*fMemFn)(x);
   }

   inline double DoDerivative (double x) const {
      return ((*fObj).*fGradMemFn)(x);
   }

   inline double DoDerivative (const double * x, unsigned int icoord ) const {
      return ((*fObj).*fGradMemFn)(x,icoord);
   }

   unsigned int fDim;
   mutable PointerToObj fObj;
   PointerToMemFn fMemFn;
   PointerToGradMemFn fGradMemFn;
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


//_______________________________________________________________________________________________
/**
   Documentation for class Functor class.
   It is used to wrap in a very simple and convenient way multi-dimensional function objects.
   It can wrap all the following types:
   <ul>
   <li> any C++ callable object implemention double operator()( const double *  )
   <li> a free C function of type double ()(const double * )
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
   Functor ()  : fImpl(0) {}


   /**
       construct from a pointer to member function (multi-dim type)
    */
   template <class PtrObj, typename MemFn>
   Functor(const PtrObj& p, MemFn memFn, unsigned int dim )
      : fImpl(new MemFunHandler<Functor, PtrObj, MemFn>(dim, p, memFn))
   {}



   /**
      construct from a callable object of multi-dimension
      with the right signature (implementing operator()(double *x)
    */
   template <typename Func>
   Functor( const Func & f, unsigned int dim ) :
      fImpl(new FunctorHandler<Functor,Func>(dim,f) )
   {}


   /**
      Destructor (no operations)
   */
   virtual ~Functor ()  {}

   /**
      Copy constructor for functor based on ROOT::Math::IMultiGenFunction
   */
   Functor(const Functor & rhs) :
      ImplBase()
   {
      if (rhs.fImpl.get() != 0)
         fImpl = std::auto_ptr<Impl>( (rhs.fImpl)->Copy() );
   }
   // need a specialization in order to call base classes and use  clone


   /**
      Assignment operator
   */
   Functor & operator = (const Functor & rhs)  {
      Functor copy(rhs);
      // swap auto_ptr by hand
      Impl * p = fImpl.release();
      fImpl.reset(copy.fImpl.release());
      copy.fImpl.reset(p);
      return *this;
   }


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const { return new Functor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const { return fImpl->NDim(); }

private :


   inline double DoEval (const double * x) const {
      return (*fImpl)(x);
   }


   std::auto_ptr<Impl> fImpl;   // pointer to base functor handler


};

//______________________________________________________________________________________
/**
   Functor1D class for one-dimensional functions.
   It is used to wrap in a very simple and convenient way:
   <ul>
   <li> any C++ callable object implemention double operator()( double  )
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
   Functor1D ()  : fImpl(0) {}

   /**
      construct from a callable object with the right signature
      implementing operator() (double x)
    */
   template <typename Func>
   Functor1D(const Func & f) :
      fImpl(new FunctorHandler<Functor1D,Func>(f) )
   {}


   /**
       construct from a pointer to member function (1D type)
    */
   template <class PtrObj, typename MemFn>
   Functor1D(const PtrObj& p, MemFn memFn)
      : fImpl(new MemFunHandler<Functor1D, PtrObj, MemFn>(p, memFn))
   {}


   /**
      Destructor (no operations)
   */
   virtual ~Functor1D ()  {}


   /**
      Copy constructor for Functor based on ROOT::Math::IGenFunction
   */
   Functor1D(const Functor1D & rhs) :
      // strange that this is required eventhough ImplBase is an abstract class
      ImplBase()
   {
      if (rhs.fImpl.get() != 0)
         fImpl = std::auto_ptr<Impl>( (rhs.fImpl)->Copy() );
   }


   /**
      Assignment operator
   */
   Functor1D & operator = (const Functor1D & rhs)  {
      Functor1D copy(rhs);
      // swap auto_ptr by hand
      Impl * p = fImpl.release();
      fImpl.reset(copy.fImpl.release());
      copy.fImpl.reset(p);
      return *this;
   }


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const { return new Functor1D(*this); }


private :

   inline double DoEval (double x) const {
      return (*fImpl)(x);
   }


   std::auto_ptr<Impl> fImpl;   // pointer to base functor handler


};

//_______________________________________________________________________________________________
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
    <li>from an function object implementing
        double operator()( const double * ) for the function evaluation and another function object implementing
        double operator() (const double *, int icoord) for the partial derivatives
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
   GradFunctor ()  : fImpl(0) {}

   /**
      construct from a callable object of multi-dimension
      implementing operator()(const double *x) and
      Derivative(const double * x,icoord)
    */
   template <typename Func>
   GradFunctor( const Func & f, unsigned int dim ) :
      fImpl(new FunctorHandler<GradFunctor,Func>(dim,f) )
   {}

   /**
       construct from a pointer to member function and member function types for function and derivative evaluations
    */
   template <class PtrObj, typename MemFn, typename GradMemFn>
   GradFunctor(const PtrObj& p, MemFn memFn, GradMemFn gradFn, unsigned int dim )
      : fImpl(new MemGradFunHandler<GradFunctor, PtrObj, MemFn, GradMemFn>(dim, p, memFn, gradFn))
   {}

   /**
      construct for Gradient Functions of multi-dimension
      Func gives the function evaluatiion, GradFunc the partial derivatives
      The function dimension is  required
    */
   template <typename Func, typename GradFunc>
   GradFunctor(const Func & f, const GradFunc & g, int dim  ) :
      fImpl(new FunctorGradHandler<GradFunctor,Func,GradFunc>(dim, f, g) )
   { }


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
      if (rhs.fImpl.get() != 0)
         fImpl = std::auto_ptr<Impl>( rhs.fImpl->Copy() );
   }

   /**
      Assignment operator
   */
   GradFunctor & operator = (const GradFunctor & rhs)  {
      GradFunctor copy(rhs);
      // swap auto_ptr by hand
      Impl * p = fImpl.release();
      fImpl.reset(copy.fImpl.release());
      copy.fImpl.reset(p);
      return *this;
   }


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const { return new GradFunctor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const { return fImpl->NDim(); }

private :


   inline double DoEval (const double * x) const {
      return (*fImpl)(x);
   }


   inline double DoDerivative (const double * x, unsigned int icoord  ) const {
      return fImpl->Derivative(x,icoord);
   }

   std::auto_ptr<Impl> fImpl;    // pointer to base grad functor handler


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

   typedef FunctorImpl<IGradientFunctionOneDim>  Impl;
   typedef IGradientFunctionOneDim::BaseFunc ImplBase;


   /**
      Default constructor
   */
   GradFunctor1D ()  : fImpl(0) {}


   /**
      construct from an object with the right signature
      implementing both operator() (double x) and Derivative(double x)
    */
   template <typename Func>
   GradFunctor1D(const Func & f) :
      fImpl(new FunctorHandler<GradFunctor1D,Func>(f) )
   {}


   /**
       construct from a pointer to class and two pointers to member functions, one for
       the function evaluation and the other for the derivative.
       The member functions must take a double as argument and return a double
    */
   template <class PtrObj, typename MemFn, typename GradMemFn>
   GradFunctor1D(const PtrObj& p, MemFn memFn, GradMemFn gradFn)
      : fImpl(new MemGradFunHandler<GradFunctor1D, PtrObj, MemFn, GradMemFn>(p, memFn, gradFn))
   {}



   /**
      construct from two 1D function objects
    */
   template <typename Func, typename GradFunc>
   GradFunctor1D(const Func & f, const GradFunc & g ) :
      fImpl(new FunctorGradHandler<GradFunctor1D,Func, GradFunc>(f, g) )
   {}

   /**
      Destructor (no operations)
   */
   virtual ~GradFunctor1D ()  {}


   /**
      Copy constructor for Functor based on ROOT::Math::IGradFunction
   */
   GradFunctor1D(const GradFunctor1D & rhs) :
      // strange that this is required eventhough Impl is an abstract class
      ImplBase()
   {
      if (rhs.fImpl.get() != 0)
         fImpl = std::auto_ptr<Impl>( rhs.fImpl->Copy()  );
   }


   /**
      Assignment operator
   */
   GradFunctor1D & operator = (const GradFunctor1D & rhs)  {
      GradFunctor1D copy(rhs);
      // swap auto_ptr by hand
      Impl * p = fImpl.release();
      fImpl.reset(copy.fImpl.release());
      copy.fImpl.reset(p);
      return *this;
   }


   // clone of the function handler (use copy-ctor)
   ImplBase * Clone() const { return new GradFunctor1D(*this); }


private :


   inline double DoEval (double x) const {
      return (*fImpl)(x);
   }


   inline double DoDerivative (double x) const {
      return fImpl->Derivative(x);
   }

   std::auto_ptr<Impl> fImpl;    // pointer to base gradient functor handler

};



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Functor */
