// @(#)root/mathcore:$Id$
// Author: L. Moneta Mon Nov 13 15:58:13 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for Functor classes.
// design is inspired by the Loki Functor

#ifndef ROOT_Math_ParamFunctor
#define ROOT_Math_ParamFunctor

// #ifndef ROOT_Math_IFunction
// #include "Math/IFunction.h"
// #endif

// #ifndef Root_Math_StaticCheck
// #include "Math/StaticCheck.h"
// #endif

//#ifndef __CINT__
//#include <memory>

#include <vector>
#include <iostream>

namespace ROOT {

namespace Math {


/** class defining the signature for multi-dim parametric functions

   @ingroup  ParamFunctor_int
 */

class ParamFunctionBase {
  public:
   virtual ~ParamFunctionBase() {}
//   virtual double operator() (const double * x, const double *p) const = 0;
   virtual double operator() (double * x, double *p) = 0;
   virtual ParamFunctionBase * Clone() const = 0;
};



/**
   ParamFunctor Handler class is responsible for wrapping any other functor and pointer to
   free C functions.
   It can be created from any function implementing the correct signature
   corresponding to the requested type

   @ingroup  ParamFunctor_int

*/
#ifndef __CINT__

template<class ParentFunctor, class Func >
class ParamFunctorHandler : public ParentFunctor::Impl {

   typedef typename ParentFunctor::Impl Base;

public:

   // constructor
   ParamFunctorHandler(const Func & fun) : fFunc(fun) {}


   virtual ~ParamFunctorHandler() {}


   // for 1D functions
   inline double operator() (double x, double *p)  {
      return fFunc(x,p);
   }
//    inline double operator() (double x, const double *p) const {
//       return fFunc(x,p);
//    }
   // for multi-dimensional functions
//    inline double operator() (const double * x, const double *p) const {
//       return fFunc(x,p);
//    }
   inline double operator() (double * x, double *p)  {
      return FuncEvaluator<Func>::Eval(fFunc,x,p);
   }

   // clone (use same pointer)
   ParamFunctorHandler  * Clone() const {
      return new ParamFunctorHandler(fFunc);
   }


private :

   Func fFunc;

   // structure to distinguish pointer types
   template <typename F> struct FuncEvaluator {
      inline static double Eval( F & f, double *x, double * p) {
         return f(x,p);
      }
   };
   template <typename F> struct FuncEvaluator<F*> {
      inline static double Eval( F * f, double *x, double * p) {
         return (*f)(x,p);
      }
   };
   template <typename F> struct FuncEvaluator<F* const> {
      inline static double Eval( const F * f, double *x, double * p) {
         return (*f)(x,p);
      }
   };
   // need maybe also volatile ?
};


#if defined(__MAKECINT__) || defined(G__DICTIONARY)
// needed since CINT initialize it with TRootIOCtor
//class TRootIOCtor;
template<class ParentFunctor>
class ParamFunctorHandler<ParentFunctor,TRootIOCtor *> : public ParentFunctor::Impl
{
public:

   ParamFunctorHandler(TRootIOCtor  *) {}

   double operator() (double *, double * )  { return 0; }
   // clone (use same pointer)
   ParamFunctorHandler  * Clone() const {
      return 0;
   }

};
#endif


/**
   ParamFunctor Handler to Wrap pointers to member functions

   @ingroup  ParamFunctor_int
*/
template <class ParentFunctor, typename PointerToObj,
          typename PointerToMemFn>
class ParamMemFunHandler : public ParentFunctor::Impl
{
   typedef typename ParentFunctor::Impl Base;


public:

   /// constructor from a pointer to the class and a pointer to the function
   ParamMemFunHandler(const PointerToObj& pObj, PointerToMemFn pMemFn)
      : fObj(pObj), fMemFn(pMemFn)
   {}

   virtual ~ParamMemFunHandler() {}

//    inline double operator() (double x, const double * p) const {
//       return ((*fObj).*fMemFn)(x,p);
//    }

   inline double operator() (double x, double * p)  {
      return ((*fObj).*fMemFn)(x,p);
   }

//    inline double operator() (const double * x, const double * p) const {
//       return ((*fObj).*fMemFn)(x,p);
//    }

   inline double operator() (double * x, double * p)  {
      return ((*fObj).*fMemFn)(x,p);
   }

   // clone (use same pointer)
   ParamMemFunHandler  * Clone() const {
      return new ParamMemFunHandler(fObj, fMemFn);
   }


private :
   ParamMemFunHandler(const ParamMemFunHandler&); // Not implemented
   ParamMemFunHandler& operator=(const ParamMemFunHandler&); // Not implemented

   PointerToObj fObj;
   PointerToMemFn fMemFn;

};

#endif



/**
   Param Functor class for Multidimensional functions.
   It is used to wrap in a very simple and convenient way
   any other C++ callable object (implemention double operator( const double *, const double * ) )
   or a member function with the correct signature,
   like Foo::EvalPar(const double *, const double *)

   @ingroup  ParamFunc

 */


class ParamFunctor   {


public:

   typedef  ParamFunctionBase Impl;


   /**
      Default constructor
   */
   ParamFunctor ()  : fImpl(0) {}


   /**
       construct from a pointer to member function (multi-dim type)
    */
   template <class PtrObj, typename MemFn>
   ParamFunctor(const PtrObj& p, MemFn memFn)
      : fImpl(new ParamMemFunHandler<ParamFunctor, PtrObj, MemFn>(p, memFn))
   {}



   /**
      construct from another generic Functor of multi-dimension
    */
   template <typename Func>
   explicit ParamFunctor( const Func & f) :
      fImpl(new ParamFunctorHandler<ParamFunctor,Func>(f) )
   {}



   // specialization used in TF1
   typedef double (* FreeFunc ) (double * , double *);
   ParamFunctor(FreeFunc f) :
      fImpl(new ParamFunctorHandler<ParamFunctor,FreeFunc>(f) )
   {
   }


   /**
      Destructor (no operations)
   */
   virtual ~ParamFunctor ()  {
      if (fImpl) delete fImpl;
   }

   /**
      Copy constructor
   */
   ParamFunctor(const ParamFunctor & rhs) :
      fImpl(0)
   {
//       if (rhs.fImpl.get() != 0)
//          fImpl = std::auto_ptr<Impl>( (rhs.fImpl)->Clone() );
      if (rhs.fImpl != 0)  fImpl = rhs.fImpl->Clone();
   }

   /**
      Assignment operator
   */
   ParamFunctor & operator = (const ParamFunctor & rhs)  {
//      ParamFunctor copy(rhs);
      // swap auto_ptr by hand
//       Impl * p = fImpl.release();
//       fImpl.reset(copy.fImpl.release());
//       copy.fImpl.reset(p);

      if(this != &rhs) {
         if (fImpl) delete fImpl;
         fImpl = 0;
         if (rhs.fImpl != 0)
            fImpl = rhs.fImpl->Clone();
      }
      return *this;
   }

   void * GetImpl() { return (void *) fImpl; }


   double operator() (double * x, double * p)  {
      return (*fImpl)(x,p);
   }



   bool Empty() const { return fImpl == 0; }


   void SetFunction(Impl * f) {
      fImpl = f;
   }

private :


   //std::auto_ptr<Impl> fImpl;
   Impl * fImpl;


};



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_ParamFunctor */
