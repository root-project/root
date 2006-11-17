// @(#)root/mathcore:$Name:  $:$Id: inc/Math/Functor.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
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

#include <vector>

namespace ROOT { 

namespace Math { 


/** 
   Functor Handler class is responsible for wrapping any other functor and pointer to 
   free C functions.
   It can be created from any function implementing the correct signature 
   corresponding to the requested type
*/ 
template<class ParentFunctor, class Func >
class FunctorHandler : public ParentFunctor::Impl { 

   typedef typename ParentFunctor::Impl Base; 

public: 

   // constructor for 1d functions 
   FunctorHandler(const Func & fun) : fDim(1), fFunc(fun) {}

   // constructor for multi-dimensional functions w/0 NDim()
   FunctorHandler(unsigned int dim, const Func & fun ) :
      fDim(dim),
      fFunc(fun) 
   {}

   // clone of the function handler (use copy-ctor) 
   FunctorHandler * Clone() const { return new FunctorHandler(*this); }

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
   Func fFunc; 

};


/** 
   Functor Handler class for gradient functions where the gradient is provided as 
   an additional functor.
   It can be created from any function implementing the correct signature 
   corresponding to the requested type
*/ 
template<class ParentFunctor, class Func, class GradFunc = Func >
class FunctorGradHandler : public ParentFunctor::Impl { 

   typedef typename ParentFunctor::Impl Base; 

public: 

   // constructor for 1d functions 
   FunctorGradHandler(const Func & fun, const GradFunc & gfun) : 
      fDim(1), 
      fFunc(fun), 
      fGradient(std::vector<GradFunc>(1))  
   {
      fGradient[0] = gfun;
   }

   // constructor for multi-dimensional functions 
   template<class GradFuncIterator> 
   FunctorGradHandler(const Func & fun, GradFuncIterator begin, GradFuncIterator end) :
      fFunc(fun), 
      fGradient(std::vector<GradFunc>(begin,end) ) 
   {
      fDim = fGradient.size();
   }

   // clone of the function handler (use copy-ctor) 
   FunctorGradHandler * Clone() const { return new FunctorGradHandler(*this); }

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
      return fGradient[0](x);
   }  

   inline double DoDerivative (const double * x, unsigned int icoord ) const { 
      return fGradient[icoord](x); 
   }  

   
   unsigned int fDim; 
   Func fFunc; 
   std::vector<GradFunc> fGradient; 

};


/**
   Functor Handler to Wrap pointers to member functions of Base type
*/
template <class ParentFunctor, typename PointerToObj,
          typename PointerToMemFn>
class MemFunHandler : public ParentFunctor::Impl
{
   typedef typename ParentFunctor::Impl Base;
   
public:
   
   /// constructor from a pointer to the class and a pointer to the function
   MemFunHandler(const PointerToObj& pObj, PointerToMemFn pMemFn) 
      : fDim(1), fObj(pObj), fMemFn(pMemFn)
   {}

   /// constructor from a pointer to the class and a pointer to the function
   MemFunHandler(unsigned int dim, const PointerToObj& pObj, PointerToMemFn pMemFn) 
      : fDim(dim), fObj(pObj), fMemFn(pMemFn)
   {}
        
   
   // clone of the function handler (use copy-ctor) 
   MemFunHandler * Clone() const { return new MemFunHandler(*this); }

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
   PointerToObj fObj;
   PointerToMemFn fMemFn; 

};

/**
   Functor Handler to Wrap pointers to member functions of Grad type
*/
template <class ParentFunctor, typename PointerToObj,
          typename PointerToMemFn, typename PointerToGradMemFn>
class MemGradFunHandler : public ParentFunctor::Impl
{
   typedef typename ParentFunctor::Impl Base;
   
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
        
   
   // clone of the function handler (use copy-ctor) 
   MemGradFunHandler * Clone() const { return new MemGradFunHandler(*this); }

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
   PointerToObj fObj;
   PointerToMemFn fMemFn;
   PointerToGradMemFn fGradMemFn;
};
  

#ifdef CHECK
/**
   Functor helper class to check on function dimension
 */
//template<class DimensionType>
struct FunctorDimHelper {
   static void CheckDimension(OneDim  ) {}
};
// template <>
// struct FunctorDimHelper<MultiDim> {
//    static void CheckDimension() {
//       STATIC_CHECK(0==1, Wrong_method_called_Multidimensional_functions);
//    }
// };
/**
   Functor helper class to check on function capability type
 */
//template<class CapabilityType>
struct FunctorCapHelper {
   static void CheckType(Gradient ) {}
};
// template <>
// struct FunctorCapHelper<Base> {
//    static void CheckType() {
//       STATIC_CHECK(0==1, Wrong_method_called_Base_Type_functions);
//    }
// };

#endif

/**
   Functor clas for Multidimensional functions
 */
template<class IFuncType>
class Functor : public IFuncType  { 


public: 

   typedef IFuncType Impl;   
   typedef typename IFuncType::BaseFunc ImplBase;   
//    typedef typename Impl::DimType DimType; 
//    typedef typename Impl::CapType CapType; 
   

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
       construct from a pointer to member function (grad type multi-dim)
    */ 
   template <class PtrObj, typename MemFn, typename GradMemFn>
   Functor(const PtrObj& p, MemFn memFn, GradMemFn gradFn, unsigned int dim = 0)
      : fImpl(new MemGradFunHandler<Functor, PtrObj, MemFn, GradMemFn>(dim, p, memFn, gradFn))
   {}


   /**
      construct from another generic Functor of multi-dimension 
    */
   template <typename Func> 
   Functor( Func f, unsigned int dim ) : 
      fImpl(new FunctorHandler<Functor,Func>(dim,f) )
   {}


   /**
      construct for Analytical Gradient Functions of multi-dimension
    */
   template <typename Func, typename FuncIterator> 
   Functor(Func f, FuncIterator begin, FuncIterator end ) : 
      fImpl(new FunctorGradHandler<Functor,Func>(f, begin, end) )
   {
      // to impl a check  if FuncType provides gradient
      //FunctorCapHelper::CheckType(CapabilityType());
   }



   /** 
      Destructor (no operations)
   */ 
   virtual ~Functor ()  {}  

   /** 
      Copy constructor
   */ 
   Functor(const Functor<ROOT::Math::IMultiGenFunction> & rhs) : 
      Impl()  
   {
      if (rhs.fImpl.get() != 0) 
         fImpl = std::auto_ptr<Impl>( (rhs.fImpl)->Clone() ); 
   } 
   // need a specialization in order to call base classes
   Functor(const Functor<ROOT::Math::IMultiGradFunction> & rhs) : 
      ImplBase(),
      Impl() 
   {
      if (rhs.fImpl.get() != 0) 
         fImpl = std::auto_ptr<Impl>( dynamic_cast<ROOT::Math::IMultiGradFunction *>( (rhs.fImpl)->Clone()) ); 
   } 

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
   IMultiGenFunction * Clone() const { return new Functor(*this); }

   // for multi-dimensional functions
   unsigned int NDim() const { return fImpl->NDim(); } 

private :


   inline double DoEval (const double * x) const { 
      return (*fImpl)(x); 
   }  


   inline double DoDerivative (const double * x, unsigned int icoord  ) const { 
      return fImpl->Derivative(x,icoord);
   }  

   std::auto_ptr<Impl> fImpl; 


}; 


/**
   Functor clas for Onedimensional functions
 */
template<class IFuncType >
class Functor1D : public IFuncType  { 


public: 

   typedef IFuncType           Impl;   
   typedef typename IFuncType::BaseFunc ImplBase; 
//    typedef typename Impl::DimType DimType; 
//    typedef typename Impl::CapType CapType; 
   

   /** 
      Default constructor
   */ 
   Functor1D ()  : fImpl(0) {}  


   /** 
       construct from a pointer to member function (1D type)
    */ 
   template <class PtrObj, typename MemFn>
   Functor1D(const PtrObj& p, MemFn memFn)
      : fImpl(new MemFunHandler<Functor1D, PtrObj, MemFn>(p, memFn))
   {}

   /** 
       construct from a pointer to member function (grad type of 1D)
    */ 
   template <class PtrObj, typename MemFn, typename GradMemFn>
   Functor1D(unsigned int dim, const PtrObj& p, MemFn memFn, GradMemFn gradFn)
      : fImpl(new MemGradFunHandler<Functor1D, PtrObj, MemFn, GradMemFn>(p, memFn, gradFn))
   {}


   /**
      construct from another generic Functor of 1D 
    */
   template <typename Func> 
   Functor1D(Func f) : 
      fImpl(new FunctorHandler<Functor1D,Func>(f) )
   {}


   /**
      construct for Analytical Gradient Functions (of 1 dimension)
    */
   template <typename Func> 
   Functor1D(Func f, Func g ) : 
      fImpl(new FunctorGradHandler<Functor1D,Func>(f, g) )
   {
      // need s to check if provides grad
      //FunctorCapHelper::CheckType(CapabilityType());
   }

   /** 
      Destructor (no operations)
   */ 
   virtual ~Functor1D ()  {}  

   /** 
      Copy constructor
   */ 
   Functor1D(const Functor1D<ROOT::Math::IGenFunction> & rhs) : 
      // strange that this is required eventhough Impl is an abstract class
      Impl()
   {
      if (rhs.fImpl.get() != 0) 
         fImpl = std::auto_ptr<Impl>( (rhs.fImpl)->Clone() ); 
   } 
   Functor1D(const Functor1D<ROOT::Math::IGradFunction> & rhs) : 
      // strange that this is required eventhough Impl is an abstract class
      ImplBase(),          
      Impl()  
   {
      if (rhs.fImpl.get() != 0) 
         fImpl = std::auto_ptr<Impl>( dynamic_cast<ROOT::Math::IGradFunction *>( (rhs.fImpl)->Clone() ) ); 
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
   IGenFunction * Clone() const { return new Functor1D(*this); }


private :

   inline double DoEval (double x) const { 
      return (*fImpl)(x); 
   }  


   inline double DoDerivative (double x) const { 
      return fImpl->Derivative(x);
   }  

   std::auto_ptr<Impl> fImpl; 


}; 



   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Functor */
