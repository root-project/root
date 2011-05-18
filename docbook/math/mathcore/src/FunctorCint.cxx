// implentation of functor for interpreted functions


#define MAKE_CINT_FUNCTOR

#include <Math/Functor.h>
#include <iostream>

#include "TClass.h"
#include "TMethodCall.h"
#include "TError.h"
#include "TInterpreter.h"

namespace ROOT {
namespace Math {

//#if defined(__MAKECINT__) || defined(G__DICTIONARY)

//Functor handler for Cint functions

template<class ParentFunctor>
class FunctorCintHandler : public ParentFunctor::Impl
{
public:
   typedef typename ParentFunctor::Impl ImplFunc;
   typedef typename ImplFunc::BaseFunc BaseFunc;

   // for 1D functor1d
   FunctorCintHandler(void * p, const char * className , const char * methodName, const char * derivName = 0 );

   // for GradFunctor1D
   FunctorCintHandler(void * p1, void * p2 );

   // for Functor
   FunctorCintHandler(void * p, unsigned int dim, const char * className , const char * methodName, const char * derivName = 0 );

   // for GradFunctor
   FunctorCintHandler(void * p1, void * p2, unsigned int dim );

   //copy ctor (need for cloning)
   FunctorCintHandler(const FunctorCintHandler<Functor> & rhs) :
      BaseFunc(),
      fDim(rhs.fDim),
      fPtr(rhs.fPtr),
      fMethodCall(rhs.fMethodCall),
      fMethodCall2(0)
   {}
   FunctorCintHandler(const FunctorCintHandler<GradFunctor> & rhs) :
      BaseFunc(),
      ImplFunc(),
      fDim(rhs.fDim),
      fPtr(rhs.fPtr),
      fMethodCall(rhs.fMethodCall),
      fMethodCall2(rhs.fMethodCall2)
   {}
   FunctorCintHandler(const FunctorCintHandler<Functor1D> & rhs) :
      BaseFunc(),
      fDim(1),
      fPtr(rhs.fPtr),
      fMethodCall(rhs.fMethodCall),
      fMethodCall2(0)
   {}
   FunctorCintHandler(const FunctorCintHandler<GradFunctor1D> & rhs) :
      BaseFunc(),
      ImplFunc(),
      fDim(1),
      fPtr(rhs.fPtr),
      fMethodCall(rhs.fMethodCall),
      fMethodCall2(rhs.fMethodCall2)
   {}

   ~FunctorCintHandler() { //no op (keep pointer to TMethodCall)
   }
   BaseFunc  * Clone() const {  return new FunctorCintHandler(*this);  }

   unsigned int NDim() const {
      return fDim;
   }


private:

   unsigned int fDim;

   void * fPtr; // pointer to callable object

   // function required by interface
   inline double DoEval (double x) const;
   inline double DoDerivative (double x) const;
   inline double DoDerivative (const double * x,unsigned int ipar ) const;
   inline double DoEval (const double * x) const;


   mutable TMethodCall *fMethodCall; // pointer to method call
   mutable TMethodCall *fMethodCall2; // pointer to second method call (for deriv)

};

//implementation of Functor methods
Functor::Functor(void * p, unsigned int dim, const char * className , const char * methodName ) :
   fImpl(new FunctorCintHandler<Functor>(p,dim,className,methodName) )
{}


Functor1D::Functor1D(void * p, const char * className , const char * methodName ) :
   fImpl(new FunctorCintHandler<Functor1D>(p,className,methodName) )
{}

GradFunctor1D::GradFunctor1D(void * p, const char * className , const char * methodName, const char * derivName ) :
   fImpl(new FunctorCintHandler<GradFunctor1D>(p,className,methodName,derivName) )
{}

GradFunctor1D::GradFunctor1D(void * p1, void * p2 ) :
   fImpl(new FunctorCintHandler<GradFunctor1D>(p1,p2) )
{}

GradFunctor::GradFunctor(void * p, unsigned int dim, const char * className , const char * methodName, const char * derivName ) :
   fImpl(new FunctorCintHandler<GradFunctor>(p,dim,className,methodName,derivName) )
{}

GradFunctor::GradFunctor(void * p1, void * p2, unsigned int dim ) :
   fImpl(new FunctorCintHandler<GradFunctor>(p1,p2,dim) )
{}


template<class PF>
FunctorCintHandler<PF>::FunctorCintHandler(void * p, const char * className , const char * methodName, const char * derivMethodName ):    fDim(1)  {

   //constructor for non-grad 1D functions
   fPtr = p;
   fMethodCall2 = 0;


   //std::cout << "creating Cint functor" << std::endl;
   fMethodCall = new TMethodCall();

   if (className == 0) {
      const char *funcname = gCint->Getp2f2funcname((void *) fPtr);

      if (funcname)
         fMethodCall->InitWithPrototype(funcname,"double");
   }
   else {

      TClass *cl = TClass::GetClass(className);

      if (cl) {

         if (methodName)
            fMethodCall->InitWithPrototype(cl,methodName,"double");
         else {
            fMethodCall->InitWithPrototype(cl,"operator()","double");
         }
         if (derivMethodName) {
            fMethodCall2 = new TMethodCall();
            fMethodCall2->InitWithPrototype(cl,derivMethodName,"double");
         }

         if (! fMethodCall->IsValid() ) {
            if (methodName)
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature %s(double ) ",className,methodName);
            else
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature operator() ( double ) ",className);
         }

         if (fMethodCall2 && ! fMethodCall2->IsValid() ) {
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature %s(double ) ",className,derivMethodName);
         }
      } else {
         Error("ROOT::Math::FunctorCintHandler","can not find any class with name %s at the address 0x%lx",className,(Long_t)fPtr);
      }
   }
}


template<class PF>
FunctorCintHandler<PF>::FunctorCintHandler(void * p1, void * p2 ) :    fDim(1) {
   //constructor for grad 1D functions
   fPtr = 0;

   //std::cout << "creating Grad Cint functor" << std::endl;
   fMethodCall = new TMethodCall();
   fMethodCall2 = new TMethodCall();

   const char *funcname = gCint->Getp2f2funcname((void *) p1);

   if (funcname)
      fMethodCall->InitWithPrototype(funcname,"double");

   const char *funcname2 = gCint->Getp2f2funcname((void *) p2);

   if (funcname2)
      fMethodCall2->InitWithPrototype(funcname2,"double");

   if (! fMethodCall->IsValid() ) {
      Error("ROOT::Math::FunctorCintHandler","No function %s found with the signature double () ( double ) at the address 0x%lx",funcname,(Long_t)fPtr);
   }
   if (! fMethodCall2->IsValid() ) {
      Error("ROOT::Math::FunctorCintHandler","No free function %s found with the signature double () ( double )",funcname2);
   }
}

template<class PF>
FunctorCintHandler<PF>::FunctorCintHandler(void * p, unsigned int ndim, const char * className , const char * methodName, const char * derivMethodName ) :    fDim(ndim) {
   // for multi-dim functions
   fPtr = p;
   fMethodCall2 = 0;

   //std::cout << "creating Cint functor" << std::endl;
   fMethodCall = new TMethodCall();

   if (className == 0) {
      const char *funcname = gCint->Getp2f2funcname((void *) fPtr);

      if (funcname)
         fMethodCall->InitWithPrototype(funcname,"const double*");
   }
   else {

      TClass *cl = TClass::GetClass(className);

      if (cl) {

         if (methodName)
            fMethodCall->InitWithPrototype(cl,methodName,"const double*");
         else {
            fMethodCall->InitWithPrototype(cl,"operator()","const double*");
         }
         if (derivMethodName) {
            fMethodCall2 = new TMethodCall();
            fMethodCall2->InitWithPrototype(cl,derivMethodName,"const double*,unsigned int");
         }

         if (! fMethodCall->IsValid() ) {
            if (methodName)
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature %s(const double *) ",className,methodName);
            else
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature operator() (const double * ) ",className);
         }

         if (fMethodCall2 && ! fMethodCall2->IsValid() ) {
               Error("ROOT::Math::FunctorCintHandler","No function found in class %s with the signature %s(const double *, unsigned int ) ",className,derivMethodName);
         }
      } else {
         Error("ROOT::Math::FunctorCintHandler","can not find any class with name %s at the address 0x%lx",className,(Long_t)fPtr);
      }
   }
}



template<class PF>
FunctorCintHandler<PF>::FunctorCintHandler(void * p1, void * p2, unsigned int dim ) :    fDim(dim) {
   //constructor for grad 1D functions

   fPtr = 0;

   //std::cout << "creating Grad Cint functor" << std::endl;
   fMethodCall = new TMethodCall();
   fMethodCall2 = new TMethodCall();

   const char *funcname = gCint->Getp2f2funcname((void *) p1);

   if (funcname)
      fMethodCall->InitWithPrototype(funcname,"const double *");

   const char *funcname2 = gCint->Getp2f2funcname((void *) p2);

   if (funcname2)
      fMethodCall2->InitWithPrototype(funcname2,"const double *,UInt_t");

   if (! fMethodCall->IsValid() ) {
      Error("ROOT::Math::FunctorCintHandler","No free function %s found with the signature double () (const double * ) ",funcname);
   }
   if (! fMethodCall2->IsValid() ) {
      Error("ROOT::Math::FunctorCintHandler","No free function %s found with the signature double () (const double *, unsigned int) ",funcname2);
   }
}

template<class PF>
inline double FunctorCintHandler<PF>::DoEval (double x) const {
   //fArgs[0] = (Long_t)&x;
   fMethodCall->ResetParam();
   fMethodCall->SetParam(x);
   double result = 0;
   // distinguish the case of free functions
   if (fPtr)
      fMethodCall->Execute(fPtr,result);
   else
      fMethodCall->Execute(result);
   //std::cout << "execute doeval for x = " << x << " on " << fPtr << "  result " << result << std::endl;
   return result;
}
template<class PF>
inline double FunctorCintHandler<PF>::DoDerivative (double x) const {
   if (!fMethodCall2) return 0;

   fMethodCall2->ResetParam();
   fMethodCall2->SetParam(x);
   double result = 0;

   // distinguish the case of free functions
   if (fPtr) {
      fMethodCall2->Execute(fPtr,result);
      //std::cout << "execute doDerivative for x = " << x << " on " << fPtr2 << "  result " << result << std::endl;
   }
   else {
      fMethodCall2->Execute(result);
   }
   return result;
}

template<class PF>
inline double FunctorCintHandler<PF>::DoEval (const double *x) const {
   // for multi-dim functions
   fMethodCall->ResetParam();
   Long_t args[1]; // for the address of x
   args[0] = (Long_t)x;
   fMethodCall->SetParamPtrs(args);
   double result = 0;
   // distinguish the case of free functions (fPtr ==0)
   if (fPtr)
      fMethodCall->Execute(fPtr,result);
   else
      fMethodCall->Execute(result);
   //std::cout << "execute doeval for x = " << x << " on " << fPtr << "  result " << result << std::endl;
   return result;
}

template<class PF>
inline double FunctorCintHandler<PF>::DoDerivative (const double *x, unsigned int ipar) const {
   // derivative for multi-dim functions
   //fMethodCall2->ResetParam();
   char * params = Form(" 0x%lx ,  %d", (ULong_t)x, ipar);

   double result = 0;
   // distinguish the case of free functions
   if (fPtr) {
      fMethodCall2->Execute(fPtr,params,result);
   }
   else {
      //std::cout << "execute doDerivative for x = " << *x <<  "  result " << result << std::endl;
      fMethodCall2->Execute(params,result);
   }
   return result;
}

} //end namespace Math

} //end namespace ROOT

#undef MAKE_CINT_FUNCTOR
