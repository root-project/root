// @(#)root/mathcore:$Name:  $:$Id: IFunction.h,v 1.1 2006/11/17 18:18:47 moneta Exp $
// Authors: L. Moneta, A. Zsenei   08/2005 


// Header file for funciton interfaces 
// 
// Generic interface for one or  multi-dimensional functions
//
// Created by: Lorenzo Moneta  : Wed Nov 13 2006
// 
// 
#ifndef ROOT_Math_IFunction
#define ROOT_Math_IFunction

/** 
@defgroup CppFunctions Function Classes and Interfaces
*/

//typedefs and tags definitions 
#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif


namespace ROOT {
namespace Math {



   /** 
       Interface for generic functions objects: 
       A template parameter, DimensionType specify the DimensionType  which can be
       single-dimension or multi-dimension onother parameter specify the 
       function capabilities. 
       Default case is function with multidimension and default capability 
       (no gradient calculation)
       - 
       @ingroup  CppFunctions
   */
   template<class DimensionType = MultiDim> 
   class IBaseFunction {
           
   public: 

      typedef  IBaseFunction BaseFunc; 


      IBaseFunction() {}

      /**
         virtual destructor 
       */
      virtual ~IBaseFunction() {}

      /** 
          Clone a function. 
          Each derived class will implement his version of the Clone method
      */
      virtual IBaseFunction * Clone() const = 0;  

      /**
         Retrieve the dimension of the function
       */
      virtual unsigned int NDim() const = 0; 

      /** 
          Evaluate the function at a point x[]. 
          Use the  a pure virtual private method Evaluate which must be implemented by sub-classes
      */      
      double operator() (const double* x) const { 
         return DoEval(x); 
      }

#ifdef LATER
      /**
         Template method to eveluate the function using the begin of an iterator
         User is responsible to provide correct size for the iterator
      */
      template <class Iterator>
      double operator() (const Iterator it ) const { 
         return DoEval( &(*it) ); 
      }
#endif


   private: 

      // use private virtual inheritance 

      /**
         Implementation of the evaluation function. Must be implemented by derived classes
      */
      virtual double DoEval(const double * x) const = 0; 


  }; 


   /** 
       Specialized Interface for one-dimensional generic functions with 
       minimal capabilities (no gradient) 
        
       @ingroup  CppFunctions
   */
   template <>
   class IBaseFunction<OneDim> { 

   public: 

      typedef  IBaseFunction BaseFunc; 

      IBaseFunction() {}

      /**
         virtual destructor 
       */
      virtual ~IBaseFunction() {}

      /** 
          Clone a function. 
          Each derived class will implement his version of the provate DoClone method
      */
      virtual IBaseFunction * Clone() const = 0;  

      /** 
          Evaluate the function at a point x 
          Use the  a pure virtual private method DoEval which must be implemented by sub-classes
      */      
      double operator() (double x) const { 
         return DoEval(x); 
      }

      /** 
          Evaluate the function at a point x[]. 
          Compatible method with multi-dimensional functions
      */      
      double operator() (const double * x) const { 
         return DoEval(*x); 
      }



   private: 

      // use private virtual inheritance 

      /// implementation of the evaluation function. Must be implemented by derived classes
      virtual double DoEval(double x) const = 0; 

   };


//-------- GRAD  functions---------------------------
   /**
      Gradient interface defining the signature for the functions to calculate the gradient

      @ingroup  CppFunctions
    */
   template <class DimensionType = MultiDim> 
   class IGradient { 

      public: 

      /// virual destructor 
      virtual ~IGradient() {}

      /** 
          Evaluate all the vector of function derivatives (gradient)  at a point x.
          Derived classes must re-implement it if more efficient than evaluting one at a time
      */
      virtual void  Gradient(const double *x, double * grad) const = 0; 

      /**
         Return the partial derivative with respect to the passed coordinate 
      */
      double Derivative(const double * x, unsigned int icoord = 0) const  { 
         return DoDerivative(x, icoord); 
      }

 
      /** 
          Optimized method to evaluate at the same time the function value and derivative at a point x.
          Often both value and derivatives are needed and it is often more efficient to compute them at the same time.
          Derived class should implement this method if performances play an important role and if it is faster to 
          evaluate value and derivative at the same time
       
      */
      virtual void FdF (const double * x, double & f, double * df) const  = 0; 


   private: 


      /**
         function to evaluate the derivative with respect each coordinate. To be implemented by the derived class 
      */ 
      virtual  double  DoDerivative(const double * x, unsigned int icoord ) const = 0; 

   };

   /**
      Specialized Gradient interface for one dimensional functions

      @ingroup  CppFunctions
    */
   template <> 
   class IGradient<OneDim> { 

   public: 

      /// virual destructor 
      virtual ~IGradient() {}

      /**
         Return the derivative of the funcition at a point x 
         Use the private method DoDerivative 
      */
      double Derivative(double x ) const  { 
         return DoDerivative(x ); 
      }

 
      /** 
          Optimized method to evaluate at the same time the function value and derivative at a point x.
          Often both value and derivatives are needed and it is often more efficient to compute them at the same time.
          Derived class should implement this method if performances play an important role and if it is faster to 
          evaluate value and derivative at the same time
       
      */
      virtual void FdF (double x, double & f, double & df) const = 0; 

      


   private: 


      /**
         function to evaluate the derivative with respect each coordinate. To be implemented by the derived class 
      */ 
      virtual  double  DoDerivative(double x ) const = 0; 

   };

/** 
   Interface for multi-dimensional functions providing a gradient calculation. 
   A method ROOT::Math::IFunction::Gradient provides the full gradient vector while 
   ROOT::Math::IFunction::Derivative provides the partial derivatives. 
   The latter bust be implemented (using ROOT::Math::IFunction::Derivative by the derived classes, 
   while the former can be overloaded if for the particular function can be implemented more efficiently.  
   @ingroup  CppFunctions
*/ 
   template<class DimensionType = MultiDim> 
   class IGradientFunction : 
      virtual public IBaseFunction<DimensionType> , 
      public IGradient<DimensionType> { 
     

   public: 

      typedef IBaseFunction<DimensionType> BaseFunc; 
      typedef IGradient<DimensionType> BaseGrad; 

//       // need default constructor with initialization of parent classes
//       IGradientFunction() : 
//           BaseFunc(), 
//           BaseGrad()
//       {}


      /** 
          Virtual Destructor (no operations)
      */ 
      virtual ~IGradientFunction () {}

      /** 
          Evaluate all the vector of function derivatives (gradient)  at a point x.
          Derived classes must re-implement it if more efficient than evaluting one at a time
      */
      virtual void  Gradient(const double *x, double * grad) const { 
         unsigned int ndim = NDim(); 
         for (unsigned int icoord  = 0; icoord < ndim; ++icoord) 
            grad[icoord] = BaseGrad::Derivative(x,icoord); 
      }
      using  BaseFunc::NDim;

      /**
         Return the partial derivative with respect to the passed coordinate 
      */
//       double Derivative(const double * x, unsigned int icoord = 0) const  { 
//          if (icoord < NDim() ) 
//             return DoDerivative(x, icoord); 
//          else 
//             return 0; 
//       }

 
      /** 
          Optimized method to evaluate at the same time the function value and derivative at a point x.
          Often both value and derivatives are needed and it is often more efficient to compute them at the same time.
          Derived class should implement this method if performances play an important role and if it is faster to 
          evaluate value and derivative at the same time
       
      */
      virtual void FdF (const double * x, double & f, double * df) const { 
         f = BaseFunc::operator()(x); 
         Gradient(x,df);
      }


   }; 


/** 
   Specialized Interface for one-dimensional functions providing a gradient calculation. 
   A method ROOT::Math::IFunction::Gradient provides the full gradient vector while 
   ROOT::Math::IFunction::Derivative provides the partial derivatives. 
   The latter bust be implemented (using ROOT::Math::IFunction::Derivative by the derived classes, 
   while the former can be overloaded if for the particular function can be implemented more efficiently.  
   @ingroup  CppFunctions
*/ 
   template <>
   class IGradientFunction<OneDim> : 
      virtual public IBaseFunction<OneDim> , 
      public IGradient<OneDim> { 
     

   public: 

      typedef IBaseFunction<ROOT::Math::OneDim> BaseFunc; 
      typedef IGradient<ROOT::Math::OneDim> BaseGrad; 

//       // need default constructor with initialization of parent classes
//       IGradientFunction() : 
//           BaseFunc(), 
//           BaseGrad()
//       {}


      /** 
          Virtual Destructor (no operations)
      */ 
      virtual ~IGradientFunction () {}


      /** 
          Optimized method to evaluate at the same time the function value and derivative at a point x.
          Often both value and derivatives are needed and it is often more efficient to compute them at the same time.
          Derived class should implement this method if performances play an important role and if it is faster to 
          evaluate value and derivative at the same time
       
      */
      virtual void FdF (double x, double & f, double & df) const { 
         f = operator()(x); 
         df = Derivative(x);
      }



   }; 



 

} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_IGenFunction */
