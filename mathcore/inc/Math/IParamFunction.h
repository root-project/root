// @(#)root/mathcore:$Id$
// Author: L. Moneta Tue Nov 14 14:20:07 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class IParamFunction

#ifndef ROOT_Math_IParamFunction
#define ROOT_Math_IParamFunction

#ifndef ROOT_Math_IFunction
#include "Math/IFunction.h"
#endif

#ifndef ROOT_Math_IParamFunctionfwd
#include "Math/IParamFunctionfwd.h"
#endif

#ifndef ROOT_Math_Util
#include "Math/Util.h"
#endif


#include <cassert> 


namespace ROOT { 

namespace Math { 


//___________________________________________________________________
/** 
    Documentation for the abstract class IBaseParam.
    It defines the interface for dealing with the function parameters
    This is used only for internal convinience, to avoid redefining the Parameter API  
    for the one and the multi-dim functions. 
    Concrete class should derive from ROOT::Math::IParamFunction and not from this class.  
    
    @ingroup  CppFunctions
*/ 

class IBaseParam  {

public: 


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IBaseParam ()  {}  


   /**
      Access the parameter values
   */
   virtual const double * Parameters() const = 0;

   /**
      Set the parameter values
      @param p vector of doubles containing the parameter values. 

      to be defined:  can user change number of params ? At the moment no. 

   */
   virtual void SetParameters(const double * p ) = 0;

    
   /**
      Return the number of Parameters
   */
   virtual unsigned int NPar() const = 0; 

   /**
      Return the name of the i-th parameter (starting from zero)
      Overwrite if want to avoid the default name ("Par_0, Par_1, ...") 
    */
   virtual std::string ParameterName(unsigned int i) const { 
      assert(i < NPar() ); 
      return "Par_" + Util::ToString(i);
   }


};

//___________________________________________________________________
/** 
   IParamFunction interface (abstract class) describing multi-dimensional parameteric functions
   It is a derived class from ROOT::Math::IBaseFunctionMultiDim and 
   ROOT::Math::IBaseParam

   Provides the interface for evaluating a function passing a coordinate vector and a parameter vector.  

   @ingroup  CppFunctions
*/ 

class IParametricFunctionMultiDim : 
         virtual public IBaseFunctionMultiDim , 
         public IBaseParam {

public: 

   typedef IBaseFunctionMultiDim  BaseFunc; 



   /**
      Evaluate function at a point x and for parameters p.
      This method may be overloaded for better efficiencies by the users 
      For example the method could not change the internal parameters value kept in the derived class. 
      This behaviour is not defined and is left intentionally to the implementation of the derived classes
   */
   virtual double operator() (const double * x, const double *  p ) 
   { 
      SetParameters(p); 
      return operator() (x); 
   }


   using BaseFunc::operator();

}; 

//___________________________________________________________________
/** 
   Specialized IParamFunction interface (abstract class) for one-dimensional parametric functions
   It is a derived class from ROOT::Math::IBaseFunctionOneDim and 
   ROOT::Math::IBaseParam

   @ingroup  CppFunctions
*/ 

class IParametricFunctionOneDim : 
         virtual public IBaseFunctionOneDim, 
         public IBaseParam { 


public: 

   typedef IBaseFunctionOneDim   BaseFunc; 


   using BaseFunc::operator();

   /**
      Evaluate function at a point x and for parameters p.
      This method may be overloaded for better efficiencies by the users 
      For example the method could not change the internal parameters value kept in the derived class. 
      This behaviour is not defined and is left intentionally to the implementation of the derived classes
   */
   virtual double operator() (double x, const double *  p ) 
   { 
      SetParameters(p); 
      return operator() (x); 
   }

   /**
      multidim-like interface
   */
   virtual double operator() (const double * x, const double *  p ) 
   { 
      return operator() (*x, p); 
   }



}; 



//_______________________________________________________________________________
/** 
   Interface (abstract class) for parametric gradient multi-dimensional functions providing 
   in addition to function evaluation and gradient with respect the coordinates 
   also the gradient with respect to the parameters, via the method ParameterGradient. 

   It is a derived class from ROOT::Math::IParametricFunctionMultiDim and 
   ROOT::Math::IGradientFunctionMultiDim. 
   Virtual inheritance is used since IBaseFunctionMultiDim is the common base of both 
   ROOT::Math::IParametricFunctionMultiDim and ROOT::Math::IGradientFunctionMultiDim. 
   The pure private virtual method DoParameterGradient must be implemented by the derived classes 
   in addition to those inherited by the base abstract classes. 

   @ingroup  CppFunctions
*/ 

class IParametricGradFunctionMultiDim : 
         public IParametricFunctionMultiDim, 
         public IGradientFunctionMultiDim   {

public: 

   typedef IParametricFunctionMultiDim                BaseParamFunc; 
   typedef IGradientFunctionMultiDim                  BaseGradFunc; 
   typedef IParametricFunctionMultiDim::BaseFunc  BaseFunc; 


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IParametricGradFunctionMultiDim ()  {}  



   using BaseParamFunc::operator();

   /**
      Evaluate the all the derivatives (gradient vector) of the function with respect to the parameters at a point x.
      It is optional to be implemented by the derived classes for better efficiency
   */
   virtual void ParameterGradient(const double * x , double * grad ) const { 
      unsigned int npar = NPar(); 
      for (unsigned int ipar  = 0; ipar < npar; ++ipar) 
         grad[ipar] = DoParameterDerivative(x,ipar); 
   } 

   /**
      Evaluate the partial derivative w.r.t a parameter ipar
    */
   double ParameterDerivative(const double * x, unsigned int ipar = 0) const { 
      return DoParameterDerivative(x, ipar); 
   }  



private: 



   /**
      Evaluate the partial derivative w.r.t a parameter ipar , to be implemented by the derived classes
    */
   virtual double DoParameterDerivative(const double * x, unsigned int ipar) const = 0;  


};

//_______________________________________________________________________________
/** 
   Interface (abstract class) for parametric one-dimensional gradient functions providing 
   in addition to function evaluation and derivative with respect the coordinates 
   also the gradient with respect to the parameters, via the method ParameterGradient. 

   It is a derived class from ROOT::Math::IParametricFunctionOneDim and 
   ROOT::Math::IGradientFunctionOneDim. 
   Virtual inheritance is used since IBaseFunctionOneDim is the common base of both 
   ROOT::Math::IParametricFunctionOneDim and ROOT::Math::IGradientFunctionOneDim. 
   The pure private virtual method DoParameterGradient must be implemented by the derived classes 
   in addition to those inherited by the base abstract classes. 

   @ingroup  CppFunctions
*/ 

class IParametricGradFunctionOneDim : 
         public IParametricFunctionOneDim, 
         public IGradientFunctionOneDim   {

public: 

   typedef IParametricFunctionOneDim            BaseParamFunc; 
   typedef IGradientFunctionOneDim              BaseGradFunc; 
   typedef IParametricFunctionOneDim::BaseFunc  BaseFunc; 


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IParametricGradFunctionOneDim ()  {}  


   using BaseParamFunc::operator();

   /**
      Evaluate the derivatives of the function with respect to the parameters at a point x.
      It is optional to be implemented by the derived classes for better efficiency if needed
   */
   virtual void ParameterGradient(double x , double * grad ) const { 
      unsigned int npar = NPar(); 
      for (unsigned int ipar  = 0; ipar < npar; ++ipar) 
         grad[ipar] = DoParameterDerivative(x,ipar); 
   } 

   /**
      Compatibility interface with multi-dimensional functions 
   */
   void ParameterGradient(const double * x , double * grad ) const { 
      ParameterGradient(*x, grad); 
   } 

   /**
      Partial derivative with respect a parameter
    */
   double ParameterDerivative(double x, unsigned int ipar = 0) const { 
      return DoParameterDerivative(x, ipar); 
   }

   /**
      Partial derivative with respect a parameter
      Compatibility interface with multi-dimensional functions 
   */
   double ParameterDerivative(const double * x, unsigned int ipar = 0) const { 
      return DoParameterDerivative(*x, ipar); 
   }



private: 


   /**
      Evaluate the gradient, to be implemented by the derived classes
    */
   virtual double DoParameterDerivative(double x, unsigned int ipar ) const = 0;  


};




   } // end namespace Math

} // end namespace ROOT



#endif /* ROOT_Math_IParamFunction */
