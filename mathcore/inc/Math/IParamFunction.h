// @(#)root/mathcore:$Name:  $:$Id: inc/Math/IParamFunction.h,v 1.0 2006/01/01 12:00:00 moneta Exp $
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

#ifndef ROOT_Math_Util
#include "Math/Util.h"
#endif

#include <vector>

#include <cassert> 


namespace ROOT { 

   namespace Math { 


/** 
   IParamFunction interface for generic parameteric function
   It is a derived class from IFunction
   @ingroup  CppFunctions
*/ 

template <class DimensionType>
class IParamFunctionBase : virtual public IBaseFunction<DimensionType> {

public: 

   typedef IBaseFunction<DimensionType> BaseFunc; 

//    /// default constructor (needed to initialize parent classes)
//    IParamFunctionBase() : 
//       BaseFunc()
//    {}


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IParamFunctionBase ()  {}  


   /**
      Access the parameter values
   */
   virtual const double * Parameters() const = 0;

   // set params values (can user change number of params ? ) 
   /**
      Set the parameter values
      @param p vector of doubles containing the parameter values. 
   */
   virtual void SetParameters(const double * p ) = 0;

//    /**
//       Set the parameters values using an iterator 
//     */
//    template <class ParIterator> 
//    bool SetParameters(ParIterator begin, ParIterator end) { 
//       std::vector<double> p(begin.end);
//       if (p.size()!= NPar() ) return false; 
//       SetParameters(&p.front());
//    }
    
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

   using BaseFunc::operator();

};

/** 
   IParamFunction interface for generic parameteric function
   It is a derived class from IFunction
   @ingroup  CppFunctions
*/ 
template<class DimensionType = MultiDim> 
class IParamFunction : public IParamFunctionBase<DimensionType> {

public: 

   typedef IParamFunctionBase<DimensionType> BaseParamFunc;
   typedef typename IParamFunctionBase<DimensionType>::BaseFunc BaseFunc; 

   /// default constructor (needed to initialize parent classes)
//    IParamFunction() : 
//       BaseParamFunc() 
//    {}


   // user may re-implement this for better efficiency
   // this method is NOT required to  change internal values of parameters. confusing ?? 
   /**
      Evaluate function at a point x and for parameters p.
      This method mey be needed for better efficiencies when for each function evaluation the parameters are changed.
   */
   virtual double operator() (const double * x, const double *  p ) 
   { 
      SetParameters(p); 
      return operator() (x); 
   }

   using BaseParamFunc::SetParameters;

   using BaseParamFunc::operator();

}; 
/** 
   IParamFunction interface for one-dimensional function
   @ingroup  CppFunctions
*/ 
template<> 
class IParamFunction<ROOT::Math::OneDim> : public IParamFunctionBase<ROOT::Math::OneDim> {

public: 

   typedef IParamFunctionBase<ROOT::Math::OneDim>           BaseParamFunc;
   typedef IParamFunctionBase<ROOT::Math::OneDim>::BaseFunc BaseFunc; 

   /// default constructor (needed to initialize parent classes)
//    IParamFunction() : 
//       BaseParamFunc() 
//    {}

   using BaseParamFunc::operator();

   // user may re-implement this for better efficiency
   // this method is NOT required to  change internal values of parameters. confusing ?? 
   /**
      Evaluate function at a point x and for parameters p.
      This method mey be needed for better efficiencies when for each function evaluation the parameters are changed.
   */
   virtual double operator() (double x, const double *  p ) 
   { 
      SetParameters(p); 
      return operator() (x); 
   }



}; 




/** 
   IParamGradFunction interface for parametric functions providing 
   the gradient
   @ingroup  CppFunctions
*/ 
template<class DimensionType=MultiDim> 
class IParamGradFunction : 
         public IParamFunction<DimensionType>, 
         public IGradientFunction<DimensionType>   {

public: 

   typedef IParamFunction<DimensionType>                     BaseParamFunc; 
   typedef IGradientFunction<DimensionType>                  BaseGradFunc; 
   typedef typename IParamFunction<DimensionType>::BaseFunc  BaseFunc; 

   /// default constructor (needed to initialize parent classes)
//    IParamGradFunction() :
//       BaseParamFunc(),  
//       BaseGradFunc()  
//    {}


   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IParamGradFunction ()  {}  


   //using BaseFunc::operator();
   using BaseParamFunc::operator();

   /**
      Evaluate the derivatives of the function with respect to the parameters at a point x.
      It is optional to be implemented by the derived classes 
   */
   void ParameterGradient(const double * x , double * grad ) const { 
      return DoParameterGradient(x, grad); 
   } 


private: 


//    /**
//       Set the parameter values
//       @param p vector of doubles containing the parameter values. 
//    */
//    virtual void DoSetParameters(const double * p ) = 0;

   /**
      Evaluate the gradient, to be implemented by the derived classes
    */
   virtual void DoParameterGradient(const double * x, double * grad) const = 0;  


};

/** 
   IParamGradFunction interface for one-dimensional function

   @ingroup  CppFunctions
*/ 
template<> 
class IParamGradFunction<ROOT::Math::OneDim> : 
         public IParamFunction<ROOT::Math::OneDim>, 
         public IGradientFunction<ROOT::Math::OneDim>   {

public: 

   typedef IParamFunction<ROOT::Math::OneDim>            BaseParamFunc; 
   typedef IGradientFunction<ROOT::Math::OneDim>         BaseGradFunc; 
   typedef IParamFunction<ROOT::Math::OneDim>::BaseFunc  BaseFunc; 

   /// default constructor (needed to initialize parent classes)
//    IParamGradFunction() :
//       BaseParamFunc(),  
//       BaseGradFunc()  
//    {}

   /** 
      Virtual Destructor (no operations)
   */ 
   virtual ~IParamGradFunction ()  {}  


   //using BaseFunc::operator();
   using BaseParamFunc::operator();

   /**
      Evaluate the derivatives of the function with respect to the parameters at a point x.
      It is optional to be implemented by the derived classes 
   */
   void ParameterGradient(double x , double * grad ) const { 
      return DoParameterGradient(x, grad); 
   } 


private: 


//    /**
//       Set the parameter values
//       @param p vector of doubles containing the parameter values. 
//    */
//    virtual void DoSetParameters(const double * p ) = 0;

   /**
      Evaluate the gradient, to be implemented by the derived classes
    */
   virtual void DoParameterGradient(double x, double * grad) const = 0;  


};




   } // end namespace Math

} // end namespace ROOT



#endif /* ROOT_Math_IParamFunction */
