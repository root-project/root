// @(#)root/mathcore:$Id$
// Author: Magdalena Slawinska 10/2007


/**********************************************************************
 *                                                                    *
 * Copyright (c) 2007  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class Minimizer

#ifndef ROOT_Math_VirtualIntegrator
#define ROOT_Math_VirtualIntegrator

#include "Math/IFunctionfwd.h"

#include "Math/Error.h"

#include "Math/IntegratorOptions.h"


#include <vector>


namespace ROOT {
namespace Math {

//___________________________________________________________________
/**
   Abstract class for all numerical integration methods (1D and multi-dim)
   Interface defining the common methods for the
   numerical integrator classes of one and multi dimensions
   The derived class VirtualIntegratorOneDim defines the methods
   for one-dimensional integration.
   The derived class VirtualIntegratorMultiDim defines the method for
   multi-dimensional integration.
   The concrete classes for one dimension (e.g. GSLIntegrator) or
   multi-dimension (e.g. GSLMCIntegrator) can be created using the
   plug-in manager.
   Users should not use directly this class but the concrete classes ROOT::Math::IntegratorOneDim or
   ROOT::Math::IntegratorMultiDim

   @ingroup  Integration

*/
class VirtualIntegrator{

public:

   /// destructor: no operation
   virtual ~VirtualIntegrator() {}

   /**
      set the desired relative Error
   */
   virtual void SetRelTolerance(double ) = 0;

   /**
      set the desired absolute Error
   */
   virtual void SetAbsTolerance(double ) = 0;

   /**
      return  the Result of the last Integral calculation
   */
   virtual double Result() const = 0;

   /**
      return the estimate of the absolute Error of the last Integral calculation
   */
   virtual double Error() const = 0;

   /**
      return the Error Status of the last Integral calculation
   */
   virtual int Status() const = 0;

   /**
      return number of function evaluations in calculating the integral
      (if integrator do not implement this function returns -1)
   */
   virtual int NEval() const { return -1; }

};


//___________________________________________________________________
/**
   Interface (abstract) class for 1D numerical integration
   It must be implemented by the concrete Integrator classes like
   ROOT::Math::GSLIntegrator.
   Plug-in's exist in ROOT to be able to instantiate the derived classes via the
   plug-in manager.
   Users should not use directly this class but the concrete classes ROOT::Math::IntegratorOneDim.

   @ingroup  Integration

*/
class VirtualIntegratorOneDim : public VirtualIntegrator {

public:

   /// destructor: no operation
   virtual ~VirtualIntegratorOneDim() {}

   /// evaluate integral
   virtual double Integral(double a, double b) = 0;

   /// set integration function
   virtual void SetFunction(const IGenFunction &) = 0;

   /// evaluate un-defined  integral (between -inf, + inf)
   virtual double Integral() = 0;

   /// evaluate integral over the (a, +inf)
   virtual double IntegralUp(double a) = 0;

   /// evaluate integral over the (-inf, b)
   virtual double IntegralLow(double b) = 0;

   /// evaluate integral with singular points
   virtual double Integral( const std::vector<double> & pts) = 0;

   /// evaluate Cauchy integral
   virtual double IntegralCauchy(double a, double b, double c) = 0;

   ///  get the option used for the integration
   /// must be implemented by derived class
   virtual ROOT::Math::IntegratorOneDimOptions Options() const = 0;

   /// return type of integrator
   virtual ROOT::Math::IntegrationOneDim::Type Type() const {
      return Options().IntegratorType();
   }

   /// set the options
   /// (should be re-implemented by derived classes -if more options than tolerance exist
   virtual void SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt) {
      SetRelTolerance(opt.RelTolerance() );
      SetAbsTolerance(opt.AbsTolerance() );
   }

};


//___________________________________________________________________
/**
   Interface (abstract) class for multi numerical integration
   It must be implemented by the concrete Integrator classes like
   ROOT::Math::GSLMCIntegrator.
   Plug-in's exist in ROOT to be able to instantiate the derived classes via the
   plug-in manager.
   Users should not use directly this class but the concrete classes ROOT::Math::IntegratorMultiDim.

   @ingroup  Integration

*/
class VirtualIntegratorMultiDim : public VirtualIntegrator {

public:

   /// destructor: no operation
   virtual ~VirtualIntegratorMultiDim() {}

   /// evaluate multi-dim integral
   virtual double Integral(const double*, const double*)  = 0;

   /// setting a multi-dim function
   virtual void SetFunction(const IMultiGenFunction &)  = 0;

   ///  get the option used for the integration
   /// implement by derived class otherwise return default ones
   virtual ROOT::Math::IntegratorMultiDimOptions Options() const = 0;

   /// return type of integrator
   virtual ROOT::Math::IntegrationMultiDim::Type Type() const {
      return Options().IntegratorType();
   }

   /// set the options (if needed must be re-implemented by derived classes)
   virtual void SetOptions(const ROOT::Math::IntegratorMultiDimOptions & opt) {
      SetRelTolerance(opt.RelTolerance() );
      SetAbsTolerance(opt.AbsTolerance() );
   }

};


}//namespace Math
}//namespace ROOT


#endif /* ROOT_Math_VirtualIntegrator */
