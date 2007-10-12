// @(#)root/mathmore:$Id$
// Authors: L. Moneta, M. Slawinska 10/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2007 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Integrator
//
//
#ifndef ROOT_Math_Integrator
#define ROOT_Math_Integrator

#ifndef ROOT_Math_IntegrationTypes
#include "Math/AllIntegrationTypes.h"
#endif

#ifndef ROOT_Math_IFunctionfwd
#include "Math/IFunctionfwd.h"
#endif

#ifndef ROOT_Math_VirtualIntegrator
#include "Math/VirtualIntegrator.h"
#endif





/**

@defgroup Integration Numerical Integration

*/



namespace ROOT {
namespace Math {




/**

User Class for performing numerical integration of a function in one or multi dimension.
It uses the plug-in manager to load advanced numerical integration algorithms from GSL, which reimplements the
algorithms used in the QUADPACK, a numerical integration package written in Fortran.

Various types of adaptive and non-adaptive integration are supported. These include
integration over infinite and semi-infinite ranges and singular integrals.

The integration type is selected using the Integration::type enumeration
in the class constructor.
The default type is adaptive integration with singularity
(ADAPTIVESINGULAR or QAGS in the QUADPACK convention) applying a Gauss-Kronrod 21-point integration rule.
In the case of ADAPTIVE type, the integration rule can also be specified via the
Integration::GKRule. The default rule is 31 points.

In the case of integration over infinite and semi-infinite ranges, the type used is always
ADAPTIVESINGULAR applying a transformation from the original interval into (0,1).

The ADAPTIVESINGULAR type is the most sophicticated type. When performances are
important, it is then recommened to use the NONADAPTIVE type in case of smooth functions or
 ADAPTIVE with a lower Gauss-Kronrod rule.

For detailed description on GSL integration algorithms see the
<A HREF="http://www.gnu.org/software/gsl/manual/gsl-ref_16.html#SEC248">GSL Manual</A>.


  @ingroup Integration

*/


class Integrator {

public:



    // constructors


    /** Constructor of one dimensional Integrator 

       @param type   integration type (adaptive, non-adaptive, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals
    */

    explicit
    Integrator(IntegrationOneDim::Type type = IntegrationOneDim::ADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int size = 1000);
    
    /** Constructor of one dimensional Integrator passing the function

       @param f      integration function (1D interface)
       @param type   integration type (adaptive, non-adaptive, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals
    */
    Integrator(const IGenFunction &f, IntegrationOneDim::Type type = IntegrationOneDim::ADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int size = 1000);

    /** Constructor of multi dimensional Integrator 

       @param type   integration type (adaptive, MC methods, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals
    */
    Integrator(IntegrationMultiDim::Type type , double absTol = 1.E-9, double relTol = 1E-6, unsigned int ncall = 100000);

    /** Constructor of multi dimensional Integrator 

       @param f      integration function (multi-dim interface) 
       @param type   integration type (adaptive, MC methods, etc..)
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals
    */
    Integrator(const IMultiGenFunction &f, IntegrationMultiDim::Type type = IntegrationMultiDim::ADAPTIVE, double absTol = 1.E-9, double relTol = 1E-6, unsigned int ncall = 100000);



#ifdef LATER0 // exclude (pass a rule) 
    /**
       generic constructor for GSL Integrator

       @param type type of integration. The possible types are defined in the Integration::Type enumeration
       @param rule Gauss-Kronrod rule. It is used only for ADAPTIVE::Integration types. The possible rules are defined in the Integration::GKRule enumeration
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals

    */

    Integrator(Integration::Type type, Integration::GKRule rule, double absTol = 1.E-9, double relTol = 1E-6, size_t size = 1000);

    Integrator(const IGenFunction &f, Integration::Type type, Integration::GKRule rule, double absTol = 1.E-9, double relTol = 1E-6, size_t size = 1000);

#endif

   /// destructor (will delete contained pointer)
   virtual ~Integrator();

   // disable copy constructur and assignment operator

private:
     Integrator(const Integrator &);
    Integrator & operator=(const Integrator &);
public:


   // template methods for generic functors

   /**
      method to set the a generic integration function
      
      @param f integration function. The function type must implement the assigment operator, <em>  double  operator() (  double  x ) </em>

   */


   /** 
       set one dimensional function for 1D integration
    */
   void SetFunction(const IGenFunction &f) { 
      fIntegrator->SetFunction(f);
   }
   
   /** 
       set a multi-dimensional function for multi-dim integration
   */
   void SetFunction( const IMultiGenFunction &f) { 
      fIntegrator->SetFunction(f);
   }


    // integration methods using a function

    /**
       evaluate the Integral of a function f over the defined interval (a,b)
       @param f integration function. The function type must implement the mathlib::IGenFunction interface
       @param a lower value of the integration interval
       @param b upper value of the integration interval
    */

   double Integral(const IGenFunction & f, double a, double b) { 
      SetFunction(f); 
      return fIntegrator->Integral(a,b);
   }

   // multi-dim integration routine
   double Integral(const IMultiGenFunction & f, const double * a, const double * b) { 
      SetFunction(f); 
      return fIntegrator->Integral(a,b);
   }


   // integration method using cached function

   /**
      evaluate the Integral over the defined interval (a,b) using the function previously set with Integrator::SetFunction method
      @param a lower value of the integration interval
      @param b upper value of the integration interval
   */

   double Integral(double a, double b) { 
      return fIntegrator->Integral(a,b);
   }

   double Integral(const double * a, const double * b) { 
      return fIntegrator->Integral(a,b);
   }


#ifdef LATER0
   /**
      evaluate the Integral of a function f over the infinite interval (-inf,+inf)
      @param f integration function. The function type must implement the mathlib::IGenFunction interface
   */
   double Integral(const IGenFunction & f);

   /**
      evaluate the Integral of a function f over the semi-infinite interval (a,+inf)
      @param f integration function. The function type must implement the mathlib::IGenFunction interface
      @param a lower value of the integration interval

   */
   double IntegralUp(const IGenFunction & f, double a );

   /**
      evaluate the Integral of a function f over the over the semi-infinite interval (-inf,b)
      @param f integration function. The function type must implement the mathlib::IGenFunction interface
      @param b upper value of the integration interval
   */
   double IntegralLow(const IGenFunction & f, double b );

   /**
      evaluate the Integral of a function f with known singular points over the defined Integral (a,b)
      @param f integration function. The function type must implement the mathlib::IGenFunction interface
      @param pts vector containing both the function singular points and the lower/upper edges of the interval. The vector must have as first element the lower edge of the integration Integral ( \a a) and last element the upper value.

   */
   double Integral(const IGenFunction & f, const std::vector<double> & pts );

   /**
      evaluate the Cauchy principal value of the integral of  a function f over the defined interval (a,b) with a singularity at c 

   */
   double IntegralCauchy(const IGenFunction & f, double a, double b, double c);

#endif


#ifdef LATER0


   /**
      evaluate the Integral over the infinite interval (-inf,+inf) using the function previously set with Integrator::SetFunction method.
   */

   double Integral( );

   /**
      evaluate the Integral of a function f over the semi-infinite interval (a,+inf) using the function previously set with Integrator::SetFunction method.
      @param a lower value of the integration interval
   */
   double IntegralUp(double a );

   /**
      evaluate the Integral of a function f over the over the semi-infinite interval (-inf,b) using the function previously set with Integrator::SetFunction method.
      @param b upper value of the integration interval
   */
   double IntegralLow( double b );

   /**
      evaluate the Integral over the defined interval (a,b) using the function previously set with Integrator::SetFunction method. The function has known singular points.
      @param pts vector containing both the function singular points and the lower/upper edges of the interval. The vector must have as first element the lower edge of the integration Integral ( \a a) and last element the upper value.

   */
   double Integral( const std::vector<double> & pts);

   /**
      evaluate the Cauchy principal value of the integral of  a function f over the defined interval (a,b) with a singularity at c 

   */
   double IntegralCauchy(double a, double b, double c);

#endif

#ifdef LATER
   /**
      return  the Result of the last Integral calculation
   */
   double Result() const;

   /**
      return the estimate of the absolute Error of the last Integral calculation
   */
   double Error() const;

   /**
      return the Error Status of the last Integral calculation
   */
   int Status() const;


   // setter for control Parameters  (getters are not needed so far )

   /**
      set the desired relative Error
   */
   void SetRelTolerance(double relTolerance);


   /**
      set the desired absolute Error
   */
   void SetAbsTolerance(double absTolerance);

   /**
      set the integration rule (Gauss-Kronrod rule).
      The possible rules are defined in the Integration::GKRule enumeration.
      The integration rule can be modified only for ADAPTIVE type integrations
   */
   void SetIntegrationRule(Integration::GKRule );

#endif


private:

   VirtualIntegrator * fIntegrator;

};





} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_Integrator */
