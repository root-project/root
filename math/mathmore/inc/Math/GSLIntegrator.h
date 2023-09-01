// @(#)root/mathmore:$Id$
// Authors: L. Moneta, A. Zsenei   08/2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2004 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  * This library is free software; you can redistribute it and/or      *
  * modify it under the terms of the GNU General Public License        *
  * as published by the Free Software Foundation; either version 2     *
  * of the License, or (at your option) any later version.             *
  *                                                                    *
  * This library is distributed in the hope that it will be useful,    *
  * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
  * General Public License for more details.                           *
  *                                                                    *
  * You should have received a copy of the GNU General Public License  *
  * along with this library (see file COPYING); if not, write          *
  * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
  * 330, Boston, MA 02111-1307 USA, or contact the author.             *
  *                                                                    *
  **********************************************************************/

// Header file for class GSLIntegrator
//
// Created by: Lorenzo Moneta  at Thu Nov 11 14:22:32 2004
//
// Last update: Thu Nov 11 14:22:32 2004
//
#ifndef ROOT_Math_GSLIntegrator
#define ROOT_Math_GSLIntegrator


#include "Math/VirtualIntegrator.h"

#include "Math/IntegrationTypes.h"

#include "Math/IFunctionfwd.h"




#include "Math/GSLFunctionAdapter.h"

#include <vector>



namespace ROOT {
namespace Math {



   class GSLIntegrationWorkspace;
   class GSLFunctionWrapper;

   //_________________________________________________________________________
   /**

   Class for performing numerical integration of a function in one dimension.
   It uses the numerical integration algorithms of GSL, which reimplements the
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
   <A HREF="http://www.gnu.org/software/gsl/manual/html_node/Numerical-Integration.html">GSL Manual</A>.


   @ingroup Integration
   */


   class GSLIntegrator : public VirtualIntegratorOneDim  {

   public:



      // constructors


      /** Default constructor of GSL Integrator for Adaptive Singular integration

      @param absTol desired absolute Error
      @param relTol desired relative Error
      @param size maximum number of sub-intervals
      */

      GSLIntegrator(double absTol = 1.E-9, double relTol = 1E-6, size_t size = 1000);




      /** constructor of GSL Integrator. In the case of Adaptive integration the Gauss-Krond rule of 31 points is used

         @param type type of integration. The possible types are defined in the Integration::Type enumeration
         @param absTol desired absolute Error
         @param relTol desired relative Error
         @param size maximum number of sub-intervals
         */


      GSLIntegrator(const Integration::Type type, double absTol = 1.E-9, double relTol = 1E-6, size_t size = 1000);


      /**
         generic constructor for GSL Integrator

       @param type type of integration. The possible types are defined in the Integration::Type enumeration
       @param rule Gauss-Kronrod rule. It is used only for ADAPTIVE::Integration types. The possible rules are defined in the Integration::GKRule enumeration
       @param absTol desired absolute Error
       @param relTol desired relative Error
       @param size maximum number of sub-intervals

       */

      GSLIntegrator(const Integration::Type type, const Integration::GKRule rule, double absTol = 1.E-9, double relTol = 1E-6, size_t size = 1000);


      /** constructor of GSL Integrator. In the case of Adaptive integration the Gauss-Krond rule of 31 points is used
          This is used by the plug-in manager (need a char * instead of enumerations)

         @param type type of integration. The possible types are defined in the Integration::Type enumeration
         @param rule Gauss-Kronrod rule (from 1 to 6)
         @param absTol desired absolute Error
         @param relTol desired relative Error
         @param size maximum number of sub-intervals
         */
      GSLIntegrator(const char *  type, int rule, double absTol, double relTol, size_t size );

      ~GSLIntegrator() override;
      //~GSLIntegrator();

      // disable copy ctrs
   private:

      GSLIntegrator(const GSLIntegrator &);
      GSLIntegrator & operator=(const GSLIntegrator &);

   public:


         // template methods for generic functors

         /**
         method to set the a generic integration function

          @param f integration function. The function type must implement the assignment operator, <em>  double  operator() (  double  x ) </em>

          */


      void SetFunction(const IGenFunction &f) override;

      /**
         Set function from a GSL pointer function type
       */
      void SetFunction( GSLFuncPointer f, void * p = nullptr);

      // methods using IGenFunction

      /**
         evaluate the Integral of a function f over the defined interval (a,b)
       @param f integration function. The function type must implement the mathlib::IGenFunction interface
       @param a lower value of the integration interval
       @param b upper value of the integration interval
       */

      double Integral(const IGenFunction & f, double a, double b);


      /**
         evaluate the Integral of a function f over the infinite interval (-inf,+inf)
       @param f integration function. The function type must implement the mathlib::IGenFunction interface
       */

      double Integral(const IGenFunction & f);

      /**
       evaluate the Cauchy principal value of the integral of  a previously defined function f over
        the defined interval (a,b) with a singularity at c
        @param a lower interval value
        @param b lower interval value
        @param c singular value of f
        */
      double IntegralCauchy(double a, double b, double c) override;

      /**
       evaluate the Cauchy principal value of the integral of  a function f over the defined interval (a,b)
        with a singularity at c
        @param f integration function. The function type must implement the mathlib::IGenFunction interface
        @param a lower interval value
        @param b lower interval value
        @param c singular value of f
      */
      double IntegralCauchy(const IGenFunction & f, double a, double b, double c);

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

      // evaluate using cached function

      /**
         evaluate the Integral over the defined interval (a,b) using the function previously set with GSLIntegrator::SetFunction method
       @param a lower value of the integration interval
       @param b upper value of the integration interval
       */

      double Integral(double a, double b) override;


      /**
         evaluate the Integral over the infinite interval (-inf,+inf) using the function previously set with GSLIntegrator::SetFunction method.
       */
      double Integral( ) override;

      /**
         evaluate the Integral of a function f over the semi-infinite interval (a,+inf) using the function previously set with GSLIntegrator::SetFunction method.
       @param a lower value of the integration interval
       */
      double IntegralUp(double a ) override;

      /**
         evaluate the Integral of a function f over the over the semi-infinite interval (-inf,b) using the function previously set with GSLIntegrator::SetFunction method.
       @param b upper value of the integration interval
       */
      double IntegralLow( double b ) override;

      /**
         evaluate the Integral over the defined interval (a,b) using the function previously set with GSLIntegrator::SetFunction method. The function has known singular points.
       @param pts vector containing both the function singular points and the lower/upper edges of the interval. The vector must have as first element the lower edge of the integration Integral ( \a a) and last element the upper value.

       */
      double Integral( const std::vector<double> & pts) override;

      // evaluate using free function pointer (same GSL signature)

      /**
         signature for function pointers used by GSL
       */
      //typedef double ( * GSLFuncPointer ) ( double, void * );

      /**
         evaluate the Integral of  of a function f over the defined interval (a,b) passing a free function pointer
       The integration function must be a free function and have a signature consistent with GSL functions:

       <em>double my_function ( double x, void * p ) { ...... } </em>

       This method is the most efficient since no internal adapter to GSL function is created.
       @param f pointer to the integration function
       @param p pointer to the Parameters of the function
       @param a lower value of the integration interval
       @param b upper value of the integration interval

       */
      double Integral(GSLFuncPointer f, void * p, double a, double b);

      /**
         evaluate the Integral  of a function f over the infinite interval (-inf,+inf) passing a free function pointer
       */
      double Integral(GSLFuncPointer f, void * p);

      /**
         evaluate the Integral of a function f over the semi-infinite interval (a,+inf) passing a free function pointer
       */
      double IntegralUp(GSLFuncPointer f, void * p, double a);

      /**
         evaluate the Integral of a function f over the over the semi-infinite interval (-inf,b) passing a free function pointer
       */
      double IntegralLow(GSLFuncPointer f, void * p, double b);

      /**
         evaluate the Integral of a function f with knows singular points over the over a defined interval passing a free function pointer
       */
      double Integral(GSLFuncPointer f, void * p, const std::vector<double> & pts);

      /**
         return  the Result of the last Integral calculation
       */
      double Result() const override;

      /**
         return the estimate of the absolute Error of the last Integral calculation
       */
      double Error() const override;

      /**
         return the Error Status of the last Integral calculation
       */
      int Status() const override;

      /**
          return number of function evaluations in calculating the integral
      */
      int NEval() const override { return fNEval; }

      // setter for control Parameters  (getters are not needed so far )

      /**
         set the desired relative Error
       */
      void SetRelTolerance(double relTolerance) override;


      /**
         set the desired absolute Error
       */
      void SetAbsTolerance(double absTolerance) override;

      /**
         set the integration rule (Gauss-Kronrod rule).
       The possible rules are defined in the Integration::GKRule enumeration.
       The integration rule can be modified only for ADAPTIVE type integrations
       */
      void SetIntegrationRule(Integration::GKRule );

      /// set the options
      void SetOptions(const ROOT::Math::IntegratorOneDimOptions & opt) override;

      ///  get the option used for the integration
      ROOT::Math::IntegratorOneDimOptions Options() const override;

      /// get type name
      IntegrationOneDim::Type GetType() const { return fType; }

      /**
          return the name
      */
      const char * GetTypeName() const;


   protected:

      // internal method to check validity of GSL function pointer
      bool CheckFunction();


   private:

      Integration::Type fType;
      Integration::GKRule fRule;
      double fAbsTol;
      double fRelTol;
      size_t fSize;
      size_t fMaxIntervals;

      // cache Error, Result and Status of integration

      double fResult;
      double fError;
      int fStatus;
      int fNEval;

      // GSLIntegrationAlgorithm * fAlgorithm;

      GSLFunctionWrapper  *     fFunction;
      GSLIntegrationWorkspace * fWorkspace;

   };





} // namespace Math
} // namespace ROOT


#endif /* ROOT_Math_GSLIntegrator */
