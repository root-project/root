// @(#)root/mathcore:$Id$
// Author: L. Moneta Fri Aug 15 2008

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2008  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

#ifndef ROOT_Math_IntegratorOptions
#define ROOT_Math_IntegratorOptions

#include "Math/AllIntegrationTypes.h"

#include <string>
#include <iostream>

namespace ROOT {


namespace Math {

   class IOptions;


//_______________________________________________________________________________
/**
    Base class for Numerical integration  options
    common in 1D and multi-dimension
    This is an internal class and is not supposed to be instantiated by the user

    @ingroup Integration
*/
class BaseIntegratorOptions {

protected:

   /// protected constructor to avoid user creating this class
   BaseIntegratorOptions();

public:

   // copy constructor
   BaseIntegratorOptions(const BaseIntegratorOptions & opt);

   /// assignment operators
   BaseIntegratorOptions & operator=(const BaseIntegratorOptions & opt);


   /// protected constructor to avoid user creating this class
   virtual ~BaseIntegratorOptions() { ClearExtra(); }


   /// name of 1D integrator
   virtual std::string  Integrator() const = 0;

   /** non-static methods for  retrivieng options */

   /// absolute tolerance
   double AbsTolerance() const { return  fAbsTolerance; }

   /// absolute tolerance
   double RelTolerance() const { return  fRelTolerance; }

   /// size of the workspace
   unsigned int WKSize() const { return fWKSize; }


   /// return extra options
   IOptions * ExtraOptions() const { return fExtraOptions; }

   /** non-static methods for setting options */


   /// set the abs tolerance
   void SetAbsTolerance(double tol) { fAbsTolerance = tol; }

   /// set the relative tolerance
   void SetRelTolerance(double tol) { fRelTolerance = tol; }

   /// set workspace size
   void SetWKSize(unsigned int size) { fWKSize = size; }

   /// set extra options (in this case pointer is cloned)
   void  SetExtraOptions(const IOptions & opt);


protected:

   void ClearExtra();

   int       fIntegType;   // Integrator type (value converted from enum)

   unsigned int fWKSize;        // workspace size
   unsigned int fNCalls;        // (max) funxtion calls
   double fAbsTolerance;        // absolute tolerance
   double fRelTolerance;        // relative tolerance


   // extra options
   ROOT::Math::IOptions *   fExtraOptions;  // extra options

};

//_______________________________________________________________________________
/**
    Numerical one dimensional integration  options

    @ingroup Integration
*/

class IntegratorOneDimOptions : public BaseIntegratorOptions {

public:


   // constructor using the default options
   // can pass a pointer to extra options (N.B. pointer will be managed by the class)
   IntegratorOneDimOptions(IOptions * extraOpts = 0);

   virtual ~IntegratorOneDimOptions() {}

   // copy constructor
   IntegratorOneDimOptions(const IntegratorOneDimOptions & rhs) :
      BaseIntegratorOptions(rhs)
   {}

   // assignment operator
   IntegratorOneDimOptions & operator=(const IntegratorOneDimOptions & rhs) {
      if (this == &rhs) return *this;
      static_cast<BaseIntegratorOptions &>(*this) = rhs;
      return *this;
   }

   // specific method for one-dim
   /// Set number of points for active integration rule.
   /// - For the GSL adaptive integrator, `n = 1,2,3,4,5,6` correspond to the 15,21,31,41,51,61-point integration rules.
   /// - For the GaussLegendre integrator, use values > 6, which correspond to the actual number of points being evaluated.
   void SetNPoints(unsigned int n) { fNCalls = n; }

   /// Number of points used by current integration rule. \see SetNPoints().
   unsigned int NPoints() const { return fNCalls; }

   /// name of 1D integrator
   std::string  Integrator() const;

   /// type of the integrator (return the enumeration type)
   IntegrationOneDim::Type IntegratorType() const { return (IntegrationOneDim::Type) fIntegType; }

   /// set 1D integrator name
   void SetIntegrator(const char * name);

   /// print all the options
   void Print(std::ostream & os = std::cout) const;

   // static methods for setting and retrieving the default options

   static void SetDefaultIntegrator(const char * name);
   static void SetDefaultAbsTolerance(double tol);
   static void SetDefaultRelTolerance(double tol);
   static void SetDefaultWKSize(unsigned int size);
   static void SetDefaultNPoints(unsigned int n);

   static std::string  DefaultIntegrator();
   static IntegrationOneDim::Type DefaultIntegratorType();
   static double DefaultAbsTolerance();
   static double DefaultRelTolerance();
   static unsigned int DefaultWKSize();
   static unsigned int DefaultNPoints();

   /// retrieve specific options - if not existing create a IOptions
   static ROOT::Math::IOptions & Default(const char * name);

   // find specific options - return 0 if not existing
   static ROOT::Math::IOptions * FindDefault(const char * name);

   /// print only the specified default options
   static void PrintDefault(const char * name = 0, std::ostream & os = std::cout);


private:


};

//_______________________________________________________________________________
/**
    Numerical multi dimensional integration  options

    @ingroup Integration
*/

class IntegratorMultiDimOptions : public BaseIntegratorOptions {

public:


   // constructor using the default options
   // can pass a pointer to extra options (N.B. pointer will be managed by the class)
   IntegratorMultiDimOptions(IOptions * extraOpts = 0);

   virtual ~IntegratorMultiDimOptions() {}

   // copy constructor
   IntegratorMultiDimOptions(const IntegratorMultiDimOptions & rhs) :
      BaseIntegratorOptions(rhs)
   {}

   // assignment operator
   IntegratorMultiDimOptions & operator=(const IntegratorMultiDimOptions & rhs) {
      if (this == &rhs) return *this;
      static_cast<BaseIntegratorOptions &>(*this) = rhs;
      return *this;
   }

   // specific method for multi-dim
   /// set maximum number of function calls
   void SetNCalls(unsigned int calls) { fNCalls = calls; }

   /// maximum number of function calls
   unsigned int NCalls() const { return fNCalls; }

   /// name of multi-dim integrator
   std::string  Integrator() const;

   /// type of the integrator (return the enumeration type)
   IntegrationMultiDim::Type IntegratorType() const { return (IntegrationMultiDim::Type) fIntegType; }

   /// set multi-dim integrator name
   void SetIntegrator(const char * name);

   /// print all the options
   void Print(std::ostream & os = std::cout) const;

   // static methods for setting and retrieving the default options

   static void SetDefaultIntegrator(const char * name);
   static void SetDefaultAbsTolerance(double tol);
   static void SetDefaultRelTolerance(double tol);
   static void SetDefaultWKSize(unsigned int size);
   static void SetDefaultNCalls(unsigned int ncall);

   static std::string DefaultIntegrator();
   static IntegrationMultiDim::Type DefaultIntegratorType();
   static double DefaultAbsTolerance();
   static double DefaultRelTolerance();
   static unsigned int DefaultWKSize();
   static unsigned int DefaultNCalls();

   // retrieve specific options
   static ROOT::Math::IOptions & Default(const char * name);

   // find specific options - return 0 if not existing
   static ROOT::Math::IOptions * FindDefault(const char * name);

   /// print only the specified default options
   static void PrintDefault(const char * name = 0, std::ostream & os = std::cout);


private:


};


   } // end namespace Math

} // end namespace ROOT

#endif
