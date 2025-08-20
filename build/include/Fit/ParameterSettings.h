// @(#)root/mathcore:$Id$
// Author: L. Moneta Thu Sep 21 16:21:48 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class ParameterSettings

#ifndef ROOT_Fit_ParameterSettings
#define ROOT_Fit_ParameterSettings

#include <string>

namespace ROOT {

   namespace Fit {


//___________________________________________________________________________________
/**
   Class, describing value, limits and step size of the parameters
   Provides functionality also to set/retrieve values, step sizes, limits and fix the
   parameters.

   To be done: add constraints (equality and inequality) as functions of the parameters

   @ingroup FitMain
*/
class ParameterSettings {

public:

   /**
      Default constructor
   */
   ParameterSettings () {}


  ///constructor for unlimited named Parameter
   ParameterSettings(const std::string & name, double val, double err) :
    fValue(val), fStepSize(err),
    fName(name)
   {}

   ///constructor for double limited Parameter. The given value should be within the given limits [min,max]
   ParameterSettings(const std::string &  name, double val, double err,
                     double min, double max) :
      fValue(val), fStepSize(err),
      fName(name)
   {
      SetLimits(min,max);
   }

   ///constructor for fixed Parameter
   ParameterSettings(const std::string &  name, double val) :
    fValue(val), fStepSize(0), fFix(true),
    fName(name)
   {}




   /// set value and name (unlimited parameter)
   void Set(const std::string & name, double value, double step) {
      SetName(name);
      SetValue(value);
      SetStepSize(step);
   }

   /// set a limited parameter. The given value should be within the given limits [min,max]
   void Set(const std::string & name, double value, double step, double lower, double upper ) {
      SetName(name);
      SetValue(value);
      SetStepSize(step);
      SetLimits(lower,upper);
   }

   /// set a fixed parameter
   void Set(const std::string & name, double value) {
      SetName(name);
      SetValue(value);
      Fix();
   }

   /// return parameter value
   double Value() const { return fValue; }
   /// return step size
   double StepSize() const { return fStepSize; }
   /// return lower limit value
   double LowerLimit() const {return fLowerLimit;}
   /// return upper limit value
   double UpperLimit() const {return fUpperLimit;}
   /// check if is fixed
   bool IsFixed() const { return fFix; }
   /// check if parameter has lower limit
   bool HasLowerLimit() const {return fHasLowerLimit; }
   /// check if parameter has upper limit
   bool HasUpperLimit() const {return fHasUpperLimit; }
   /// check if is bound
   bool IsBound() const {   return fHasLowerLimit || fHasUpperLimit;  }
   /// check if is double bound (upper AND lower limit)
   bool IsDoubleBound() const { return fHasLowerLimit && fHasUpperLimit;  }
   /// return name
   const std::string & Name() const { return fName; }

   /** interaction **/

   /// set name
   void SetName(const std::string & name ) { fName = name; }

   /// fix  the parameter
   void Fix() {fFix = true;}
   /// release the parameter
   void Release() {fFix = false;}
   /// set the value
   void SetValue(double val) {fValue = val;}
   /// set the step size
   void SetStepSize(double err) {fStepSize = err;}
   void SetLimits(double low, double up);
   /// set a single upper limit
   void SetUpperLimit(double up) {
    fLowerLimit = 0.;
    fUpperLimit = up;
    fHasLowerLimit = false;
    fHasUpperLimit = true;
   }
   /// set a single lower limit
   void SetLowerLimit(double low) {
      fLowerLimit = low;
      fUpperLimit = 0.;
      fHasLowerLimit = true;
      fHasUpperLimit = false;
   }

   /// remove all limit
   void RemoveLimits() {
      fLowerLimit = 0.;
      fUpperLimit = 0.;
      fHasLowerLimit = false;
      fHasUpperLimit = false;
   }

private:

   double fValue = 0.0;          ///< parameter value
   double fStepSize = 0.1;       ///< parameter step size (used by minimizer)
   bool fFix = false;            ///< flag to control if parameter is fixed
   double fLowerLimit = 0.0;     ///< lower parameter limit
   double fUpperLimit = 0.0;     ///< upper parameter limit
   bool fHasLowerLimit = false;  ///< flag to control lower parameter limit
   bool fHasUpperLimit = false;  ///< flag to control upper parameter limit

   std::string fName;    ///< parameter name

};

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_ParameterSettings */
