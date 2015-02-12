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

#ifndef ROOT_Math_Error
#include "Math/Error.h"
#endif


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
   ParameterSettings () :  
    fValue(0.), fStepSize(0.1), fFix(false), 
    fLowerLimit(0.), fUpperLimit(0.), fHasLowerLimit(false), fHasUpperLimit(false), 
    fName("") 
   {}

  
  ///constructor for unlimited named Parameter
   ParameterSettings(const std::string & name, double val, double err) :
    fValue(val), fStepSize(err), fFix(false), 
    fLowerLimit(0.), fUpperLimit(0.), fHasLowerLimit(false), fHasUpperLimit(false), 
    fName(name) 
   {}
  
   ///constructor for double limited Parameter
   ParameterSettings(const std::string &  name, double val, double err, 
                     double min, double max) :
      fValue(val), fStepSize(err), fFix(false), 
      fLowerLimit(0.), fUpperLimit(0.), fHasLowerLimit(false), fHasUpperLimit(false), 
      fName(name) 
   { 
      SetLimits(min,max); 
   }

   ///constructor for fixed Parameter
   ParameterSettings(const std::string &  name, double val) : 
    fValue(val), fStepSize(0), fFix(true), 
    fLowerLimit(0.), fUpperLimit(0.), fHasLowerLimit(false), fHasUpperLimit(false), 
    fName(name)  
   {}




   /// set value and name (unlimited parameter) 
   void Set(const std::string & name, double value, double step) { 
      SetName(name); 
      SetValue(value); 
      SetStepSize(step);
   }

   /// set a limited parameter
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


   /** 
      Destructor (no operations)
   */ 
   ~ParameterSettings ()  {}  

   /// copy constructor and assignment operators (leave them to the compiler) 

public: 

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
   /// set a double side limit, 
   /// if low == up the parameter is fixed  if low > up the limits are removed
   void SetLimits(double low, double up) {
      
      if ( low > up ) { 
         RemoveLimits(); 
         return; 
      }
      if (low == up && low == fValue) { 
         Fix();           
         return; 
      }
      if (low > fValue || up < fValue) { 
         MATH_INFO_MSG("ParameterSettings","lower/upper bounds outside current parameter value. The value will be set to (low+up)/2 ");
         fValue = 0.5 * (up+low);
      }
      fLowerLimit = low; 
      fUpperLimit = up;
      fHasLowerLimit = true; 
      fHasUpperLimit = true;
   }
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
  
      

protected: 


private: 

   double fValue;        // parameter value
   double fStepSize;     // parameter step size (used by minimizer)
   bool fFix;            // flag to control if parameter is fixed 
   double fLowerLimit;   // lower parameter limit
   double fUpperLimit;   // upper parameter limit
   bool fHasLowerLimit;  // flag to control lower parameter limit
   bool fHasUpperLimit;  // flag to control upper parameter limit

   std::string fName;    // parameter name

}; 

   } // end namespace Fit

} // end namespace ROOT


#endif /* ROOT_Fit_ParameterSettings */
