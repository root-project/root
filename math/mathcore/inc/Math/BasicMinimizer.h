// @(#)root/mathmore:$Id$
// Author: L. Moneta Oct 2012

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/


// Header file for class BasicMinimizer

#ifndef ROOT_Math_BasicMinimizer
#define ROOT_Math_BasicMinimizer

#include "Math/Minimizer.h"


#include "Math/IFunctionfwd.h"

#include "Math/IParamFunctionfwd.h"

#include "Math/MinimTransformVariable.h"


#include <vector>
#include <map>
#include <string>



namespace ROOT {

namespace Math {

   class MinimTransformFunction;



//_______________________________________________________________________________
/**
   Base Minimizer class, which defines the basic functionality of various minimizer
   implementations (apart from Minuit and Minuit2)
   It provides support for storing parameter values, step size,
   parameter transformation etc.. in case real minimizer implementations do not provide
   such functionality.
   This is an internal class and should not be used directly by the user

   @ingroup MultiMin
*/


class BasicMinimizer : public ROOT::Math::Minimizer {

public:

   /**
      Default constructor
   */
   BasicMinimizer ( );


   /**
      Destructor
   */
   ~BasicMinimizer () override;

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   BasicMinimizer(const BasicMinimizer &) : Minimizer() {}

   /**
      Assignment operator
   */
   BasicMinimizer & operator = (const BasicMinimizer & rhs) {
      if (this == &rhs) return *this;  // time saving self-test
      return *this;
   }

public:

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   /// set gradient the function to minimize
   void SetFunction(const ROOT::Math::IMultiGradFunction & func) override;

   /// set free variable
   bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) override;


   /// set lower limit variable  (override if minimizer supports them )
   bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower ) override;
   /// set upper limit variable (override if minimizer supports them )
   bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) override;
   /// set upper/lower limited variable (override if minimizer supports them )
   bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */) override;
   /// set fixed variable (override if minimizer supports them )
   bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */) override;
   /// set the value of an existing variable
   bool SetVariableValue(unsigned int ivar, double val ) override;
   /// set the values of all existing variables (array must be dimensioned to the size of existing parameters)
   bool SetVariableValues(const double * x) override;
   /// set the step size of an already existing variable
   bool SetVariableStepSize(unsigned int ivar, double step ) override;
   /// set the lower-limit of an already existing variable
   bool SetVariableLowerLimit(unsigned int ivar, double lower) override;
   /// set the upper-limit of an already existing variable
   bool SetVariableUpperLimit(unsigned int ivar, double upper) override;
   /// set the limits of an already existing variable
   bool SetVariableLimits(unsigned int ivar, double lower, double upper) override;
   /// fix an existing variable
   bool FixVariable(unsigned int ivar) override;
   /// release an existing variable
   bool ReleaseVariable(unsigned int ivar) override;
   /// query if an existing variable is fixed (i.e. considered constant in the minimization)
   /// note that by default all variables are not fixed
   bool IsFixedVariable(unsigned int ivar)  const override;
   /// get variable settings in a variable object (like ROOT::Fit::ParamsSettings)
   bool GetVariableSettings(unsigned int ivar, ROOT::Fit::ParameterSettings & varObj) const override;
   /// get name of variables (override if minimizer support storing of variable names)
   std::string VariableName(unsigned int ivar) const override;
   /// get index of variable given a variable given a name
   /// return -1 if variable is not found
   int VariableIndex(const std::string & name) const override;

   /// method to perform the minimization
    bool Minimize() override;

   /// return minimum function value
   double MinValue() const override { return fMinVal; }

   /// return  pointer to X values at the minimum
   const double *  X() const override { return &fValues.front(); }

   /// number of dimensions
   unsigned int NDim() const override { return fDim; }

   /// number of free variables (real dimension of the problem)
   unsigned int NFree() const override;

   /// total number of parameter defined
   virtual unsigned int NPar() const { return fValues.size(); }

   /// return pointer to used objective function
   const ROOT::Math::IMultiGenFunction * ObjFunction() const { return fObjFunc; }

   /// return pointer to used gradient object function  (NULL if gradient is not supported)
   const ROOT::Math::IMultiGradFunction * GradObjFunction() const;

   /// return transformation function (NULL if not having a transformation)
   const ROOT::Math::MinimTransformFunction * TransformFunction() const;

   /// print result of minimization
   void PrintResult() const;

   /// accessor methods
   virtual const double * StepSizes() const { return &fSteps.front(); }

protected:

   bool CheckDimension() const;

   bool CheckObjFunction() const;

   MinimTransformFunction * CreateTransformation(std::vector<double> & startValues, const ROOT::Math::IMultiGradFunction * func = 0);

   void SetFinalValues(const double * x);

   void SetMinValue(double val) { fMinVal = val; }

private:

   // dimension of the function to be minimized
   unsigned int fDim;

   const ROOT::Math::IMultiGenFunction * fObjFunc;

   double fMinVal;
   std::vector<double> fValues;
   std::vector<double> fSteps;
   std::vector<std::string> fNames;
   std::vector<ROOT::Math::EMinimVariableType> fVarTypes;       ///< vector specifying the type of variables
   std::map< unsigned int, std::pair<double, double> > fBounds; ///< map specifying the bound using as key the parameter index

};

   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Math_BasicMinimizer */
