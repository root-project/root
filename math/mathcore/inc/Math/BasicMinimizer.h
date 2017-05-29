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
   Base Minimizer class, which defines the basic funcionality of various minimizer
   implementations (apart from Minuit and Minuit2)
   It provides support for storing parameter values, step size,
   parameter transofrmation etc.. in case real minimizer impelmentations do not provide
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
   virtual ~BasicMinimizer ();

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
   virtual void SetFunction(const ROOT::Math::IMultiGenFunction & func);

   /// set gradient the function to minimize
   virtual void SetFunction(const ROOT::Math::IMultiGradFunction & func);

   /// set free variable
   virtual bool SetVariable(unsigned int ivar, const std::string & name, double val, double step);


   /// set lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower );
   /// set upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper );
   /// set upper/lower limited variable (override if minimizer supports them )
   virtual bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */);
   /// set fixed variable (override if minimizer supports them )
   virtual bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */);
   /// set the value of an existing variable
   virtual bool SetVariableValue(unsigned int ivar, double val );
   /// set the values of all existing variables (array must be dimensioned to the size of existing parameters)
   virtual bool SetVariableValues(const double * x);
   /// set the step size of an already existing variable
   virtual bool SetVariableStepSize(unsigned int ivar, double step );
   /// set the lower-limit of an already existing variable
   virtual bool SetVariableLowerLimit(unsigned int ivar, double lower);
   /// set the upper-limit of an already existing variable
   virtual bool SetVariableUpperLimit(unsigned int ivar, double upper);
   /// set the limits of an already existing variable
   virtual bool SetVariableLimits(unsigned int ivar, double lower, double upper);
   /// fix an existing variable
   virtual bool FixVariable(unsigned int ivar);
   /// release an existing variable
   virtual bool ReleaseVariable(unsigned int ivar);
   /// query if an existing variable is fixed (i.e. considered constant in the minimization)
   /// note that by default all variables are not fixed
   virtual bool IsFixedVariable(unsigned int ivar)  const;
   /// get variable settings in a variable object (like ROOT::Fit::ParamsSettings)
   virtual bool GetVariableSettings(unsigned int ivar, ROOT::Fit::ParameterSettings & varObj) const;
   /// get name of variables (override if minimizer support storing of variable names)
   virtual std::string VariableName(unsigned int ivar) const;
   /// get index of variable given a variable given a name
   /// return -1 if variable is not found
   virtual int VariableIndex(const std::string & name) const;

   /// method to perform the minimization
   virtual  bool Minimize();

   /// return minimum function value
   virtual double MinValue() const { return fMinVal; }

   /// return  pointer to X values at the minimum
   virtual const double *  X() const { return &fValues.front(); }

   /// number of dimensions
   virtual unsigned int NDim() const { return fDim; }

   /// number of free variables (real dimension of the problem)
   virtual unsigned int NFree() const;

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
   std::vector<ROOT::Math::EMinimVariableType> fVarTypes;  // vector specifyng the type of variables
   std::map< unsigned int, std::pair<double, double> > fBounds; // map specifying the bound using as key the parameter index

};

   } // end namespace Fit

} // end namespace ROOT



#endif /* ROOT_Math_BasicMinimizer */
