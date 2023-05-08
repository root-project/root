// @(#)root/minuit:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TMinuitMinimizer

#ifndef ROOT_TMinuitMinimizer
#define ROOT_TMinuitMinimizer

#include "Math/Minimizer.h"

#include "Rtypes.h"

#include <vector>
#include <string>

class TMinuit;

namespace ROOT {

   namespace Minuit {


      // enumeration specifying the type of TMinuit minimizers
      enum EMinimizerType {
         kMigrad,
         kSimplex,
         kCombined,
         kMigradImproved,
         kScan,
         kSeek
      };

   }
}



/**
   TMinuitMinimizer class:
   ROOT::Math::Minimizer implementation based on TMinuit

   @ingroup TMinuit
*/
class TMinuitMinimizer  : public ROOT::Math::Minimizer {

public:

   /**
      Default constructor
   */
   TMinuitMinimizer ( ROOT::Minuit::EMinimizerType type = ROOT::Minuit::kMigrad, unsigned int ndim = 0);

   /**
      Constructor from a char * (used by PM)
   */
   TMinuitMinimizer ( const char * type , unsigned int ndim = 0);

   /**
      Destructor (no operations)
   */
   ~TMinuitMinimizer () override;

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   TMinuitMinimizer(const TMinuitMinimizer &);

   /**
      Assignment operator
   */
   TMinuitMinimizer & operator = (const TMinuitMinimizer & rhs);

public:

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGradFunction & func) override;

   /// set free variable
   bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) override;

   /// set upper/lower limited variable (override if minimizer supports them )
   bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */) override;

   /// set lower limit variable  (override if minimizer supports them )
   bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower ) override;

   /// set upper limit variable (override if minimizer supports them )
   bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) override;

   /// set fixed variable (override if minimizer supports them )
   bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */) override;

   /// set the value of an existing variable
   bool SetVariableValue(unsigned int , double ) override;

   /// set the step size of an existing variable
   bool SetVariableStepSize(unsigned int , double ) override;
   /// set the lower-limit of an existing variable
   bool SetVariableLowerLimit(unsigned int , double ) override;
   /// set the upper-limit of an existing variable
   bool SetVariableUpperLimit(unsigned int , double ) override;
   /// set the limits of an existing variable
   bool SetVariableLimits(unsigned int ivar, double lower, double upper) override;
   /// fix an existing variable
   bool FixVariable(unsigned int) override;
   /// release an existing variable
   bool ReleaseVariable(unsigned int) override;
   /// query if an existing variable is fixed (i.e. considered constant in the minimization)
   /// note that by default all variables are not fixed
   bool IsFixedVariable(unsigned int) const override;
   /// get variable settings in a variable object (like ROOT::Fit::ParamsSettings)
   bool GetVariableSettings(unsigned int, ROOT::Fit::ParameterSettings & ) const override;


   /// method to perform the minimization
    bool Minimize() override;

   /// return minimum function value
   double MinValue() const override;

   /// return expected distance reached from the minimum
   double Edm() const override;

   /// return  pointer to X values at the minimum
   const double *  X() const override { return &fParams.front(); }

   /// return pointer to gradient values at the minimum
   const double *  MinGradient() const override { return nullptr; } // not available in Minuit2

   /// number of function calls to reach the minimum
   unsigned int NCalls() const override;

   /// this is <= Function().NDim() which is the total
   /// number of variables (free+ constrained ones)
   unsigned int NDim() const override { return fDim; }

   /// number of free variables (real dimension of the problem)
   /// this is <= Function().NDim() which is the total
   unsigned int NFree() const override;

   /// minimizer provides error and error matrix
   bool ProvidesError() const override { return true; }

   /// return errors at the minimum
   const double * Errors() const override { return  &fErrors.front(); }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   double CovMatrix(unsigned int i, unsigned int j) const override {
      return ( fCovar.size() > (i + fDim* j) ) ? fCovar[i + fDim* j] : 0;
   }

   /**
       Fill the passed array with the  covariance matrix elements
       if the variable is fixed or const the value is zero.
       The array will be filled as cov[i *ndim + j]
       The ordering of the variables is the same as in errors and parameter value.
       This is different from the direct interface of Minuit2 or TMinuit where the
       values were obtained only to variable parameters
   */
   bool GetCovMatrix(double * cov) const override;

   /**
       Fill the passed array with the Hessian matrix elements
       The Hessian matrix is the matrix of the second derivatives
       and is the inverse of the covariance matrix
       If the variable is fixed or const the values for that variables are zero.
       The array will be filled as h[i *ndim + j]
   */
   bool GetHessianMatrix(double * h) const override;

   ///return status of covariance matrix
   int CovMatrixStatus() const override;

   ///global correlation coefficient for variable i
   double GlobalCC(unsigned int ) const override;

   /// minos error for variable i, return false if Minos failed
   bool GetMinosError(unsigned int i, double & errLow, double & errUp, int = 0) override;

   /// minos status code of last Minos run
   /// minos status = -1   : Minos is not run
   ///              =  0   : last MINOS run was succesfull
   ///              >  0   : some problems encountered when running MINOS
   int MinosStatus() const override { return fMinosStatus; }

   /**
      perform a full calculation of the Hessian matrix for error calculation
    */
   bool Hesse() override;

   /**
      scan a parameter i around the minimum. A minimization must have been done before,
      return false if it is not the case
    */
   bool Scan(unsigned int i, unsigned int &nstep, double * x, double * y, double xmin = 0, double xmax = 0) override;

   /**
      find the contour points (xi,xj) of the function for parameter i and j around the minimum
      The contour will be find for value of the function = Min + ErrorUp();
    */
   bool Contour(unsigned int i, unsigned int j, unsigned int & npoints, double *xi, double *xj) override;


   void PrintResults() override;

   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const;

   /// get name of variables (override if minimizer support storing of variable names)
   std::string VariableName(unsigned int ivar) const override;

   /// get index of variable given a variable given a name
   /// return always -1 . (It is Not implemented)
   int VariableIndex(const std::string & name) const override;

   /// static function to switch on/off usage of static global TMinuit instance (gMinuit)
   /// By default it is used (i.e. is on). Method returns the previous state
   bool static UseStaticMinuit(bool on = true);

   /// suppress the minuit warnings (if called with false will enable them)
   /// By default they are suppressed only when the printlevel is <= 0
   void SuppressMinuitWarnings(bool nowarn=true);

   /// set debug mode. Return true if setting was successfull
   bool SetDebug(bool on = true);

protected:

   /// implementation of FCN for Minuit
   static void Fcn( int &, double * , double & f, double * , int);
   /// implementation of FCN for Minuit when user provided gradient is used
   static void FcnGrad( int &, double * g, double & f, double * , int);

   /// initialize the TMinuit instance
   void InitTMinuit(int ndim);

   /// reset
   void DoClear();

   ///release a parameter that is fixed  when it is redefined
   void DoReleaseFixParameter( int ivar);

   /// retrieve minimum parameters and errors from TMinuit
   void RetrieveParams();

   /// retrieve error matrix from TMinuit
   void RetrieveErrorMatrix();

   /// check TMinuit instance
   bool CheckMinuitInstance() const;

   ///check parameter
   bool CheckVarIndex(unsigned int ivar) const;


private:

   bool fUsed;
   bool fMinosRun;
   unsigned int fDim;
   int fMinosStatus = -1;         // Minos status code
   std::vector<double> fParams;   // vector of output values
   std::vector<double> fErrors;   // vector of output errors
   std::vector<double> fCovar;    // vector storing the covariance matrix

   ROOT::Minuit::EMinimizerType fType;
   TMinuit * fMinuit;

   static TMinuit * fgMinuit;

   static bool fgUsed;  // flag to control if static instance has done minimization
   static bool fgUseStaticMinuit; // flag to control if using global TMInuit instance (gMinuit)

   ClassDef(TMinuitMinimizer,1)  //Implementation of Minimizer interface using TMinuit

};



#endif /* ROOT_TMinuitMinimizer */
