// @(#)root/minuit:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TLinearMinimizer

#ifndef ROOT_TLinearMinimizer
#define ROOT_TLinearMinimizer

#include "Math/Minimizer.h"

#include "Rtypes.h"

#include <vector>
#include <string>

class TLinearFitter;




/**
   TLinearMinimizer class: minimizer implementation based on TMinuit.
*/
class TLinearMinimizer  : public ROOT::Math::Minimizer {

public:

   /**
      Default constructor
   */
   TLinearMinimizer (int type = 0);

   /**
      Constructor from a char * (used by PM)
   */
   TLinearMinimizer ( const char * type );

   /**
      Destructor (no operations)
   */
   ~TLinearMinimizer () override;

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   TLinearMinimizer(const TLinearMinimizer &);

   /**
      Assignment operator
   */
   TLinearMinimizer & operator = (const TLinearMinimizer & rhs);

public:

   /// set the fit model function
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGradFunction & func) override;

   /// set free variable (dummy impl. )
   bool SetVariable(unsigned int , const std::string & , double , double ) override { return false; }

   /// set fixed variable (override if minimizer supports them )
   bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */) override;

   /// method to perform the minimization
    bool Minimize() override;

   /// return minimum function value
   double MinValue() const override { return fMinVal; }

   /// return expected distance reached from the minimum
   double Edm() const override { return 0; }

   /// return  pointer to X values at the minimum
   const double *  X() const override { return &fParams.front(); }

   /// return pointer to gradient values at the minimum
   const double *  MinGradient() const override { return 0; } // not available in Minuit2

   /// number of function calls to reach the minimum
   unsigned int NCalls() const override { return 0; }

   /// this is <= Function().NDim() which is the total
   /// number of variables (free+ constrained ones)
   unsigned int NDim() const override { return fDim; }

   /// number of free variables (real dimension of the problem)
   /// this is <= Function().NDim() which is the total
   unsigned int NFree() const override { return fNFree; }

   /// minimizer provides error and error matrix
   bool ProvidesError() const override { return true; }

   /// return errors at the minimum
   const double * Errors() const override { return  (fErrors.empty()) ? 0 : &fErrors.front(); }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   double CovMatrix(unsigned int i, unsigned int j) const override {
      return (fCovar.empty()) ? 0 : fCovar[i + fDim* j];
   }

   /// return covariance matrix status
   int CovMatrixStatus() const override {
      if (fCovar.size() == 0) return 0;
      return (fStatus ==0) ? 3 : 1;
   }

   /// return reference to the objective function
   ///virtual const ROOT::Math::IGenFunction & Function() const;




protected:

private:

   bool fRobust;
   unsigned int fDim;
   unsigned int fNFree;
   double fMinVal;
   std::vector<double> fParams;
   std::vector<double> fErrors;
   std::vector<double> fCovar;

   const ROOT::Math::IMultiGradFunction * fObjFunc;
   TLinearFitter * fFitter;

   ClassDef(TLinearMinimizer,1)  //Implementation of the Minimizer interface using TLinearFitter

};



#endif /* ROOT_TLinearMinimizer */
