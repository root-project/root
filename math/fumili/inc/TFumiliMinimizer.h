// @(#)root/fumili:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class TFumiliMinimizer

#ifndef ROOT_TFumiliMinimizer
#define ROOT_TFumiliMinimizer

#include "Math/Minimizer.h"

#include "Math/FitMethodFunction.h"

#include "Rtypes.h"
#include <vector>
#include <string>

class TFumili;



// namespace ROOT {

//    namespace Math {

//       class BasicFitMethodFunction<ROOT::Math::IMultiGenFunction>;
//       class BasicFitMethodFunction<ROOT::Math::IMultiGradFunction>;

//    }
// }



/**
   TFumiliMinimizer class: minimizer implementation based on TFumili.
*/
class TFumiliMinimizer  : public ROOT::Math::Minimizer {

public:

   /**
      Default constructor (an argument is needed by plug-in manager)
   */
   TFumiliMinimizer (int dummy=0 );


   /**
      Destructor (no operations)
   */
   ~TFumiliMinimizer () override;

private:
   // usually copying is non trivial, so we make this unaccessible

   /**
      Copy constructor
   */
   TFumiliMinimizer(const TFumiliMinimizer &);

   /**
      Assignment operator
   */
   TFumiliMinimizer & operator = (const TFumiliMinimizer & rhs);

public:

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGenFunction & func) override;

   /// set the function to minimize
   void SetFunction(const ROOT::Math::IMultiGradFunction & func) override;

   /// set free variable
   bool SetVariable(unsigned int ivar, const std::string & name, double val, double step) override;

   /// set upper/lower limited variable (override if minimizer supports them )
   bool SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double /* lower */, double /* upper */) override;

#ifdef LATER
   /// set lower limit variable  (override if minimizer supports them )
   virtual bool SetLowerLimitedVariable(unsigned int  ivar , const std::string & name , double val , double step , double lower );
   /// set upper limit variable (override if minimizer supports them )
   virtual bool SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper );
#endif

   /// set fixed variable (override if minimizer supports them )
   bool SetFixedVariable(unsigned int /* ivar */, const std::string & /* name */, double /* val */) override;

   /// set the value of an existing variable
   bool SetVariableValue(unsigned int ivar, double val ) override;

   /// method to perform the minimization
    bool Minimize() override;

   /// return minimum function value
   double MinValue() const override { return fMinVal; }

   /// return expected distance reached from the minimum
   double Edm() const override { return fEdm; }

   /// return  pointer to X values at the minimum
   const double *  X() const override { return &fParams.front(); }

   /// return pointer to gradient values at the minimum
   const double *  MinGradient() const override { return nullptr; } // not available

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
   const double * Errors() const override { return  &fErrors.front(); }

   /** return covariance matrices elements
       if the variable is fixed the matrix is zero
       The ordering of the variables is the same as in errors
   */
   double CovMatrix(unsigned int i, unsigned int j) const override {
      return fCovar[i + fDim* j];
   }

   /*
     return covariance matrix status
   */
   int CovMatrixStatus() const override {
      if (fCovar.size() == 0) return 0;
      return (fStatus ==0) ? 3 : 1;
   }





protected:

   /// implementation of FCN for Fumili
   static void Fcn( int &, double * , double & f, double * , int);
   /// implementation of FCN for Fumili when user provided gradient is used
   //static void FcnGrad( int &, double * g, double & f, double * , int);

   /// static function implementing the evaluation of the FCN (it uses static instance fgFumili)
   static double EvaluateFCN(const double * x, double * g);

private:


   unsigned int fDim;
   unsigned int fNFree;
   double fMinVal;
   double fEdm;
   std::vector<double> fParams;
   std::vector<double> fErrors;
   std::vector<double> fCovar;

   TFumili * fFumili;

   // need to have a static copy of the function
   //NOTE: This is NOT thread safe.
   static ROOT::Math::FitMethodFunction * fgFunc;
   static ROOT::Math::FitMethodGradFunction * fgGradFunc;

   static TFumili * fgFumili; // static instance (used by fcn function)

   ClassDef(TFumiliMinimizer,1)  //Implementation of Minimizer interface using TFumili

};



#endif /* ROOT_TFumiliMinimizer */
