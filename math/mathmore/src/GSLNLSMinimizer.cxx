// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLNLSMinimizer

#include "Math/GSLNLSMinimizer.h"

#include "Math/MinimTransformFunction.h"
#include "Math/MultiNumGradFunction.h"

#include "Math/Error.h"
#include "GSLMultiFit.h"
#include "gsl/gsl_errno.h"


#include "Math/FitMethodFunction.h"
//#include "Math/Derivator.h"

#include <iostream> 
#include <iomanip>
#include <cassert>
#include <memory>

namespace ROOT { 

   namespace Math { 


// class to implement transformation of chi2 function
// in general could make template on the fit method function type

class FitTransformFunction : public FitMethodFunction { 

public:
   
   FitTransformFunction(const FitMethodFunction & f, const std::vector<EMinimVariableType> & types, const std::vector<double> & values, 
                              const std::map<unsigned int, std::pair<double, double> > & bounds) : 
      FitMethodFunction( f.NDim(), f.NPoints() ),
      fFunc(f),
      fTransform(new MinimTransformFunction( new MultiNumGradFunction(f), types, values, bounds) ), 
      fGrad( std::vector<double>(f.NDim() ) )
   {
      // constructor
      // need to pass to MinimTransformFunction a new pointer which will be managed by the class itself
      // pass a gradient pointer although it will not be used byb the class 
   }

   ~FitTransformFunction() { 
      assert(fTransform); 
      delete fTransform; 
   }

   // re-implement data element
   virtual double DataElement(const double *  x, unsigned i, double * g = 0) const { 
      // transform from x internal to x external 
      const double * xExt = fTransform->Transformation(x); 
      if ( g == 0) return fFunc.DataElement( xExt, i );
      // use gradient 
      double val =  fFunc.DataElement( xExt, i, &fGrad[0]); 
      // transform gradient 
      fTransform->GradientTransformation( x, &fGrad.front(), g);  
      return val; 
   }


   IMultiGenFunction * Clone() const {
      // not supported
      return 0; 
   }

   // dimension (this is number of free dimensions)
   unsigned int NDim() const { 
      return fTransform->NDim(); 
   }

   unsigned int NTot() const { 
      return fTransform->NTot(); 
   }

   // forward of transformation functions
   const double * Transformation( const double * x) const { return fTransform->Transformation(x); }


   /// inverse transformation (external -> internal)
   void  InvTransformation(const double * xext,  double * xint) const { fTransform->InvTransformation(xext,xint); }

   /// inverse transformation for steps (external -> internal) at external point x
   void  InvStepTransformation(const double * x, const double * sext,  double * sint) const { fTransform->InvStepTransformation(x,sext,sint); }

   ///transform gradient vector (external -> internal) at internal point x
   void GradientTransformation(const double * x, const double *gext, double * gint) const   { fTransform->GradientTransformation(x,gext,gint); }

   void MatrixTransformation(const double * x, const double *cint, double * cext) const { fTransform->MatrixTransformation(x,cint,cext); }

private: 

   double DoEval(const double * x) const { 
      return fFunc( fTransform->Transformation(x) );
   }

   const FitMethodFunction & fFunc;                  // pointer to original fit method function 
   MinimTransformFunction * fTransform;        // pointer to transformation function
   mutable std::vector<double> fGrad;          // cached vector of gradient values
   
};




// GSLNLSMinimizer implementation

GSLNLSMinimizer::GSLNLSMinimizer( int /* ROOT::Math::EGSLNLSMinimizerType type */ ) : 
   fDim(0), 
   fNFree(0),
   fSize(0),
   fObjFunc(0), 
   fMinVal(0)
{
   // Constructor implementation : create GSLMultiFit wrapper object
   fGSLMultiFit = new GSLMultiFit( /*type */ ); 
   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   fEdm = -1; 
   fLSTolerance = 0.0001; 
   SetMaxIterations(100);
   SetPrintLevel(1);
}

GSLNLSMinimizer::~GSLNLSMinimizer () { 
   assert(fGSLMultiFit != 0); 
   delete fGSLMultiFit; 
//   if (fObjFunc) delete fObjFunc; 
}

bool GSLNLSMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set variable in minimizer - support only free variables 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   if (ivar == fValues.size() ) { 
      fValues.push_back(val); 
      fNames.push_back(name);
      fSteps.push_back(step); 
      fVarTypes.push_back(kDefault); 
   }
   else { 
      fValues[ivar] = val; 
      fNames[ivar] = name;
      fSteps[ivar] = step; 
      fVarTypes[ivar] = kDefault; 

      // remove bounds if needed
      std::map<unsigned  int, std::pair<double, double> >::iterator iter = fBounds.find(ivar); 
      if ( iter !=  fBounds.end() ) fBounds.erase (iter); 

   }
   return true; 
}

bool GSLNLSMinimizer::SetLowerLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower) { 
   //MATH_WARN_MSGVAL("GSLNLSMinimizer::SetLowerLimitedVariable","Ignore lower limit on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, lower);
   fVarTypes[ivar] = kLowBound; 
   return true;  
}
bool GSLNLSMinimizer::SetUpperLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double upper) { 
   //MATH_WARN_MSGVAL("GSLNLSMinimizer::SetUpperLimitedVariable","Ignore upper limit on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( upper, upper);
   fVarTypes[ivar] = kUpBound; 
   return true;  
}
bool GSLNLSMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper ) { 
   //MATH_WARN_MSGVAL("GSLNLSMinimizer::SetLimitedVariable","Ignore bounds on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, upper);
   fVarTypes[ivar] = kBounds; 
   return true;  
}

bool GSLNLSMinimizer::SetFixedVariable(unsigned int ivar , const std::string & name , double val ) {   
   /// set fixed variable (override if minimizer supports them )
   bool ret = SetVariable(ivar, name, val, 0.); 
   if (!ret) return false; 
   fVarTypes[ivar] = kFix; 
   return true;  
}

bool GSLNLSMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set variable value in minimizer 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   fValues[ivar] = val; 
   return true; 
}

bool GSLNLSMinimizer::SetVariableValues( const double * x) { 
   // set all variable values in minimizer 
   if (x == 0) return false; 
   std::copy(x,x+fValues.size(), fValues.begin() );
   return true; 
}


      
void GSLNLSMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) { 
   // set the function to minimizer 
   // need to create vector of functions to be passed to GSL multifit
   // support now only CHi2 implementation
   
   const ROOT::Math::FitMethodFunction * chi2Func = dynamic_cast<const ROOT::Math::FitMethodFunction *>(&func); 
   if (chi2Func == 0) { 
      if (PrintLevel() > 0) std::cout << "GSLNLSMinimizer: Invalid function set - only Chi2Func supported" << std::endl;
      return;
   } 
   fSize = chi2Func->NPoints(); 
   fDim = chi2Func->NDim(); 
   fNFree = fDim;

   // use vector by value 
   fResiduals.reserve(fSize);   
   for (unsigned int i = 0; i < fSize; ++i) { 
      fResiduals.push_back( LSResidualFunc(*chi2Func, i) ); 
   }
   // keep pointers to the chi2 function
   fObjFunc = chi2Func; 
 }

void GSLNLSMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & func ) { 
   // set the function to minimizer using gradient interface
   // not supported yet, implemented using the other SetFunction
   return SetFunction(static_cast<const ROOT::Math::IMultiGenFunction &>(func) );
}


bool GSLNLSMinimizer::Minimize() { 
   // set initial parameters of the minimizer
   int debugLevel = PrintLevel(); 


   assert (fGSLMultiFit != 0);   
   if (fResiduals.size() !=  fSize || fObjFunc == 0) {
      MATH_ERROR_MSG("GSLNLSMinimizer::Minimize","Function has not been  set");
      return false; 
   }

   unsigned int npar = fValues.size();
   if (npar == 0 || npar < fDim) { 
       MATH_ERROR_MSGVAL("GSLNLSMinimizer::Minimize","Wrong number of parameters",npar);
       return false; 
   }
   
   // set residual functions and check if a transformation is needed

   bool doTransform = (fBounds.size() > 0); 
   unsigned int ivar = 0; 
   while (!doTransform && ivar < fVarTypes.size() ) {
      doTransform = (fVarTypes[ivar++] != kDefault );
   }
   std::vector<double> startValues(fValues.begin(), fValues.end() );

   std::auto_ptr<FitTransformFunction> trFunc; 

   // in case of transformation wrap residual functions in new transformation functions
   // and transform from external variables  to internals one
   if (doTransform)   {  
      trFunc.reset(new FitTransformFunction ( *fObjFunc, fVarTypes, fValues, fBounds ) ); 
      for (unsigned int ires = 0; ires < fResiduals.size(); ++ires) {
         fResiduals[ires] = LSResidualFunc(*trFunc, ires);
      }

      trFunc->InvTransformation(&fValues.front(), &startValues[0]); 
      fNFree = trFunc->NDim(); // actual dimension  
      assert(fValues.size() == trFunc->NTot() ); 
      startValues.resize( fNFree );
   }

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLNLSMinimizer " << fGSLMultiFit->Name() << std::endl; 

//    // use a global step size = min (step vectors) 
//    double stepSize = 1; 
//    for (unsigned int i = 0; i < fSteps.size(); ++i) 
//       //stepSize += fSteps[i]; 
//       if (fSteps[i] < stepSize) stepSize = fSteps[i]; 

   int iret = fGSLMultiFit->Set( fResiduals, &startValues.front() );  
   if (iret) { 
      MATH_ERROR_MSGVAL("GSLNLSMinimizer::Minimize","Error setting the residual functions ",iret);
      return false; 
   }

   if (debugLevel >=1 ) std::cout <<"GSLNLSMinimizer: Start iterating......... "  << std::endl; 

   // start iteration 
   unsigned  int iter = 0; 
   int status; 
   bool minFound = false; 
   do { 
      status = fGSLMultiFit->Iterate(); 

      if (debugLevel >=1) { 
         std::cout << "----------> Iteration " << iter << " / " << MaxIterations() << " status " << gsl_strerror(status)  << std::endl; 
         const double * x = fGSLMultiFit->X();
         if (trFunc.get()) x = trFunc->Transformation(x);
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << (*fObjFunc)(x) << std::endl; 
         std::cout.precision(pr);
         std::cout << "            X Values : "; 
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << " " << fNames[i] << " = " << x[i]; 
         std::cout << std::endl; 
      }

      if (status) break; 

      // check also the delta in X()
      status = fGSLMultiFit->TestDelta( Tolerance(), Tolerance() );
      if (status == GSL_SUCCESS) {
         minFound = true; 
      }

      // double-check with the gradient
      int status2 = fGSLMultiFit->TestGradient( Tolerance() );
      if ( minFound && status2 != GSL_SUCCESS) {
         // check now edm 
         fEdm = fGSLMultiFit->Edm(); 
         if (fEdm > Tolerance() ) { 
            // continue the iteration
            status = status2; 
            minFound = false; 
         }
      }

      if (debugLevel >=1) { 
         std::cout  << "          after Gradient and Delta tests:  " << gsl_strerror(status); 
         if (fEdm > 0) std::cout << ", edm is:  " << fEdm;
         std::cout << std::endl;
      }

      iter++;

   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );

   // check edm 
   fEdm = fGSLMultiFit->Edm();
   if ( fEdm < Tolerance() ) { 
      minFound = true; 
   }

   // save state with values and function value
   const double * x = fGSLMultiFit->X(); 
   if (x == 0) return false; 

   // check to see if a transformation need to be applied 
   if (trFunc.get() != 0) { 
      const double * xtrans = trFunc->Transformation(x);  
      std::copy(xtrans, xtrans + trFunc->NTot(),  fValues.begin() ); 
   }
   else { 
      std::copy(x, x +fDim, fValues.begin() ); 
   }

   fMinVal =  (*fObjFunc)(&fValues.front() );
   fStatus = status; 

   fErrors.resize(fDim);

   // get errors from cov matrix 
   if (fGSLMultiFit->CovarMatrix() ) fCovMatrix.resize(fDim*fDim);
      
   if (minFound) { 

      if (trFunc.get() != 0) { 
         trFunc->MatrixTransformation(x, fGSLMultiFit->CovarMatrix(), &fCovMatrix[0] ); 
      }
      else {
         const double * m =  fGSLMultiFit->CovarMatrix();
         std::copy(m, m+ fDim*fDim, fCovMatrix.begin() );
      }
   
      for (unsigned int i = 0; i < fDim; ++i)
         fErrors[i] = std::sqrt(fCovMatrix[i*fDim + i]);

      if (debugLevel >=1 ) { 
         std::cout << "GSLNLSMinimizer: Minimum Found" << std::endl;  
         int pr = std::cout.precision(18);
         std::cout << "FVAL         = " << fMinVal << std::endl;
         std::cout << "Edm          = " << fEdm    << std::endl;
         std::cout.precision(pr);
         std::cout << "NIterations  = " << iter << std::endl;
         std::cout << "NFuncCalls   = " << fObjFunc->NCalls() << std::endl;
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << std::setw(12) <<  fNames[i] << " = " << std::setw(12) << fValues[i] << "   +/-   " << std::setw(12) << fErrors[i] << std::endl; 
      }

      return true; 
   }
   else { 
      if (debugLevel >=1 ) { 
         std::cout << "GSLNLSMinimizer: Minimization did not converge" << std::endl;  
         std::cout << "FVAL         = " << fMinVal << std::endl;
         std::cout << "Edm   = " << fGSLMultiFit->Edm() << std::endl;
         std::cout << "Niterations  = " << iter << std::endl;
      }
      return false; 
   }
   return false; 
}

const double * GSLNLSMinimizer::MinGradient() const {
   // return gradient (internal values)
   return fGSLMultiFit->Gradient(); 
}


double GSLNLSMinimizer::CovMatrix(unsigned int i , unsigned int j ) const { 
   // return covariance matrix element
   if ( fCovMatrix.size() == 0) return 0;  
   if (i > fDim || j > fDim) return 0; 
   return fCovMatrix[i*fDim + j];
}

int GSLNLSMinimizer::CovMatrixStatus( ) const { 
   // return covariance  matrix status = 0 not computed,
   // 1 computed but is approximate because minimum is not valid, 3 is fine
   if ( fCovMatrix.size() == 0) return 0;  
   // case minimization did not finished correctly
   if (fStatus != GSL_SUCCESS) return 1;
   return 3;
}


   } // end namespace Math

} // end namespace ROOT

