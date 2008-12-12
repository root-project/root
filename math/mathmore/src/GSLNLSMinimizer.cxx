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
#include "GSLMultiFit.h"
#include "gsl/gsl_errno.h"


#include "Math/FitMethodFunction.h"
//#include "Math/Derivator.h"

#include <iostream> 
#include <cassert>

namespace ROOT { 

   namespace Math { 




// GSLNLSMinimizer implementation

GSLNLSMinimizer::GSLNLSMinimizer( int /* ROOT::Math::EGSLNLSMinimizerType type */ ) : 
   fDim(0), 
   fObjFunc(0),
   fCovMatrix(0)
{
   // Constructor implementation : create GSLMultiFit wrapper object
   fGSLMultiFit = new GSLMultiFit( /*type */ ); 
   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   fLSTolerance = 0.0001; 
   SetMaxIterations(100);
   SetPrintLevel(3);
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
   }
   else { 
      fValues[ivar] = val; 
      fNames[ivar] = name;
      fSteps[ivar] = step; 
   }
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
   // need to create vector of funcitons to be passed to GSL multifit
   // support now only CHi2 implementation
   
   const ROOT::Math::FitMethodFunction * chi2Func = dynamic_cast<const ROOT::Math::FitMethodFunction *>(&func); 
   if (chi2Func == 0) { 
      if (PrintLevel() > 0) std::cout << "GSLNLSMinimizer: Invalid function set - only Chi2Func supported" << std::endl;
      return;
   } 
   fSize = chi2Func->NPoints(); 
   fDim = chi2Func->NDim(); 

   // use vector by value 
   fResiduals.reserve(fSize);   
   for (unsigned int i = 0; i < fSize; ++i) { 
      fResiduals.push_back( LSResidualFunc(*chi2Func, i) ); 
   }
   // keep pointers to the chi2 function
   fObjFunc = chi2Func; 
 }

void GSLNLSMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & /* func */) { 
   // set the function to minimizer (need to clone ??)
   // not supported yet 
   return; 
}


bool GSLNLSMinimizer::Minimize() { 
   // set initial parameters of the minimizer
   int debugLevel = PrintLevel(); 


   assert (fGSLMultiFit != 0);   
   if (fResiduals.size() !=  fSize) {
      std::cout << "GSLNLSMinimizer : Error - wrong residual size." << std::endl;
      return false; 
   }
   

//    // use a global step size = min (step vectors) 
//    double stepSize = 1; 
//    for (unsigned int i = 0; i < fSteps.size(); ++i) 
//       //stepSize += fSteps[i]; 
//       if (fSteps[i] < stepSize) stepSize = fSteps[i]; 

   int iret = fGSLMultiFit->Set( fResiduals, &fValues.front() );  
   if (iret) { 
      std::cout << "GSLNLSMinimizer : Error setting residual functions, iret = " << iret << std::endl;
      return false; 
   }


   if (debugLevel >=1 ) std::cout <<"Minimize using GSLNLSMinimizer " << fGSLMultiFit->Name() << std::endl; 


   //std::cout <<"print Level " << debugLevel << std::endl; 
   //debugLevel = 3; 

   // start iteration 
   unsigned  int iter = 0; 
   int status; 
   bool minFound = false; 
   do { 
      status = fGSLMultiFit->Iterate(); 

      if (debugLevel >=1) { 
         std::cout << "----------> Iteration " << iter << " / " << MaxIterations() << " status " << gsl_strerror(status)  << std::endl; 
         const double * x = fGSLMultiFit->X();
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

      // double-check with thegradient
      int status2 = fGSLMultiFit->TestGradient( Tolerance() );
      if ( minFound && status2 != GSL_SUCCESS) {
         // check now edm 
         double edm = fGSLMultiFit->Edm(); 
         if (debugLevel >=1) { 
            std::cout  << "          Gradient test failed, edm is:  " << edm  << std::endl; 
         }
         if (edm > Tolerance() ) { 
            // continue the iteration
            status = status2; 
            minFound = false; 
         }
      }

      if (debugLevel >=1) { 
         std::cout  << "          after Gradient and Delta tests:  " << gsl_strerror(status)  << std::endl; 
      }

      iter++;

   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );

   // check edm 
   double edmval = fGSLMultiFit->Edm();
   if (edmval < Tolerance() ) { 
      minFound = true; 
   }

   // save state with values and function value
   const double * x = fGSLMultiFit->X(); 
   if (x == 0) return false; 
   std::copy(x, x +fDim, fValues.begin() ); 
   fMinVal =  (*fObjFunc)(x);
   fStatus = status; 

   fErrors.resize(fDim);
      
   if (minFound) { 
      if (debugLevel >=1 ) { 
         std::cout << "GSLNLSMinimizer: Minimum Found" << std::endl;  
         int pr = std::cout.precision(18);
         std::cout << "FVAL         = " << fMinVal << std::endl;
         std::cout << "Edm   = " << edmval << std::endl;
         std::cout.precision(pr);
         std::cout << "Niterations  = " << iter << std::endl;
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << fNames[i] << "\t  = " << fValues[i] << std::endl; 
      }
      // get errors from cov matrix
      fCovMatrix = fGSLMultiFit->CovarMatrix(); 
      for (unsigned int i = 0; i < fDim; ++i)
         fErrors[i] = std::sqrt(fCovMatrix[i*fDim + i]);

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
   return fGSLMultiFit->Gradient(); 
}


double GSLNLSMinimizer::CovMatrix(unsigned int i , unsigned int j ) const { 
   if (!fCovMatrix) return 0;  
   if (i > fDim || j > fDim) return 0; 
   return fCovMatrix[i*fDim + j];
}

   } // end namespace Math

} // end namespace ROOT

