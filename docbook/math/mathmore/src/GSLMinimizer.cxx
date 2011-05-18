// @(#)root/mathmore:$Id$
// Author: L. Moneta Tue Dec 19 15:41:39 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 * This library is free software; you can redistribute it and/or      *
 * modify it under the terms of the GNU General Public License        *
 * as published by the Free Software Foundation; either version 2     *
 * of the License, or (at your option) any later version.             *
 *                                                                    *
 * This library is distributed in the hope that it will be useful,    *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of     *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU   *
 * General Public License for more details.                           *
 *                                                                    *
 * You should have received a copy of the GNU General Public License  *
 * along with this library (see file COPYING); if not, write          *
 * to the Free Software Foundation, Inc., 59 Temple Place, Suite      *
 * 330, Boston, MA 02111-1307 USA, or contact the author.             *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLMinimizer

#include "Math/GSLMinimizer.h"

#include "GSLMultiMinimizer.h"

#include "Math/MultiNumGradFunction.h"
#include "Math/FitMethodFunction.h"

#include "Math/MinimTransformFunction.h"

#include <cassert>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here
#include <limits> 

namespace ROOT { 

   namespace Math { 


GSLMinimizer::GSLMinimizer( ROOT::Math::EGSLMinimizerType type) : 
   fDim(0), 
   fObjFunc(0),
   fMinVal(0)
{
   // Constructor implementation : create GSLMultiMin wrapper object
   //std::cout << "create GSL Minimizer of type " << type << std::endl;

   fGSLMultiMin = new GSLMultiMinimizer((ROOT::Math::EGSLMinimizerType) type); 
   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   fLSTolerance = 0.1; // line search tolerance (use fixed)
   int niter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   if (niter <=0 ) niter = 1000; 
   SetMaxIterations(niter);
   SetPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel());
}

GSLMinimizer::GSLMinimizer( const char *  type) : 
   fDim(0), 
   fObjFunc(0),
   fMinVal(0)
{
   // Constructor implementation from a string 
   std::string algoname(type);
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower ); 

   ROOT::Math::EGSLMinimizerType algo =   kVectorBFGS2; // default value

   if (algoname == "conjugatefr") algo = kConjugateFR;   
   if (algoname == "conjugatepr") algo = kConjugatePR; 
   if (algoname == "bfgs") algo = kVectorBFGS; 
   if (algoname == "bfgs2") algo = kVectorBFGS2; 
   if (algoname == "steepestdescent") algo = kSteepestDescent; 
 

   //std::cout << "create GSL Minimizer of type " << algo << std::endl;

   fGSLMultiMin = new GSLMultiMinimizer(algo); 
   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   fLSTolerance = 0.1; // use 10**-4 
   int niter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   if (niter <=0 ) niter = 1000; 
   SetMaxIterations(niter);
   SetPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel());
}


GSLMinimizer::~GSLMinimizer () { 
   assert(fGSLMultiMin != 0); 
   delete fGSLMultiMin; 
   if (fObjFunc) delete fObjFunc; 
}

bool GSLMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
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

bool GSLMinimizer::SetLowerLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower) { 
   //MATH_WARN_MSGVAL("GSLMinimizer::SetLowerLimitedVariable","Ignore lower limit on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, lower);
   fVarTypes[ivar] = kLowBound; 
   return true;  
}
bool GSLMinimizer::SetUpperLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double upper ) { 
   //MATH_WARN_MSGVAL("GSLMinimizer::SetUpperLimitedVariable","Ignore upper limit on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( upper, upper);
   fVarTypes[ivar] = kUpBound; 
   return true;  
}

bool GSLMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper) { 
   //MATH_WARN_MSGVAL("GSLMinimizer::SetLimitedVariable","Ignore bounds on variable ",ivar);
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, upper);
   fVarTypes[ivar] = kBounds; 
   return true;  
}

bool GSLMinimizer::SetFixedVariable(unsigned int ivar , const std::string & name , double val ) {   
   /// set fixed variable (override if minimizer supports them )
   bool ret = SetVariable(ivar, name, val, 0.); 
   if (!ret) return false; 
   fVarTypes[ivar] = kFix; 
   return true;  
}


bool GSLMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set variable value in minimizer 
   // no change to transformation or variable status
   if (ivar > fValues.size() ) return false; 
   fValues[ivar] = val; 
   return true; 
}

bool GSLMinimizer::SetVariableValues( const double * x) { 
   // set all variable values in minimizer 
   if (x == 0) return false; 
   std::copy(x,x+fValues.size(), fValues.begin() );
   return true; 
}
      

void GSLMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) { 
   // set the function to minimizer 
   // need to calculate numerically the derivatives: do via class MultiNumGradFunction
   fObjFunc = new MultiNumGradFunction( func); 
   fDim = fObjFunc->NDim(); 
}

void GSLMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & func) { 
   // set the function to minimizer 
   fObjFunc = dynamic_cast<const ROOT::Math::IMultiGradFunction *>( func.Clone()); 
   assert(fObjFunc != 0);
   fDim = fObjFunc->NDim(); 
}

unsigned int GSLMinimizer::NCalls() const {
   // return numbr of function calls 
   // if method support 
   const ROOT::Math::MultiNumGradFunction * fnumgrad = dynamic_cast<const ROOT::Math::MultiNumGradFunction *>(fObjFunc);
   if (fnumgrad) return fnumgrad->NCalls();
   const ROOT::Math::FitMethodGradFunction * ffitmethod = dynamic_cast<const ROOT::Math::FitMethodGradFunction *>(fObjFunc);
   if (ffitmethod) return ffitmethod->NCalls();   
   // not supported in the other case
   return 0; 
}

bool GSLMinimizer::Minimize() { 
   // set initial parameters of the minimizer

   if (fGSLMultiMin == 0) return false; 
   if (fObjFunc == 0) { 
      MATH_ERROR_MSG("GSLMinimizer::Minimize","Function has not been set");
      return false; 
   }

   unsigned int npar = fValues.size(); 
   if (npar == 0 || npar < fObjFunc->NDim()  ) { 
      MATH_ERROR_MSGVAL("GSLMinimizer::Minimize","Wrong number of parameters",npar);
      return false;
   }

   // use a global step size = modules of  step vectors 
   double stepSize = 0; 
   for (unsigned int i = 0; i < fSteps.size(); ++i)  
      stepSize += fSteps[i]*fSteps[i]; 
   stepSize = std::sqrt(stepSize);

   const double eps = std::numeric_limits<double>::epsilon();  
   if (stepSize < eps) {
      MATH_ERROR_MSGVAL("GSLMinimizer::Minimize","Step size is too small",stepSize);
      return false;
   }

   // check if a transformation is needed 
   bool doTransform = (fBounds.size() > 0); 
   unsigned int ivar = 0; 
   while (!doTransform && ivar < fVarTypes.size() ) {
      doTransform = (fVarTypes[ivar++] != kDefault );
   }


   std::vector<double> startValues(fValues.begin(), fValues.end() );

   MinimTransformFunction * trFunc  = 0; 

   // in case of transformation wrap objective function in a new transformation function
   // and transform from external variables  to internals one
   if (doTransform)   {   
      trFunc =  new MinimTransformFunction ( fObjFunc, fVarTypes, fValues, fBounds ); 
      trFunc->InvTransformation(&fValues.front(), &startValues[0]); 
      startValues.resize( trFunc->NDim() );
      fObjFunc = trFunc; 
   }

//    std::cout << " f has transform " << doTransform << "  " << fBounds.size() << "   " << startValues.size() <<  " ndim " << fObjFunc->NDim() << std::endl;   std::cout << "InitialValues external : "; 
//    for (int i = 0; i < fValues.size(); ++i) std::cout << fValues[i] << "  "; 
//    std::cout << "\n";
//    std::cout << "InitialValues internal : "; 
//    for (int i = 0; i < startValues.size(); ++i) std::cout << startValues[i] << "  "; 
//    std::cout << "\n";


   // set parameters in internal GSL minimization class 
   fGSLMultiMin->Set(*fObjFunc, &startValues.front(), stepSize, fLSTolerance ); 


   int debugLevel = PrintLevel(); 

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLMinimizer " << fGSLMultiMin->Name() << std::endl; 


   //std::cout <<"print Level " << debugLevel << std::endl; 
   //debugLevel = 3; 

   // start iteration 
   unsigned  int iter = 0; 
   int status; 
   bool minFound = false; 
   bool iterFailed = false; 
   do { 
      status = fGSLMultiMin->Iterate(); 
      if (status) { 
         iterFailed = true;
         break; 
      }

      status = fGSLMultiMin->TestGradient( Tolerance() );
      if (status == GSL_SUCCESS) {
         minFound = true; 
      }

      if (debugLevel >=3) { 
         std::cout << "----------> Iteration " << iter << std::endl; 
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << fGSLMultiMin->Minimum() << std::endl; 
         std::cout.precision(pr);
         std::cout << "            X Values : "; 
         const double * xtmp = fGSLMultiMin->X();
         std::cout << std::endl; 
         if (trFunc != 0 ) { 
            xtmp  = trFunc->Transformation(xtmp);  
         }
         for (unsigned int i = 0; i < NDim(); ++i) {
            std::cout << " " << fNames[i] << " = " << xtmp[i];
         // avoid nan
         // if (std::isnan(xtmp[i])) status = -11;
         }
         std::cout << std::endl; 
      }


      iter++;


   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );

   // save state with values and function value
   double * x = fGSLMultiMin->X(); 
   if (x == 0) return false; 

   // check to see if a transformation need to be applied 
   if (trFunc != 0) { 
      const double * xtrans = trFunc->Transformation(x);  
      assert(fValues.size() == trFunc->NTot() ); 
      assert( trFunc->NTot() == NDim() );
      std::copy(xtrans, xtrans + trFunc->NTot(),  fValues.begin() ); 
   }
   else { 
      // case of no transformation applied 
      assert( fValues.size() == NDim() ); 
      std::copy(x, x + NDim(),  fValues.begin() ); 
   }

   fMinVal =  fGSLMultiMin->Minimum(); 

   fStatus = status; 

      
   if (minFound) { 
      if (debugLevel >=1 ) { 
         std::cout << "GSLMinimizer: Minimum Found" << std::endl;  
         int pr = std::cout.precision(18);
         std::cout << "FVAL         = " << fMinVal << std::endl;
         std::cout.precision(pr);
//      std::cout << "Edm   = " << fState.Edm() << std::endl;
         std::cout << "Niterations  = " << iter << std::endl;
         unsigned int ncalls = NCalls(); 
         if (ncalls) std::cout << "NCalls     = " << ncalls << std::endl;
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << fNames[i] << "\t  = " << fValues[i] << std::endl; 
      }
      return true; 
   }
   else { 
      if (debugLevel >= -1 ) { 
         std::cout << "GSLMinimizer: Minimization did not converge" << std::endl;  
         if (iterFailed) { 
            if (status == GSL_ENOPROG) // case status 27
               std::cout << "\t Iteration is not making progress towards solution" << std::endl;
            else 
               std::cout << "\t Iteration failed with status " << status << std::endl;

            if (debugLevel >= 1) {
               double * g = fGSLMultiMin->Gradient();
               double dg2 = 0; 
               for (unsigned int i = 0; i < fDim; ++i) dg2 += g[i] * g[1];  
               std::cout << "Grad module is " << std::sqrt(dg2) << std::endl; 
               for (unsigned int i = 0; i < fDim; ++i) 
                  std::cout << fNames[i] << "\t  = " << fValues[i] << std::endl; 
               std::cout << "FVAL         = " << fMinVal << std::endl;
//      std::cout << "Edm   = " << fState.Edm() << std::endl;
               std::cout << "Niterations  = " << iter << std::endl;
            }
         }
      }
      return false; 
   }
   return false; 
}

const double * GSLMinimizer::MinGradient() const {
   // return gradient (internal values) 
   return fGSLMultiMin->Gradient(); 
}

const MinimTransformFunction * GSLMinimizer::TransformFunction() const { 
   return dynamic_cast<const MinimTransformFunction *>(fObjFunc);
}

   } // end namespace Math

} // end namespace ROOT

