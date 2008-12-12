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

#include "Math/NumGradFunction.h"

#include <cassert>

#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here

namespace ROOT { 

   namespace Math { 


GSLMinimizer::GSLMinimizer( ROOT::Math::EGSLMinimizerType type) : 
   fDim(0), 
   fObjFunc(0)
{
   // Constructor implementation : create GSLMultiMin wrapper object
   //std::cout << "create GSL Minimizer of type " << type << std::endl;

   fGSLMultiMin = new GSLMultiMinimizer((ROOT::Math::EGSLMinimizerType) type); 
   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   fLSTolerance = 0.1; // use 10**-4 
   SetMaxIterations(1000);
   SetPrintLevel(3);
}

GSLMinimizer::GSLMinimizer( const char *  type) : 
   fDim(0), 
   fObjFunc(0)
{
   // Constructor implementation from a string 
   std::string algoname(type);
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower ); 

   ROOT::Math::EGSLMinimizerType algo =  kConjugateFR;   // default value 
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
   SetMaxIterations(1000);
   SetPrintLevel(3);
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
   }
   else { 
      fValues[ivar] = val; 
      fNames[ivar] = name;
      fSteps[ivar] = step; 
   }
   return true; 
}

bool GSLMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set variable value in minimizer 
   // no transformation implemented - so far
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
   // need to calculate numerical the derivatives since are not supported
   fObjFunc = new MultiNumGradFunction( func); 
   fDim = fObjFunc->NDim(); 
}

void GSLMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & func) { 
   // set the function to minimizer (need to clone ??)
   fObjFunc = dynamic_cast< const ROOT::Math::IMultiGradFunction *>(func.Clone() ); 
   fDim = func.NDim(); 
}


bool GSLMinimizer::Minimize() { 
   // set initial parameters of the minimizer

   if (fGSLMultiMin == 0) return false; 
   if (fObjFunc == 0) return false; 


   // use a global step size = min (step vectors) 
   double stepSize = 1; 
   for (unsigned int i = 0; i < fSteps.size(); ++i) 
      //stepSize += fSteps[i]; 
      if (fSteps[i] < stepSize) stepSize = fSteps[i]; 

   fGSLMultiMin->Set(*fObjFunc, &fValues.front(), stepSize, fLSTolerance ); 


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

      if (debugLevel >=1) { 
         std::cout << "----------> Iteration " << iter << std::endl; 
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << fGSLMultiMin->Minimum() << std::endl; 
         std::cout.precision(pr);
         std::cout << "            X Values : "; 
         double * x = fGSLMultiMin->X();
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << " " << fNames[i] << " = " << x[i]; 
         std::cout << std::endl; 
      }
      iter++;

   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );

   // save state with values and function value
   double * x = fGSLMultiMin->X(); 
   if (x == 0) return false; 
   std::copy(x, x +fDim, fValues.begin() ); 
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
         for (unsigned int i = 0; i < fDim; ++i) 
            std::cout << fNames[i] << "\t  = " << fValues[i] << std::endl; 
      }
      return true; 
   }
   else { 
      if (debugLevel >= -1 ) { 
         std::cout << "GSLMinimizer: Minimization did not converge" << std::endl;  
         if (iterFailed) { 
            std::cout << "\t Iteration failed with status " << status << std::endl;
            double * g = fGSLMultiMin->Gradient();
            double dg2 = 0; 
            for (unsigned int i = 0; i < fDim; ++i) dg2 += g[i] * g[1];  
            std::cout << "Grad module is " << std::sqrt(dg2) << std::endl; 
         }
         std::cout << "FVAL         = " << fMinVal << std::endl;
//      std::cout << "Edm   = " << fState.Edm() << std::endl;
         std::cout << "Niterations  = " << iter << std::endl;
      }
      return false; 
   }
   return false; 
}

const double * GSLMinimizer::MinGradient() const {
   return fGSLMultiMin->Gradient(); 
}

   } // end namespace Math

} // end namespace ROOT

