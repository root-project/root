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
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <functional>
#include <ctype.h>   // need to use c version of tolower defined here
#include <limits>

namespace ROOT {

   namespace Math {


GSLMinimizer::GSLMinimizer( ROOT::Math::EGSLMinimizerType type) :
   BasicMinimizer()
{
   // Constructor implementation : create GSLMultiMin wrapper object
   //std::cout << "create GSL Minimizer of type " << type << std::endl;

   fGSLMultiMin = new GSLMultiMinimizer((ROOT::Math::EGSLMinimizerType) type);

   fLSTolerance = 0.1; // line search tolerance (use fixed)
   int niter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   if (niter <=0 ) niter = 1000;
   SetMaxIterations(niter);
   SetPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel());
}

GSLMinimizer::GSLMinimizer( const char *  type) :    BasicMinimizer()
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

   fLSTolerance = 0.1; // use 10**-4
   int niter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   if (niter <=0 ) niter = 1000;
   SetMaxIterations(niter);
   SetPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel());
}


GSLMinimizer::~GSLMinimizer () {
   assert(fGSLMultiMin != 0);
   delete fGSLMultiMin;
}



void GSLMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) {
   // set the function to minimizer
   // need to calculate numerically the derivatives: do via class MultiNumGradFunction
   // no need to clone the passed function
   ROOT::Math::MultiNumGradFunction gradFunc(func);
   // function is cloned inside so can be delete afterwards
   // called base class method setfunction
   // (note: write explicitly otherwise it will call back itself)
   BasicMinimizer::SetFunction(gradFunc);
}


unsigned int GSLMinimizer::NCalls() const {
   // return numbr of function calls
   // if method support
   const ROOT::Math::MultiNumGradFunction * fnumgrad = dynamic_cast<const ROOT::Math::MultiNumGradFunction *>(ObjFunction());
   if (fnumgrad) return fnumgrad->NCalls();
   const ROOT::Math::FitMethodGradFunction * ffitmethod = dynamic_cast<const ROOT::Math::FitMethodGradFunction *>(ObjFunction());
   if (ffitmethod) return ffitmethod->NCalls();
   // not supported in the other case
   return 0;
}

bool GSLMinimizer::Minimize() {
   // set initial parameters of the minimizer

   if (fGSLMultiMin == 0) return false;
   const ROOT::Math::IMultiGradFunction * function = GradObjFunction();
   if (function == 0) {
      MATH_ERROR_MSG("GSLMinimizer::Minimize","Function has not been set");
      return false;
   }

   unsigned int npar = NPar();
   unsigned int ndim = NDim();
   if (npar == 0 || npar < NDim()  ) {
      MATH_ERROR_MSGVAL("GSLMinimizer::Minimize","Wrong number of parameters",npar);
      return false;
   }
   if (npar > ndim  ) {
      MATH_WARN_MSGVAL("GSLMinimizer::Minimize","number of parameters larger than function dimension - ignore extra parameters",npar);
   }

   const double eps = std::numeric_limits<double>::epsilon();

   std::vector<double> startValues;
   std::vector<double> steps(StepSizes(), StepSizes()+npar);

   MinimTransformFunction * trFunc  =  CreateTransformation(startValues);
   if (trFunc) {
      function = trFunc;
      // need to transform also  the steps
      trFunc->InvStepTransformation(X(), StepSizes(), &steps[0]);
      steps.resize(trFunc->NDim());
   }

   // in case all parameters are free - just evaluate the function
   if (NFree() == 0) {
      MATH_INFO_MSG("GSLMinimizer::Minimize","There are no free parameter - just compute the function value");
      double fval = (*function)((double*)0);   // no need to pass parameters
      SetFinalValues(&startValues[0]);
      SetMinValue(fval);
      fStatus = 0;
      return true;
   }

   // use a global step size = modules of  step vectors
   double stepSize = 0;
   for (unsigned int i = 0; i < steps.size(); ++i)
      stepSize += steps[i]*steps[i];
   stepSize = std::sqrt(stepSize);
   if (stepSize < eps) {
      MATH_ERROR_MSGVAL("GSLMinimizer::Minimize","Step size is too small",stepSize);
      return false;
   }


   // set parameters in internal GSL minimization class
   fGSLMultiMin->Set(*function, &startValues.front(), stepSize, fLSTolerance );


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

      if (debugLevel >=2) {
         std::cout << "----------> Iteration " << std::setw(4) << iter;
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << fGSLMultiMin->Minimum() << std::endl;
         std::cout.precision(pr);
         if (debugLevel >=3) {
            std::cout << "            Parameter Values : ";
            const double * xtmp = fGSLMultiMin->X();
            std::cout << std::endl;
            if (trFunc != 0 ) {
               xtmp  = trFunc->Transformation(xtmp);
            }
            for (unsigned int i = 0; i < NDim(); ++i) {
               std::cout << " " << VariableName(i) << " = " << xtmp[i];
               // avoid nan
               // if (std::isnan(xtmp[i])) status = -11;
            }
            std::cout << std::endl;
         }
      }


      iter++;


   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );


   // save state with values and function value
   double * x = fGSLMultiMin->X();
   if (x == 0) return false;
   SetFinalValues(x);

   double minVal =  fGSLMultiMin->Minimum();
   SetMinValue(minVal);

   fStatus = status;


   if (minFound) {
      if (debugLevel >=1 ) {
         std::cout << "GSLMinimizer: Minimum Found" << std::endl;
         int pr = std::cout.precision(18);
         std::cout << "FVAL         = " << MinValue() << std::endl;
         std::cout.precision(pr);
//      std::cout << "Edm   = " << fState.Edm() << std::endl;
         std::cout << "Niterations  = " << iter << std::endl;
         unsigned int ncalls = NCalls();
         if (ncalls) std::cout << "NCalls     = " << ncalls << std::endl;
         for (unsigned int i = 0; i < NDim(); ++i)
            std::cout << VariableName(i) << "\t  = " << X()[i] << std::endl;
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
               for (unsigned int i = 0; i < NDim(); ++i) dg2 += g[i] * g[1];
               std::cout << "Grad module is " << std::sqrt(dg2) << std::endl;
               for (unsigned int i = 0; i < NDim(); ++i)
                  std::cout << VariableName(i) << "\t  = " << X()[i] << std::endl;
               std::cout << "FVAL         = " << MinValue() << std::endl;
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


   } // end namespace Math

} // end namespace ROOT

