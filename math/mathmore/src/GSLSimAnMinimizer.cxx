// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLSimAnMinimizer

#include "Math/GSLSimAnMinimizer.h"
#include "Math/WrappedParamFunction.h"


#include <iostream> 
#include <cassert>

namespace ROOT { 

   namespace Math { 



// GSLSimAnMinimizer implementation

GSLSimAnMinimizer::GSLSimAnMinimizer( int /* ROOT::Math::EGSLSimAnMinimizerType type */ ) : 
   fDim(0), 
   fNFix(0),
   fObjFunc(0)
{
   // Constructor implementation : create GSLMultiFit wrapper object

   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 
   fVarFix.reserve(10); 

   SetMaxIterations(100);
   SetPrintLevel(3);
}

GSLSimAnMinimizer::~GSLSimAnMinimizer () { 
//   if (fObjFunc) delete fObjFunc; 
   // if I have fixed variables need to delete
   if ( fNFix   && fObjFunc) delete fObjFunc;
}

bool GSLSimAnMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set variable in minimizer - support only free variables 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   if (ivar == fValues.size() ) { 
      fValues.push_back(val); 
      fNames.push_back(name);
      // step is the simmulated annealing scale
      fSteps.push_back( step   ); 
      fVarFix.push_back( false);

   }
   else { 
      fValues[ivar] = val; 
      fNames[ivar] = name;
      fSteps[ivar] = step; 
      fVarFix[ivar] = false; 
   }
   return true; 
}


bool GSLSimAnMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) { 
   // set variable in minimizer - support only free variables 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   if (ivar == fValues.size() ) { 
      fValues.push_back(val); 
      fNames.push_back(name);
      // step is the simmulated annealing scale
      fSteps.push_back( 0   ); 
      fVarFix.push_back( true);
   }
   else { 
      fValues[ivar] = val; 
      fNames[ivar] = name;
      fSteps[ivar] = 0; 
      fVarFix[ivar] = true; 
   }
   return true; 
}

bool GSLSimAnMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set variable value in minimizer 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   fValues[ivar] = val; 
   return true; 
}

bool GSLSimAnMinimizer::SetVariableValues( const double * x) { 
   // set all variable values in minimizer 
   if (x == 0) return false; 
   std::copy(x,x+fValues.size(), fValues.begin() );
   return true; 
}

      
void GSLSimAnMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) { 
   // set the function to minimize
   
   // keep pointers to the chi2 function
   fObjFunc = &func; 
   fDim = func.NDim();
}

void GSLSimAnMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & /* func */) { 
   // set the function to minimizer (need to clone ??)
   // not supported yet 
   return; 
}


bool GSLSimAnMinimizer::Minimize() { 
   // set initial parameters of the minimizer
   int debugLevel = PrintLevel(); 

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLSimAnMinimizer " << std::endl; 


   // check for fixed variables
   std::vector<unsigned int> fixedVars; 
   std::vector<unsigned int> freeVars; 
   std::vector<double> fixedValues; 

   //std::cout << " fDim " << fDim << std::endl;
   //std::cout << " fVarFix.size " << fVarFix.size() << std::endl;

   assert ( fVarFix.size() == fDim);

   for (unsigned int i = 0; i < fDim; ++i) { 
      if (fVarFix[i]) { 
         fixedVars.push_back(i); 
         fixedValues.push_back(fValues[i] );
      }
      else 
         freeVars.push_back(i);
   }

   fNFix = fixedVars.size(); 
   // need to create an adapter 
   if ( fNFix > 0) 
      fObjFunc = new ROOT::Math::WrappedParamFunctionGen<const ROOT::Math::IMultiGenFunction *> (fObjFunc, fDim-fNFix, fNFix, &fixedValues.front(), &fixedVars.front() ); 



   // adapt  the steps (use largers) 
   for (unsigned int i = 0; i < fSteps.size() ; ++i) 
      fSteps[i] *= 10; 

   // correct for free fariables
   std::vector<double> xvar;
   std::vector<double> steps;

   for (unsigned int i = 0; i < fDim; ++i) {
      if (!fVarFix[i]) { 
         xvar.push_back(fValues[i] );
         steps.push_back( 10* fSteps[i] ); // adapt the steps
      }
   }




   std::vector<double> xmin(xvar.size() );
   int iret = fSolver.Solve(*fObjFunc, &xvar.front(), &steps.front(), &xmin[0], debugLevel );

   unsigned int j = 0; 
   for (unsigned int i = 0; i < fDim; ++i) {
      if (!fVarFix[i]) { 
         fValues[i] = xmin[j];
         j++;
      }
   }

   //fValues = xmin;


   fMinVal = (*fObjFunc)(&xmin.front() );


   if (debugLevel >=1 ) { 
      if (iret == 0)  
         std::cout << "GSLSimAnMinimizer: Minimum Found" << std::endl;  
      else 
         std::cout << "GSLSimAnMinimizer: Error in solving" << std::endl;  

      int pr = std::cout.precision(18);
      std::cout << "FVAL         = " << fMinVal << std::endl;
      std::cout.precision(pr);
      for (unsigned int i = 0; i < fDim; ++i) 
         std::cout << fNames[i] << "\t  = " << fValues[i] << std::endl; 
   }


   return ( iret == 0) ? true : false; 
}



   } // end namespace Math

} // end namespace ROOT

