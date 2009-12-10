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
#include "Math/Error.h"

#include "Math/MinimTransformFunction.h"
#include "Math/MultiNumGradFunction.h"   // needed to use transformation function

#include <iostream> 
#include <cassert>

namespace ROOT { 

   namespace Math { 



// GSLSimAnMinimizer implementation

GSLSimAnMinimizer::GSLSimAnMinimizer( int /* ROOT::Math::EGSLSimAnMinimizerType type */ ) : 
   fDim(0), 
   fOwnFunc(false),
   fObjFunc(0), 
   fMinVal(0)
{
   // Constructor implementation : create GSLMultiFit wrapper object

   fValues.reserve(10); 
   fNames.reserve(10); 
   fSteps.reserve(10); 

   SetMaxIterations(100);
   SetPrintLevel(0);
}

GSLSimAnMinimizer::~GSLSimAnMinimizer () { 
   if ( fOwnFunc   && fObjFunc) delete fObjFunc;
}

bool GSLSimAnMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set variable in minimizer - support only free variables 
   // no transformation implemented - so far
   if (ivar > fValues.size() ) return false; 
   if (ivar == fValues.size() ) { 
      fValues.push_back(val); 
      fNames.push_back(name);
      // step is the simmulated annealing scale
      fSteps.push_back( step ); 
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

bool GSLSimAnMinimizer::SetLowerLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower ) { 
   bool ret =  SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, lower);
   fVarTypes[ivar] = kLowBound; 
   return true;  
}
bool GSLSimAnMinimizer::SetUpperLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double upper) { 
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( upper, upper);
   fVarTypes[ivar] = kUpBound; 
   return true;  
}
bool GSLSimAnMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper ) { 
   bool ret = SetVariable(ivar, name, val, step); 
   if (!ret) return false; 
   fBounds[ivar] = std::make_pair( lower, upper);
   fVarTypes[ivar] = kBounds; 
   return true;  
}



bool GSLSimAnMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) { 
   /// set fixed variable (override if minimizer supports them )
   // use zero step size 
   bool ret = SetVariable(ivar, name, val, 0.); 
   if (!ret) return false; 
   fVarTypes[ivar] = kFix; 
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

void GSLSimAnMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & func ) { 
   // set the function to minimize
   // use the other methods
   SetFunction( static_cast<const ROOT::Math::IMultiGenFunction &>(func) ); 
}


bool GSLSimAnMinimizer::Minimize() { 
   // set initial parameters of the minimizer
   int debugLevel = PrintLevel(); 

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLSimAnMinimizer " << std::endl; 


   // adapt  the steps (use largers) 
   for (unsigned int i = 0; i < fSteps.size() ; ++i) 
      fSteps[i] *= 10; 

   // vector of internal values (copied by default) 
   std::vector<double> xvar (fValues );
   std::vector<double> steps (fSteps);


   // check if a transformation is needed 
   bool doTransform = (fBounds.size() > 0); 
   unsigned int ivar = 0; 
   while (!doTransform && ivar < fVarTypes.size() ) {
      doTransform = (fVarTypes[ivar++] != kDefault );
   }

   // if needed do transformation and wrap objective function in a new transformation function
   // and transform from external variables (and steps)   to internals one 
   MinimTransformFunction * trFunc  = 0; 
   if (doTransform)   {   
      // since objective function is gradient build a gradient function for the transformation
      // although gradient is not needed

      trFunc =  new MinimTransformFunction ( new MultiNumGradFunction( *fObjFunc), fVarTypes, fValues, fBounds ); 

      trFunc->InvTransformation(&fValues.front(), &xvar[0]); 

      // need to transform also  the steps 
      trFunc->InvStepTransformation(&fValues.front(), &fSteps.front(), &steps[0]); 

      xvar.resize( trFunc->NDim() );
      steps.resize( trFunc->NDim() ); 

      fObjFunc = trFunc; 
      fOwnFunc = true; // flag to indicate we need to delete the function
   }

   assert (xvar.size() == steps.size() );         

   // output vector 
   std::vector<double> xmin(xvar.size() ); 
  
   int iret = fSolver.Solve(*fObjFunc, &xvar.front(), &steps.front(), &xmin[0], (debugLevel > 1) );

   fMinVal = (*fObjFunc)(&xmin.front() );

   // get the result (transform if needed) 
   if (trFunc != 0) { 
      const double * xtrans = trFunc->Transformation(&xmin.front());  
      assert(fValues.size() == trFunc->NTot() ); 
      assert( trFunc->NTot() == NDim() );
      std::copy(xtrans, xtrans + trFunc->NTot(),  fValues.begin() ); 
   }
   else { 
      // case of no transformation applied 
      assert( fValues.size() == xmin.size() ); 
      std::copy(xmin.begin(), xmin.end(),  fValues.begin() ); 
   }



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

