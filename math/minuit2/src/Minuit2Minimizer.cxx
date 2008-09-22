// @(#)root/minuit2:$Id$
// Author: L. Moneta Wed Oct 18 11:48:00 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class Minuit2Minimizer

#include "Minuit2/Minuit2Minimizer.h"

#include "Math/IFunction.h"

#include "Minuit2/FCNAdapter.h"
#include "Minuit2/FumiliFCNAdapter.h"
#include "Minuit2/FCNGradAdapter.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/MnHesse.h"
#include "Minuit2/MinuitParameter.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/VariableMetricMinimizer.h"
#include "Minuit2/SimplexMinimizer.h"
#include "Minuit2/CombinedMinimizer.h"
#include "Minuit2/ScanMinimizer.h"
#include "Minuit2/FumiliMinimizer.h"
 


#include <cassert> 
#include <iostream> 
#include <algorithm>
#include <functional>


namespace ROOT { 

namespace Minuit2 { 


Minuit2Minimizer::Minuit2Minimizer(ROOT::Minuit2::EMinimizerType type ) : 
   fDim(0),
   fMinimizer(0),
   fMinuitFCN(0),
   fMinimum(0)   
{
   // Default constructor implementation depending on minimizer type 
   SetMinimizerType(type); 
}

Minuit2Minimizer::Minuit2Minimizer(const char *  type ) : 
   fDim(0),
   fMinimizer(0),
   fMinuitFCN(0),
   fMinimum(0)   
{   
   // constructor from a string

   std::string algoname(type);
   // tolower() is not an  std function (Windows)
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower ); 

   EMinimizerType algoType = kMigrad; 
   if (algoname == "simplex")   algoType = kSimplex; 
   if (algoname == "minimize" ) algoType = kCombined; 
   if (algoname == "scan" )     algoType = kScan; 
   if (algoname == "fumili" )   algoType = kFumili;
  
   SetMinimizerType(algoType);
}

void Minuit2Minimizer::SetMinimizerType(ROOT::Minuit2::EMinimizerType type) {
   // Set  minimizer algorithm type 
   fUseFumili = false;
   switch (type) { 
   case ROOT::Minuit2::kMigrad: 
      //std::cout << "Minuit2Minimizer: minimize using MIGRAD " << std::endl;
      SetMinimizer( new ROOT::Minuit2::VariableMetricMinimizer() );
      return;
   case ROOT::Minuit2::kSimplex: 
      //std::cout << "Minuit2Minimizer: minimize using SIMPLEX " << std::endl;
      SetMinimizer( new ROOT::Minuit2::SimplexMinimizer() );
      return;
   case ROOT::Minuit2::kCombined: 
      SetMinimizer( new ROOT::Minuit2::CombinedMinimizer() );
      return;
   case ROOT::Minuit2::kScan: 
      SetMinimizer( new ROOT::Minuit2::ScanMinimizer() );
      return;
   case ROOT::Minuit2::kFumili:          
      SetMinimizer( new ROOT::Minuit2::FumiliMinimizer() );
      fUseFumili = true;
      return;
   default: 
      //migrad minimizer
      SetMinimizer( new ROOT::Minuit2::VariableMetricMinimizer() );

   }
}


Minuit2Minimizer::~Minuit2Minimizer() 
{
   // Destructor implementation.
   if (fMinimizer) delete fMinimizer; 
   if (fMinuitFCN) delete fMinuitFCN; 
   if (fMinimum)   delete fMinimum; 
}

Minuit2Minimizer::Minuit2Minimizer(const Minuit2Minimizer &) : 
   ROOT::Math::Minimizer()
{
   // Implementation of copy constructor.
}

Minuit2Minimizer & Minuit2Minimizer::operator = (const Minuit2Minimizer &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


void Minuit2Minimizer::Clear() { 
   // delete the state in case of consecutive minimizations
   fState = MnUserParameterState();
}


// set variables 

bool Minuit2Minimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set a free variable. 
   //Add if not existing or set value if exists already
   // this is implemented in MnUserParameterState::Add
   //   std::cout << " add parameter " << name << "  " <<  val << std::endl;
   fState.Add(name.c_str(), val, step); 
   unsigned int minuit2Index = fState.Index(name.c_str() ); 
   if ( minuit2Index != ivar) 
      std::cout << "Minuit2Minimizer:  WARNING: variable " << name 
                << " has a different index. Correct index is " <<  minuit2Index << std::endl;
   return true; 
}

bool Minuit2Minimizer::SetLowerLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower ) {
   // add a lower bounded variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetLowerLimit(ivar, lower);
   return true;
}

bool Minuit2Minimizer::SetUpperLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double upper ) {
   // add a upper bounded variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetUpperLimit(ivar, upper);
   return true;
}



bool Minuit2Minimizer::SetLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower , double upper) {
   // add a double bound variable
   if (!SetVariable(ivar, name, val, step) ) return false;
   fState.SetLimits(ivar, lower, upper);
   return true;
}

bool Minuit2Minimizer::SetFixedVariable(unsigned int ivar , const std::string & name , double val ) {
   // add a fixed variable
   if (!SetVariable(ivar, name, val, 0.0) ) return false;
   fState.Fix(ivar);
   return true;
}


void Minuit2Minimizer::SetFunction(const  ROOT::Math::IMultiGenFunction & func) { 
   // set function to be minimized
   if (fMinuitFCN) delete fMinuitFCN;
   fDim = func.NDim(); 
   if (!fUseFumili) {
      fMinuitFCN = new ROOT::Minuit2::FCNAdapter<ROOT::Math::IMultiGenFunction> (func, ErrorUp() );
   }
   else { 
      // for Fumili the fit method function interface is required
      const ROOT::Math::FitMethodFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodFunction *>(&func);
      if (!fcnfunc) {
         MN_ERROR_MSG("Minuit2Minimizer: Wrong Fit method function for Fumili");
         return;
      }
      fMinuitFCN = new ROOT::Minuit2::FumiliFCNAdapter<ROOT::Math::FitMethodFunction> (*fcnfunc, fDim, ErrorUp() );
   }
}

void Minuit2Minimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & func) { 
   // set function to be minimized
   fDim = func.NDim(); 
   if (fMinuitFCN) delete fMinuitFCN;
   if (!fUseFumili) { 
      fMinuitFCN = new ROOT::Minuit2::FCNGradAdapter<ROOT::Math::IMultiGradFunction> (func, ErrorUp() );
   }
   else { 
      // for Fumili the fit method function interface is required
      const ROOT::Math::FitMethodGradFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodGradFunction*>(&func);
      if (!fcnfunc) {
         MN_ERROR_MSG("Minuit2Minimizer: Wrong Fit method function for Fumili");
         return;
      }
      fMinuitFCN = new ROOT::Minuit2::FumiliFCNAdapter<ROOT::Math::FitMethodGradFunction> (*fcnfunc, fDim, ErrorUp() );
   }
}
                                   
bool Minuit2Minimizer::Minimize() { 
   // perform the minimization
   // store a copy of FunctionMinimum 
   assert(fMinuitFCN != 0 );
   assert(GetMinimizer() != 0 );
   // delete result of previous minimization
   if (fMinimum) delete fMinimum; 
   fMinimum = 0;


   int maxfcn = MaxFunctionCalls(); 
   double tol = Tolerance();
   int strategy = Strategy(); 
   fMinuitFCN->SetErrorDef(ErrorUp() );

   if (PrintLevel() >=1)
      std::cout << "Minuit2Minimizer: Minimize with max iterations " << maxfcn << " edmval " << tol << " strategy " 
                << strategy << std::endl; 

#ifdef USE_ROOT_ERROR
   // switch off Minuit2 printing
   int prevErrorIgnoreLevel = gErrorIgnoreLevel; 
   if (PrintLevel() ==0)  
      // switch off printing of info messages in Minuit2
      gErrorIgnoreLevel = 1001;
#endif 
      

   const ROOT::Minuit2::FCNGradientBase * gradFCN = dynamic_cast<const ROOT::Minuit2::FCNGradientBase *>( fMinuitFCN ); 
   if ( gradFCN != 0) {
      // use gradient
      //SetPrintLevel(3);
      ROOT::Minuit2::FunctionMinimum min =  GetMinimizer()->Minimize(*gradFCN, fState, ROOT::Minuit2::MnStrategy(strategy), maxfcn, tol);
      fMinimum = new ROOT::Minuit2::FunctionMinimum (min);    
   }
   else {
      ROOT::Minuit2::FunctionMinimum min = GetMinimizer()->Minimize(*GetFCN(), fState, ROOT::Minuit2::MnStrategy(strategy), maxfcn, tol);
      fMinimum = new ROOT::Minuit2::FunctionMinimum (min);    
   }

   // check if Hesse needs to be run 
   if (fMinimum->IsValid() && IsValidError() && fMinimum->State().Error().Dcovar() != 0 ) {
      // run Hesse (Hesse will add results in the last state of fMinimum
      ROOT::Minuit2::MnHesse hesse(strategy );
      hesse( *GetFCN(), *fMinimum, maxfcn); 
   }



#ifdef USE_ROOT_ERROR
//restore previous printing level
   if (PrintLevel() ==0)  
      gErrorIgnoreLevel = prevErrorIgnoreLevel;
#endif 

   
   fState = fMinimum->UserState(); 
   bool ok =  ExamineMinimum(*fMinimum);
   //fMinimum = 0; 
   return ok; 
}

bool  Minuit2Minimizer::ExamineMinimum(const ROOT::Minuit2::FunctionMinimum & min) {  
   /// study the function minimum      
   
   // debug ( print all the states) 
   int debugLevel = PrintLevel(); 
   if (debugLevel >= 3) { 
      
      const std::vector<ROOT::Minuit2::MinimumState>& iterationStates = min.States();
      std::cout << "Number of iterations " << iterationStates.size() << std::endl;
      for (unsigned int i = 0; i <  iterationStates.size(); ++i) {
         //std::cout << iterationStates[i] << std::endl;                                                                       
         const ROOT::Minuit2::MinimumState & st =  iterationStates[i];
         std::cout << "----------> Iteration " << i << std::endl;
         int pr = std::cout.precision(12);
         std::cout << "            FVAL = " << st.Fval() << " Edm = " << st.Edm() << " Nfcn = " << st.NFcn() << std::endl;
         std::cout.precision(pr);
         std::cout << "            Error matrix change = " << st.Error().Dcovar() << std::endl;
         std::cout << "            Parameters : ";
         // need to transform from internal to external 
         for (int j = 0; j < st.size() ; ++j) std::cout << " p" << j << " = " << fState.Int2ext( j, st.Vec()(j) );
         std::cout << std::endl;
      }
   }
   // print result 
   if (min.IsValid() ) {
      if (debugLevel >=1 ) { 
         std::cout << "Minuit2Minimizer: Minimum Found" << std::endl; 
         int pr = std::cout.precision(18);
         std::cout << "FVAL  = " << fState.Fval() << std::endl;
         std::cout << "Edm   = " << fState.Edm() << std::endl;
         std::cout.precision(pr);
         std::cout << "Nfcn  = " << fState.NFcn() << std::endl;
         std::vector<double> par = fState.Params();
         std::vector<double> err = fState.Errors();
         for (unsigned int i = 0; i < fState.Params().size(); ++i) 
            std::cout << fState.Parameter(i).Name() << "\t  = " << par[i] << "\t  +/-  " << err[i] << std::endl; 
      }
      fStatus = 0; 
      return true;
   }
   else { 
      if (debugLevel >= 1)  {
         std::cout << "Minuit2Minimizer::Minimization DID not converge !" << std::endl; 
         std::cout << "FVAL  = " << fState.Fval() << std::endl;
         std::cout << "Edm   = " << fState.Edm() << std::endl;
         std::cout << "Nfcn  = " << fState.NFcn() << std::endl;
      }
      if (min.HasMadePosDefCovar() ) { 
         if (debugLevel >= 1) std::cout << "      Covar was made pos def" << std::endl;
         fStatus = 1; 
         return false; 
      }
      if (min.HesseFailed() ) { 
         if (debugLevel >= 1) std::cout << "      Hesse is not valid" << std::endl;
         fStatus = 2; 
         return false;
      }
      if (min.IsAboveMaxEdm() ) { 
         if (debugLevel >= 1) std::cout << "      Edm is above max" << std::endl;
         fStatus = 3; 
         return false; 
      }
      if (min.HasReachedCallLimit() ) { 
         if (debugLevel >= 1) std::cout << "      Reached call limit" << std::endl;
         fStatus = 4;
         return false; 
      }
      fStatus =  5;
      return false; 
   }
   return true;
}

double Minuit2Minimizer::CovMatrix(unsigned int i, unsigned int j) const { 
   // get value of covariance matrices (transform from external to internal indices)
   if ( i >= fDim || i >= fDim) return 0;  
   if ( Status()  || !fState.HasCovariance()    ) return 0; // no info available when minimization has failed
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   unsigned int l = fState.IntOfExt(j); 
   return fState.Covariance()(k,l); 
}

double Minuit2Minimizer::Correlation(unsigned int i, unsigned int j) const { 
   // get correlation between parameter i and j 
   if ( i >= fDim || i >= fDim) return 0;  
   if ( Status()  || !fState.HasCovariance()    ) return 0; // no info available when minimization has failed
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   if (fState.Parameter(j).IsFixed() || fState.Parameter(j).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   unsigned int l = fState.IntOfExt(j); 
   double cij =  fState.IntCovariance()(k,l); 
   double tmp =  std::sqrt( std::abs ( fState.IntCovariance()(k,k) * fState.IntCovariance()(l,l) ) );
   if (tmp > 0 ) return cij/tmp; 
   return 0; 
}

double Minuit2Minimizer::GlobalCC(unsigned int i) const { 
   // get global correlation coefficient for the parameter i. This is a number between zero and one which gives 
   // the correlation between the i-th parameter  and that linear combination of all other parameters which 
   // is most strongly correlated with i.

   if ( i >= fDim || i >= fDim) return 0;  
    // no info available when minimization has failed or has some problems
   if ( Status()  || !fState.HasGlobalCC()    ) return 0; 
   if (fState.Parameter(i).IsFixed() || fState.Parameter(i).IsConst() ) return 0; 
   unsigned int k = fState.IntOfExt(i); 
   return fState.GlobalCC().GlobalCC()[k]; 
}


bool Minuit2Minimizer::GetMinosError(unsigned int i, double & errLow, double & errUp) { 

   errLow = 0; errUp = 0; 

   assert( fMinuitFCN );

   // need to know if parameter is const or fixed 
   if ( fState.Parameter(i).IsConst() || fState.Parameter(i).IsFixed() ) { 
      return false; 
   }

   int debugLevel = PrintLevel(); 

   // to run minos I need function minimum class 
   // redo minimization from current state
//    ROOT::Minuit2::FunctionMinimum min =  
//       GetMinimizer()->Minimize(*GetFCN(),fState, ROOT::Minuit2::MnStrategy(strategy), MaxFunctionCalls(), Tolerance());
//    fState = min.UserState();
   if (fMinimum == 0) { 
      std::cout << "Minuit2Minimizer::GetMinosErrors:  failed - no function minimum existing" << std::endl;
      return false;
   }
   
   if (!fMinimum->IsValid() ) { 
      std::cout << "Minuit2Minimizer::MINOS failed due to invalid function minimum" << std::endl;
      return false;
   }

   fMinuitFCN->SetErrorDef(ErrorUp() );

   // if error def has been changed update it in FunctionMinimum
   if (ErrorUp() != fMinimum->Up() ) 
      fMinimum->SetErrorDef(ErrorUp() );


   ROOT::Minuit2::MnMinos minos( *fMinuitFCN, *fMinimum);
   // chech if variable is not fixed 

   ROOT::Minuit2::MinosError me = minos.Minos(i);
   // print error message in Minos
   if (debugLevel == 0) {
      if (!me.IsValid() ) { 
         std::cout << "Error running Minos for parameter " << i << std::endl; 
         if ( fStatus%100 == 0 )  fStatus += 10; 
         return false; 
      }
   }
   if (debugLevel >= 1) {
      if (!me.LowerValid() )  
         std::cout << "Minos:  Invalid lower error for parameter " << i << std::endl; 
      if(me.AtLowerLimit()) 
         std::cout << "Minos:  Parameter  is at Lower limit."<<std::endl;
      if(me.AtLowerMaxFcn())
         std::cout << "Minos:  Maximum number of function calls exceeded when running for lower error" <<std::endl;   
      if(me.LowerNewMin() )
         std::cout << "Minos:  New Minimum found while running Minos for lower error" <<std::endl;     
         
      if (!me.UpperValid() )  
         std::cout << "Minos:  Invalid upper error for parameter " << i << std::endl; 
      if(me.AtUpperLimit()) 
         std::cout << "Minos:  Parameter  is at Upper limit."<<std::endl;
      if(me.AtUpperMaxFcn())
         std::cout << "Minos:  Maximum number of function calls exceeded when running for upper error" <<std::endl;   
      if(me.UpperNewMin() )
         std::cout << "Minos:  New Minimum found while running Minos for upper error" <<std::endl;     
         
   }
         
   errLow = me.Lower();
   errUp = me.Upper();
      
   
//    if (debugLevel >= 3) {
//       for(std::vector<ROOT::Minuit2::MinosError>::const_iterator ime = fMinosErrors.begin();
//           ime != fMinosErrors.end(); ime++) 
//          std::cout<<*ime<<std::endl;
//       }

   return true;
} 

} // end namespace Minuit2

} // end namespace ROOT

