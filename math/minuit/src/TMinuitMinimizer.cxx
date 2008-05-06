// @(#)root/minuit:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TMinuitMinimizer

#include "TMinuitMinimizer.h"
#include "Math/IFunction.h"

#include "TMinuit.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <functional>

//______________________________________________________________________________
//
//  TMinuitMinimizer class implementing the ROOT::Math::Minimizer interface using 
//  TMinuit. 
//  This class is normally instantiates using the plug-in manager 
//  (plug-in with name Minuit or TMinuit)
//  In addition the user can choose the minimizer algorithm: Migrad (the default one), Simplex, or Minimize (combined Migrad + Simplex)
//  
//__________________________________________________________________________________________

// initialize the static instances

ROOT::Math::IMultiGenFunction * TMinuitMinimizer::fgFunc = 0; 
TMinuit * TMinuitMinimizer::fgMinuit = 0; 

ClassImp(TMinuitMinimizer)


TMinuitMinimizer::TMinuitMinimizer(ROOT::Minuit::EMinimizerType type ) : 
   fDim(0),
   fStrategy(1),
   fType(type), 
   fMinuit(fgMinuit)
{
   // Constructor for TMinuitMinimier class via an enumeration specifying the minimization 
   // algorithm type. Supported types are : kMigrad, kSimplex, kCombined (a combined 
   // Migrad + Simplex minimization) and kMigradImproved (a Migrad mininimization folloed by an 
   // improved search for global minima). The default type is Migrad (kMigrad). 
}

TMinuitMinimizer::TMinuitMinimizer(const char *  type ) : 
   fDim(0),
   fStrategy(1),
   fMinuit(fgMinuit)
{
   // constructor from a char * for the algorithm type, used by the plug-in manager
   // The names supported (case unsensitive) are: 
   //  Migrad (default), Simplex, Minimize (for the combined Migrad+ Simplex) and Migrad_imp 

   // select type from the string
   std::string algoname(type);
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower ); 

   ROOT::Minuit::EMinimizerType algoType = ROOT::Minuit::kMigrad; 
   if (algoname == "simplex")   algoType = ROOT::Minuit::kSimplex; 
   if (algoname == "minimize" ) algoType = ROOT::Minuit::kCombined; 
   if (algoname == "migrad_imp" ) algoType = ROOT::Minuit::kMigradImproved; 

   fType = algoType; 
}

TMinuitMinimizer::~TMinuitMinimizer() 
{
   // Destructor implementation.
   if (fMinuit) delete fMinuit; 
}

TMinuitMinimizer::TMinuitMinimizer(const TMinuitMinimizer &) : 
   Minimizer()
{
   // Implementation of copy constructor (it is private).
}

TMinuitMinimizer & TMinuitMinimizer::operator = (const TMinuitMinimizer &rhs) 
{
   // Implementation of assignment operator (private)
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}



void TMinuitMinimizer::SetFunction(const  ROOT::Math::IMultiGenFunction & func) { 
   // Set the objective function to be minimized, by passing a function object implement the 
   // basic multi-dim Function interface. In this case the derivatives will be 
   // calculated by Minuit 

   // Here a TMinuit instance is created since only at this point we know the number of parameters 
   // needed to create TMinuit

   fDim = func.NDim(); 

#ifdef USE_STATIC_TMINUIT
   if (fgMinuit == 0) fgMinuit =  new TMinuit(fDim);
   fMinuit = fgMinuit; 
#else
   if (fMinuit) { 
      //std::cout << "delete previously existing TMinuit " << (int) fMinuit << std::endl; 
      delete fMinuit;  
   }
   fMinuit =  new TMinuit(fDim);
#endif
   
   fDim = func.NDim(); 
   
   // assign to the static pointer (NO Thread safety here)
   fgFunc = const_cast<ROOT::Math::IMultiGenFunction *>(&func); 
   fMinuit->SetFCN(&TMinuitMinimizer::Fcn);

   // set print level in TMinuit
   double arglist[1];
   // TMinuit level is shift by 1 -1 means 0;
   arglist[0] = PrintLevel() - 1;
   int ierr= 0; 

   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

}

void TMinuitMinimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & func) { 
   // Set the objective function to be minimized, by passing a function object implement the 
   // multi-dim gradient Function interface. In this case the function derivatives are provided  
   // by the user via this interface and there not calculated by Minuit. 

   fDim = func.NDim(); 

#ifdef USE_STATIC_TMINUIT
   if (fgMinuit == 0) fgMinuit =  new TMinuit(fDim);
   fMinuit = fgMinuit; 
#else
   if (fMinuit) delete fMinuit;  
   fMinuit =  new TMinuit(fDim);
#endif
   
   fDim = func.NDim(); 
   
   // assign to the static pointer (NO Thread safety here)
   fgFunc = const_cast<ROOT::Math::IMultiGradFunction *>(&func); 
   fMinuit->SetFCN(&TMinuitMinimizer::FcnGrad);

   // set print level in TMinuit
   double arglist[1];
   // TMinuit level is shift by 1 -1 means 0;
   arglist[0] = PrintLevel() - 1;
   int ierr= 0; 

   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

   // set gradient 
   // use default case to check for derivative calculations (not force it) 
   fMinuit->mnexcm("SET GRAD",arglist,0,ierr);
}

void TMinuitMinimizer::Fcn( int &, double * , double & f, double * x , int /* iflag */) { 
   // implementation of FCN static function used internally by TMinuit.
   // Adapt IMultiGenFunction interface to TMinuit FCN static function
   f = fgFunc->operator()(x);
}

void TMinuitMinimizer::FcnGrad( int &, double * g, double & f, double * x , int iflag ) { 
   // implementation of FCN static function used internally by TMinuit.
   // Adapt IMultiGradFunciton interface to TMinuit FCN static function in the case of user 
   // provided gradient.
   ROOT::Math::IMultiGradFunction * gFunc = dynamic_cast<ROOT::Math::IMultiGradFunction *> ( fgFunc); 

   assert(gFunc != 0);
   f = gFunc->operator()(x);

   // calculates also derivatives 
   if (iflag == 2) gFunc->Gradient(x,g);
}

bool TMinuitMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) { 
   // set a free variable.
   if (fMinuit == 0) { 
      std::cerr << "TMinuitMinimizer: ERROR : invalid TMinuit pointer. Set function first " << std::endl;
   }
   fMinuit->DefineParameter(ivar , name.c_str(), val, step, 0., 0. ); 
   return true; 
}

bool TMinuitMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper) { 
   // set a limited variable.
   if (fMinuit == 0) { 
      std::cerr << "TMinuitMinimizer: ERROR : invalid TMinuit pointer. Set function first " << std::endl;
   }
   fMinuit->DefineParameter(ivar, name.c_str(), val, step, lower, upper ); 
   return true; 
}
#ifdef LATER
bool Minuit2Minimizer::SetLowerLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower ) {
    // add a lower bounded variable as a double bound one, using a very large number for the upper limit
   double s = val-lower; 
   double upper = s*1.0E15; 
   if (s != 0)  upper = 1.0E15;
   return SetLimitedVariable(ivar, name, val, step, lower,upper);
}
#endif


bool TMinuitMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) { 
   // set a fixed variable.
   if (fMinuit == 0) { 
      std::cerr << "TMinuitMinimizer: ERROR : invalid TMinuit pointer. Set function first " << std::endl;
   }


   fMinuit->DefineParameter(ivar, name.c_str(), val, 0., val, val ); 
   fMinuit->FixParameter(ivar);
   return true; 
}

bool TMinuitMinimizer::Minimize() { 
   // perform the minimization using the algorithm chosen previously by the user  
   // By default Migrad is used. 
   // Return true if the found minimum is valid and update internal chached values of 
   // minimum values, errors and covariance matrix. 

   assert(fMinuit != 0 );


   double arglist[10]; 
   int ierr = 0; 


   // set error and print level 
   arglist[0] = ErrorUp(); 
   fMinuit->mnexcm("SET Err",arglist,1,ierr);

   int printlevel = PrintLevel(); 
   arglist[0] = printlevel - 1;
   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

   // suppress warning in case Printlevel() == 0
   if (printlevel == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);


   arglist[0] = MaxFunctionCalls(); 
   arglist[1] = Tolerance(); 
   
   int nargs = 2; 

   switch (fType){  
   case ROOT::Minuit::kMigrad: 
      // case of Migrad 
      fMinuit->mnexcm("MIGRAD",arglist,nargs,ierr);
      break; 
   case ROOT::Minuit::kCombined: 
      // case of combined (Migrad+ simplex)
      fMinuit->mnexcm("MINIMIZE",arglist,nargs,ierr);
      break; 
   case ROOT::Minuit::kSimplex: 
      // case of Simlex
      fMinuit->mnexcm("SIMPLEX",arglist,nargs,ierr);
      break; 
   default: 
      // default: use Migrad 
      fMinuit->mnexcm("MIGRAD",arglist,nargs,ierr);

   }

   // run improved if needed
   if (ierr == 0 && fType == ROOT::Minuit::kMigradImproved) 
      fMinuit->mnexcm("IMPROVE",arglist,1,ierr);

   // check if Hesse needs to be run 
   if (ierr == 0 && IsValidError() ) { 
      fMinuit->mnexcm("HESSE",arglist,1,ierr);
   }


   int ntot; 
   int istat;
   int nfree; 
   double errdef = 0;
   fMinuit->mnstat(fMinVal,fEdm,errdef,nfree,ntot,istat);
   assert (static_cast<unsigned int>(ntot) == fDim); 
   assert( nfree == fMinuit->GetNumFreePars() );
   fNFree = nfree;
   assert (errdef == ErrorUp());
   

   // get parameter values 
   fParams.resize( fDim); 
   fErrors.resize( fDim); 
   for (unsigned int i = 0; i < fDim; ++i) { 
      fMinuit->GetParameter( i, fParams[i], fErrors[i]);
   }
   // get covariance matrix
   // store global min results (only if minimization is OK)  
   if (ierr == 0) { 
      fCovar.resize(fDim*fDim); 
      fMinuit->mnemat(&fCovar.front(), fDim); 

      // need to re-run Minos again if requested
      fMinosRun = false; 

      return true;
   }


   else {
      return false; 
   }

}

bool TMinuitMinimizer::GetMinosError(unsigned int i, double & errLow, double & errUp) { 
   // Perform Minos analysis for the given parameter  i 

   // if Minos is not run run it 
   if (!fMinosRun) { 
      double arglist[2];
      int ierr = 0; 

      // set error and print level 
      arglist[0] = ErrorUp(); 
      fMinuit->mnexcm("SET Err",arglist,1,ierr);
      
      std::cout << "print level " << PrintLevel() << std::endl;
      arglist[0] = PrintLevel()-1; 
      fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

      arglist[0] = MaxFunctionCalls(); 
      arglist[1] = Tolerance(); 
   
      int nargs = 2; 
      fMinuit->mnexcm("MINOS",arglist,nargs,ierr);
      if (ierr != 0 ) return false; 

      fMinosRun = true; 
   }

   double errParab = 0; 
   double gcor = 0; 
   // what returns if parameter fixed or constant or at limit ? 
   fMinuit->mnerrs(i,errUp,errLow, errParab, gcor); 
   return true; 

}


//    } // end namespace Fit

// } // end namespace ROOT

