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

#include "TGraph.h" // needed for scan 
#include "TError.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <functional>
#include <cmath>

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

#define USE_STATIC_TMINUIT

ROOT::Math::IMultiGenFunction * TMinuitMinimizer::fgFunc = 0; 
TMinuit * TMinuitMinimizer::fgMinuit = 0;
bool TMinuitMinimizer::fgUsed = false; 

ClassImp(TMinuitMinimizer)


TMinuitMinimizer::TMinuitMinimizer(ROOT::Minuit::EMinimizerType type ) : 
   fUsed(false),
   fMinosRun(false),
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
   fUsed(false),
   fMinosRun(false),
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
   if (algoname == "migradimproved" ) algoType = ROOT::Minuit::kMigradImproved; 
   if (algoname == "scan" )           algoType = ROOT::Minuit::kScan; 
   if (algoname == "seek" )           algoType = ROOT::Minuit::kSeek; 

   fType = algoType; 
}

TMinuitMinimizer::~TMinuitMinimizer() 
{
   // Destructor implementation.
#ifndef USE_STATIC_TMINUIT
   if (fMinuit) delete fMinuit; 
#endif
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
   if (fgMinuit == 0) {
      fgUsed = false;
      fgMinuit =  new TMinuit(fDim);
   }
   else if (fgMinuit->GetNumPars() != int(fDim) ) { 
      delete fgMinuit; 
      fgUsed = false;
      fgMinuit =  new TMinuit(fDim);
   }

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

   // switch off gradient calculations
   fMinuit->mnexcm("SET NOGrad",arglist,0,ierr);
}

void TMinuitMinimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & func) { 
   // Set the objective function to be minimized, by passing a function object implement the 
   // multi-dim gradient Function interface. In this case the function derivatives are provided  
   // by the user via this interface and there not calculated by Minuit. 

   fDim = func.NDim(); 

#ifdef USE_STATIC_TMINUIT
   if (fgMinuit == 0) {
      fgUsed = false; 
      fgMinuit =  new TMinuit(fDim);
   }
   else if (fgMinuit->GetNumPars() != int(fDim) ) { 
      delete fgMinuit; 
      fgUsed = false; 
      fgMinuit =  new TMinuit(fDim);
   }

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
   // by default do not check gradient calculation 
   // it cannot be done here, check can be done only after having defined the parameters
   arglist[0] = 1; 
   fMinuit->mnexcm("SET GRAD",arglist,1,ierr);
}

void TMinuitMinimizer::Fcn( int &, double * , double & f, double * x , int /* iflag */) { 
   // implementation of FCN static function used internally by TMinuit.
   // Adapt IMultiGenFunction interface to TMinuit FCN static function
   f = fgFunc->operator()(x);
}

void TMinuitMinimizer::FcnGrad( int &, double * g, double & f, double * x , int iflag ) { 
   // implementation of FCN static function used internally by TMinuit.
   // Adapt IMultiGradFunction interface to TMinuit FCN static function in the case of user 
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
      Error("SetVariable","invalid TMinuit pointer. Need to call first SetFunction"); 
      return false; 
   }

#ifdef USE_STATIC_TMINUIT
   fUsed = fgUsed; 
#endif

   // clear after minimization when setting params
   if (fUsed) DoClear(); 


   fMinuit->DefineParameter(ivar , name.c_str(), val, step, 0., 0. ); 
   return true; 
}

bool TMinuitMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper) { 
   // set a limited variable.
   if (fMinuit == 0) { 
      Error("SetVariable","invalid TMinuit pointer. Need to call first SetFunction"); 
      return false; 
   }

#ifdef USE_STATIC_TMINUIT
   fUsed = fgUsed; 
#endif

   // clear after minimization when setting params
   if (fUsed) DoClear(); 

   fMinuit->DefineParameter(ivar, name.c_str(), val, step, lower, upper ); 
   return true; 
}
#ifdef LATER
bool Minuit2Minimizer::SetLowerLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower ) {
    // add a lower bounded variable as a double bound one, using a very large number for the upper limit

   if (fMinuit == 0) { 
      Error("SetVariable","invalid TMinuit pointer. Need to call first SetFunction"); 
      return false; 
   }

#ifdef USE_STATIC_TMINUIT
   fUsed = fgUsed; 
#endif

   // clear after minimization when setting params
   if (fUsed) DoClear(); 

   double s = val-lower; 
   double upper = s*1.0E15; 
   if (s != 0)  upper = 1.0E15;
   return SetLimitedVariable(ivar, name, val, step, lower,upper);
}
#endif


bool TMinuitMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) { 
   // set a fixed variable.
   if (fMinuit == 0) { 
      Error("SetVariable","invalid TMinuit pointer. Need to call first SetFunction"); 
      return false; 
   }

   // clear after minimization when setting params
#ifdef USE_STATIC_TMINUIT
   fUsed = fgUsed; 
#endif

   // clear after minimization when setting params
   if (fUsed) DoClear(); 

   // put an arbitrary step (0.1*abs(value) otherwise TMinuit consider the parameter as constant
   // constant parameters are treated differently (they are ignored inside TMinuit and not considered in the
   // total list of parameters) 
   double step = ( val != 0) ? 0.1 * std::abs(val) : 0.1;
   fMinuit->DefineParameter(ivar, name.c_str(), val, step, 0., 0. ); 
   fMinuit->FixParameter(ivar);
   return true; 
}

bool TMinuitMinimizer::SetVariableValue(unsigned int ivar, double val) { 
   // set the value of an existing variable
   // parameter must exist or return false

   if (fMinuit == 0) { 
      Error("SetVariable","invalid TMinuit pointer. Need to call first SetFunction"); 
      return false; 
   }

   double arglist[2]; 
   int ierr = 0; 
   arglist[0] = ivar+1;  // TMinuit starts from 1 
   arglist[1] = val;
   fMinuit->mnexcm("SET PAR",arglist,2,ierr);
   return (ierr==0);
}

std::string TMinuitMinimizer::VariableName(unsigned int ivar) const { 
   // return the variable name
   if (!fMinuit || (int) ivar > fMinuit->fNu) return std::string();
return std::string(fMinuit->fCpnam[ivar]);
}

int TMinuitMinimizer::VariableIndex(const std::string & ) const { 
   // return variable index
   Error("TMinuitMinimizer::VariableIndex"," find index of a variable from its name  is not implemented in TMinuit");
   return -1;
}

bool TMinuitMinimizer::Minimize() { 
   // perform the minimization using the algorithm chosen previously by the user  
   // By default Migrad is used. 
   // Return true if the found minimum is valid and update internal chached values of 
   // minimum values, errors and covariance matrix. 
   // Status of minimizer is set to: 
   // migradResult + 10*minosResult + 100*hesseResult + 1000*improveResult


   if (fMinuit == 0) { 
      Error("Minimize","invalid TMinuit pointer. Need to call first SetFunction and SetVariable"); 
      return false; 
   }


   // total number of parameter defined in Minuit is fNu
   if (fMinuit->fNu <  static_cast<int>(fDim) ) { 
      Error("Minimize","The total number of defined parameters is different than the function dimension, npar = %d, dim = %d",fMinuit->fNu, fDim);
      return false; 
   }

   double arglist[10]; 
   int ierr = 0; 


   // set error and print level 
   arglist[0] = ErrorDef(); 
   fMinuit->mnexcm("SET Err",arglist,1,ierr);

   int printlevel = PrintLevel(); 
   arglist[0] = printlevel - 1;
   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

   // suppress warning in case Printlevel() == 0 
   if (printlevel == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);

   // set precision if needed
   if (Precision() > 0)  { 
      arglist[0] = Precision();
      fMinuit->mnexcm("SET EPS",arglist,1,ierr);
   }

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
   case ROOT::Minuit::kScan: 
      // case of Scan (scan all parameters with default values)
      nargs = 0; 
      fMinuit->mnexcm("SCAN",arglist,nargs,ierr);
      break; 
   case ROOT::Minuit::kSeek: 
      // case of Seek (random find minimum in a hypercube around current parameter values
      // use Tolerance as measures for standard deviation (if < 1) used default value in Minuit ( supposed to be  3)
      nargs = 1; 
      if (arglist[1] >= 1.) nargs = 2; 
      fMinuit->mnexcm("SEEK",arglist,nargs,ierr);
      break; 
   default: 
      // default: use Migrad 
      fMinuit->mnexcm("MIGRAD",arglist,nargs,ierr);

   }

#ifdef USE_STATIC_TMINUIT
   fgUsed = true; 
#endif
   fUsed = true;

   fStatus = ierr; 
   int minErrStatus = ierr;

   // run improved if needed
   if (ierr == 0 && fType == ROOT::Minuit::kMigradImproved) {
      fMinuit->mnexcm("IMPROVE",arglist,1,ierr);
      fStatus += 1000*ierr; 
   }

   // check if Hesse needs to be run 
   if (ierr == 0 && IsValidError() ) { 
      fMinuit->mnexcm("HESSE",arglist,1,ierr);
      fStatus += 100*ierr; 
   }

   // retrieve parameters and errors  from TMinuit
   RetrieveParams(); 

   if (minErrStatus == 0) { 

      // store global min results (only if minimization is OK)
      // ignore cases when Hesse or IMprove return error different than zero
      RetrieveErrorMatrix(); 

      // need to re-run Minos again if requested
      fMinosRun = false; 

      return true;

   }
   return false; 

}

void TMinuitMinimizer::RetrieveParams() {
   // retrieve from TMinuit minimum parameter values 
   // and errors

   assert(fMinuit != 0); 

   // get parameter values 
   if (fParams.size() != fDim) fParams.resize( fDim); 
   if (fErrors.size() != fDim) fErrors.resize( fDim); 
   for (unsigned int i = 0; i < fDim; ++i) { 
      fMinuit->GetParameter( i, fParams[i], fErrors[i]);
   }
}

void TMinuitMinimizer::RetrieveErrorMatrix() {
   // get covariance error matrix from TMinuit
   // when some parameters are fixed filled the corresponding rows and column with zero's

   assert(fMinuit != 0); 

   unsigned int nfree = NFree();

   unsigned int ndim2 = fDim*fDim; 
   if (fCovar.size() != ndim2 )  fCovar.resize(fDim*fDim); 
   if (nfree >= fDim) { // no fixed parameters 
      fMinuit->mnemat(&fCovar.front(), fDim); 
   } 
   else { 
      // case of fixed params need to take care 
      std::vector<double> tmpMat(nfree*nfree); 
      fMinuit->mnemat(&tmpMat.front(), nfree); 
      
      unsigned int l = 0; 
      for (unsigned int i = 0; i < fDim; ++i) { 
         
         if ( fMinuit->fNiofex[i] > 0 ) {  // not fixed ?
            unsigned int m = 0; 
            for (unsigned int j = 0; j <= i; ++j) { 
               if ( fMinuit->fNiofex[j] > 0 ) {  //not fixed
                  fCovar[i*fDim + j] = tmpMat[l*nfree + m];
                  fCovar[j*fDim + i] = fCovar[i*fDim + j]; 
                  m++;
               }
            }
            l++;
         }
      }
      
   }
}

unsigned int TMinuitMinimizer::NCalls() const { 
   // return total number of function calls
   if (fMinuit == 0) return 0; 
   return fMinuit->fNfcn;
}

double TMinuitMinimizer::MinValue() const { 
   // return minimum function value

   // use part of code from mnstat
   if (!fMinuit) return 0; 
   double minval = fMinuit->fAmin; 
   if (minval == fMinuit->fUndefi) return 0; 
   return minval; 
}

double TMinuitMinimizer::Edm() const { 
   // return expected distance from the minimum

   // use part of code from mnstat
   if (!fMinuit) return -1; 
   if (fMinuit->fAmin == fMinuit->fUndefi || fMinuit->fEDM == fMinuit->fBigedm) return fMinuit->fUp; 
   return fMinuit->fEDM; 
}

unsigned int TMinuitMinimizer::NFree() const { 
    // return number of free parameters 
   if (!fMinuit) return 0; 
   if (fMinuit->fNpar < 0) return 0; 
   return fMinuit->fNpar; 
}

int TMinuitMinimizer::CovMatrixStatus() const { 
   // return status of covariance matrix 
   //           status:  0= not calculated at all
   //                    1= approximation only, not accurate
   //                    2= full matrix, but forced positive-definite
   //                    3= full accurate covariance matrix

   // use part of code from mnstat
   if (!fMinuit) return 0; 
   if (fMinuit->fAmin == fMinuit->fUndefi) return 0; 
   return fMinuit->fISW[1];
}

double TMinuitMinimizer::GlobalCC(unsigned int i) const { 
   // global correlation coefficient for parameter i 
   if (!fMinuit) return 0; 
   if (!fMinuit->fGlobcc) return 0;
   if (int(i) >= fMinuit->fNu) return 0; 
   // get internal number in Minuit
   int iin = fMinuit->fNiofex[i];  
   // index in TMinuit starts from 1 
   if (iin < 1) return 0; 
   return fMinuit->fGlobcc[iin-1];   
}

bool TMinuitMinimizer::GetMinosError(unsigned int i, double & errLow, double & errUp, int ) { 
   // Perform Minos analysis for the given parameter  i 

   if (fMinuit == 0) { 
      Error("GetMinosError","invalid TMinuit pointer. Need to call first SetFunction and SetVariable"); 
      return false; 
   }

   double arglist[2];
   int ierr = 0; 

   // if Minos is not run run it 
   if (!fMinosRun) { 

      // set error and print level 
      arglist[0] = ErrorDef(); 
      fMinuit->mnexcm("SET Err",arglist,1,ierr);
      
      arglist[0] = PrintLevel()-1; 
      fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

      // suppress warning in case Printlevel() == 0 
      if (PrintLevel() == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);

      // set precision if needed
      if (Precision() > 0)  { 
         arglist[0] = Precision();
         fMinuit->mnexcm("SET EPS",arglist,1,ierr);
      }

   }

   // syntax of MINOS is MINOS [maxcalls] [parno]
   // if parno = 0 all parameters are done 
   arglist[0] = MaxFunctionCalls(); 
   arglist[1] = i+1;  // par number starts from 1 in TMInuit
   
   int nargs = 2; 
   fMinuit->mnexcm("MINOS",arglist,nargs,ierr);
   fStatus += 10*ierr;

   fMinosRun = true; 

   double errParab = 0; 
   double gcor = 0; 
   // what returns if parameter fixed or constant or at limit ? 
   fMinuit->mnerrs(i,errUp,errLow, errParab, gcor); 

   if (fStatus%100 != 0 ) return false; 
   return true; 

}

void TMinuitMinimizer::DoClear() { 
   // reset TMinuit

   fMinuit->mncler();
   
   //reset the internal Minuit random generator to its initial state
   double val = 3;
   int inseed = 12345;
   fMinuit->mnrn15(val,inseed);

   fUsed = false; 

#ifdef USE_STATIC_TMINUIT
   fgUsed = false; 
#endif

}


void TMinuitMinimizer::PrintResults() { 
   // print-out results using classic Minuit format (mnprin)
   if (fMinuit == 0) return; 

   // print minimizer result
   if (PrintLevel() > 2) 
      fMinuit->mnprin(4,fMinuit->fAmin);
   else
      fMinuit->mnprin(3,fMinuit->fAmin);
}


bool TMinuitMinimizer::Contour(unsigned int ipar, unsigned int jpar, unsigned int &npoints, double * x, double * y) {
   // contour plot for parameter i and j
   // need a valid FunctionMinimum otherwise exits
   if (fMinuit == 0) { 
      Error("TMinuitMinimizer::Contour"," invalid TMinuit instance");
      return false;
   }

   // set error and print level 
   double arglist[1];
   int ierr = 0; 
   arglist[0] = ErrorDef(); 
   fMinuit->mnexcm("SET Err",arglist,1,ierr);
      
   arglist[0] = PrintLevel()-1; 
   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

   // suppress warning in case Printlevel() == 0 
   if (PrintLevel() == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);

   // set precision if needed
   if (Precision() > 0)  { 
      arglist[0] = Precision();
      fMinuit->mnexcm("SET EPS",arglist,1,ierr);
   }


   if (npoints < 4) { 
      Error("Contour","Cannot make contour with so few points");
      return false; 
   }
   int npfound = 0; 
   npoints -= 1;   // remove always one point in TMinuit
   // parameter numbers in mncont start from zero
   fMinuit->mncont( ipar,jpar,npoints, x, y,npfound); 
   if (npfound<4) {
      // mncont did go wrong
      Error("Contour","Cannot find more than 4 points");
      return false;
   }
   if (npfound!=(int)npoints) {
      // mncont did go wrong
      Warning("Contour","Returning only %d points ",npfound);
      npoints = npfound;
   }
   return true; 
   
}

bool TMinuitMinimizer::Scan(unsigned int ipar, unsigned int & nstep, double * x, double * y, double xmin, double xmax) { 
   // scan a parameter (variable) around the minimum value
   // the parameters must have been set before 
   // if xmin=0 && xmax == 0  by default scan around 2 sigma of the error
   // if the errors  are also zero then scan from min and max of parameter range
   // (if parameters are limited Minuit scan from min and max instead of 2 sigma by default)
   // (force in that case to use errors)

   // scan is not implemented for TMinuit, the way to return the array is only via the graph 
   if (fMinuit == 0) { 
      Error("TMinuitMinimizer::Scan"," invalid TMinuit instance");
      return false;
   }

   // case of default xmin and xmax
   if (xmin >= xmax && (int) ipar < fMinuit->GetNumPars() ) { 
      double val = 0; double err = 0;
      TString name; 
      double xlow = 0; double xup = 0 ;
      int iuint = 0; 
      // in mnpout index starts from ze
      fMinuit->mnpout( ipar, name, val, err, xlow, xup, iuint);
      // redefine 2 sigma for all parameters by default (TMinuit does 1 sigma and range if limited)
      if (iuint > 0 && err > 0) { 
         xmin = val - 2.*err; 
         xmax = val + 2 * err; 
      }
   }
   
   double arglist[4]; 
   int ierr = 0; 

   arglist[0] = PrintLevel()-1; 
   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);
   // suppress warning in case Printlevel() == 0 
   if (PrintLevel() == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);

   // set precision if needed
   if (Precision() > 0)  { 
      arglist[0] = Precision();
      fMinuit->mnexcm("SET EPS",arglist,1,ierr);
   }

   if (nstep == 0) return false; 
   arglist[0] = ipar+1;  // TMinuit starts from 1 
   arglist[1] = nstep+2; // TMinuit deletes two points
   int nargs = 2; 
   if (xmax > xmin ) { 
      arglist[2] = xmin; 
      arglist[3] = xmax; 
      nargs = 4; 
   }
   fMinuit->mnexcm("SCAN",arglist,nargs,ierr);
   if (ierr) { 
      Error("TMinuitMinimizer::Scan"," Error executing command SCAN");
      return false;      
   }
   // get TGraph object
   TGraph * gr = dynamic_cast<TGraph *>(fMinuit->GetPlot() );
   if (!gr) {
      Error("TMinuitMinimizer::Scan"," Error in returned graph object");
      return false;      
   }
   nstep = std::min(gr->GetN(), (int) nstep); 


   std::copy(gr->GetX(), gr->GetX()+nstep, x);
   std::copy(gr->GetY(), gr->GetY()+nstep, y);
   nstep = gr->GetN(); 
   return true; 
}

bool TMinuitMinimizer::Hesse() { 
   // perform calculation of Hessian

   if (fMinuit == 0) { 
      Error("Hesse","invalid TMinuit pointer. Need to call first SetFunction and SetVariable"); 
      return false; 
   }


   double arglist[10]; 
   int ierr = 0; 

   // set error and print level 
   arglist[0] = ErrorDef(); 
   fMinuit->mnexcm("SET ERR",arglist,1,ierr);

   int printlevel = PrintLevel(); 
   arglist[0] = printlevel - 1;
   fMinuit->mnexcm("SET PRINT",arglist,1,ierr);

   // suppress warning in case Printlevel() == 0 
   if (printlevel == 0)    fMinuit->mnexcm("SET NOW",arglist,0,ierr);

   // set precision if needed
   if (Precision() > 0)  { 
      arglist[0] = Precision();
      fMinuit->mnexcm("SET EPS",arglist,1,ierr);
   }

   arglist[0] = MaxFunctionCalls(); 

   fMinuit->mnexcm("HESSE",arglist,1,ierr);
   fStatus += 100*ierr; 

   if (ierr != 0) return false;    

   // retrieve results (parameter and error matrix)
   // only if result is OK

   RetrieveParams(); 
   RetrieveErrorMatrix(); 

   return true;
}


//    } // end namespace Fit

// } // end namespace ROOT

