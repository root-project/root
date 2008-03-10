// @(#)root/minuit:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TLinearMinimizer

#include "TLinearMinimizer.h"
#include "Math/IParamFunction.h"
#include "TF1.h"
#include "Fit/Chi2FCN.h"

#include "TLinearFitter.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <functional>



// namespace ROOT { 

//    namespace Fit { 


// structure used for creating the TF1 representing the basis functions
// they are the derivatives w.r.t the parameters of the model function 
template<class Func> 
struct BasisFunction { 
   BasisFunction(Func & f, int k) : 
      fKPar(k), 
      fFunc(&f) 
   {}

   double operator() ( double * x, double *)  { 
      return fFunc->ParameterDerivative(x,fKPar); 
   }

   unsigned int fKPar; // param component
   Func * fFunc; 
};


//______________________________________________________________________________
//
//  TLinearMinimizer, simple class implementing the ROOT::Math::Minimizer interface using 
//  TLinearFitter. 
//  This class uses TLinearFitter to find directly (by solving a system of linear equations) 
//  the minimum of a 
//  least-square function which has a linear dependence in the fit parameters. 
//  This class is not used directly, but via the ROOT::Fitter class, when calling the 
//  LinearFit method. It is instantiates using the plug-in manager (plug-in name is "Linear")
//  
//__________________________________________________________________________________________


ClassImp(TLinearMinimizer)


TLinearMinimizer::TLinearMinimizer(int ) : 
   fDim(0),
   fObjFunc(0),
   fFitter(0)
{
   // Default constructor implementation.
   // type is not used - needed for consistency with other minimizer plug-ins
}


TLinearMinimizer::~TLinearMinimizer() 
{
   // Destructor implementation.
   if (fFitter) delete fFitter; 
}

TLinearMinimizer::TLinearMinimizer(const TLinearMinimizer &) : 
   Minimizer()
{
   // Implementation of copy constructor.
}

TLinearMinimizer & TLinearMinimizer::operator = (const TLinearMinimizer &rhs) 
{
   // Implementation of assignment operator.
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}


void TLinearMinimizer::SetFunction(const  IObjFunction & ) { 
   // Set function to be minimized. Flag an error since only support Gradient objective functions

   Error("SetFunction1","Wrong type of function used for Linear fitter");
}


void TLinearMinimizer::SetFunction(const  IGradObjFunction & objfunc) { 
   // Set the function to be minimized. The function must be a Chi2 gradient function 
   // When performing a linear fit we need the basis functions, which are the partial derivatives with respect to the parameters of the model function.

   typedef ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGradFunction> Chi2Func; 
   const Chi2Func * chi2func = dynamic_cast<const Chi2Func *>(&objfunc); 
   if (chi2func ==0) { 
      Error("SetFunction2","Wrong type of function used for Linear fitter");
      return; 
   }
   fObjFunc = chi2func;

   // get model function
   typedef  Chi2Func::IModelFunction ModelFunc; 
   const ModelFunc & modfunc =  chi2func->ModelFunction(); 
   fDim = chi2func->NDim(); // number of parameters
   fNFree = fDim;

   // get the basis functions (derivatives of the modelfunc)
   TObjArray flist; 
   for (unsigned int i = 0; i < fDim; ++i) { 
      BasisFunction<const ModelFunc> bf(modfunc,i); 
      std::string fname = "f" + ROOT::Math::Util::ToString(i);
      TF1 * f = new TF1(fname.c_str(),ROOT::Math::ParamFunctor(bf));
      //f->SetDirectory(0);
      flist.Add(f);
   }

   // create TLinearFitter (do it now because olny now now the coordinate dimensions)
   if (fFitter) delete fFitter; // reset by deleting previous copy
   fFitter = new TLinearFitter( static_cast<const ModelFunc::BaseFunc&>(modfunc).NDim() ); 
   fFitter->StoreData(false); 

   fFitter->SetBasisFunctions(&flist); 

   // get the fitter data
   const ROOT::Fit::BinData & data = chi2func->Data(); 
   // add the data but not store them 
   for (unsigned int i = 0; i < data.Size(); ++i) { 
      double y = 0; 
      const double * x = data.GetPoint(i,y); 
      double ey = 1;
      if (! data.Opt().fErrors1) { 
         ey = data.Error(i); 
      } 
      // interface should take a double *
      fFitter->AddPoint( const_cast<double *>(x) , y, ey); 
   }

}


bool TLinearMinimizer::SetFixedVariable(unsigned int ivar, const std::string & /* name */ , double val) { 
   // set a fixed variable.
   if (!fFitter) return false; 
   fFitter->FixParameter(ivar, val);
   return true; 
}

bool TLinearMinimizer::Minimize() { 
   // find directly the minimum of the chi2 function 
   // solving the linear equation. Use  TVirtualFitter::Eval. 

   if (fFitter == 0 || fObjFunc == 0) return false;

   int iret = fFitter->Eval(); 
   
   if (iret != 0) { 
      Warning("Minimize","TLinearFitter failed in finding the solution");  
      return false; 
   }
   

   // get parameter values 
   fParams.resize( fDim); 
   fErrors.resize( fDim); 
   for (unsigned int i = 0; i < fDim; ++i) { 
      fParams[i] = fFitter->GetParameter( i);
      fErrors[i] = fFitter->GetParError( i ); 
   }
   fCovar.resize(fDim*fDim); 
   double * cov = fFitter->GetCovarianceMatrix();
   std::copy(cov,cov+fDim*fDim,fCovar.begin() );

   // calculate chi2 value
   
   fMinVal = (*fObjFunc)(&fParams.front());

   return true;

}


//    } // end namespace Fit

// } // end namespace ROOT

