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
#include "TUUID.h"
#include "TROOT.h"
#include "Fit/BasicFCN.h"
#include "Fit/BinData.h"
#include "Fit/Chi2FCN.h"

#include "TLinearFitter.h"
#include "TVirtualMutex.h"

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
   BasisFunction(const Func & f, int k) :
      fKPar(k),
      fFunc(&f)
   {}

   double operator() ( double * x, double *)  {
      return fFunc->ParameterDerivative(x,fKPar);
   }

   unsigned int fKPar; // param component
   const Func * fFunc;
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


ClassImp(TLinearMinimizer);


TLinearMinimizer::TLinearMinimizer(int ) :
   fRobust(false),
   fDim(0),
   fNFree(0),
   fMinVal(0),
   fObjFunc(0),
   fFitter(0)
{
   // Default constructor implementation.
   // type is not used - needed for consistency with other minimizer plug-ins
}

TLinearMinimizer::TLinearMinimizer ( const char * type ) :
   fRobust(false),
   fDim(0),
   fNFree(0),
   fMinVal(0),
   fObjFunc(0),
   fFitter(0)
{
   // constructor passing a type of algorithm, (supported now robust via LTS regression)

   // select type from the string
   std::string algoname(type);
   std::transform(algoname.begin(), algoname.end(), algoname.begin(), (int(*)(int)) tolower );

   if (algoname.find("robust") != std::string::npos) fRobust = true;

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


void TLinearMinimizer::SetFunction(const  ROOT::Math::IMultiGenFunction & ) {
   // Set function to be minimized. Flag an error since only support Gradient objective functions

   Error("TLinearMinimizer::SetFunction(IMultiGenFunction)","Wrong type of function used for Linear fitter");
}


void TLinearMinimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & objfunc) {
   // Set the function to be minimized. The function must be a Chi2 gradient function
   // When performing a linear fit we need the basis functions, which are the partial derivatives with respect to the parameters of the model function.

   typedef ROOT::Fit::Chi2FCN<ROOT::Math::IMultiGradFunction> Chi2Func;
   const Chi2Func * chi2func = dynamic_cast<const Chi2Func *>(&objfunc);
   if (chi2func ==0) {
      Error("TLinearMinimizer::SetFunction(IMultiGradFunction)","Wrong type of function used for Linear fitter");
      return;
   }
   fObjFunc = chi2func;

   // need to get the gradient parametric model function
   typedef  ROOT::Math::IParamMultiGradFunction ModelFunc;
   const  ModelFunc * modfunc = dynamic_cast<const ModelFunc*>( &(chi2func->ModelFunction()) );
   assert(modfunc != 0);

   fDim = chi2func->NDim(); // number of parameters
   fNFree = fDim;
   // get the basis functions (derivatives of the modelfunc)
   TObjArray flist(fDim);
   flist.SetOwner(kFALSE);  // we do not want to own the list - it will be owned by the TLinearFitter class
   for (unsigned int i = 0; i < fDim; ++i) {
      // t.b.f: should not create TF1 classes
      // when creating TF1 (if onother function with same name exists it is
      // deleted since it is added in function list in gROOT
      // fix the problem using meaniful names (difficult to re-produce)
      BasisFunction<ModelFunc > bf(*modfunc,i);
      TUUID u;
      std::string fname = "_LinearMinimimizer_BasisFunction_" +
         std::string(u.AsString() );
      TF1 * f = new TF1(fname.c_str(),ROOT::Math::ParamFunctor(bf),0,1,0,1,TF1::EAddToList::kNo);
      flist.Add(f);
   }

   // create TLinearFitter (do it now because olny now now the coordinate dimensions)
   if (fFitter) delete fFitter; // reset by deleting previous copy
   fFitter = new TLinearFitter( static_cast<const ModelFunc::BaseFunc&>(*modfunc).NDim() );

   fFitter->StoreData(fRobust); //  need a copy of data in case of robust fitting

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

   int iret = 0;
   if (!fRobust)
      iret = fFitter->Eval();
   else {
      // robust fitting - get h parameter using tolerance (t.b. improved)
      double h = Tolerance();
      if (PrintLevel() >  0)
         std::cout << "TLinearMinimizer: Robust fitting with h = " << h << std::endl;
      iret = fFitter->EvalRobust(h);
   }
   fStatus = iret;

   if (iret != 0) {
      Warning("Minimize","TLinearFitter failed in finding the solution");
      return false;
   }


   // get parameter values
   fParams.resize( fDim);
   // no error available for robust fitting
   if (!fRobust) fErrors.resize( fDim);
   for (unsigned int i = 0; i < fDim; ++i) {
      fParams[i] = fFitter->GetParameter( i);
      if (!fRobust) fErrors[i] = fFitter->GetParError( i );
   }
   fCovar.resize(fDim*fDim);
   double * cov = fFitter->GetCovarianceMatrix();

   if (!fRobust && cov) std::copy(cov,cov+fDim*fDim,fCovar.begin() );

   // calculate chi2 value

   fMinVal = (*fObjFunc)(&fParams.front());

   return true;

}


//    } // end namespace Fit

// } // end namespace ROOT

