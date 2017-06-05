// @(#)root/fumili:$Id$
// Author: L. Moneta Wed Oct 25 16:28:55 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class TFumiliMinimizer

#include "TFumiliMinimizer.h"
#include "Math/IFunction.h"
#include "Math/Util.h"
#include "TError.h"

#include "TFumili.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <functional>


// setting USE_FUMILI_FUNCTION will use the Derivatives provided by Fumili
// instead of what proided in FitUtil::EvalChi2Residual
// t.d.: use still standard Chi2 but replace model function
// with a gradient function where gradient is computed by TFumili
// since TFumili knows the step size can calculate it better
// Derivative in FUmili are very fast (1 extra call for each parameter)
// + 1 function evaluation
//
//#define USE_FUMILI_FUNCTION
#ifdef USE_FUMILI_FUNCTION
bool gUseFumiliFunction = true;
//#include "FumiliFunction.h"
// fit method function used in TFumiliMinimizer

#include "Fit/PoissonLikelihoodFCN.h"
#include "Fit/LogLikelihoodFCN.h"
#include "Fit/Chi2FCN.h"
#include "TF1.h"
#include "TFumili.h"

template<class MethodFunc>
class FumiliFunction  : public ROOT::Math::FitMethodFunction {

   typedef ROOT::Math::FitMethodFunction::BaseFunction BaseFunction;

public:
   FumiliFunction(TFumili * fumili,  const ROOT::Math::FitMethodFunction * func) :
      ROOT::Math::FitMethodFunction(func->NDim(), func->NPoints() ),
      fFumili(fumili),
      fObjFunc(0)
   {
      fObjFunc = dynamic_cast<const MethodFunc *>(func);
      assert(fObjFunc != 0);

      // create TF1 class from model function
      fModFunc = new TF1("modfunc",ROOT::Math::ParamFunctor( &fObjFunc->ModelFunction() ) );
      fFumili->SetUserFunc(fModFunc);
   }

   ROOT::Math::FitMethodFunction::Type_t Type() const { return fObjFunc->Type();  }

   FumiliFunction * Clone() const { return new FumiliFunction(fFumili, fObjFunc); }


   // recalculate data elemet using Fumili stuff
   double DataElement(const double * /*par */, unsigned int i, double * g) const {

      // parameter values are inside TFumili

      // suppose type is bin likelihood
      unsigned int npar = fObjFunc->NDim();
      double  y = 0;
      double invError = 0;
      const double *x = fObjFunc->Data().GetPoint(i,y,invError);
      double fval  = fFumili->EvalTFN(g,const_cast<double *>( x));
      fFumili->Derivatives(g, const_cast<double *>( x));

      if ( fObjFunc->Type() == ROOT::Math::FitMethodFunction::kLogLikelihood) {
         double logPdf =   y * ROOT::Math::Util::EvalLog( fval) - fval;
         for (unsigned int k = 0; k < npar; ++k) {
            g[k] *= ( y/fval - 1.) ;//* pdfval;
         }

 //         std::cout << "x = " << x[0] << " logPdf = " << logPdf << " grad";
//          for (unsigned int ipar = 0; ipar < npar; ++ipar)
//             std::cout << g[ipar] << "\t";
//          std::cout << std::endl;

         return logPdf;
      }
      else if (fObjFunc->Type() == ROOT::Math::FitMethodFunction::kLeastSquare ) {
         double resVal = (y-fval)*invError;
         for (unsigned int k = 0; k < npar; ++k) {
            g[k] *= -invError;
         }
         return resVal;
      }

      return 0;
   }


private:

   double DoEval(const double *x ) const {
      return (*fObjFunc)(x);
   }

   TFumili * fFumili;
   const MethodFunc * fObjFunc;
   TF1 * fModFunc;

};
#else
bool gUseFumiliFunction = false;
#endif
//______________________________________________________________________________
//
//  TFumiliMinimizer class implementing the ROOT::Math::Minimizer interface using
//  TFumili.
//  This class is normally instantiates using the plug-in manager
//  (plug-in with name Fumili or TFumili)
//  In addition the user can choose the minimizer algorithm: Migrad (the default one), Simplex, or Minimize (combined Migrad + Simplex)
//
//__________________________________________________________________________________________

// initialize the static instances

ROOT::Math::FitMethodFunction * TFumiliMinimizer::fgFunc = 0;
ROOT::Math::FitMethodGradFunction * TFumiliMinimizer::fgGradFunc = 0;
TFumili * TFumiliMinimizer::fgFumili = 0;


ClassImp(TFumiliMinimizer);


TFumiliMinimizer::TFumiliMinimizer(int  ) :
   fDim(0),
   fNFree(0),
   fMinVal(0),
   fEdm(-1),
   fFumili(0)
{
   // Constructor for TFumiliMinimier class

   // construct with npar = 0 (by default a value of 25 is used in TFumili for allocating the arrays)
#ifdef USE_STATIC_TMINUIT
   // allocate here only the first time
   if (fgFumili == 0) fgFumili =  new TFumili(0);
   fFumili = fgFumili;
#else
   if (fFumili) delete fFumili;
   fFumili =  new TFumili(0);
   fgFumili = fFumili;
#endif

}


TFumiliMinimizer::~TFumiliMinimizer()
{
   // Destructor implementation.
   if (fFumili) delete fFumili;
}

TFumiliMinimizer::TFumiliMinimizer(const TFumiliMinimizer &) :
   Minimizer()
{
   // Implementation of copy constructor (it is private).
}

TFumiliMinimizer & TFumiliMinimizer::operator = (const TFumiliMinimizer &rhs)
{
   // Implementation of assignment operator (private)
   if (this == &rhs) return *this;  // time saving self-test
   return *this;
}



void TFumiliMinimizer::SetFunction(const  ROOT::Math::IMultiGenFunction & func) {
   // Set the objective function to be minimized, by passing a function object implement the
   // basic multi-dim Function interface. In this case the derivatives will be
   // calculated by Fumili

   // Here a TFumili instance is created since only at this point we know the number of parameters
   // needed to create TFumili
   fDim = func.NDim();
   fFumili->SetParNumber(fDim);

   // for Fumili the fit method function interface is required
   const ROOT::Math::FitMethodFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodFunction *>(&func);
   if (!fcnfunc) {
      Error("SetFunction","Wrong Fit method function type used for Fumili");
      return;
   }
   // assign to the static pointer (NO Thread safety here)
   fgFunc = const_cast<ROOT::Math::FitMethodFunction *>(fcnfunc);
   fgGradFunc = 0;
   fFumili->SetFCN(&TFumiliMinimizer::Fcn);

#ifdef USE_FUMILI_FUNCTION
   if (gUseFumiliFunction) {
      if (fcnfunc->Type() == ROOT::Math::FitMethodFunction::kLogLikelihood)
         fgFunc = new FumiliFunction<ROOT::Fit::PoissonLikelihoodFCN<ROOT::Math::FitMethodFunction::BaseFunction> >(fFumili,fcnfunc);
      else if (fcnfunc->Type() == ROOT::Math::FitMethodFunction::kLeastSquare)
         fgFunc = new FumiliFunction<ROOT::Fit::Chi2FCN<ROOT::Math::FitMethodFunction::BaseFunction> >(fFumili,fcnfunc);
   }
#endif

}

void TFumiliMinimizer::SetFunction(const  ROOT::Math::IMultiGradFunction & func) {
   // Set the objective function to be minimized, by passing a function object implement the
   // multi-dim gradient Function interface. In this case the function derivatives are provided
   // by the user via this interface and there not calculated by Fumili.

   fDim = func.NDim();
   fFumili->SetParNumber(fDim);

   // for Fumili the fit method function interface is required
   const ROOT::Math::FitMethodGradFunction * fcnfunc = dynamic_cast<const ROOT::Math::FitMethodGradFunction *>(&func);
   if (!fcnfunc) {
      Error("SetFunction","Wrong Fit method function type used for Fumili");
      return;
   }
   // assign to the static pointer (NO Thread safety here)
   fgFunc = 0;
   fgGradFunc = const_cast<ROOT::Math::FitMethodGradFunction  *>(fcnfunc);
   fFumili->SetFCN(&TFumiliMinimizer::Fcn);

}

void TFumiliMinimizer::Fcn( int & , double * g , double & f, double * x , int /* iflag */) {
   // implementation of FCN static function used internally by TFumili.
   // Adapt IMultiGenFunction interface to TFumili FCN static function
   f = TFumiliMinimizer::EvaluateFCN(const_cast<double*>(x),g);
}

// void TFumiliMinimizer::FcnGrad( int &, double * g, double & f, double * x , int iflag ) {
//    // implementation of FCN static function used internally by TFumili.
//    // Adapt IMultiGradFunction interface to TFumili FCN static function in the case of user
//    // provided gradient.
//    ROOT::Math::IMultiGradFunction * gFunc = dynamic_cast<ROOT::Math::IMultiGradFunction *> ( fgFunc);

//    assert(gFunc != 0);
//    f = gFunc->operator()(x);

//    // calculates also derivatives
//    if (iflag == 2) gFunc->Gradient(x,g);
// }

double TFumiliMinimizer::EvaluateFCN(const double * x, double * grad) {
   // function callaed to evaluate the FCN at the value x
   // calculates also the matrices of the second derivatives of the objective function needed by FUMILI


   //typedef FumiliFCNAdapter::Function Function;



   // reset
//    assert(grad.size() == npar);
//    grad.assign( npar, 0.0);
//    hess.assign( hess.size(), 0.0);

   double sum = 0;
   unsigned int ndata = 0;
   unsigned int npar = 0;
   if (fgFunc) {
      ndata = fgFunc->NPoints();
      npar = fgFunc->NDim();
      fgFunc->UpdateNCalls();
   }
   else if (fgGradFunc) {
      ndata = fgGradFunc->NPoints();
      npar = fgGradFunc->NDim();
      fgGradFunc->UpdateNCalls();
   }

   // eventually store this matrix as static member to optimize speed
   std::vector<double> gf(npar);
   std::vector<double> hess(npar*(npar+1)/2);

   // reset gradients
   for (unsigned int ipar = 0; ipar < npar; ++ipar)
      grad[ipar] = 0;


   //loop on the data points
//#define DEBUG
#ifdef DEBUG
   std::cout << "=============================================";
   std::cout << "par = ";
   for (unsigned int ipar = 0; ipar < npar; ++ipar)
      std::cout << x[ipar] << "\t";
   std::cout << std::endl;
   if (fgFunc) std::cout << "type " << fgFunc->Type() << std::endl;
#endif


   // assume for now least-square
   // since TFumili doet not use errodef I must diveide chi2 by 2
   if ( (fgFunc && fgFunc->Type() == ROOT::Math::FitMethodFunction::kLeastSquare) ||
        (fgGradFunc && fgGradFunc->Type() == ROOT::Math::FitMethodGradFunction::kLeastSquare) ) {

      double fval = 0;
      for (unsigned int i = 0; i < ndata; ++i) {
         // calculate data element and gradient
         // DataElement returns (f-y)/s and gf is derivatives of model function multiplied by (-1/sigma)
         if (gUseFumiliFunction) {
            fval = fgFunc->DataElement( x, i, &gf[0]);
         }
         else {
            if (fgFunc != 0)
               fval = fgFunc->DataElement(x, i, &gf[0]);
            else
               fval = fgGradFunc->DataElement(x, i, &gf[0]);
         }

         // t.b.d should protect for bad  values of fval
         sum += fval*fval;

         // to be check (TFumili uses a factor of 1/2 for chi2)

         for (unsigned int j = 0; j < npar; ++j) {
            grad[j] +=  fval * gf[j];
            for (unsigned int k = j; k < npar; ++ k) {
               int idx =  j + k*(k+1)/2;
               hess[idx] += gf[j] * gf[k];
            }
         }
      }
   }
   else if ( (fgFunc && fgFunc->Type() == ROOT::Math::FitMethodFunction::kLogLikelihood) ||
             (fgGradFunc && fgGradFunc->Type() == ROOT::Math::FitMethodGradFunction::kLogLikelihood) ) {



      double fval = 0;

      //std::cout << "\t x "  << x[0] << "  " << x[1] << "  " << x[2] << std::endl;

      for (unsigned int i = 0; i < ndata; ++i) {

         if (gUseFumiliFunction) {
            fval = fgFunc->DataElement( x, i, &gf[0]);
         }
         else {
            // calculate data element and gradient
            if (fgFunc != 0)
               fval = fgFunc->DataElement(x, i, &gf[0]);
            else
               fval = fgGradFunc->DataElement(x, i, &gf[0]);
         }

         // protect for small values of fval
         //      std::cout << i << "  "  << fval << " log " << " grad " << gf[0] << "  " << gf[1] << "  " << gf[2] << std::endl;
//         sum -= ROOT::Math::Util::EvalLog(fval);
         sum -= fval;

         for (unsigned int j = 0; j < npar; ++j) {
            double gfj = gf[j];// / fval;
            grad[j] -= gfj;
            for (unsigned int k = j; k < npar; ++ k) {
               int idx =  j + k*(k+1)/2;
               hess[idx] +=  gfj * gf[k];// / (fval );
            }
         }
      }
   }
   else {
      Error("EvaluateFCN"," type of fit method is not supported, it must be chi2 or log-likelihood");
   }

   // now TFumili excludes fixed prameter in second-derivative matrix
   // ned to get them using the static instance of TFumili
   double * zmatrix = fgFumili->GetZ();
   double * pl0 = fgFumili->GetPL0(); // parameter limits
   assert(zmatrix != 0);
   assert(pl0 != 0);
   unsigned int k = 0;
   unsigned int l = 0;
   for (unsigned int i = 0; i < npar; ++i) {
         for (unsigned int j = 0; j <= i; ++j) {
            if (pl0[i] > 0 && pl0[j] > 0) { // only for non-fixed parameters
               zmatrix[l++] = hess[k];
            }
            k++;
         }
   }

#ifdef DEBUG
   std::cout << "FCN value " << sum << " grad ";
   for (unsigned int ipar = 0; ipar < npar; ++ipar)
      std::cout << grad[ipar] << "\t";
   std::cout << std::endl << std::endl;
#endif


   return 0.5*sum; // fumili multiply then by 2

}



bool TFumiliMinimizer::SetVariable(unsigned int ivar, const std::string & name, double val, double step) {
   // set a free variable.
   if (fFumili == 0) {
      Error("SetVariableValue","invalid TFumili pointer. Set function first ");
      return false;
   }
#ifdef DEBUG
   std::cout << "set variable " << ivar << " " << name << " value " << val << " step " << step << std::endl;
#endif

   int ierr = fFumili->SetParameter(ivar , name.c_str(), val, step, 0., 0. );
   if (ierr) {
      Error("SetVariable","Error for parameter %d ",ivar);
      return false;
   }
   return true;
}

bool TFumiliMinimizer::SetLimitedVariable(unsigned int ivar, const std::string & name, double val, double step, double lower, double upper) {
   // set a limited variable.
   if (fFumili == 0) {
      Error("SetVariableValue","invalid TFumili pointer. Set function first ");
      return false;
   }
#ifdef DEBUG
   std::cout << "set limited variable " << ivar << " " << name << " value " << val << " step " << step << std::endl;
#endif
   int ierr = fFumili->SetParameter(ivar, name.c_str(), val, step, lower, upper );
   if (ierr) {
      Error("SetLimitedVariable","Error for parameter %d ",ivar);
      return false;
   }
   return true;
}
#ifdef LATER
bool Fumili2Minimizer::SetLowerLimitedVariable(unsigned int ivar , const std::string & name , double val , double step , double lower ) {
    // add a lower bounded variable as a double bound one, using a very large number for the upper limit
   double s = val-lower;
   double upper = s*1.0E15;
   if (s != 0)  upper = 1.0E15;
   return SetLimitedVariable(ivar, name, val, step, lower,upper);
}
#endif


bool TFumiliMinimizer::SetFixedVariable(unsigned int ivar, const std::string & name, double val) {
   // set a fixed variable.
   if (fFumili == 0) {
      Error("SetVariableValue","invalid TFumili pointer. Set function first ");
      return false;
   }


   int ierr = fFumili->SetParameter(ivar, name.c_str(), val, 0., val, val );
   fFumili->FixParameter(ivar);

#ifdef DEBUG
   std::cout << "Fix variable " << ivar << " " << name << " value " << std::endl;
#endif

   if (ierr) {
      Error("SetFixedVariable","Error for parameter %d ",ivar);
      return false;
   }
   return true;
}

bool TFumiliMinimizer::SetVariableValue(unsigned int ivar, double val) {
   // set the variable value
   if (fFumili == 0) {
      Error("SetVariableValue","invalid TFumili pointer. Set function first ");
      return false;
   }
   TString name = fFumili->GetParName(ivar);
   double  oldval, verr, vlow, vhigh = 0;
   int ierr = fFumili->GetParameter( ivar, &name[0], oldval, verr, vlow, vhigh);
   if (ierr) {
      Error("SetVariableValue","Error for parameter %d ",ivar);
      return false;
   }
#ifdef DEBUG
   std::cout << "set variable " << ivar << " " << name << " value "
             << val << " step " <<  verr << std::endl;
#endif

   ierr = fFumili->SetParameter(ivar , name , val, verr, vlow, vhigh );
   if (ierr) {
      Error("SetVariableValue","Error for parameter %d ",ivar);
      return false;
   }
   return true;
}

bool TFumiliMinimizer::Minimize() {
   // perform the minimization using the algorithm chosen previously by the user
   // By default Migrad is used.
   // Return true if the found minimum is valid and update internal chached values of
   // minimum values, errors and covariance matrix.

   if (fFumili == 0) {
      Error("SetVariableValue","invalid TFumili pointer. Set function first ");
      return false;
   }

   // need to set static instance to be used when calling FCN
   fgFumili = fFumili;


   double arglist[10];

   // error cannot be set in TFumili (always the same)
//    arglist[0] = ErrorUp();
//    fFumili->ExecuteCommand("SET Err",arglist,1);

   int printlevel = PrintLevel();
   // not implemented in TFumili yet
   //arglist[0] = printlevel - 1;
   //fFumili->ExecuteCommand("SET PRINT",arglist,1,ierr);

   // suppress warning in case Printlevel() == 0
   if (printlevel == 0)    fFumili->ExecuteCommand("SET NOW",arglist,0);
   else fFumili->ExecuteCommand("SET WAR",arglist,0);


   // minimize: use ExecuteCommand instead of Minimize to set tolerance and maxiter

   arglist[0] = MaxFunctionCalls();
   arglist[1] = Tolerance();

   if (printlevel > 0)
      std::cout << "Minimize using TFumili with tolerance = " << Tolerance()
                << " max calls " << MaxFunctionCalls() << std::endl;

   int iret = fFumili->ExecuteCommand("MIGRAD",arglist,2);
   fStatus = iret;
   //int iret = fgFumili->Minimize();

   // Hesse and IMP not implemented
//    // run improved if needed
//    if (ierr == 0 && fType == ROOT::Fumili::kMigradImproved)
//       fFumili->mnexcm("IMPROVE",arglist,1,ierr);

//    // check if Hesse needs to be run
//    if (ierr == 0 && IsValidError() ) {
//       fFumili->mnexcm("HESSE",arglist,1,ierr);
//    }


   int ntot;
   int nfree;
   double errdef = 0; // err def is not used by Fumili
   fFumili->GetStats(fMinVal,fEdm,errdef,nfree,ntot);

   if (printlevel > 0)
      fFumili->PrintResults(printlevel,fMinVal);


   assert (static_cast<unsigned int>(ntot) == fDim);
   assert( nfree == fFumili->GetNumberFreeParameters() );
   fNFree = nfree;


   // get parameter values and correlation matrix
   // fumili stores only lower part of diagonal matrix of the free parameters
   fParams.resize( fDim);
   fErrors.resize( fDim);
   fCovar.resize(fDim*fDim);
   const double * cv = fFumili->GetCovarianceMatrix();
   unsigned int l = 0;
   for (unsigned int i = 0; i < fDim; ++i) {
      fParams[i] = fFumili->GetParameter( i );
      fErrors[i] = fFumili->GetParError( i );

      if ( !fFumili->IsFixed(i) ) {
         for (unsigned int j = 0; j <=i ; ++j) {
            if ( !fFumili->IsFixed(j) ) {
               fCovar[i*fDim + j] = cv[l];
               fCovar[j*fDim + i] = fCovar[i*fDim + j];
               l++;
            }
         }
      }
   }

   return (iret==0) ? true : false;
}


//    } // end namespace Fit

// } // end namespace ROOT

