// @(#)root/mathmore:$Id$
// Author: L. Moneta Wed Dec 20 17:16:32 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class GSLNLSMinimizer

#include "Math/GSLNLSMinimizer.h"

#include "Math/MinimTransformFunction.h"
#include "Math/MultiNumGradFunction.h"

#include "Math/Error.h"
#include "GSLMultiFit.h"
#include "gsl/gsl_errno.h"


#include "Math/FitMethodFunction.h"
//#include "Math/Derivator.h"

#include <iostream>
#include <iomanip>
#include <cassert>

namespace ROOT {

   namespace Math {


/// Internal class used by GSLNLSMinimizer to implement the transformation of the chi2
/// function used by GSL Non-linear Least-square fitting
/// The class is template on the FitMethodFunction type to support both gradient and non
/// gradient functions
template<class FMFunc>
class FitTransformFunction : public FMFunc {

public:

   FitTransformFunction(const FMFunc & f, std::unique_ptr<MinimTransformFunction> transFunc ) :
      FMFunc( f.NDim(), f.NPoints() ),
      fFunc(f),
      fTransform(std::move(transFunc)),
      fGrad( std::vector<double>(f.NDim() ) )
   {
      // constructor from a given FitMethodFunction and  Transformation object.
      // Ownership of the transformation object is passed to this class
   }

   virtual ~FitTransformFunction()  {
   }

   // re-implement data element
   virtual double DataElement(const double *  x, unsigned i, double * g = 0) const  {
      // transform from x internal to x external
      const double * xExt = fTransform->Transformation(x);
      if ( g == 0) return fFunc.DataElement( xExt, i );
      // use gradient
      double val =  fFunc.DataElement( xExt, i, &fGrad[0]);
      // transform gradient
      fTransform->GradientTransformation( x, &fGrad.front(), g);
      return val;
   }


   virtual IMultiGenFunction * Clone() const {
      // not supported
      return nullptr;
   }

   // dimension (this is number of free dimensions)
   virtual unsigned int NDim() const  {
      return fTransform->NDim();
   }

   unsigned int NTot() const {
      return fTransform->NTot();
   }

   // forward of transformation functions
   const double * Transformation( const double * x) const { return fTransform->Transformation(x); }


   /// inverse transformation (external -> internal)
   void  InvTransformation(const double * xext,  double * xint) const { fTransform->InvTransformation(xext,xint); }

   /// inverse transformation for steps (external -> internal) at external point x
   void  InvStepTransformation(const double * x, const double * sext,  double * sint) const { fTransform->InvStepTransformation(x,sext,sint); }

   ///transform gradient vector (external -> internal) at internal point x
   void GradientTransformation(const double * x, const double *gext, double * gint) const   { fTransform->GradientTransformation(x,gext,gint); }

   void MatrixTransformation(const double * x, const double *cint, double * cext) const { fTransform->MatrixTransformation(x,cint,cext); }

private:

   // objects of this class are not meant for copying or assignment
   FitTransformFunction(const FitTransformFunction& rhs);
   FitTransformFunction& operator=(const FitTransformFunction& rhs);

   virtual double DoEval(const double * x) const  {
      return fFunc( fTransform->Transformation(x) );
   }

   virtual double DoDerivative(const double * /* x */, unsigned int /*icoord*/) const  {
      // not used
      throw std::runtime_error("FitTransformFunction::DoDerivative");
      return 0;
   }

   bool fOwnTransformation;
   const FMFunc & fFunc;                  // pointer to original fit method function
   std::unique_ptr<MinimTransformFunction> fTransform;        // pointer to transformation function
   mutable std::vector<double> fGrad;          // cached vector of gradient values

};



// GSLNLSMinimizer implementation

GSLNLSMinimizer::GSLNLSMinimizer( int type )
{
   // Constructor implementation : create GSLMultiFit wrapper object
   const gsl_multifit_fdfsolver_type * gsl_type = 0; // use default type defined in GSLMultiFit
   if (type == 1) gsl_type =   gsl_multifit_fdfsolver_lmsder; // scaled lmder version
   if (type == 2) gsl_type =   gsl_multifit_fdfsolver_lmder; // unscaled version

   fGSLMultiFit = new GSLMultiFit( gsl_type );

   fEdm = -1;

   // default tolerance and max iterations
   int niter = ROOT::Math::MinimizerOptions::DefaultMaxIterations();
   if (niter <= 0) niter = 100;
   SetMaxIterations(niter);

   fLSTolerance = ROOT::Math::MinimizerOptions::DefaultTolerance();
   if (fLSTolerance <=0) fLSTolerance = 0.0001; // default internal value

   SetPrintLevel(ROOT::Math::MinimizerOptions::DefaultPrintLevel());
}

GSLNLSMinimizer::~GSLNLSMinimizer () {
   assert(fGSLMultiFit != 0);
   delete fGSLMultiFit;
}



void GSLNLSMinimizer::SetFunction(const ROOT::Math::IMultiGenFunction & func) {
   // set the function to minimizer
   // need to create vector of functions to be passed to GSL multifit
   // support now only CHi2 implementation

   // call base class method. It will clone the function and set number of dimensions
   BasicMinimizer::SetFunction(func);
   fNFree = NDim();
   fUseGradFunction = false;
 }

void GSLNLSMinimizer::SetFunction(const ROOT::Math::IMultiGradFunction & func ) {
   // set the function to minimizer using gradient interface
   BasicMinimizer::SetFunction(func);
   fUseGradFunction = true;
}


bool GSLNLSMinimizer::Minimize() {

   if (ObjFunction() == nullptr) {
      MATH_ERROR_MSG("GSLNLSMinimizer::Minimize","Function has not been  set");
      return false;
   }
   // check type of function (if it provides gradient)
   auto fitFunc = (!fUseGradFunction) ? dynamic_cast<const ROOT::Math::FitMethodFunction *>(ObjFunction()) : nullptr;
   auto fitGradFunc = (fUseGradFunction) ? dynamic_cast<const ROOT::Math::FitMethodGradFunction *>(ObjFunction()) : nullptr;
   if (fitFunc == nullptr && fitGradFunc == nullptr) {
      if (PrintLevel() > 0) std::cout << "GSLNLSMinimizer: Invalid function set - only FitMethodFunction types are supported" << std::endl;
      return false;
   }

   if (fitGradFunc)
      return DoMinimize<ROOT::Math::FitMethodGradFunction>(*fitGradFunc);
   else
      return DoMinimize<ROOT::Math::FitMethodFunction>(*fitFunc);
}

template<class Func>
bool GSLNLSMinimizer::DoMinimize(const Func & fitFunc) {

   unsigned int size = fitFunc.NPoints();
   fNCalls = 0;  // reset number of function calls

   std::vector<LSResidualFunc<Func>> residualFuncs;
   residualFuncs.reserve(size);

   // set initial parameters of the minimizer
   int debugLevel = PrintLevel();

   assert (fGSLMultiFit != nullptr);

   unsigned int npar = NPar();
   unsigned int ndim = NDim();
   if (npar == 0 || npar < ndim) {
       MATH_ERROR_MSGVAL("GSLNLSMinimizer::Minimize","Wrong number of parameters",npar);
       return false;
   }

   // set residual functions and check if a transformation is needed
   std::vector<double> startValues;

   // transformation need a grad function.
   std::unique_ptr<MultiNumGradFunction> gradFunction;
   std::unique_ptr<MinimTransformFunction> trFuncRaw;
   if (!fUseGradFunction) {
      gradFunction = std::make_unique<MultiNumGradFunction>(fitFunc);
      trFuncRaw.reset(CreateTransformation(startValues, gradFunction.get()));
   }
   else {
      // use pointer stored in BasicMinimizer
      trFuncRaw.reset(CreateTransformation(startValues));
   }
   // need to transform in a FitTransformFunction which is set in the residual functions
   std::unique_ptr<FitTransformFunction<Func>> trFunc;
   if (trFuncRaw) {
      //pass ownership of trFuncRaw to FitTransformFunction
      trFunc = std::make_unique<FitTransformFunction<Func>>(fitFunc, std::move(trFuncRaw));
      assert(npar == trFunc->NTot() );
      for (unsigned int ires = 0; ires < size; ++ires) {
         residualFuncs.emplace_back(LSResidualFunc<Func>(*trFunc, ires));
      }
   }  else {
      for (unsigned int ires = 0; ires < size; ++ires) {
        residualFuncs.emplace_back( LSResidualFunc<Func>(fitFunc, ires) );
      }
   }

   if (debugLevel >=1 ) std::cout <<"Minimize using GSLNLSMinimizer "  << std::endl;


   int iret = fGSLMultiFit->Set( residualFuncs, &startValues.front() );

   if (iret) {
      MATH_ERROR_MSGVAL("GSLNLSMinimizer::Minimize","Error setting the residual functions ",iret);
      return false;
   }

   if (debugLevel >=1 ) std::cout <<"GSLNLSMinimizer: " << fGSLMultiFit->Name() << " - start iterating......... "  << std::endl;

   // start iteration
   unsigned  int iter = 0;
   int status;
   bool minFound = false;
   do {
      status = fGSLMultiFit->Iterate();

      if (debugLevel >=1) {
         std::cout << "----------> Iteration " << iter << " / " << MaxIterations() << " status " << gsl_strerror(status)  << std::endl;
         const double * x = fGSLMultiFit->X();
         if (trFunc) x = trFunc->Transformation(x);
         int pr = std::cout.precision(18);
         std::cout << "            FVAL = " << (fitFunc)(x) << std::endl;
         std::cout.precision(pr);
         std::cout << "            X Values : ";
         for (unsigned int i = 0; i < NDim(); ++i)
            std::cout << " " << VariableName(i) << " = " << X()[i];
         std::cout << std::endl;
      }

      if (status) break;

      // check also the delta in X()
      status = fGSLMultiFit->TestDelta( Tolerance(), Tolerance() );
      if (status == GSL_SUCCESS) {
         minFound = true;
      }

      // double-check with the gradient
      int status2 = fGSLMultiFit->TestGradient( Tolerance() );
      if ( minFound && status2 != GSL_SUCCESS) {
         // check now edm
         fEdm = fGSLMultiFit->Edm();
         if (fEdm > Tolerance() ) {
            // continue the iteration
            status = status2;
            minFound = false;
         }
      }

      if (debugLevel >=1) {
         std::cout  << "          after Gradient and Delta tests:  " << gsl_strerror(status);
         if (fEdm > 0) std::cout << ", edm is:  " << fEdm;
         std::cout << std::endl;
      }

      iter++;

   }
   while (status == GSL_CONTINUE && iter < MaxIterations() );

   // check edm
   fEdm = fGSLMultiFit->Edm();
   if ( fEdm < Tolerance() ) {
      minFound = true;
   }

   // save state with values and function value
   const double * x = fGSLMultiFit->X();
   if (x == 0) return false;
   // apply transformation outside SetFinalValues(..)
   // because trFunc is not a MinimTransformFunction but a FitTransFormFunction
   if (trFunc)  x = trFunc->Transformation(x);
   SetFinalValues(x);

   SetMinValue( (fitFunc)(x) );
   fStatus = status;
   fNCalls = fitFunc.NCalls();
   fErrors.resize(NDim());

   // get errors from cov matrix
   const double * cov =  fGSLMultiFit->CovarMatrix();
   if (cov) {

      fCovMatrix.resize(ndim*ndim);

      if (trFunc) {
         trFunc->MatrixTransformation(x, fGSLMultiFit->CovarMatrix(), fCovMatrix.data() );
      }
      else {
         std::copy(cov, cov + fCovMatrix.size(), fCovMatrix.begin() );
      }

      for (unsigned int i = 0; i < ndim; ++i)
         fErrors[i] = std::sqrt(fCovMatrix[i*ndim + i]);
   }

   if (minFound) {

      if (debugLevel >=1 ) {
         std::cout << "GSLNLSMinimizer: Minimum Found" << std::endl;
         int pr = std::cout.precision(18);
         std::cout << "FVAL         = " << MinValue() << std::endl;
         std::cout << "Edm          = " << fEdm    << std::endl;
         std::cout.precision(pr);
         std::cout << "NIterations  = " << iter << std::endl;
         std::cout << "NFuncCalls   = " << fitFunc.NCalls() << std::endl;
         for (unsigned int i = 0; i < NDim(); ++i)
            std::cout << std::setw(12) <<  VariableName(i) << " = " << std::setw(12) << X()[i] << "   +/-   " << std::setw(12) << fErrors[i] << std::endl;
      }

      return true;
   }
   else {
      if (debugLevel >=0 ) {
         std::cout << "GSLNLSMinimizer: Minimization did not converge: " << std::endl;
         if (status == GSL_ENOPROG) // case status 27
            std::cout << "\t iteration is not making progress towards solution" << std::endl;
         else
            std::cout << "\t failed with status " << status << std::endl;
      }
      if (debugLevel >=1 ) {
         std::cout << "FVAL         = " << MinValue() << std::endl;
         std::cout << "Edm   = " << fGSLMultiFit->Edm() << std::endl;
         std::cout << "Niterations  = " << iter << std::endl;
      }
      return false;
   }
   return false;
}

const double * GSLNLSMinimizer::MinGradient() const {
   // return gradient (internal values)
   return fGSLMultiFit->Gradient();
}


double GSLNLSMinimizer::CovMatrix(unsigned int i , unsigned int j ) const {
   // return covariance matrix element
   unsigned int ndim = NDim();
   if ( fCovMatrix.size() == 0) return 0;
   if (i > ndim || j > ndim) return 0;
   return fCovMatrix[i*ndim + j];
}

int GSLNLSMinimizer::CovMatrixStatus( ) const {
   // return covariance  matrix status = 0 not computed,
   // 1 computed but is approximate because minimum is not valid, 3 is fine
   if ( fCovMatrix.size() == 0) return 0;
   // case minimization did not finished correctly
   if (fStatus != GSL_SUCCESS) return 1;
   return 3;
}


   } // end namespace Math

} // end namespace ROOT

