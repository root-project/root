// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
   BayesianCalculator class
**/

// include other header files

#include "RooAbsFunc.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooBrentRootFinder.h"
#include "RooFormulaVar.h"
#include "RooGenericPdf.h"
#include "RooPlot.h"
#include "RooProdPdf.h"
#include "RooDataSet.h"

// include header file of this class 
#include "RooStats/BayesianCalculator.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/RooStatsUtils.h"

#include "Math/IFunction.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/Integrator.h"
#include "Math/RootFinder.h"
#include "Math/BrentMinimizer1D.h"
#include "RooFunctor.h"
#include "RooFunctor1DBinding.h"
#include "RooTFnBinding.h"
#include "RooMsgService.h"

#include "TAxis.h"
#include "TF1.h"
#include "TH1.h"
#include "TMath.h"
#include "TCanvas.h"

#include <map>
#include <cmath>

//#include "TRandom.h"
#include "RConfigure.h"

ClassImp(RooStats::BayesianCalculator)

namespace RooStats { 


// first some utility classes and functions 

#ifdef R__HAS_MATHMORE
   const ROOT::Math::RootFinder::EType kRootFinderType = ROOT::Math::RootFinder::kGSL_BRENT; 
#else 
   const ROOT::Math::RootFinder::EType kRootFinderType = ROOT::Math::RootFinder::kBRENT; 
#endif




struct  LikelihoodFunction { 
   LikelihoodFunction(RooFunctor & f, RooFunctor * prior = 0, double offset = 0) : 
      fFunc(f), fPrior(prior),
      fOffset(offset), fMaxL(0) {
      fFunc.binding().resetNumCall(); 
   }

   double operator() (const double *x ) const { 
      double nll = fFunc(x) - fOffset;
      double likelihood =  std::exp(-nll);

      if (fPrior) likelihood *= (*fPrior)(x);

      int nCalls = fFunc.binding().numCall();
      if (nCalls > 0 && nCalls % 1000 == 0) { 
         ooccoutD((TObject*)0,Eval) << "Likelihood evaluation ncalls = " << nCalls
                                    << " x0 " << x[0] << "  nll = " << nll+fOffset;
         if (fPrior) ooccoutD((TObject*)0,Eval) << " prior(x) = " << (*fPrior)(x); 
         ooccoutD((TObject*)0,Eval) << " likelihood " << likelihood                                     
                                    << " max Likelihood " << fMaxL << std::endl;                                            
      }
 
      if  (likelihood > fMaxL ) { 
         fMaxL = likelihood; 
         if ( likelihood > 1.E10) { 
            ooccoutW((TObject*)0,Eval) << "LikelihoodFunction::()  WARNING - Huge likelihood value found for  parameters ";
            for (int i = 0; i < fFunc.nObs(); ++i) 
               ooccoutW((TObject*)0,Eval) << " x[" << i << " ] = " << x[i]; 
            ooccoutW((TObject*)0,Eval) << "  nll = " << nll << " L = " << likelihood << std::endl;
         }
      }

      return likelihood; 
   }

   // for the 1D case
   double operator() (double x) const { 
      // just call the previous method
      double tmp = x; 
      return (*this)(&tmp); 
   }

   RooFunctor & fFunc;     // functor representing the nll function 
   RooFunctor * fPrior;     // functor representing the prior function 
   double fOffset;         //  offset used to bring the nll in a reasanble range for computing the exponent
   mutable double fMaxL;
};

//______________________________________________________________________

// Posterior CDF Function class 
// for integral of posterior function in nuisance and POI
// 1-Dim function as function of the poi 

class PosteriorCdfFunction : public ROOT::Math::IGenFunction { 

public:

   PosteriorCdfFunction(RooAbsReal & nll, RooAbsReal & prior, RooArgList & bindParams, const char * integType = 0, double nllMinimum = 0) : 
      fFunctor(nll, bindParams, RooArgList() ),  // functor 
      fPriorFunc(prior, bindParams, RooArgList() ),  // could be skipped in case of uniform priors
      fLikelihood(fFunctor, &fPriorFunc, nllMinimum),         // integral of exp(-nll) function
      fIntegrator(ROOT::Math::IntegratorMultiDim::GetType(integType) ),  // integrator 
      fXmin(bindParams.getSize() ),               // vector of parameters (min values) 
      fXmax(bindParams.getSize() ),               // vector of parameter (max values) 
      fNorm(1.0), fNormErr(0.0), fOffset(0), fMaxPOI(0), 
      fHasNorm(false),  fUseOldValues(true), fError(false) 
   {     

      fIntegrator.SetFunction(fLikelihood, bindParams.getSize() );
         
      ooccoutD((TObject*)0,NumIntegration) << "PosteriorCdfFunction::Compute integral of posterior in nuisance and poi. " 
                                           << " nllMinimum is " << nllMinimum << std::endl; 
   
      std::vector<double> par(bindParams.getSize());
      for (unsigned int i = 0; i < fXmin.size(); ++i) { 
         RooRealVar & var = (RooRealVar &) bindParams[i]; 
         fXmin[i] = var.getMin(); 
         fXmax[i] = var.getMax();
         par[i] = var.getVal();
         ooccoutD((TObject*)0,NumIntegration) << "PosteriorFunction::Integrate" << var.GetName() 
                                              << " in interval [ " <<  fXmin[i] << " , " << fXmax[i] << " ] " << std::endl;
      } 

      fIntegrator.Options().Print(ooccoutD((TObject*)0,NumIntegration));

      // store max POI value because it will be changed when evaluating the function 
      fMaxPOI = fXmax[0];

      // compute first the normalization with  the poi 
      fNorm = (*this)( fMaxPOI );  
      if (fError) ooccoutE((TObject*)0,NumIntegration) << "PosteriorFunction::Error computing normalization - norm = " << fNorm << std::endl;
      fHasNorm = true; 
      fNormCdfValues.insert(std::make_pair(fXmin[0], 0) );
      fNormCdfValues.insert(std::make_pair(fXmax[0], 1.0) );

   }
   
   // copy constructor 
   // need special treatment because integrator 
   // has no copy constuctor
   PosteriorCdfFunction(const PosteriorCdfFunction & rhs) : 
      ROOT::Math::IGenFunction(),
      fFunctor(rhs.fFunctor),
      fPriorFunc(rhs.fPriorFunc),
      fLikelihood(fFunctor, &fPriorFunc, rhs.fLikelihood.fOffset),  
      fIntegrator(ROOT::Math::IntegratorMultiDim::GetType( rhs.fIntegrator.Name().c_str() ) ),  // integrator 
      fXmin( rhs.fXmin), 
      fXmax( rhs.fXmax), 
      fNorm( rhs.fNorm), 
      fNormErr( rhs.fNormErr), 
      fOffset(rhs.fOffset), 
      fMaxPOI(rhs.fMaxPOI), 
      fHasNorm(rhs.fHasNorm), 
      fUseOldValues(rhs.fUseOldValues), 
      fError(rhs.fError), 
      fNormCdfValues(rhs.fNormCdfValues)
   { 
      fIntegrator.SetFunction(fLikelihood, fXmin.size() );
   }
                                   
    
   bool HasError() const { return fError; }


   ROOT::Math::IGenFunction * Clone() const { 
      ooccoutD((TObject*)0,NumIntegration) << " cloning function .........." << std::endl;
      return new PosteriorCdfFunction(*this); 
   }

   // offset value for computing the root
   void SetOffset(double offset) { fOffset = offset; }

private:

   // make assignment operator private
   PosteriorCdfFunction& operator=(const PosteriorCdfFunction &) { 
      return *this; 
   }

   double DoEval (double x) const {
 
      // evaluate cdf at poi value x by integrating poi from [xmin,x] and all the nuisances  
      fXmax[0] = x;
      if (x <= fXmin[0] ) return -fOffset; 
      // could also avoid a function evaluation at maximum
      if (x >= fMaxPOI && fHasNorm) return 1. - fOffset;  // cdf is bound to these values

      // computes the integral using a previous cdf estimate
      double  normcdf0 = 0; 
      if (fHasNorm && fUseOldValues) { 
         // look in the map of the stored cdf values the closes one
         std::map<double,double>::iterator itr = fNormCdfValues.upper_bound(x); 
         itr--;   // upper bound returns a poistion 1 up of the value we want
         if (itr != fNormCdfValues.end() ) { 
            fXmin[0] = itr->first; 
            normcdf0 = itr->second;
            // ooccoutD((TObject*)0,NumIntegration) << "PosteriorCdfFunction:   computing integral between in poi interval : " 
            //                                      << fXmin[0] << " -  " << fXmax[0] << std::endl; 
         }
      }

      fFunctor.binding().resetNumCall();  // reset number of calls for debug

      double cdf = fIntegrator.Integral(&fXmin[0],&fXmax[0]);  
      double error = fIntegrator.Error(); 
      double normcdf =  cdf/fNorm;  // normalize the cdf 

      ooccoutD((TObject*)0,NumIntegration) << "PosteriorCdfFunction: poi = [" << fXmin[0] << " , " 
                                           << fXmax[0] << "] integral =  " << cdf << " +/- " << error 
                                           << "  norm-integ = " << normcdf << " cdf(x) = " << normcdf+normcdf0 
                                           << " ncalls = " << fFunctor.binding().numCall() << std::endl; 

      if (TMath::IsNaN(cdf) || cdf > std::numeric_limits<double>::max()) { 
         ooccoutE((TObject*)0,NumIntegration) << "PosteriorFunction::Error computing integral - cdf = " 
                                              << cdf << std::endl;
         fError = true; 
      }

      if (cdf != 0 && error/cdf > 0.2 ) 
         oocoutW((TObject*)0,NumIntegration) << "PosteriorCdfFunction: integration error  is larger than 20 %   x0 = " << fXmin[0]  
                                              << " x = " << x << " cdf(x) = " << cdf << " +/- " << error << std::endl;

      if (!fHasNorm) { 
         oocoutI((TObject*)0,NumIntegration) << "PosteriorCdfFunction - integral of posterior = " 
                                             << cdf << " +/- " << error << std::endl; 
         fNormErr = error;
         return cdf; 
      }

      normcdf += normcdf0;

      // store values in the map
      if (fUseOldValues) { 
         fNormCdfValues.insert(std::make_pair(x, normcdf) );
      }

      double errnorm = sqrt( error*error + normcdf*normcdf * fNormErr * fNormErr )/fNorm;  
      if (normcdf > 1. + 3 * errnorm) {
         oocoutW((TObject*)0,NumIntegration) << "PosteriorCdfFunction: normalized cdf values is larger than 1" 
                                              << " x = " << x << " normcdf(x) = " << normcdf << " +/- " << error/fNorm << std::endl;
      }

      return normcdf - fOffset;  // apply an offset (for finding the roots) 
   }

   mutable RooFunctor fFunctor;                         // functor binding nll 
   mutable RooFunctor fPriorFunc;                         // functor binding the prior 
   LikelihoodFunction fLikelihood;              // likelihood function
   mutable ROOT::Math::IntegratorMultiDim  fIntegrator; // integrator  (mutable because Integral() is not const
   mutable std::vector<double> fXmin;    // min value of parameters (poi+nuis) - 
   mutable std::vector<double> fXmax;   // max value of parameters (poi+nuis) - max poi changes so it is mutable
   double fNorm;      // normalization value (computed in ctor) 
   mutable double fNormErr;    // normalization error value (computed in ctor) 
   double fOffset;   // offset for computing the root 
   double fMaxPOI;  // maximum value of POI 
   bool fHasNorm; // flag to control first call to the function 
   bool fUseOldValues;  // use old cdf values
   mutable bool fError;     // flag to indicate if a numerical evaluation error occurred
   mutable std::map<double,double> fNormCdfValues; 
};

//__________________________________________________________________
// Posterior Function class 
// 1-Dim function as function of the poi 
// and it integrated all the nuisance parameters 

class PosteriorFunction : public ROOT::Math::IGenFunction { 

public: 


   PosteriorFunction(RooAbsReal & nll, RooAbsReal & prior, RooRealVar & poi, RooArgList & nuisParams, const char * integType = 0, double
                     norm = 1.0,  double nllOffset = 0, int niter = 0) :
      fFunctor(nll, nuisParams, RooArgList() ),
      fPriorFunc(prior, nuisParams, RooArgList() ),  // create prior, it could have some dependence on nuisance parameters
      fLikelihood(fFunctor, &fPriorFunc, nllOffset), 
      fPoi(&poi),
      fXmin(nuisParams.getSize() ),
      fXmax(nuisParams.getSize() ), 
      fNorm(norm),
      fError(0)
   { 

      ooccoutD((TObject*)0,NumIntegration) << "PosteriorFunction::Evaluate the posterior function by integrating the nuisances: " << std::endl;
      for (unsigned int i = 0; i < fXmin.size(); ++i) { 
         RooRealVar & var = (RooRealVar &) nuisParams[i]; 
         fXmin[i] = var.getMin(); 
         fXmax[i] = var.getMax();
         ooccoutD((TObject*)0,NumIntegration) << "PosteriorFunction::Integrate " << var.GetName() 
                                              << " in interval [" <<  fXmin[i] << " , " << fXmax[i] << " ] " << std::endl;
      }
      if (fXmin.size() == 1) { // 1D case  
         fIntegratorOneDim = std::auto_ptr<ROOT::Math::Integrator>(
            new ROOT::Math::Integrator(ROOT::Math::IntegratorOneDim::GetType(integType) ) );
         fIntegratorOneDim->SetFunction(fLikelihood);
         // interested only in relative tolerance
         //fIntegratorOneDim->SetAbsTolerance(1.E-300);
         fIntegratorOneDim->Options().Print(ooccoutD((TObject*)0,NumIntegration) );
      }
      else if (fXmin.size() > 1) { // multiDim case          
         fIntegratorMultiDim = 
            std::auto_ptr<ROOT::Math::IntegratorMultiDim>(
               new ROOT::Math::IntegratorMultiDim(ROOT::Math::IntegratorMultiDim::GetType(integType) ) );
         fIntegratorMultiDim->SetFunction(fLikelihood, fXmin.size());
         ROOT::Math::IntegratorMultiDimOptions opt = fIntegratorMultiDim->Options();
         if (niter > 0) { 
            opt.SetNCalls(niter); 
            fIntegratorMultiDim->SetOptions(opt);
         }
         //fIntegratorMultiDim->SetAbsTolerance(1.E-300);
         // print the options
         opt.Print(ooccoutD((TObject*)0,NumIntegration) );
      }
   }
      
      
   ROOT::Math::IGenFunction * Clone() const { 
      assert(1); 
      return 0; // cannot clone this function for integrator 
   } 

   double Error() const { return fError;}
   
   
private: 
   double DoEval (double x) const { 

      // evaluate posterior function at a poi value x by integrating all nuisance parameters

      fPoi->setVal(x);
      fFunctor.binding().resetNumCall();  // reset number of calls for debug

      double f = 0; 
      double error = 0; 
      if (fXmin.size() == 1) { // 1D case  
         f = fIntegratorOneDim->Integral(fXmin[0],fXmax[0]); 
         error = fIntegratorOneDim->Error();
      }
      else if (fXmin.size() > 1) { // multi-dim case
         f = fIntegratorMultiDim->Integral(&fXmin[0],&fXmax[0]); 
         error = fIntegratorMultiDim->Error();
      } else { 
         // no integration to be done
         f = fLikelihood(&x);
      }

      // debug 
      ooccoutD((TObject*)0,NumIntegration) << "PosteriorFunction:  POI value  =  " 
                                           << x << "\tf(x) =  " << f << " +/- " << error 
                                           << "  norm-f(x) = " << f/fNorm 
                                           << " ncalls = " << fFunctor.binding().numCall() << std::endl; 




      if (f != 0 && error/f > 0.2 ) 
         ooccoutW((TObject*)0,NumIntegration) << "PosteriorFunction::DoEval - Error from integration in " 
                                              << fXmin.size() <<  " Dim is larger than 20 % " 
                                              << "x = " << x << " p(x) = " << f << " +/- " << error << std::endl;

      fError = error / fNorm; 
      return f / fNorm;
   }

   mutable RooFunctor fFunctor; 
   mutable RooFunctor fPriorFunc; 
   LikelihoodFunction fLikelihood; 
   RooRealVar * fPoi;
   std::auto_ptr<ROOT::Math::Integrator>  fIntegratorOneDim; 
   std::auto_ptr<ROOT::Math::IntegratorMultiDim>  fIntegratorMultiDim; 
   std::vector<double> fXmin; 
   std::vector<double> fXmax; 
   double fNorm;
   mutable double fError;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Posterior function obtaining sampling  toy MC for th enuisance according to their pdf  
class PosteriorFunctionFromToyMC : public ROOT::Math::IGenFunction { 

public: 


   PosteriorFunctionFromToyMC(RooAbsReal & nll, RooAbsReal & prior, RooAbsPdf & pdf, RooRealVar & poi, RooArgList & nuisParams, double
                              nllOffset = 0, int niter = 0, bool redoToys = true ) :
      fFunctor(nll, nuisParams, RooArgList() ),
      fPriorFunc(prior, nuisParams, RooArgList() ), // create prior, it could have some dependence on nuisance parameters
      fLikelihood(fFunctor, &fPriorFunc, nllOffset), 
      fPdf(&pdf),
      fPoi(&poi),
      fNuisParams(nuisParams),
      fGenParams(0),
      fNumIterations(niter),
      fError(-1),
      fRedoToys(redoToys)
   { 
      if (niter == 0) fNumIterations = 100; // default value 

      ooccoutI((TObject*)0,InputArguments) << "PosteriorFunctionFromToyMC::Evaluate the posterior function by randomizing the nuisances:  niter " << fNumIterations << std::endl;

      ooccoutI((TObject*)0,InputArguments) << "PosteriorFunctionFromToyMC::Pdf used for randomizing the nuisance is " << fPdf->GetName() << std::endl; 
      // check that pdf contains  the nuisance 
      RooArgSet * vars = fPdf->getVariables(); 
      for (int i = 0; i < fNuisParams.getSize(); ++i) { 
         if (!vars->find( fNuisParams[i].GetName() ) ) { 
            ooccoutW((TObject*)0,InputArguments) << "Nuisance parameter " << fNuisParams[i].GetName() 
                                                 << " is not part of sampling pdf. " 
                                                 << " A uniform distribution will be generated " << std::endl;
         }
      }
      delete vars;

      if (!fRedoToys) { 
         ooccoutI((TObject*)0,InputArguments) << "PosteriorFunctionFromToyMC::Generate nuisance toys only one time (for all POI points)" << std::endl; 
         GenerateToys();
      }
   }

   virtual ~PosteriorFunctionFromToyMC() { if (fGenParams) delete fGenParams; }

   // generate first n-samples of the nuisance parameters 
   void GenerateToys() const {    
      if (fGenParams) delete fGenParams;
      fGenParams = fPdf->generate(fNuisParams, fNumIterations);
      if(fGenParams==0) {
         ooccoutE((TObject*)0,InputArguments) << "PosteriorFunctionFromToyMC - failed to generate nuisance parameters" << std::endl;
      }
   }

   double Error() const { return fError;}

   ROOT::Math::IGenFunction * Clone() const { 
      // use defsult copy constructor 
      //return new PosteriorFunctionFromToyMC(*this);
      //  clone not implemented  
      assert(1);
      return 0;
   }

private:
   // evaluate the posterior at the poi value x 
   double DoEval( double x) const { 

      int npar = fNuisParams.getSize();
      assert (npar > 0);  

      
      // generate the toys 
      if (fRedoToys) GenerateToys();
      if (!fGenParams) return 0;

      // evaluate posterior function at a poi value x by integrating all nuisance parameters

      fPoi->setVal(x);

      // loop over all of the generate data 
      double sum = 0; 
      double sum2 = 0; 

      for(int iter=0; iter<fNumIterations; ++iter) {

         // get the set of generated parameters and set the nuisance parameters to the generated values
         std::vector<double> p(npar); 
         for (int i = 0; i < npar; ++i) { 
            const RooArgSet* genset=fGenParams->get(iter);
            RooAbsArg * arg = genset->find( fNuisParams[i].GetName() );
            RooRealVar * var = dynamic_cast<RooRealVar*>(arg);
            assert( arg!= 0);
            p[i] = var->getVal();
            ((RooRealVar &) fNuisParams[i]).setVal(p[i]);
         }

         // evaluate now the likelihood function 
         double fval =  fLikelihood( &p.front() );

         // liklihood already must contained the pdf we have sampled 
         // so we must divided by it. The value must be normalized on all 
         // other parameters 
         RooArgSet arg(fNuisParams);
         double nuisPdfVal = fPdf->getVal(&arg); 
         fval /= nuisPdfVal;


         if( fval > std::numeric_limits<double>::max()  ) {
            ooccoutE((TObject*)0,Eval) <<  "BayesianCalculator::EvalPosteriorFunctionFromToy : " 
                        << "Likelihood evaluates to infinity " << std::endl;
            ooccoutE((TObject*)0,Eval) <<  "poi value =  " << x << std::endl; 
            ooccoutE((TObject*)0,Eval) <<  "Nuisance  parameter values :  ";
            for (int i = 0; i < npar; ++i)  
               ooccoutE((TObject*)0,Eval) << fNuisParams[i].GetName() << " = " << p[i] << " ";
            ooccoutE((TObject*)0,Eval) <<  " - return 0   " << std::endl;

            fError = 1.E30; 
            return 0;
         }
         if(  TMath::IsNaN(fval) ) {
            ooccoutE((TObject*)0,Eval) <<  "BayesianCalculator::EvalPosteriorFunctionFromToy : " 
                        << "Likelihood is a NaN " << std::endl;
            ooccoutE((TObject*)0,Eval) <<  "poi value =  " << x << std::endl; 
            ooccoutE((TObject*)0,Eval) <<  "Nuisance  parameter values :  ";
            for (int i = 0; i < npar; ++i)  
               ooccoutE((TObject*)0,Eval) << fNuisParams[i].GetName() << " = " << p[i] << " ";
            ooccoutE((TObject*)0,Eval) <<  " - return 0   " << std::endl;
            fError = 1.E30; 
            return 0;
         }


         
         sum += fval; 
         sum2 += fval*fval; 
      }
   
      // compute the average and variance 
      double val = sum/double(fNumIterations);
      double dval2 = std::max( sum2/double(fNumIterations) - val*val, 0.0);
      fError = std::sqrt( dval2 / fNumIterations);

      // debug 
      ooccoutD((TObject*)0,NumIntegration) << "PosteriorFunctionFromToyMC:  POI value  =  " 
                                           << x << "\tp(x) =  " << val << " +/- " << fError << std::endl;
      

      if (val != 0 && fError/val > 0.2 ) {
         ooccoutW((TObject*)0,NumIntegration) << "PosteriorFunctionFromToyMC::DoEval" 
                                              << " - Error in estimating posterior is larger than 20% ! " 
                                              << "x = " << x << " p(x) = " << val << " +/- " << fError << std::endl;
      }


      return val; 
   }

   mutable RooFunctor fFunctor; 
   mutable RooFunctor fPriorFunc; 
   LikelihoodFunction fLikelihood; 
   mutable RooAbsPdf * fPdf;
   RooRealVar * fPoi;
   RooArgList fNuisParams;
   mutable RooDataSet * fGenParams;
   int fNumIterations;
   mutable double fError; 
   bool fRedoToys;                    // do toys every iteration

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation of BayesianCalculator 
/////////////////////////////////////////////////////////////////////////////////////////////////////

BayesianCalculator::BayesianCalculator() :
   fData(0),
   fPdf(0),
   fPriorPOI(0),
   fNuisancePdf(0),
   fProductPdf (0), fLogLike(0), fLikelihood (0), fIntegratedLikelihood (0), fPosteriorPdf(0), 
   fPosteriorFunction(0), fApproxPosterior(0),
   fLower(0), fUpper(0),
   fNLLMin(0),
   fSize(0.05), fLeftSideFraction(0.5), 
   fBrfPrecision(0.00005), 
   fNScanBins(-1),
   fValidInterval(false)
{
   // default constructor
}

BayesianCalculator::BayesianCalculator( /* const char* name,  const char* title, */						   
						    RooAbsData& data,
                                                    RooAbsPdf& pdf,
						    const RooArgSet& POI,
						    RooAbsPdf& priorPOI,
						    const RooArgSet* nuisanceParameters ) :
   //TNamed( TString(name), TString(title) ),
   fData(&data),
   fPdf(&pdf),
   fPOI(POI),
   fPriorPOI(&priorPOI),
   fNuisancePdf(0),
   fProductPdf (0), fLogLike(0), fLikelihood (0), fIntegratedLikelihood (0), fPosteriorPdf(0),
   fPosteriorFunction(0), fApproxPosterior(0),
   fLower(0), fUpper(0), 
   fNLLMin(0),
   fSize(0.05), fLeftSideFraction(0.5), 
   fBrfPrecision(0.00005), 
   fNScanBins(-1),
   fNumIterations(0),
   fValidInterval(false)
{
   // constructor
   if (nuisanceParameters) fNuisanceParameters.add(*nuisanceParameters); 
}

BayesianCalculator::BayesianCalculator( RooAbsData& data,
                       ModelConfig & model) : 
   fData(&data), 
   fPdf(model.GetPdf()),
   fPriorPOI( model.GetPriorPdf()),
   fNuisancePdf(0),
   fProductPdf (0), fLogLike(0), fLikelihood (0), fIntegratedLikelihood (0), fPosteriorPdf(0),
   fPosteriorFunction(0), fApproxPosterior(0),
   fLower(0), fUpper(0), 
   fNLLMin(0),
   fSize(0.05), fLeftSideFraction(0.5), 
   fBrfPrecision(0.00005), 
   fNScanBins(-1),
   fNumIterations(0),
   fValidInterval(false)
{
   // constructor from Model Config
   SetModel(model);
}


BayesianCalculator::~BayesianCalculator()
{
   // destructor
   ClearAll(); 
}

void BayesianCalculator::ClearAll() const { 
   // clear cached pdf objects
   if (fProductPdf) delete fProductPdf; 
   if (fLogLike) delete fLogLike; 
   if (fLikelihood) delete fLikelihood; 
   if (fIntegratedLikelihood) delete fIntegratedLikelihood; 
   if (fPosteriorPdf) delete fPosteriorPdf;    
   if (fPosteriorFunction) delete fPosteriorFunction; 
   if (fApproxPosterior) delete fApproxPosterior; 
   fPosteriorPdf = 0; 
   fPosteriorFunction = 0; 
   fProductPdf = 0;
   fLogLike = 0; 
   fLikelihood = 0; 
   fIntegratedLikelihood = 0; 
   fLower = 0;
   fUpper = 0;
   fNLLMin = 0; 
   fValidInterval = false;
}

void BayesianCalculator::SetModel(const ModelConfig & model) {
   // set the model
   fPdf = model.GetPdf();
   fPriorPOI =  model.GetPriorPdf(); 
   // assignment operator = does not do a real copy the sets (must use add method) 
   fPOI.removeAll();
   fNuisanceParameters.removeAll();
   if (model.GetParametersOfInterest()) fPOI.add( *(model.GetParametersOfInterest()) );
   if (model.GetNuisanceParameters())  fNuisanceParameters.add( *(model.GetNuisanceParameters() ) );

   // invalidate the cached pointers
   ClearAll(); 
}



RooAbsReal* BayesianCalculator::GetPosteriorFunction() const
{
   // build and return the posterior function (not normalized) as a RooAbsReal
   // the posterior is obtained from the product of the likelihood function and the
   // prior pdf which is then intergated in the nuisance parameters (if existing).
   // A prior function for the nuisance can be specified either in the prior pdf object
   // or in the model itself. If no prior nuisance is specified, but prior parameters are then
   // the integration is performed assuming a flat prior for the nuisance parameters.        

   if (fIntegratedLikelihood) return fIntegratedLikelihood; 
   if (fLikelihood) return fLikelihood; 

   // run some sanity checks
   if (!fPdf ) {
      coutE(InputArguments) << "BayesianCalculator::GetPosteriorPdf - missing pdf model" << std::endl;
      return 0;
   }
   if (!fPriorPOI) { 
      coutE(InputArguments) << "BayesianCalculator::GetPosteriorPdf - missing prior pdf" << std::endl;
      return 0;
   }
   if (fPOI.getSize() == 0) {
      coutE(InputArguments) << "BayesianCalculator::GetPosteriorPdf - missing parameter of interest" << std::endl;
      return 0;
   }
   if (fPOI.getSize() > 1) { 
      coutE(InputArguments) << "BayesianCalculator::GetPosteriorPdf - current implementation works only on 1D intervals" << std::endl;
      return 0; 
   }


   RooArgSet* constrainedParams = fPdf->getParameters(*fData);
   // remove the constant parameters
   RemoveConstantParameters(constrainedParams);
   
   //constrainedParams->Print("V");

   // use RooFit::Constrain() to be sure constraints terms are taken into account
   fLogLike = fPdf->createNLL(*fData, RooFit::Constrain(*constrainedParams) );



   ccoutD(Eval) <<  "BayesianCalculator::GetPosteriorFunction : " 
                << " pdf value " <<  fPdf->getVal() 
                << " neglogLikelihood = " << fLogLike->getVal() 
                << " priorPOI value " << fPriorPOI->getVal() << std::endl;

   // check that likelihood evaluation is not inifinity 
   if ( fLogLike->getVal() > std::numeric_limits<double>::max() ) {
      coutE(Eval) <<  "BayesianCalculator::GetPosteriorFunction : " 
                  << " Negative log likelihood evaluates to infinity " << std::endl 
                  << " Non-const Parameter values : ";
      RooArgList p(*constrainedParams);
      for (int i = 0; i < p.getSize(); ++i) {
         RooRealVar * v = dynamic_cast<RooRealVar *>(&p[i] );
         if (v!=0) ccoutE(Eval) << v->GetName() << " = " << v->getVal() << "   ";
      }
      ccoutE(Eval) << std::endl;
      ccoutE(Eval) << "--  Perform a full likelihood fit of the model before or set more reasanable parameter values"  
                   << std::endl; 
      coutE(Eval) << "BayesianCalculator::GetPosteriorFunction : " << " cannot compute posterior function "  << std::endl; 
      return 0;
   }

   // if pdf evaluates to zero, should be fixed, but this will
   // stop error messages.
   fLogLike->setEvalErrorLoggingMode(RooAbsReal::CountErrors);



   // need do find minimum of log-likelihood in the range to shift function 
   // to avoid numerical errors when we compute the likelihood (overflows in the exponent)
   // N.B.: this works for only 1 parameter of interest otherwise Minuit should be used for finding the minimum
   RooFunctor * nllFunc = fLogLike->functor(fPOI);
   ROOT::Math::Functor1D wnllFunc(*nllFunc);
   RooRealVar* poi = dynamic_cast<RooRealVar*>( fPOI.first() ); 
   assert(poi);

   ROOT::Math::BrentMinimizer1D minim; 
   minim.SetFunction(wnllFunc,poi->getMin(),poi->getMax() );
   bool ret  = minim.Minimize(100,1.E-3,1.E-3);
   fNLLMin = 0; 
   if (ret) fNLLMin = minim.FValMinimum();

   ccoutD(Eval) << "BayesianCalculator::GetPosteriorFunction : minimum of NLL vs POI for POI =  " 
          << poi->getVal() << " min NLL = " << fNLLMin << std::endl;

   delete nllFunc;

   delete constrainedParams;


   if ( fNuisanceParameters.getSize() == 0 ||  fIntegrationType.Contains("ROOFIT") ) { 

      ccoutD(Eval) << "BayesianCalculator::GetPosteriorFunction : use ROOFIT integration  " 
                   << std::endl;

#ifdef DOLATER // (not clear why this does not work)
      // need to make in this case a likelihood from the nll and make the product with the prior
      TString likeName = TString("likelihood_times_prior_") + TString(fPriorPOI->GetName());   
      TString formula; 
      formula.Form("exp(-@0+%f+log(@1))",fNLLMin);
      fLikelihood = new RooFormulaVar(likeName,formula,RooArgList(*fLogLike,*fPriorPOI));
#else
      // here use RooProdPdf (not very nice) but working

      if (fLogLike) delete fLogLike; 
      // // create a unique name for the product pdf 
      TString prodName = TString("product_") + TString(fPdf->GetName()) + TString("_") + TString(fPriorPOI->GetName() );   
      fProductPdf = new RooProdPdf(prodName,"",RooArgList(*fPdf,*fPriorPOI));

      RooArgSet* constrParams = fPdf->getParameters(*fData);
      // remove the constant parameters
      RemoveConstantParameters(constrParams);
      fLogLike = fProductPdf->createNLL(*fData, RooFit::Constrain(*constrParams) );
      delete constrParams;

      TString likeName = TString("likelihood_times_prior_") + TString(fProductPdf->GetName());   
      TString formula; 
      formula.Form("exp(-@0+%f)",fNLLMin);
      fLikelihood = new RooFormulaVar(likeName,formula,RooArgList(*fLogLike));
#endif

      
      // if no nuisance parameter we can just return the likelihood funtion
      if (fNuisanceParameters.getSize() == 0) { 
         fIntegratedLikelihood = fLikelihood; 
         fLikelihood = 0; 
      }
      else 
         // case of using RooFit for the integration
         fIntegratedLikelihood = fLikelihood->createIntegral(fNuisanceParameters);

      return fIntegratedLikelihood;
   }

   else if ( fIntegrationType.Contains("TOYMC") ) { 
      // compute the posterior as expectation values of the likelihood function 
      // sampling on the nuisance parameters 

      RooArgList nuisParams(fNuisanceParameters); 

      bool doToysEveryIteration = true;
      // if type is 1-TOYMC or TOYMC-1
      if ( fIntegrationType.Contains("1") || fIntegrationType.Contains("ONE")  ) doToysEveryIteration = false;

      RooAbsPdf * samplingPdf = (fNuisancePdf) ? fNuisancePdf : fPdf;
      fPosteriorFunction = new PosteriorFunctionFromToyMC(*fLogLike, *fPriorPOI, *samplingPdf, *poi, nuisParams, fNLLMin,
                                                          fNumIterations, doToysEveryIteration ); 

      TString name = "toyposteriorfunction_from_"; 
      name += fLogLike->GetName();  
      fIntegratedLikelihood = new RooFunctor1DBinding(name,name,*fPosteriorFunction,*poi);
      
      // need to scan likelihood in this case
      if (fNScanBins <= 0) fNScanBins = 100;
      
   }

   else  { 

      // use ROOT integration method if there are nuisance parameters 

      RooArgList nuisParams(fNuisanceParameters); 
      fPosteriorFunction = new PosteriorFunction(*fLogLike, *fPriorPOI, *poi, nuisParams, fIntegrationType, 1.,fNLLMin, fNumIterations ); 
      
      TString name = "posteriorfunction_from_"; 
      name += fLogLike->GetName();  
      fIntegratedLikelihood = new RooFunctor1DBinding(name,name,*fPosteriorFunction,*poi);

   }

   //fIntegratedLikelihood->setEvalErrorLoggingMode(RooAbsReal::CountErrors);

   // ccoutD(Eval) << "BayesianCalculator::GetPosteriorFunction : use ROOT numerical integration algorithm. "; 
   // ccoutD(Eval) << " Integrated log-likelihood = " << fIntegratedLikelihood->getVal() << std::endl;

   
   return fIntegratedLikelihood;  

}

RooAbsPdf* BayesianCalculator::GetPosteriorPdf() const
{
   /// build and return the posterior pdf (i.e posterior function normalized to all range of poi
   ///NOTE: user must delete the returned object 
   
   RooAbsReal * plike = GetPosteriorFunction();
   if (!plike) return 0;

   
   // create a unique name on the posterior from the names of the components
   TString posteriorName = this->GetName() + TString("_posteriorPdf_") + plike->GetName(); 

   RooAbsPdf * posteriorPdf = new RooGenericPdf(posteriorName,"@0",*plike);

   return posteriorPdf;
}


RooPlot* BayesianCalculator::GetPosteriorPlot(bool norm, double precision ) const
{
  /// return a RooPlot with the posterior  and the credibility region

   GetPosteriorFunction(); 

   // if a scan is requested approximate the posterior
   if (fNScanBins > 0) 
      ApproximatePosterior();

   RooAbsReal * posterior = fIntegratedLikelihood; 
   if (norm) { 
      // delete and re-do always posterior pdf (could be invalid after approximating it)
      if (fPosteriorPdf) delete fPosteriorPdf; 
      fPosteriorPdf = GetPosteriorPdf();
      posterior = fPosteriorPdf;
   }
   if (!posterior) return 0;

   if (!fValidInterval) GetInterval();

   RooAbsRealLValue* poi = dynamic_cast<RooAbsRealLValue*>( fPOI.first() );
   assert(poi);


   RooPlot* plot = poi->frame();

   // try to reduce some error messages
   posterior->setEvalErrorLoggingMode(RooAbsReal::CountErrors);

   plot->SetTitle(TString("Posterior probability of parameter \"")+TString(poi->GetName())+TString("\""));  
   posterior->plotOn(plot,RooFit::Range(fLower,fUpper,kFALSE),RooFit::VLines(),RooFit::DrawOption("F"),RooFit::MoveToBack(),RooFit::FillColor(kGray),RooFit::Precision(precision));
   posterior->plotOn(plot);
   plot->GetYaxis()->SetTitle("posterior function");
   
   return plot; 
}

void BayesianCalculator::SetIntegrationType(const char * type) { 
   fIntegrationType = TString(type); 
   fIntegrationType.ToUpper(); 
}



SimpleInterval* BayesianCalculator::GetInterval() const
{
  /// returns a SimpleInterval with lower and upper bounds on the
  /// parameter of interest specified in the constructor. 
  /// Using the method (to be called before SetInterval) SetLeftSideTailFraction the user can choose the type of interval.
  /// By default the returned interval is a central interval with the confidence level specified 
  /// previously in the constructor ( LeftSideTailFraction = 0.5). 
  ///  For lower limit use SetLeftSideTailFraction = 1
  ///  For upper limit use SetLeftSideTailFraction = 0
  ///  for shortest intervals use SetLeftSideTailFraction = -1 or call the method SetShortestInterval()
  /// NOTE: The BayesianCaluclator covers only the case with one
  /// single parameter of interest

   if (fValidInterval) 
      coutW(Eval) << "BayesianCalculator::GetInterval - recomputing interval for the same CL and same model" << std::endl;

   RooRealVar* poi = dynamic_cast<RooRealVar*>( fPOI.first() );
   if (!poi) { 
      coutE(Eval) << "BayesianCalculator::GetInterval - no parameter of interest is set " << std::endl;
      return 0; 
   } 

   // get integrated likelihood (posterior function) 
   GetPosteriorFunction();

   if (fLeftSideFraction < 0 ) { 
      // compute short intervals
      ComputeShortestInterval(); 
   }
   else {
      // compute the other intervals

      double lowerCutOff = fLeftSideFraction * fSize; 
      double upperCutOff = 1. - (1.- fLeftSideFraction) * fSize; 


      if (fNScanBins > 0) { 
         ComputeIntervalFromApproxPosterior(lowerCutOff, upperCutOff);
      }
      
      else { 
         // use integration method if there are nuisance parameters 
         if (fNuisanceParameters.getSize() > 0) { 
            ComputeIntervalFromCdf(lowerCutOff, upperCutOff);      
         }
         else { 
            // case of no nuisance - just use createCdf from roofit
            ComputeIntervalUsingRooFit(lowerCutOff, upperCutOff);      
         }
         // case cdf failed (scan then the posterior)
         if (!fValidInterval) { 
            fNScanBins = 100;
            coutW(Eval) << "BayesianCalculator::GetInterval - computing integral from cdf failed - do a scan in "
                        << fNScanBins << " nbins " << std::endl;
            ComputeIntervalFromApproxPosterior(lowerCutOff, upperCutOff);
         }
      }
   }

   if (!fValidInterval) { 
      fLower = 1; fUpper = 0;
      coutE(Eval) << "BayesianCalculator::GetInterval - cannot compute a valid interval - return a dummy [1,0] interval"
      <<  std::endl;
   }
   else {
      coutI(Eval) << "BayesianCalculator::GetInterval - found a valid interval : [" << fLower << " , " 
                << fUpper << " ]" << std::endl;
   }
   
   TString interval_name = TString("BayesianInterval_a") + TString(this->GetName());
   SimpleInterval * interval = new SimpleInterval(interval_name,*poi,fLower,fUpper,ConfidenceLevel());
   interval->SetTitle("SimpleInterval from BayesianCalculator");
   
   return interval;
}

double BayesianCalculator::GetMode() const { 
   /// Returns the value of the parameter for the point in
   /// parameter-space that is the most likely.
   ///  How do we do if there are points that are equi-probable?
   /// use approximate posterior
   /// t.b.d use real function to find the mode

   ApproximatePosterior(); 
   TH1 * h = fApproxPosterior->GetHistogram();
   return h->GetBinCenter(h->GetMaximumBin() );
   //return  fApproxPosterior->GetMaximumX();
}

void BayesianCalculator::ComputeIntervalUsingRooFit(double lowerCutOff, double upperCutOff ) const { 
   // compute the interval using RooFit

   coutI(Eval) <<  "BayesianCalculator: Compute interval using RooFit:  posteriorPdf + createCdf + RooBrentRootFinder " << std::endl;

   RooRealVar* poi = dynamic_cast<RooRealVar*>( fPOI.first() ); 
   assert(poi);

   fValidInterval = false;
   if (!fPosteriorPdf) fPosteriorPdf = (RooAbsPdf*) GetPosteriorPdf();
   if (!fPosteriorPdf) return;
         
   RooAbsReal* cdf = fPosteriorPdf->createCdf(fPOI,RooFit::ScanNoCdf());
         
   RooAbsFunc* cdf_bind = cdf->bindVars(fPOI,&fPOI);
   RooBrentRootFinder brf(*cdf_bind);
   brf.setTol(fBrfPrecision); // set the brf precision
   
   double tmpVal = poi->getVal();  // patch used because findRoot changes the value of poi
   
   bool ret = true; 
   if (lowerCutOff > 0) { 
      double y = lowerCutOff;
      ret &= brf.findRoot(fLower,poi->getMin(),poi->getMax(),y);
   } 
   else 
      fLower = poi->getMin(); 

   if (upperCutOff < 1.0) { 
      double y=upperCutOff;
      ret &= brf.findRoot(fUpper,poi->getMin(),poi->getMax(),y);
   }
   else 
      fUpper = poi->getMax();
   if (!ret)  coutE(Eval) << "BayesianCalculator::GetInterval "
                           << "Error returned from Root finder, estimated interval is not fully correct" 
                           << std::endl;
   else 
      fValidInterval = true; 
   

   poi->setVal(tmpVal); // patch: restore the original value of poi
   
   delete cdf_bind;
   delete cdf;
}

void BayesianCalculator::ComputeIntervalFromCdf(double lowerCutOff, double upperCutOff ) const { 
   // compute the interval using Cdf integration

   fValidInterval = false; 

   coutI(InputArguments) <<  "BayesianCalculator:GetInterval Compute the interval from the posterior cdf " << std::endl;

   RooRealVar* poi = dynamic_cast<RooRealVar*>( fPOI.first() ); 
   assert(poi);
   if (GetPosteriorFunction() == 0) { 
      coutE(InputArguments) <<  "BayesianCalculator::GetInterval() cannot make posterior Function " << std::endl;      
      return;
   }

   // need to remove the constant parameters
   RooArgList bindParams; 
   bindParams.add(fPOI);
   bindParams.add(fNuisanceParameters);
   
   // this code could be put inside the PosteriorCdfFunction
         
   //bindParams.Print("V");
         
   PosteriorCdfFunction cdf(*fLogLike, *fPriorPOI, bindParams, fIntegrationType, fNLLMin ); 
   if( cdf.HasError() ) { 
      coutE(Eval) <<  "BayesianCalculator: Numerical error computing CDF integral - try a different method " << std::endl;      
      return;
   }
                  
   //find the roots 

   ROOT::Math::RootFinder rf(kRootFinderType); 
         
   ccoutD(Eval) << "BayesianCalculator::GetInterval - finding roots of posterior using RF " << rf.Name() 
                << " with precision = " << fBrfPrecision;
         
   if (lowerCutOff > 0) { 
      cdf.SetOffset(lowerCutOff);
      ccoutD(NumIntegration) << "Integrating posterior to get cdf and search lower limit at p =" << lowerCutOff << std::endl;
      bool ok = rf.Solve(cdf, poi->getMin(),poi->getMax() , 200,fBrfPrecision, fBrfPrecision); 
      if( cdf.HasError() ) 
         coutW(Eval) <<  "BayesianCalculator: Numerical error integrating the  CDF   " << std::endl;      
      if (!ok) {
         coutE(NumIntegration) << "BayesianCalculator::GetInterval - Error from root finder when searching lower limit !" << std::endl;
         return;
      }
      fLower = rf.Root(); 
   }
   else { 
      fLower = poi->getMin(); 
   }
   if (upperCutOff < 1.0) {  
      cdf.SetOffset(upperCutOff);
      ccoutD(NumIntegration) << "Integrating posterior to get cdf and search upper interval limit at p =" << upperCutOff << std::endl;
      bool ok = rf.Solve(cdf, fLower,poi->getMax() , 200, fBrfPrecision, fBrfPrecision); 
      if( cdf.HasError() ) 
         coutW(Eval) <<  "BayesianCalculator: Numerical error integrating the  CDF   " << std::endl;            
      if (!ok)  {
         coutE(NumIntegration) << "BayesianCalculator::GetInterval - Error from root finder when searching upper limit !" << std::endl;
         return;
      }   
      fUpper = rf.Root(); 
   }
   else { 
      fUpper = poi->getMax(); 
   }

   fValidInterval = true; 
}

void BayesianCalculator::ApproximatePosterior() const { 
   // approximate posterior in nbins using a TF1 
   // scan the poi values and evaluate the posterior at each point 
   // and save the result in a cloned TF1 
   // For each point the posterior is evaluated by integrating the nuisance 
   // parameters 

   if (fApproxPosterior) { 
      // if number of bins of existing function is >= requested one - no need to redo the scan
      if (fApproxPosterior->GetNpx() >= fNScanBins) return;  
      // otherwise redo the scan
      delete fApproxPosterior; 
      fApproxPosterior = 0;
   }      

   RooAbsReal * posterior = GetPosteriorFunction();
   if (!posterior) return; 

   // try to reduce some error messages
   posterior->setEvalErrorLoggingMode(RooAbsReal::CountErrors);

   TF1 * tmp = posterior->asTF(fPOI); 
   assert(tmp != 0);
   // binned the function in nbins and evaluate at thos points
   if (fNScanBins > 0)  tmp->SetNpx(fNScanBins);  // if not use default of TF1 (which is 100)

   coutI(Eval) << "BayesianCalculator - scan posterior function in nbins = " << tmp->GetNpx() << std::endl;

   fApproxPosterior = (TF1*) tmp->Clone();
   // save this function for future reuse 
   // I can delete now original posterior and use this approximated copy
   delete tmp;
   TString name = posterior->GetName() + TString("_approx");
   TString title = posterior->GetTitle() + TString("_approx");
   RooAbsReal * posterior2 = new RooTFnBinding(name,title,fApproxPosterior,fPOI);
   if (posterior == fIntegratedLikelihood) { 
      delete fIntegratedLikelihood;
      fIntegratedLikelihood = posterior2; 
   }
   else if (posterior == fLikelihood) { 
      delete fLikelihood; 
      fLikelihood = posterior2;
   }
   else {
      assert(1); // should never happen this case
   }
}

void BayesianCalculator::ComputeIntervalFromApproxPosterior(double lowerCutOff, double upperCutOff ) const { 
   // compute the interval using the approximate posterior function

   ccoutD(Eval) <<  "BayesianCalculator: Compute interval from the approximate posterior " << std::endl;

   ApproximatePosterior();
   if (!fApproxPosterior) return;

   double prob[2]; 
   double limits[2];
   prob[0] = lowerCutOff;
   prob[1] = upperCutOff; 
   fApproxPosterior->GetQuantiles(2,limits,prob);
   fLower = limits[0]; 
   fUpper = limits[1];
   fValidInterval = true; 
}

void BayesianCalculator::ComputeShortestInterval( ) const { 
   // compute the shortest interval
   coutI(Eval) << "BayesianCalculator - computing shortest interval with CL = " << 1.-fSize << std::endl;

   // compute via the approx posterior function
   ApproximatePosterior(); 
   if (!fApproxPosterior) return;

   TH1D * h1 = dynamic_cast<TH1D*>(fApproxPosterior->GetHistogram() );
   assert(h1 != 0);
   h1->SetName(fApproxPosterior->GetName());
   // get bins and sort them 
   double * bins = h1->GetArray(); 
   int n = h1->GetSize()-2; // exclude under/overflow bins
   std::vector<int> index(n);
   TMath::Sort(n, bins, &index[0]); 
   // find cut off as test size
   double sum = 0; 
   double actualCL = 0;
   double upper =  h1->GetXaxis()->GetXmin();
   double lower =  h1->GetXaxis()->GetXmax();
   double norm = h1->GetSumOfWeights();

   for (int i = 0; i < n; ++i)  {
      int idx = index[i]; 
      double p = bins[ idx] / norm;
      sum += p;
      if (sum > 1.-fSize ) { 
         actualCL = sum - p;
         break; 
      }

      if ( h1->GetBinLowEdge(idx) < lower) 
         lower = h1->GetBinLowEdge(idx);
      if ( h1->GetXaxis()->GetBinUpEdge(idx) > upper) 
         upper = h1->GetXaxis()->GetBinUpEdge(idx);
   }

   ccoutD(Eval) << "BayesianCalculator::ComputeShortestInterval - actual interval CL = " 
                << actualCL << " difference from requested is " << (actualCL-(1.-fSize))/fSize*100. << "%  "
                << " limits are [ " << lower << " , " << " upper ] " << std::endl;


   if (lower < upper) { 
      fLower = lower;
      fUpper = upper; 



      if ( std::abs(actualCL-(1.-fSize)) > 0.1*(1.-fSize) )
         coutW(Eval) << "BayesianCalculator::ComputeShortestInterval - actual interval CL = " 
                     << actualCL << " differs more than 10% from desired CL value - must increase nbins " 
                     << n << " to an higher value " << std::endl;
   }
   else 
      coutE(Eval) << "BayesianCalculator::ComputeShortestInterval " << n << " bins are not sufficient " << std::endl;

   fValidInterval = true; 

}




} // end namespace RooStats

