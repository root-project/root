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
   LikelihoodFunction(RooFunctor & f, double offset = 0) : fFunc(f), fOffset(offset) {}

   double operator() (const double *x ) const { 
      double nll = fFunc(x) - fOffset;
      double likelihood =  std::exp(-nll);
//       ooccoutD((TObject*)0,NumIntegration)  
//      std::cout     << "x[0] = " << x[0] << " x[1] = " << x[1]  << " func " << fFunc(x) << " nll = " << nll << "offset " << fOffset << " likelihood = " << likelihood << std::endl;
      return likelihood; 
   }

   // for the 1D case
   double operator() (double x) const { 
      double tmp = x; 
      return (*this)(&tmp); 
   }

   RooFunctor & fFunc;     // functor representing the nll function 
   double fOffset;         //  offset used to bring the nll in a reasanble range for computing the exponent whch can be computed 
};

class PosteriorCdfFunction : public ROOT::Math::IGenFunction { 

public:
   PosteriorCdfFunction(ROOT::Math::IntegratorMultiDim & ig, double * xmin, double * xmax) : 
      fIntegrator(ig), fXmin(xmin), fXmax(xmax), fNorm(1.0), fOffset(0.0), fMaxX(xmax[0]), 
      fHasNorm(false), fUseOldValues(true) 
   {
      // compute first the normalization with  the poi 
      fNorm = (*this)(xmax[0] );  
      fHasNorm = true; 
      fNormCdfValues.insert(std::make_pair(xmin[0], 0) );
      fNormCdfValues.insert(std::make_pair(xmax[0], 1.0) );

   }

//    PosteriorCdfFunction(ROOT::Math::IntegratorMultiDim & ig, double * xmin, double * xmax, double norm) : 
//       fIntegrator(ig), fXmin(xmin), fXmax(xmax), fNorm(norm), fOffset(0.0), fMaxX(xmax[0]), fHasNorm(true)  {
//       // compute posterior cdf from a normalization given from outside 
//    }

   ROOT::Math::IGenFunction * Clone() const { 
      ooccoutD((TObject*)0,NumIntegration) << " cloning function .........." << std::endl;
      return new PosteriorCdfFunction(*this); 
   }

   void SetOffset(double offset) { fOffset = offset; }

private:
   double DoEval (double x) const { 
      // evaluate cdf at poi value x by integrating poi from [-xmin,x] and all the nuisances  
      fXmax[0] = x;
      if (x <= fXmin[0] ) return -fOffset; 
      // could also avoid a function evaluation at maximum
      if (x >= fMaxX && fHasNorm) return 1. - fOffset;  // cdf is bound to these values

      // computes the intergral using previous cdf estimate
      double  normcdf0 = 0; 
      if (fHasNorm && fUseOldValues) { 
         // look in the map of the stored cdf values the closes one
         std::map<double,double>::iterator itr = fNormCdfValues.upper_bound(x); 
         itr--;   // upper bound returns a poistion 1 up of the value we want
         if (itr != fNormCdfValues.end() ) { 
            fXmin[0] = itr->first; 
            normcdf0 = itr->second;
            ooccoutD((TObject*)0,NumIntegration) << "PosteriorCdfFunction:   position for x = " << x << " is  = " << itr->first << std::endl; 
         }
      }


      double cdf = fIntegrator.Integral(fXmin,fXmax);  
      double error = fIntegrator.Error(); 
      double normcdf =  cdf/fNorm;  // normalize the cdf 
      ooccoutD((TObject*)0,NumIntegration) << "PosteriorCdfFunction:   x0 = " << fXmin[0] << " x =  " 
                                          << x << " cdf(x) =  " << cdf << " +/- " << error 
                                          << "  norm-cdf(x) = " << normcdf << std::endl; 
      if (cdf != 0 && error/cdf > 0.2 ) 
         oocoutW((TObject*)0,NumIntegration) << "PosteriorCdfFunction: integration error  is larger than 20 %   x0 = " << fXmin[0]  
                                              << " x = " << x << " cdf(x) = " << cdf << " +/- " << error << std::endl;

      if (!fHasNorm) { 
         oocoutI((TObject*)0,NumIntegration) << "PosteriorCdfFunction - integral of posterior = " 
                                             << cdf << " +/- " << error << std::endl; 
         return cdf; 
      }

      normcdf += normcdf0;

      // store values in the map
      if (fUseOldValues) { 
         fNormCdfValues.insert(std::make_pair(x, normcdf) );
      }

      if (normcdf > 1. + 3 * error/fNorm) {
         oocoutW((TObject*)0,NumIntegration) << "PosteriorCdfFunction: normalized cdf values is larger than 1" 
                                              << " x = " << x << " normcdf(x) = " << normcdf << " +/- " << error/fNorm << std::endl;
      }

      return normcdf - fOffset;  // apply an offset (for finding the roots) 
   }

   ROOT::Math::IntegratorMultiDim & fIntegrator; 
   mutable double * fXmin; 
   mutable double * fXmax; 
   double fNorm; 
   double fOffset;
   double fMaxX;  // maxumum value of x 
   bool fHasNorm; // flag to control first call to the function 
   bool fUseOldValues;  // use old cdf values
   mutable std::map<double,double> fNormCdfValues; 
};

class PosteriorFunction : public ROOT::Math::IGenFunction { 

public: 


   PosteriorFunction(RooAbsReal & nll, RooRealVar & poi, RooArgList & nuisParams, const char * integType = 0, double norm = 1.0, double nllOffset = 0) :
      fFunctor(nll, nuisParams, RooArgList() ),
      fLikelihood(fFunctor, nllOffset), 
      fPoi(&poi),
      fXmin(nuisParams.getSize() ),
      fXmax(nuisParams.getSize() ), 
      fNorm(norm)      
   { 

      for (unsigned int i = 0; i < fXmin.size(); ++i) { 
         RooRealVar & var = (RooRealVar &) nuisParams[i]; 
         fXmin[i] = var.getMin(); 
         fXmax[i] = var.getMax();
      }
      if (fXmin.size() == 1) { // 1D case  
         fIntegratorOneDim = std::auto_ptr<ROOT::Math::Integrator>(
            new ROOT::Math::Integrator(ROOT::Math::IntegratorOneDim::GetType(integType) ) );
         fIntegratorOneDim->SetFunction(fLikelihood);
         // interested only in relative tolerance
         //fIntegratorOneDim->SetAbsTolerance(1.E-300);
      }
      else if (fXmin.size() > 1) { // multiDim case          
         fIntegratorMultiDim = 
            std::auto_ptr<ROOT::Math::IntegratorMultiDim>(
               new ROOT::Math::IntegratorMultiDim(ROOT::Math::IntegratorMultiDim::GetType(integType) ) );
         fIntegratorMultiDim->SetFunction(fLikelihood, fXmin.size());
         //fIntegratorMultiDim->SetAbsTolerance(1.E-300);
      }
   }
      
      
   ROOT::Math::IGenFunction * Clone() const { 
      assert(1); 
      return 0; // cannot clone this function for integrator 
   } 
   
private: 
   double DoEval (double x) const { 
      // evaluate posterior function at a poi value x by integrating all nuisance parameters

      fPoi->setVal(x);
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

      if (f != 0 && error/f > 0.2 ) 
         ooccoutW((TObject*)0,NumIntegration) << "PosteriorFunction::DoEval - Error from integration in " 
                                              << fXmin.size() <<  " Dim is larger than 20 % " 
                                              << "x = " << x << " p(x) = " << f << " +/- " << error << std::endl;

      return f / fNorm;
   }

   RooFunctor fFunctor; 
   LikelihoodFunction fLikelihood; 
   RooRealVar * fPoi;
   std::auto_ptr<ROOT::Math::Integrator>  fIntegratorOneDim; 
   std::auto_ptr<ROOT::Math::IntegratorMultiDim>  fIntegratorMultiDim; 
   std::vector<double> fXmin; 
   std::vector<double> fXmax; 
   double fNorm;
};



////////////////////////////////////////////////////////////////////////////////////////////////////////////

BayesianCalculator::BayesianCalculator() :
  fData(0),
  fPdf(0),
  fPriorPOI(0),
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
  fProductPdf (0), fLogLike(0), fLikelihood (0), fIntegratedLikelihood (0), fPosteriorPdf(0),
  fPosteriorFunction(0), fApproxPosterior(0),
  fLower(0), fUpper(0), 
  fNLLMin(0),
  fSize(0.05), fLeftSideFraction(0.5), 
  fBrfPrecision(0.00005), 
  fNScanBins(-1),
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
   fProductPdf (0), fLogLike(0), fLikelihood (0), fIntegratedLikelihood (0), fPosteriorPdf(0),
   fPosteriorFunction(0), fApproxPosterior(0),
   fLower(0), fUpper(0), 
   fNLLMin(0),
   fSize(0.05), fLeftSideFraction(0.5), 
   fBrfPrecision(0.00005), 
   fNScanBins(-1),
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

   // create a unique name for the product pdf 
   TString prodName = TString("product_") + TString(fPdf->GetName()) + TString("_") + TString(fPriorPOI->GetName() );   
   fProductPdf = new RooProdPdf(prodName,"",RooArgList(*fPdf,*fPriorPOI));

   RooArgSet* constrainedParams = fProductPdf->getParameters(*fData);
   // remove the constant parameters
   RemoveConstantParameters(constrainedParams);
   
   //constrainedParams->Print("V");

   // use RooFit::Constrain() to make product of likelihood with prior pdf
   fLogLike = fProductPdf->createNLL(*fData, RooFit::Constrain(*constrainedParams) );

   ccoutD(Eval) <<  "BayesianCalculator::GetPosteriorFunction : " 
                << " pdf x priors = " <<  fProductPdf->getVal() 
                << " neglogLikelihood = " << fLogLike->getVal() << std::endl;

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


      // case of no nuisance parameters 
      TString likeName = TString("likelihood_") + TString(fProductPdf->GetName());   
      TString formula; 
      formula.Form("exp(-@0+%f)",fNLLMin);
      fLikelihood = new RooFormulaVar(likeName,formula,RooArgList(*fLogLike));
      
      // if no nuisance parameter we can just return the likelihood funtion
      if (fNuisanceParameters.getSize() == 0) return fLikelihood; 

      // case of using RooFit for the integration
      fIntegratedLikelihood = fLikelihood->createIntegral(fNuisanceParameters);
   }

   else  { 

      // use ROOT integration method if there are nuisance parameters 

      RooArgList nuisParams(fNuisanceParameters); 
      fPosteriorFunction = new PosteriorFunction(*fLogLike, *poi, nuisParams, fIntegrationType, 1.,fNLLMin ); 
      
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

   if (!fLikelihood) GetPosteriorFunction(); 

   // if a scan is requested approximate the posterior
   if (fNScanBins > 0) ApproximatePosterior();

   RooAbsReal * posterior = fIntegratedLikelihood; 
   if (norm) posterior = fPosteriorPdf; 
   if (!posterior) { 
      posterior = GetPosteriorFunction();
      if (norm) { 
         if (fPosteriorPdf) delete fPosteriorPdf;
         fPosteriorPdf = GetPosteriorPdf();
         posterior = fPosteriorPdf;
      }
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
            // case of no nuisance - just use createCdf from roofit
         }
         else { 
            ComputeIntervalUsingRooFit(lowerCutOff, upperCutOff);      
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
   if (!ret) coutE(Eval) << "BayesianCalculator::GetInterval "
                         << "Error returned from Root finder, estimated interval is not fully correct" 
                         << std::endl;

   poi->setVal(tmpVal); // patch: restore the original value of poi
   
   delete cdf_bind;
   delete cdf;
   fValidInterval = true; 
}

void BayesianCalculator::ComputeIntervalFromCdf(double lowerCutOff, double upperCutOff ) const { 
   // compute the interval using Cdf integration

   coutI(Eval) <<  "BayesianCalculator: Compute the interval from the posterior cdf " << std::endl;

   RooRealVar* poi = dynamic_cast<RooRealVar*>( fPOI.first() ); 
   assert(poi);
   if (GetPosteriorFunction() == 0) return; 

   // need to remove the constant parameters
   RooArgList bindParams; 
   bindParams.add(fPOI);
   bindParams.add(fNuisanceParameters);
   
   // this code could be put inside the PosteriorCdfFunction
         
   RooFunctor functor_nll(*fLogLike, bindParams, RooArgList());
         
   
   // compute the integral of the exp(-nll) function
   LikelihoodFunction fll(functor_nll, fNLLMin);
      
   ROOT::Math::IntegratorMultiDim ig(ROOT::Math::IntegratorMultiDim::GetType(fIntegrationType)); 
   ig.SetFunction(fll,bindParams.getSize()); 
   
   std::vector<double> pmin(bindParams.getSize());
   std::vector<double> pmax(bindParams.getSize());
   std::vector<double> par(bindParams.getSize());
   for (unsigned int i = 0; i < pmin.size(); ++i) { 
      RooRealVar & var = (RooRealVar &) bindParams[i]; 
      pmin[i] = var.getMin(); 
      pmax[i] = var.getMax();
      par[i] = var.getVal();
   } 
         
         
   //bindParams.Print("V");
         
   PosteriorCdfFunction cdf(ig, &pmin[0], &pmax[0] ); 
         
         
   //find the roots 

   ROOT::Math::RootFinder rf(kRootFinderType); 
         
   ccoutD(Eval) << "BayesianCalculator::GetInterval - finding roots of posterior using RF " << rf.Name() 
                << " with precision = " << fBrfPrecision;
         
   if (lowerCutOff > 0) { 
      cdf.SetOffset(lowerCutOff);
      ccoutD(NumIntegration) << "Integrating posterior to get cdf and search lower limit at p =" << lowerCutOff << std::endl;
      bool ok = rf.Solve(cdf, poi->getMin(),poi->getMax() , 200,fBrfPrecision, fBrfPrecision); 
      if (!ok) 
         coutE(NumIntegration) << "BayesianCalculator::GetInterval - Error from root finder when searching lower limit !" << std::endl;
      fLower = rf.Root(); 
   }
   else { 
      fLower = poi->getMin(); 
   }
   if (upperCutOff < 1.0) {  
      cdf.SetOffset(upperCutOff);
      ccoutD(NumIntegration) << "Integrating posterior to get cdf and search upper interval limit at p =" << upperCutOff << std::endl;
      bool ok = rf.Solve(cdf, fLower,poi->getMax() , 200, fBrfPrecision, fBrfPrecision); 
      if (!ok) 
         coutE(NumIntegration) << "BayesianCalculator::GetInterval - Error from root finder when searching upper limit !" << std::endl;
      
      fUpper = rf.Root(); 
   }
   else { 
      fUpper = poi->getMax(); 
   }
   fValidInterval = true; 
}

void BayesianCalculator::ApproximatePosterior() const { 
   // approximate posterior in nbins using a TF1 and
   // scan the values

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

