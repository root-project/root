// @(#)root/roostats:$Id: ModelConfig.h 27519 2009-02-19 13:31:41Z pellicci $
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
#include "RooAbsRealLValue.h"
#include "RooArgSet.h"
#include "RooBrentRootFinder.h"
#include "RooFormulaVar.h"
#include "RooGenericPdf.h"
#include "RooPlot.h"
#include "RooProdPdf.h"

// include header file of this class 
#include "RooStats/BayesianCalculator.h"
#include "RooStats/ModelConfig.h"

ClassImp(RooStats::BayesianCalculator)

namespace RooStats { 




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
  fNuisanceParameters(*nuisanceParameters)
{
   // constructor
   fLowerLimit = -999;
   fUpperLimit = +999;
}

BayesianCalculator::BayesianCalculator( RooAbsData& data,
                       ModelConfig & model) : 
   fData(&data), 
   fPdf(model.GetPdf()),
   fPOI( *model.GetParametersOfInterest() ),
   fPriorPOI( model.GetPriorPdf()),
   fNuisanceParameters( *model.GetNuisanceParameters() )
{
   // constructor from Model Config
}


BayesianCalculator::~BayesianCalculator()
{
   // destructor
}


// RooAbsPdf* BayesianNumIntCalculator::GetPosteriorPdf()
// {
//   if (fPosteriorPdf==0) {
//     fProductPdf = new RooProdPdf("product","",RooArgList(*fPdf,*fPriorPOI));
//     RooAbsReal* nll = product->createNLL(*fData);
//     fLikelihood = new RooFormulaVar("fLikelihood","exp(-@0)",RooArgList(*nll));
//     fIntegratedLikelihood = fLikelihood->createIntegral(*fNuisanceParameters);

//     TString posterior_name = this->GetName();
//     posterior_name += "_posteriorPdf";
//     fPosteriorPdf = new RooGenericPdf(posterior_name,"@0",fIntegratedLikelihood);
//   }
//   return fPosteriorPdf;
// }


RooPlot* BayesianCalculator::PlotPosterior()
{
   RooProdPdf posterior("posterior","",RooArgList(*fPdf,*fPriorPOI));
   RooAbsReal* nll = posterior.createNLL(*fData);
   RooFormulaVar like("like","exp(-@0)",RooArgList(*nll));

   like.createIntegral(fNuisanceParameters);

   RooAbsRealLValue * poi = dynamic_cast<RooAbsRealLValue *>(fPOI.first() );
   assert(poi );
   RooPlot* plot = poi->frame();
   like.plotOn(plot);
   //like.plotOn(plot,RooFit::Range(fLowerLimit,fUpperLimit),RooFit::FillColor(kYellow));
   //plot->GetYaxis()->SetTitle("posterior probability");
   plot->Draw();

   delete nll;
   return plot; 
}


SimpleInterval* BayesianCalculator::GetInterval() const
{
   // returns a SimpleInterval with the lower/upper limit on the scanned variable

   if (!fPdf || !fPriorPOI) return 0; 
   if (fPOI.getSize() > 1) { 
      std::cerr << "BayesianCalculator::GetInterval - current implementation works only on 1D intervals" << std::endl;
      return 0; 
   }

   RooProdPdf posterior("posterior","",RooArgList(*fPdf,*fPriorPOI));
   RooAbsReal* nll = posterior.createNLL(*fData);
   RooFormulaVar like("like","exp(-@0)",RooArgList(*nll));

   RooAbsReal * like2 = &like; 
   if (fNuisanceParameters.getSize() > 0 ) 
      like2 = like.createIntegral(fNuisanceParameters);

   RooGenericPdf pp("pp","@0",*like2);
   RooAbsReal* cdf = pp.createCdf(fPOI);
   

   RooAbsFunc* cdf_bind = cdf->bindVars(fPOI,&fPOI);
   RooBrentRootFinder brf(*cdf_bind);
   brf.setTol(0.00005);
   
   RooAbsRealLValue * poi = dynamic_cast<RooAbsRealLValue *>( fPOI.first()); 
   assert(poi);
   
   double y = fSize;
   brf.findRoot(fLowerLimit,poi->getMin(),poi->getMax(),y);
   
   y=1-fSize;
   brf.findRoot(fUpperLimit,poi->getMin(),poi->getMax(),y);
   
   delete cdf_bind;
   delete cdf;
   delete nll;

   TString interval_name = TString("BayesianInterval_a") + TString(this->GetName());
   SimpleInterval* interval = new SimpleInterval(interval_name,"SimpleInterval from BayesianCalculator",poi,fLowerLimit,fUpperLimit);
  
   return interval;
}

} // end namespace RooStats

