// @(#)root/roostats:$Id$
// Author: Sven Kreiss    June 2010
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/ToyMCSampler.h"

#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif

#ifndef ROO_DATA_HIST
#include "RooDataHist.h"
#endif

#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif

#include "TCanvas.h"
#include "RooPlot.h"
#include "RooRandom.h"

#include "RooStudyManager.h"
#include "RooStats/ToyMCStudy.h"
#include "RooSimultaneous.h"

#include "TMath.h"


ClassImp(RooStats::ToyMCSampler)

namespace RooStats {

class NuisanceParametersSampler {
   // Helper for ToyMCSampler. Handles all of the nuisance parameter related
   // functions. Once instantiated, it gives a new nuisance parameter point
   // at each call to nextPoint(...).

   public:
      NuisanceParametersSampler(RooAbsPdf *prior=NULL, const RooArgSet *parameters=NULL, Int_t nToys=1000, Bool_t asimov=kFALSE) :
         fPrior(prior),
         fParams(parameters),
         fNToys(nToys),
         fExpected(asimov),
         fPoints(NULL),
         fIndex(0)
      {
         if(prior) Refresh();
      }

      void NextPoint(RooArgSet& nuisPoint, Double_t& weight) {
         // Assigns new nuisance parameter point to members of nuisPoint.
         // nuisPoint can be more objects than just the nuisance
         // parameters.

         // check whether to get new set of nuisanceParPoints
         if (fIndex >= fNToys) {
            Refresh();
            fIndex = 0;
         }

         // get value
         nuisPoint =  *fPoints->get(fIndex++);
         weight = fPoints->weight();

         // check whether result will have any influence
         if(fPoints->weight() == 0.0) {
            oocoutI((TObject*)NULL,Generation) << "Weight 0 encountered. Skipping." << endl;
            NextPoint(nuisPoint, weight);
         }
      }


   protected:

      void Refresh() {
         // Creates the initial set of nuisance parameter points. It also refills the
         // set with new parameter points if called repeatedly. This helps with
         // adaptive sampling as the required number of nuisance parameter points
         // might increase during the run.

         if (!fPrior || !fParams) return;

         if (fPoints) delete fPoints;

         if (fExpected) {
            // UNDER CONSTRUCTION
            oocoutI((TObject*)NULL,InputArguments) << "Using expected nuisance parameters." << endl;

            int nBins = fNToys;

            // From FeldmanCousins.cxx:
            // set nbins for the POI
            TIter it2 = fParams->createIterator();
            RooRealVar *myarg2;
            while ((myarg2 = dynamic_cast<RooRealVar*>(it2.Next()))) {
              myarg2->setBins(nBins);
            }


            fPoints = fPrior->generateBinned(
               *fParams,
               RooFit::ExpectedData(),
               RooFit::NumEvents(1) // for Asimov set, this is only a scale factor
            );
            if(fPoints->numEntries() != fNToys) {
               fNToys = fPoints->numEntries();
               oocoutI((TObject*)NULL,InputArguments) <<
                  "Adjusted number of toys to number of bins of nuisance parameters: " << fNToys << endl;
            }

/*
            // check
            TCanvas *c1 = new TCanvas;
            RooPlot *p = dynamic_cast<RooRealVar*>(fParams->first())->frame();
            fPoints->plotOn(p);
            p->Draw();
            for(int x=0; x < fPoints->numEntries(); x++) {
               fPoints->get(x)->Print("v");
               cout << fPoints->weight() << endl;
            }
*/

         }else{
            oocoutI((TObject*)NULL,InputArguments) << "Using randomized nuisance parameters." << endl;

            fPoints = fPrior->generate(*fParams, fNToys);
         }
      }


   private:
      RooAbsPdf *fPrior;           // prior for nuisance parameters
      const RooArgSet *fParams;    // nuisance parameters
      Int_t fNToys;
      Bool_t fExpected;

      RooAbsData *fPoints;         // generated nuisance parameter points
      Int_t fIndex;                // current index in fPoints array
};






Bool_t ToyMCSampler::CheckConfig(void) {
   // only checks, no guessing/determination (do this in calculators,
   // e.g. using ModelConfig::GuessObsAndNuisance(...))
   bool goodConfig = true;

   if(!fTestStat) { ooccoutE((TObject*)NULL,InputArguments) << "Test statistic not set." << endl; goodConfig = false; }
   if(!fObservables) { ooccoutE((TObject*)NULL,InputArguments) << "Observables not set." << endl; goodConfig = false; }
   if(!fNullPOI) { ooccoutE((TObject*)NULL,InputArguments) << "Parameter values used to evaluate for test statistic  not set." << endl; goodConfig = false; }
   if(!fPdf) { ooccoutE((TObject*)NULL,InputArguments) << "Pdf not set." << endl; goodConfig = false; }

   return goodConfig;
}



SamplingDistribution* ToyMCSampler::GetSamplingDistribution(RooArgSet& paramPointIn) {
   // Use for serial and parallel runs.

   // ======= S I N G L E   R U N ? =======
   if(!fProofConfig)
      return GetSamplingDistributionSingleWorker(paramPointIn);


   // ======= P A R A L L E L   R U N =======
   CheckConfig();

   // turn adaptive sampling off if given
   if(fToysInTails) {
      fToysInTails = 0;
      oocoutW((TObject*)NULL, InputArguments)
         << "Adaptive sampling in ToyMCSampler is not supported for parallel runs."
         << endl;
   }

   // adjust number of toys on the slaves to keep the total number of toys constant
   Int_t totToys = fNToys;
   fNToys = (int)ceil((double)fNToys / (double)fProofConfig->GetNExperiments()); // round up

   // create the study instance for parallel processing
   ToyMCStudy toymcstudy;
   toymcstudy.SetToyMCSampler(*this);
   toymcstudy.SetParamPointOfInterest(paramPointIn);

   // temporary workspace for proof to avoid messing with TRef
   RooWorkspace w(fProofConfig->GetWorkspace());
   RooStudyManager studymanager(w, toymcstudy);
   studymanager.runProof(fProofConfig->GetNExperiments(), fProofConfig->GetHost(), fProofConfig->GetShowGui());

   SamplingDistribution *result = new SamplingDistribution(GetSamplingDistName().c_str(), GetSamplingDistName().c_str());
   toymcstudy.merge(*result);

   // reset the number of toys
   fNToys = totToys;

   return result;
}

SamplingDistribution* ToyMCSampler::GetSamplingDistributionSingleWorker(RooArgSet& paramPointIn) {
   // This is the main function for serial runs. It is called automatically
   // from inside GetSamplingDistribution when no ProofConfig is given.
   // You should not call this function yourself. This function should
   // be used by ToyMCStudy on the workers (ie. when you explicitly want
   // a serial run although ProofConfig is present).

   CheckConfig();

   std::vector<Double_t> testStatVec;
   std::vector<Double_t> testStatWeights;

   // important to cache the paramPoint b/c test statistic might 
   // modify it from event to event
   RooArgSet *paramPoint = (RooArgSet*) paramPointIn.snapshot();
   RooArgSet *allVars = fPdf->getVariables();
   RooArgSet *allVars2 = 0;   //importance density variables 
   if (fImportanceDensity) { 
      // in case of importance sampling include in allVars 
      // also any extra variables defined for the importance sampling
      allVars2 = fImportanceDensity->getVariables(); 
      allVars->add(*allVars2);
   }
   RooArgSet *saveAll = (RooArgSet*) allVars->snapshot();

   // create nuisance parameter points
   NuisanceParametersSampler *np = NULL;
   if(fPriorNuisance && fNuisancePars)
      np = new NuisanceParametersSampler(fPriorNuisance, fNuisancePars, fNToys, fExpectedNuisancePar);

   // counts the number of toys in the limits set for adaptive sampling
   // (taking weights into account)
   Double_t toysInTails = 0.0;

   for (Int_t i = 0; i < fMaxToys; ++i) {

      // status update
      if ( i% 500 == 0 && i>0 ) {
         oocoutP((TObject*)0,Generation) << "generated toys: " << i << " / " << fNToys;
         if (fToysInTails) ooccoutP((TObject*)0,Generation) << " (tails: " << toysInTails << " / " << fToysInTails << ")" << std::endl;
         else ooccoutP((TObject*)0,Generation) << endl;
      }

      // create toy data and calculate value of test statistic
      Double_t value, weight;
      if (np) { // use nuisance parameters?


         // set variables to requested parameter point
         *allVars = *paramPoint;

         // get nuisance parameter point and weight
         np->NextPoint(*allVars, weight);

         // generate toy data for this parameter point
         RooAbsData* toydata = GenerateToyData(*allVars);
         // evaluate test statistic, that only depends on null POI
         value = fTestStat->Evaluate(*toydata, *fNullPOI);



         if(fImportanceDensity) {
            // Importance Sampling: adjust weight
            // Source: presentation by Michael Woodroofe

            // NOTE that importance density is used only when sampling also the  
            // nuisance parameters
            // One has to be careful not to have nuisance parameter in the snapshot 
            // otherwise they will be not smeared in GenerateToyData


            // get the NLLs of the importance density and the pdf to sample
            if (fImportanceSnapshot)   *allVars = *fImportanceSnapshot;

            RooAbsReal *impNLL = fImportanceDensity->createNLL(*toydata, RooFit::Extended(kFALSE), RooFit::CloneData(kFALSE));
            double impNLLVal = impNLL->getVal();
            delete impNLL;


            *allVars = *paramPoint;
            RooAbsReal *pdfNLL = fPdf->createNLL(*toydata, RooFit::Extended(kFALSE), RooFit::CloneData(kFALSE));
            double pdfNLLVal = pdfNLL->getVal();
            delete pdfNLL;

            // L(pdf) / L(imp)  =  exp( NLL(imp) - NLL(pdf) )
            weight *= exp(impNLLVal - pdfNLLVal);


         }

         delete toydata;

      }else{

         // set variables to requested parameter point
         *allVars = *paramPoint;
         // generate toy data for this parameter point
         RooAbsData* toydata = GenerateToyData(*allVars);
         // evaluate test statistic, that only depends on null POI
         value = fTestStat->Evaluate(*toydata, *fNullPOI);
         weight = -1.;
         delete toydata;
      }

      // check for nan
      if(value != value) {
         oocoutW((TObject*)NULL, Generation) << "skip: " << value << ", " << weight << endl;
         continue;
      }

      // add results
      testStatVec.push_back(value);
      if(weight >= 0.) testStatWeights.push_back(weight);

      // adaptive sampling checks
      if (value <= fAdaptiveLowLimit  ||  value >= fAdaptiveHighLimit) {
         if(weight >= 0.) toysInTails += weight;
         else toysInTails += 1.;
      }
      if (toysInTails >= fToysInTails  &&  i+1 >= fNToys) break;
   }


   // clean up
   *allVars = *saveAll;
   delete saveAll;
   delete allVars;
   if (allVars2) delete allVars2;
   if(np) delete np;

   // return
   if (testStatWeights.size()) {
      return new SamplingDistribution(
         fSamplingDistName.c_str(),
         fSamplingDistName.c_str(),
         testStatVec,
         testStatWeights,
         fTestStat->GetVarName()
      );

   }
   return new SamplingDistribution(
      fSamplingDistName.c_str(),
      fSamplingDistName.c_str(),
      testStatVec,
      fTestStat->GetVarName()
   );
}


RooAbsData* ToyMCSampler::GenerateToyData(RooArgSet& /*nullPOI*/) const {
   // This method generates a toy data set for the given parameter point taking
   // global observables into account.

   if (fObservables == NULL) { 
      ooccoutE((TObject*)NULL,InputArguments) << "Observables not set." << endl; 
      return 0; 
   }

   RooArgSet observables(*fObservables);
   if(fGlobalObservables  &&  fGlobalObservables->getSize()) {
      observables.remove(*fGlobalObservables);

      // generate one set of global observables and assign it
      // has problem for sim pdfs
      RooSimultaneous* simPdf = dynamic_cast<RooSimultaneous*>(fPdf);
      if(!simPdf){
	RooDataSet *one = fPdf->generate(*fGlobalObservables, 1);
	const RooArgSet *values = one->get();
	RooArgSet *allVars = fPdf->getVariables();
	*allVars = *values;
	delete allVars;
	delete values;
	delete one;
      } else {

         //try fix for sim pdf
         TIterator* iter = simPdf->indexCat().typeIterator() ;
         RooCatType* tt = NULL;
         while((tt=(RooCatType*) iter->Next())) {
            
            // Get pdf associated with state from simpdf
            RooAbsPdf* pdftmp = simPdf->getPdf(tt->GetName()) ;
            
            // Generate only global variables defined by the pdf associated with this state
            RooArgSet* globtmp = pdftmp->getObservables(*fGlobalObservables) ;
            RooDataSet* tmp = pdftmp->generate(*globtmp,1) ;
	  
            // Transfer values to output placeholder
            *globtmp = *tmp->get(0) ;
            
            // Cleanup
            delete globtmp ;
            delete tmp ;
         }
      }
   } 
   

   RooAbsData* data = NULL;

   if(!fImportanceDensity) {
      // no Importance Sampling
      data = Generate(*fPdf, observables);
   }else{

      // Importance Sampling
      RooArgSet* allVars = fPdf->getVariables();
      RooArgSet* allVars2 = fImportanceDensity->getVariables();
      allVars->add(*allVars2);
      const RooArgSet* saveVars = (const RooArgSet*)allVars->snapshot();

      // the number of events generated is either the given fNEvents or
      // in case this is not given, the expected number of events of
      // the pdf with a Poisson fluctuation
      int forceEvents = 0;
      if(fNEvents == 0) {
         forceEvents = (int)fPdf->expectedEvents(observables);
         forceEvents = RooRandom::randomGenerator()->Poisson(forceEvents);
      }

      // need to be careful here not to overwrite the current state of the
      // nuisance parameters, ie they must not be part of the snapshot
      if(fImportanceSnapshot) *allVars = *fImportanceSnapshot;

      // generate with the parameters configured in this class
      //   NULL => no protoData
      //   overwriteEvents => replaces fNEvents it would usually take
      data = Generate(*fImportanceDensity, observables, NULL, forceEvents);

      *allVars = *saveVars;
      delete allVars;
      delete allVars2;
      delete saveVars;
   }

   return data;
}

RooAbsData* ToyMCSampler::Generate(RooAbsPdf &pdf, RooArgSet &observables, const RooDataSet* protoData, int forceEvents) const {
   // This is the generate function to use in the context of the ToyMCSampler
   // instead of the standard RooAbsPdf::generate(...).
   // It takes into account whether the number of events is given explicitly
   // or whether it should use the expected number of events. It also takes
   // into account the option to generate a binned data set (ie RooDataHist).

   if(fProtoData) {
      protoData = fProtoData;
      forceEvents = protoData->numEntries();
   }

   RooAbsData *data = NULL;
   int events = forceEvents;
   if(events == 0) events = fNEvents;

   if(events == 0) {
      if( pdf.canBeExtended() && pdf.expectedEvents(observables) > 0) {
         if(fGenerateBinned) {
            if(protoData) data = pdf.generateBinned(observables, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true));
            else          data = pdf.generateBinned(observables, RooFit::Extended());
         }else{
            if(protoData) data = pdf.generate      (observables, RooFit::Extended(), RooFit::ProtoData(*protoData, true, true));
            else          data = pdf.generate      (observables, RooFit::Extended());
         }
      }else{
         oocoutE((TObject*)0,InputArguments)
            << "ToyMCSampler: Error : pdf is not extended and number of events per toy is zero"
            << endl;
      }
   }else{
      if(fGenerateBinned) {
         if(protoData) data = pdf.generateBinned(observables, events, RooFit::ProtoData(*protoData, true, true));
         else          data = pdf.generateBinned(observables, events);
      }else{
         if(protoData) data = pdf.generate      (observables, events, RooFit::ProtoData(*protoData, true, true));
         else          data = pdf.generate      (observables, events);
      }
   }

   return data;
}





} // end namespace RooStats
