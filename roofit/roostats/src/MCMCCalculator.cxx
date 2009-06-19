// @(#)root/roostats:$Id: MCMCCalculator.cxx 28978 2009-06-17 14:33:31Z kbelasco $
// Author: Kevin Belasco        17/06/2009

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooRealVar.h"
#include "RooNLLVar.h"
#include "RooGlobalFunc.h"
#include "RooDataSet.h"
#include "TRandom.h"
#include "TH1.h"
#include "TMath.h"
#include "TFile.h"
#include "RooStats/MCMCCalculator.h"
#include "RooStats/MCMCInterval.h"

ClassImp(RooStats::MCMCCalculator);

using namespace RooFit;
using namespace RooStats;

MCMCCalculator::MCMCCalculator()
{
   // default constructor
   fWS = NULL;
   fPOI = NULL;
   fNuisParams = NULL;
   fOwnsWorkspace = false;
   fPropFunc = NULL;
   fPdfName = NULL;
   fDataName = NULL;
   fNumIters = 0;
}

MCMCCalculator::MCMCCalculator(RooWorkspace& ws, RooAbsData& data,
                RooAbsPdf& pdf, RooArgSet& paramsOfInterest,
                ProposalFunction& proposalFunction, Int_t numIters,
                Double_t size)
{
   fOwnsWorkspace = false;
   SetWorkspace(ws);
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);
   SetTestSize(size);
   SetProposalFunction(proposalFunction);
   fNumIters = numIters;
}

MCMCCalculator::MCMCCalculator(RooAbsData& data, RooAbsPdf& pdf,
                RooArgSet& paramsOfInterest, ProposalFunction& proposalFunction,
                Int_t numIters, Double_t size)
{
   // alternate constructor
   fWS = new RooWorkspace();
   fOwnsWorkspace = true;
   SetData(data);
   SetPdf(pdf);
   SetParameters(paramsOfInterest);
   SetTestSize(size);
   SetProposalFunction(proposalFunction);
   fNumIters = numIters;
}

MCMCInterval* MCMCCalculator::GetInterval() const
{
   // Main interface to get a RooStats::ConfInterval.  

   RooAbsPdf* pdf = fWS->pdf(fPdfName);
   RooAbsData* data = fWS->data(fDataName);
   if (!data || !pdf || !fPOI) return 0;

   RooArgSet x;
   RooArgSet xPrime;
   RooRealVar* w = new RooRealVar("w", "weight", 0);
   RooArgSet* parameters = pdf->getParameters(data);
   RemoveConstantParameters(parameters);
   x.addClone(*parameters);
   x.addOwned(*w);
   xPrime.addClone(*parameters);

   RooDataSet* points = new RooDataSet("points", "Markov Chain", x, WeightVar(*w));

   TRandom gen;
   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RooAbsReal* nll = pdf->createNLL(*data, Constrain(*constrainedParams));
   delete constrainedParams;

   Int_t weight = 0;
   for (int i = 0; i < fNumIters; i++) {
     //       cout << "Iteration # " << i << endl;
     if (i % 100 == 0){
       fprintf(stdout, ".");
       fflush(NULL);
     }

      fPropFunc->Propose(xPrime);

      RooStats::SetParameters(&xPrime, nll->getParameters(*data));
      Double_t xPrimeNLL = nll->getVal();
      RooStats::SetParameters(&x, nll->getParameters(*data));
      Double_t xNLL = nll->getVal();
      Double_t diff = xPrimeNLL - xNLL;

      if (!fPropFunc->IsSymmetric(xPrime, x))
         diff += TMath::Log(fPropFunc->GetProposalDensity(xPrime, x)) - 
                 TMath::Log(fPropFunc->GetProposalDensity(x, xPrime));

      if (diff < 0.0) {
         // The proposed point (xPrime) has a higher likelihood than the
         // current (x), so go there

         // add the current point with the current weight
         points->addFast(x, (Double_t)weight);
         // reset the weight and go to xPrime
         weight = 1;
         RooStats::SetParameters(&xPrime, &x);
      }
      else {
         // generate numbers on a log distribution to decide
         // whether to go to xPrime or stay at x
         Double_t rand = TMath::Log(gen.Uniform(1.0));
         if (-1.0 * rand >= diff) {
            // we chose to go to the new proposed point xPrime
            // even though it has a lower likelihood than x

            // add the current point with the current weight
            points->addFast(x, (Double_t)weight);
            // reset the weight and go to xPrime
            weight = 1;
            RooStats::SetParameters(&xPrime, &x);
         } else {
            // stay at current point x
            weight++;
         }
      }
   }
   // make sure to add the last point
   points->addFast(x, (Double_t)weight);

   //TFile chainDataFile("chainData.root", "recreate");
   //points->Write();
   //chainDataFile.Close();

   MCMCInterval* interval = new MCMCInterval("mcmcinterval", "MCMCInterval",
                                             *fPOI, *points);
   interval->SetConfidenceLevel(1.0 - fSize);
   return interval;
}
