/* @(#)root/roostats:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__ 

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ namespace RooStats;
#pragma link C++ namespace RooStats::NumberCountingUtils;

// for auto-loading namespaces
#ifdef USE_FOR_AUTLOADING
#pragma link C++ class RooStats::NumberCountingUtils;
#pragma link C++ class RooStats;
#endif

#pragma link C++ class RooStats::SPlot+;
#pragma link C++ class RooStats::NumberCountingPdfFactory+;


#pragma link C++ class RooStats::HypoTestResult+;
#pragma link C++ class RooStats::ConfInterval+; // interface, not concrete
#pragma link C++ class RooStats::SimpleInterval+;
#pragma link C++ class RooStats::LikelihoodInterval+;
#pragma link C++ class RooStats::PointSetInterval+;

#pragma link C++ class RooStats::ModelConfig+;

#pragma link C++ class RooStats::IntervalCalculator+; // interface, not concrete
#pragma link C++ class RooStats::HypoTestCalculator+; // interface, not concrete
#pragma link C++ class RooStats::CombinedCalculator+; // interface, not concrete
#pragma link C++ class RooStats::ProfileLikelihoodCalculator+; 

#pragma link C++ class RooStats::MCMCCalculator+; 
#pragma link C++ class RooStats::MCMCInterval+; 
#pragma link C++ class RooStats::ProposalFunction+; 
#pragma link C++ class RooStats::UniformProposal+; 

#pragma link C++ class RooStats::HybridCalculator+;
#pragma link C++ class RooStats::HybridPlot+;
#pragma link C++ class RooStats::HybridResult+;

#pragma link C++ class RooStats::TestStatSampler+; // interface, not concrete
#pragma link C++ class RooStats::DebuggingSampler+;
#pragma link C++ class RooStats::ToyMCSampler+;

#pragma link C++ class RooStats::TestStatistic+; // interface
#pragma link C++ class RooStats::DebuggingTestStat+;
#pragma link C++ class RooStats::ProfileLikelihoodTestStat+;
#pragma link C++ class RooStats::NumEventsTestStat+;

#pragma link C++ class RooStats::SamplingDistribution+;
#pragma link C++ class RooStats::NeymanConstruction+;
#pragma link C++ class RooStats::FeldmanCousins+;

// in progress
#pragma link C++ class RooStats::ConfidenceBelt+; 
#pragma link C++ class RooStats::AcceptanceRegion+; 
#pragma link C++ class RooStats::SamplingSummary+; 
#pragma link C++ class RooStats::SamplingSummaryLookup+; 

#pragma link C++ class RooStats::BernsteinCorrection+;

#pragma link C++ class RooStats::SamplingDistPlot+;
#pragma link C++ class RooStats::LikelihoodIntervalPlot+;

#pragma link C++ function RooStats::NumberCountingUtils::BinomialExpZ(Double_t , Double_t ,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialWithTauExpZ(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialObsZ(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialWithTauObsZ(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialExpP(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialWithTauExpP(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialObsP(Double_t,Double_t,Double_t);
#pragma link C++ function RooStats::NumberCountingUtils::BinomialWithTauObsP(Double_t,Double_t,Double_t);

#pragma link C++ function RooStats::PValueToSignificance(Double_t);
#pragma link C++ function RooStats::SignificanceToPValue(Double_t);
#pragma link C++ function RooStats::RemoveConstantParameters(RooArgSet* set);
#pragma link C++ function RooStats::SetParameters(const RooArgSet* , RooArgSet* );

#endif
