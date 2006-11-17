// @(#)root/tmva $Id: RuleFit.cxx,v 1.28 2006/11/16 22:51:59 helgevoss Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Rule                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class describung a 'rule'                                               *
 *      Each internal node of a tree defines a rule from all the parental nodes.  *
 *      A rule with 0 or 1 nodes in the list is a root rule -> corresponds to a0. *
 *      Input: a decision tree (in the constructor)                               *
 *             its coefficient                                                    *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <algorithm>

#include "TMVA/RuleFit.h"
#include "TMVA/MethodRuleFit.h"

ClassImp(TMVA::RuleFit)
   ;

//_______________________________________________________________________
TMVA::RuleFit::RuleFit( const TMVA::MethodRuleFit *rfbase,
                        const std::vector< TMVA::DecisionTree *> & forest,
                        const std::vector<TMVA::Event *> & trainingEvents,
                        Double_t samplefrac )
   : fLogger( "RuleFit" )
{
   // constructor
   Initialise( rfbase, forest, trainingEvents, samplefrac );
}

//_______________________________________________________________________
TMVA::RuleFit::RuleFit()
   : fLogger( "RuleFit" )
{
   // default constructor
}

//_______________________________________________________________________
TMVA::RuleFit::~RuleFit()
{
   // destructor
}

//_______________________________________________________________________
void TMVA::RuleFit::Initialise(  const TMVA::MethodRuleFit *rfbase,
                                 const std::vector< TMVA::DecisionTree *> & forest,
                                 const std::vector< TMVA::Event *> & events,
                                 Double_t sampfrac )
{
   // initialize the parameters of the RuleFit method
   fMethodRuleFit = rfbase;
   std::vector< TMVA::DecisionTree *>::const_iterator itrDtree=forest.begin();
   for (; itrDtree!=forest.end(); ++itrDtree ) fForest.push_back( *itrDtree );

   ForestStatistics();

   SetTrainingEvents( events, sampfrac );

   // Initialize RuleEnsemble
   fRuleEnsemble.Initialize( this );

   // Make the model - Rule + Linear (if fDoLinear is true)
   fLogger << kINFO << "make model" << Endl;
   fRuleEnsemble.MakeModel();

   // Initialize RuleFitParams
   fRuleFitParams.SetRuleFit( this );
}

//_______________________________________________________________________
void TMVA::RuleFit::Copy( const TMVA::RuleFit& other )
{
   // copy method
   if(this != &other) {
      fMethodRuleFit   = other.GetMethodRuleFit();
      fTrainingEvents  = other.GetTrainingEvents();
      fSubsampleEvents = other.GetSubsampleEvents();
   
      fForest       = other.GetForest();
      fRuleEnsemble = other.GetRuleEnsemble();
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::ForestStatistics()
{
   // summary of statistics of all trees
   // * end-nodes: average and spread
   
   UInt_t ntrees = fForest.size();
   Double_t nt   = Double_t(ntrees);
   const TMVA::DecisionTree *tree;
   Double_t sumn2 = 0;
   Double_t sumn  = 0;
   Double_t nd;
   for (UInt_t i=0; i<ntrees; i++) {
      tree = fForest[i];
      nd = Double_t(tree->GetNNodes());
      sumn  += nd;
      sumn2 += nd*nd;
   }
   Double_t var = (sumn2 - (sumn*sumn/nt))/(nt-1);
   fLogger << kINFO << "nodes in trees, average& variance = " << sumn/nt << " , " << var << Endl;
}

//_______________________________________________________________________
void TMVA::RuleFit::FitCoefficients()
{
   //
   // Fit the coefficients for the rule ensemble
   //
   //    fRuleFitParams.SetGDNPathSteps( 100 );
   //    fRuleFitParams.SetGDPathStep( 0.01 );
   //    fRuleFitParams.SetGDTau( 0.0 );
   fRuleFitParams.MakeGDPath();
}

//_______________________________________________________________________
void TMVA::RuleFit::CalcImportance()
{
   // calculates the importance of each rule

   fLogger << kINFO << "calculating importance" << Endl;
   fRuleEnsemble.CalcImportance();
   fRuleEnsemble.CleanupRules();
   fRuleEnsemble.CleanupLinear();
   fRuleEnsemble.CalcVarImportance();
   fLogger << kINFO << "filling rule statistics" << Endl;
   fRuleEnsemble.RuleStatistics();
}

//_______________________________________________________________________
Double_t TMVA::RuleFit::EvalEvent( const TMVA::Event& e )
{
   // evaluate single event

   return fRuleEnsemble.EvalEvent( e );
}

//_______________________________________________________________________
void TMVA::RuleFit::SetTrainingEvents( const std::vector<TMVA::Event *>& el, Double_t sampfrac )
{
   // set the training events randomly

   UInt_t neve = el.size();
   if (neve==0) fLogger << kWARNING << "an empty sample of training events was given" << Endl;

   // copy vector
   fTrainingEvents.clear();
   for (UInt_t i=0; i<el.size(); i++) {
      fTrainingEvents.push_back(static_cast< const TMVA::Event *>(el[i]));
   }

   // Re-shuffle the vector, ie, recreate it in a random order
   std::random_shuffle( fTrainingEvents.begin(), fTrainingEvents.end() );

   // Divide into subsamples
   Int_t istep = static_cast<Int_t>(neve*sampfrac);
   // Any change to nsub -> see also MethodRuleFit::MakeForest()
   Int_t nsub  = static_cast<Int_t>(1.0/sampfrac);

   for (Int_t s=0; s<nsub; s++) {
      fSubsampleEvents.push_back( s*istep );
   }
   fSubsampleEvents.push_back( neve ); // last index is the total length
   fLogger << kINFO << "created " << nsub << " training samples of in total "
           << fTrainingEvents.size() << " events" << Endl;
}

//_______________________________________________________________________
void TMVA::RuleFit::GetSubsampleEvents(Int_t sub, UInt_t& ibeg, UInt_t& iend) const
{
   // get the events for subsample sub

   Int_t nsub = GetNSubsamples();
   if (nsub==0) {
      fLogger << kFATAL << "<GetSubsampleEvents> - wrong size, not properly initialised! BUG!!!" << Endl;
   }
   ibeg = 0;
   iend = 0;
   if (sub<0) {
      ibeg = 0;
      iend = fTrainingEvents.size() - 1;
   }
   if (sub<nsub) {
      ibeg = fSubsampleEvents[sub];
      iend = fSubsampleEvents[sub+1];
   }
}
