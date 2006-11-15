// @(#)root/tmva $Id: MethodRuleFit.cxx,v 1.40 2006/11/02 15:44:50 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodRuleFit                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch>  - Iowa State U., USA     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      Iowa State U.                                                             *
 *      MPI-KP Heidelberg, Germany,                                               * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//
// J Friedman's RuleFit method
//_______________________________________________________________________

#include "TMVA/MethodRuleFit.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMatrix.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodRuleFit)
   ;
 
//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   //
   InitRuleFit();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   if (HasTrainingTree()) {
      // fill the STL Vector with the event sample
      this->InitEventSample();
   }
   else{
      fLogger << kWARNING << "no training Tree given: you will not be allowed to call ::Train etc." << Endl;
   }

   InitMonitorNtuple();
}

//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( DataSet& theData,
                                    TString theWeightFile,
                                    TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir )
{
   // constructor from weight file
   InitRuleFit();

   DeclareOptions();
}

//_______________________________________________________________________
TMVA::MethodRuleFit::~MethodRuleFit( void )
{
   // destructor
   for (UInt_t i=0; i<fEventSample.size(); i++) delete fEventSample[i];
   for (UInt_t i=0; i<fForest.size(); i++)      delete fForest[i];
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::DeclareOptions() 
{
   DeclareOptionRef(fGDTau=0.0,            "GDTau",          "gradient-directed path: fit threshhold");
   DeclareOptionRef(fGDPathStep=0.01,      "GDStep",         "gradient-directed path: step size");
   DeclareOptionRef(fGDNPathSteps=100,     "GDNSteps",       "gradient-directed path: number of steps");
   DeclareOptionRef(fGDErrNsigma=1.0,      "GDErrNsigma",    "threshold for error-rate");
   DeclareOptionRef(fMinimp=0.01,          "MinImp",         "minimum rule importance accepted");
   DeclareOptionRef(fNodeMinEvents=10,     "nEventsMin",     "minimum number of events in a leaf node");
   DeclareOptionRef(fNTrees=-1,            "nTrees",         "number of trees in forest.");
   DeclareOptionRef(fSampleFraction=-1,    "SampleFraction", "fraction of events used to train each tree");
   DeclareOptionRef(fNCuts=20,             "nCuts",          "number of steps during node cut optimisation");
   DeclareOptionRef(fRuleMaxDist=0.001,    "RuleMaxDist",    "max distance allowed between equal rules");
   //
   DeclareOptionRef(fSepTypeS="GiniIndex", "SeparationType", "separation criterion for node splitting");
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   //
   DeclareOptionRef(fModelTypeS="ModRuleLinear", "Model", "model to be used");
   AddPreDefVal(TString("ModRule"));
   AddPreDefVal(TString("ModRuleLinear"));
   AddPreDefVal(TString("ModLinear"));
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::ProcessOptions() 
{
   MethodBase::ProcessOptions();

   if     (fSepTypeS == "misclassificationerror") fSepType = new TMVA::MisClassificationError();
   else if(fSepTypeS == "giniindex")              fSepType = new TMVA::GiniIndex();
   else if(fSepTypeS == "crossentropy")           fSepType = new TMVA::CrossEntropy();
   else                                           fSepType = new TMVA::SdivSqrtSplusB();

   if      (fModelTypeS == "ModLinear" ) fRuleFit.SetModelLinear();
   else if (fModelTypeS == "ModRule" )   fRuleFit.SetModelRules();
   else                                  fRuleFit.SetModelFull();

   fRuleFit.GetRuleFitParamsPtr()->SetGDTau(fGDTau);
   fRuleFit.GetRuleFitParamsPtr()->SetGDPathStep(fGDPathStep);
   fRuleFit.GetRuleFitParamsPtr()->SetGDNPathSteps(fGDNPathSteps);
   fRuleFit.SetImportanceCut(fMinimp);
   fRuleFit.SetMaxRuleDist(fRuleMaxDist);
   fRuleFit.GetRuleFitParamsPtr()->SetGDErrNsigma(fGDErrNsigma);
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitMonitorNtuple()
{
   fMonitorNtuple= new TTree("MonitorNtuple_RuleFit","RuleFit variables");
   fMonitorNtuple->Branch("importance",&fNTImportance,"importance/D");
   fMonitorNtuple->Branch("support",&fNTSupport,"support/D");
   fMonitorNtuple->Branch("coefficient",&fNTCoefficient,"coefficient/D");
   fMonitorNtuple->Branch("ncuts",&fNTNcuts,"ncuts/I");
   fMonitorNtuple->Branch("type",&fNTType,"type/I");
   fMonitorNtuple->Branch("ptag",&fNTPtag,"ptag/D");
   fMonitorNtuple->Branch("pss",&fNTPss,"pss/D");
   fMonitorNtuple->Branch("psb",&fNTPsb,"psb/D");
   fMonitorNtuple->Branch("pbs",&fNTPbs,"pbs/D");
   fMonitorNtuple->Branch("pbb",&fNTPbb,"pbb/D");
   fMonitorNtuple->Branch("soversb",&fNTSSB,"soversb/D");
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitRuleFit( void )
{
   // default initialisation
   SetMethodName( "RuleFit" );
   SetMethodType( TMVA::Types::RuleFit );
   SetTestvarName();
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitEventSample( void )
{
   // write all Events from the Tree into a vector of TMVA::Events, that are
   // more easily manipulated.
   // This method should never be called without existing trainingTree, as it
   // the vector of events from the ROOT training tree
   if (!HasTrainingTree()) fLogger << kFATAL << "<Init> Data().TrainingTree() is zero pointer" << Endl;

   Int_t nevents = Data().GetNEvtTrain();
   for (Int_t ievt=0; ievt<nevents; ievt++){
      ReadTrainingEvent(ievt);
      //      Float_t weight = GetEventWeight();
      fEventSample.push_back(new TMVA::Event(Data().Event()));
      //       if (fSignalFraction > 0){
      //          if (!(fEventSample.back()->IsSignal())) {
      //             fEventSample.back()->SetWeight(fSignalFraction*fEventSample.back()->GetWeight());
      //          }
      //       }
   }
   if (fSampleFraction<=0) {
      Double_t n = static_cast<Double_t>(nevents);
      fSampleFraction = min( 0.5, (100.0 +6.0*sqrt(n))/n);
   }
   //
   //   std::random_shuffle(fEventSample.begin(), fEventSample.end());
   //
   fLogger << kINFO << "set sample fraction to " << fSampleFraction << Endl;
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::BuildTree( TMVA::DecisionTree *dt, std::vector< TMVA::Event *> & el )
{
   if (dt==0) return;
   dt->BuildTree(el);
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeForest()
{
   // make a forest of decisiontrees
   const Int_t nevents = static_cast<Int_t>(fEventSample.size());
   const Int_t nsubeve = static_cast<Int_t>(nevents*fSampleFraction);

   // Note, any change here, do the same in RuleFit::SetTrainingEvents().
   if (fNTrees<1) fNTrees = static_cast<Int_t>(1.0/fSampleFraction);

   fLogger << kINFO << "creating a forest of " << fNTrees << " decision trees" << Endl;
   fLogger << kINFO << "each tree is built using subsamples of " << nsubeve << " events" << Endl;
   TMVA::Timer timer( fNTrees, GetName() );

   std::vector<TMVA::Event*> eventSubSample;
   std::vector<TMVA::Event*> eventSampleCopy;
   eventSubSample.resize(nsubeve);
   eventSampleCopy.resize(nevents);
   //
   for (Int_t ie=0; ie<nevents; ie++) {
      eventSampleCopy[ie] = fEventSample[ie];
   }
   Double_t fsig;
   Int_t nsig,nbkg;
   for (Int_t i=0; i<fNTrees; i++) {
      //      timer.DrawProgressBar(i);
      std::random_shuffle(eventSampleCopy.begin(), eventSampleCopy.end());
      nsig=0;
      nbkg=0;
      for (Int_t ie = 0; ie<nsubeve; ie++) {
         eventSubSample[ie] = eventSampleCopy[ie];
         if (eventSubSample[ie]->IsSignal()) nsig++;
         else nbkg++;
      }
      fsig = Double_t(nsig)/Double_t(nsig+nbkg);

      fForest.push_back( new DecisionTree( fSepType, fNodeMinEvents, fNCuts ) );
      BuildTree(fForest.back(),eventSubSample);
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::Train( void )
{
   // training of rules

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   // Make forest of decision trees
   if (fRuleFit.GetRuleEnsemble().DoRules()) MakeForest();

   // Init RuleFit object and create rule ensemble
   fRuleFit.Initialise( this, fForest, GetTrainingEvents(), fSampleFraction );

   // Fit the rules
   fLogger << kINFO << "fitting rule coefficients" << Endl;
   fRuleFit.FitCoefficients();

   // Calculate importance
   fLogger << kINFO << "calculating rule and variable importance" << Endl;
   fRuleFit.CalcImportance();

   // Output results and fill monitor ntuple
   fLogger << kINFO << fRuleFit.GetRuleEnsemble();

   UInt_t nrules = fRuleFit.GetRuleEnsemble().GetRulesConst().size();
   const Rule *rule;
   for (UInt_t i=0; i<nrules; i++ ) {
      rule           = fRuleFit.GetRuleEnsemble().GetRulesConst(i);
      fNTImportance   = rule->GetRelImportance();
      fNTSupport      = rule->GetSupport();
      fNTCoefficient  = rule->GetCoefficient();
      fNTType         = (rule->IsSignalRule() ? 1:-1 );
      fNTNcuts        = fRuleFit.GetRuleEnsemble().GetRulesNCuts(i);
      fNTPtag         = fRuleFit.GetRuleEnsemble().GetRulePTag(i); // should be identical with support
      fNTPss          = fRuleFit.GetRuleEnsemble().GetRulePSS(i);
      fNTPsb          = fRuleFit.GetRuleEnsemble().GetRulePSB(i);
      fNTPbs          = fRuleFit.GetRuleEnsemble().GetRulePBS(i);
      fNTPbb          = fRuleFit.GetRuleEnsemble().GetRulePBB(i);
      fNTSSB          = rule->GetSSB();
      fMonitorNtuple->Fill();
   }
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodRuleFit::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new TMVA::Ranking( GetName(), "Variable Importance" );

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new TMVA::Rank( GetInputExp(ivar), fRuleFit.GetRuleEnsemble().GetVarImportance(ivar) ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteWeightsToStream( ostream & o ) const
{  
   fRuleFit.GetRuleEnsemble().PrintRaw(o);
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::ReadWeightsFromStream( istream & istr )
{
   // read rules from stream
   fRuleFit.GetRuleEnsemblePtr()->ReadRaw(istr);
}

//_______________________________________________________________________
Double_t TMVA::MethodRuleFit::GetMvaValue()
{
   // returns MVA value for given event
   return fRuleFit.EvalEvent( Data().Event() );
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteMonitoringHistosToFile( void ) const
{
   // write special monitoring histograms to file - not implemented for RuleFit
   fLogger << kINFO << "write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;

   fMonitorNtuple->Write();
}
