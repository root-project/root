// @(#)root/tmva $Id: MethodRuleFit.cxx,v 1.11 2007/01/30 11:24:16 brun Exp $
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
 *      MPI-K Heidelberg, Germany ,                                               * 
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
#include "TRandom.h"
#include "TMath.h"
#include "TMatrix.h"
#include "TDirectory.h"
#include "Riostream.h"
#include <algorithm>

ClassImp(TMVA::MethodRuleFit)
 
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

   fRuleFit.SetMsgType( fLogger.GetMinType() );

   if (HasTrainingTree()) {
      // fill the STL Vector with the event sample
      this->InitEventSample();
   }
   else {
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
   // define the options (their key words) that can be set in the option string 
   // know options:
   // GDTau          <float>      gradient-directed path: fit threshhold, default
   // GDTauMin       <float>      gradient-directed path: fit threshhold, min
   // GDTauMax       <float>      gradient-directed path: fit threshhold, max
   // GDNTau         <int>        gradient-directed path: fit threshhold, N(tau)
   // GDTauScan      <int>        gradient-directed path: fit threshhold, number of points to scan
   // GDStep         <float>      gradient-directed path: step size       
   // GDNSteps       <float>      gradient-directed path: number of steps 
   // GDErrScale     <float>      stop scan when error>scale*errmin       
   // MinImp         <float>      minimum rule importance accepted        
   // nEventsMin     <float>      minimum number of events in a leaf node 
   // nTrees         <float>      number of trees in forest.              
   // SampleFraction <float>      fraction of events used to train each tree
   // nCuts          <float>      number of steps during node cut optimisation
   // LinQuantile    <float>      quantile of linear terms (remove outliers)
   // RuleMinDist    <float>      min distance allowed between rules
   // 
   // SeparationType <string>     separation criterion for node splitting
   //    available values are:    GiniIndex <default>
   //                             MisClassificationError
   //                             CrossEntropy
   //                             SDivSqrtSPlusB
   // 
   // Model          <string>     model to be used
   //    available values are:    ModRuleLinear <default>
   //                             ModRule
   //                             ModLinear

   DeclareOptionRef(fGDTau=0.6,            "GDTau",          "gradient-directed path: default fit threshold");
   DeclareOptionRef(fGDTauMin=0.0,         "GDTauMin",       "gradient-directed path: min fit threshold (tau)");
   DeclareOptionRef(fGDTauMax=1.0,         "GDTauMax",       "gradient-directed path: max fit threshold (tau)");
   DeclareOptionRef(fGDNTau=1,             "GDNTau",         "gradient-directed path: N(tau)");
   DeclareOptionRef(fGDTauScan=200,        "GDTauScan",      "gradient-directed path: number of points scanning for best tau");
   DeclareOptionRef(fGDPathStep=0.01,      "GDStep",         "gradient-directed path: step size");
   DeclareOptionRef(fGDNPathSteps=10000,   "GDNSteps",       "gradient-directed path: number of steps");
   DeclareOptionRef(fGDErrScale=1.1,       "GDErrScale",     "stop scan when error>scale*errmin");
   // tree options
   DeclareOptionRef(fNodeMinEvents=-1,     "nEventsMin",     "OBSOLETE:minimum number of events in a leaf node");
   DeclareOptionRef(fMinFracNEve=0.1,      "fEventsMin",     "minimum fraction of events in a splittable node");
   DeclareOptionRef(fMaxFracNEve=0.4,      "fEventsMax",     "maximum fraction of events in a splittable node");
   DeclareOptionRef(fNTrees=-1,            "nTrees",         "number of trees in forest.");
   DeclareOptionRef(fSampleFraction=-1,    "SampleFraction", "fraction of events used to train each tree");
   DeclareOptionRef(fSubSampleFraction=0.4,"SubFraction",    "fraction of subsamples used for the fitting");
   DeclareOptionRef(fNCuts=20,             "nCuts",          "number of steps during node cut optimisation");
   DeclareOptionRef(fSepTypeS="GiniIndex", "SeparationType", "separation criterion for node splitting");
   AddPreDefVal(TString("MisClassificationError"));
   AddPreDefVal(TString("GiniIndex"));
   AddPreDefVal(TString("CrossEntropy"));
   AddPreDefVal(TString("SDivSqrtSPlusB"));
   // --- DO NOT INCLUDE IN RELEASE YET ---
   fPruneMethodS = "none";
//    DeclareOptionRef(fPruneMethodS="CostComplexity", "PruneMethod", "Pruning method: Expected Error or Cost Complexity");
//    AddPreDefVal(TString("ExpectedError"));
//    AddPreDefVal(TString("CostComplexity"));
//    AddPreDefVal(TString("CostComplexity2"));
//    AddPreDefVal(TString("NONE"));

//    DeclareOptionRef(fPruneStrength=3.5, "PruneStrength", "a parameter to adjust the amount of pruning. Should be large enouth such that overtraining is avoided, or negative == automatic (takes time)");

   DeclareOptionRef(fLinQuantile=0.025,    "LinQuantile",    "quantile of linear terms (remove outliers)");

   // rule cleanup options
   DeclareOptionRef(fRuleMinDist=0.001,    "RuleMinDist",    "min distance between rules");
   DeclareOptionRef(fMinimp=0.01,          "MinImp",         "minimum rule importance accepted");

   // rule model option
   DeclareOptionRef(fModelTypeS="ModRuleLinear", "Model", "model to be used");
   AddPreDefVal(TString("ModRule"));
   AddPreDefVal(TString("ModRuleLinear"));
   AddPreDefVal(TString("ModLinear"));
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::ProcessOptions() 
{
   // process the options specified by the user   
   MethodBase::ProcessOptions();

   fSepTypeS.ToLower();
   if     (fSepTypeS == "misclassificationerror") fSepType = new TMVA::MisClassificationError();
   else if(fSepTypeS == "giniindex")              fSepType = new TMVA::GiniIndex();
   else if(fSepTypeS == "crossentropy")           fSepType = new TMVA::CrossEntropy();
   else                                           fSepType = new TMVA::SdivSqrtSplusB();

   fModelTypeS.ToLower();
   if      (fModelTypeS == "modlinear" ) fRuleFit.SetModelLinear();
   else if (fModelTypeS == "modrule" )   fRuleFit.SetModelRules();
   else                                  fRuleFit.SetModelFull();

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror" )   fPruneMethod  = TMVA::DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity" )  fPruneMethod  = TMVA::DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "costcomplexity2" ) fPruneMethod  = TMVA::DecisionTree::kMCC;
   else                                          fPruneMethodS = "none";

   fRuleFit.GetRuleEnsemblePtr()->SetLinQuantile(fLinQuantile);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTau(fGDTauMin,fGDTauMax);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTau(fGDTau);
   fRuleFit.GetRuleFitParamsPtr()->SetGDNTau(fGDNTau);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTauScan(fGDTauScan);
   fRuleFit.GetRuleFitParamsPtr()->SetGDPathStep(fGDPathStep);
   fRuleFit.GetRuleFitParamsPtr()->SetGDNPathSteps(fGDNPathSteps);
   fRuleFit.SetImportanceCut(fMinimp);
   fRuleFit.SetRuleMinDist(fRuleMinDist);
   fRuleFit.GetRuleFitParamsPtr()->SetGDErrScale(fGDErrScale);

   // range check
   Bool_t minmod=kFALSE;
   Bool_t maxmod=kFALSE;
   if (fMinFracNEve<0.0) {
      fMinFracNEve=0.0;
      minmod=kTRUE;
   }
   if (fMaxFracNEve>1.0) {
      fMaxFracNEve=1.0;
      maxmod=kTRUE;
   }
   if (fMaxFracNEve<fMinFracNEve) {
      fMaxFracNEve=fMinFracNEve;
      maxmod=kTRUE;
   }
   if (minmod) fLogger << kWARNING << "illegal fEventsMin - set to new value = " << fMinFracNEve << Endl;
   if (maxmod) fLogger << kWARNING << "illegal fEventsMax - set to new value = " << fMaxFracNEve << Endl;
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitMonitorNtuple()
{
   // initialize the monitoring ntuple
   BaseDir()->cd();
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
   SetMethodType( TMVA::Types::kRuleFit );
   SetTestvarName();

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );
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
      fEventSample.push_back(new TMVA::Event(GetEvent()));
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
   std::random_shuffle(fEventSample.begin(), fEventSample.end());
   //
   fLogger << kVERBOSE << "set sub-sample fraction to " << fSampleFraction << Endl;
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::BuildTree( TMVA::DecisionTree *dt, std::vector< TMVA::Event *> & el )
{
   // build the decision tree
   if (dt==0) return;
   dt->BuildTree(el);
   if (fPruneMethodS!="none") {
      dt->SetPruneMethod(fPruneMethod);
      dt->SetPruneStrength(fPruneStrength);
      dt->PruneTree();
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeForest()
{
   // make a forest of decisiontrees
   const Int_t nevents = static_cast<Int_t>(fEventSample.size());
   const Int_t nsubeve = static_cast<Int_t>(nevents*fSampleFraction);
   const Int_t ntreesf = static_cast<Int_t>(0.5+(1.0/fSampleFraction));
   // flag, if true => all trees are generated with unique samples
   // it is true when fNTrees<1 or fNTrees== [1.0/fSampleFraction]
   Bool_t doUniqueSamples;


   // Note, any change here, do the same in RuleFit::SetTrainingEvents().
   if (fNTrees<1) fNTrees = ntreesf;
   
   doUniqueSamples = (fNTrees <= ntreesf);

   fLogger << kINFO << "creating a forest of " << fNTrees << " decision trees" << Endl;
   fLogger << kINFO << "each tree is built using subsamples of " << nsubeve << " events" << Endl;
   fLogger << kINFO << "training samples are " << (doUniqueSamples ? "NOT ":"") << "overlapping" << Endl;
   //
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
   Int_t itree=0;
   Int_t itofs;
   //
   TRandom rndGen;
   //
   Int_t nminRnd;
   //
   for (Int_t i=0; i<fNTrees; i++) {
      //      timer.DrawProgressBar(i);
      if (!doUniqueSamples) {
         std::random_shuffle(eventSampleCopy.begin(), eventSampleCopy.end());
         itree=0;
      } else {
         itree=i;
      }
      nsig=0;
      nbkg=0;
      itofs = itree*nsubeve;
      for (Int_t ie = itofs; ie<itofs+nsubeve; ie++) {
         eventSubSample[ie-itofs] = eventSampleCopy[ie];
         if (eventSubSample[ie-itofs]->IsSignal()) nsig++;
         else nbkg++;
      }
      fsig = Double_t(nsig)/Double_t(nsig+nbkg);
      if ((nbkg==0) || (nsig==0)) {
         fLogger << kFATAL << "BUG TRAP: only signal or bkg (not both) in sample for making forest, nsig,nbkg = "
                 << nsig << " , " << nbkg << Endl;
      }
      if ((fsig>0.7) || (fsig<0.3)) {
         fLogger << kFATAL << "BUG TRAP: number of signal or bkg not roughly equal in sample for making forest, nsig,nbkg = "
                 << nsig << " , " << nbkg << Endl;
      }
      TMVA::SeparationBase *qualitySepType = new TMVA::GiniIndex();
      // generate random number of events
      // do not implement the above in this release...just set it to default
      //      nminRnd = fNodeMinEvents;
      DecisionTree *dt;
      Bool_t tryAgain=kTRUE;
      Int_t ntries=0;
      const Int_t ntriesMax=10;
      while (tryAgain) {
         Double_t frnd = rndGen.Uniform( fMinFracNEve, fMaxFracNEve );
         nminRnd = Int_t(frnd*Double_t(nsubeve));
         dt = new DecisionTree( fSepType, nminRnd, fNCuts, qualitySepType );
         BuildTree(dt,eventSubSample);
         if (dt->GetNNodes()<3) {
            delete dt;
            dt=0;
         }
         ntries++;
         tryAgain = ((dt==0) && (ntries<ntriesMax));
      }
      if (dt) {
         fForest.push_back(dt);
      } else {
         fLogger << kWARNING << "------------------------------------------------------------------" << Endl;
         fLogger << kWARNING << " failed growing a tree even after " << ntriesMax << " trials" << Endl;
         fLogger << kWARNING << " possible solutions: " << Endl;
         fLogger << kWARNING << "   1. increase the number of training events" << Endl;
         fLogger << kWARNING << "   2. set a lower min fraction cut (fEventsMin)" << Endl;
         fLogger << kWARNING << "   3. maybe also decrease the max fraction cut (fEventsMax)" << Endl;
         fLogger << kWARNING << " if the above warning occurs rarely, it can be ignored" << Endl;
         fLogger << kWARNING << "------------------------------------------------------------------" << Endl;
      }

      fLogger << kDEBUG << "built tree with minimum cut at N = " << nminRnd
              << " => N(nodes) = " << fForest.back()->GetNNodes()
              << " ; n(tries) = " << ntries
              << Endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeForestRnd()
{
   // test forest generation - NOTE: this is not used in the normal release
   const Int_t nevents = static_cast<Int_t>(fEventSample.size());
   const Int_t nsubeve = static_cast<Int_t>(nevents*fSampleFraction);
   const Int_t ntreesf = static_cast<Int_t>(0.5+(1.0/fSampleFraction));
   // flag, if true => all trees are generated with unique samples
   // it is true when fNTrees<1 or fNTrees== [1.0/fSampleFraction]
   Bool_t doUniqueSamples;


   // Note, any change here, do the same in RuleFit::SetTrainingEvents().
   if (fNTrees<1) fNTrees = ntreesf;
   
   doUniqueSamples = (fNTrees <= ntreesf);

   fLogger << kINFO << "creating a forest of " << fNTrees << " decision trees" << Endl;
   fLogger << kINFO << "each tree is built using subsamples of " << nsubeve << " events" << Endl;
   fLogger << kINFO << "training samples are " << (doUniqueSamples ? "NOT ":"") << "overlapping" << Endl;
   //
   TMVA::Timer timer( fNTrees, GetName() );

   std::vector<TMVA::Event*> eventSubSample;
   std::vector<TMVA::Event*> eventSampleCopy;
   eventSubSample.resize(nsubeve);
   eventSampleCopy.resize(nevents);
   //
   for (Int_t ie=0; ie<nevents; ie++) {
      eventSampleCopy[ie] = fEventSample[ie];
   }
   Int_t nsig,nbkg;
   Int_t itree=0;
   Int_t itofs;
   //
   TRandom rndGen;
   const Int_t nmin  = 20;
   const Int_t nmax  = nsubeve-1;
   const Int_t ntst  = nmax-nmin+1;
   Int_t    nval;
   Double_t nfrac;
   Double_t nnsum;
   Double_t nnsum2;
   //
   for (Int_t n=0; n<ntst; n++) {
      nval = nmin+n;
      nnsum=0;
      nnsum2=0;
      //      nfrac = rndGen.Uniform(0.01,0.9); //
      nfrac = Double_t(nval)/Double_t(nsubeve);
      nval = Int_t(nfrac*nsubeve);
      for (Int_t i=0; i<fNTrees; i++) {
         //      timer.DrawProgressBar(i);
         if (!doUniqueSamples) {
            std::random_shuffle(eventSampleCopy.begin(), eventSampleCopy.end());
            itree=0;
         } else {
            itree=i;
         }
         nsig=0;
         nbkg=0;
         itofs = itree*nsubeve;
         for (Int_t ie = itofs; ie<itofs+nsubeve; ie++) {
            eventSubSample[ie-itofs] = eventSampleCopy[ie];
            if (eventSubSample[ie-itofs]->IsSignal()) nsig++;
            else nbkg++;
         }
         TMVA::SeparationBase *qualitySepType = new TMVA::GiniIndex();
         DecisionTree *dt = new DecisionTree( fSepType, nval, fNCuts, qualitySepType );
         BuildTree(dt,eventSubSample);
         Int_t nr = fRuleFit.GetRuleEnsemblePtr()->CalcNRules(dt);
         Int_t nn = (nr/2) + 1;
         nnsum  += nn;
         nnsum2 += nn*nn;
         delete dt;
      }
      Double_t nnave = nnsum/fNTrees;
      Double_t nnsig = TMath::Sqrt( (nnsum2 - (nnsum*nnsum/Double_t(fNTrees)))/Double_t(fNTrees-1) );
      std::cout << "NNDIST: " << nfrac << " " << nnave << " " << nnsig << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::Train( void )
{
   // training of rules

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   // timer
   TMVA::Timer timer( 1, GetName() );

   // test tree nmin cut -> for debug purposes
   // the routine will generate trees with stopping cut on N(eve) given by
   // a fraction between [20,N(eve)-1].
   // 
   //   MakeForestRnd();
   //   exit(1);
   //
   // Make forest of decision trees
   if (fRuleFit.GetRuleEnsemble().DoRules()) MakeForest();

   // Init RuleFit object and create rule ensemble
   fRuleFit.Initialise( this, fForest, GetTrainingEvents(), fSampleFraction );

   // Fit the rules
   fLogger << kVERBOSE << "fitting rule coefficients" << Endl;
   fRuleFit.FitCoefficients();

   // print timing info
   fLogger << kINFO << "train elapsed time: " << timer.GetElapsedTime() << Endl;

   // Calculate importance
   fLogger << kVERBOSE << "calculating rule and variable importance" << Endl;
   fRuleFit.CalcImportance();

   // Output results and fill monitor ntuple
   fRuleFit.GetRuleEnsemblePtr()->Print();
   //
   fLogger << kVERBOSE << "filling rule ntuple" << Endl;
   UInt_t nrules = fRuleFit.GetRuleEnsemble().GetRulesConst().size();
   const Rule *rule;
   for (UInt_t i=0; i<nrules; i++ ) {
      rule            = fRuleFit.GetRuleEnsemble().GetRulesConst(i);
      fNTImportance   = rule->GetRelImportance();
      fNTSupport      = rule->GetSupport();
      fNTCoefficient  = rule->GetCoefficient();
      fNTType         = (rule->IsSignalRule() ? 1:-1 );
      fNTNcuts        = rule->GetRuleCut()->GetNcuts();
      fNTPtag         = fRuleFit.GetRuleEnsemble().GetRulePTag(i); // should be identical with support
      fNTPss          = fRuleFit.GetRuleEnsemble().GetRulePSS(i);
      fNTPsb          = fRuleFit.GetRuleEnsemble().GetRulePSB(i);
      fNTPbs          = fRuleFit.GetRuleEnsemble().GetRulePBS(i);
      fNTPbb          = fRuleFit.GetRuleEnsemble().GetRulePBB(i);
      fNTSSB          = rule->GetSSB();
      fMonitorNtuple->Fill();
   }
   fLogger << kVERBOSE << "done training" << Endl;

   fRuleFit.MakeVisHists();
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodRuleFit::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new TMVA::Ranking( GetName(), "Importance" );

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new TMVA::Rank( GetInputExp(ivar), fRuleFit.GetRuleEnsemble().GetVarImportance(ivar) ) );
   }

   return fRanking;
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteWeightsToStream( ostream & o ) const
{  
   // write the rules to an ostream
   fRuleFit.GetRuleEnsemble().PrintRaw(o);
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::ReadWeightsFromStream( istream & istr )
{
   // read rules from an istream

   fRuleFit.GetRuleEnsemblePtr()->ReadRaw(istr);
}

//_______________________________________________________________________
Double_t TMVA::MethodRuleFit::GetMvaValue()
{
   // returns MVA value for given event
   return fRuleFit.EvalEvent( GetEvent() );
}

//_______________________________________________________________________
void  TMVA::MethodRuleFit::WriteMonitoringHistosToFile( void ) const
{
   // write special monitoring histograms to file (here ntuple)
   fLogger << kINFO << "write monitoring ntuple to file: " << BaseDir()->GetPath() << Endl;

   fMonitorNtuple->Write();
}
