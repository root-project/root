// @(#)root/tmva $Id: MethodRuleFit.cxx,v 1.12 2007/04/19 06:53:02 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRuleFit                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header file for description)                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch>  - Iowa State U., USA     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 * 
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
#include "TMVA/RuleFitAPI.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TRandom.h"
#include "TMath.h"
#include "TMatrix.h"
#include "TDirectory.h"
#include "Riostream.h"
#include <algorithm>
#include <list>


ClassImp(TMVA::MethodRuleFit)
 
//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( TString jobName, TString methodTitle, DataSet& theData, 
                                    TString theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   //

   InitRuleFit();

   // interpretation of configuration option string
   DeclareOptions();
   ParseOptions();
   ProcessOptions();

   // Select what weight to use in the 'importance' rule visualisation plots.
   // Note that if UseCoefficientsVisHists() is selected, the following weight is used:
   //    w = rule coefficient * rule support
   // The support is a positive number which is 0 if no events are accepted by the rule.
   // Normally the importance gives more useful information.
   //
   //fRuleFit.UseCoefficientsVisHists();
   fRuleFit.UseImportanceVisHists();

   fRuleFit.SetMsgType( fLogger.GetMinType() );

   if (HasTrainingTree()) {
      // fill the STL Vector with the event sample
      this->InitEventSample();
   }
   else {
      fLogger << kWARNING << "No training Tree given: you will not be allowed to call ::Train etc." << Endl;
   }

   InitMonitorNtuple();
}

//_______________________________________________________________________
TMVA::MethodRuleFit::MethodRuleFit( DataSet& theData,
                                    TString theWeightFile,
                                    TDirectory* theTargetDir )
   : MethodBase( theData, theWeightFile, theTargetDir )
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
   // know options.
   //---------
   // general
   //---------
   // RuleFitModule  <string>     
   //    available values are:    RFTMVA      - use TMVA implementation
   //                             RFFriedman  - use Friedmans original implementation
   //----------------------
   // Path search (fitting)
   //----------------------
   // GDTau          <float>      gradient-directed path: fit threshhold, default
   // GDTauPrec      <float>      gradient-directed path: precision of estimated tau
   // GDStep         <float>      gradient-directed path: step size       
   // GDNSteps       <float>      gradient-directed path: number of steps 
   // GDErrScale     <float>      stop scan when error>scale*errmin       
   //-----------------
   // Tree generation
   //-----------------
   // fEventsMin     <float>      minimum fraction of events in a splittable node
   // fEventsMax     <float>      maximum fraction of events in a splittable node
   // nTrees         <float>      number of trees in forest.
   // ForestType     <string>
   //    available values are:    Random    - create forest using random subsample
   //                             AdaBoost  - create forest with boosted events
   //
   //-----------------
   // Model creation
   //-----------------
   // RuleMinDist    <float>      min distance allowed between rules
   // MinImp         <float>      minimum rule importance accepted        
   // Model          <string>     model to be used
   //    available values are:    ModRuleLinear <default>
   //                             ModRule
   //                             ModLinear
   //
   //-----------------
   // Friedmans module
   //-----------------
   // RFWorkDir      <string>     directory where Friedmans module (rf_go.exe) is installed
   // RFNrules       <int>        maximum number of rules allowed
   // RFNendnodes    <int>        average number of end nodes in the forest of trees
   //
   DeclareOptionRef(fGDTau=-1,             "GDTau",          "Gradient-directed path: default fit cut-off");
   DeclareOptionRef(fGDTauPrec=0.01,       "GDTauPrec",      "Gradient-directed path: precision of tau");
   DeclareOptionRef(fGDPathStep=0.01,      "GDStep",         "Gradient-directed path: step size");
   DeclareOptionRef(fGDNPathSteps=10000,   "GDNSteps",       "Gradient-directed path: number of steps");
   DeclareOptionRef(fGDErrScale=1.1,       "GDErrScale",     "Stop scan when error>scale*errmin");
   DeclareOptionRef(fGDPathEveFrac=0.5,    "GDPathEveFrac",  "Fraction of events used for the path search");
   DeclareOptionRef(fGDValidEveFrac=0.5,   "GDValidEveFrac", "Fraction of events used for the validation");
   // tree options
   DeclareOptionRef(fMinFracNEve=0.1,      "fEventsMin",     "Minimum fraction of events in a splittable node");
   DeclareOptionRef(fMaxFracNEve=0.9,      "fEventsMax",     "Maximum fraction of events in a splittable node");
   DeclareOptionRef(fNTrees=20,            "nTrees",         "Number of trees in forest.");

   DeclareOptionRef(fForestTypeS="AdaBoost",  "ForestType",   "Method to use for forest generation");
   AddPreDefVal(TString("AdaBoost"));
   AddPreDefVal(TString("Random"));
   // rule cleanup options
   DeclareOptionRef(fRuleMinDist=0.001,    "RuleMinDist",    "Minimum distance between rules");
   DeclareOptionRef(fMinimp=0.01,          "MinImp",         "Minimum rule importance accepted");
   // rule model option
   DeclareOptionRef(fModelTypeS="ModRuleLinear", "Model",    "Model to be used");
   AddPreDefVal(TString("ModRule"));
   AddPreDefVal(TString("ModRuleLinear"));
   AddPreDefVal(TString("ModLinear"));

   DeclareOptionRef(fRuleFitModuleS="RFTMVA",  "RuleFitModule","Which RuleFit module to use");
   AddPreDefVal(TString("RFTMVA"));
   AddPreDefVal(TString("RFFriedman"));

   DeclareOptionRef(fRFWorkDir="./rulefit", "RFWorkDir",    "Friedmans RuleFit module: working dir");
   DeclareOptionRef(fRFNrules=2000,         "RFNrules",     "Friedmans RuleFit module: maximum number of rules");
   DeclareOptionRef(fRFNendnodes=4,         "RFNendnodes",  "Friedmans RuleFit module: average number of end nodes");
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::ProcessOptions() 
{
   // process the options specified by the user   
   MethodBase::ProcessOptions();

   fRuleFitModuleS.ToLower();
   if      (fRuleFitModuleS == "rftmva")     fUseRuleFitJF = kFALSE;
   else if (fRuleFitModuleS == "rffriedman") fUseRuleFitJF = kTRUE;
   else                                      fUseRuleFitJF = kTRUE;

   fSepTypeS.ToLower();
   if      (fSepTypeS == "misclassificationerror") fSepType = new MisClassificationError();
   else if (fSepTypeS == "giniindex")              fSepType = new GiniIndex();
   else if (fSepTypeS == "crossentropy")           fSepType = new CrossEntropy();
   else                                            fSepType = new SdivSqrtSplusB();

   fModelTypeS.ToLower();
   if      (fModelTypeS == "modlinear" ) fRuleFit.SetModelLinear();
   else if (fModelTypeS == "modrule" )   fRuleFit.SetModelRules();
   else                                  fRuleFit.SetModelFull();

   fPruneMethodS.ToLower();
   if      (fPruneMethodS == "expectederror" )   fPruneMethod  = DecisionTree::kExpectedErrorPruning;
   else if (fPruneMethodS == "costcomplexity" )  fPruneMethod  = DecisionTree::kCostComplexityPruning;
   else if (fPruneMethodS == "costcomplexity2" ) fPruneMethod  = DecisionTree::kMCC;
   else                                          fPruneMethod  = DecisionTree::kNoPruning;

   fForestTypeS.ToLower();
   if      (fForestTypeS == "random" )   fUseBoost = kFALSE;
   else if (fForestTypeS == "adaboost" ) fUseBoost = kTRUE;
   else                                  fUseBoost = kTRUE;
   //
   // if creating the forest by boosting the events
   // the full training sample is used per tree
   // -> only true for the TMVA version of RuleFit.
   if (fUseBoost && (!fUseRuleFitJF)) fTreeEveFrac = 1.0;

   // check event fraction for tree generation
   // if <0 set to automatic number
   if (fTreeEveFrac<=0) {
      Int_t nevents = Data().GetNEvtTrain();
      Double_t n = static_cast<Double_t>(nevents);
      fTreeEveFrac = min( 0.5, (100.0 +6.0*sqrt(n))/n);
   }
   // verify ranges of options
   Tools::VerifyRange(fLogger, "nTrees",        fNTrees,0,100000,20);
   Tools::VerifyRange(fLogger, "MinImp",        fMinimp,0.0,1.0,0.0);
   Tools::VerifyRange(fLogger, "GDTauPrec",     fGDTauPrec,1e-5,5e-1);
   Tools::VerifyRange(fLogger, "GDTauMin",      fGDTauMin,0.0,1.0);
   Tools::VerifyRange(fLogger, "GDTauMax",      fGDTauMax,fGDTauMin,1.0);
   Tools::VerifyRange(fLogger, "GDPathStep",    fGDPathStep,0.0,100.0,0.01);
   Tools::VerifyRange(fLogger, "GDErrScale",    fGDErrScale,1.0,100.0,1.1);
   Tools::VerifyRange(fLogger, "GDPathEveFrac", fGDPathEveFrac,0.01,0.9,0.5);
   Tools::VerifyRange(fLogger, "GDValidEveFrac",fGDValidEveFrac,0.01,1.0-fGDPathEveFrac,1.0-fGDPathEveFrac);
   Tools::VerifyRange(fLogger, "fEventsMin",    fMinFracNEve,0.0,1.0);
   Tools::VerifyRange(fLogger, "fEventsMax",    fMaxFracNEve,fMinFracNEve,1.0);

   fRuleFit.GetRuleEnsemblePtr()->SetLinQuantile(fLinQuantile);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTauRange(fGDTauMin,fGDTauMax);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTau(fGDTau);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTauPrec(fGDTauPrec);
   fRuleFit.GetRuleFitParamsPtr()->SetGDTauScan(fGDTauScan);
   fRuleFit.GetRuleFitParamsPtr()->SetGDPathStep(fGDPathStep);
   fRuleFit.GetRuleFitParamsPtr()->SetGDNPathSteps(fGDNPathSteps);
   fRuleFit.GetRuleFitParamsPtr()->SetGDErrScale(fGDErrScale);
   fRuleFit.SetImportanceCut(fMinimp);
   fRuleFit.SetRuleMinDist(fRuleMinDist);


   // check if Friedmans module is used.
   // print a message concerning the options.
   if (fUseRuleFitJF) {
      fLogger << kINFO << "" << Endl;
      fLogger << kINFO << "--------------------------------------" <<Endl;
      fLogger << kINFO << "Friedmans RuleFit module is selected." << Endl;
      fLogger << kINFO << "Only the following options are used:" << Endl;
      fLogger << kINFO <<  Endl;
      fLogger << kINFO << Tools::Color("bold") << "   Model"        << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   RFWorkDir"    << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   RFNrules"     << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   RFNendnodes"  << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   GDNPathSteps" << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   GDPathStep"   << Tools::Color("reset") << Endl;
      fLogger << kINFO << Tools::Color("bold") << "   GDErrScale"   << Tools::Color("reset") << Endl;
      fLogger << kINFO << "--------------------------------------" <<Endl;
      fLogger << kINFO << Endl;
   }
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
   fMonitorNtuple->Branch("nvars",&fNTNvars,"nvars/I");
   fMonitorNtuple->Branch("type",&fNTType,"type/I");
   fMonitorNtuple->Branch("ptag",&fNTPtag,"ptag/D");
   fMonitorNtuple->Branch("pss",&fNTPss,"pss/D");
   fMonitorNtuple->Branch("psb",&fNTPsb,"psb/D");
   fMonitorNtuple->Branch("pbs",&fNTPbs,"pbs/D");
   fMonitorNtuple->Branch("pbb",&fNTPbb,"pbb/D");
   fMonitorNtuple->Branch("soversb",&fNTSSB,"soversb/D");
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitRuleFit()
{
   // default initialization
   SetMethodName( "RuleFit" );
   SetMethodType( TMVA::Types::kRuleFit );
   SetTestvarName();

   // the minimum requirement to declare an event signal-like
   SetSignalReferenceCut( 0.0 );

   // set variables that used to be options
   // any modifications are then made in ProcessOptions()
   fLinQuantile   = 0.025;       // Quantile of linear terms (remove outliers)
   fTreeEveFrac   = -1.0;        // Fraction of events used to train each tree
   fNCuts         = 20;          // Number of steps during node cut optimisation
   fSepTypeS      = "GiniIndex"; // Separation criterion for node splitting; see BDT
   fPruneMethodS  = "NONE";      // Pruning method; see BDT
   fPruneStrength = 3.5;         // Pruning strength; see BDT
   fGDTauMin      = 0.0;         // Gradient-directed path: min fit threshold (tau)
   fGDTauMax      = 1.0;         // Gradient-directed path: max fit threshold (tau)
   fGDTauScan     = 1000;        // Gradient-directed path: number of points scanning for best tau

}

//_______________________________________________________________________
void TMVA::MethodRuleFit::InitEventSample( void )
{
   // write all Events from the Tree into a vector of Events, that are
   // more easily manipulated.
   // This method should never be called without existing trainingTree, as it
   // the vector of events from the ROOT training tree
   if (!HasTrainingTree()) fLogger << kFATAL << "<Init> Data().TrainingTree() is zero pointer" << Endl;

   Int_t nevents = Data().GetNEvtTrain();
   for (Int_t ievt=0; ievt<nevents; ievt++){
      ReadTrainingEvent(ievt);
      fEventSample.push_back( new Event(GetEvent()) );
   }
   if (fTreeEveFrac<=0) {
      Double_t n = static_cast<Double_t>(nevents);
      fTreeEveFrac = min( 0.5, (100.0 +6.0*sqrt(n))/n);
   }
   if (fTreeEveFrac>1.0) fTreeEveFrac=1.0;
   //
   std::random_shuffle(fEventSample.begin(), fEventSample.end());
   //
   fLogger << kVERBOSE << "Set sub-sample fraction to " << fTreeEveFrac << Endl;
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::Train( void )
{
   // training of rules

   if (fUseRuleFitJF) {
      TrainJFRuleFit();
   } 
   else {
      TrainTMVARuleFit();
   }
   fRuleFit.GetRuleEnsemblePtr()->ClearRuleMap();
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::TrainTMVARuleFit( void )
{
   // training of rules using TMVA implementation

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
   if (IsNormalised()) fLogger << kFATAL << "\"Normalise\" option cannot be used with RuleFit; " 
                               << "please remove the optoin from the configuration string, or "
                               << "use \"!Normalise\""
                               << Endl;

   // timer
   Timer timer( 1, GetName() );

   // test tree nmin cut -> for debug purposes
   // the routine will generate trees with stopping cut on N(eve) given by
   // a fraction between [20,N(eve)-1].
   // 
   //   MakeForestRnd();
   //   exit(1);
   //

   // Init RuleFit object and create rule ensemble
   // + make forest & rules
   fRuleFit.Initialize( this );

   // Make forest of decision trees
   //   if (fRuleFit.GetRuleEnsemble().DoRules()) fRuleFit.MakeForest();

   // Fit the rules
   fLogger << kVERBOSE << "Fitting rule coefficients ..." << Endl;
   fRuleFit.FitCoefficients();

   // print timing info
   fLogger << kINFO << "Elapsed time: " << timer.GetElapsedTime() << Endl;

   // Calculate importance
   fLogger << kVERBOSE << "Computing rule and variable importance" << Endl;
   fRuleFit.CalcImportance();

   // Output results and fill monitor ntuple
   fRuleFit.GetRuleEnsemblePtr()->Print();
   //
   fLogger << kVERBOSE << "Filling rule ntuple" << Endl;
   UInt_t nrules = fRuleFit.GetRuleEnsemble().GetRulesConst().size();
   const Rule *rule;
   for (UInt_t i=0; i<nrules; i++ ) {
      rule            = fRuleFit.GetRuleEnsemble().GetRulesConst(i);
      fNTImportance   = rule->GetRelImportance();
      fNTSupport      = rule->GetSupport();
      fNTCoefficient  = rule->GetCoefficient();
      fNTType         = (rule->IsSignalRule() ? 1:-1 );
      fNTNvars        = rule->GetRuleCut()->GetNvars();
      fNTNcuts        = rule->GetRuleCut()->GetNcuts();
      fNTPtag         = fRuleFit.GetRuleEnsemble().GetRulePTag(i); // should be identical with support
      fNTPss          = fRuleFit.GetRuleEnsemble().GetRulePSS(i);
      fNTPsb          = fRuleFit.GetRuleEnsemble().GetRulePSB(i);
      fNTPbs          = fRuleFit.GetRuleEnsemble().GetRulePBS(i);
      fNTPbb          = fRuleFit.GetRuleEnsemble().GetRulePBB(i);
      fNTSSB          = rule->GetSSB();
      fMonitorNtuple->Fill();
   }
   fLogger << kVERBOSE << "Training done" << Endl;

   fRuleFit.MakeVisHists();

   fRuleFit.MakeDebugHists();
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::TrainJFRuleFit( void )
{
   // training of rules using Jerome Friedmans implementation

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   fRuleFit.InitPtrs( this );
   fRuleFit.SetTrainingEvents( GetTrainingEvents() );

   RuleFitAPI *rfAPI = new RuleFitAPI( this, &fRuleFit, fLogger.GetMinType() );

   rfAPI->WelcomeMessage();

   // timer
   Timer timer( 1, GetName() );

   fLogger << kINFO << "Training ..." << Endl;
   rfAPI->TrainRuleFit();
   fLogger << kINFO << "Elapsed time: " << timer.GetElapsedTime() << Endl;

   fLogger << kVERBOSE << "reading model summary from rf_go.exe output" << Endl;
   rfAPI->ReadModelSum();

   //   fRuleFit.GetRuleEnsemblePtr()->MakeRuleMap();

   fLogger << kVERBOSE << "calculating rule and variable importance" << Endl;
   fRuleFit.CalcImportance();

   // Output results and fill monitor ntuple
   fRuleFit.GetRuleEnsemblePtr()->Print();
   //
   fRuleFit.MakeVisHists();

   delete rfAPI;

   fLogger << kVERBOSE << "done training" << Endl;
}

//_______________________________________________________________________
const TMVA::Ranking* TMVA::MethodRuleFit::CreateRanking() 
{
   // computes ranking of input variables

   // create the ranking object
   fRanking = new Ranking( GetName(), "Importance" );

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fRanking->AddRank( *new Rank( GetInputExp(ivar), fRuleFit.GetRuleEnsemble().GetVarImportance(ivar) ) );
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
   BaseDir()->cd();
   fLogger << kINFO << "write monitoring ntuple to file: " << BaseDir()->GetPath() << Endl;
   fMonitorNtuple->Write();
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << std::endl;
   fout << "};" << std::endl;
   fout << "void   " << className << "::Initialize(){}" << std::endl;
   fout << "void   " << className << "::Clear(){}" << std::endl;
   fout << "double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const {" << std::endl;
   fout << "   double rval=" << setprecision(10) << fRuleFit.GetRuleEnsemble().GetOffset() << ";" << std::endl;
   MakeClassRuleCuts(fout);
   MakeClassLinear(fout);
   fout << "   return rval;" << std::endl;
   fout << "}" << std::endl;

}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeClassRuleCuts( std::ostream& fout ) const
{
   // print out the rule cuts
   if (!fRuleFit.GetRuleEnsemble().DoRules()) {
      fout << "   //" << std::endl;
      fout << "   // ==> MODEL CONTAINS NO RULES <==" << std::endl;
      fout << "   //" << std::endl;
      return;
   }
   const RuleEnsemble *rens = &(fRuleFit.GetRuleEnsemble());
   const std::vector< Rule* > *rules = &(rens->GetRulesConst());
   const RuleCut *ruleCut;
   //
   std::list< std::pair<Double_t,Int_t> > sortedRules;
   for (UInt_t ir=0; ir<rules->size(); ir++) {
      sortedRules.push_back( std::pair<Double_t,Int_t>( (*rules)[ir]->GetImportance()/rens->GetImportanceRef(),ir ) );
   }
   sortedRules.sort();
   //
   fout << "   //" << std::endl;
   fout << "   // here follows all rules ordered in importance (most important first)" << std::endl;
   fout << "   // at the end of each line, the relative importance of the rule is given" << std::endl;
   fout << "   //" << std::endl;
   //
   for ( std::list< std::pair<double,int> >::reverse_iterator itpair = sortedRules.rbegin();
         itpair != sortedRules.rend(); itpair++ ) {
      UInt_t ir     = itpair->second;
      Double_t impr = itpair->first;
      ruleCut = (*rules)[ir]->GetRuleCut();
      if (impr<rens->GetImportanceCut()) fout << "   //" << std::endl;
      fout << "   if (" << std::flush;
      for (UInt_t ic=0; ic<ruleCut->GetNvars(); ic++) {
         Double_t sel    = ruleCut->GetSelector(ic);
         Double_t valmin = ruleCut->GetCutMin(ic);
         Double_t valmax = ruleCut->GetCutMax(ic);
         Bool_t   domin  = ruleCut->GetCutDoMin(ic);
         Bool_t   domax  = ruleCut->GetCutDoMax(ic);
         //
         if (ic>0) fout << "&&" << std::flush;
         if (domin) {
            fout << "(" << setprecision(10) << valmin << std::flush;
            fout << "<inputValues[" << sel << "])" << std::flush;
         }
         if (domax) {
            if (domin) fout << "&&" << std::flush;
            fout << "(inputValues[" << sel << "]" << std::flush;
            fout << "<" << setprecision(10) << valmax << ")" <<std::flush;
         }
      }
      fout << ") rval+=" << setprecision(10) << (*rules)[ir]->GetCoefficient() << ";" << std::flush;
      fout << "   // importance = " << Form("%3.3f",impr) << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::MakeClassLinear( std::ostream& fout ) const
{
   // print out the linear terms
   if (!fRuleFit.GetRuleEnsemble().DoLinear()) {
      fout << "   //" << std::endl;
      fout << "   // ==> MODEL CONTAINS NO LINEAR TERMS <==" << std::endl;
      fout << "   //" << std::endl;
      return;
   }
   fout << "   //" << std::endl;
   fout << "   // here follows all linear terms" << std::endl;
   fout << "   // at the end of each line, the relative importance of the term is given" << std::endl;
   fout << "   //" << std::endl;
   const RuleEnsemble *rens = &(fRuleFit.GetRuleEnsemble());
   UInt_t nlin = rens->GetNLinear();
   for (UInt_t il=0; il<nlin; il++) {
      if (rens->IsLinTermOK(il)) {
         Double_t norm = rens->GetLinNorm(il);
         Double_t imp  = rens->GetLinImportance(il)/rens->GetImportanceRef();
         fout << "   rval+="
              << setprecision(10) << rens->GetLinCoefficients(il)*norm << "*std::min(" << setprecision(10) << rens->GetLinDP(il)
              << ", std::max( inputValues[" << il << "]," << setprecision(10) << rens->GetLinDM(il) << "));"
              << std::flush;
         fout << "   // importance = " << Form("%3.3f",imp) << std::endl;
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodRuleFit::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "This method uses a collection of so called rules to create a" << Endl;
   fLogger << "discriminating scoring function. Each rule consists of a series" << Endl;
   fLogger << "of cuts in parameter space. The ensemble of rules are created" << Endl;
   fLogger << "from a forest of decision trees, trained using the training data." << Endl;
   fLogger << "Each node (apart from the root) corresponds to one rule." << Endl;
   fLogger << "The scoring function is then obtained by linearly combining" << Endl;
   fLogger << "the rules. A fitting procedure is applied to find the optimum" << Endl;
   fLogger << "set of coefficients. The goal is to find a model with few rules" << Endl;
   fLogger << "but with a strong discriminating power." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "There are two important considerations to make when optimising:" << Endl;
   fLogger << Endl;
   fLogger << "  1. topology of the decision tree forest" << Endl;
   fLogger << "  2. fitting of the coefficients" << Endl;
   fLogger << Endl;
   fLogger << "The maximum complexity of the rules is defined by the size of" << Endl;
   fLogger << "the trees. Large trees will yield many complex rules and capture" << Endl;
   fLogger << "higher order correlations. On the other hand, small trees will" << Endl;
   fLogger << "lead to a smaller ensemble with simple rules, only capable of" << Endl;
   fLogger << "modeling simple structures." << Endl;
   fLogger << "Several parameters exists for controlling the complexity of the" << Endl;
   fLogger << "rule ensemble." << Endl;
   fLogger << Endl;
   fLogger << "The fitting procedure searches for a minimum using a gradient" << Endl;
   fLogger << "directed path. Apart from step size and number of steps, the" << Endl;
   fLogger << "evolution of the path is defined by a cut-off parameter, tau." << Endl;
   fLogger << "This parameter is unknown and depends on the training data." << Endl;
   fLogger << "A large value will tend to give large weights to a few rules." << Endl;
   fLogger << "Similarily, a small value will lead to a large set of rules" << Endl;
   fLogger << "with similar weights." << Endl;
   fLogger << Endl;
   fLogger << "A final point is the model used; rules and/or linear terms." << Endl;
   fLogger << "For a given training sample, the result may improve by adding" << Endl;
   fLogger << "linear terms. If best performance is optained using only linear" << Endl;
   fLogger << "terms, it is very likely that the Fisher discriminant would be" << Endl;
   fLogger << "a better choice. Ideally the fitting procedure should be able to" << Endl;
   fLogger << "make this choice by giving appropriate weights for either terms." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "I.  TUNING OF RULE ENSEMBLE:" << Endl;
   fLogger << Endl;
   fLogger << "   " << Tools::Color("bold") << "ForestType  " << Tools::Color("reset")
           << ": Recomended is to use the default <AdaBoost>." << Endl;
   fLogger << "   " << Tools::Color("bold") << "nTrees      " << Tools::Color("reset")
           << ": More trees leads to more rules but also slow" << Endl;
   fLogger << "                 performance. With too few trees the risk is" << Endl;
   fLogger << "                 that the rule ensemble becomes too simple." << Endl;
   fLogger << "   " << Tools::Color("bold") << "fEventsMin  " << Tools::Color("reset") << Endl;
   fLogger << "   " << Tools::Color("bold") << "fEventsMax  " << Tools::Color("reset")
           << ": With a lower min, more large trees will be generated" << Endl;
   fLogger << "                 leading to more complex rules." << Endl;
   fLogger << "                 With a higher max, more small trees will be" << Endl;
   fLogger << "                 generated leading to more simple rules." << Endl;
   fLogger << "                 By changing this range, the average complexity" << Endl;
   fLogger << "                 of the rule ensemble can be controlled." << Endl;
   fLogger << "   " << Tools::Color("bold") << "RuleMinDist " << Tools::Color("reset")
           << ": By increasing the minimum distance between" << Endl;
   fLogger << "                 rules, fewer and more diverse rules will remain." << Endl;
   fLogger << "                 Initially it's a good idea to keep this small" << Endl;
   fLogger << "                 or zero and let the fitting do the selection of" << Endl;
   fLogger << "                 rules. In order to reduce the ensemble size," << Endl;
   fLogger << "                 the value can then be increased." << Endl;
   fLogger << Endl;
   //         "|--------------------------------------------------------------|"
   fLogger << "II. TUNING OF THE FITTING:" << Endl;
   fLogger << Endl;
   fLogger << "   " << Tools::Color("bold") << "GDPathEveFrac " << Tools::Color("reset")
           << ": fraction of events in path evaluation" << Endl;
   fLogger << "                 Increasing this fraction will improve the path" << Endl;
   fLogger << "                 finding. However, a too high value will give few" << Endl;
   fLogger << "                 unique events available for error estimation." << Endl;
   fLogger << "                 It is recomended to usethe default = 0.5." << Endl;
   fLogger << "   " << Tools::Color("bold") << "GDTau         " << Tools::Color("reset")
           << ": cutoff parameter tau" << Endl;
   fLogger << "                 By default this value is set to -1.0." << Endl;
   //         "|----------------|---------------------------------------------|"
   fLogger << "                 This means that the cut off parameter is" << Endl;
   fLogger << "                 automatically estimated. In most cases" << Endl;
   fLogger << "                 this should be fine. However, you may want" << Endl;
   fLogger << "                 to fix this value if you already know it" << Endl;
   fLogger << "                 and want to reduce on training time." << Endl;
   fLogger << "   " << Tools::Color("bold") << "GDTauPrec     " << Tools::Color("reset")
           << ": precision of estimated tau" << Endl;
   fLogger << "                 Increase this precision to find a more" << Endl;
   fLogger << "                 optimum cut-off parameter." << Endl;
   fLogger << "   " << Tools::Color("bold") << "GDNStep       " << Tools::Color("reset")
           << ": number of steps in path search" << Endl;
   fLogger << "                 If the number of steps is too small, then" << Endl;
   fLogger << "                 the program will give a warning message." << Endl;
   fLogger << Endl;
   fLogger << "III. WARNING MESSAGES" << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("red") << "Risk(i+1)>=Risk(i) in path" << Tools::Color("reset") << Endl;
   fLogger << Tools::Color("red") << "Chaotic behaviour of risk evolution." << Tools::Color("reset") << Endl;
   //         "|----------------|---------------------------------------------|"
   fLogger << "                 The error rate was still decreasing at the end" << Endl;
   fLogger << "                 By construction the Risk should always decrease." << Endl;
   fLogger << "                 However, if the training sample is too small or" << Endl;
   fLogger << "                 the model is overtrained, such warnings can" << Endl;
   fLogger << "                 occur." << Endl;
   fLogger << "                 The warnings can safely be ignored if only a" << Endl;
   fLogger << "                 few (<3) occur. If more warnings are generated," << Endl;
   fLogger << "                 the fitting fails." << Endl;
   fLogger << "                 A remedy may be to increase the value" << Endl;
   fLogger << "                 "
           << Tools::Color("bold") << "GDValidEveFrac" << Tools::Color("reset")
           << " to 1.0 (or a larger value)." << Endl;
   fLogger << "                 In addition, if "
           << Tools::Color("bold") << "GDPathEveFrac" << Tools::Color("reset")
           << " is too high" << Endl;
   fLogger << "                 the same warnings may occur since the events" << Endl;
   fLogger << "                 used for error estimation are also used for" << Endl;
   fLogger << "                 path estimation." << Endl;
   fLogger << "                 Another possibility is to modify the model - " << Endl;
   fLogger << "                 See above on tuning the rule ensemble." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("red") << "The error rate was still decreasing at the end of the path"
           << Tools::Color("reset") << Endl;
   fLogger << "                 Too few steps in path! Increase "
           << Tools::Color("bold") << "GDNSteps" <<  Tools::Color("reset") << "." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("red") << "Reached minimum early in the search" << Tools::Color("reset") << Endl;

   fLogger << "                 Minimum was found early in the fitting. This" << Endl;
   fLogger << "                 may indicate that the used step size "
           << Tools::Color("bold") << "GDStep" <<  Tools::Color("reset") << "." << Endl;
   fLogger << "                 was too large. Reduce it and rerun." << Endl;
   fLogger << "                 If the results still are not OK, modify the" << Endl;
   fLogger << "                 model either by modifying the rule ensemble" << Endl;
   fLogger << "                 or add/remove linear terms" << Endl;
   fLogger << Endl;
}
