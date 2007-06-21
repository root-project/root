// @(#)root/tmva $Id: MethodCuts.cxx,v 1.15 2007/06/19 13:26:21 brun Exp $ 
// Author: Andreas Hoecker, Matt Jachowski, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodCuts                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//Begin_Html
/*
  Multivariate optimisation of signal efficiency for given background  
  efficiency, applying rectangular minimum and maximum requirements.

  <p>
  Also implemented is a "decorrelate/diagonlized cuts approach",            
  which improves over the uncorrelated cuts ansatz by            
  transforming linearly the input variables into a diagonal space,     
  using the square-root of the covariance matrix.

  <p>
  <font size="-1">
  Other optimisation criteria, such as maximising the signal significance-
  squared, S^2/(S+B), with S and B being the signal and background yields, 
  correspond to a particular point in the optimised background rejection 
  versus signal efficiency curve. This working point requires the knowledge 
  of the expected yields, which is not the case in general. Note also that 
  for rare signals, Poissonian statistics should be used, which modifies 
  the significance criterion. 
  </font>

  <p>
  The rectangular cut of a volume in the variable space is performed using 
  a binary tree to sort the training events. This provides a significant 
  reduction in computing time (up to several orders of magnitudes, depending
  on the complexity of the problem at hand).

  <p>
  Technically, optimisation is achieved in TMVA by two methods:

  <ol>
  <li>Monte Carlo generation using uniform priors for the lower cut value, 
  and the cut width, thrown within the variable ranges. 

  <li>A Genetic Algorithm (GA) searches for the optimal ("fittest") cut sample.
  The GA is configurable by many external settings through the option 
  string. For difficult cases (such as many variables), some tuning 
  may be necessary to achieve satisfying results
  </ol>

  <p>
  <font size="-1">
  Attempts to use Minuit fits (Simplex ot Migrad) instead have not shown 
  superior results, and often failed due to convergence at local minima. 
  </font>

  <p>
  The tests we have performed so far showed that in generic applications, 
  the GA is superior to MC sampling, and hence GA is the default method.
  It is worthwhile trying both anyway.

  <b>Decorrelated (or "diagonalized") Cuts</b>

  <p>
  See class description for Method Likelihood for a detailed explanation.
*/
//End_Html


#include <stdio.h>
#include "time.h"
#include "Riostream.h"
#include "TH1F.h"
#include "TObjString.h"
#include "TDirectory.h"
#include "TMath.h"

#include "TMVA/MethodCuts.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/MinuitFitter.h"
#include "TMVA/MCFitter.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Interval.h"

ClassImp(TMVA::MethodCuts)

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( TString jobName, TString methodTitle, DataSet& theData, 
                              TString theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{ 
   // standard constructor
   // see below for option string format

   InitCuts();

   // interpretation of configuration option string
   DeclareOptions();
   ParseOptions();
   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( DataSet& theData, 
                              TString theWeightFile,  
                              TDirectory* theTargetDir )
   : MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // construction from weight file
   InitCuts();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodCuts::InitCuts( void ) 
{
   // default initialisation called by all constructors
   SetMethodName( "Cuts" );
   SetMethodType( Types::kCuts );  
   SetTestvarName();

   fVarHistS          = fVarHistB = 0;                 
   fVarHistS_smooth   = fVarHistB_smooth = 0;
   fVarPdfS           = fVarPdfB = 0; 
   fFitParams         = 0;
   fEffBvsSLocal      = 0;
   fBinaryTreeS       = fBinaryTreeB = 0;
   fEffSMin           = 0;
   fEffSMax           = 0; 
   fTrainEffBvsS      = 0;
   fTrainRejBvsS      = 0;

   // vector with fit results
   fNpar      = 2*GetNvar();
   fRangeSign = new vector<Int_t>   ( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*fRangeSign)[ivar] = +1;

   fMeanS     = new vector<Double_t>( GetNvar() ); 
   fMeanB     = new vector<Double_t>( GetNvar() ); 
   fRmsS      = new vector<Double_t>( GetNvar() );  
   fRmsB      = new vector<Double_t>( GetNvar() );  

   // get the variable specific options, first initialize default
   fFitParams = new vector<EFitParameters>( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*fFitParams)[ivar] = kNotEnforced;

   fFitMethod = kUseMonteCarlo;
   fTestSignalEff = -1;

   // create LUT for cuts
   fCutMin = new Double_t*[GetNvar()];
   fCutMax = new Double_t*[GetNvar()];
   for (Int_t i=0;i<GetNvar();i++) {
      fCutMin[i] = new Double_t[fNbins];
      fCutMax[i] = new Double_t[fNbins];
   }
  
   // init
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      for (Int_t ibin=0; ibin<fNbins; ibin++) {
         fCutMin[ivar][ibin] = 0;
         fCutMax[ivar][ibin] = 0;
      }
   }

   fTmpCutMin = new Double_t[GetNvar()];
   fTmpCutMax = new Double_t[GetNvar()];
}

//_______________________________________________________________________
TMVA::MethodCuts::~MethodCuts( void )
{
   // destructor
   delete fRangeSign;
   delete fMeanS;
   delete fMeanB;
   delete fRmsS;
   delete fRmsB;
   for (Int_t i=0;i<GetNvar();i++) {
      if (fCutMin[i]   != NULL) delete [] fCutMin[i];
      if (fCutMax[i]   != NULL) delete [] fCutMax[i];
      if (fCutRange[i] != NULL) delete fCutRange[i];
   }

   delete[] fCutMin;
   delete[] fCutMax;

   delete[] fTmpCutMin;
   delete[] fTmpCutMax;

   if (NULL != fBinaryTreeS) delete fBinaryTreeS;
   if (NULL != fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
void TMVA::MethodCuts::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // know options:
   // Method             <string> Minimization method
   //    available values are:        MC Monte Carlo <default>
   //                                 GA Genetic Algorithm
   //                                 SA Simulated annealing
   //
   // EffMethod          <string> Efficiency selection method
   //    available values are:        EffSel <default>
   //                                 EffPDF
   //
   // VarProp            <string> Property of variable 1 for the MC method (taking precedence over the
   //    globale setting. The same values as for the global option are available. Variables 1..10 can be
   //    set this way
   //
   // CutRangeMin/Max    <float>  user-defined ranges in which cuts are varied

   DeclareOptionRef(fFitMethodS = "GA", "FitMethod", "Minimization Method");
   AddPreDefVal(TString("GA"));
   AddPreDefVal(TString("SA"));
   AddPreDefVal(TString("MC"));
   AddPreDefVal(TString("MINUIT"));

   // selection type
   DeclareOptionRef(fEffMethodS = "EffSel", "EffMethod", "Selection Method");
   AddPreDefVal(TString("EffSel"));
   AddPreDefVal(TString("EffPDF"));

   // cut ranges 
   fCutRange.resize(GetNvar());
   fCutRangeMin = new Double_t[GetNvar()];
   fCutRangeMax = new Double_t[GetNvar()];
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fCutRange[ivar] = 0;
      fCutRangeMin[ivar] = fCutRangeMax[ivar] = -1;
   }

   DeclareOptionRef( fCutRangeMin, GetNvar(), "CutRangeMin", "Minimum of allowed cut range (set per variable)" );
   DeclareOptionRef( fCutRangeMax, GetNvar(), "CutRangeMax", "Maximum of allowed cut range (set per variable)" );   

   fAllVarsI = new TString[GetNvar()];

   for (int i=0; i<GetNvar(); i++) fAllVarsI[i] = "NotEnforced";  

   DeclareOptionRef(fAllVarsI, GetNvar(), "VarProp", "Categorisation of cuts");  
   AddPreDefVal(TString("NotEnforced"));
   AddPreDefVal(TString("FMax"));
   AddPreDefVal(TString("FMin"));
   AddPreDefVal(TString("FSmart"));
   AddPreDefVal(TString("FVerySmart"));
}

//_______________________________________________________________________
void TMVA::MethodCuts::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();

   if      (fFitMethodS == "MC" ) fFitMethod = kUseMonteCarlo;
   else if (fFitMethodS == "GA" ) fFitMethod = kUseGeneticAlgorithm;
   else if (fFitMethodS == "SA" ) fFitMethod = kUseSimulatedAnnealing;
   else if (fFitMethodS == "MINUIT" ) {
      fFitMethod = kUseMinuit;
      fLogger << kWARNING << "poor performance of MINUIT in MethodCuts; preferred fit method: GA" << Endl;
   }
   else {
      fLogger << kFATAL << "unknown minimization method: " << fFitMethodS << Endl;
   }

   if      (fEffMethodS == "EFFSEL" ) fEffMethod = kUseEventSelection; // highly recommended
   else if (fEffMethodS == "EFFPDF" ) fEffMethod = kUsePDFs;
   else                               fEffMethod = kUseEventSelection;

   // options output
   fLogger << kINFO << Form("Use optimization method: '%s'\n", 
                            (fFitMethod == kUseMonteCarlo) ? "Monte Carlo" : "Genetic Algorithm" );
   fLogger << kINFO << Form("Use efficiency computation method: '%s'\n", 
                            (fEffMethod == kUseEventSelection) ? "Event Selection" : "PDF" );

   // cut ranges
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fCutRange[ivar] = new Interval( fCutRangeMin[ivar], fCutRangeMax[ivar] );
   }

   // individual options
   int maxVar = GetNvar();
   for (Int_t ivar=0; ivar<maxVar; ivar++) {
      EFitParameters theFitP = kNotEnforced;      
      if (fAllVarsI[ivar] == "" || fAllVarsI[ivar] == "NotEnforced") theFitP = kNotEnforced;
      else if (fAllVarsI[ivar] == "FMax" )                           theFitP = kForceMax;
      else if (fAllVarsI[ivar] == "FMin" )                           theFitP = kForceMin;
      else if (fAllVarsI[ivar] == "FSmart" )                         theFitP = kForceSmart;
      else if (fAllVarsI[ivar] == "FVerySmart" )                     theFitP = kForceVerySmart;
      else {
         fLogger << kFATAL << "unknown value \'" << fAllVarsI[ivar]
                 << "\' for fit parameter option " << Form("VarProp[%i]",ivar+1) << Endl;
      }
      (*fFitParams)[ivar] = theFitP;
      
      if (theFitP != kNotEnforced) 
         fLogger << kINFO << "Use \"" << fAllVarsI[ivar] 
                 << "\" cuts for variable: " << "'" << (*fInputVars)[ivar] << "'" << Endl;
   }

   // -----------------------------------------------------------------------------------
   // interpret for MC use  
   //
   if (fFitMethod == kUseMonteCarlo) {
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         TString theFitOption = ( ((*fFitParams)[ivar] == kNotEnforced) ? "NotEnforced" :
                                  ((*fFitParams)[ivar] == kForceMin   ) ? "ForceMin"    :
                                  ((*fFitParams)[ivar] == kForceMax   ) ? "ForceMax"    :
                                  ((*fFitParams)[ivar] == kForceSmart ) ? "ForceSmart"  :
                                  ((*fFitParams)[ivar] == kForceVerySmart ) ? "ForceVerySmart"  : "other" );
         
         fLogger << kINFO << Form("Option for variable: %s: '%s' (#: %i)\n",
                                  (const char*)(*fInputVars)[ivar], (const char*)theFitOption, 
                                  (Int_t)(*fFitParams)[ivar] );
      }
   }

   // decorrelate option will be last option, if it is specified
   if (GetVariableTransform() == Types::kDecorrelated)
      fLogger << kINFO << "Use decorrelated variable set" << Endl;
   else if (GetVariableTransform() == Types::kPCA)
      fLogger << kINFO << "Use principal component transformation" << Endl;
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::GetMvaValue()
{
   // cut evaluation: returns 1.0 if event passed, 0.0 otherwise

   // sanity check
   if (fCutMin == NULL || fCutMax == NULL || fNbins == 0) {
      fLogger << kFATAL << "<Eval_Cuts> fCutMin/Max have zero pointer. "
              << "Did you book Cuts ?" << Endl;
   }

   // sanity check
   if (fTestSignalEff > 0) {
      // get efficiency bin
      Int_t ibin = Int_t((fTestSignalEff - fEffSMin)/(fEffSMax - fEffSMin)*Double_t(fNbins));
      if (ibin < 0      ) ibin = 0;
      if (ibin >= fNbins) ibin = fNbins - 1;
    
      Bool_t passed = kTRUE;
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         passed &= ( (GetEventVal(ivar) >= fCutMin[ivar][ibin]) && 
                     (GetEventVal(ivar) <= fCutMax[ivar][ibin]) );

      return passed ? 1. : 0. ;
   }
   else return 0;
}

//_______________________________________________________________________
void  TMVA::MethodCuts::Train( void )
{
   // training method: here the cuts are optimised for the training sample
   
   // perform basic sanity chacks
   if (!SanityChecks()) fLogger << kFATAL << "Basic sanity checks failed" << Endl;

   if (fEffMethod == kUsePDFs) CreateVariablePDFs(); // create PDFs for variables

   // create binary trees (global member variables) for signal and background
   if (fBinaryTreeS != 0) delete fBinaryTreeS;
   if (fBinaryTreeB != 0) delete fBinaryTreeB;

   // the variables may be transformed by a transformation method: to coherently 
   // treat signal and background one must decide which transformation type shall 
   // be used: our default is signal-type
   fBinaryTreeS = new BinarySearchTree();
   fBinaryTreeS->Fill( *this, Data().GetTrainingTree(), 1 );
   fBinaryTreeB = new BinarySearchTree();
   fBinaryTreeB->Fill( *this, Data().GetTrainingTree(), 0 );

   for (UInt_t ivar =0; ivar < Data().GetNVariables(); ivar++) {
      (*fMeanS)[ivar] = fBinaryTreeS->Mean(Types::kSignal, ivar);
      (*fRmsS)[ivar]  = fBinaryTreeS->RMS (Types::kSignal, ivar);
      (*fMeanB)[ivar] = fBinaryTreeB->Mean(Types::kBackground, ivar);
      (*fRmsB)[ivar]  = fBinaryTreeB->RMS (Types::kBackground, ivar);

      // update interval ?
      Double_t xmin = TMath::Min(fBinaryTreeS->Min(Types::kSignal, ivar), fBinaryTreeB->Min(Types::kBackground, ivar));
      Double_t xmax = TMath::Max(fBinaryTreeS->Max(Types::kSignal, ivar), fBinaryTreeB->Max(Types::kBackground, ivar));

      if (fCutRange[ivar]->GetMin() == fCutRange[ivar]->GetMax()) {
         fCutRange[ivar]->SetMin( xmin );
         fCutRange[ivar]->SetMax( xmax );
      }         
      else if (xmin > fCutRange[ivar]->GetMin()) fCutRange[ivar]->SetMin( xmin );
      else if (xmax < fCutRange[ivar]->GetMax()) fCutRange[ivar]->SetMax( xmax );
   }   

   vector<TH1F*> signalDist, bkgDist;

   // this is important: reset the branch addresses of the training tree to the current event
   Data().ResetCurrentTree();

   fEffBvsSLocal = new TH1F( GetTestvarName() + "_effBvsSLocal", 
                             TString(GetName()) + " efficiency of B vs S", fNbins, 0.0, 1.0 );

   // init
   for (Int_t ibin=1; ibin<=fNbins; ibin++) fEffBvsSLocal->SetBinContent( ibin, -0.1 );

   // --------------------------------------------------------------------------
   if (fFitMethod == kUseGeneticAlgorithm || fFitMethod == kUseMonteCarlo || fFitMethod == kUseMinuit) {

      // ranges
      vector<Interval*> ranges;

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {

         Int_t nbins = 0;
         if (Data().GetVarType(ivar) == 'I') {
            nbins = Int_t(fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin()) + 1;
         }

         EFitParameters fitParam = (*fFitParams)[ivar];

         if (fitParam == kForceSmart) {
            if ((*fMeanS)[ivar] > (*fMeanB)[ivar]) fitParam = kForceMax;
            else                                   fitParam = kForceMin;          
         }

         if (fitParam == kForceMin){
            ranges.push_back( new Interval( fCutRange[ivar]->GetMin(), fCutRange[ivar]->GetMin(), nbins ) );
            ranges.push_back( new Interval( 0, fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), nbins ) );
         }
         else if (fitParam == kForceMax){
            ranges.push_back( new Interval( fCutRange[ivar]->GetMin(), fCutRange[ivar]->GetMax(), nbins ) );
            ranges.push_back( new Interval( fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), 
                                            fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), nbins ) );
         }
         else{
            ranges.push_back( new Interval( fCutRange[ivar]->GetMin(), fCutRange[ivar]->GetMax(), nbins ) );
            ranges.push_back( new Interval( 0, fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), nbins ) );
         }
      }

      // create the fitter
      FitterBase* fitter = NULL;
      
      switch (fFitMethod) {
      case kUseGeneticAlgorithm:
         fitter = new GeneticFitter( *this, Form("%sFitter_GA",     GetName()), ranges, GetOptions() );
         break;
      case kUseMonteCarlo:
         fitter = new MCFitter     ( *this, Form("%sFitter_MC",     GetName()), ranges, GetOptions() );
         break;
      case kUseMinuit:
         fitter = new MinuitFitter ( *this, Form("%sFitter_MINUIT", GetName()), ranges, GetOptions() );
         break;
      default:
         fLogger << kFATAL << "Wrong fit method: " << fFitMethod << Endl;
      }

      fitter->CheckForUnusedOptions();

      fitter->Run();      

      // clean up
      for (UInt_t ivar=0; ivar<ranges.size(); ivar++) delete ranges[ivar];
   }
   // --------------------------------------------------------------------------
   else fLogger << kFATAL << "unknown minization method: " << fFitMethod << Endl;

   if (fBinaryTreeS != 0) { delete fBinaryTreeS; fBinaryTreeS = 0; }
   if (fBinaryTreeB != 0) { delete fBinaryTreeB; fBinaryTreeB = 0; }
}

//_______________________________________________________________________
void TMVA::MethodCuts::Test( TTree* )
{
   // not used 
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::EstimatorFunction( std::vector<Double_t>& par )
{
   // returns estimator for "cut fitness" used by GA
   return ComputeEstimator( par );
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::ComputeEstimator( std::vector<Double_t>& par )
{
   // returns estimator for "cut fitness" used by GA
   // there are two requirements:
   // 1) the signal efficiency must be equal to the required one in the 
   //    efficiency scan
   // 2) the background efficiency must be as small as possible
   // the requirement 1) has priority over 2)

   // caution: the npar gives the _free_ parameters
   // however: the "par" array contains all parameters

   // determine cuts
   Double_t effS = 0, effB = 0;
   this->MatchParsToCuts( par, &fTmpCutMin[0], &fTmpCutMax[0] );

   // retrieve signal and background efficiencies for given cut
   switch (fEffMethod) {
   case kUsePDFs:
      this->GetEffsfromPDFs( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB );
      break;
   case kUseEventSelection:
      this->GetEffsfromSelection( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB);
      break;
   default:
      this->GetEffsfromSelection( &fTmpCutMin[0], &fTmpCutMax[0], effS, effB);
   }

   Double_t eta = 0;      
   
   // test for a estimator function which optimizes on the whole background-rejection signal-efficiency plot
   
   // get the backg-reject. and sig-eff for the parameters given to this function
   // effS, effB
      
   // get the "best signal eff" for the backg-reject.
   // determine bin
   Int_t    ibinS = (Int_t)(effS*Float_t(fNbins) + 1);
   if (ibinS < 1     ) ibinS = 1;
   if (ibinS > fNbins) ibinS = fNbins;
      
   Double_t effBH       = fEffBvsSLocal->GetBinContent( ibinS );
   Double_t effBH_left  = fEffBvsSLocal->GetBinContent( ibinS-1 );
   Double_t effBH_right = fEffBvsSLocal->GetBinContent( ibinS+1 );

   Double_t average = (effBH_left+effBH_right)/2.;
   if (effBH < effB) average = effBH;

   // if the average of the bin right and left is larger than this one, add the difference to 
   // the actual value of the estimator (because you can do at least so much better)
   eta = ( -TMath::Abs(effBH-average) +( 1. - (effBH - effB) ) ) / (1+effS); 

   // if a point is found which is better than an existing one, ... replace it. 
   // preliminary best event -> backup
   if (effBH < 0 || effBH > effB) {
      fEffBvsSLocal->SetBinContent( ibinS, effB );
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         fCutMin[ivar][ibinS-1] = fTmpCutMin[ivar]; // bin 1 stored in index 0
         fCutMax[ivar][ibinS-1] = fTmpCutMax[ivar];
      }
   }
   
   // attention!!! this value is not good for a decision for MC, .. its designed for GA
   // but .. it doesn't matter, as MC samplings are independent from the former ones
   // and the replacement of the best variables by better ones is done about 10 lines above. 
   // ( if (effBH < 0 || effBH > effB) { .... )
   return eta;
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchParsToCuts( const std::vector<Double_t> & par, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates parameters into cuts
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Int_t ipar = 2*ivar;
      cutMin[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] : par[ipar] - par[ipar+1];
      cutMax[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] + par[ipar+1] : par[ipar]; 
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchCutsToPars( std::vector<Double_t>& par, 
                                        Double_t** cutMinAll, Double_t** cutMaxAll, Int_t ibin )
{
   // translate the cuts into parameters
   if (ibin < 1 || ibin > fNbins) fLogger << kFATAL << "::MatchCutsToPars: bin error: "
                                          << ibin << Endl;
   
   const Int_t nvar = GetNvar();
   Double_t *cutMin = new Double_t[nvar];
   Double_t *cutMax = new Double_t[nvar];
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      cutMin[ivar] = cutMinAll[ivar][ibin-1];
      cutMax[ivar] = cutMaxAll[ivar][ibin-1];
   }
   
   MatchCutsToPars( par, cutMin, cutMax );
   delete [] cutMin;
   delete [] cutMax;
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchCutsToPars( std::vector<Double_t>& par, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates cuts into parameters
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Int_t ipar = 2*ivar;
      par[ipar]   = ((*fRangeSign)[ivar] > 0) ? cutMin[ivar] : cutMax[ivar];
      par[ipar+1] = cutMax[ivar] - cutMin[ivar];
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetEffsfromPDFs( Double_t* cutMin, Double_t* cutMax,
                                        Double_t& effS, Double_t& effB )
{
   // compute signal and background efficiencies from PDFs 
   // for given cut sample
   effS = 1.0;
   effB = 1.0;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      effS *= (*fVarPdfS)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
      effB *= (*fVarPdfB)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetEffsfromSelection( Double_t* cutMin, Double_t* cutMax,
                                             Double_t& effS, Double_t& effB)
{
   // compute signal and background efficiencies from event counting 
   // for given cut sample
   Float_t nTotS = 0, nTotB = 0;
   Float_t nSelS = 0, nSelB = 0;  
      
   Volume* volume = new Volume( cutMin, cutMax, GetNvar() );
  
   // search for all events lying in the volume, and add up their weights
   nSelS = fBinaryTreeS->SearchVolume( volume );
   nSelB = fBinaryTreeB->SearchVolume( volume );

   delete volume;

   // total number of "events" (sum of weights) as reference to compute efficiency
   nTotS = fBinaryTreeS->GetSumOfWeights();
   nTotB = fBinaryTreeB->GetSumOfWeights();
   
   // sanity check
   if (nTotS == 0 && nTotB == 0) {
      fLogger << kFATAL << "<GetEffsfromSelection> fatal error in zero total number of events:"
              << " nTotS, nTotB: " << nTotS << " " << nTotB << " ***" << Endl;
   }

   // efficiencies
   if (nTotS == 0 ) {
      effS = 0;
      effB = nSelB/nTotB;
      fLogger << kWARNING << "<ComputeEstimator> zero number of signal events" << Endl;
   }
   else if (nTotB == 0) {
      effB = 0;
      effS = nSelS/nTotS;
      fLogger << kWARNING << "<ComputeEstimator> zero number of background events" << Endl;
   }
   else {
      effS = nSelS/nTotS;
      effB = nSelB/nTotB;
   }  
}

//_______________________________________________________________________
void TMVA::MethodCuts::CreateVariablePDFs( void )
{
   // for PDF method: create efficiency reference histograms and PDFs

   // create list of histograms and PDFs
   fVarHistS        = new vector<TH1*>( GetNvar() );
   fVarHistB        = new vector<TH1*>( GetNvar() );
   fVarHistS_smooth = new vector<TH1*>( GetNvar() );
   fVarHistB_smooth = new vector<TH1*>( GetNvar() );
   fVarPdfS         = new vector<PDF*>( GetNvar() );
   fVarPdfB         = new vector<PDF*>( GetNvar() );

   Int_t nsmooth = 0;

   for (Int_t ivar=0; ivar<GetNvar(); ivar++) { 

      // ---- signal
      TString histTitle = (*fInputVars)[ivar] + " signal training";
      TString histName  = (*fInputVars)[ivar] + "_sig";
      TString drawOpt   = (*fInputVars)[ivar] + ">>h(";
      drawOpt += fNbins;
      drawOpt += ")";

      // selection
      Data().GetTrainingTree()->Draw( drawOpt, "type==1", "goff" );
      (*fVarHistS)[ivar] = (TH1F*)gDirectory->Get("h");
      (*fVarHistS)[ivar]->SetName(histName);
      (*fVarHistS)[ivar]->SetTitle(histTitle);

      // make copy for smoothed histos
      (*fVarHistS_smooth)[ivar] = (TH1F*)(*fVarHistS)[ivar]->Clone();
      histTitle =  (*fInputVars)[ivar] + " signal training  smoothed ";
      histTitle += nsmooth;
      histTitle +=" times";
      histName =  (*fInputVars)[ivar] + "_sig_smooth";
      (*fVarHistS_smooth)[ivar]->SetName(histName);
      (*fVarHistS_smooth)[ivar]->SetTitle(histTitle);

      // smooth
      (*fVarHistS_smooth)[ivar]->Smooth(nsmooth);

      // ---- background
      histTitle = (*fInputVars)[ivar] + " background training";
      histName  = (*fInputVars)[ivar] + "_bgd";
      drawOpt   = (*fInputVars)[ivar] + ">>h(";
      drawOpt += fNbins;
      drawOpt += ")";

      Data().GetTrainingTree()->Draw( drawOpt, "type==0", "goff" );
      (*fVarHistB)[ivar] = (TH1F*)gDirectory->Get("h");
      (*fVarHistB)[ivar]->SetName(histName);
      (*fVarHistB)[ivar]->SetTitle(histTitle);

      // make copy for smoothed histos
      (*fVarHistB_smooth)[ivar] = (TH1F*)(*fVarHistB)[ivar]->Clone();
      histTitle  = (*fInputVars)[ivar]+" background training  smoothed ";
      histTitle += nsmooth;
      histTitle +=" times";
      histName   = (*fInputVars)[ivar]+"_bgd_smooth";
      (*fVarHistB_smooth)[ivar]->SetName(histName);
      (*fVarHistB_smooth)[ivar]->SetTitle(histTitle);

      // smooth
      (*fVarHistB_smooth)[ivar]->Smooth(nsmooth);

      // create PDFs
      (*fVarPdfS)[ivar] = new PDF( (*fVarHistS_smooth)[ivar], PDF::kSpline2 );
      (*fVarPdfB)[ivar] = new PDF( (*fVarHistB_smooth)[ivar], PDF::kSpline2 );
   }                  
}

//_______________________________________________________________________
Bool_t TMVA::MethodCuts::SanityChecks( void )
{
   // basic checks to ensure that assumptions on variable order are satisfied
   Bool_t        isOK = kTRUE;

   TObjArrayIter branchIter( Data().GetTrainingTree()->GetListOfBranches(), kIterForward );
   TBranch*      branch = 0;
   Int_t         ivar   = -1;
   while ((branch = (TBranch*)branchIter.Next()) != 0) {
      TString branchName = branch->GetName();

      if (branchName != "type" && branchName != "weight" && branchName != "boostweight") {

         // determine mean and rms to obtain appropriate starting values
         ivar++;
         if ((*fInputVars)[ivar] != branchName) {
            fLogger << kWARNING << "<SanityChecks> mismatch in variables" << Endl;
            isOK = kFALSE;
         }
      }
   }  

   return isOK;
}

//_______________________________________________________________________
void  TMVA::MethodCuts::WriteWeightsToStream( ostream & o ) const
{
   // first the dimensions
   o << "OptimisationMethod " << "nbins:" << endl;
   o << ((fEffMethod == kUseEventSelection) ? "Fit-EventSelection" : 
         (fEffMethod == kUsePDFs) ? "Fit-PDF" : "Monte-Carlo") << "  " ;
   o << fNbins << endl;

   o << "Below are the optimised cuts for " << GetNvar() << " variables:"  << endl;
   o << "Format: ibin(hist) effS effB cutMin[ivar=0] cutMax[ivar=0]"
     << " ... cutMin[ivar=n-1] cutMax[ivar=n-1]" << endl;
   for (Int_t ibin=0; ibin<fNbins; ibin++) {
      o << setw(4) << ibin+1 << "  "    
        << setw(8)<< fEffBvsSLocal->GetBinCenter( ibin +1 ) << "  " 
        << setw(8)<< fEffBvsSLocal->GetBinContent( ibin +1 ) << "  ";  
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         o <<setw(10)<< fCutMin[ivar][ibin] << "  " << setw(10) << fCutMax[ivar][ibin] << "  ";
      o << endl;
   }
}

//_______________________________________________________________________
void  TMVA::MethodCuts::ReadWeightsFromStream( istream& istr )
{
   // read the cuts from stream
   TString dummy;
   UInt_t  dummyInt;

   // first the dimensions   
   istr >> dummy >> dummy;
   istr >> dummy >> fNbins;

   // get rid of one read-in here because we read in once all ready to check for decorrelation
   istr >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummyInt >> dummy ;
   
   // sanity check
   if (dummyInt != Data().GetNVariables()) {
      fLogger << kFATAL << "<ReadWeightsFromStream> fatal error: mismatch "
              << "in number of variables: " << dummyInt << " != " << Data().GetNVariables() << Endl;
   }
   SetNvar(dummyInt);

   // print some information
   if (fFitMethod == kUseMonteCarlo) {
      fLogger << kINFO << "Read cuts optimised using "<< fNRandCuts << " MC events" << Endl;
   }
   else if (fFitMethod == kUseGeneticAlgorithm) {
      fLogger << kINFO << "Read cuts optimised using Genetic Algorithm" << Endl;
   }
   else if (fFitMethod == kUseSimulatedAnnealing) {
      fLogger << kINFO << "Read cuts optimised using Si,ulated Annealing" << Endl;
   }
   else {
      fLogger << kWARNING << "unknown method: " << fFitMethod << Endl;
   }
   fLogger << kINFO << "in " << fNbins << " signal efficiency bins and for " << GetNvar() << " variables" << Endl;
   
   // now read the cuts
   char buffer[200];
   istr.getline(buffer,200);
   istr.getline(buffer,200);

   Int_t   tmpbin;
   Float_t tmpeffS, tmpeffB;
   if (fEffBvsSLocal!=0) delete fEffBvsSLocal;
   fEffBvsSLocal = new TH1F( GetTestvarName() + "_effBvsSLocal", 
                             TString(GetName()) + " efficiency of B vs S", fNbins, 0.0, 1.0 );

   for (Int_t ibin=0; ibin<fNbins; ibin++) {
      istr >> tmpbin >> tmpeffS >> tmpeffB;
      fEffBvsSLocal->SetBinContent( ibin+1, tmpeffB );

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         istr >> fCutMin[ivar][ibin] >> fCutMax[ivar][ibin];
      }
   }

   fEffSMin = fEffBvsSLocal->GetBinCenter(1);
   fEffSMax = fEffBvsSLocal->GetBinCenter(fNbins);
}

//_______________________________________________________________________
void TMVA::MethodCuts::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   fLogger << kINFO << "write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
  
   fEffBvsSLocal->Write();

   // save reference histograms to file
   if (fEffMethod == kUsePDFs) {
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) { 
         (*fVarHistS)[ivar]->Write();    
         (*fVarHistB)[ivar]->Write();
         (*fVarHistS_smooth)[ivar]->Write();    
         (*fVarHistB_smooth)[ivar]->Write();
         (*fVarPdfS)[ivar]->GetPDFHist()->Write();
         (*fVarPdfB)[ivar]->GetPDFHist()->Write();
      }
   }  
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::GetTrainingEfficiency( TString theString)
{
   // - overloaded function to create background efficiency (rejection) versus
   //   signal efficiency plot (first call of this function)
   // - the function returns the signal efficiency at background efficiency
   //   indicated in theString
   //
   // "theString" must have two entries:
   // [0]: "Efficiency"
   // [1]: the value of background efficiency at which the signal efficiency 
   //      is to be returned

   // parse input string for required background efficiency
   TList* list  = Tools::ParseFormatLine( theString );
   // sanity check
   if (list->GetSize() != 2) {
      fLogger << kFATAL << "<GetTrainingEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05" << Endl;
      return -1;
   }
   
   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   Bool_t firstPass = (NULL == fTrainEffBvsS || NULL == fTrainRejBvsS);

   // first round ? --> create histograms
   if (firstPass) {

      if (fBinaryTreeS != 0) delete fBinaryTreeS;
      if (fBinaryTreeB != 0) delete fBinaryTreeB;
      fBinaryTreeS = new BinarySearchTree();
      fBinaryTreeS->Fill( *this, Data().GetTrainingTree(), 1 );
      fBinaryTreeB = new BinarySearchTree();
      fBinaryTreeB->Fill( *this, Data().GetTrainingTree(), 0 );
      // there is no really good equivalent to the fEffS; fEffB (efficiency vs cutvalue)
      // for the "Cuts" method (unless we had only one cut). Maybe later I might add here
      // histograms for each of the cuts...but this would require also a change in the 
      // base class, and it is not really necessary, as we get exactly THIS info from the
      // "evaluateAllVariables" anyway.

      // now create efficiency curve: background versus signal
      if (NULL != fTrainEffBvsS) delete fTrainEffBvsS; 
      if (NULL != fTrainRejBvsS) delete fTrainRejBvsS; 
    
      fTrainEffBvsS = new TH1F( GetTestvarName() + "_trainingEffBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fTrainRejBvsS = new TH1F( GetTestvarName() + "_trainingRejBvsS", GetTestvarName() + "", fNbins, 0, 1 );

      // use root finder

      // make the background-vs-signal efficiency plot
      Double_t* tmpCutMin = new Double_t[GetNvar()];
      Double_t* tmpCutMax = new Double_t[GetNvar()];
      for (Int_t bini=1; bini<=fNbins; bini++) {
         for (Int_t ivar=0; ivar <GetNvar(); ivar++){
            tmpCutMin[ivar] = fCutMin[ivar][bini-1];
            tmpCutMax[ivar] = fCutMax[ivar][bini-1];
         }
         // find cut value corresponding to a given signal efficiency
         Double_t effS, effB;
         this->GetEffsfromSelection( &tmpCutMin[0], &tmpCutMax[0], effS, effB);    

         // and fill histograms
         fTrainEffBvsS->SetBinContent( bini, effB     );    
         fTrainRejBvsS->SetBinContent( bini, 1.0-effB ); 
      }

      delete[] tmpCutMin;
      delete[] tmpCutMax;

      // create splines for histogram
      fGraphTrainEffBvsS = new TGraph( fTrainEffBvsS );
      fSplTrainEffBvsS   = new TSpline1( "trainEffBvsS", fGraphTrainEffBvsS );
   }

   // must exist...
   if (NULL == fSplTrainEffBvsS) return 0.0;

   // now find signal efficiency that corresponds to required background efficiency
   Double_t effS, effB, effS_ = 0, effB_ = 0;
   Int_t    nbins_ = 1000;

   // loop over efficiency bins until the background eff. matches the requirement
   for (Int_t bini=1; bini<=nbins_; bini++) {
      // get corresponding signal and background efficiencies
      effS = (bini - 0.5)/Float_t(nbins_);
      effB = fSplTrainEffBvsS->Eval( effS );

      // find signal efficiency that corresponds to required background efficiency
      if ((effB - effBref)*(effB_ - effBref) < 0) break;
      effS_ = effS;
      effB_ = effB;  
   }

   return 0.5*(effS + effS_);
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::GetEfficiency( TString theString, TTree* theTree, Double_t& effSerr )
{
   // - overloaded function to create background efficiency (rejection) versus
   //   signal efficiency plot (first call of this function)
   // - the function returns the signal efficiency at background efficiency
   //   indicated in theString
   //
   // "theString" must have two entries:
   // [0]: "Efficiency"
   // [1]: the value of background efficiency at which the signal efficiency 
   //      is to be returned

   if (theTree == 0); // dummy 

   // parse input string for required background efficiency
   TList* list  = TMVA::Tools::ParseFormatLine( theString, ":" );

   // sanity check
   Bool_t computeArea = kFALSE;
   if      (!list || list->GetSize() < 2) computeArea = kTRUE; // the area is computed 
   else if (list->GetSize() > 2) {
      fLogger << kFATAL << "<GetEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05, or empty string" << Endl;
      return -1;
   }
   
   // first round ? --> create histograms
   if (fEffBvsS == NULL || fRejBvsS == NULL) {

      if (fBinaryTreeS!=0) delete fBinaryTreeS;
      if (fBinaryTreeB!=0) delete fBinaryTreeB;

      // the variables may be transformed by a transformation method: to coherently 
      // treat signal and background one must decide which transformation type shall 
      // be used: our default is signal-type
      fBinaryTreeS = new BinarySearchTree();
      fBinaryTreeS->Fill( *this, Data().GetTestTree(), 1 );
      fBinaryTreeB = new BinarySearchTree();
      fBinaryTreeB->Fill( *this, Data().GetTestTree(), 0 );

      // there is no really good equivalent to the fEffS; fEffB (efficiency vs cutvalue)
      // for the "Cuts" method (unless we had only one cut). Maybe later I might add here
      // histograms for each of the cuts...but this would require also a change in the 
      // base class, and it is not really necessary, as we get exactly THIS info from the
      // "evaluateAllVariables" anyway.

      // now create efficiency curve: background versus signal
      if (NULL != fEffBvsS)delete fEffBvsS; 
      if (NULL != fRejBvsS)delete fRejBvsS; 
    
      fEffBvsS = new TH1F( GetTestvarName() + "_effBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fRejBvsS = new TH1F( GetTestvarName() + "_rejBvsS", GetTestvarName() + "", fNbins, 0, 1 );

      // use root finder

      // make the background-vs-signal efficiency plot
      Double_t* tmpCutMin = new Double_t[GetNvar()];
      Double_t* tmpCutMax = new Double_t[GetNvar()];
      for (Int_t bini=1; bini<=fNbins; bini++) {
         for (Int_t ivar=0; ivar <GetNvar(); ivar++){
            tmpCutMin[ivar] = fCutMin[ivar][bini-1];
            tmpCutMax[ivar] = fCutMax[ivar][bini-1];
         }
         // find cut value corresponding to a given signal efficiency
         Double_t effS, effB;
         this->GetEffsfromSelection( &tmpCutMin[0], &tmpCutMax[0], effS, effB);    

         // and fill histograms
         fEffBvsS->SetBinContent( bini, effB     );    
         fRejBvsS->SetBinContent( bini, 1.0-effB ); 
      }

      delete[] tmpCutMin;
      delete[] tmpCutMax;

      // create splines for histogram
      fGrapheffBvsS = new TGraph( fEffBvsS );
      fSpleffBvsS   = new TSpline1( "effBvsS", fGrapheffBvsS );
   }

   // must exist...
   if (NULL == fSpleffBvsS) return 0.0;

   // now find signal efficiency that corresponds to required background efficiency
   Double_t effS = 0, effB = 0, effS_ = 0, effB_ = 0;
   Int_t    nbins_ = 1000;

   if (computeArea) {

      // compute area of rej-vs-eff plot
      Double_t integral = 0;
      for (Int_t bini=1; bini<=nbins_; bini++) {
         
         // get corresponding signal and background efficiencies
         effS = (bini - 0.5)/Float_t(nbins_);
         effB = fSpleffBvsS->Eval( effS );
         integral += (1.0 - effB);
      }   
      integral /= nbins_;      
      
      return integral;
   }
   else {

      // that will be the value of the efficiency retured (does not affect
      // the efficiency-vs-bkg plot which is done anyway.
      Float_t effBref = atof( ((TObjString*)list->At(1))->GetString() );      

      // loop over efficiency bins until the background eff. matches the requirement
      for (Int_t bini=1; bini<=nbins_; bini++) {
         // get corresponding signal and background efficiencies
         effS = (bini - 0.5)/Float_t(nbins_);
         effB = fSpleffBvsS->Eval( effS );
         
         // find signal efficiency that corresponds to required background efficiency
         if ((effB - effBref)*(effB_ - effBref) < 0) break;
         effS_ = effS;
         effB_ = effB;  
      }

      effS = 0.5*(effS + effS_);
      
      effSerr = 0;
      if (Data().GetNEvtSigTest() > 0) 
         effSerr = TMath::Sqrt( effS*(1.0 - effS)/Double_t(Data().GetNEvtSigTest()) );
   
      return effS;

   }

   return -1;
}
 
//_______________________________________________________________________
void TMVA::MethodCuts::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << endl;
   fout << "};" << endl;
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The optimisation of rectangular cuts performed by TMVA maximises " << Endl;
   fLogger << "the background rejection at given signal efficiency, and scans " << Endl;
   fLogger << "over the full range of the latter quantity. Three optimisation" << Endl;
   fLogger << "methods are optional: Monte Carlo sampling (MC), a Genetics Algo-," << Endl;
   fLogger << "rithm (GA), and Simulated Annealing (SA - depreciated at present). " << Endl;
   fLogger << "GA is expected to perform best." << Endl;
   fLogger << Endl;
   fLogger << "The difficulty to find the optimal cuts strongly increases with" << Endl;
   fLogger << "the dimensionality (number of input variables) of the problem." << Endl;
   fLogger << "This behavior is due to the non-uniqueness of the solution space."<<  Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "If the dimensionality exceeds, say, 4 input variables, it is " << Endl;
   fLogger << "advisable to scrutinize the separation power of the variables," << Endl;
   fLogger << "and to remove the weakest ones. If some among the input variables" << Endl;
   fLogger << "can be described by a single cut (e.g., because signal tends to be" << Endl;
   fLogger << "larger than background), this can be indicated to MethodCuts via" << Endl;
   fLogger << "the \"Fsmart\" options (see option string). Choosing this option" << Endl;
   fLogger << "reduces the number of requirements for the variable from 2 (min/max)" << Endl;
   fLogger << "to a single one (TMVA finds out whether it is to be interpreted as" << Endl;
   fLogger << "min or max)." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << "" << Endl;
   fLogger << "Monte Carlo sampling:" << Endl;
   fLogger << "" << Endl;
   fLogger << "Apart form the \"Fsmart\" option for the variables, the only way" << Endl;
   fLogger << "to improve the MC sampling is to increase the sampling rate. This" << Endl;
   fLogger << "is done via the configuration option \"MC_NRandCuts\". The execution" << Endl;
   fLogger << "time scales linearly with the sampling rate." << Endl;
   fLogger << "" << Endl;
   fLogger << "Genetic Algorithm:" << Endl;
   fLogger << "" << Endl;
   fLogger << "The algorithm terminates if no significant fitness increase has" << Endl;
   fLogger << "been achieved within the last \"nsteps\" steps of the calculation." << Endl;
   fLogger << "Wiggles in the ROC curve or constant background rejection of 1" << Endl;
   fLogger << "indicate that the GA failed to always converge at the true maximum" << Endl;
   fLogger << "fitness. In such a case, it is recommended to broaden the search " << Endl;
   fLogger << "by increasing the population size (\"popSize\") and to give the GA " << Endl;
   fLogger << "more time to find improvements by increasing the number of steps" << Endl;
   fLogger << "(\"nsteps\")" << Endl;
   fLogger << "  -> increase \"popSize\" (at least >10 * number of variables)" << Endl;
   fLogger << "  -> increase \"nsteps\"" << Endl;
   fLogger << "" << Endl;
}
