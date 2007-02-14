// @(#)root/tmva $Id: MethodCuts.cxx,v 1.10 2006/11/20 15:35:28 brun Exp $ 
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
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
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
#include "TDirectory.h"
#include "TH1F.h"
#include "TObjString.h"

#ifndef ROOT_TMVA_MethodCuts
#include "TMVA/MethodCuts.h"
#endif
#ifndef ROOT_TMVA_GeneticCuts
#include "TMVA/GeneticCuts.h"
#endif
#ifndef ROOT_TMVA_SimulatedAnnealingCuts
#include "TMVA/SimulatedAnnealingCuts.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif

ClassImp(TMVA::MethodCuts)
   ;

// init global variables
TMVA::MethodCuts* TMVA::MethodCuts::fgThisCuts = NULL;

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( TString jobName, TString methodTitle, DataSet& theData, 
                              TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{ 
   // standard constructor
   // ---------------------------------------------------------------------------------- 
   // format of option string: "OptMethod:EffMethod:Option_var1:...:Option_varn:Decorr"
   // "OptMethod" can be:
   //     - "GA"    : Genetic Algorithm (recommended)
   //     - "SA"    : Simulated Annealing
   //     - "MC"    : Monte-Carlo optimization 
   // "EffMethod" can be:
   //     - "EffSel": compute efficiency by event counting
   //     - "EffPDF": compute efficiency from PDFs
   // === For "GA" method ======
   // "Option_var1++" are (see GA for explanation of parameters):
   //     - fGA_nsteps        
   //     - fGA_cycles        
   //     - fGA_popSize
   //     - fGA_SC_steps        
   //     - fGA_SC_offsteps 
   //     - fGA_SC_factor   
   // === For "SA" method ======
   // "Option_var1++" are (see SA for explanation of parameters):
   //     - fSA_MaxCalls                
   //     - fSA_TemperatureGradient      
   //     - fSA_UseAdaptiveTemperature    
   //     - fSA_InitialTemperature        
   //     - fSA_MinTemperature        
   //     - fSA_Eps                       
   //     - fSA_NFunLoops                 
   //     - fSA_NEps                      
   // === For "MC" method ======
   // "Option_var1" is number of random samples
   // "Option_var2++" can be 
   //     - "FMax"  : ForceMax   (the max cut is fixed to maximum of variable i)
   //     - "FMin"  : ForceMin   (the min cut is fixed to minimum of variable i)
   //     - "FSmart": ForceSmart (the min or max cut is fixed to min/max, based on mean value)
   //     - Adding "All" to "option_vari", eg, "AllFSmart" will use this option for all variables
   //     - if "option_vari" is empty (== ""), no assumptions on cut min/max are made
   // "Decorr" can be:
   //     - omitted : Decorrelation not used
   //     - "D"     : Decorrelates variables, evaluation events decorrelated with signal decorrelation matrix
   //     - "DS"    : Decorrelates variables, evaluation events decorrelated with signal decorrelation matrix
   //     - "DB"    : Decorrelates variables, evaluation events decorrelated with background decorrelation matrix
   // ---------------------------------------------------------------------------------- 

   InitCuts();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( DataSet& theData, 
                              TString theWeightFile,  
                              TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
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
   SetMethodType( TMVA::Types::kCuts );  
   SetTestvarName();

   fConstrainType     = kConstrainEffS;
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

   // static pointer to this object
   fgThisCuts         = this;

   // vector with fit results
   fNpar      = 2*GetNvar();
   fRangeSign = new vector<Int_t>   ( GetNvar() );
   fMeanS     = new vector<Double_t>( GetNvar() ); 
   fMeanB     = new vector<Double_t>( GetNvar() ); 
   fRmsS      = new vector<Double_t>( GetNvar() );  
   fRmsB      = new vector<Double_t>( GetNvar() );  
   fXmin      = new vector<Double_t>( GetNvar() );  
   fXmax      = new vector<Double_t>( GetNvar() );  

   // get the variable specific options, first initialize default
   fFitParams = new vector<EFitParameters>( GetNvar() );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*fFitParams)[ivar] = kNotEnforced;

   fRandom    = new TRandom( 0 ); // set seed
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
   fLogger << kVERBOSE << "Destructor called" << Endl;

   delete fRangeSign;
   delete fRandom;
   delete fMeanS;
   delete fMeanB;
   delete fRmsS;
   delete fRmsB;
   delete fXmin;
   delete fXmax;  
   for (Int_t i=0;i<GetNvar();i++) {
      if (fCutMin[i] != NULL) delete [] fCutMin[i];
      if (fCutMax[i] != NULL) delete [] fCutMax[i];
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
   // MC_NRandCuts       <int>    Number of random cuts to estimate the efficiency for the MC method
   // MC_AllVarProp      <string> Property of all variables for the MC method
   //    available values are:        AllNotEnforced <default>
   //                                 AllFMax
   //                                 AllFMin
   //                                 AllFSmart
   //                                 AllFVerySmart
   // MC_Var1Prop        <string> Property of variable 1 for the MC method (taking precedence over the
   //    globale setting. The same values as for the global option are available. Variables 1..10 can be
   //    set this way
   //
   //
   // GA_nsteps           <int>   Number of steps for the genetic algorithm
   // GA_cycles           <int>   Number of generations for the genetic algorithm
   // GA_popSize          <int>   Size of the population for the genetic algorithm
   // GA_SC_steps         <int>   Number of steps for the genetic algorithm
   // GA_SC_offsteps      <int>    for the genetic algorithm
   // GA_SC_factor        <float>  for the genetic algorithm
   //
   //
   // SA_MaxCalls                 <int>      maximum number of calls for simulated annealing
   // SA_TemperatureGradient      <float>    temperature gradient for simulated annealing
   // SA_UseAdaptiveTemperature   <bool>     use of adaptive temperature for simulated annealing
   // SA_InitialTemperature       <float>    initial temperature for simulated annealing
   // SA_MinTemperature           <float>    minimum temperature for simulated annealing 
   // SA_Eps                      <int>      number of epochs for simulated annealing
   // SA_NFunLoops                <int>      number of loops for simulated annealing      
   // SA_NEps                     <int>      number of epochs for simulated annealing


   DeclareOptionRef(fFitMethodS="MC", "Method", "Minimization Method");
   AddPreDefVal(TString("GA"));
   AddPreDefVal(TString("SA"));
   AddPreDefVal(TString("MC"));

   // selection type
   DeclareOptionRef(fEffMethodS = "EffSel", "EffMethod", "Selection Method");
   AddPreDefVal(TString("EffSel"));
   AddPreDefVal(TString("EffPDF"));

   // MC options
   fNRandCuts         = 100000;
   DeclareOptionRef(fNRandCuts=100000,          "MC_NRandCuts", "");  

   DeclareOptionRef(fAllVars="AllNotEnforced",  "MC_AllVarProp", "");  
   AddPreDefVal(TString("AllNotEnforced"));
   AddPreDefVal(TString("AllFMax"));
   AddPreDefVal(TString("AllFMin"));
   AddPreDefVal(TString("AllFSmart"));
   AddPreDefVal(TString("AllFVerySmart"));
   for(int i=0; i<10; i++) {
      DeclareOptionRef(fAllVarsI[i]="NotEnforced", Form("MC_Var%iProp",i+1), "");  
      AddPreDefVal(TString("NotEnforced"));
      AddPreDefVal(TString("FMax"));
      AddPreDefVal(TString("FMin"));
      AddPreDefVal(TString("FSmart"));
      AddPreDefVal(TString("FVerySmart"));
   }

   // GA option
   fGA_cycles         = 3;
   fGA_SC_steps       = 10;
   fGA_popSize        = 100;
   fGA_SC_offsteps    = 5;
   fGA_SC_factor      = 0.95;
   fGA_nsteps         = 30;
   DeclareOptionRef(fGA_nsteps,      "GA_nsteps",      "");
   DeclareOptionRef(fGA_cycles,      "GA_cycles",      "");
   DeclareOptionRef(fGA_popSize,     "GA_popSize",     "");
   DeclareOptionRef(fGA_SC_steps,    "GA_SC_steps",    "");
   DeclareOptionRef(fGA_SC_offsteps, "GA_SC_offsteps", "");
   DeclareOptionRef(fGA_SC_factor,   "GA_SC_factor",   "");

   // SA options
   fSA_MaxCalls               = 5000000;
   fSA_TemperatureGradient    = 0.7;
   fSA_UseAdaptiveTemperature = kTRUE;
   fSA_InitialTemperature     = 100000;
   fSA_MinTemperature         = 500;
   fSA_Eps                    = 1e-04;
   fSA_NFunLoops              = 5;
   fSA_NEps                   = 4; // needs to be at leas 2 !
   DeclareOptionRef(fSA_MaxCalls,               "SA_MaxCalls", "");
   DeclareOptionRef(fSA_TemperatureGradient,    "SA_TemperatureGradient", ""); 
   DeclareOptionRef(fSA_UseAdaptiveTemperature, "SA_UseAdaptiveTemperature", "");  
   DeclareOptionRef(fSA_InitialTemperature,     "SA_InitialTemperature", "");  
   DeclareOptionRef(fSA_MinTemperature,         "SA_MinTemperature", "");
   DeclareOptionRef(fSA_Eps,                    "SA_Eps", "");  
   DeclareOptionRef(fSA_NFunLoops,              "SA_NFunLoops", "");  
   DeclareOptionRef(fSA_NEps,                   "SA_NEps", "");           
}

//_______________________________________________________________________
void TMVA::MethodCuts::ProcessOptions() 
{
   // process user options
   MethodBase::ProcessOptions();

   if      (fFitMethodS == "MC" ) fFitMethod = kUseMonteCarlo;
   else if (fFitMethodS == "GA" ) fFitMethod = kUseGeneticAlgorithm;
   else if (fFitMethodS == "SA" ) fFitMethod = kUseSimulatedAnnealing;
   else {
      fLogger << kFATAL << "unknown minimization method: " << fFitMethodS << Endl;
   }

   if      (fEffMethodS == "EFFSEL" ) fEffMethod = kUseEventSelection; // highly recommended
   else if (fEffMethodS == "EFFPDF" ) fEffMethod = kUsePDFs;
   else                               fEffMethod = kUseEventSelection;


   // options output
   fLogger << kINFO << Form("use optimization method: '%s'\n", 
                            (fFitMethod == kUseMonteCarlo) ? "Monte Carlo" : "Genetic Algorithm" );
   fLogger << kINFO << Form("use efficiency computation method: '%s'\n", 
                            (fEffMethod == kUseEventSelection) ? "Event Selection" : "PDF" );

   // -----------------------------------------------------------------------------------
   // interpret for MC use  
   //
   if (fFitMethod == kUseMonteCarlo) {
      if (fNRandCuts <= 1) {
         fLogger << kFATAL << "invalid number of MC events: " << fNRandCuts << Endl;
      }
    
      fLogger << kINFO << "generate " << fNRandCuts << " random cut samples" << Endl;
  
      if (fAllVars!="AllNotEnforced") { // options are specified

         EFitParameters theFitP = kNotEnforced;
         if      (fAllVars == "AllNotEnforced") theFitP = kNotEnforced;
         else if (fAllVars == "AllFMax"       ) theFitP = kForceMax;
         else if (fAllVars == "AllFMin"       ) theFitP = kForceMin;
         else if (fAllVars == "AllFSmart"     ) theFitP = kForceSmart;
         else if (fAllVars == "AllFVerySmart" ) theFitP = kForceVerySmart;
         else {
            fLogger << kFATAL << "unknown value \'" << fAllVars
                    << "\' for fit parameter option MC_AllVarProp" << Endl;
         }
         for (Int_t ivar=0; ivar<GetNvar(); ivar++) (*fFitParams)[ivar] = theFitP;
         if (theFitP != kNotEnforced) fLogger << "use 'smart' cuts" << Endl;

      } 
      else { // individual options

         int maxVar = GetNvar()<=10?GetNvar():10;
         for (Int_t ivar=0; ivar<maxVar; ivar++) {
            EFitParameters theFitP = kNotEnforced;
            if (fAllVarsI[ivar] == "" || fAllVarsI[ivar] == "NotEnforced") theFitP = kNotEnforced;
            else if (fAllVarsI[ivar] == "FMax" )                           theFitP = kForceMax;
            else if (fAllVarsI[ivar] == "FMin" )                           theFitP = kForceMin;
            else if (fAllVarsI[ivar] == "FSmart" )                         theFitP = kForceSmart;
            else if (fAllVarsI[ivar] == "FVerySmart" )                     theFitP = kForceVerySmart;
            else {
               fLogger << kFATAL << "unknown value \'" << fAllVarsI[ivar]
                       << "\' for fit parameter option " << Form("MC_Var%iProp",ivar+1) << Endl;
            }
            (*fFitParams)[ivar] = theFitP;
            
            if (theFitP != kNotEnforced) 
               fLogger << kINFO << "use 'smart' cuts for variable: " 
                       << "'" << (*fInputVars)[ivar] << "'" << Endl;
         }        
      }
      
      fLogger << kINFO << Form("number of MC events to be generated: %i\n", fNRandCuts );
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         TString theFitOption = ( ((*fFitParams)[ivar] == kNotEnforced) ? "NotEnforced" :
                                  ((*fFitParams)[ivar] == kForceMin   ) ? "ForceMin"    :
                                  ((*fFitParams)[ivar] == kForceMax   ) ? "ForceMax"    :
                                  ((*fFitParams)[ivar] == kForceSmart ) ? "ForceSmart"  :
                                  ((*fFitParams)[ivar] == kForceVerySmart ) ? "ForceVerySmart"  : "other" );
         
         fLogger << kINFO << Form("option for variable: %s: '%s' (#: %i)\n",
                                  (const char*)(*fInputVars)[ivar], (const char*)theFitOption, 
                                  (Int_t)(*fFitParams)[ivar] );
      }

   }

   // decorrelate option will be last option, if it is specified
   if      (GetPreprocessingMethod() == Types::kDecorrelated)
      fLogger << kINFO << "use decorrelated variable set" << Endl;
   else if (GetPreprocessingMethod() == Types::kPCA)
      fLogger << kINFO << "use principal component preprocessing" << Endl;
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
         passed &= ( (Data().Event().GetVal(ivar) >= fCutMin[ivar][ibin]) && 
                     (Data().Event().GetVal(ivar) <= fCutMax[ivar][ibin]) );

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

   // get background efficiency for which the signal efficiency
   // ought to be maximized
   fConstrainType = kConstrainEffS;

   // create binary trees (global member variables) for signal and background
   if (fBinaryTreeS != 0) delete fBinaryTreeS;
   if (fBinaryTreeB != 0) delete fBinaryTreeB;

   // the variables may be transformed by a preprocessing method: to coherently 
   // treat signal and background one must decide which preprocessing type shall 
   // be used: our default is signal-type
   fBinaryTreeS = new TMVA::BinarySearchTree();
   fBinaryTreeS->Fill( Data(), Data().GetTrainingTree(), 1, GetPreprocessingMethod(), GetPreprocessingType() );
   fBinaryTreeB = new TMVA::BinarySearchTree();
   fBinaryTreeB->Fill( Data(), Data().GetTrainingTree(), 0, GetPreprocessingMethod(), GetPreprocessingType() );

   vector<TH1F*> signalDist, bkgDist;

   // get statistics and variable distributions
   for (UInt_t ivar = 0; ivar < Data().GetNVariables(); ivar++) {

      const TString& varname = Data().GetInternalVarName(ivar);

      Statistics( TMVA::Types::kTraining, varname,
                  (*fMeanS)[ivar], (*fMeanB)[ivar], 
                  (*fRmsS)[ivar], (*fRmsB)[ivar], 
                  (*fXmin)[ivar], (*fXmax)[ivar] );            
   }

   // this is important: reset the branch addresses of the training tree to the current event
   Data().ResetCurrentTree();

   // determine eff(B) versus eff(S) plot
   fConstrainType = kConstrainEffS;

   Int_t ibin=0;
   fEffBvsSLocal = new TH1F( GetTestvarName() + "_effBvsSLocal", 
                             TString(GetName()) + " efficiency of B vs S", fNbins, 0.0, 1.0 );

   // init
   for (ibin=1; ibin<=fNbins; ibin++) fEffBvsSLocal->SetBinContent( ibin, -0.1 );

   // --------------------------------------------------------------------------
   if (fFitMethod == kUseMonteCarlo) {
    
      // generate MC cuts
      Double_t* cutMin = new Double_t[GetNvar()];
      Double_t* cutMax = new Double_t[GetNvar()];
    
      // MC loop
      fLogger << kINFO << "Generating " << fNRandCuts 
              << " cycles (random cuts) in " << GetNvar() << " variables ... patience please" << Endl;

      Int_t nBinsFilled=0, nBinsFilledAt=0;

      // timing of MC
      TMVA::Timer timer( fNRandCuts, GetName() ); 

      for (Int_t imc=0; imc<fNRandCuts; imc++) {

         // generate random cuts
         for (Int_t ivar=0; ivar<GetNvar(); ivar++) {

            EFitParameters fitParam = (*fFitParams)[ivar];

            if (fitParam == kForceSmart) {
               if ((*fMeanS)[ivar] > (*fMeanB)[ivar]) fitParam = kForceMax;
               else                                   fitParam = kForceMin;          
            }

            if (fitParam == kForceMin) 
               cutMin[ivar] = (*fXmin)[ivar];
            else
               cutMin[ivar] = fRandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];

            if (fitParam == kForceMax) 
               cutMax[ivar] = (*fXmax)[ivar];
            else
               cutMax[ivar] = fRandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
        
            if (fitParam == kForceVerySmart){
               // generate random cut parameters gaussian distrubuted around the variable values
               // where the difference between signal and background is maximal
          
               // get the variable distributions:
               cutMin[ivar] = fRandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];
               cutMax[ivar] = fRandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
               // ..... to be continued (Helge)
            }

            if (cutMax[ivar] < cutMin[ivar]) {
               fLogger << kFATAL << "<Train>: mismatch with cuts" << Endl;
            }
         }

         // event loop
         Double_t effS = 0, effB = 0;
         GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
    
         // determine bin
         Int_t    ibinS = (Int_t)(effS*Float_t(fNbins) + 1);
         if (ibinS < 1     ) ibinS = 1;
         if (ibinS > fNbins) ibinS = fNbins;
      
         // linear extrapolation 
         // (not done at present --> MC will be slightly biased !
         //  the bias increases with the bin width)
         Double_t effBH = fEffBvsSLocal->GetBinContent( ibinS );

         // preliminary best event -> backup
         if (effBH < 0 || effBH > effB) {
            fEffBvsSLocal->SetBinContent( ibinS, effB );
            for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
               fCutMin[ivar][ibinS-1] = cutMin[ivar]; // bin 1 stored in index 0
               fCutMax[ivar][ibinS-1] = cutMax[ivar];
            }
         }

         // some output to make waiting less boring
         Int_t nout = 1000;
         if ((Int_t)imc%nout == 0  || imc == fNRandCuts-1) {
            Int_t nbinsF = 0, ibin_;
            for (ibin_=0; ibin_<fNbins; ibin_++)
               if (fEffBvsSLocal->GetBinContent( ibin_ +1   ) >= 0) nbinsF++;
            if (nBinsFilled!=nbinsF) {
               nBinsFilled = nbinsF;
               nBinsFilledAt = imc;
            }
        
            timer.DrawProgressBar( imc );
            if (imc == fNRandCuts-1 ) 
               fLogger << kINFO << Form( "fraction of efficiency bins filled: %3.1f              ",
                                         nbinsF/Float_t(fNbins) ) << Endl;
         }
      } // end of MC loop

      fLogger << kVERBOSE << "fraction of filled eff. bins did not increase" 
              << " anymore after "<< nBinsFilledAt << " cycles" << Endl;

      // get elapsed time
      fLogger << kINFO << "elapsed time: " << timer.GetElapsedTime() 
              << "                            " << Endl;  

      delete[] cutMin;
      delete[] cutMax;

   }
   // --------------------------------------------------------------------------
   else if (fFitMethod == kUseGeneticAlgorithm) {

      // ranges
      vector<LowHigh_t*> ranges;
    
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         (*fRangeSign)[ivar] = +1;    
         ranges.push_back( new LowHigh_t( (*fXmin)[ivar], (*fXmax)[ivar] ) );
         ranges.push_back( new LowHigh_t( 0, (*fXmax)[ivar] - (*fXmin)[ivar] ) );
      }

      fLogger << kINFO << "GA: calculation, please be patient ..." << Endl;

      // timing of MC
      TMVA::Timer timer1( fGA_cycles, GetName() ); 

      // precalculation
      for (Int_t cycle = 0; cycle < fGA_cycles; cycle++) {

         timer1.DrawProgressBar( cycle );

         // ---- perform series of fits to achieve best convergence
         
         // "m_ga_spread" times the number of variables
         TMVA::GeneticCuts ga( fGA_popSize, ranges, this ); 
    
         ga.CalculateFitness();
         ga.GetGeneticPopulation().TrimPopulation();
    
         do {
            ga.Init();
            ga.CalculateFitness();
            ga.SpreadControl( fGA_SC_steps, fGA_SC_offsteps, fGA_SC_factor );
         } while (!ga.HasConverged( fGA_nsteps, 0.0001 ));                
      }

      // get elapsed time
      fLogger << kINFO << "GA: elapsed time: " << timer1.GetElapsedTime() 
              << "                            " << Endl;  
   }
   // --------------------------------------------------------------------------
   else if (fFitMethod == kUseSimulatedAnnealing) {

      // ranges
      vector<LowHigh_t*> ranges;
      vector<Double_t>   par;
    
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         (*fRangeSign)[ivar] = +1;    
         ranges.push_back( new LowHigh_t( (*fXmin)[ivar], (*fXmax)[ivar] ) );
         ranges.push_back( new LowHigh_t( 0, (*fXmax)[ivar] - (*fXmin)[ivar] ) );

         // central values of parameters
         par.push_back( (ranges[2*ivar]->first   + ranges[2*ivar]->second)/2.0 );
         par.push_back( (ranges[2*ivar+1]->first + ranges[2*ivar+1]->second)/2.0 );
      }

      TMVA::SimulatedAnnealingCuts saCuts( ranges );

      // set driving parameters
      saCuts.SetMaxCalls    ( fSA_MaxCalls );              
      saCuts.SetTempGrad    ( fSA_TemperatureGradient );   
      saCuts.SetUseAdaptTemp( fSA_UseAdaptiveTemperature );
      saCuts.SetInitTemp    ( fSA_InitialTemperature );    
      saCuts.SetMinTemp     ( fSA_MinTemperature );
      saCuts.SetNumFunLoops ( fSA_NFunLoops );                   
      saCuts.SetAccuracy    ( fSA_Eps );             
      saCuts.SetNEps        ( fSA_NEps );                  

      fLogger << kINFO << "SA: entree, please be patient ..." << Endl;

      // timing of SA
      TMVA::Timer timer( fNbins, GetName() ); 

      Double_t* cutMin = new Double_t[GetNvar()];
      Double_t* cutMax = new Double_t[GetNvar()];      
      for (ibin=1; ibin<=fNbins; ibin++) {

         timer.DrawProgressBar( ibin );

         fEffRef = fEffBvsSLocal->GetBinCenter( ibin );

         Double_t effS = 0, effB = 0;
         this->MatchParsToCuts     ( par, &cutMin[0], &cutMax[0] );
         this->GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
      
         for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
            fCutMin[ivar][ibin-1] = cutMin[ivar]; // bin 1 stored in index 0
            fCutMax[ivar][ibin-1] = cutMax[ivar];
         }
      }
      delete [] cutMin;
      delete [] cutMax;

      // get elapsed time
      
      fLogger << kINFO << "SA: elapsed time: " << timer.GetElapsedTime() 
              << "                            " << Endl;  

   }
   // --------------------------------------------------------------------------
   else fLogger << kFATAL << "unknown minization method: " << fFitMethod << Endl;

   if (fBinaryTreeS != 0) { delete fBinaryTreeS; fBinaryTreeS = 0; }
   if (fBinaryTreeB != 0) { delete fBinaryTreeB; fBinaryTreeB = 0; }
}

void TMVA::MethodCuts::Test( TTree* )
{
   // not used 
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::ComputeEstimator( const std::vector<Double_t>& par )
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
   
   
   // test for a fitnessfunction which optimizes on the whole background-rejection signal-efficiency plot
   
   // get the backg-reject. and sig-eff for the parameters given to this function
   // effS, effB
      
   // get the "best signal eff" for the backg-reject.
   // determine bin
   Int_t    ibinS = (Int_t)(effS*Float_t(fNbins) + 1);
   if (ibinS < 1     ) ibinS = 1;
   if (ibinS > fNbins) ibinS = fNbins;
      
   Double_t effBH = fEffBvsSLocal->GetBinContent( ibinS );

   eta = ( 1. - (effBH - effB) ) / (1+effS);


   // if a point is found which is better than an existing one, ... replace it. 
   // preliminary best event -> backup
   if (effBH < 0 || effBH > effB) {
      fEffBvsSLocal->SetBinContent( ibinS, effB );
      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         fCutMin[ivar][ibinS-1] = fTmpCutMin[ivar]; // bin 1 stored in index 0
         fCutMax[ivar][ibinS-1] = fTmpCutMax[ivar];
      }
   }
   
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
void TMVA::MethodCuts::MatchCutsToPars( Double_t* par, 
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
      
   TMVA::Volume* volume = new TMVA::Volume( cutMin, cutMax, GetNvar() );
  
   nSelS = fBinaryTreeS->SearchVolume( volume );
   nSelB = fBinaryTreeB->SearchVolume( volume );
  
   nTotS = Float_t(fBinaryTreeS->GetSumOfWeights());
   nTotB = Float_t(fBinaryTreeB->GetSumOfWeights());
    
   delete volume;

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
   else if ( nTotB == 0) {
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
   fVarHistS        = new vector<TH1*>    ( GetNvar() );
   fVarHistB        = new vector<TH1*>    ( GetNvar() );
   fVarHistS_smooth = new vector<TH1*>    ( GetNvar() );
   fVarHistB_smooth = new vector<TH1*>    ( GetNvar() );
   fVarPdfS         = new vector<TMVA::PDF*>( GetNvar() );
   fVarPdfB         = new vector<TMVA::PDF*>( GetNvar() );

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
      (*fVarPdfS)[ivar] = new TMVA::PDF( (*fVarHistS_smooth)[ivar], TMVA::PDF::kSpline2 );
      (*fVarPdfB)[ivar] = new TMVA::PDF( (*fVarHistB_smooth)[ivar], TMVA::PDF::kSpline2 );
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
   o << "OptimisationMethod " << "nRandCuts " << "nbins:" << endl;
   o << ((fEffMethod == kUseEventSelection) ? "Fit-EventSelection" : 
         (fEffMethod == kUsePDFs) ? "Fit-PDF" : "Monte-Carlo") << "  " ;
   o << fNRandCuts << "  ";
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
   istr >> dummy >> dummy >> dummy;
   istr >> dummy >> fNRandCuts >> fNbins;

   // get rid of one read-in here because we read in once all ready to check for decorrelation
   istr >> dummy >> dummy >> dummy >> dummy >> dummy >> dummy >> dummyInt >> dummy ;
   
   // sanity check
   if (dummyInt != Data().GetNVariables()) {
      fLogger << kFATAL << "<ReadWeightsFromStream> fatal error: mismatch "
              << "in number of variables: " << dummyInt << " != " << Data().GetNVariables() << Endl;
   }
   SetNvar(dummyInt);

   fLogger << kINFO << "Read cuts from "<< fNRandCuts << " MC events"
           << " in " << fNbins << " efficiency bins and for " << GetNvar() << " variables" << Endl;
   
   char buffer[200];
   istr.getline(buffer,200);
   istr.getline(buffer,200);

   // read histogram and cuts
   Int_t   tmpbin;
   Float_t tmpeffS, tmpeffB;
   for (Int_t ibin=0; ibin<fNbins; ibin++) {
      istr >> tmpbin >> tmpeffS >> tmpeffB;

      if (ibin == 0       ) fEffSMin = tmpeffS;
      if (ibin == fNbins-1) fEffSMax = tmpeffS;

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
         istr >> fCutMin[ivar][ibin] >> fCutMax[ivar][ibin];
      }
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::WriteMonitoringHistosToFile( void ) const
{
   // write histograms and PDFs to file for monitoring purposes

   fLogger << kINFO << "write monitoring histograms to file: " << BaseDir()->GetPath() << Endl;
  
   BaseDir()->cd();

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
   TList* list  = TMVA::Tools::ParseFormatLine( theString );
   // sanity check
   if (list->GetSize() != 2) {
      fLogger << kWARNING << "<GetTrainingEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05" << Endl;
      return -1;
   }
   
   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   fLogger << kVERBOSE << "<GetTrainingEfficiency> compute eff(S) at eff(B) = " 
           << effBref << Endl;
   
   Bool_t firstPass = (NULL == fTrainEffBvsS || NULL == fTrainRejBvsS);

   // first round ? --> create histograms
   if (firstPass) {

      if (fBinaryTreeS != 0) delete fBinaryTreeS;
      if (fBinaryTreeB != 0) delete fBinaryTreeB;
      fBinaryTreeS = new TMVA::BinarySearchTree();
      fBinaryTreeS->Fill( Data(), Data().GetTrainingTree(), 1 );
      fBinaryTreeB = new TMVA::BinarySearchTree();
      fBinaryTreeB->Fill( Data(), Data().GetTrainingTree(), 0 );
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
      fSplTrainEffBvsS   = new TMVA::TSpline1( "trainEffBvsS", fGraphTrainEffBvsS );
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
Double_t TMVA::MethodCuts::GetEfficiency( TString theString, TTree* /*theTree*/ )
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
   TList* list  = TMVA::Tools::ParseFormatLine( theString );
   // sanity check
   if (list->GetSize() != 2) {
      fLogger << kWARNING << "<GetEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05" << Endl;
      return -1;
   }
   
   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   fLogger << kVERBOSE << "<GetEfficiency> compute eff(S) at eff(B) = " << effBref << Endl;

   // first round ? --> create histograms
   if ( fEffBvsS == NULL ||   fRejBvsS == NULL) {

      if (fBinaryTreeS!=0) delete fBinaryTreeS;
      if (fBinaryTreeB!=0) delete fBinaryTreeB;

      // the variables may be transformed by a preprocessing method: to coherently 
      // treat signal and background one must decide which preprocessing type shall 
      // be used: our default is signal-type
      fBinaryTreeS = new TMVA::BinarySearchTree();
      fBinaryTreeS->Fill( Data(), Data().GetTestTree(), 1, GetPreprocessingMethod(), GetPreprocessingType() );
      fBinaryTreeB = new TMVA::BinarySearchTree();
      fBinaryTreeB->Fill( Data(), Data().GetTestTree(), 0, GetPreprocessingMethod(), GetPreprocessingType() );

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
      fSpleffBvsS   = new TMVA::TSpline1( "effBvsS", fGrapheffBvsS );
   }

   // must exist...
   if (NULL == fSpleffBvsS) return 0.0;

   // now find signal efficiency that corresponds to required background efficiency
   Double_t effS, effB, effS_ = 0, effB_ = 0;
   Int_t    nbins_ = 1000;

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

   return 0.5*(effS + effS_);
}
 
