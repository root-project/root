// @(#)root/tmva $Id$
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
/* Begin_Html
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

End_Html */
//_______________________________________________________________________

#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include "time.h"
#include "Riostream.h"
#include "TH1F.h"
#include "TObjString.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TGraph.h"
#include "TSpline.h"
#include "TRandom3.h"

#include "TMVA/MethodCuts.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/MinuitFitter.h"
#include "TMVA/MCFitter.h"
#include "TMVA/Tools.h"
#include "TMVA/Timer.h"
#include "TMVA/Interval.h"
#include "TMVA/TSpline1.h"
#include "TMVA/SimulatedAnnealingFitter.h"
#include "TMVA/Config.h"

ClassImp(TMVA::MethodCuts)

const Double_t TMVA::MethodCuts::fgMaxAbsCutVal = 1.0e30;

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( const TString& jobName, const TString& methodTitle, DataSet& theData, 
                              const TString& theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
   , fCutRangeMin(0)
   , fCutRangeMax(0)
   , fAllVarsI(0)  
{ 
   // standard constructor
   // see below for option string format

   InitCuts();

   // interpretation of configuration option string
   SetConfigName( TString("Method") + GetMethodName() );
   DeclareOptions();
   ParseOptions();
   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodCuts::MethodCuts( DataSet& theData, 
                              const TString& theWeightFile,  
                              TDirectory* theTargetDir )
   : MethodBase( theData, theWeightFile, theTargetDir ) 
   , fCutRangeMin(0)
   , fCutRangeMax(0)
   , fAllVarsI(0)  
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
   delete fFitParams;

   if (fCutRangeMin) delete[] fCutRangeMin;
   if (fCutRangeMax) delete[] fCutRangeMax;
   if (fAllVarsI)    delete[] fAllVarsI;

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

   DeclareOptionRef(fFitMethodS = "GA", "FitMethod", "Minimization Method (GA, SA, and MC are the primary methods to be used; the others have been introduced for testing purposes and are depreciated)");
   AddPreDefVal(TString("GA"));
   AddPreDefVal(TString("SA"));
   AddPreDefVal(TString("MC"));
   AddPreDefVal(TString("MCEvents"));
   AddPreDefVal(TString("MINUIT"));
   AddPreDefVal(TString("EventScan"));

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

   // sanity check, do not allow the input variables to be normalised, because this 
   // only creates problems when interpreting the cuts
   if (IsNormalised()) {
      fLogger << kWARNING << "Normalisation of the input variables for cut optimisation is not" << Endl;
      fLogger << kWARNING << "supported because this provides intransparent cut values, and no" << Endl;
      fLogger << kWARNING << "improvement in the performance of the algorithm." << Endl;
      fLogger << kWARNING << "Please remove \"Normalise\" option from booking option string" << Endl;
      fLogger << kWARNING << "==> Will reset normalisation flag to \"False\"" << Endl;
      SetNormalised( kFALSE );
   }

   if      (fFitMethodS == "MC"      ) fFitMethod = kUseMonteCarlo;
   else if (fFitMethodS == "MCEvents") fFitMethod = kUseMonteCarloEvents;
   else if (fFitMethodS == "GA"      ) fFitMethod = kUseGeneticAlgorithm;
   else if (fFitMethodS == "SA"      ) fFitMethod = kUseSimulatedAnnealing;
   else if (fFitMethodS == "MINUIT"  ) {
      fFitMethod = kUseMinuit;
      fLogger << kWARNING << "poor performance of MINUIT in MethodCuts; preferred fit method: GA" << Endl;
   }
   else if (fFitMethodS == "EventScan" ) fFitMethod = kUseEventScan;
   else fLogger << kFATAL << "unknown minimization method: " << fFitMethodS << Endl;

   if      (fEffMethodS == "EFFSEL" ) fEffMethod = kUseEventSelection; // highly recommended
   else if (fEffMethodS == "EFFPDF" ) fEffMethod = kUsePDFs;
   else                               fEffMethod = kUseEventSelection;

   // options output
   fLogger << kINFO << Form("Use optimization method: '%s'\n", 
                            (fFitMethod == kUseMonteCarlo) ? "Monte Carlo" : 
                            (fFitMethod == kUseMonteCarlo) ? "Monte-Carlo-Event sampling" : 
                            (fFitMethod == kUseEventScan)  ? "Full Event Scan (slow)" :
                            (fFitMethod == kUseMinuit)     ? "MINUIT" : "Genetic Algorithm" );
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
   if (fFitMethod == kUseMonteCarlo || fFitMethod == kUseMonteCarloEvents || fFitMethod == kUseGeneticAlgorithm) {
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
   if      (GetVariableTransform() == Types::kDecorrelated) {
      fLogger << kINFO << "Use decorrelated variable set" << Endl;
   }
   else if (GetVariableTransform() == Types::kPCA) {
      fLogger << kINFO << "Use principal component transformation" << Endl;
   }
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
      Int_t ibin = fEffBvsSLocal->FindBin( fTestSignalEff );
      if      (ibin < 0      ) ibin = 0;
      else if (ibin >= fNbins) ibin = fNbins - 1;
    
      Bool_t passed = kTRUE;
      for (Int_t ivar=0; ivar<GetNvar(); ivar++)
         passed &= ( (GetEventVal(ivar) >  fCutMin[ivar][ibin]) && 
                     (GetEventVal(ivar) <= fCutMax[ivar][ibin]) );

      return passed ? 1. : 0. ;
   }
   else return 0;
}

//_______________________________________________________________________
void TMVA::MethodCuts::PrintCuts( Double_t effS ) const
{
   // print cuts

   std::vector<Double_t> cutsMin;
   std::vector<Double_t> cutsMax;
   Int_t ibin = fEffBvsSLocal->FindBin( effS );
   
   GetCuts( effS, cutsMin, cutsMax );

   // retrieve variable expressions (could be transformations)
   std::vector<TString>* varVec = GetVarTransform().GetTransformationStrings();
   if (!varVec) fLogger << kFATAL << "Big troubles in \"PrintCuts\": zero varVec pointer" << Endl;

   UInt_t maxL = 0;
   for (UInt_t ivar=0; ivar<cutsMin.size(); ivar++) {
      if ((UInt_t)(*varVec)[ivar].Length() > maxL) maxL = (*varVec)[ivar].Length();
   }
   UInt_t maxLine = 20+maxL+16;

   for (UInt_t i=0; i<maxLine; i++) fLogger << "-";
   fLogger << Endl;
   fLogger << kINFO << "Cut values for requested signal efficiency: " << effS << Endl;
   fLogger << kINFO << "Corresponding background efficiency       : " << fEffBvsSLocal->GetBinContent( ibin +1 ) 
           << ")" << Endl;
   if (GetVariableTransform() != Types::kNone) {
      fLogger << kINFO << "NOTE: The cuts are applied to TRANFORMED variables (see explicit transformation below)" << Endl;
      fLogger << kINFO << "      Name of the transformation: \"" << GetVarTransform().GetName() << "\"" << Endl;
   }
   for (UInt_t i=0; i<maxLine; i++) fLogger << "-";
   fLogger << Endl;
   for (UInt_t ivar=0; ivar<cutsMin.size(); ivar++) {
      fLogger << kINFO 
              << "Cut[" << setw(2) << ivar << "]: " 
              << setw(10) << cutsMin[ivar] 
              << " < " 
              << setw(maxL) << (*varVec)[ivar]
              << " <= " 
              << setw(10) << cutsMax[ivar] << Endl;
   }
   for (UInt_t i=0; i<maxLine; i++) fLogger << "-";
   fLogger << Endl;

   delete varVec; // yes, ownership has been given to us
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetCuts( Double_t effS, Double_t* cutMin, Double_t* cutMax ) const
{
   // retrieve cut values for given signal efficiency
   // assume vector of correct size !!

   std::vector<Double_t> cMin( GetNvar() );
   std::vector<Double_t> cMax( GetNvar() );
   GetCuts( effS, cMin, cMax );
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      cutMin[ivar] = cMin[ivar];
      cutMax[ivar] = cMax[ivar];
   }   
}

//_______________________________________________________________________
void TMVA::MethodCuts::GetCuts( Double_t effS, 
                                std::vector<Double_t>& cutMin, 
                                std::vector<Double_t>& cutMax ) const
{
   // retrieve cut values for given signal efficiency

   // find corresponding bin
   Int_t ibin = fEffBvsSLocal->FindBin( effS );
   if      (ibin < 0      ) ibin = 0;
   else if (ibin >= fNbins) ibin = fNbins - 1;

   cutMin.clear();
   cutMax.clear();
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      cutMin.push_back( fCutMin[ivar][ibin]  );
      cutMax.push_back( fCutMax[ivar][ibin] );
   }   
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

      // redefine ranges to be slightly smaller and larger than xmin and xmax, respectively
      Double_t eps = 0.01*(xmax - xmin);
      xmin -= eps;
      xmax += eps;

      if (TMath::Abs(fCutRange[ivar]->GetMin() - fCutRange[ivar]->GetMax()) < 1.0e-300 ) {
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
   if (fFitMethod == kUseGeneticAlgorithm || 
       fFitMethod == kUseMonteCarlo       || 
       fFitMethod == kUseMinuit           || 
       fFitMethod == kUseSimulatedAnnealing) {

      // ranges
      vector<Interval*> ranges;

      for (Int_t ivar=0; ivar<GetNvar(); ivar++) {

         Int_t nbins = 0;
         if (Data().GetVarType(ivar) == 'I') {
            nbins = Int_t(fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin()) + 1;
         }

         if ((*fFitParams)[ivar] == kForceSmart) {
            if ((*fMeanS)[ivar] > (*fMeanB)[ivar]) (*fFitParams)[ivar] = kForceMax;
            else                                   (*fFitParams)[ivar] = kForceMin;          
         }         

         if ((*fFitParams)[ivar] == kForceMin){
            ranges.push_back( new Interval( fCutRange[ivar]->GetMin(), fCutRange[ivar]->GetMin(), nbins ) );
            ranges.push_back( new Interval( 0, fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), nbins ) );
         }
         else if ((*fFitParams)[ivar] == kForceMax){
            ranges.push_back( new Interval( fCutRange[ivar]->GetMin(), fCutRange[ivar]->GetMax(), nbins ) );
            ranges.push_back( new Interval( fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), 
                                            fCutRange[ivar]->GetMax() - fCutRange[ivar]->GetMin(), nbins ) );
         }
         else {
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
      case kUseSimulatedAnnealing:
         fitter = new SimulatedAnnealingFitter( *this, Form("%sFitter_SA", GetName()), ranges, GetOptions() );
         break;
      default:
         fLogger << kFATAL << "Wrong fit method: " << fFitMethod << Endl;
      }

      fitter->CheckForUnusedOptions();

      // perform the fit
      fitter->Run();      

      // clean up
      for (UInt_t ivar=0; ivar<ranges.size(); ivar++) delete ranges[ivar];
   }
   // --------------------------------------------------------------------------
   else if (fFitMethod == kUseEventScan) {

      Int_t nevents = Data().GetNEvtTrain();
      Int_t ic = 0;

      // timing of MC
      Int_t nsamples = Int_t(0.5*nevents*(nevents - 1));
      Timer timer( nsamples, GetName() ); 

      fLogger << kINFO << "Running full event scan: " << Endl;
      for (Int_t ievt1=0; ievt1<nevents; ievt1++) {
         for (Int_t ievt2=ievt1+1; ievt2<nevents; ievt2++) {

            EstimatorFunction( ievt1, ievt2 );

            // what's the time please?
            ic++;
            if ((nsamples<10000) || ic%10000 == 0) timer.DrawProgressBar( ic );
         }
      }
   }
   // --------------------------------------------------------------------------
   else if (fFitMethod == kUseMonteCarloEvents) {

      Int_t  nsamples = 200000;
      UInt_t seed     = 100;
      DeclareOptionRef( nsamples, "SampleSize", "Number of Monte-Carlo-Event samples" );  
      DeclareOptionRef( seed,     "Seed",       "Seed for the random generator (0 takes random seeds)" );  
      ParseOptions();

      Int_t nevents = Data().GetNEvtTrain();
      Int_t ic = 0;

      // timing of MC
      Timer timer( nsamples, GetName() ); 

      // random generator
      TRandom *rnd = new TRandom3( seed );

      fLogger << kINFO << "Running Monte-Carlo-Event sampling over " << nsamples << " events" << Endl;
      std::vector<Double_t> pars( 2*GetNvar() );
      for (Int_t itoy=0; itoy<nsamples; itoy++) {

         for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
            
            // generate minimum and delta cuts for this variable

            // retrieve signal events
            Bool_t isSignal = kFALSE;
            Int_t    ievt1, ievt2;
            Double_t evt1, evt2;
            Int_t nbreak = 0;
            while (!isSignal) {
               ievt1 = Int_t(rnd->Uniform(0.,1.)*nevents);
               ievt2 = Int_t(rnd->Uniform(0.,1.)*nevents);

               ReadTrainingEvent( ievt1 );
               isSignal = IsSignalEvent();
               evt1 = GetEventVal( ivar );

               ReadTrainingEvent( ievt2 );
               isSignal &= IsSignalEvent();
               evt2 = GetEventVal( ivar );
               
               if (nbreak++ > 10000) fLogger << kFATAL << "<MCEvents>: could not find signal events" 
                                             << " after 10000 trials - do you have signal events in your sample ?" 
                                             << Endl;
               isSignal = 1;
            }

            // sort
            if (evt1 > evt2) { Double_t z = evt1; evt1 = evt2; evt2 = z; }
            pars[2*ivar]   = evt1;
            pars[2*ivar+1] = evt2 - evt1;
         }

         // compute estimator
         EstimatorFunction( pars );
         
         // what's the time please?
         ic++;
         if ((nsamples<1000) || ic%1000 == 0) timer.DrawProgressBar( ic );
      }

      delete rnd;
   }
   // --------------------------------------------------------------------------
   else fLogger << kFATAL << "Unknown minimisation method: " << fFitMethod << Endl;

   if (fBinaryTreeS != 0) { delete fBinaryTreeS; fBinaryTreeS = 0; }
   if (fBinaryTreeB != 0) { delete fBinaryTreeB; fBinaryTreeB = 0; }

   // force cut ranges within limits
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      for (Int_t ibin=0; ibin<fNbins; ibin++) {

         if ((*fFitParams)[ivar] == kForceMin && fCutMin[ivar][ibin] > -fgMaxAbsCutVal) {
            fCutMin[ivar][ibin] = -fgMaxAbsCutVal;
         }
         if ((*fFitParams)[ivar] == kForceMax && fCutMax[ivar][ibin] < fgMaxAbsCutVal) {
            fCutMax[ivar][ibin] = fgMaxAbsCutVal;
         }
      }
   }

   // some output
   PrintCuts( 0.1 );
   PrintCuts( 0.2 );
   PrintCuts( 0.3 );
   PrintCuts( 0.4 );
   PrintCuts( 0.5 );
   PrintCuts( 0.6 );
   PrintCuts( 0.7 );
   PrintCuts( 0.8 );
   PrintCuts( 0.9 );
   //   exit(1);
}

//_______________________________________________________________________
void TMVA::MethodCuts::Test( TTree* )
{
   // not used 
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::EstimatorFunction( Int_t ievt1, Int_t ievt2 )
{
   // for full event scan
   const Int_t nvar = GetNvar();
   Double_t* evt1 = new Double_t[nvar];
   Double_t* evt2 = new Double_t[nvar];
   
   ReadTrainingEvent( ievt1 );
   if (!IsSignalEvent()) return -1;
   for (Int_t ivar=0; ivar<nvar; ivar++) evt1[ivar] = GetEventVal( ivar );

   ReadTrainingEvent( ievt2 );
   if (!IsSignalEvent()) return -1;
   for (Int_t ivar=0; ivar<nvar; ivar++) evt2[ivar] = GetEventVal( ivar );

   // determine cuts
   std::vector<Double_t> pars;
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      Double_t cutMin;
      Double_t cutMax;
      if (evt1[ivar] < evt2[ivar]) {
         cutMin = evt1[ivar];
         cutMax = evt2[ivar];
      }
      else {
         cutMin = evt2[ivar];
         cutMax = evt1[ivar];
      }

      pars.push_back( cutMin );
      pars.push_back( cutMax - cutMin );
   }

   delete [] evt1;
   delete [] evt2;

   return ComputeEstimator( pars );
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::EstimatorFunction( std::vector<Double_t>& pars )
{
   // returns estimator for "cut fitness" used by GA
   return ComputeEstimator( pars );
}

//_______________________________________________________________________
Double_t TMVA::MethodCuts::ComputeEstimator( std::vector<Double_t>& pars )
{
   // returns estimator for "cut fitness" used by GA
   // there are two requirements:
   // 1) the signal efficiency must be equal to the required one in the 
   //    efficiency scan
   // 2) the background efficiency must be as small as possible
   // the requirement 1) has priority over 2)

   // caution: the npar gives the _free_ parameters
   // however: the "pars" array contains all parameters

   // determine cuts
   Double_t effS = 0, effB = 0;
   this->MatchParsToCuts( pars, &fTmpCutMin[0], &fTmpCutMax[0] );

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
      
   // get best background rejection for given signal efficiency
   Int_t ibinS = fEffBvsSLocal->FindBin( effS );      

   Double_t effBH       = fEffBvsSLocal->GetBinContent( ibinS );
   Double_t effBH_left  = (ibinS > 1       ) ? fEffBvsSLocal->GetBinContent( ibinS-1 ) : effBH;
   Double_t effBH_right = (ibinS < fNbins-1) ? fEffBvsSLocal->GetBinContent( ibinS+1 ) : effBH;

   Double_t average = (effBH_left+effBH_right)/2.;
   if (effBH < effB) average = effBH;

   // if the average of the bin right and left is larger than this one, add the difference to 
   // the current value of the estimator (because you can do at least so much better)
   eta = ( -TMath::Abs(effBH-average) + ( 1. - (effBH - effB) ) ) / (1+effS); 

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
void TMVA::MethodCuts::MatchParsToCuts( const std::vector<Double_t> & pars, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates parameters into cuts
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Int_t ipar = 2*ivar;
      cutMin[ivar] = ((*fRangeSign)[ivar] > 0) ? pars[ipar] : pars[ipar] - pars[ipar+1];
      cutMax[ivar] = ((*fRangeSign)[ivar] > 0) ? pars[ipar] + pars[ipar+1] : pars[ipar]; 
   }
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchCutsToPars( std::vector<Double_t>& pars, 
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
   
   MatchCutsToPars( pars, cutMin, cutMax );
   delete [] cutMin;
   delete [] cutMax;
}

//_______________________________________________________________________
void TMVA::MethodCuts::MatchCutsToPars( std::vector<Double_t>& pars, 
                                        Double_t* cutMin, Double_t* cutMax )
{
   // translates cuts into parameters
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      Int_t ipar = 2*ivar;
      pars[ipar]   = ((*fRangeSign)[ivar] > 0) ? cutMin[ivar] : cutMax[ivar];
      pars[ipar+1] = cutMax[ivar] - cutMin[ivar];
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

   // NOTE: The signal efficiency written out into 
   //       the weight file does not correspond to the center of the bin within which the 
   //       background rejection is maximised (as before) but to the lower left edge of it. 
   //       This is because the cut optimisation algorithm determines the best background 
   //       rejection for all signal efficiencies belonging into a bin. Since the best background 
   //       rejection is in general obtained for the lowest possible signal efficiency, the 
   //       reference signal efficeincy is the lowest value in the bin.
   for (Int_t ibin=0; ibin<fNbins; ibin++) {
      Double_t effS = fEffBvsSLocal->GetBinCenter ( ibin + 1 ) - 0.5*fEffBvsSLocal->GetBinWidth( ibin + 1 );
      if (TMath::Abs(effS) < 1e-10) effS = 0;
      o << setw(4) << ibin+1 << "  "    
        << setw(8)<< effS  << "  " 
        << setw(8)<< fEffBvsSLocal->GetBinContent( ibin + 1 ) << "  ";  
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
      fLogger << kINFO << "Read cuts optimised using sample of"<< fNRandCuts << " MC events" << Endl;
   }
   else if (fFitMethod == kUseMonteCarloEvents) {
      fLogger << kINFO << "Read cuts optimised using sample of"<< fNRandCuts << " MC-Event events" << Endl;
   }
   else if (fFitMethod == kUseGeneticAlgorithm) {
      fLogger << kINFO << "Read cuts optimised using Genetic Algorithm" << Endl;
   }
   else if (fFitMethod == kUseSimulatedAnnealing) {
      fLogger << kINFO << "Read cuts optimised using Simulated Annealing algorithm" << Endl;
   }
   else if (fFitMethod == kUseEventScan) {
      fLogger << kINFO << "Read cuts optimised using Full Event Scan" << Endl;
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
   TList* list  = gTools().ParseFormatLine( theString );
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

   delete list;

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
      fTrainEffBvsS->SetDirectory(0);
      fTrainRejBvsS->SetDirectory(0);

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
Double_t TMVA::MethodCuts::GetEfficiency( TString theString, TTree*, Double_t& effSerr )
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
   TList* list  = TMVA::gTools().ParseFormatLine( theString, ":" );

   if (list->GetSize() > 2) {
      delete list;
      fLogger << kFATAL << "<GetEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05, or empty string" << Endl;
      return -1;
   }

   // sanity check
   Bool_t computeArea = (list->GetSize() < 2); // the area is computed 

   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref = (computeArea?1.:atof( ((TObjString*)list->At(1))->GetString() ));

   delete list;

   
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
      if (NULL != fEffBvsS) delete fEffBvsS; 
      if (NULL != fRejBvsS) delete fRejBvsS; 
    
      fEffBvsS = new TH1F( GetTestvarName() + "_effBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fRejBvsS = new TH1F( GetTestvarName() + "_rejBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fEffBvsS->SetDirectory(0);
      fRejBvsS->SetDirectory(0);

      // use root finder

      // make the background-vs-signal efficiency plot
      Double_t* tmpCutMin = new Double_t[GetNvar()];
      Double_t* tmpCutMax = new Double_t[GetNvar()];
      for (Int_t bini=1; bini<=fNbins; bini++) {
         for (Int_t ivar=0; ivar <GetNvar(); ivar++) {
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

      delete [] tmpCutMin;
      delete [] tmpCutMax;

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
   TString bold    = gConfig().WriteOptionsReference() ? "<b>" : "";
   TString resbold = gConfig().WriteOptionsReference() ? "</b>" : "";
   TString brk     = gConfig().WriteOptionsReference() ? "<br>" : "";

   fLogger << Endl;
   fLogger << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The optimisation of rectangular cuts performed by TMVA maximises " << Endl;
   fLogger << "the background rejection at given signal efficiency, and scans " << Endl;
   fLogger << "over the full range of the latter quantity. Three optimisation" << Endl;
   fLogger << "methods are optional: Monte Carlo sampling (MC), a Genetics" << Endl;
   fLogger << "Algorithm (GA), and Simulated Annealing (SA). GA and SA are"  << Endl;
   fLogger << "expected to perform best." << Endl;
   fLogger << Endl;
   fLogger << "The difficulty to find the optimal cuts strongly increases with" << Endl;
   fLogger << "the dimensionality (number of input variables) of the problem." << Endl;
   fLogger << "This behavior is due to the non-uniqueness of the solution space."<<  Endl;
   fLogger << Endl;
   fLogger << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
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
   fLogger << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   fLogger << "" << Endl;
   fLogger << bold << "Monte Carlo sampling:" << resbold << Endl;
   fLogger << "" << Endl;
   fLogger << "Apart form the \"Fsmart\" option for the variables, the only way" << Endl;
   fLogger << "to improve the MC sampling is to increase the sampling rate. This" << Endl;
   fLogger << "is done via the configuration option \"MC_NRandCuts\". The execution" << Endl;
   fLogger << "time scales linearly with the sampling rate." << Endl;
   fLogger << "" << Endl;
   fLogger << bold << "Genetic Algorithm:" << resbold << Endl;
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
   fLogger << bold << "Simulated Annealing (SA) algorithm:" << resbold << Endl;
   fLogger << "" << Endl;
   fLogger << "\"Increasing Adaptive\" approach:" << Endl;
   fLogger << "" << Endl;
   fLogger << "The algorithm seeks local minima and explores their neighborhood, while" << Endl;
   fLogger << "changing the ambient temperature depending on the number of failures" << Endl;
   fLogger << "in the previous steps. The performance can be improved by increasing" << Endl;
   fLogger << "the number of iteration steps (\"MaxCalls\"), or by adjusting the" << Endl;
   fLogger << "minimal temperature (\"MinTemperature\"). Manual adjustments of the" << Endl;
   fLogger << "speed of the temperature increase (\"TemperatureScale\" and \"AdaptiveSpeed\")" << Endl;
   fLogger << "to individual data sets should also help. Summary:" << brk << Endl;
   fLogger << "  -> increase \"MaxCalls\"" << brk << Endl;
   fLogger << "  -> adjust   \"MinTemperature\"" << brk << Endl;
   fLogger << "  -> adjust   \"TemperatureScale\"" << brk << Endl;
   fLogger << "  -> adjust   \"AdaptiveSpeed\"" << Endl;
   fLogger << "" << Endl;   
   fLogger << "\"Decreasing Adaptive\" approach:" << Endl;
   fLogger << "" << Endl;
   fLogger << "The algorithm calculates the initial temperature (based on the effect-" << Endl;
   fLogger << "iveness of large steps) and the multiplier that ensures to reach the" << Endl;
   fLogger << "minimal temperature with the requested number of iteration steps." << Endl;
   fLogger << "The performance can be improved by adjusting the minimal temperature" << Endl;
   fLogger << " (\"MinTemperature\") and by increasing number of steps (\"MaxCalls\"):" << brk << Endl;
   fLogger << "  -> increase \"MaxCalls\"" << brk << Endl;
   fLogger << "  -> adjust   \"MinTemperature\"" << Endl;
   fLogger << " " << Endl;
   fLogger << "Other kernels:" << Endl;
   fLogger << "" << Endl;
   fLogger << "Alternative ways of counting the temperature change are implemented. " << Endl;
   fLogger << "Each of them starts with the maximum temperature (\"MaxTemperature\")" << Endl;
   fLogger << "and descreases while changing the temperature according to a given" << Endl;
   fLogger << "prescription:" << brk << Endl;
   fLogger << "CurrentTemperature =" << brk << Endl;
   fLogger << "  - Sqrt: InitialTemperature / Sqrt(StepNumber+2) * TemperatureScale" << brk << Endl;
   fLogger << "  - Log:  InitialTemperature / Log(StepNumber+2) * TemperatureScale" << brk << Endl;
   fLogger << "  - Homo: InitialTemperature / (StepNumber+2) * TemperatureScale" << brk << Endl;
   fLogger << "  - Sin:  ( Sin( StepNumber / TemperatureScale ) + 1 ) / (StepNumber + 1) * InitialTemperature + Eps" << brk << Endl;
   fLogger << "  - Geo:  CurrentTemperature * TemperatureScale" << Endl;
   fLogger << "" << Endl;
   fLogger << "Their performance can be improved by adjusting initial temperature" << Endl;
   fLogger << "(\"InitialTemperature\"), the number of iteration steps (\"MaxCalls\")," << Endl;
   fLogger << "and the multiplier that scales the termperature descrease" << Endl;
   fLogger << "(\"TemperatureScale\")" << brk << Endl;
   fLogger << "  -> increase \"MaxCalls\"" << brk << Endl;
   fLogger << "  -> adjust   \"InitialTemperature\"" << brk << Endl;
   fLogger << "  -> adjust   \"TemperatureScale\"" << brk << Endl;
   fLogger << "  -> adjust   \"KernelTemperature\"" << Endl;
}
