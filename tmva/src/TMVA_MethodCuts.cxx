// @(#)root/tmva $Id: TMVA_MethodCuts.cxx,v 1.2 2006/05/08 12:59:13 brun Exp $ 
// Author: Andreas Hoecker, Peter Speckmayer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate Data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCuts                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Multivariate optimisation of signal efficiency for given background  
// efficiency, using rectangular minimum and maximum requirements on    
//_______________________________________________________________________

#include <stdio.h>
#include "time.h"
#include "TMVA_MethodCuts.h"
#include "TMVA_GeneticCuts.h"
#include "TMVA_Tools.h"
#include "TMVA_Timer.h"
#include "Riostream.h"
#include "TH1F.h"
#include "TObjString.h"

#define DEBUG_TMVA_MethodCuts kTRUE

ClassImp(TMVA_MethodCuts)

// init global variables
TMVA_MethodCuts* TMVA_MethodCuts::fThisCuts = NULL;

//_______________________________________________________________________
TMVA_MethodCuts::TMVA_MethodCuts( TString jobName, vector<TString>* theVariables,  
				  TTree* theTree, TString theOption, TDirectory* theTargetDir )
  : TMVA_MethodBase( jobName, theVariables, theTree, theOption, theTargetDir )
{ 
  InitCuts();

  // ---------------------------------------------------------------------------------- 
  // interpret option string
  // format of option string: Method:nRandCuts:Option_var1:...:Option_varn
  // "Method" can be:
  //     - "MC"    : Monte-Carlo optimization (recommended)
  //     - "FitSel": Minuit Fit: "FitSel_Migrad" or "FitSel_Simplex": using event selection
  //     - "FitPDF": Minuit Fit: "FitPDF_Migrad" or "FitPDF_Simplex" PDF-based
  //                 (only useful for uncorrelated input variables)
  // "option_vari" can be 
  //     - "FMax"  : ForceMax   (the max cut is fixed to maximum of variable i)
  //     - "FMin"  : ForceMin   (the min cut is fixed to minimum of variable i)
  //     - "FSmart": ForceSmart (the min or max cut is fixed to min/max, based on mean value)
  //     - Adding "All" to "option_vari", eg, "AllFSmart" will use this option for all variables
  //     - if "option_vari" is empty (== ""), no assumptions on cut min/max are made
  // ---------------------------------------------------------------------------------- 
  TList* list  = TMVA_Tools::ParseFormatLine( fOptions );

  if (list->GetSize()<1) {
    fOptions = "MC:10000:";
    cout << "--- " << GetName() << ": problems with options string, using default: " 
	 << fOptions << endl;
    list  = TMVA_Tools::ParseFormatLine( fOptions );
  }  

  // interpret string
  // which optimisation Method
  TString s = ((TObjString*)list->At(0))->GetString();
  s.ToUpper();
  if      (s.Contains( "MC"     )) fFitMethod = UseMonteCarlo;
  else if (s.Contains( "GA"     )) fFitMethod = UseGeneticAlgorithm;
  else {
    cout << "--- " << GetName() << ": unknown entry in field 0 of option string: " 
	 << s << " ==> abort" << endl;
    exit(1);
  }

  if (list->GetSize() > 1) { // options are specified
    s = ((TObjString*)list->At(1))->GetString();
    s.ToUpper();

    if      (s.Contains( "EFFSEL" )) fEffMethod = UseEventSelection; // highly recommended
    else if (s.Contains( "EFFPDF" )) fEffMethod = UsePDFs;
    else                             fEffMethod = UseEventSelection;
  }

  // options output
  cout << "--- " << GetName() << ": interpret options string: '" << fOptions << "'" << endl;
  printf( "--- %s: --> use optimization method: '%s'\n", 
	  GetName(), (fFitMethod == UseMonteCarlo) ? "Monte Carlo" : "Genetic Algorithm" );
  printf( "--- %s: --> use efficiency computation method: '%s'\n", 
	  GetName(), (fEffMethod == UseEventSelection) ? "Event Selection" : "PDF" );

  // -----------------------------------------------------------------------------------
  // interpret for MC use  
  //
  switch (fFitMethod) {

  case UseMonteCarlo:

    if (list->GetSize() > 2) {

      s = ((TObjString*)list->At(2))->GetString();
      fNRandCuts = atoi( s );
      if (fNRandCuts <= 1) {
	cout << "--- " << GetName() << ": invalid number of MC events: " <<  fNRandCuts 
	     << " in field 2 of option string: " << s << " ==> abort" << endl;
	exit(1);
      }
    }
    
    cout << "--- " << GetName() << ": generate " << fNRandCuts << " random cut samples"
	 << endl;
  
    if (list->GetSize() > 3) { // options are specified

      s = ((TObjString*)list->At(3))->GetString();
      s.ToUpper();
      if (s.Contains( "ALL" )) { // one option sets all the others
	FitParameters theFitP = NotEnforced;
	if      (s.Contains( "FMAX"   )) theFitP = ForceMax;
	else if (s.Contains( "FMIN"   )) theFitP = ForceMin;
	else if (s.Contains( "FSMART" )) theFitP = ForceSmart;
	else if (s.Contains( "FVERYSMART" )) theFitP = ForceVerySmart;
	else {
	  cout << "--- " << GetName() << ": unknown fit parameter option "
	       << " in field 2 of option string: " << s << " ==> abort" << endl;
	  exit(1);
	}
	for (Int_t ivar=0; ivar<fNvar; ivar++) (*fFitParams)[ivar] = theFitP;

	if (theFitP != NotEnforced) 
	  cout << "--- " << GetName() << ": use 'smart' cuts" << endl;
      }
      else { // individual options
	for (Int_t ivar=0; ivar<fNvar; ivar++) {
	  if (list->GetSize() >= 3+ivar) {
	    s = ((TObjString*)list->At(2+ivar))->GetString();
	    s.ToUpper();
	    FitParameters theFitP = NotEnforced;
	    if      (s == "" || s == "NOTENFORCED") theFitP = NotEnforced;
	    else if (s.Contains( "FMAX"   )) theFitP = ForceMax;
	    else if (s.Contains( "FMIN"   )) theFitP = ForceMin;
	    else if (s.Contains( "FSMART" )) theFitP = ForceSmart;
	    else if (s.Contains( "FVERYSMART" )) theFitP = ForceVerySmart;
	    else {
	      cout << "--- " << GetName() << ": unknown fit parameter option "
		   << " in field " << ivar+3 << " (var: " << ivar 
		   << " of option string: " << s << " ==> abort" << endl;
	      exit(1);
	    }
	    (*fFitParams)[ivar] = theFitP;

	    if (theFitP != NotEnforced) 
	      cout << "--- " << GetName() << ": use 'smart' cuts for variable: " 
		   << "'" << (*fInputVars)[ivar] << "'" << endl;
	  }	
	}
      }      
    }
    break;

  // -----------------------------------------------------------------------------------
  // interpret for GA use  
  //
  case UseGeneticAlgorithm:
    
    if (list->GetSize() > 2) {      
      s = ((TObjString*)list->At(2))->GetString(); fGa_nsteps = atoi( s );
      if (list->GetSize() > 3) {      
	s = ((TObjString*)list->At(3))->GetString(); fGa_preCalc = atoi( s );
	if (list->GetSize() > 4) {      
	  s = ((TObjString*)list->At(4))->GetString(); fGa_SC_steps = atoi( s );
	  if (list->GetSize() > 5) {      
	    s = ((TObjString*)list->At(5))->GetString(); fGa_SC_offsteps = atoi( s );
	    if (list->GetSize() > 6) {      
	      s = ((TObjString*)list->At(6))->GetString(); fGa_SC_factor = atof( s );
	    }
	  }
	}
      }
    }
    break;
    
  default:

    cout << "--- " << GetName() << ": Error: unknown method: " << fFitMethod 
	 << " ==> abort" << endl;
    exit(1);
  }

  
  if (fFitMethod == UseMonteCarlo) 
    printf( "--- %s: --> number of MC events to be generated: %i\n", GetName(), fNRandCuts );
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    TString theFitOption = ( ((*fFitParams)[ivar] == NotEnforced) ? "NotEnforced" :
			     ((*fFitParams)[ivar] == ForceMin   ) ? "ForceMin"    :
			     ((*fFitParams)[ivar] == ForceMax   ) ? "ForceMax"    :
			     ((*fFitParams)[ivar] == ForceSmart ) ? "ForceSmart"  :
			     ((*fFitParams)[ivar] == ForceVerySmart ) ? "ForceVerySmart"  : "other" );
    
    printf( "--- %s: --> option for variable: %s: '%s' (#: %i)\n",
	    GetName(), (const char*)(*fInputVars)[ivar], (const char*)theFitOption, 
	    (Int_t)(*fFitParams)[ivar] );
  }
  // ---------------------------------------------------------------------------------- 

}

//_______________________________________________________________________
TMVA_MethodCuts::TMVA_MethodCuts( vector<TString> *theVariables, 
				  TString theWeightFile,  
				  TDirectory* theTargetDir )
  : TMVA_MethodBase( theVariables, theWeightFile, theTargetDir ) 
{
  InitCuts();
}

//_______________________________________________________________________
void TMVA_MethodCuts::InitCuts( void ) 
{
  fMethodName        = "Cuts";
  fMethod            = TMVA_Types::Cuts;  
  fTestvar           = fTestvarPrefix+GetMethodName();
  fConstrainType     = ConstrainEffS;
  fVarHistS          = fVarHistB = 0;		 
  fVarHistS_smooth   = fVarHistB_smooth = 0;
  fVarPdfS           = fVarPdfB = 0; 
  fFitParams         = 0;
  fEffBvsSLocal      = 0;
  fBinaryTreeS       = fBinaryTreeB = 0;
  fEffSMin           = 0;
  fEffSMax           = 0; 

  // MC defaults
  fNRandCuts         = 100000;

  // GA defaults
  fGa_preCalc        = 3;
  fGa_SC_steps       = 10;
  fGa_SC_offsteps    = 5;
  fGa_SC_factor      = 0.95;
  fGa_nsteps         = 30;

  fThisCuts          = this;

  // vector with fit results
  fNpar      = 2*fNvar;
  fPar0      = new vector<Double_t>( fNpar );
  fRangeSign = new vector<Int_t>   ( fNvar );
  fMeanS     = new vector<Double_t>( fNvar ); 
  fMeanB     = new vector<Double_t>( fNvar ); 
  fRmsS      = new vector<Double_t>( fNvar );  
  fRmsB      = new vector<Double_t>( fNvar );  
  fXmin      = new vector<Double_t>( fNvar );  
  fXmax      = new vector<Double_t>( fNvar );  

  // get the variable specific options, first initialize default
  fFitParams = new vector<FitParameters>( fNvar );
  for (Int_t ivar=0; ivar<fNvar; ivar++) (*fFitParams)[ivar] = NotEnforced;

  fTrandom   = new TRandom( 0 ); // set seed
  fFitMethod = UseMonteCarlo;
  fTestSignalEff = -1;

  // create LUT for cuts
  fCutMin = new Double_t*[fNvar];
  fCutMax = new Double_t*[fNvar];
  for (Int_t i=0;i<fNvar;i++) {
    fCutMin[i] = new Double_t[fNbins];
    fCutMax[i] = new Double_t[fNbins];
  }
  
  // init
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    for (Int_t ibin=0; ibin<fNbins; ibin++) {
      fCutMin[ivar][ibin] = 0;
      fCutMax[ivar][ibin] = 0;
    }
  }
}

//_______________________________________________________________________
TMVA_MethodCuts::~TMVA_MethodCuts( void )
{
  if (Verbose()){
    cout << "--- TMVA_MethodCuts: Destructor called " << endl;
  }
  delete fPar0;
  delete fRangeSign;
  delete fTrandom;
  delete fMeanS;
  delete fMeanB;
  delete fRmsS;
  delete fRmsB;
  delete fXmin;
  delete fXmax;  
  for (Int_t i=0;i<fNvar;i++) {
    if (fCutMin[i] != NULL) delete [] fCutMin[i];
    if (fCutMax[i] != NULL) delete [] fCutMax[i];
  }

  if (NULL != fBinaryTreeS) delete fBinaryTreeS;
  if (NULL != fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
Double_t TMVA_MethodCuts::GetMvaValue( TMVA_Event *e )
{
  // evaluation
   
  // sanity check
  if (fCutMin == NULL || fCutMax == NULL || fNbins == 0) {
    cerr << "--- " << GetName() << "::Eval_Cuts: Fatal Error: fCutMin/Max have zero pointer. "
	 << "Did you book Cuts ? ==> abort" << endl;
    exit(1);
  }

  // sanity check
  if (fTestSignalEff > 0) {  
    // get efficiency bin
    Int_t ibin = int((fTestSignalEff - fEffSMin)/(fEffSMax - fEffSMin)*Double_t(fNbins));
    if (ibin < 0       ) ibin = 0;
    if (ibin >= fNbins) ibin = fNbins - 1;
    
    Bool_t passed = kTRUE;
    for (size_t ivar=0; ivar<e->GetData().size(); ivar++) {
      passed *= (e->GetData()[ivar] >= fCutMin[ivar][ibin] && e->GetData()[ivar] <= fCutMax[ivar][ibin]);
    }
    return (Double_t)passed;
  }
  else return 0;
}

//_______________________________________________________________________
void  TMVA_MethodCuts::Train( void )
{
  // trainning method
   
  // perform basic sanity chacks
  if (!SanityChecks()) {
    cout << "--- " << GetName() << ": Error: Basic sanity checks failed ==> abort"
	 << endl;
    exit(1);
  }

  if (fEffMethod == UsePDFs) CreateVariablePDFs(); // create PDFs for variables

  // get background efficiency for which the signal efficiency
  // ought to be maximized
  fConstrainType = ConstrainEffS;

  // create binary trees (global member variables) for signal and background
  Int_t dummy;
  fBinaryTreeS = new TMVA_BinarySearchTree();
  fBinaryTreeS->Fill( fTrainingTree, fInputVars, dummy, 1 );
  fBinaryTreeB = new TMVA_BinarySearchTree();
  fBinaryTreeB->Fill( fTrainingTree, fInputVars, dummy, 0 );

  // init basic statistics
  TObjArrayIter branchIter( fTrainingTree->GetListOfBranches(), kIterForward );
  TBranch*      branch = 0;
  Int_t         ivar   = -1;
  const Int_t nvar = this->fNvar;
  TString       branchName[nvar];
  Float_t       branchVar[nvar];
  Int_t         theType;

  vector<TH1F*> signalDist, bkgDist;

  while ((branch = (TBranch*)branchIter.Next()) != 0) {
    // note: allowed are only variables with minimum and maximum cut
    //       i.e., no distinct cut regions are supported
    if ((TString)branch->GetName() == "type") {
      fTrainingTree->SetBranchAddress( branch->GetName(), &theType );
    }
    else {
      ++ivar;
      branchName[ivar] = branch->GetName();
      fTrainingTree->SetBranchAddress( branchName[ivar],  &branchVar[ivar] );

      // determine mean and rms to obtain appropriate starting values
      TMVA_Tools::ComputeStat( fTrainingTree, branchName[ivar],
			       (*fMeanS)[ivar], (*fMeanB)[ivar], 
			       (*fRmsS)[ivar], (*fRmsB)[ivar], 
			       (*fXmin)[ivar], (*fXmax)[ivar] );
      
      // I want to use these distributions later to steer the MC-Method a bit into the 
      // direction where the difference in the distributions for BKG and Signal are largest
      TString name = Form( "sigDistVar%d",ivar );
      signalDist.push_back( (TH1F*)TMVA_Tools::projNormTH1F( fTrainingTree, branchName[ivar], name, 50,
							     (*fXmin)[ivar], (*fXmax)[ivar],
							     "type==1" ) );
      name = Form( "bkgDistVar%d",ivar );
      bkgDist.push_back( (TH1F*)TMVA_Tools::projNormTH1F( fTrainingTree, branchName[ivar], name,50,
							  (*fXmin)[ivar],(*fXmax)[ivar],
							  "type==0" ) );
  
      if ((*fInputVars)[ivar] != branchName[ivar]) {
	cout << "Error in: " << GetName() << "::Train: mismatch in variables ==> abort: "
	     << ivar << " " << (*fInputVars)[ivar] << " " << branchName[ivar]
	     << endl;
	exit(1);
      }
    }
  }

  // determine eff(B) versus eff(S) plot
  fConstrainType = ConstrainEffS;

  Int_t ibin=0;
  fEffBvsSLocal = new TH1F( fTestvar + "_effBvsSLocal", 
			     TString(GetName()) + " efficiency of B vs S", 
			     fNbins, 0.0, 1.0 );

  // init
  for (ibin=1; ibin<=fNbins; ibin++) fEffBvsSLocal->SetBinContent( ibin, -0.1 );

  // --------------------------------------------------------------------------
  if (fFitMethod == UseMonteCarlo) {
    
    // generate MC cuts
    Double_t cutMin[nvar];
    Double_t cutMax[nvar];
    
    // MC loop
    cout << "--- " << GetName() << ": Generating " << fNRandCuts 
	 << " cycles (random cuts) ... patience please" << endl;

    Int_t nBinsFilled=0, nBinsFilledAt=0;

    // timing of MC
    TMVA_Timer timer( fNRandCuts, GetName() ); 

    for (Int_t imc=0; imc<fNRandCuts; imc++) {

      // generate random cuts
      for (Int_t ivar=0; ivar<fNvar; ivar++) {

	FitParameters fitParam = (*fFitParams)[ivar];

	if (fitParam == ForceSmart) {
	  if ((*fMeanS)[ivar] > (*fMeanB)[ivar]) fitParam = ForceMax;
	  else                                     fitParam = ForceMin;	  
	}

	if (fitParam == ForceMin) 
	  cutMin[ivar] = (*fXmin)[ivar];
	else
	  cutMin[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];

	if (fitParam == ForceMax) 
	  cutMax[ivar] = (*fXmax)[ivar];
	else
	  cutMax[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
	
	if (fitParam == ForceVerySmart){
	  // generate random cut parameters gaussian distrubuted around the variable values
	  // where the difference between signal and background is maximal
	  
	  // get the variable distributions:
	  cutMin[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - (*fXmin)[ivar]) + (*fXmin)[ivar];
	  cutMax[ivar] = fTrandom->Rndm()*((*fXmax)[ivar] - cutMin[ivar]   ) + cutMin[ivar];
	  // ..... to be continued (Helge)
	}

	if (cutMax[ivar] < cutMin[ivar]) {
	  cout << "--- " << GetName() << ": Error in ::Train: mismatch with cuts ==> abort"
	       << endl;
	  exit(1);
	}
      }

      // event loop
      Double_t effS = 0, effB = 0;
      GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
    
      // determine bin
      Int_t    ibinS = (Int_t)(effS*Float_t(fNbins) + 1);
      if (ibinS < 1      ) ibinS = 1;
      if (ibinS > fNbins) ibinS = fNbins;
      
      // linear extrapolation 
      // (not done at present --> MC will be slightly biased !
      //  the bias increases with the bin width)
      Double_t effBH = fEffBvsSLocal->GetBinContent( ibinS );

      // preliminary best event -> backup
      if (effBH < 0 || effBH > effB) {
	fEffBvsSLocal->SetBinContent( ibinS, effB );
	for (Int_t ivar=0; ivar<fNvar; ivar++) {
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
	  printf( "--- %s: fraction of efficiency bins filled: %3.1f\n",
		  GetName(), nbinsF/Float_t(fNbins) );
      }
    } // end of MC loop

    if (this->Verbose()){
      cout << "--- TMVA_MethodCuts: fraction of filled eff. bins did not increase" 
	   << " anymore after "<< nBinsFilledAt << " cycles" << endl;
    }

    // get elapsed time
    cout << "--- " << GetName() << ": elapsed time: " << timer.GetElapsedTime() 
	 << endl;    

  }
  // --------------------------------------------------------------------------
  else if (fFitMethod == UseGeneticAlgorithm) {

    // ranges
    vector<LowHigh*> ranges;
    
    for (Int_t ivar=0; ivar<fNvar; ivar++) {
      (*fRangeSign)[ivar] = +1;    
      ranges.push_back( new LowHigh( (*fXmin)[ivar], (*fXmax)[ivar] ) );
      ranges.push_back( new LowHigh( 0, (*fXmax)[ivar] - (*fXmin)[ivar] ) );
    }

    TMVA_GeneticCuts *bestResultsStore = new TMVA_GeneticCuts( 0, ranges ); 
    TMVA_GeneticCuts *bestResults      = new TMVA_GeneticCuts( 0, ranges );

    cout << "--- " << GetName() << ": GA: entree, please be patient ..." << endl;

    // timing of MC
    TMVA_Timer timer1( fGa_preCalc*fNbins, GetName() ); 

    // precalculation
    for (Int_t preCalc = 0; preCalc < fGa_preCalc; preCalc++) {

      for (Int_t ibin=1; ibin<=fNbins; ibin++) {

	timer1.DrawProgressBar( ibin + preCalc*fNbins );

	fEffRef = fEffBvsSLocal->GetBinCenter( ibin );

	// ---- perform series of fits to achieve best convergence

	// "m_ga_spread" times the number of variables
	TMVA_GeneticCuts ga( ranges.size() * 10, ranges ); 

	ga.population.addPopulation( &bestResults->population );
	ga.calculateFitness();
	ga.population.trimPopulation();

	do {
	  ga.init();
	  ga.calculateFitness();
	  ga.spreadControl( fGa_SC_steps, fGa_SC_offsteps, fGa_SC_factor );
	} while (!ga.hasConverged( Int_t(fGa_nsteps*0.67), 0.0001 ));
	
	bestResultsStore->population.giveHint( ga.population.getGenes( 0 )->factors );
      }
      bestResults = bestResultsStore;
      bestResultsStore = new TMVA_GeneticCuts( 0, ranges );
		
    }

    bestResults->init();

    // main run
    cout << "--- " << GetName() << ": GA: start main course                                    " 
	 << endl;

    // timing of MC
    TMVA_Timer timer2( fNbins, GetName() ); 

    for (ibin=1; ibin<=fNbins; ibin++) {

      timer2.DrawProgressBar( ibin );
      
      fEffRef = fEffBvsSLocal->GetBinCenter( ibin );

      // ---- perform series of fits to achieve best convergence

      TMVA_GeneticCuts ga( ranges.size() * 10, ranges ); // 10 times the number of variables
      ga.spread = 0.1;
      ga.population.addPopulation( &bestResults->population );
      ga.calculateFitness();
      ga.population.trimPopulation();
      do {
	ga.init();
	ga.calculateFitness();
	ga.spreadControl( fGa_SC_steps, fGa_SC_offsteps, fGa_SC_factor );
      } while (!ga.hasConverged( fGa_nsteps, 0.00001 ));

      Int_t n;
      const Int_t nvar2 = 2*this->fNvar;
      Double_t par[nvar2];

      n = 0;
      for( vector< Double_t >::iterator vec = ga.population.getGenes( 0 )->factors.begin(); 
	   vec < ga.population.getGenes( 0 )->factors.end(); vec++ ){
	par[n] = (*vec);
	n++;
      }

      Double_t effS = 0, effB = 0;
      Double_t cutMin[nvar];
      Double_t cutMax[nvar];
      this->MatchParsToCuts( par, &cutMin[0], &cutMax[0] );
      this->GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);

      for (Int_t ivar=0; ivar<fNvar; ivar++) {
	fCutMin[ivar][ibin-1] = cutMin[ivar]; // bin 1 stored in index 0
	fCutMax[ivar][ibin-1] = cutMax[ivar];
      }
    }

    // get elapsed time
    cout << "--- " << GetName() << ": GA: elapsed time: " << timer1.GetElapsedTime() 
	 << endl;    

  }
  // --------------------------------------------------------------------------
  else {
    cerr << "--- " << GetName() << ": Error: unknown minization method: "
	 << fFitMethod << " ==> abort" << endl;
    exit(1);    
  }

  // write weights and technical histos to file
  WriteWeightsToFile();
  WriteHistosToFile();
  delete fEffBvsSLocal;
  if (fBinaryTreeS) delete fBinaryTreeS;
  if (fBinaryTreeB) delete fBinaryTreeB;
}

//_______________________________________________________________________
Double_t TMVA_MethodCuts::ComputeEstimator( Double_t *par, Int_t /*npar*/ )
{
  // caution: the npar gives the _free_ parameters
  // however: the "par" array contains all parameters

  // determine cuts
  Double_t effS = 0, effB = 0;
  const Int_t nvar = this->fNvar;
  Double_t cutMin[nvar];
  Double_t cutMax[nvar];
  this->MatchParsToCuts( par, &cutMin[0], &cutMax[0] );

  // retrieve signal and background efficiencies for given cut
  switch (fEffMethod) {
  case UsePDFs:
    this->GetEffsfromPDFs( &cutMin[0], &cutMax[0], effS, effB );
    break;
  case UseEventSelection:
    this->GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
    break;
  default:
    this->GetEffsfromSelection( &cutMin[0], &cutMax[0], effS, effB);
  }

  // compute estimator
  const Double_t epsilon = 1.0e-06;
  Double_t eta;  
  if (fConstrainType == ConstrainEffS) {
    if (TMath::Abs(effS - fEffRef) > 0.001) eta = TMath::Abs(effB) + TMath::Abs(effS - fEffRef)/epsilon;
    else eta = TMath::Abs(effB);
  }
  else if (fConstrainType == ConstrainEffB) {
    eta = ( pow( (effB - fEffRef)/epsilon, 1 ) +
	    pow( 1.0/((effS > 0) ? effS : epsilon), 2 ) );
  }
  else eta = 0;

  return eta;
}

//_______________________________________________________________________
void TMVA_MethodCuts::MatchParsToCuts( Double_t* par, 
				       Double_t* cutMin, Double_t* cutMax )
{
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    Int_t ipar = 2*ivar;
    cutMin[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] : par[ipar] - par[ipar+1];
    cutMax[ivar] = ((*fRangeSign)[ivar] > 0) ? par[ipar] + par[ipar+1] : par[ipar]; 
  }
}

//_______________________________________________________________________
void TMVA_MethodCuts::MatchCutsToPars( Double_t* par, 
				       Double_t* cutMin, Double_t* cutMax )
{
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    Int_t ipar = 2*ivar;
    par[ipar]   = ((*fRangeSign)[ivar] > 0) ? cutMin[ivar] : cutMax[ivar];
    par[ipar+1] = cutMax[ivar] - cutMin[ivar];
  }
}

//_______________________________________________________________________
void TMVA_MethodCuts::GetEffsfromPDFs( Double_t* cutMin, Double_t* cutMax,
				       Double_t& effS, Double_t& effB )
{
  effS = 1.0;
  effB = 1.0;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    effS *= (*fVarPdfS)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
    effB *= (*fVarPdfB)[ivar]->GetIntegral( cutMin[ivar], cutMax[ivar] );
  }
}

//_______________________________________________________________________
void TMVA_MethodCuts::GetEffsfromSelection( Double_t* cutMin, Double_t* cutMax,
					    Double_t& effS, Double_t& effB)
{
  Float_t nTotS = 0, nTotB = 0;
  Float_t nSelS = 0, nSelB = 0;  
  
  TMVA_Volume* volume = new TMVA_Volume( cutMin, cutMax, fNvar );
  
  nSelS = fBinaryTreeS->SearchVolume( volume );
  nSelB = fBinaryTreeB->SearchVolume( volume );
  
  nTotS = Float_t(fBinaryTreeS->GetSumOfWeights());
  nTotB = Float_t(fBinaryTreeB->GetSumOfWeights());
    
  delete volume;

  // sanity check
  if (nTotS == 0 && nTotB == 0) {
    cout << "--- " << GetName() 
	 << ": fatal error in::ComputeEstimator: zero total number of events:"
	 << " nTotS, nTotB: " << nTotS << " " << nTotB << " ***"
	 << endl;
    exit(1);
  }

  // efficiencies
  if (nTotS == 0 ) {
    effS = 0;
    effB = nSelB/nTotB;
    cout << "--- " << GetName() 
	 << ": Warning in ::ComputeEstimator: zero number of events signal Events:\n";
  }
  else if ( nTotB == 0) {
    effB = 0;
    effS = nSelS/nTotS;
    cout << "--- " << GetName() 
	 << ": Warning in ::ComputeEstimator: zero number of events background Events:\n";
  }
  else {
    effS = nSelS/nTotS;
    effB = nSelB/nTotB;
  }  
}

//_______________________________________________________________________
void TMVA_MethodCuts::CreateVariablePDFs( void )
{
  // create list of histograms and PDFs
  fVarHistS        = new vector<TH1*>    ( fNvar );
  fVarHistB        = new vector<TH1*>    ( fNvar );
  fVarHistS_smooth = new vector<TH1*>    ( fNvar );
  fVarHistB_smooth = new vector<TH1*>    ( fNvar );
  fVarPdfS         = new vector<TMVA_PDF*>( fNvar );
  fVarPdfB         = new vector<TMVA_PDF*>( fNvar );

  Int_t nsmooth = 0;

  for (Int_t ivar=0; ivar<fNvar; ivar++) { 

    // ---- signal
    TString histTitle = (*fInputVars)[ivar] + " signal training";
    TString histName  = (*fInputVars)[ivar] + "_sig";
    TString drawOpt   = (*fInputVars)[ivar] + ">>h(";
    drawOpt += fNbins;
    drawOpt += ")";

    // selection
    fTrainingTree->Draw( drawOpt, "type==1", "goff" );
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

    fTrainingTree->Draw( drawOpt, "type==0", "goff" );
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
    (*fVarPdfS)[ivar] = new TMVA_PDF( (*fVarHistS_smooth)[ivar], TMVA_PDF::Spline2 );
    (*fVarPdfB)[ivar] = new TMVA_PDF( (*fVarHistB_smooth)[ivar], TMVA_PDF::Spline2 );
  }		  
}

//_______________________________________________________________________
Bool_t TMVA_MethodCuts::SanityChecks( void )
{
  // basic checks to ensure that assumptions on variable order are satisfied
  Bool_t        isOK = kTRUE;

  TObjArrayIter branchIter( fTrainingTree->GetListOfBranches(), kIterForward );
  TBranch*      branch = 0;
  Int_t         ivar   = -1;
  while ((branch = (TBranch*)branchIter.Next()) != 0) {
    TString branchName = branch->GetName();

    if (branchName != "type") {

      // determine mean and rms to obtain appropriate starting values
      ivar++;
      if ((*fInputVars)[ivar] != branchName) {
	cout << "Error in: " << GetName() << "::SanityChecks: mismatch in variables ==> abort"
	     << endl;
	isOK = kFALSE;
      }
    }
  }  

  return isOK;
}

//_______________________________________________________________________
void TMVA_MethodCuts::CheckErr( TString cmd, Int_t errFlag )
{
  if (errFlag != 0) 
    cout << "--- " << GetName() << ": Problem in ::cmd: " << cmd
	 << " / error code: " << errFlag <<" *** " << endl;
}

//_______________________________________________________________________
void  TMVA_MethodCuts::WriteWeightsToFile( void )
{
  // write weights to file
  // though we could write the root effBvsS histogram directly, we
  // prefer here to put everything into a human-readable form  
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": creating weight file: " << fname << endl;
  ofstream fout( fname );
  if (!fout.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::WriteWeightsToFile: "
         << "unable to open output  weight file: " << fname << endl;
    exit(1);
  }
  // write variable names and min/max
  // NOTE: the latter values are mandatory for the normalisation
  // in the reader application !!!
  fout << this->GetMethodName() <<endl;
  fout << "NVars= " << fNvar <<endl; 
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    TString var = (*fInputVars)[ivar];
    fout << var << "  " << GetXminNorm( var ) << "  " << GetXmaxNorm( var )
	 << endl;
  }
  
  // first the dimensions
  fout << "OptimisationMethod " << "nRandCuts " << "nbins:" << endl;
  fout << ((fEffMethod == UseEventSelection) ? "Fit-EventSelection" : 
	   (fEffMethod == UsePDFs) ? "Fit-PDF" : "Monte-Carlo") << "  " ;
  fout << fNRandCuts << "  ";
  fout << fNbins << endl;

  //  fout << endl;
  fout << "the optimised cuts for " << fNvar << " variables"  << endl;
  fout << "format: ibin(hist) effS effB cutMin[ivar=0] cutMax[ivar=0]"
       << " ... cutMin[ivar=n-1] cutMax[ivar=n-1]" << endl;
  Int_t ibin, ivar;
  for (ibin=0; ibin<fNbins; ibin++) {
    fout << setw(4) << ibin+1 << "  "    
	 << setw(8)<< fEffBvsSLocal->GetBinCenter( ibin +1 ) << "  " 
	 << setw(8)<< fEffBvsSLocal->GetBinContent( ibin +1 ) << "  ";  
    for (ivar=0; ivar<fNvar; ivar++)
      fout <<setw(10)<< fCutMin[ivar][ibin] << "  " << setw(10) << fCutMax[ivar][ibin] << "  ";
    fout << endl;
  }
}
  
//_______________________________________________________________________
void  TMVA_MethodCuts::ReadWeightsFromFile( void )
{
  // read weights from file
  // though we could write the root effBvsS histogram directly, we
  // prefer here to put everything into a human-readable form  
  TString fname = GetWeightFileName();
  cout << "--- " << GetName() << ": reading weight file: " << fname << endl;
  ifstream fin( fname );
  if (!fin.good( )) { // file not found --> Error
    cout << "--- " << GetName() << ": Error in ::ReadWeightsFromFile: "
         << "unable to open input file: " << fname << endl;
    exit(1);
  }

  // read variable names and min/max
  // NOTE: the latter values are mandatory for the normalisation
  // in the reader application !!!
  TString var, dummy;
  Double_t xmin, xmax;
  fin >> dummy;
  this->SetMethodName(dummy);
  fin >> dummy >> fNvar;
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    fin >> var >> xmin >> xmax;

    // sanity check
    if (var != (*fInputVars)[ivar]) {
      cout << "--- " << GetName() << ": Error while reading weight file; "
           << "unknown variable: " << var << " at position: " << ivar << ". "
           << "Expected variable: " << (*fInputVars)[ivar] << " ==> abort" 
	   << endl;
      exit(1);
    }

    // set min/max
    this->SetXminNorm( ivar, xmin );
    this->SetXmaxNorm( ivar, xmax );
  }

  // first the dimensions
  fin >> dummy >> dummy >> dummy;
  fin >> dummy >> fNRandCuts >> fNbins;
  cout << "--- " << GetName() << ": Read cuts from "<< fNRandCuts << " MC events"
       << " in " << fNbins << " efficiency bins " << endl;
  fin >> dummy >> dummy >> dummy >> dummy >>fNvar>>dummy ;

  char buffer[200];
  fin.getline(buffer,200);
  fin.getline(buffer,200);

  // read histogram and cuts
  Int_t   ibin, ivar;
  Int_t   tmpbin;
  Float_t tmpeffS, tempeffB;
  for (ibin=0; ibin<fNbins; ibin++) {
    fin >> tmpbin >> tmpeffS >> tempeffB;

    if (ibin == 0        ) fEffSMin = tmpeffS;
    if (ibin == fNbins-1) fEffSMax = tmpeffS;

    for (ivar=0; ivar<fNvar; ivar++) {
      fin >> fCutMin[ivar][ibin] >> fCutMax[ivar][ibin];
    }
  }
}

//_______________________________________________________________________
void  TMVA_MethodCuts::WriteHistosToFile( void )
{
  cout << "--- " << GetName() << ": write " << GetName() 
       << " special histos to file: " << fBaseDir->GetPath() << endl;
  
  fEffBvsSLocal->Write();

  // save reference histograms to file
  if (fEffMethod == UsePDFs) {
    gDirectory->GetListOfKeys()->Print();
    fBaseDir->mkdir(GetName()+GetMethodName())->cd();  
    for (Int_t ivar=0; ivar<fNvar; ivar++) { 
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
void TMVA_MethodCuts::TestInitLocal( TTree *theTree ) 
 {
   cout << "--- " << GetName() << ": called TestInitLocal " <<endl;

  // create binary trees (global member variables) for signal and background
   Int_t dummy;
   fBinaryTreeS = new TMVA_BinarySearchTree();
   fBinaryTreeS->Fill( theTree, fInputVars, dummy, 1 );
   fBinaryTreeB = new TMVA_BinarySearchTree();
   fBinaryTreeB->Fill( theTree, fInputVars, dummy, 0 );
 }

//_______________________________________________________________________
Double_t TMVA_MethodCuts::GetEfficiency( TString theString, TTree * /*theTree*/ )
{
  // parse input string for required background efficiency
  TList*  list  = TMVA_Tools::ParseFormatLine( theString );
  // sanity check
  if (list->GetSize() != 2) {
    cout << "--- " << GetName() << ": Error in::GetEfficiency: wrong number of arguments"
	 << " in string: " << theString
	 << " | required format, e.g., Efficiency:0.05" << endl;
    return -1;
  }

  // that will be the value of the efficiency retured (does not affect
  // the efficiency-vs-bkg plot which is done anyway.
  Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

  if (Verbose()) 
    cout << "--- " << GetName() << "::GetEfficiency <verbose>: compute eff(S) at eff(B) = " 
	 << effBref << endl;

  // first round ? --> create histograms
  if ( fEffBvsS == NULL ||   fRejBvsS == NULL) {
    // there is no really good equivalent to the fEffS; fEffB (efficiency vs cutvalue)
    // for the "Cuts" method (unless we had only one cut). Maybe later I might add here
    // histograms for each of the cuts...but this would require also a change in the 
    // base class, and it is not really necessary, as we get exactly THIS info from the
    // "evaluateAllVariables" anyway.

    // now create efficiency curve: background versus signal
    //    if (NULL != fEffBvsS)fEffBvsS->Delete();
    //    if (NULL != fRejBvsS)fRejBvsS->Delete();
    if (NULL != fEffBvsS)delete fEffBvsS; 
    if (NULL != fRejBvsS)delete fRejBvsS; 
    
    fEffBvsS = new TH1F( fTestvar + "_effBvsS", fTestvar + "", fNbins, 0, 1 );
    fRejBvsS = new TH1F( fTestvar + "_rejBvsS", fTestvar + "", fNbins, 0, 1 );
    // use root finder

    // make the background-vs-signal efficiency plot
    const Int_t nvar = this->fNvar;
    for (Int_t bini=1; bini<=fNbins; bini++) {
      Double_t tmpCutMin[nvar], tmpCutMax[nvar];
      for (Int_t ivar=0; ivar <fNvar; ivar++){
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

    // create splines for histogram
    fGrapheffBvsS = new TGraph( fEffBvsS );
    fSpleffBvsS   = new TMVA_TSpline1( "effBvsS", fGrapheffBvsS );
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

  return fEffSatB = 0.5*(effS + effS_);
}
