// @(#)root/tmva $Id: TMVA_MethodBase.cpp,v 1.13 2006/05/02 23:27:40 helgevoss Exp $   
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodBase                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
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
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodBase.cpp,v 1.13 2006/05/02 23:27:40 helgevoss Exp $ 
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Virtual base class for all MVA method                                
//                                                                      
//_______________________________________________________________________

#include "string"
#include "TMVA_MethodBase.h"
#include "TMVA_Event.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TMVA_Timer.h"
#include <stdlib.h>
#include "TMVA_Tools.h"
#include "TMVA_RootFinder.h"
#include "TMVA_PDF.h"
#include "TObjString.h"
#include "TQObject.h"
#include "TSpline.h"
#include "TMatrix.h"
#include "TMath.h"

#define DEBUG_TMVA_MethodBase          kFALSE
#define TMVA_MethodBase_MaxIterations_ 200
#define Use_Splines_for_Eff_          kTRUE
#define Thats_Big__                   1.0e30

#define NBIN_HIST_PLOT    100
#define NBIN_HIST_HIGH    10000

ClassImp(TMVA_MethodBase)
  
//_______________________________________________________________________
TMVA_MethodBase::TMVA_MethodBase( TString jobName, 
				  vector<TString>* theVariables,  
				  TTree*  theTree, 
				  TString theOption,
				  TDirectory*  theBaseDir) :
  fJobName      ( jobName ),
  fTrainingTree ( theTree ), 
  fInputVars    ( theVariables ),
  fOptions      ( theOption ),
  fBaseDir      ( theBaseDir ),
  fWeightFile   ( "" )
{
// default constructur
  this->Init();
  // parse option string and search for verbose
  // after that, remove the verbose option to not interfere with method-specific options
  TList*  list = TMVA_Tools::ParseFormatLine( fOptions );
  TString O;
  for (Int_t i=0; i<list->GetSize(); i++) {
    TString s = ((TObjString*)list->At(i))->GetString();
    s.ToUpper();
    if (s == "V") {
      fVerbose = kTRUE;
      if (i == list->GetSize()-1) O.Chop();
    }
    else {
      O += (TString)((TObjString*)list->At(i))->GetString();
      if (i < list->GetSize()-1) O += ":";
    }
  }
  fOptions = O;

  // default extension for weight files
  fFileExtension = "weights";
  fFileDir       = "weights";
  gSystem->MakeDirectory( fFileDir );


  // init the normalization vectors
  InitNorm( fTrainingTree );
}

//_______________________________________________________________________
TMVA_MethodBase::TMVA_MethodBase( vector<TString> *theVariables, 
				  TString weightFile, 
				  TDirectory*  theBaseDir) 
  : fJobName      ( "" ),
    fTrainingTree ( NULL ), 
    fInputVars    ( theVariables ),
    fOptions      ( "" ),
    fBaseDir      ( theBaseDir ),
    fWeightFile   ( weightFile )
{
// constructor used for Testing + Application of the MVA, only (no training), using given WeightFiles
  this->Init();
  fJobName       = "";   //not used 
}

//_______________________________________________________________________
void TMVA_MethodBase::Init(){
  fVerbose       = kFALSE;
  fIsOK          = kTRUE;
  fNvar = fInputVars->size();
  fXminNorm      = 0;
  fXmaxNorm      = 0;
  fMeanS         = -1; // it is nice to have them "initialized". Every method
  fMeanB         = -1; // but "MethodCuts" sets them later
  fRmsS          = -1;
  fRmsB          = -1;

  fNbins         = NBIN_HIST_PLOT;
  fNbinsH        = NBIN_HIST_HIGH;

  fHistS_plotbin = NULL;
  fHistB_plotbin = NULL;
  fHistS_highbin = NULL;
  fHistB_highbin = NULL;
  fEffS          = NULL;
  fEffB          = NULL;
  fEffBvsS       = NULL;
  fRejBvsS       = NULL;
  fHistBhatS     = NULL;
  fHistBhatB     = NULL;
  fHistMuS       = NULL;
  fHistMuB       = NULL;
  fTestvarPrefix = "MVA_";

  // init variable bounds
  fXminNorm = new vector<Double_t>( fNvar );
  fXmaxNorm = new vector<Double_t>( fNvar );
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    (*fXminNorm)[ivar] = +Thats_Big__;
    (*fXmaxNorm)[ivar] = -Thats_Big__;
  }

  // define "this" pointer
  ResetThisBase();
}

//_______________________________________________________________________
TMVA_MethodBase::~TMVA_MethodBase( void )
{
/// default destructur
  if (Verbose()) cout << "--- TMVA_MethodCuts: Destructor called " << endl;

   if (NULL != fXminNorm) delete fXminNorm;
   if (NULL != fXmaxNorm) delete fXmaxNorm;
}

//_______________________________________________________________________
void TMVA_MethodBase::InitNorm( TTree* theTree )
{
  // if trainingsTree exists, fill min/max vector
  if (NULL != theTree) {
    for (Int_t ivar=0; ivar<fNvar; ivar++) {
      this->SetXminNorm( ivar, theTree->GetMinimum( (*fInputVars)[ivar] ) );
      this->SetXmaxNorm( ivar, theTree->GetMaximum( (*fInputVars)[ivar] ) );
    }
  }       
  else {
    cout << "--- " << GetName() 
	 << ":InitNorm Error: tree has zero pointer ==> abort" << endl;
    exit(1);
  }
  if (Verbose()) {
    cout << "--- " << GetName() << " <verbose>: set minNorm/maxNorm to: " << endl;    
    cout << setprecision(3); 
    for (Int_t ivar=0; ivar<fNvar; ivar++) 
      cout << "    " << (*fInputVars)[ivar] 
	   << "\t: [" << GetXminNorm( ivar ) << "\t, " << GetXmaxNorm( ivar ) << "\t] " << endl;
    cout << setprecision(5); // reset to better value
  }
}

//_______________________________________________________________________
void TMVA_MethodBase::SetWeightFileName( void ) 
{  
  fWeightFile =  fFileDir + "/" +fJobName + "_" + fMethodName + "." + fFileExtension;
}

//_______________________________________________________________________
void TMVA_MethodBase::SetWeightFileName( TString theWeightFile)
{  
  fWeightFile = theWeightFile;
}

//_______________________________________________________________________
TString TMVA_MethodBase::GetWeightFileName( void ) 
{  
  if (fWeightFile == "") this->SetWeightFileName();  
  return fWeightFile;
}

//_______________________________________________________________________
Bool_t TMVA_MethodBase::CheckSanity( TTree* theTree )
{
  // if no tree is given, use the trainingTree
  TTree* tree = (0 != theTree) ? theTree : fTrainingTree;
  
  // the input variables must exist in the tree
  vector<TString>::iterator itrVar    = fInputVars->begin();
  vector<TString>::iterator itrVarEnd = fInputVars->end();
  Bool_t found = kTRUE;
  for (; itrVar != itrVarEnd; itrVar++) 
    if (0 == tree->FindBranch( *itrVar )) found = kFALSE;

  return found;
}

//_______________________________________________________________________
void TMVA_MethodBase::AppendToMethodName( TString methodNameSuffix )
{
  fMethodName += "_";
  fTestvar += "_";
  fMethodName += methodNameSuffix;
  fTestvar += methodNameSuffix;
}

//_______________________________________________________________________
void TMVA_MethodBase::SetWeightFileDir( TString fileDir )
{
  fFileDir = fileDir; 
  gSystem->MakeDirectory( fFileDir );
}

// ---------------------------------------------------------------------------------------
// ----- methods related to renormalization of variables ---------------------------------
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
Double_t TMVA_MethodBase::Norm( TString var, Double_t x ) const
{
  return TMVA_Tools::NormVariable( x, GetXminNorm( var ), GetXmaxNorm( var ) );
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::Norm( Int_t ivar, Double_t x ) const
{
  return TMVA_Tools::NormVariable( x, GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
}

//_______________________________________________________________________
void TMVA_MethodBase::UpdateNorm( Int_t ivar, Double_t x ) 
{
  if (x < GetXminNorm( ivar )) SetXminNorm( ivar, x );
  if (x > GetXmaxNorm( ivar )) SetXmaxNorm( ivar, x );
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetXminNorm( TString var ) const
{
  for (Int_t ivar=0; ivar<fNvar; ivar++)
    if (var == (*fInputVars)[ivar]) return (*fXminNorm)[ivar];

  cout << "--- " << GetName() << ": Error in ::GetXminNorm: variable not found ==> abort " 
       << var << endl;
  exit(1);
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetXmaxNorm( TString var ) const
{
  for (Int_t ivar=0; ivar<fNvar; ivar++)
    if (var == (*fInputVars)[ivar]) return (*fXmaxNorm)[ivar];

  cout << "--- " << GetName() << ": Error in ::GetXmaxNorm: variable not found ==> abort " 
       << var << endl;
  exit(1);
}

//_______________________________________________________________________
void TMVA_MethodBase::SetXminNorm( TString var, Double_t x ) 
{
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    if (var == (*fInputVars)[ivar]) {
      (*fXminNorm)[ivar] = x;
      return;
    }
  }

  cout << "--- " << GetName() << ": Error in ::SetXminNorm: variable not found ==> abort " 
       << var << endl;
  exit(1);
}

//_______________________________________________________________________
void TMVA_MethodBase::SetXmaxNorm( TString var, Double_t x ) 
{
  for (Int_t ivar=0; ivar<fNvar; ivar++) {
    if (var == (*fInputVars)[ivar]) {
      (*fXmaxNorm)[ivar] = x;
      return;
    }
  }

  cout << "--- " << GetName() << ": Error in ::SetXmaxNorm: variable not found ==> abort " 
       << var << endl;
  exit(1);
}
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
void TMVA_MethodBase::TestInit(TTree* theTestTree)
{
   
  //  fTestTree       = theTestTree;
  fHistS_plotbin  = fHistB_plotbin = 0;
  fHistS_highbin  = fHistB_highbin = 0;
  fEffS           = fEffB = fEffBvsS = fRejBvsS = 0;
  fGraphS         = fGraphB = 0;
  fEffSatB        = 0;
  fSeparation     = 0;
  fCutOrientation = Positive;
  fSplS           = fSplB = 0;
  fSplRefS        = fSplRefB = 0;


  // sanity checks: tree must exist, and theVar must be in tree
  if (0 == theTestTree || 
      ( 0 == theTestTree->FindBranch( fTestvar ) && !(GetMethodName().Contains("Cuts")))){
    cout<<"--- "<< GetName() << ": Error in TestInit: test variable "<<fTestvar
	<<" not found in tree"<<endl;

    fIsOK = kFALSE;  
  }

  // now call the TestInitLocal for possible individual initialisation
  // of each method
  this->TestInitLocal(theTestTree);

} 

//_______________________________________________________________________
void TMVA_MethodBase::PrepareEvaluationTree( TTree* testTree )
{
  // sanity checks
  if (0 == testTree) {
    cout << "--- " << GetName() 
	 << ": PrepareEvaluationTree Error: testTree has zero pointer ==> exit(1)"
	 << endl;
    exit(1);
  }

  // checks that all variables in input vector indeed exist in the testTree
  if (!CheckSanity( testTree )) {
    cout << "--- " << GetName() 
	 << ": PrepareEvaluationTree Error: sanity check failed" << endl;
    exit(1);
  }

  // read the coefficients
  this->ReadWeightsFromFile();

  // fill a new branch into the testTree with the MVA-value of the method
  Double_t myMVA;
  TBranch *newBranch = testTree->Branch( fTestvar, &myMVA, fTestvar + "/D" );
  
  // use timer
  TMVA_Timer timer( testTree->GetEntries(), GetName(), kTRUE ); 
  
  for (Int_t ievt=0; ievt<testTree->GetEntries(); ievt++) {
    if ((Int_t)ievt%100 == 0) timer.DrawProgressBar( ievt );
    TMVA_Event *e = new TMVA_Event( testTree, ievt, fInputVars );
    myMVA = this->GetMvaValue( e );
    newBranch->Fill();
    delete e;
  }
  cout << "--- " << GetName() << ": elapsed time for evaluation of " 
       << testTree->GetEntries() <<  " events: "
       << timer.GetElapsedTime() << "       " << endl;    
}

//_______________________________________________________________________
void TMVA_MethodBase::Test( TTree *theTestTree )
{
  // basic statistics operations are made in base class
  // note: cannot directly modify private class members
  Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
  TMVA_Tools::ComputeStat( theTestTree, fTestvar, meanS, meanB, rmsS, rmsB, xmin, xmax );

  // choose reasonable histogram ranges, by removing outliers
  Double_t nrms = 4;
  xmin = TMath::Max( TMath::Min(meanS - nrms*rmsS, meanB - nrms*rmsB ), xmin );
  xmax = TMath::Min( TMath::Max(meanS + nrms*rmsS, meanB + nrms*rmsB ), xmax );

  fMeanS = meanS; fMeanB = meanB;
  fRmsS  = rmsS;  fRmsB  = rmsB;
  fXmin  = xmin;  fXmax  = xmax;  

  // determine cut orientation
  fCutOrientation = (fMeanS > fMeanB) ? Positive : Negative;

  // fill 2 types of histograms for the various analyses
  // this one is for actual plotting
  fHistS_plotbin = TMVA_Tools::projNormTH1F( theTestTree, fTestvar, 
					     fTestvar + "_S", 
					     fNbins, fXmin, fXmax, "type == 1" );
  fHistB_plotbin = TMVA_Tools::projNormTH1F( theTestTree, fTestvar, 
					     fTestvar + "_B",  
					     fNbins, fXmin, fXmax, "type == 0" );

  // need histograms with even more bins for efficiency calculation and integration
  fHistS_highbin = TMVA_Tools::projNormTH1F( theTestTree, fTestvar, 
					     fTestvar + "_S_high",  
					     fNbinsH, fXmin, fXmax, "type == 1" );
  fHistB_highbin = TMVA_Tools::projNormTH1F( theTestTree, fTestvar, 
					     fTestvar + "_B_high",  
					     fNbinsH, fXmin, fXmax, "type == 0" );

  // create PDFs from histograms, using default splines, and no additional smoothing
  fSplS = new TMVA_PDF( fHistS_plotbin, TMVA_PDF::Spline2, 0 ); 
  fSplB = new TMVA_PDF( fHistB_plotbin, TMVA_PDF::Spline2, 0  ); 

}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetEfficiency( TString theString, TTree *theTree )
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
  //that will be the value of the efficiency retured (does not affect
  //the efficiency-vs-bkg plot which is done anyway.
  Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

  if (DEBUG_TMVA_MethodBase) 
    cout << "--- " << GetName() << "::GetEfficiency : compute eff(S) at eff(B) = " << effBref << endl;

  // sanity check
  if (fHistS_highbin->GetNbinsX() != fHistB_highbin->GetNbinsX() ||
      fHistS_plotbin->GetNbinsX() != fHistB_plotbin->GetNbinsX()) {
    cout << "--- " << GetName() 
	 << "WARNING: in GetEfficiency() binning mismatch between signal and background histos"<<endl;  
    fIsOK = kFALSE;
    return -1.0;
  }

  // create histogram

  // first, get efficiency histograms for signal and background
  Double_t xmin = fHistS_highbin->GetXaxis()->GetXmin();
  Double_t xmax = fHistS_highbin->GetXaxis()->GetXmax();

  // first round ? --> create histograms
  Bool_t firstPass = kFALSE;
  if (NULL == fEffS && NULL == fEffB) firstPass = kTRUE;

  if (firstPass) {

    fEffS = new TH1F( fTestvar + "_effS", fTestvar + " (signal)",     fNbinsH, xmin, xmax );
    fEffB = new TH1F( fTestvar + "_effB", fTestvar + " (background)", fNbinsH, xmin, xmax );

    // sign if cut
    Int_t sign = (fCutOrientation == Positive) ? +1 : -1;

    // this method is unbinned
    for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {
      
      TH1* theHist = 0;
      if ((Int_t)TMVA_Tools::GetValue( theTree, ievt, "type" ) == 1) { // this is signal
	theHist = fEffS;
      }
      else { // this is background
	theHist = fEffB;
      }
      
      Double_t theVal = TMVA_Tools::GetValue( theTree, ievt, fTestvar );
      for (Int_t bin=1; bin<=fNbinsH; bin++) 
	if (sign*theVal > sign*theHist->GetBinCenter( bin )) theHist->AddBinContent( bin );
    }

    // renormalize to maximum
    fEffS->Scale( 1.0/(fEffS->GetMaximum() > 0 ? fEffS->GetMaximum() : 1) );
    fEffB->Scale( 1.0/(fEffB->GetMaximum() > 0 ? fEffB->GetMaximum() : 1) );

    // now create efficiency curve: background versus signal
    fEffBvsS = new TH1F( fTestvar + "_effBvsS", fTestvar + "", fNbins, 0, 1 );
    fRejBvsS = new TH1F( fTestvar + "_rejBvsS", fTestvar + "", fNbins, 0, 1 );
    // use root finder
    // spline background efficiency plot
    // note that there is a bin shift when going from a TH1F object to a TGraph :-(
    if (Use_Splines_for_Eff_) {
      fGraphS   = new TGraph( fEffS );
      fGraphB   = new TGraph( fEffB );
      fSplRefS  = new TMVA_TSpline1( "spline2_signal",     fGraphS );
      fSplRefB  = new TMVA_TSpline1( "spline2_background", fGraphB );   

      // verify spline sanity
      if (Verbose())
	cout << "--- " << GetName() 
	     << "::GetEfficiency <verbose>: verify signal and background eff. splines" << endl;
      TMVA_Tools::CheckSplines( fEffS, fSplRefS );
      TMVA_Tools::CheckSplines( fEffB, fSplRefB );
    }

    // make the background-vs-signal efficiency plot

    // create root finder
    // reset static "this" pointer before calling external function
    ResetThisBase();
    TMVA_RootFinder rootFinder( &IGetEffForRoot, fXmin, fXmax );

    Double_t effB = 0;
    for (Int_t bini=1; bini<=fNbins; bini++) {
      
      // find cut value corresponding to a given signal efficiency
      Double_t effS = fEffBvsS->GetBinCenter( bini );

      Double_t cut  = rootFinder.Root( effS );
      
      // retrieve background efficiency for given cut
      if (Use_Splines_for_Eff_)
	effB = fSplRefB->Eval( cut );
      else
	effB = fEffB->GetBinContent( fEffB->FindBin( cut ) );

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
  for (Int_t bini=1; bini<=nbins_; bini++) {
    
    // get corresponding signal and background efficiencies
    effS = (bini - 0.5)/Float_t(nbins_);
    effB = fSpleffBvsS->Eval( effS );

    // find signal efficiency that corresponds to required background efficiency
    if ((effB - effBref)*(effB_ - effBref) < 0) break;
    effS_ = effS;
    effB_ = effB;  
  }

  return fEffSatB = 0.5*(effS + effS_);//the mean between bin above and bin below

}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetSignificance( void )
{
  // compute significance of mean difference
  // significance = |<S> - <B>|/Sqrt(RMS_S2 + RMS_B2)
  Double_t rms = sqrt(pow(fRmsS,2) + pow(fRmsB,2));

  return fSignificance = (rms > 0) ? fabs(fMeanS - fMeanB)/rms : 0;
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetSeparation( void )
{
  // compute "separation" defined as
  // <s2> = (1/2) Int_-oo..+oo { (S(x)2 - B(x)2)/(S(x) + B(x)) dx }
  fSeparation = 0;

  Int_t nstep  = 1000;
  Double_t intBin = (fXmax - fXmin)/nstep;
  for (Int_t bin=0; bin<nstep; bin++) {
    Double_t x = (bin + 0.5)*intBin + fXmin;
    Double_t S = fSplS->GetVal( x );
    Double_t B = fSplB->GetVal( x );
    // separation
    if (S + B > 0) fSeparation += 0.5*pow(S - B,2)/(S + B); 
  }
  fSeparation *= intBin;
  
  return fSeparation;
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetmuTransform( TTree *theTree )
{
  //---------------------------------------------------------------------------------------
  // Authors     : Francois Le Diberder and Muriel Pivk
  // Reference   : Muriel Pivk,
  //               "Etude de la violation de CP dans la désintégration 
  //                B0 -> h+ h- (h = pi, K) auprès du détecteur BaBar à SLAC",
  //               PhD thesis at Universite de Paris VI-VII, LPNHE (IN2P3/CNRS), Paris, 2003
  //               http://tel.ccsd.cnrs.fr/documents/archives0/00/00/29/91/index_fr.html
  //
  // Definitions : Bhat = PDFbackground(x)/(PDFbackground(x) + PDFsignal(x))
  //               mu   = mu(b) = Int_0B Bhat[b'] db'
  //---------------------------------------------------------------------------------------

  // create Bhat distribution function
  Int_t nbin  = 70;
  fHistBhatS = new TH1F( fTestvar + "_BhatS", fTestvar + ": Bhat (S)", nbin, 0.0, 1.0 );
  fHistBhatB = new TH1F( fTestvar + "_BhatB", fTestvar + ": Bhat (B)", nbin, 0.0, 1.0 );

  fHistBhatS->Sumw2();
  fHistBhatB->Sumw2();

  vector<Double_t>* BhatB = new vector<Double_t>;
  vector<Double_t>* BhatS = new vector<Double_t>;
  Int_t ievt;
  for (ievt=0; ievt<theTree->GetEntries(); ievt++) {    
    Double_t x    = TMVA_Tools::GetValue( theTree, ievt, fTestvar );
    Double_t S    = fSplS->GetVal( x );
    Double_t B    = fSplB->GetVal( x );
    Double_t Bhat = 0;
    if (B + S > 0) Bhat = B/(B + S);

    if ((Int_t)TMVA_Tools::GetValue( theTree, ievt, "type" ) == 1) { // this is signal
      BhatS->push_back ( Bhat );
      fHistBhatS->Fill( Bhat );
    }
    else {
      BhatB->push_back ( Bhat );
      fHistBhatB->Fill( Bhat );
    }
  }

  // normalize histograms
  fHistBhatS->Scale( 1.0/((fHistBhatS->GetEntries() > 0 ? fHistBhatS->GetEntries() : 1) / nbin) );
  fHistBhatB->Scale( 1.0/((fHistBhatB->GetEntries() > 0 ? fHistBhatB->GetEntries() : 1) / nbin) );

  TMVA_PDF* yB = new TMVA_PDF( fHistBhatB, TMVA_PDF::Spline2, 100 );

  Int_t nevtS = BhatS->size();
  Int_t nevtB = BhatB->size();

  // get the mu-transform
  Int_t nbinMu = 50;
  fHistMuS = new TH1F( fTestvar + "_muTransform_S", 
			fTestvar + ": mu-Transform (S)", nbinMu, 0.0, 1.0 );
  fHistMuB = new TH1F( fTestvar + "_muTransform_B", 
			fTestvar + ": mu-Transform (B)", nbinMu, 0.0, 1.0 );
  
  // signal
  for (ievt=0; ievt<nevtS; ievt++) {    
    Double_t w = yB->GetVal( (*BhatS)[ievt] );
    if (w > 0) fHistMuS->Fill( 1.0 - (*BhatS)[ievt], 1.0/w );
  }

  // background (must be flat)
  for (ievt=0; ievt<nevtB; ievt++) {          
    Double_t w = yB->GetVal( (*BhatB)[ievt] );
    if (w > 0) fHistMuB->Fill( 1.0 - (*BhatB)[ievt], 1.0/w );
  }

  // normalize mu-transforms
  TMVA_Tools::NormHist( fHistMuS );
  TMVA_Tools::NormHist( fHistMuB );

  // determine the mu-transform value, which is defined as 
  // the average of the signal mu-transform Int_[0,1] { S(mu) dmu }
  // this average is 0.5 for background, by definition
  TMVA_PDF* thePdf = new TMVA_PDF( fHistMuS, TMVA_PDF::Spline2 );
  Double_t intS = 0;
  Int_t    nstp = 10000;
  for (Int_t istp=0; istp<nstp; istp++) {
    Double_t x = (istp + 0.5)/Double_t(nstp);
    intS += x*thePdf->GetVal( x );
  }
  intS /= Double_t(nstp);

  delete yB;
  delete thePdf;
  delete BhatB;
  delete BhatS;

  return intS; // return average mu-transform for signal
}

//_______________________________________________________________________
void TMVA_MethodBase::WriteHistosToFile( TDirectory* targetDir )
{
  targetDir->cd();
  if (0 != fHistS_plotbin) fHistS_plotbin->Write();
  if (0 != fHistB_plotbin) fHistB_plotbin->Write();
  if (0 != fHistS_highbin) fHistS_highbin->Write();
  if (0 != fHistB_highbin) fHistB_highbin->Write();
  if (0 != fEffS         ) fEffS->Write();
  if (0 != fEffB         ) fEffB->Write();
  if (0 != fEffBvsS      ) fEffBvsS->Write();
  if (0 != fRejBvsS      ) fRejBvsS->Write();
  if (0 != fHistBhatS    ) fHistBhatS->Write();
  if (0 != fHistBhatB    ) fHistBhatB->Write();
  if (0 != fHistMuS      ) fHistMuS->Write();
  if (0 != fHistMuB      ) fHistMuB->Write();
}

// ----------------------- r o o t   f i n d i n g ----------------------------

TMVA_MethodBase* TMVA_MethodBase::fThisBase = NULL;

//_______________________________________________________________________
Double_t TMVA_MethodBase::IGetEffForRoot( Double_t theCut ) 
{
  return TMVA_MethodBase::ThisBase()->GetEffForRoot( theCut );
}

//_______________________________________________________________________
Double_t TMVA_MethodBase::GetEffForRoot( Double_t theCut ) 
{
  Double_t retval;

  // retrieve the class object
  if (Use_Splines_for_Eff_)
    retval = fSplRefS->Eval( theCut );
  else
    retval = fEffS->GetBinContent( fEffS->FindBin( theCut ) );

  // caution: here we take some "forbidden" action to hide a problem:
  // in some cases, in particular for likelihood, the binned efficiency distributions
  // do not equal 1, at xmin, and 0 at xmax; of course, in principle we have the 
  // unbinned information available in the trees, but the unbinned minimization is
  // too slow, and we don't need to do a precision measurement here. Hence, we force
  // this property.
  Double_t eps = 1.0e-5;
  if      (theCut-fXmin < eps) retval = (GetCutOrientation() == Positive) ? 1.0 : 0.0;
  else if (fXmax-theCut < eps) retval = (GetCutOrientation() == Positive) ? 0.0 : 1.0;

  return retval;
};

