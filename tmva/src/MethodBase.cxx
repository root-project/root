// @(#)root/tmva $Id: MethodBase.cxx,v 1.4 2006/08/31 11:03:37 rdm Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodBase                                                      *
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
 * $Id: MethodBase.cxx,v 1.4 2006/08/31 11:03:37 rdm Exp $
 **********************************************************************************/

//_______________________________________________________________________
//
// Virtual base class for all MVA method
//
// MethodBase hosts several specific evaluation methods
//
// The kind of MVA that provides optimal performance in an analysis strongly
// depends on the particular application. The evaluation factory provides a
// number of numerical benchmark results to directly assess the performance
// of the MVA training on the independent test sample. These are:
// <ul>
//   <li> The <i>signal efficiency</i> at three representative background efficiencies
//        (which is 1 &minus; rejection).</li>
//   <li> The <i>significance</I> of an MVA estimator, defined by the difference
//        between the MVA mean values for signal and background, divided by the
//        quadratic sum of their root mean squares.</li>
//   <li> The <i>separation</i> of an MVA <i>x</i>, defined by the integral
//        &frac12;&int;(S(x) &minus; B(x))<sup>2</sup>/(S(x) + B(x))dx, where
//        S(x) and B(x) are the signal and background distributions, respectively.
//        The separation is zero for identical signal and background MVA shapes,
//        and it is one for disjunctive shapes.
//   <li> <a name="mu_transform">
//        The average, &int;x &mu;(S(x))dx, of the signal &mu;-transform.
//        The &mu;-transform of an MVA denotes the transformation that yields
//        a uniform background distribution. In this way, the signal distributions
//        S(x) can be directly compared among the various MVAs. The stronger S(x)
//        peaks towards one, the better is the discrimination of the MVA. The
//        &mu;-transform is
//        <a href=http://tel.ccsd.cnrs.fr/documents/archives0/00/00/29/91/index_fr.html>documented here</a>.
// </ul>
// The MVA standard output also prints the linear correlation coefficients between
// signal and background, which can be useful to eliminate variables that exhibit too
// strong correlations.
//_______________________________________________________________________

#include <string>
#include <stdlib.h>
#include "TROOT.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TQObject.h"
#include "TSpline.h"
#include "TMatrix.h"
#include "TMath.h"
#include "Riostream.h"

#include "TMVA/MethodBase.h"
#include "TMVA/Event.h"
#include "TMVA/Timer.h"
#include "TMVA/Tools.h"
#include "TMVA/RootFinder.h"
#include "TMVA/PDF.h"

const Bool_t   DEBUG_TMVA_MethodBase=kFALSE;
const Int_t    MethodBase_MaxIterations_=200;
const Bool_t   Use_Splines_for_Eff_=kTRUE;
const Double_t Thats_Big__=1.0e30;

const int NBIN_HIST_PLOT=100;
const int NBIN_HIST_HIGH=10000;

//ClassImp(TMVA::MethodBase)

//_______________________________________________________________________
TMVA::MethodBase::MethodBase( TString jobName,
                              vector<TString>* theVariables,
                              TTree*  theTree,
                              TString theOption,
                              TDirectory*  theBaseDir)
   : fJobName      ( jobName ),
     fTrainingTree ( theTree ),
     fInputVars    ( theVariables ),
     fOptions      ( theOption  ),
     fBaseDir      ( theBaseDir ),
     fWeightFile   ( "" ),
     fVerbose      ( kTRUE )
{
   // standard constructur

   this->Init();
   // parse option string and search for verbose
   // after that, remove the verbose option to not interfere with method-specific options
   TList*  list = TMVA::Tools::ParseFormatLine( fOptions );
   TString opt;
   for (Int_t i=0; i<list->GetSize(); i++) {
      TString s = ((TObjString*)list->At(i))->GetString();
      s.ToUpper();
      if (s == "V") {
         fVerbose = kTRUE;
         if (i == list->GetSize()-1) opt.Chop();
      }
      else {
         opt += (TString)((TObjString*)list->At(i))->GetString();
         if (i < list->GetSize()-1) opt += ":";
      }
   }
   fOptions = opt;

   for (Int_t i=0; i<list->GetSize(); i++) list->At(i)->Delete();
   delete list;

   // default extension for weight files
   fFileExtension = "weights";
   fFileDir       = "weights";
   gSystem->MakeDirectory( fFileDir );

   // init the normalization vectors
   InitNorm( fTrainingTree );
}

//_______________________________________________________________________
TMVA::MethodBase::MethodBase( vector<TString> *theVariables,
                              TString weightFile,
                              TDirectory*  theBaseDir)
   : fJobName      ( "" ),
     fTrainingTree ( NULL ),
     fInputVars    ( theVariables ),
     fOptions      ( "" ),
     fBaseDir      ( theBaseDir ),
     fWeightFile   ( weightFile ),
     fVerbose      ( kTRUE )
{
   // constructor used for Testing + Application of the MVA,
   // only (no training), using given WeightFiles

   this->Init();
   fJobName       = "";   //not used
}

//_______________________________________________________________________
void TMVA::MethodBase::Init()
{
   // default initialisation called by all constructors

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
TMVA::MethodBase::~MethodBase( void )
{
   // default destructur
   if (Verbose()) cout << "--- TMVA::MethodCuts: Destructor called " << endl;

   if (NULL != fXminNorm) delete fXminNorm;
   if (NULL != fXmaxNorm) delete fXmaxNorm;
}

//_______________________________________________________________________
void TMVA::MethodBase::InitNorm( TTree* theTree )
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
void TMVA::MethodBase::SetWeightFileName( void )
{
   // build weight file name
   fWeightFile =  fFileDir + "/" + fJobName + "_" + fMethodName + "." + fFileExtension;
}

//_______________________________________________________________________
void TMVA::MethodBase::SetWeightFileName( TString theWeightFile)
{
   // set the weight file name (depreciated)
   fWeightFile = theWeightFile;
}

//_______________________________________________________________________
TString TMVA::MethodBase::GetWeightFileName( void )
{
   // retrieve weight file name
   if (fWeightFile == "") this->SetWeightFileName();
   return fWeightFile;
}

//_______________________________________________________________________
Bool_t TMVA::MethodBase::CheckSanity( TTree* theTree )
{
   // tree sanity checks

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
void TMVA::MethodBase::AppendToMethodName( TString methodNameSuffix )
{
   // appends a suffix to the standard method name
   // to this is useful to run several instances of the same
   // method, e.g., to test different configuration sets

   fMethodName += "_";
   fTestvar += "_";
   fMethodName += methodNameSuffix;
   fTestvar += methodNameSuffix;
}

//_______________________________________________________________________
void TMVA::MethodBase::SetWeightFileDir( TString fileDir )
{
   // set directory of weight file

   fFileDir = fileDir;
   gSystem->MakeDirectory( fFileDir );
}

// ---------------------------------------------------------------------------------------
// ----- methods related to renormalization of variables ---------------------------------
// ---------------------------------------------------------------------------------------

//_______________________________________________________________________
Double_t TMVA::MethodBase::Norm( TString var, Double_t x ) const
{
   // renormalises variable with respect to its min and max
   return TMVA::Tools::NormVariable( x, GetXminNorm( var ), GetXmaxNorm( var ) );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::Norm( Int_t ivar, Double_t x ) const
{
   // renormalises variable with respect to its min and max
   return TMVA::Tools::NormVariable( x, GetXminNorm( ivar ), GetXmaxNorm( ivar ) );
}

//_______________________________________________________________________
void TMVA::MethodBase::UpdateNorm( Int_t ivar, Double_t x )
{
   // check and update norm
   if (x < GetXminNorm( ivar )) SetXminNorm( ivar, x );
   if (x > GetXmaxNorm( ivar )) SetXmaxNorm( ivar, x );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetXminNorm( TString var ) const
{
   // retrieves minimum for variable
   for (Int_t ivar=0; ivar<fNvar; ivar++)
      if (var == (*fInputVars)[ivar]) return (*fXminNorm)[ivar];

   cout << "--- " << GetName() << ": Error in ::GetXminNorm: variable not found ==> abort "
        << var << endl;
   exit(1);
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetXmaxNorm( TString var ) const
{
   // retrieves maximum for variable
   for (Int_t ivar=0; ivar<fNvar; ivar++)
      if (var == (*fInputVars)[ivar]) return (*fXmaxNorm)[ivar];

   cout << "--- " << GetName() << ": Error in ::GetXmaxNorm: variable not found ==> abort "
        << var << endl;
   exit(1);
}

//_______________________________________________________________________
void TMVA::MethodBase::SetXminNorm( TString var, Double_t x )
{
   // set minimum for variable
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
void TMVA::MethodBase::SetXmaxNorm( TString var, Double_t x )
{
   // set maximum for variable
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
void TMVA::MethodBase::TestInit(TTree* theTestTree)
{
   // initialisation of MVA testing

   //  fTestTree       = theTestTree;
   fHistS_plotbin  = fHistB_plotbin = 0;
   fHistS_highbin  = fHistB_highbin = 0;
   fEffS           = fEffB = fEffBvsS = fRejBvsS = 0;
   fGraphS         = fGraphB = 0;
   fCutOrientation = kPositive;
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
void TMVA::MethodBase::PrepareEvaluationTree( TTree* testTree )
{
   // prepare tree branch with the method's discriminating variable

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
   TMVA::Timer timer( testTree->GetEntries(), GetName(), kTRUE );

   for (Int_t ievt=0; ievt<testTree->GetEntries(); ievt++) {
      if ((Int_t)ievt%100 == 0) timer.DrawProgressBar( ievt );
      TMVA::Event *e = new TMVA::Event( testTree, ievt, fInputVars );
      myMVA = this->GetMvaValue( e );
      newBranch->Fill();
      delete e;
   }
   cout << "--- " << GetName() << ": elapsed time for evaluation of "
        << testTree->GetEntries() <<  " events: "
        << timer.GetElapsedTime() << "       " << endl;
}

//_______________________________________________________________________
void TMVA::MethodBase::Test( TTree *theTestTree )
{
   // test the method - not much is done here... mainly furthor initialisation

   // basic statistics operations are made in base class
   // note: cannot directly modify private class members
   Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;
   TMVA::Tools::ComputeStat( theTestTree, fTestvar, meanS, meanB, rmsS, rmsB, xmin, xmax );

   // choose reasonable histogram ranges, by removing outliers
   Double_t nrms = 4;
   xmin = TMath::Max( TMath::Min(meanS - nrms*rmsS, meanB - nrms*rmsB ), xmin );
   xmax = TMath::Min( TMath::Max(meanS + nrms*rmsS, meanB + nrms*rmsB ), xmax );

   fMeanS = meanS; fMeanB = meanB;
   fRmsS  = rmsS;  fRmsB  = rmsB;
   fXmin  = xmin;  fXmax  = xmax;

   // determine cut orientation
   fCutOrientation = (fMeanS > fMeanB) ? kPositive : kNegative;

   // fill 2 types of histograms for the various analyses
   // this one is for actual plotting
   fHistS_plotbin = TMVA::Tools::projNormTH1F( theTestTree, fTestvar,
                                               fTestvar + "_S",
                                               fNbins, fXmin, fXmax, "type == 1" );
   fHistB_plotbin = TMVA::Tools::projNormTH1F( theTestTree, fTestvar,
                                               fTestvar + "_B",
                                               fNbins, fXmin, fXmax, "type == 0" );

   // need histograms with even more bins for efficiency calculation and integration
   fHistS_highbin = TMVA::Tools::projNormTH1F( theTestTree, fTestvar,
                                               fTestvar + "_S_high",
                                               fNbinsH, fXmin, fXmax, "type == 1" );
   fHistB_highbin = TMVA::Tools::projNormTH1F( theTestTree, fTestvar,
                                               fTestvar + "_B_high",
                                               fNbinsH, fXmin, fXmax, "type == 0" );

   // create PDFs from histograms, using default splines, and no additional smoothing
   fSplS = new TMVA::PDF( fHistS_plotbin, TMVA::PDF::kSpline2, 0 );
   fSplB = new TMVA::PDF( fHistB_plotbin, TMVA::PDF::kSpline2, 0  );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetEfficiency( TString theString, TTree *theTree )
{
   // fill background efficiency (resp. rejection) versus signal efficiency plots
   // returns signal efficiency at background efficiency indicated in theString

   // parse input string for required background efficiency
   TList*  list  = TMVA::Tools::ParseFormatLine( theString );
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
      Int_t sign = (fCutOrientation == kPositive) ? +1 : -1;

      // this method is unbinned
      for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

         TH1* theHist = 0;
         if ((Int_t)TMVA::Tools::GetValue( theTree, ievt, "type" ) == 1) { // this is signal
            theHist = fEffS;
         }
         else { // this is background
            theHist = fEffB;
         }

         Double_t theVal = TMVA::Tools::GetValue( theTree, ievt, fTestvar );
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
         fSplRefS  = new TMVA::TSpline1( "spline2_signal",     fGraphS );
         fSplRefB  = new TMVA::TSpline1( "spline2_background", fGraphB );

         // verify spline sanity
         if (Verbose())
            cout << "--- " << GetName()
                 << "::GetEfficiency <verbose>: verify signal and background eff. splines" << endl;
         TMVA::Tools::CheckSplines( fEffS, fSplRefS );
         TMVA::Tools::CheckSplines( fEffB, fSplRefB );
      }

      // make the background-vs-signal efficiency plot

      // create root finder
      // reset static "this" pointer before calling external function
      ResetThisBase();
      TMVA::RootFinder rootFinder( &IGetEffForRoot, fXmin, fXmax );

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
      fSpleffBvsS   = new TMVA::TSpline1( "effBvsS", fGrapheffBvsS );
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

   return 0.5*(effS + effS_); // the mean between bin above and bin below

}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetSignificance( void )
{
   // compute significance of mean difference
   // significance = |<S> - <B>|/Sqrt(RMS_S2 + RMS_B2)
   Double_t rms = sqrt(pow(fRmsS,2) + pow(fRmsB,2));

   return (rms > 0) ? TMath::Abs(fMeanS - fMeanB)/rms : 0;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetSeparation( void )
{
   // compute "separation" defined as
   // <s2> = (1/2) Int_-oo..+oo { (S(x)2 - B(x)2)/(S(x) + B(x)) dx }
   Double_t separation = 0;

   Int_t nstep  = 1000;
   Double_t intBin = (fXmax - fXmin)/nstep;
   for (Int_t bin=0; bin<nstep; bin++) {
      Double_t x = (bin + 0.5)*intBin + fXmin;
      Double_t s = fSplS->GetVal( x );
      Double_t b = fSplB->GetVal( x );
      // separation
      if (s + b > 0) separation += 0.5*pow(s - b,2)/(s + b);
   }
   separation *= intBin;

   return separation;
}


//_______________________________________________________________________
Double_t TMVA::MethodBase::GetOptimalSignificance(Double_t SignalEvents,
                                                  Double_t BackgroundEvents,
                                                  Double_t & optimal_significance_value  ) const
{
   // plot significance, S/Sqrt(S^2 + B^2), curve for given number
   // of signal and background events; returns cut for optimal significance
   // also returned via reference is the optimal significance

   if (Verbose()) cout << "--- " << GetName() << ": Get optimal significance ..." << endl;

   Double_t optimal_significance(0);
   Double_t effS(0),effB(0),significance(0);
   TH1F *temp_histogram = new TH1F("temp", "temp", fNbinsH, fXmin, fXmax );

   if (SignalEvents <= 0 || BackgroundEvents <= 0) {
      cout << "--- " << GetName() << ": ERROR in "
           << "'TMVA::MethodBase::GetOptimalSignificance'"
           << "number of signal or background events is <= 0 ==> abort"
           << endl;
      exit(1);
   }

   cout << "--- " << GetName() << ": using ratio SignalEvents/BackgroundEvents = "
        << SignalEvents/BackgroundEvents << endl;

   if ((fEffS == 0) || (fEffB == 0)) {
      cout<<"--- "<< GetName() <<": efficiency histograms empty !"<<endl;
      cout<<"--- "<< GetName() <<": no optimal cut, return 0"<<endl;
      return 0;
   }

   for (Int_t bin=1; bin<=fNbinsH; bin++) {
      effS = fEffS->GetBinContent( bin );
      effB = fEffB->GetBinContent( bin );

      // put significance into a histogram
      significance = sqrt(SignalEvents) * ( effS )/sqrt( effS + ( BackgroundEvents / SignalEvents) * effB  );

      temp_histogram->SetBinContent(bin,significance);
   }

   // find maximum in histogram
   optimal_significance = temp_histogram->GetBinCenter( temp_histogram->GetMaximumBin() );
   optimal_significance_value = temp_histogram->GetBinContent( temp_histogram->GetMaximumBin() );

   // delete
   temp_histogram->Delete();

   cout << "--- " << GetName() << ": optimal cut at      : " << optimal_significance << endl;
   cout << "--- " << GetName() << ": optimal significance: " << optimal_significance_value << endl;

   return optimal_significance;
}


//_______________________________________________________________________
Double_t TMVA::MethodBase::GetmuTransform( TTree *theTree )
{
   // computes Mu-transform
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

   vector<Double_t>* aBhatB = new vector<Double_t>;
   vector<Double_t>* aBhatS = new vector<Double_t>;
   Int_t ievt;
   for (ievt=0; ievt<theTree->GetEntries(); ievt++) {
      Double_t x    = TMVA::Tools::GetValue( theTree, ievt, fTestvar );
      Double_t s    = fSplS->GetVal( x );
      Double_t b    = fSplB->GetVal( x );
      Double_t aBhat = 0;
      if (b + s > 0) aBhat = b/(b + s);

      if ((Int_t)TMVA::Tools::GetValue( theTree, ievt, "type" ) == 1) { // this is signal
         aBhatS->push_back ( aBhat );
         fHistBhatS->Fill( aBhat );
      }
      else {
         aBhatB->push_back ( aBhat );
         fHistBhatB->Fill( aBhat );
      }
   }

   // normalize histograms
   fHistBhatS->Scale( 1.0/((fHistBhatS->GetEntries() > 0 ? fHistBhatS->GetEntries() : 1) / nbin) );
   fHistBhatB->Scale( 1.0/((fHistBhatB->GetEntries() > 0 ? fHistBhatB->GetEntries() : 1) / nbin) );

   TMVA::PDF* yB = new TMVA::PDF( fHistBhatB, TMVA::PDF::kSpline2, 100 );

   Int_t nevtS = aBhatS->size();
   Int_t nevtB = aBhatB->size();

   // get the mu-transform
   Int_t nbinMu = 50;
   fHistMuS = new TH1F( fTestvar + "_muTransform_S",
                        fTestvar + ": mu-Transform (S)", nbinMu, 0.0, 1.0 );
   fHistMuB = new TH1F( fTestvar + "_muTransform_B",
                        fTestvar + ": mu-Transform (B)", nbinMu, 0.0, 1.0 );

   // signal
   for (ievt=0; ievt<nevtS; ievt++) {
      Double_t w = yB->GetVal( (*aBhatS)[ievt] );
      if (w > 0) fHistMuS->Fill( 1.0 - (*aBhatS)[ievt], 1.0/w );
   }

   // background (must be flat)
   for (ievt=0; ievt<nevtB; ievt++) {
      Double_t w = yB->GetVal( (*aBhatB)[ievt] );
      if (w > 0) fHistMuB->Fill( 1.0 - (*aBhatB)[ievt], 1.0/w );
   }

   // normalize mu-transforms
   TMVA::Tools::NormHist( fHistMuS );
   TMVA::Tools::NormHist( fHistMuB );

   // determine the mu-transform value, which is defined as
   // the average of the signal mu-transform Int_[0,1] { S(mu) dmu }
   // this average is 0.5 for background, by definition
   TMVA::PDF* thePdf = new TMVA::PDF( fHistMuS, TMVA::PDF::kSpline2 );
   Double_t intS = 0;
   Int_t    nstp = 10000;
   for (Int_t istp=0; istp<nstp; istp++) {
      Double_t x = (istp + 0.5)/Double_t(nstp);
      intS += x*thePdf->GetVal( x );
   }
   intS /= Double_t(nstp);

   delete yB;
   delete thePdf;
   delete aBhatB;
   delete aBhatS;

   return intS; // return average mu-transform for signal
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteHistosToFile( TDirectory* targetDir )
{
   // writes all MVA evaluation histograms to file

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

TMVA::MethodBase* TMVA::MethodBase::fgThisBase = NULL;

//_______________________________________________________________________
Double_t TMVA::MethodBase::IGetEffForRoot( Double_t theCut )
{
   // interface for RootFinder
   return TMVA::MethodBase::GetThisBase()->GetEffForRoot( theCut );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetEffForRoot( Double_t theCut )
{
   // returns efficiency as function of cut

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
   if      (theCut-fXmin < eps) retval = (GetCutOrientation() == kPositive) ? 1.0 : 0.0;
   else if (fXmax-theCut < eps) retval = (GetCutOrientation() == kPositive) ? 0.0 : 1.0;

   return retval;
}

