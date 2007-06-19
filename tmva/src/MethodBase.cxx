// @(#)root/tmva $Id: MethodBase.cxx,v 1.19 2007/05/10 06:36:22 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBase                                                            *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
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
 *                                                                                *
 **********************************************************************************/

//_______________________________________________________________________
//Begin_Html
/*
  Virtual base Class for all MVA method
  MethodBase hosts several specific evaluation methods

  The kind of MVA that provides optimal performance in an analysis strongly
  depends on the particular application. The evaluation factory provides a
  number of numerical benchmark results to directly assess the performance
  of the MVA training on the independent test sample. These are:
  <ul>
  <li> The <i>signal efficiency</i> at three representative background efficiencies
  (which is 1 &minus; rejection).</li>
  <li> The <i>significance</I> of an MVA estimator, defined by the difference
  between the MVA mean values for signal and background, divided by the
  quadratic sum of their root mean squares.</li>
  <li> The <i>separation</i> of an MVA <i>x</i>, defined by the integral
  &frac12;&int;(S(x) &minus; B(x))<sup>2</sup>/(S(x) + B(x))dx, where
  S(x) and B(x) are the signal and background distributions, respectively.
  The separation is zero for identical signal and background MVA shapes,
  and it is one for disjunctive shapes.
  <li> <a name="mu_transform">
  The average, &int;x &mu;(S(x))dx, of the signal &mu;-transform.
  The &mu;-transform of an MVA denotes the transformation that yields
  a uniform background distribution. In this way, the signal distributions
  S(x) can be directly compared among the various MVAs. The stronger S(x)
  peaks towards one, the better is the discrimination of the MVA. The
  &mu;-transform is
  <a href=http://tel.ccsd.cnrs.fr/documents/archives0/00/00/29/91/index_fr.html>documented here</a>.
  </ul>
  The MVA standard output also prints the linear correlation coefficients between
  signal and background, which can be useful to eliminate variables that exhibit too
  strong correlations.
*/
//End_Html
//_______________________________________________________________________

#include <string>
#include <sstream>
#include <fstream>
#include <stdlib.h>

#include "TROOT.h"
#include "TSystem.h"
#include "TObjString.h"
#include "TQObject.h"
#include "TSpline.h"
#include "TMatrix.h"
#include "TMath.h"
#include "TFile.h"
#include "TKey.h"

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif
#ifndef ROOT_TMVA_RootFinder
#include "TMVA/RootFinder.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif
#ifndef ROOT_TMVA_VariableIdentityTransform
#include "TMVA/VariableIdentityTransform.h"
#endif
#ifndef ROOT_TMVA_VariableDecorrTransform
#include "TMVA/VariableDecorrTransform.h"
#endif
#ifndef ROOT_TMVA_VariablePCATransform
#include "TMVA/VariablePCATransform.h"
#endif
#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
#endif

ClassImp(TMVA::MethodBase)

const Int_t    MethodBase_MaxIterations_ = 200;
const Bool_t   Use_Splines_for_Eff_      = kTRUE;

const int      NBIN_HIST_PLOT = 100;
const int      NBIN_HIST_HIGH = 10000;

//_______________________________________________________________________
TMVA::MethodBase::MethodBase( TString      jobName,
                              TString      methodTitle,
                              DataSet&     theData,
                              TString      theOption,
                              TDirectory*  theBaseDir)
   : IMethod(),
     Configurable               ( theOption ),
     fData                      ( theData ),
     fSignalReferenceCut        ( 0.5 ),
     fVariableTransformType     ( Types::kSignal ),
     fJobName                   ( jobName ),
     fMethodName                ( "MethodBase"  ),
     fMethodType                ( Types::kMaxMethod ),
     fMethodTitle               ( methodTitle ),
     fTestvar                   ( "" ),
     fTestvarPrefix             ( "MVA_" ),
     fTMVATrainingVersion       ( TMVA_VERSION_CODE ),
     fROOTTrainingVersion       ( ROOT_VERSION_CODE ),
     fNvar                      ( theData.GetNVariables() ),
     fBaseDir                   ( 0 ),
     fMethodBaseDir             ( theBaseDir ),
     fWeightFile                ( "" ),
     fVarTransform              ( 0 ),
     fTrainEffS                 ( 0 ),
     fTrainEffB                 ( 0 ),
     fTrainEffBvsS              ( 0 ),
     fTrainRejBvsS              ( 0 ),
     fGraphTrainS               ( 0 ),
     fGraphTrainB               ( 0 ),
     fGraphTrainEffBvsS         ( 0 ),
     fSplTrainS                 ( 0 ),
     fSplTrainB                 ( 0 ),
     fSplTrainEffBvsS           ( 0 ),
     fMeanS                     ( -1 ),
     fMeanB                     ( -1 ),
     fRmsS                      ( -1 ),
     fRmsB                      ( -1 ),
     fXmin                      ( 1e30 ),
     fXmax                      ( -1e30 ),
     fVariableTransform         ( Types::kNone ),
     fVerbose                   ( kFALSE ),
     fHelp                      ( kFALSE ),
     fIsMVAPdfs                 ( kFALSE ),
     fTxtWeightsOnly            ( kTRUE ),
     fSplRefS                   ( 0 ),
     fSplRefB                   ( 0 ),
     fLogger                    ( this )
{
   // standard constructur
   this->Init();

   // interpretation of configuration option string
   DeclareOptions();

   // default extension for weight files
   fFileDir = gConfig().GetIONames().fWeightFileDir;
   gSystem->MakeDirectory( fFileDir );
}

//_______________________________________________________________________
TMVA::MethodBase::MethodBase( DataSet&     theData,
                              TString      weightFile,
                              TDirectory*  theBaseDir )
   : IMethod(),
     Configurable(""),
     fData                      ( theData ),
     fSignalReferenceCut        ( 0.5 ),
     fVariableTransformType     ( Types::kSignal ),
     fJobName                   ( "" ),
     fMethodName                ( "MethodBase"  ),
     fMethodType                ( Types::kMaxMethod ),
     fMethodTitle               ( "" ),
     fTestvar                   (""),
     fTestvarPrefix             ("MVA_"),
     fTMVATrainingVersion       ( 0 ),
     fROOTTrainingVersion       ( 0 ),
     fNvar                      ( theData.GetNVariables() ),
     fBaseDir                   ( theBaseDir ),
     fMethodBaseDir             ( 0 ),
     fWeightFile                ( weightFile ),
     fVarTransform              ( 0 ),
     fTrainEffS                 ( 0 ),
     fTrainEffB                 ( 0 ),
     fTrainEffBvsS              ( 0 ),
     fTrainRejBvsS              ( 0 ),
     fGraphTrainS               ( 0 ),
     fGraphTrainB               ( 0 ),
     fGraphTrainEffBvsS         ( 0 ),
     fSplTrainS                 ( 0 ),
     fSplTrainB                 ( 0 ),
     fSplTrainEffBvsS           ( 0 ),
     fMeanS                     ( -1 ),
     fMeanB                     ( -1 ),
     fRmsS                      ( -1 ),
     fRmsB                      ( -1 ),
     fXmin                      ( 1e30 ),
     fXmax                      ( -1e30 ),
     fVariableTransform         ( Types::kNone ),
     fVerbose                   ( kFALSE ),
     fHelp                      ( kFALSE ),
     fIsMVAPdfs                 ( kFALSE ),
     fTxtWeightsOnly            ( kTRUE ),
     fSplRefS                   ( 0 ),
     fSplRefB                   ( 0 ),
     fLogger                    ( this )
{
   // constructor used for Testing + Application of the MVA,
   // only (no training), using given WeightFiles

   this->Init();

   DeclareOptions();
}

//_______________________________________________________________________
TMVA::MethodBase::~MethodBase( void )
{
   // default destructur
   if (fInputVars != 0) { fInputVars->clear(); delete fInputVars; }
   if (fRanking   != 0) delete fRanking;
   if (fMVAPdfS   != 0) delete fMVAPdfS;
   if (fMVAPdfB   != 0) delete fMVAPdfB;
}

//_______________________________________________________________________
void TMVA::MethodBase::Init()
{
   // default initialization called by all constructors
   fIsOK            = kTRUE;

   fNbins           = NBIN_HIST_PLOT;
   fNbinsH          = NBIN_HIST_HIGH;

   fRanking         = NULL;

   fHistS_plotbin   = NULL;
   fHistB_plotbin   = NULL;
   fProbaS_plotbin  = NULL;
   fProbaB_plotbin  = NULL;
   fRarityS_plotbin = NULL;
   fRarityB_plotbin = NULL;
   fHistS_highbin   = NULL;
   fHistB_highbin   = NULL;
   fEffS            = NULL;
   fEffB            = NULL;
   fEffBvsS         = NULL;
   fRejBvsS         = NULL;
   finvBeffvsSeff   = NULL;
   fHistBhatS       = NULL;
   fHistBhatB       = NULL;
   fHistMuS         = NULL;
   fHistMuB         = NULL;
   fMVAPdfS         = NULL;
   fMVAPdfB         = NULL;

   // temporary until the move to DataSet is complete
   fInputVars = new vector<TString>;
   for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++)
      fInputVars->push_back(Data().GetInternalVarName(ivar));

   // define "this" pointer
   ResetThisBase();
}

//_______________________________________________________________________
void TMVA::MethodBase::DeclareOptions()
{
   // define the options (their key words) that can be set in the option string
   // here the options valid for ALL MVA methods are declared.
   // know options: VariableTransform=None,Decorrelated,PCA  to use transformed variables
   //                                                        instead of the original ones
   //               VariableTransformType=Signal,Background  which decorrelation matrix to use
   //                                                        in the method. Only the Likelihood
   //                                                        Method can make proper use of independent
   //                                                        transformations of signal and background
   //               fNbinsMVAPdf   = 50 Number of bins used to create a PDF of MVA
   //               fNsmoothMVAPdf =  2 Number of times a histogram is smoothed before creating the PDF
   //               fIsMVAPdfs          create PDFs for the MVA outputs
   //               V                   for Verbose output (!V) for non verbos
   //               H                   for Help message

   DeclareOptionRef( fUseDecorr=kFALSE, "D", "use-decorrelated-variables flag (depreciated)" );

   DeclareOptionRef( fNormalise=kFALSE, "Normalise", "Normalise input variables" ); // don't change the default !!!

   DeclareOptionRef( fVarTransformString="None", "VarTransform", "Variable transformation method" );
   AddPreDefVal( TString("None") );
   AddPreDefVal( TString("Decorrelate") );
   AddPreDefVal( TString("PCA") );

   DeclareOptionRef( fVariableTransformTypeString="Signal", "VarTransformType",
                     "Use signal or background events for var transform" );
   AddPreDefVal( TString("Signal") );
   AddPreDefVal( TString("Background") );

   DeclareOptionRef( fNbinsMVAPdf   = 60, "NbinsMVAPdf",   "Number of bins used to create MVA PDF" );
   DeclareOptionRef( fNsmoothMVAPdf = 2,  "NsmoothMVAPdf", "Number of smoothing iterations for MVA PDF" );

   DeclareOptionRef( fVerbose,   "V",       "Verbose mode" );

   DeclareOptionRef( fVerbosityLevelString="Info", "VerboseLevel", "Verbosity level" );
   AddPreDefVal( TString("Debug") );
   AddPreDefVal( TString("Verbose") );
   AddPreDefVal( TString("Info") );
   AddPreDefVal( TString("Warning") );
   AddPreDefVal( TString("Error") );
   AddPreDefVal( TString("Fatal") );

   DeclareOptionRef( fHelp,             "H",             "Print classifier-specific help message" );
   DeclareOptionRef( fIsMVAPdfs=kFALSE, "CreateMVAPdfs", "Create PDFs for classifier outputs" );

   DeclareOptionRef( fTxtWeightsOnly=kTRUE, "TxtWeightFilesOnly", "if True, write all weights as text files" );
}

//_______________________________________________________________________
void TMVA::MethodBase::ProcessOptions()
{
   // the option string is decoded, for availabel options see "DeclareOptions"

   if      (fVarTransformString == "None")         fVariableTransform = Types::kNone;
   else if (fVarTransformString == "Decorrelate" ) fVariableTransform = Types::kDecorrelated;
   else if (fVarTransformString == "PCA" )         fVariableTransform = Types::kPCA;
   else {
      fLogger << kFATAL << "<ProcessOptions> Variable transform '"
              << fVarTransformString << "' unknown." << Endl;
   }

   // for backward compatibility
   if ((fVariableTransform == Types::kNone) && fUseDecorr) fVariableTransform = Types::kDecorrelated;

   if      (fVariableTransformTypeString == "Signal")      fVariableTransformType = Types::kSignal;
   else if (fVariableTransformTypeString == "Background" ) fVariableTransformType = Types::kBackground;
   else {
      fLogger << kFATAL << "<ProcessOptions> Variable transformation type '"
              << fVariableTransformTypeString << "' unknown." << Endl;
   }

   // retrieve variable transformer
   if (fVarTransform == 0) fVarTransform = Data().GetTransform( fVariableTransform );

   if      (fVerbosityLevelString == "Debug"   ) fLogger.SetMinType( kDEBUG );
   else if (fVerbosityLevelString == "Verbose" ) fLogger.SetMinType( kVERBOSE );
   else if (fVerbosityLevelString == "Info"    ) fLogger.SetMinType( kINFO );
   else if (fVerbosityLevelString == "Warning" ) fLogger.SetMinType( kWARNING );
   else if (fVerbosityLevelString == "Error"   ) fLogger.SetMinType( kERROR );
   else if (fVerbosityLevelString == "Fatal"   ) fLogger.SetMinType( kFATAL );
   else {
      fLogger << kFATAL << "<ProcessOptions> Verbosity level type '"
              << fVerbosityLevelString << "' unknown." << Endl;
   }

   if (Verbose()) fLogger.SetMinType( kVERBOSE );
}

//_______________________________________________________________________
void TMVA::MethodBase::PrintHelpMessage() const
{
   // prints out classifier-specific help method

   //         "|--------------------------------------------------------------|"
   fLogger << kINFO << Endl;
   fLogger << Tools::Color("bold")
           << "================================================================" 
           << Tools::Color( "reset" )
           << Endl;
   fLogger << Tools::Color("bold")
           << "H e l p   f o r   c l a s s i f i e r   [ " << GetName() << " ] :" 
           << Tools::Color( "reset" )
           << Endl;

   // print method-specific help message
   GetHelpMessage(); 

   fLogger << Endl;
   fLogger << "<Suppress this message by specifying \"!H\" in the booking option>" << Endl;
   fLogger << Tools::Color("bold")
           << "================================================================" 
           << Tools::Color( "reset" )
           << Endl;
   fLogger << Endl;
}

//_______________________________________________________________________
void TMVA::MethodBase::TrainMethod()
{
   // train the classifier method
   if (Help()) PrintHelpMessage();

   // all histograms should be created in the method's subdirectory
   BaseDir()->cd();

   // call training of derived classifier
   Train();

   // create PDFs for the signal and background MVA distributions
   if (IsMVAPdfs()) CreateMVAPdfs();

   // write the current classifier state into stream
   // produced are one text file and one ROOT file
   WriteStateToFile();

   // produce standalone make class
   MakeClass();

   // write additional monitoring histograms to main target file (not the weight file)
   // again, make sure the histograms go into the method's subdirectory
   BaseDir()->cd();

   WriteMonitoringHistosToFile();
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetProba( Double_t mvaVal, Double_t ap_sig )
{
   // compute likelihood ratio
   if (!fMVAPdfS || !fMVAPdfB) {
      fLogger << kWARNING << "<GetProba> MVA PDFs for Signal and Backgroud don't exist" << Endl;
      return 0;
   }
   Double_t p_s = fMVAPdfS->GetVal( mvaVal );
   Double_t p_b = fMVAPdfB->GetVal( mvaVal );

   Double_t denom = p_s*ap_sig + p_b*(1 - ap_sig);

   return (denom > 0) ? (p_s*ap_sig) / denom : -1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetRarity( Double_t mvaVal, Types::ESBType reftype ) const
{
   // compute rarity:
   // R(x) = Integrate_[-oo..x] { PDF(x') dx' }
   // where PDF(x) is the PDF of the classifier's signal or background distribution

   if ((reftype == Types::kSignal && !fMVAPdfS) || (reftype == Types::kBackground && !fMVAPdfB)) {
      fLogger << kWARNING << "<GetRarity> Required MVA PDF for Signal or Backgroud does not exist: "
              << "select option \"CreateMVAPdfs\"" << Endl;
      return 0.0;
   }

   PDF* thePdf = reftype == Types::kSignal ? fMVAPdfS : fMVAPdfB;

   return thePdf->GetIntegral( thePdf->GetXmin(), mvaVal );
}

//_______________________________________________________________________
void TMVA::MethodBase::CreateMVAPdfs()
{
   // Create PDFs of the MVA output variables

   fLogger << kINFO << "<CreateMVAPdfs> Using " << fNbinsMVAPdf << " bins and smooth "
           << fNsmoothMVAPdf << " times" << Endl;

   vector<Double_t>* sigVec = new vector<Double_t>;
   vector<Double_t>* bkgVec = new vector<Double_t>;

   Double_t minVal=9999;
   Double_t maxVal=-9999;
   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {
      ReadTrainingEvent(ievt);
      Double_t theVal = this->GetMvaValue();
      if (minVal>theVal) minVal = theVal;
      if (maxVal<theVal) maxVal = theVal;
      if (IsSignalEvent()) sigVec->push_back(theVal);
      else                 bkgVec->push_back(theVal);
   }

   // create histograms that serve as basis to create the MVA Pdfs
   TH1* histMVAPdfS = new TH1F( GetMethodName() + "_tr_S", GetMethodName() + "_tr_S",
                                fNbinsMVAPdf, minVal, maxVal );
   TH1* histMVAPdfB = new TH1F( GetMethodName() + "_tr_B", GetMethodName() + "_tr_B",
                                fNbinsMVAPdf, minVal, maxVal );

   // compute sum of weights properly
   histMVAPdfS->Sumw2();
   histMVAPdfB->Sumw2();

   // fill histograms
   for (Int_t i=0; i<(Int_t)sigVec->size(); i++) histMVAPdfS->Fill( (*sigVec)[i] );
   for (Int_t i=0; i<(Int_t)bkgVec->size(); i++) histMVAPdfB->Fill( (*bkgVec)[i] );

   // cleanup
   delete sigVec;
   delete bkgVec;

   // normalisation
   Tools::NormHist( histMVAPdfS );
   Tools::NormHist( histMVAPdfB );

   // momentary hack for ROOT problem
   histMVAPdfS->Write();
   histMVAPdfB->Write();

   // create PDFs
   fMVAPdfS = new PDF( histMVAPdfS, PDF::kSpline2, fNsmoothMVAPdf );
   fMVAPdfB = new PDF( histMVAPdfB, PDF::kSpline2, fNsmoothMVAPdf );

   fMVAPdfS->ValidatePDF( histMVAPdfS );
   fMVAPdfB->ValidatePDF( histMVAPdfB );

   fLogger << kINFO
           << Form( "<CreateMVAPdfs> Separation from histogram (PDF): %1.3f (%1.3f)",
                    GetSeparation( histMVAPdfS, histMVAPdfB ), GetSeparation( fMVAPdfS, fMVAPdfB ) )
           << Endl;

   delete histMVAPdfS;
   delete histMVAPdfB;
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteStateToStream( std::ostream& tf, Bool_t isClass ) const
{
   // general method used in writing the header of the weight files where
   // the used variables, variable transformation type etc. is specified

   TString prefix = "";

   tf << prefix << "#GEN -*-*-*-*-*-*-*-*-*-*-*- general info -*-*-*-*-*-*-*-*-*-*-*-" << endl << prefix << endl;
   tf << prefix << "Method         : " << GetMethodName() << "::" << GetMethodTitle() << endl;
   tf.setf(std::ios::left);
   tf << prefix << "TMVA Release   : " << setw(10) << GetTrainingTMVAVersionString() << "    [" << GetTrainingTMVAVersionCode() << "]" << endl;
   tf << prefix << "ROOT Release   : " << setw(10) << GetTrainingROOTVersionString() << "    [" << GetTrainingROOTVersionCode() << "]" << endl;
   tf << prefix << "Creator        : " << gSystem->GetUserInfo()->fUser << endl;
   tf << prefix << "Date           : "; TDatime *d = new TDatime; tf << d->AsString() << endl; delete d;
   tf << prefix << "Host           : " << gSystem->GetBuildNode() << endl;
   tf << prefix << "Dir            : " << gSystem->WorkingDirectory() << endl;
   tf << prefix << "Training events: " << Data().GetNEvtTrain() << endl;
   tf << prefix << endl;

   // First write all options
   tf << prefix << endl << prefix << "#OPT -*-*-*-*-*-*-*-*-*-*-*-*- options -*-*-*-*-*-*-*-*-*-*-*-*-" << endl << prefix << endl;
   WriteOptionsToStream( tf, prefix );
   tf << prefix << endl;

   // Second write variable info
   tf << prefix << endl << prefix << "#VAR -*-*-*-*-*-*-*-*-*-*-*-* variables *-*-*-*-*-*-*-*-*-*-*-*-" << endl << prefix << endl;
   GetVarTransform().WriteVarsToStream( tf, prefix );
   tf << prefix << endl;

   // Third write transformation data if available
   // this is weight file specific information: not written if class is made
   if (!isClass) {
      tf << endl << "#MAT -*-*-*-*-*-*-*-*-* transformation data -*-*-*-*-*-*-*-*-*-" << endl;
      GetVarTransform().WriteTransformationToStream( tf );
      tf << endl;

      // Fourth write the MVA variable distributions
      if (IsMVAPdfs()) {
         tf << endl << "#MVAPDFS -*-*-*-*-*-*-*-*-*-*-* MVA PDFS -*-*-*-*-*-*-*-*-*-*-*-" << endl;
         tf << *fMVAPdfS << endl;
         tf << *fMVAPdfB << endl;
         tf << endl;
      }

      // Lst, write weights
      tf << endl << "#WGT -*-*-*-*-*-*-*-*-*-*-*-*- weights -*-*-*-*-*-*-*-*-*-*-*-*-" << endl << endl;

      // no weights when standalone class is produced, which would be duplication
      WriteWeightsToStream( tf ); 
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteStateToStream( TFile& rf ) const
{
   // write reference MVA distributions (and other information)
   // to a ROOT type weight file

   rf.cd();
   if (fMVAPdfS && fMVAPdfB) {
      fMVAPdfS->Write("MVA_PDF_Signal");
      fMVAPdfB->Write("MVA_PDF_Background");
   }

   WriteWeightsToStream( rf );
}

//_______________________________________________________________________
void TMVA::MethodBase::ReadStateFromStream( TFile& rf )
{
   // write reference MVA distributions (and other information)
   // to a ROOT type weight file

   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory( 0 ); // this avoids the binding of the hists in PDF to the current ROOT file
   fMVAPdfS = (PDF*)rf.Get( "MVA_PDF_Signal" );
   fMVAPdfB = (PDF*)rf.Get( "MVA_PDF_Background" );
   TH1::AddDirectory( addDirStatus );

   ReadWeightsFromStream( rf );
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteStateToFile() const
{
   // write options and weights to file
   // note that each one text file for the main configuration information
   // and one ROOT file for ROOT objects are created

   // ---- create the text file
   TString tfname( GetWeightFileName() );
   fLogger << kINFO << "Creating weight file in text format: "
           << Tools::Color("lightblue") << tfname << Tools::Color("reset") << Endl;

   ofstream tfile( tfname );
   if (!tfile.good()) { // file could not be opened --> Error
      fLogger << kFATAL << "<WriteStateToFile> "
              << "Unable to open output weight file: " << tfname << Endl;
   }
   WriteStateToStream( tfile );

   tfile.close();

   if ( ! fTxtWeightsOnly ) {
      // ---- create the ROOT file
      TString rfname( tfname ); rfname.ReplaceAll( ".txt", ".root" );
      fLogger << kINFO << "Creating weight file in root format: "
              << Tools::Color("lightblue") << rfname << Tools::Color("reset") << Endl;
      TFile* rfile = TFile::Open( rfname, "RECREATE" );
      WriteStateToStream( *rfile );
      rfile->Close();
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::ReadStateFromFile()
{
   // Function to write options and weights to file

   // get the filename
   TString tfname(GetWeightFileName());

   fLogger << kINFO << "Reading weight file: "
           << Tools::Color("lightblue") << tfname << Tools::Color("reset") << Endl;

   ifstream fin( tfname );
   if (!fin.good()) { // file not found --> Error
      fLogger << kFATAL << "<ReadStateFromFile> "
                 << "Unable to open input weight file: " << tfname << Endl;
   }

   ReadStateFromStream(fin);
   fin.close();

   if (!fTxtWeightsOnly) {
      // ---- read the ROOT file
      TString rfname( tfname ); rfname.ReplaceAll( ".txt", ".root" );
      fLogger << kINFO << "Reading root weight file: "
              << Tools::Color("lightblue") << rfname << Tools::Color("reset") << Endl;
      TFile* rfile = TFile::Open( rfname, "READ" );
      ReadStateFromStream( *rfile );
      rfile->Close();
   }
}

//_______________________________________________________________________
bool TMVA::MethodBase::GetLine(std::istream& fin, char * buf )
{
   // reads one line from the input stream
   // checks for certain keywords and interprets
   // the line if keywords are found
   fin.getline(buf,512);
   TString line(buf);
   if (line.BeginsWith("TMVA Release")) {
      Ssiz_t start = line.First('[')+1;
      Ssiz_t length = line.Index("]",start)-start;
      TString code = line(start,length);
      std::stringstream s(code.Data());
      s >> fTMVATrainingVersion;
      fLogger << kINFO << "Classifier was trained with TMVA Version: " << GetTrainingTMVAVersionString() << Endl;
   }
   if (line.BeginsWith("ROOT Release")) {
      Ssiz_t start = line.First('[')+1;
      Ssiz_t length = line.Index("]",start)-start;
      TString code = line(start,length);
      std::stringstream s(code.Data());
      s >> fROOTTrainingVersion;
      fLogger << kINFO << "Classifier was trained with ROOT Version: " << GetTrainingROOTVersionString() << Endl;
   }

   return true;
}

//_______________________________________________________________________
void TMVA::MethodBase::ReadStateFromStream( std::istream& fin )
{
   // read the header from the weight files of the different MVA methods
   char buf[512];

   // first read the method name
   GetLine(fin,buf);
   while (!TString(buf).BeginsWith("Method")) GetLine(fin,buf);
   TString ls(buf);
   Int_t idx1 = ls.First(':')+2; Int_t idx2 = ls.Index(' ',idx1)-idx1; if (idx2<0) idx2=ls.Length();
   TString fullname = ls(idx1,idx2);
   idx1 = fullname.First(':');
   Int_t idxtit = (idx1<0 ? fullname.Length() : idx1);
   TString methodName  = fullname(0, idxtit);
   TString methodTitle;
   Bool_t notit;
   if (idx1<0) {
      methodTitle=methodName;
      notit=kTRUE;
   } 
   else {
      methodTitle=fullname(idxtit+2,fullname.Length()-1);
      notit=kFALSE;
   }
   fLogger << kINFO << "Read method with name <" << methodName << "> and title <" << methodTitle << ">" << Endl;
   this->SetMethodName( methodName );
   this->SetMethodTitle( methodTitle );

   // now the question is whether to read the variables first or the options (well, of course the order
   // of writing them needs to agree)
   //
   // the option "Decorrelation" is needed to decide if the variables we
   // read are decorrelated or not
   //
   // the variables are needed by some methods (TMLP) to build the NN
   // which is done in ProcessOptions so for the time being we first Read and Parse the options then
   // we read the variables, and then we process the options

   // now read all options
   GetLine(fin,buf);
   while (!TString(buf).BeginsWith("#OPT")) GetLine(fin,buf);

   ReadOptionsFromStream(fin);
   ParseOptions(Verbose());

   fLogger << kINFO << "Create VariableTransformation \"" << fVarTransformString << "\"" << Endl;
   if (fVarTransformString == "None" ) {
      fVarTransform = new VariableIdentityTransform( Data().GetVariableInfos() );
   }
   else if (fVarTransformString == "Decorrelate" ) {
      fVarTransform = new VariableDecorrTransform( Data().GetVariableInfos() );
   }
   else if (fVarTransformString == "PCA" ) {
      fVarTransform = new VariablePCATransform( Data().GetVariableInfos() );
   }

   // Now read variable info
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("#VAR")) fin.getline(buf,512);
   GetVarTransform().ReadVarsFromStream(fin);

   // now we process the options (of the derived class)
   ProcessOptions();

   // Now read decorrelation matrix if available
   if (GetVariableTransform() != Types::kNone) {
      fin.getline(buf,512);
      while (!TString(buf).BeginsWith("#MAT")) fin.getline(buf,512);
      GetVarTransform().ReadTransformationFromStream(fin);
   }

   if (IsMVAPdfs()) {
      // Now read the MVA PDFs
      fin.getline(buf,512);
      while (!TString(buf).BeginsWith("#MVAPDFS")) fin.getline(buf,512);
      if (fMVAPdfS!=0) { delete fMVAPdfS; fMVAPdfS = 0; }
      if (fMVAPdfB!=0) { delete fMVAPdfB; fMVAPdfB = 0; }
      fMVAPdfS = new PDF();
      fMVAPdfB = new PDF();
      fMVAPdfS->SetReadingVersion( GetTrainingTMVAVersionCode() );
      fMVAPdfB->SetReadingVersion( GetTrainingTMVAVersionCode() );
      fin >> *fMVAPdfS;
      fin >> *fMVAPdfB;
   }

   // Now read weights
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("#WGT")) fin.getline(buf,512);
   fin.getline(buf,512);
   
   ReadWeightsFromStream( fin );
}

//_______________________________________________________________________
TDirectory* TMVA::MethodBase::BaseDir() const
{
   // returns the ROOT directory where info/histograms etc of the
   // corresponding MVA method instance are stored

   if (fBaseDir != 0) {
      return fBaseDir;
   }
   TDirectory* methodDir = MethodBaseDir();
   if (methodDir==0) {
      fLogger << kFATAL << "MethodBase::BaseDir() - MethodBaseDir() return a NUL pointer!" << Endl;
   }
   TDirectory* dir = 0;

   TString defaultDir = GetMethodTitle();

   TObject* o = methodDir->FindObject(defaultDir);
   if (o!=0 && o->InheritsFrom("TDirectory")) dir = (TDirectory*)o;
   if (dir != 0) {
      return dir;
   }

   TDirectory *sdir = methodDir->mkdir(defaultDir);
   return sdir;
}

//_______________________________________________________________________
TDirectory* TMVA::MethodBase::MethodBaseDir() const
{
   // returns the ROOT directory where all instances of the
   // corresponding MVA method are stored

   if (fMethodBaseDir != 0) return fMethodBaseDir;

   TDirectory* dir = 0;

   TString defaultDir = "Method_" + GetMethodName();

   TObject* o = Data().BaseRootDir()->FindObject(defaultDir);
   if (o!=0 && o->InheritsFrom("TDirectory")) dir = (TDirectory*)o;
   if (dir != 0) return dir;

   return Data().BaseRootDir()->mkdir(defaultDir);
}

//_______________________________________________________________________
void TMVA::MethodBase::SetWeightFileName( TString theWeightFile)
{
   // set the weight file name (depreciated)
   fWeightFile = theWeightFile;
}

//_______________________________________________________________________
TString TMVA::MethodBase::GetWeightFileName() const
{
   // retrieve weight file name
   if (fWeightFile!="") return fWeightFile;

   // the default consists of
   // directory/jobname_methodname_suffix.extension.{root/txt}
   TString suffix = "";
   return  fFileDir + "/" + fJobName + "_" + GetMethodTitle() + suffix + "." + gConfig().GetIONames().fWeightFileExtension + ".txt";
}

//_______________________________________________________________________
Bool_t TMVA::MethodBase::CheckSanity( TTree* theTree )
{
   // tree sanity checks

   // if no tree is given, use the trainingTree
   TTree* tree = (0 != theTree) ? theTree : Data().GetTrainingTree();

   // the input variables must exist in the tree
   for (Int_t i=0; i<GetNvar(); i++)
      if (0 == tree->FindBranch( GetInputVar(i) )) return kFALSE;

   return kTRUE;
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
   return Tools::NormVariable( x, GetXmin( var ), GetXmax( var ) );
}

//______________________________________________________________________
Double_t TMVA::MethodBase::Norm( Int_t ivar, Double_t x ) const
{
   // renormalises variable with respect to its min and max
   return Tools::NormVariable( x, GetXmin( ivar ), GetXmax( ivar ) );
}

//_______________________________________________________________________
void TMVA::MethodBase::TestInit(TTree* theTestTree)
{
   // initialization of MVA testing
   if (theTestTree == 0)
      theTestTree = Data().GetTestTree(); // sets theTestTree to the TestTree in the DataSet,
   // decorrelation properly taken care of

   fHistS_plotbin   = fHistB_plotbin = 0;
   fProbaS_plotbin  = fProbaB_plotbin = 0;
   fRarityS_plotbin = fRarityB_plotbin = 0;
   fHistS_highbin   = fHistB_highbin = 0;
   fEffS            = fEffB = fEffBvsS = fRejBvsS = 0;
   fGraphS          = fGraphB = 0;
   fCutOrientation  = kPositive;
   fSplS            = fSplB = 0;
   fSplRefS         = fSplRefB = 0;

   // sanity checks: tree must exist, and theVar must be in tree
   if (0 == theTestTree) {
      fLogger << kFATAL << "<TestInit> Test tree has zero pointer " << Endl;
      fIsOK = kFALSE;
   }
   if ( 0 == theTestTree->FindBranch( GetTestvarName() ) && !(GetMethodName().Contains("Cuts"))) {
      fLogger << kFATAL << "<TestInit> Test variable " << GetTestvarName()
              << " not found in tree" << Endl;
      fIsOK = kFALSE;
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::PrepareEvaluationTree( TTree* testTree )
{
   // prepare tree branch with the method's discriminating variable

   if (0 == testTree) testTree = Data().GetTestTree();

   // sanity checks
   // checks that all variables in input vector indeed exist in the testTree
   if (!CheckSanity( testTree )) {
      fLogger << kFATAL << "<PrepareEvaluationTree> Sanity check failed" << Endl;
   }

   // read the coefficients
   this->ReadStateFromFile();

   // use timer
   Timer timer( testTree->GetEntries(), GetName(), kTRUE );
   Data().BaseRootDir()->cd();
   Float_t myMVA   = 0;
   Float_t myProba = 0;
   TBranch* newBranchMVA   = testTree->Branch( GetTestvarName(), &myMVA, GetTestvarName() + "/F", 128000 );
   newBranchMVA->SetFile(testTree->GetDirectory()->GetFile());
   TBranch* newBranchProba = 0;
   if (IsMVAPdfs()) {
      newBranchProba = testTree->Branch( GetProbaName(), &myProba, GetProbaName() + "/F", 128000 );
      newBranchProba->SetFile(testTree->GetDirectory()->GetFile());
   }

   fLogger << kINFO << "Preparing evaluation tree... " << Endl;
   for (Int_t ievt=0; ievt<testTree->GetEntries(); ievt++) {

      ReadTestEvent(ievt);
      
      // fill the MVA output value for this event
      newBranchMVA->SetAddress( &myMVA ); // only when the tree changed, but we don't know when that is
      myMVA = (Float_t)GetMvaValue();
      newBranchMVA->Fill();

      // fill corresponding signal probabilities
      if (newBranchProba) {
         newBranchProba->SetAddress( &myProba ); // only when the tree changed, but we don't know when that is
         myProba = (Float_t)GetProba( myMVA, 0.5 );
         newBranchProba->Fill();
      }

      // print progress
      Int_t modulo = Int_t(testTree->GetEntries()/100);
      if (ievt%modulo == 0) timer.DrawProgressBar( ievt );
   }

   Data().BaseRootDir()->Write("",TObject::kOverwrite);

   fLogger << kINFO << "Elapsed time for evaluation of "
           << testTree->GetEntries() <<  " events: "
           << timer.GetElapsedTime() << "       " << Endl;

   newBranchMVA  ->ResetAddress();
   if (newBranchProba) newBranchProba->ResetAddress();
}

//_______________________________________________________________________
void TMVA::MethodBase::Test( TTree *theTestTree )
{
   // test the method - not much is done here... mainly furthor initialization

   // If Empty tree: sets theTestTree to the TestTree in the DataSet,
   // decorrelation properly taken care of
   if (theTestTree == 0) theTestTree = Data().GetTestTree();

   // basic statistics operations are made in base class
   // note: cannot directly modify private class members
   Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;

   TMVA::Tools::ComputeStat( theTestTree, GetTestvarName(), meanS, meanB, rmsS, rmsB, xmin, xmax );

   // choose reasonable histogram ranges, by removing outliers
   Double_t nrms = 10;
   xmin = TMath::Max( TMath::Min(meanS - nrms*rmsS, meanB - nrms*rmsB ), xmin );
   xmax = TMath::Min( TMath::Max(meanS + nrms*rmsS, meanB + nrms*rmsB ), xmax );

   fMeanS = meanS; fMeanB = meanB;
   fRmsS  = rmsS;  fRmsB  = rmsB;
   fXmin  = xmin;  fXmax  = xmax;

   // determine cut orientation
   fCutOrientation = (fMeanS > fMeanB) ? kPositive : kNegative;

   // fill 2 types of histograms for the various analyses
   // this one is for actual plotting

   Double_t sxmax = fXmax+0.00001;
   if (fHistS_plotbin  ) { delete fHistS_plotbin;   fHistS_plotbin   = 0; }
   if (fHistB_plotbin  ) { delete fHistB_plotbin;   fHistB_plotbin   = 0; }
   if (fProbaS_plotbin ) { delete fProbaS_plotbin;  fProbaS_plotbin  = 0; }
   if (fProbaB_plotbin ) { delete fProbaB_plotbin;  fProbaB_plotbin  = 0; }
   if (fRarityS_plotbin) { delete fRarityS_plotbin; fRarityS_plotbin = 0; }
   if (fRarityB_plotbin) { delete fRarityB_plotbin; fRarityB_plotbin = 0; }
   if (fHistS_highbin  ) { delete fHistS_highbin;   fHistS_highbin   = 0; }
   if (fHistB_highbin  ) { delete fHistB_highbin;   fHistB_highbin   = 0; }

   fHistS_plotbin  = new TH1F( GetTestvarName() + "_S",GetTestvarName() + "_S", fNbins, fXmin, sxmax );
   fHistB_plotbin  = new TH1F( GetTestvarName() + "_B",GetTestvarName() + "_B", fNbins, fXmin, sxmax );

   if (IsMVAPdfs()) {
      fProbaS_plotbin = new TH1F( GetTestvarName() + "_Proba_S",GetTestvarName() + "_Proba_S", fNbins, 0.0, 1.0 );
      fProbaB_plotbin = new TH1F( GetTestvarName() + "_Proba_B",GetTestvarName() + "_Proba_B", fNbins, 0.0, 1.0 );

      fRarityS_plotbin = new TH1F( GetTestvarName() + "_Rarity_S",GetTestvarName() + "_Rarity_S", fNbins, 0.0, 1.0 );
      fRarityB_plotbin = new TH1F( GetTestvarName() + "_Rarity_B",GetTestvarName() + "_Rarity_B", fNbins, 0.0, 1.0 );
   }
   fHistS_highbin  = new TH1F( GetTestvarName() + "_S_high",GetTestvarName() + "_S_high", fNbinsH, fXmin, sxmax );
   fHistB_highbin  = new TH1F( GetTestvarName() + "_B_high",GetTestvarName() + "_B_high", fNbinsH, fXmin, sxmax );

   // enable quadratic errors
   fHistS_plotbin ->Sumw2();
   fHistB_plotbin ->Sumw2();
   if (IsMVAPdfs()) {
      fProbaS_plotbin->Sumw2();
      fProbaB_plotbin->Sumw2();

      fRarityS_plotbin->Sumw2();
      fRarityB_plotbin->Sumw2();
   }
   fHistS_highbin ->Sumw2();
   fHistB_highbin ->Sumw2();

   // fill the histograms
   theTestTree->ResetBranchAddresses();
   Float_t  v, p;
   Float_t  w;
   Int_t    t;
   TBranch* vbranch = theTestTree->GetBranch( GetTestvarName() );
   TBranch* pbranch = 0;
   if (IsMVAPdfs()) pbranch = theTestTree->GetBranch( GetProbaName() );
   TBranch* wbranch = theTestTree->GetBranch( "weight" );
   TBranch* tbranch = theTestTree->GetBranch( "type" );
   if (!vbranch || !wbranch || !tbranch)
      fLogger << kFATAL << "<Test> Mismatch in test tree: "
              << vbranch << " " << pbranch << " " << wbranch << " " << tbranch << Endl;

   vbranch->SetAddress( &v );
   if (pbranch) pbranch->SetAddress( &p );
   wbranch->SetAddress( &w );
   tbranch->SetAddress( &t );   
   fLogger << kINFO << "Loop over test events and fill histograms with classifier response ..." << Endl;
   if (pbranch) fLogger << kINFO << "Also filling probability and rarity histograms (on request) ..." << Endl;
   for (Int_t ievt=0; ievt<Data().GetNEvtTest(); ievt++) {

      theTestTree->GetEntry(ievt);

      if (t == 1) {
         fHistS_plotbin ->Fill( v, w );
         if (pbranch) {
            fProbaS_plotbin ->Fill( p, w );
            fRarityS_plotbin->Fill( GetRarity( v ), w );
         }

         fHistS_highbin ->Fill( v, w );
      }
      else {
         fHistB_plotbin ->Fill( v, w );
         if (pbranch) {
            fProbaB_plotbin->Fill( p, w );
            fRarityB_plotbin->Fill( GetRarity( v ), w );
         }

         fHistB_highbin ->Fill( v, w );
      }
   }

   theTestTree->ResetBranchAddresses();

   Tools::NormHist( fHistS_plotbin  );
   Tools::NormHist( fHistB_plotbin  );
   if (pbranch) {
      Tools::NormHist( fProbaS_plotbin );
      Tools::NormHist( fProbaB_plotbin );

      Tools::NormHist( fRarityS_plotbin );
      Tools::NormHist( fRarityB_plotbin );
   }
   Tools::NormHist( fHistS_highbin  );
   Tools::NormHist( fHistB_highbin  );

   fHistS_plotbin ->SetDirectory(0);
   fHistB_plotbin ->SetDirectory(0);
   if (pbranch) {
      fProbaS_plotbin->SetDirectory(0);
      fProbaB_plotbin->SetDirectory(0);

      fRarityS_plotbin->SetDirectory(0);
      fRarityB_plotbin->SetDirectory(0);
   }
   fHistS_highbin ->SetDirectory(0);
   fHistB_highbin ->SetDirectory(0);

   // create PDFs from histograms, using default splines, and no additional smoothing
   fSplS = new PDF( fHistS_plotbin, PDF::kSpline2 );
   fSplB = new PDF( fHistB_plotbin, PDF::kSpline2 );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetEfficiency( TString theString, TTree *theTree, Double_t& effSerr )
{
   // fill background efficiency (resp. rejection) versus signal efficiency plots
   // returns signal efficiency at background efficiency indicated in theString

   if (theTree == 0) fLogger << kFATAL << "<GetEfficiency> Object theTree has zero pointer" << Endl;

   // parse input string for required background efficiency
   TList* list  = Tools::ParseFormatLine( theString );

   // sanity check
   Bool_t computeArea = kFALSE;
   if      (!list || list->GetSize() < 2) computeArea = kTRUE; // the area is computed
   else if (list->GetSize() > 2) {
      fLogger << kFATAL << "<GetEfficiency> Wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05, or empty string" << Endl;
      return -1;
   }

   // sanity check
   if (fHistS_highbin->GetNbinsX() != fHistB_highbin->GetNbinsX() ||
       fHistS_plotbin->GetNbinsX() != fHistB_plotbin->GetNbinsX()) {
      fLogger << kWARNING << "<GetEfficiency> Binning mismatch between signal and background histos" << Endl;
      fIsOK = kFALSE;
      return -1.0;
   }

   // create histograms

   // first, get efficiency histograms for signal and background
   Double_t xmin = fHistS_highbin->GetXaxis()->GetXmin();
   Double_t xmax = fHistS_highbin->GetXaxis()->GetXmax();

   // first round ? --> create histograms
   Bool_t firstPass = kFALSE;

   static Double_t nevtS;

   if (NULL == fEffS && NULL == fEffB) firstPass = kTRUE;

   if (firstPass) {

      fEffS = new TH1F( GetTestvarName() + "_effS", GetTestvarName() + " (signal)",     fNbinsH, xmin, xmax );
      fEffB = new TH1F( GetTestvarName() + "_effB", GetTestvarName() + " (background)", fNbinsH, xmin, xmax );

      // sign if cut
      Int_t sign = (fCutOrientation == kPositive) ? +1 : -1;

      // this method is unbinned
      Int_t   theType;
      Float_t theVal;
      theTree->ResetBranchAddresses();
      TBranch* brType = theTree->GetBranch("type");
      TBranch* brVal  = theTree->GetBranch(GetTestvarName());
      if (brVal == 0) {
         fLogger << kFATAL << "Could not find variable "
                 << GetTestvarName() << " in tree " << theTree->GetName() << Endl;
      }
      brType->SetAddress(&theType);
      brVal ->SetAddress(&theVal );

      nevtS = 0;
      for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

         // read the tree
         brType->GetEntry(ievt);
         brVal ->GetEntry(ievt);

         // select histogram depending on if sig or bgd
         TH1* theHist = (theType == 1) ? fEffS : fEffB;

         // count signal and background events in tree
         if (theType == 1) nevtS++;

         TAxis* axis   = theHist->GetXaxis();
         Int_t  maxbin = Int_t((theVal - axis->GetXmin())/(axis->GetXmax() - axis->GetXmin())*fNbinsH) + 1;
         if (sign > 0 && maxbin > fNbinsH) continue; // can happen... event doesn't count
         if (sign < 0 && maxbin < 1      ) continue; // can happen... event doesn't count
         if (sign > 0 && maxbin < 1      ) maxbin = 1;
         if (sign < 0 && maxbin > fNbinsH) maxbin = fNbinsH;

         if (sign > 0)
            for (Int_t ibin=1; ibin<=maxbin; ibin++) theHist->AddBinContent( ibin );
         else if (sign < 0)
            for (Int_t ibin=maxbin+1; ibin<=fNbinsH; ibin++) theHist->AddBinContent( ibin );
         else
            fLogger << kFATAL << "<GetEfficiency> Mismatch in sign" << Endl;
      }
      theTree->ResetBranchAddresses();

      // renormalise to maximum
      fEffS->Scale( 1.0/(fEffS->GetMaximum() > 0 ? fEffS->GetMaximum() : 1) );
      fEffB->Scale( 1.0/(fEffB->GetMaximum() > 0 ? fEffB->GetMaximum() : 1) );

      // now create efficiency curve: background versus signal
      fEffBvsS = new TH1F( GetTestvarName() + "_effBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fEffBvsS->SetXTitle( "signal eff" );
      fEffBvsS->SetYTitle( "backgr eff" );
      fRejBvsS = new TH1F( GetTestvarName() + "_rejBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fRejBvsS->SetXTitle( "signal eff" );
      fRejBvsS->SetYTitle( "backgr rejection (1-eff)" );
      finvBeffvsSeff = new TH1F( GetTestvarName() + "_invBeffvsSeff",
                                 GetTestvarName() + "", fNbins, 0, 1 );
      finvBeffvsSeff->SetXTitle( "signal eff" );
      finvBeffvsSeff->SetYTitle( "inverse backgr. eff (1/eff)" );

      // use root finder
      // spline background efficiency plot
      // note that there is a bin shift when going from a TH1F object to a TGraph :-(
      if (Use_Splines_for_Eff_) {
         fGraphS   = new TGraph( fEffS );
         fGraphB   = new TGraph( fEffB );
         fSplRefS  = new TSpline1( "spline2_signal",     fGraphS );
         fSplRefB  = new TSpline1( "spline2_background", fGraphB );

         // verify spline sanity
         Tools::CheckSplines( fEffS, fSplRefS );
         Tools::CheckSplines( fEffB, fSplRefB );
      }

      // make the background-vs-signal efficiency plot

      // create root finder
      // reset static "this" pointer before calling external function
      ResetThisBase();
      RootFinder rootFinder( &IGetEffForRoot, fXmin, fXmax );

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
         if (effB>std::numeric_limits<double>::epsilon())
            finvBeffvsSeff->SetBinContent( bini, 1.0/effB );
      }

      // create splines for histogram
      fGrapheffBvsS = new TGraph( fEffBvsS );
      fSpleffBvsS   = new TSpline1( "effBvsS", fGrapheffBvsS );

      // search for overlap point where, when cutting on it,
      // one would obtain: eff_S = rej_B = 1 - eff_B
      Double_t effS, rejB, effS_ = 0, rejB_ = 0;
      Int_t    nbins_ = 5000;
      for (Int_t bini=1; bini<=nbins_; bini++) {

         // get corresponding signal and background efficiencies
         effS = (bini - 0.5)/Float_t(nbins_);
         rejB = 1.0 - fSpleffBvsS->Eval( effS );

         // find signal efficiency that corresponds to required background efficiency
         if ((effS - rejB)*(effS_ - rejB_) < 0) break;
         effS_ = effS;
         rejB_ = rejB;
      }

      // find cut that corresponds to signal efficiency and update signal-like criterion
      Double_t cut = rootFinder.Root( 0.5*(effS + effS_) );
      SetSignalReferenceCut( cut );
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
      Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

      // find precise efficiency value
      for (Int_t bini=1; bini<=nbins_; bini++) {

         // get corresponding signal and background efficiencies
         effS = (bini - 0.5)/Float_t(nbins_);
         effB = fSpleffBvsS->Eval( effS );

         // find signal efficiency that corresponds to required background efficiency
         if ((effB - effBref)*(effB_ - effBref) < 0) break;
         effS_ = effS;
         effB_ = effB;
      }

      // take mean between bin above and bin below
      effS = 0.5*(effS + effS_);

      effSerr = 0;
      if (nevtS > 0) effSerr = TMath::Sqrt( effS*(1.0 - effS)/nevtS );

      return effS;
   }

   return -1;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetTrainingEfficiency( TString theString)
{
   // fill background efficiency (resp. rejection) versus signal efficiency plots
   // returns signal efficiency at background efficiency indicated in theString

   // parse input string for required background efficiency
   TList*  list  = Tools::ParseFormatLine( theString );
   // sanity check

   if (list->GetSize() != 2) {
      fLogger << kFATAL << "<GetTrainingEfficiency> Wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05" << Endl;
      return -1;
   }
   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   // sanity check
   if (fHistS_highbin->GetNbinsX() != fHistB_highbin->GetNbinsX() ||
       fHistS_plotbin->GetNbinsX() != fHistB_plotbin->GetNbinsX()) {
      fLogger << kFATAL << "<GetTrainingEfficiency> Binning mismatch between signal and background histos"
              << Endl;
      fIsOK = kFALSE;
      return -1.0;
   }

   // create histogram

   // first, get efficiency histograms for signal and background
   Double_t xmin = fHistS_highbin->GetXaxis()->GetXmin();
   Double_t xmax = fHistS_highbin->GetXaxis()->GetXmax();

   // first round ? --> create histograms
   Bool_t firstPass = kFALSE;
   if (NULL == fTrainEffS && NULL == fTrainEffB) firstPass = kTRUE;

   if (firstPass) {

      fTrainEffS = new TH1F( GetTestvarName() + "_trainingEffS", GetTestvarName() + " (signal)",
                             fNbinsH, xmin, xmax );
      fTrainEffB = new TH1F( GetTestvarName() + "_trainingEffB", GetTestvarName() + " (background)",
                             fNbinsH, xmin, xmax );

      // sign if cut
      Int_t sign = (fCutOrientation == kPositive) ? +1 : -1;

      // this method is unbinned
      for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {
         ReadTrainingEvent(ievt);

         TH1* theHist = (IsSignalEvent() ? fTrainEffS : fTrainEffB);

         Double_t theVal = this->GetMvaValue();

         TAxis* axis   = theHist->GetXaxis();
         Int_t  maxbin = Int_t((theVal - axis->GetXmin())/(axis->GetXmax() - axis->GetXmin())*fNbinsH) + 1;
         if (sign > 0 && maxbin > fNbinsH) continue; // can happen... event doesn't count
         if (sign < 0 && maxbin < 1      ) continue; // can happen... event doesn't count
         if (sign > 0 && maxbin < 1      ) maxbin = 1;
         if (sign < 0 && maxbin > fNbinsH) maxbin = fNbinsH;

         if (sign > 0)
            for (Int_t ibin=1; ibin<=maxbin; ibin++) theHist->AddBinContent( ibin );
         else if (sign < 0)
            for (Int_t ibin=maxbin+1; ibin<=fNbinsH; ibin++) theHist->AddBinContent( ibin );
         else
            fLogger << kFATAL << "<GetEfficiency> Mismatch in sign" << Endl;

      }

      // renormalise to maximum
      fTrainEffS->Scale( 1.0/(fTrainEffS->GetMaximum() > 0 ? fTrainEffS->GetMaximum() : 1) );
      fTrainEffB->Scale( 1.0/(fTrainEffB->GetMaximum() > 0 ? fTrainEffB->GetMaximum() : 1) );

      // now create efficiency curve: background versus signal
      fTrainEffBvsS = new TH1F( GetTestvarName() + "_trainingEffBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fTrainRejBvsS = new TH1F( GetTestvarName() + "_trainingRejBvsS", GetTestvarName() + "", fNbins, 0, 1 );

      // use root finder
      // spline background efficiency plot
      // note that there is a bin shift when going from a TH1F object to a TGraph :-(
      if (Use_Splines_for_Eff_) {
         fGraphTrainS   = new TGraph( fTrainEffS );
         fGraphTrainB   = new TGraph( fTrainEffB );
         fSplTrainRefS  = new TSpline1( "spline2_signal",     fGraphTrainS );
         fSplTrainRefB  = new TSpline1( "spline2_background", fGraphTrainB );

         // verify spline sanity
         Tools::CheckSplines( fTrainEffS, fSplTrainRefS );
         Tools::CheckSplines( fTrainEffB, fSplTrainRefB );
      }

      // make the background-vs-signal efficiency plot

      // create root finder
      // reset static "this" pointer before calling external function
      ResetThisBase();
      RootFinder rootFinder(&IGetEffForRoot, fXmin, fXmax );

      Double_t effB = 0;
      for (Int_t bini=1; bini<=fNbins; bini++) {

         // find cut value corresponding to a given signal efficiency
         Double_t effS = fTrainEffBvsS->GetBinCenter( bini );

         Double_t cut  = rootFinder.Root( effS );

         // retrieve background efficiency for given cut
         if (Use_Splines_for_Eff_)
            effB = fSplTrainRefB->Eval( cut );
         else
            effB = fTrainEffB->GetBinContent( fTrainEffB->FindBin( cut ) );

         // and fill histograms
         fTrainEffBvsS->SetBinContent( bini, effB     );
         fTrainRejBvsS->SetBinContent( bini, 1.0-effB );
      }

      // create splines for histogram
      fGraphTrainEffBvsS = new TGraph( fTrainEffBvsS );
      fSplTrainEffBvsS   = new TSpline1( "effBvsS", fGraphTrainEffBvsS );
   }

   // must exist...
   if (NULL == fSplTrainEffBvsS) return 0.0;

   // now find signal efficiency that corresponds to required background efficiency
   Double_t effS, effB, effS_ = 0, effB_ = 0;
   Int_t    nbins_ = 1000;
   for (Int_t bini=1; bini<=nbins_; bini++) {

      // get corresponding signal and background efficiencies
      effS = (bini - 0.5)/Float_t(nbins_);
      effB = fSplTrainEffBvsS->Eval( effS );

      // find signal efficiency that corresponds to required background efficiency
      if ((effB - effBref)*(effB_ - effBref) < 0) break;
      effS_ = effS;
      effB_ = effB;
   }

   return 0.5*(effS + effS_); // the mean between bin above and bin below
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetSignificance( void ) const
{
   // compute significance of mean difference
   // significance = |<S> - <B>|/Sqrt(RMS_S2 + RMS_B2)
   Double_t rms = sqrt( fRmsS*fRmsS + fRmsB*fRmsB );

   return (rms > 0) ? TMath::Abs(fMeanS - fMeanB)/rms : 0;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetSeparation( TH1* histoS, TH1* histoB ) const
{
   // compute "separation" defined as
   // <s2> = (1/2) Int_-oo..+oo { (S(x)2 - B(x)2)/(S(x) + B(x)) dx }

   Double_t xmin = histoS->GetXaxis()->GetXmin();
   Double_t xmax = histoB->GetXaxis()->GetXmax();
   // sanity check
   if (xmin != histoB->GetXaxis()->GetXmin() || xmax != histoB->GetXaxis()->GetXmax()) {
      fLogger << kFATAL << "<GetSeparation> Mismatch in histogram limits: "
              << xmin << " " << histoB->GetXaxis()->GetXmin()
              << xmax << " " << histoB->GetXaxis()->GetXmax()  << Endl;
   }

   Double_t separation = 0;
   Int_t nstep  = histoS->GetNbinsX();
   Double_t intBin = (xmax - xmin)/nstep;
   for (Int_t bin=0; bin<nstep; bin++) {
      Double_t s = histoS->GetBinContent(bin);
      Double_t b = histoB->GetBinContent(bin);
      if (s + b > 0) separation += 0.5*(s - b)*(s - b)/(s + b);
   }
   separation *= intBin;
   return separation;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetSeparation( PDF* pdfS, PDF* pdfB ) const
{
   // compute "separation" defined as
   // <s2> = (1/2) Int_-oo..+oo { (S(x)2 - B(x)2)/(S(x) + B(x)) dx }

   // note, if zero pointers given, use internal pdf
   // sanity check first
   if (!pdfS && pdfB || pdfS && !pdfB)
      fLogger << kFATAL << "<GetSeparation> Mismatch in pdfs" << Endl;
   if (!pdfS) pdfS = fSplS;
   if (!pdfB) pdfB = fSplB;

   Double_t xmin = pdfS->GetXmin();
   Double_t xmax = pdfS->GetXmax();
   // sanity check
   if (xmin != pdfB->GetXmin() || xmax != pdfB->GetXmax()) {
      fLogger << kFATAL << "<GetSeparation> Mismatch in PDF limits: "
              << xmin << " " << pdfB->GetXmin() << xmax << " " << pdfB->GetXmax()  << Endl;
   }

   Double_t separation = 0;
   Int_t nstep  = 100;
   Double_t intBin = (xmax - xmin)/nstep;
   for (Int_t bin=0; bin<nstep; bin++) {
      Double_t x = (bin + 0.5)*intBin + xmin;
      Double_t s = pdfS->GetVal( x );
      Double_t b = pdfB->GetVal( x );
      // separation
      if (s + b > 0) separation += 0.5*(s - b)*(s - b)/(s + b);
   }
   separation *= intBin;

   return separation;
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetOptimalSignificance(Double_t SignalEvents,
                                                  Double_t BackgroundEvents,
                                                  Double_t& optimal_significance_value  ) const
{
   // plot significance, S/Sqrt(S^2 + B^2), curve for given number
   // of signal and background events; returns cut for optimal significance
   // also returned via reference is the optimal significance

   Double_t optimal_significance(0);
   Double_t effS(0),effB(0),significance(0);
   TH1F *temp_histogram = new TH1F("temp", "temp", fNbinsH, fXmin, fXmax );

   if (SignalEvents <= 0 || BackgroundEvents <= 0) {
      fLogger << kFATAL << "<GetOptimalSignificance> "
              << "Number of signal or background events is <= 0 ==> abort"
              << Endl;
   }

   fLogger << kINFO << "Using ratio SignalEvents/BackgroundEvents = "
           << SignalEvents/BackgroundEvents << Endl;

   if ((fEffS == 0) || (fEffB == 0)) {
      fLogger << kWARNING << "Efficiency histograms empty !" << Endl;
      fLogger << kWARNING << "no optimal cut found, return 0" << Endl;
      return 0;
   }

   for (Int_t bin=1; bin<=fNbinsH; bin++) {
      effS = fEffS->GetBinContent( bin );
      effB = fEffB->GetBinContent( bin );

      // put significance into a histogram
      significance = sqrt(SignalEvents)*( effS )/sqrt( effS + ( BackgroundEvents / SignalEvents) * effB  );

      temp_histogram->SetBinContent(bin,significance);
   }

   // find maximum in histogram
   optimal_significance = temp_histogram->GetBinCenter( temp_histogram->GetMaximumBin() );
   optimal_significance_value = temp_histogram->GetBinContent( temp_histogram->GetMaximumBin() );

   // delete
   temp_histogram->Delete();

   fLogger << kINFO << "Optimal cut at      : " << optimal_significance << Endl;
   fLogger << kINFO << "Optimal significance: " << optimal_significance_value << Endl;

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
   fHistBhatS = new TH1F( GetTestvarName() + "_BhatS", GetTestvarName() + ": Bhat (S)", nbin, 0.0, 1.0 );
   fHistBhatB = new TH1F( GetTestvarName() + "_BhatB", GetTestvarName() + ": Bhat (B)", nbin, 0.0, 1.0 );

   fHistBhatS->Sumw2();
   fHistBhatB->Sumw2();

   vector<Double_t>* aBhatB = new vector<Double_t>;
   vector<Double_t>* aBhatS = new vector<Double_t>;

   Float_t x;
   TBranch* br = theTree->GetBranch(GetTestvarName());
   for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {
      ReadEvent(theTree,ievt);
      br->SetAddress(&x);
      br->GetEvent(ievt);
      Double_t s = fSplS->GetVal( x );
      Double_t b = fSplB->GetVal( x );
      Double_t aBhat = 0;
      if (b + s > 0) aBhat = b/(b + s);

      if (IsSignalEvent()) { // this is signal
         aBhatS->push_back ( aBhat );
         fHistBhatS->Fill( aBhat );
      }
      else {
         aBhatB->push_back ( aBhat );
         fHistBhatB->Fill( aBhat );
      }
   }

   // normalise histograms
   fHistBhatS->Scale( 1.0/((fHistBhatS->GetEntries() > 0 ? fHistBhatS->GetEntries() : 1) / nbin) );
   fHistBhatB->Scale( 1.0/((fHistBhatB->GetEntries() > 0 ? fHistBhatB->GetEntries() : 1) / nbin) );

   PDF* yB = new PDF( fHistBhatB, PDF::kSpline2, 100 );

   Int_t nevtS = aBhatS->size();
   Int_t nevtB = aBhatB->size();

   // get the mu-transform
   Int_t nbinMu = 50;
   fHistMuS = new TH1F( GetTestvarName() + "_muTransform_S",
                        GetTestvarName() + ": mu-Transform (S)", nbinMu, 0.0, 1.0 );
   fHistMuB = new TH1F( GetTestvarName() + "_muTransform_B",
                        GetTestvarName() + ": mu-Transform (B)", nbinMu, 0.0, 1.0 );

   // signal
   for (Int_t ievt=0; ievt<nevtS; ievt++) {
      Double_t w = yB->GetVal( (*aBhatS)[ievt] );
      if (w > 0) fHistMuS->Fill( 1.0 - (*aBhatS)[ievt], 1.0/w );
   }

   // background (must be flat)
   for (Int_t ievt=0; ievt<nevtB; ievt++) {
      Double_t w = yB->GetVal( (*aBhatB)[ievt] );
      if (w > 0) fHistMuB->Fill( 1.0 - (*aBhatB)[ievt], 1.0/w );
   }

   // normalise mu-transforms
   Tools::NormHist( fHistMuS );
   Tools::NormHist( fHistMuB );

   // determine the mu-transform value, which is defined as
   // the average of the signal mu-transform Int_[0,1] { S(mu) dmu }
   // this average is 0.5 for background, by definition
   PDF* thePdf = new PDF( fHistMuS, PDF::kSpline2 );
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
void TMVA::MethodBase::Statistics( Types::ETreeType treeType, const TString& theVarName,
                                   Double_t& meanS, Double_t& meanB,
                                   Double_t& rmsS,  Double_t& rmsB,
                                   Double_t& xmin,  Double_t& xmax,
                                   Bool_t    norm )
{
   // calculates rms,mean, xmin, xmax of the event variable
   // this can be either done for the variables as they are or for
   // normalised variables (in the range of 0-1) if "norm" is set to kTRUE

   Long64_t entries = ( (treeType == Types::kTesting ) ? Data().GetNEvtTest() :
                        (treeType == Types::kTraining) ? Data().GetNEvtTrain() : -1 );

   // sanity check
   if (entries <=0)
      fLogger << kFATAL << "<CalculateEstimator> Wrong tree type: " << treeType << Endl;

   // index of the wanted variable
   UInt_t varIndex = Data().FindVar( theVarName );

   // first fill signal and background in arrays before analysis
   Double_t* varVecS  = new Double_t[entries];
   Double_t* varVecB  = new Double_t[entries];
   xmin               = +1e20;
   xmax               = -1e20;
   Long64_t nEventsS  = -1;
   Long64_t nEventsB  = -1;

   // take into account event weights
   meanS = 0;
   meanB = 0;
   rmsS  = 0;
   rmsB  = 0;
   Double_t sumwS = 0, sumwB = 0;

   // loop over all training events
   for (Int_t i = 0; i < entries; i++) {

      if (treeType == Types::kTesting ) ReadTestEvent(i);
      else                              ReadTrainingEvent(i);

      Double_t theVar = (norm) ? GetEventValNormalised(varIndex) : GetEventVal(varIndex);


      Double_t weight = GetEventWeight();

      if (IsSignalEvent()) {
         sumwS += weight;
         meanS += weight*theVar;
         rmsS  += weight*theVar*theVar;
      }
      else {
         sumwB += weight;
         meanB += weight*theVar;
         rmsB  += weight*theVar*theVar;
      }

      if (IsSignalEvent()) varVecS[++nEventsS] = theVar;
      else                 varVecB[++nEventsB] = theVar;

      xmin = TMath::Min( xmin, theVar );
      xmax = TMath::Max( xmax, theVar );
   }
   ++nEventsS;
   ++nEventsB;

   meanS = meanS/sumwS;
   meanB = meanB/sumwB;
   rmsS  = TMath::Sqrt( rmsS/sumwS - meanS*meanS );
   rmsB  = TMath::Sqrt( rmsB/sumwB - meanB*meanB );

   delete [] varVecS;
   delete [] varVecB;
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteEvaluationHistosToFile( TDirectory* targetDir )
{
   // writes all MVA evaluation histograms to file
   if (targetDir!=0) targetDir->cd();
   else              BaseDir()->cd();
   if (0 != fHistS_plotbin  ) fHistS_plotbin->Write();
   if (0 != fHistB_plotbin  ) fHistB_plotbin->Write();
   if (0 != fProbaS_plotbin ) fProbaS_plotbin->Write();
   if (0 != fProbaB_plotbin ) fProbaB_plotbin->Write();
   if (0 != fRarityS_plotbin) fRarityS_plotbin->Write();
   if (0 != fRarityB_plotbin) fRarityB_plotbin->Write();
   if (0 != fHistS_highbin  ) fHistS_highbin->Write();
   if (0 != fHistB_highbin  ) fHistB_highbin->Write();
   if (0 != fEffS           ) fEffS->Write();
   if (0 != fEffB           ) fEffB->Write();
   if (0 != fEffBvsS        ) fEffBvsS->Write();
   if (0 != fRejBvsS        ) fRejBvsS->Write();
   if (0 != finvBeffvsSeff  ) finvBeffvsSeff->Write();
   if (0 != fHistBhatS      ) fHistBhatS->Write();
   if (0 != fHistBhatB      ) fHistBhatB->Write();
   if (0 != fHistMuS        ) fHistMuS->Write();
   if (0 != fHistMuB        ) fHistMuB->Write();

   if (0 != fMVAPdfS) {
      fMVAPdfS->GetOriginalHist()->Write();
      fMVAPdfS->GetSmoothedHist()->Write();
      fMVAPdfS->GetPDFHist()->Write();
   }
   if (0 != fMVAPdfB) {
      fMVAPdfB->GetOriginalHist()->Write();
      fMVAPdfB->GetSmoothedHist()->Write();
      fMVAPdfB->GetPDFHist()->Write();
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::MakeClass( const TString& theClassFileName ) const
{
   // create reader class for classifier

   // the default consists of
   TString classFileName = "";
   if (theClassFileName == "")
      classFileName = fFileDir + "/" + fJobName + "_" + GetMethodTitle() + ".class.C";
   else
      classFileName = theClassFileName;

   TString className = TString("Read") + GetMethodTitle();

   TString tfname( classFileName );
   fLogger << kINFO << "Creating standalone response class : "
           << Tools::Color("lightblue") << classFileName << Tools::Color("reset") << Endl;

   ofstream fout( classFileName );
   if (!fout.good()) { // file could not be opened --> Error
      fLogger << kFATAL << "<MakeClass> Unable to open file: " << classFileName << Endl;
   }

   // now create the class
   // preamble
   fout << "// Class: " << className << endl;
   fout << "// Automatically generated by MethodBase::MakeClass" << endl << "//" << endl;

   // print general information and configuration state
   fout << endl;
   fout << "/* configuration options =====================================================" << endl << endl;
   WriteStateToStream( fout, kTRUE );
   fout << endl;
   fout << "============================================================================ */" << endl;

   // generate the class
   fout << "" << endl;
   fout << "#include <vector>" << endl;
   fout << "#include <cmath>" << endl;
   fout << "#include <string>" << endl;
   fout << "#include <iostream>" << endl;
   fout << "" << endl;
   // now if the classifier needs to write some addicional classes for its response implementation
   // this code goes here: (at least the header declarations need to come before the main class
   this->MakeClassSpecificHeader( fout, className );

   fout << "#ifndef IClassifierReader__def" << endl;
   fout << "#define IClassifierReader__def" << endl;
   fout << endl;
   fout << "class IClassifierReader {" << endl;
   fout << endl;
   fout << " public:" << endl;
   fout << endl;
   fout << "   // constructor" << endl;
   fout << "   IClassifierReader() {}" << endl;
   fout << "   virtual ~IClassifierReader() {}" << endl;
   fout << endl;
   fout << "   // return classifier response" << endl;
   fout << "   virtual double GetMvaValue( const std::vector<double>& inputValues ) const = 0;" << endl;
   fout << "};" << endl;
   fout << endl;
   fout << "#endif" << endl;
   fout << endl;
   fout << "class " << className << " : public IClassifierReader {" << endl;
   fout << endl;
   fout << " public:" << endl;
   fout << endl;
   fout << "   // constructor" << endl;
   fout << "   " << className << "( std::vector<std::string>& theInputVars ) " << endl;
   fout << "      : IClassifierReader()," << endl;
   fout << "        fClassName( \"" << className << "\" )," << endl;
   fout << "        fStatusIsClean( true )," << endl;
   fout << "        fNvars( " << GetNvar() << " )," << endl;
   fout << "        fIsNormalised( " << (IsNormalised() ? "true" : "false") << " )" << endl;
   fout << "   {      " << endl;
   fout << "      // the training input variables" << endl;
   fout << "      const char* inputVars[] = { ";
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "\"" << GetOriginalVarName(ivar) << "\"";
      if (ivar<GetNvar()-1) fout << ", ";
   }
   fout << " };" << endl;
   fout << endl;
   fout << "      // sanity checks" << endl;
   fout << "      if (theInputVars.size() <= 0) {" << endl;
   fout << "         std::cout << \"Problem in class \\\"\" << fClassName << \"\\\": empty input vector\" << std::endl;" << endl;
   fout << "         fStatusIsClean = false;" << endl;
   fout << "      }" << endl;
   fout << endl;
   fout << "      if (theInputVars.size() != fNvars) {" << endl;
   fout << "         std::cout << \"Problem in class \\\"\" << fClassName << \"\\\": mismatch in number of input values: \"" << endl;
   fout << "                   << theInputVars.size() << \" != \" << fNvars << std::endl;" << endl;
   fout << "         fStatusIsClean = false;" << endl;
   fout << "      }" << endl;
   fout << endl;
   fout << "      // validate input variables" << endl;
   fout << "      for (size_t ivar = 0; ivar < theInputVars.size(); ivar++) {" << endl;
   fout << "         if (theInputVars[ivar] != inputVars[ivar]) {" << endl;
   fout << "            std::cout << \"Problem in class \\\"\" << fClassName << \"\\\": mismatch in input variable names\" << std::endl" << endl;
   fout << "                      << \" for variable [\" << ivar << \"]: \" << theInputVars[ivar].c_str() << \" != \" << inputVars[ivar] << std::endl;" << endl;
   fout << "            fStatusIsClean = false;" << endl;
   fout << "         }" << endl;
   fout << "      }" << endl;
   fout << endl;
   fout << "      // initialize min and max vectors (for normalisation)" << endl;
   for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
      fout << "      fVmin[" << ivar << "] = " << setprecision(15) << GetXmin( ivar ) << ";" << endl;
      fout << "      fVmax[" << ivar << "] = " << setprecision(15) << GetXmax( ivar ) << ";" << endl;
   }
   fout << endl;
   fout << "      // initialize input variable types" << endl;
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fout << "      fType[" << ivar << "] = \'" << Data().GetVarType(ivar) << "\';" << endl;
   }
   fout << endl;
   fout << "      // initialize constants" << endl;
   fout << "      Initialize();" << endl;
   fout << endl;
   if (fVariableTransform != Types::kNone) {
      fout << "      // initialize transformation" << endl;
      fout << "      InitTransform();" << endl;
   }
   fout << "   }" << endl;
   fout << endl;
   fout << "   // destructor" << endl;
   fout << "   virtual ~" << className << "() {" << endl;
   fout << "      Clear(); // method-specific" << endl;
   fout << "   }" << endl;
   fout << endl;
   fout << "   // the classifier response" << endl;
   fout << "   // \"inputValues\" is a vector of input values in the same order as the " << endl;
   fout << "   // variables given to the constructor" << endl;
   fout << "   double GetMvaValue( const std::vector<double>& inputValues ) const" << endl;
   fout << "   {" << endl;
   fout << "      // classifier response value" << endl;
   fout << "      double retval = 0;" << endl;
   fout << endl;
   fout << "      // classifier response, sanity check first" << endl;
   fout << "      if (!fStatusIsClean) {" << endl;
   fout << "         std::cout << \"Problem in class \\\"\" << fClassName << \"\\\": cannot return classifier response\"" << endl;
   fout << "                   << \" because status is dirty\" << std::endl;" << endl;
   fout << "         retval = 0;" << endl;
   fout << "      }" << endl;
   fout << "      else {" << endl;
   fout << "         if (IsNormalised()) {" << endl;
   fout << "            // normalise variables" << endl;
   fout << "            std::vector<double> iV;" << endl;
   fout << "            int ivar = 0;" << endl;
   fout << "            for (std::vector<double>::const_iterator varIt = inputValues.begin();" << endl;
   fout << "                 varIt != inputValues.end(); varIt++, ivar++) {" << endl;
   fout << "               iV.push_back(NormVariable( *varIt, fVmin[ivar], fVmax[ivar] ));" << endl;
   fout << "            }" << endl;
   if (fVariableTransform != Types::kNone && GetMethodType() != Types::kLikelihood )
      fout << "            Transform( iV, 0 );" << endl;
   fout << "            retval = GetMvaValue__( iV );" << endl;
   fout << "         }" << endl;
   fout << "         else {" << endl;
   if (fVariableTransform != Types::kNone && GetMethodType() != Types::kLikelihood )
      fout << "            Transform( inputValues, 0 );" << endl;
   fout << "            retval = GetMvaValue__( inputValues );" << endl;
   fout << "         }" << endl;
   fout << "      }" << endl;
   fout << endl;
   fout << "      return retval;" << endl;
   fout << "   }" << endl;
   fout << endl;
   fout << " private:" << endl;
   fout << endl;
   fout << "   // method-specific destructor" << endl;
   fout << "   void Clear();" << endl;
   fout << endl;
   if (fVariableTransform != Types::kNone) {
      fout << "   // input variable transformation" << endl;
      GetVarTransform().MakeFunction(fout, className,1);
      fout << "   void InitTransform();" << endl;
      fout << "   void Transform( std::vector<double> & iv, int sigOrBgd ) const;" << endl;
      fout << endl;
   }
   fout << "   // common member variables" << endl;
   fout << "   const char* fClassName;" << endl;
   fout << "   bool fStatusIsClean;" << endl;
   fout << endl;
   fout << "   const size_t fNvars;" << endl;
   fout << "   size_t GetNvar()           const { return fNvars; }" << endl;
   fout << "   char   GetType( int ivar ) const { return fType[ivar]; }" << endl;
   fout << endl;
   fout << "   // normalisation of input variables" << endl;
   fout << "   const bool fIsNormalised;" << endl;
   fout << "   bool IsNormalised() const { return fIsNormalised; }" << endl;
   fout << "   double fVmin[" << GetNvar() << "];" << endl;
   fout << "   double fVmax[" << GetNvar() << "];" << endl;
   fout << "   double NormVariable( double x, double xmin, double xmax ) const {" << endl;
   fout << "      // normalise to output range: [-1, 1]" << endl;
   fout << "      return 2*(x - xmin)/(xmax - xmin) - 1.0;" << endl;
   fout << "   }" << endl;
   fout << endl;
   fout << "   // type of input variable: 'F' or 'I'" << endl;
   fout << "   char   fType[" << GetNvar() << "];" << endl;
   fout << endl;
   fout << "   // initialize internal variables" << endl;
   fout << "   void Initialize();" << endl;
   fout << "   double GetMvaValue__( const std::vector<double>& inputValues ) const;" << endl;
   fout << "" << endl;
   fout << "   // private members (method specific)" << endl;

   // call the classifier specific output (the classifier must close the class !)
   MakeClassSpecific( fout, className );

   // create output for transformation - if any
   if (fVariableTransform != Types::kNone) GetVarTransform().MakeFunction(fout, className,2);

   // close the file
   fout.close();
}

// ----------------------- r o o t   f i n d i n g ----------------------------

TMVA::MethodBase* TMVA::MethodBase::fgThisBase = NULL;

//_______________________________________________________________________
Double_t TMVA::MethodBase::IGetEffForRoot( Double_t theCut )
{
   // interface for RootFinder
   return MethodBase::GetThisBase()->GetEffForRoot( theCut );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetEffForRoot( Double_t theCut )
{
   // returns efficiency as function of cut
   Double_t retval=0;

   // retrieve the class object
   if (Use_Splines_for_Eff_) {
      retval = fSplRefS->Eval( theCut );
   }
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

//_______________________________________________________________________
void  TMVA::MethodBase::WriteMonitoringHistosToFile( void ) const
{
   // write special monitoring histograms to file - not implemented for this method
   fLogger << kINFO << "No monitoring histograms written" << Endl;
}

//_______________________________________________________________________
TString TMVA::MethodBase::GetTrainingTMVAVersionString() const
{
   // calculates the TMVA version string from the training version code on the fly
   UInt_t a = GetTrainingTMVAVersionCode() & 0xff0000; a>>=16;
   UInt_t b = GetTrainingTMVAVersionCode() & 0x00ff00; b>>=8;
   UInt_t c = GetTrainingTMVAVersionCode() & 0x0000ff;

   return TString(Form("%i.%i.%i",a,b,c));
}

//_______________________________________________________________________
TString TMVA::MethodBase::GetTrainingROOTVersionString() const
{
   // calculates the ROOT version string from the training version code on the fly
   UInt_t a = GetTrainingROOTVersionCode() & 0xff0000; a>>=16;
   UInt_t b = GetTrainingROOTVersionCode() & 0x00ff00; b>>=8;
   UInt_t c = GetTrainingROOTVersionCode() & 0x0000ff;

   return TString(Form("%i.%02i/%02i",a,b,c));
}

