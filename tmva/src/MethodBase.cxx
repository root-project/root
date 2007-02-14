// @(#)root/tmva $Id: MethodBase.cxx,v 1.13 2007/01/23 11:26:36 brun Exp $
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
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
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
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: MethodBase.cxx,v 1.13 2007/01/23 11:26:36 brun Exp $
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
#include <fstream>
#include <stdlib.h>
#include "TSystem.h"
#include "TObjString.h"
#include "TQObject.h"
#include "TSpline.h"
#include "TMatrix.h"
#include "TH1.h"
#include "TMath.h"
#include "TDirectory.h"
#include <stdexcept>

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_RootFinder
#include "TMVA/RootFinder.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif

ClassImp(TMVA::MethodBase)
   ;

const Bool_t   DEBUG_TMVA_MethodBase     = kFALSE;
const Int_t    MethodBase_MaxIterations_ = 200;
const Bool_t   Use_Splines_for_Eff_      = kTRUE;

const int      NBIN_HIST_PLOT = 100;
const int      NBIN_HIST_HIGH = 10000;

const TString  BC_blue   = "\033[34m" ;
const TString  EC__      = "\033[0m"    ;
const TString  BC_yellow = "\033[1;33m";
const TString  BC_lgreen = "\033[1;32m";


//_______________________________________________________________________
TMVA::MethodBase::MethodBase( TString      jobName,
                              TString      methodTitle,
                              DataSet&     theData,
                              TString      theOption,
                              TDirectory*  theBaseDir)  
   : IMethod(),
     fData                      ( theData ),     
     fJobName                   ( jobName ),
     fMethodTitle               ( methodTitle ),
     fOptions                   ( theOption  ),
     fBaseDir                   ( theBaseDir ),
     fWeightFile                ( "" ),
     fWeightFileType            (kTEXT),
     fTrainEffS                 (0),
     fTrainEffB                 (0),
     fTrainEffBvsS              (0),
     fTrainRejBvsS              (0),
     fGraphTrainS               (0),        
     fGraphTrainB               (0),     
     fGraphTrainEffBvsS         (0),
     fSplTrainS                 (0),       
     fSplTrainB                 (0),       
     fSplTrainEffBvsS           (0),
     fUseDecorr                 (kFALSE),
     fPreprocessingMethod       (Types::kNone),
     fVerbose                   (kFALSE),
     fHelp                      (kFALSE),
     fLooseOptionCheckingEnabled(kTRUE),
     fSplRefS                   (0),
     fSplRefB                   (0),
     fLogger                    (this)
{  
   // standard constructur
   this->Init();

   DeclareOptions();

   // local copy of the variable ranges 
   // used for normalization
   // if Method constructed from Dataset
   // the ranges will be taken from there
   for (Int_t corr=0; corr!=Types::kMaxPreprocessingMethod; corr++) {
      fXminNorm[corr] = new Double_t[Data().GetNVariables()];
      fXmaxNorm[corr] = new Double_t[Data().GetNVariables()];
   }

   for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
      SetXmin(ivar, Data().GetXmin(ivar, Types::kNone), Types::kNone);
      SetXmax(ivar, Data().GetXmax(ivar, Types::kNone), Types::kNone);
   }


   // default extension for weight files
   fFileExtension = "weights";
   fFileDir       = "weights";
   gSystem->MakeDirectory( fFileDir );
}

//_______________________________________________________________________
TMVA::MethodBase::MethodBase( DataSet&     theData,
                              TString      weightFile,
                              TDirectory*  theBaseDir )
   : IMethod(),
     fData          ( theData ),
     fJobName       ( "" ),
     fOptions       ( "" ),
     fBaseDir       ( theBaseDir ),
     fWeightFile    ( weightFile ),
     fWeightFileType( kTEXT ),
     fUseDecorr     ( kFALSE ),
     fPreprocessingMethod(Types::kNone),
     fVerbose       ( kTRUE ),
     fHelp          ( kFALSE ),
     fLogger        (this)
{
   // constructor used for Testing + Application of the MVA, 
   // only (no training), using given WeightFiles
  
   this->Init();

   //   TMVA::MethodBase::DeclareOptions();
   DeclareOptions();

   fXminNorm[0] = fXminNorm[1] = fXmaxNorm[0] = fXmaxNorm[1] = 0;
}

//_______________________________________________________________________
TMVA::MethodBase::~MethodBase( void )
{
   // default destructur
}

//_______________________________________________________________________
namespace TMVA
{
   // helper functions to make ParseOptions cleaner

   Bool_t IsLastOption(TString& theOpt)
   {
      return (theOpt.First(':')<0);
   }

   //-----------

   void SeparateOptions(TString& theOpt, TList& loo)
   {
      while (theOpt.Length()>0) {
         if (IsLastOption(theOpt)) {
            loo.Add(new TObjString(theOpt));
            theOpt = "";
         } 
         else {
            TString toSave = theOpt(0,theOpt.First(':'));
            loo.Add(new TObjString(toSave.Data()));
            theOpt = theOpt(theOpt.First(':')+1,theOpt.Length());
         }
      }
   }  
}

//-----------------

void TMVA::MethodBase::ParseOptions( Bool_t verbose ) 
{
   // options parser

   if (verbose) {
      fLogger << kINFO << "parsing option string: " << Endl;
      fLogger << kINFO << "\"" << fOptions << "\"" << Endl;
   }
   
   TString theOpt(fOptions);
   TList loo; // the List Of Options in the parsed string
   
   theOpt = theOpt.Strip(TString::kLeading, ':');
   
   // separate the options by the ':' marker
   SeparateOptions(theOpt, loo);
   
   // loop over the declared options and check for their availability
   TListIter decOptIt(&fListOfOptions); // declared options
   TListIter setOptIt(&loo);   // parsed options
   while (TObjString * os = (TObjString*) setOptIt()) {
      TString s = os->GetString();
      bool paramParsed = false;
      if (s.Contains('=')) { // desired way of setting an option: "...:optname=optvalue:..."
         TString optname = s(0,s.First('=')); optname.ToLower();
         TString optval = s(s.First('=')+1,s.Length());
         OptionBase * decOpt = 0;
         TListIter optIt(&fListOfOptions);
         while ( (decOpt = (OptionBase*)optIt()) !=0) {
            TString predOptName(decOpt->GetName());
            predOptName.ToLower();
            if (predOptName == optname) break;
         }
         if (decOpt!=0) {
            if (decOpt->IsSet())
               if (verbose) 
                  fLogger << kWARNING << "value for option " << decOpt->GetName() 
                          << " was previously set to " << decOpt->GetValue() << Endl;

            if (!decOpt->HasPreDefinedVal() || (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(optval)) ) {
               decOpt->SetValue(optval);
               paramParsed = kTRUE;
            } 
            else fLogger << kFATAL << "option " << decOpt->TheName() 
                         << " has no predefined value " << optval << Endl;               
         } 
         else fLogger << kFATAL << "option " << optname << " not found!" << Endl;
      }

      // boolean variables can be specified by just their name (!name), 
      // which will set the to true (false):  ...:V:...:!S:..
      if (!paramParsed) { 
         bool hasNot = false;
         if (s.BeginsWith("!")) { s.Remove(0,1); hasNot=true; }
         TString optname(s);optname.ToLower();
         OptionBase * decOpt = 0;
         TListIter optIt(&fListOfOptions);
         while ( (decOpt = (OptionBase*)optIt()) !=0) {
            if (dynamic_cast<Option<bool>*>(decOpt)==0) continue; // not a boolean option
            TString predOptName(decOpt->GetName());
            predOptName.ToLower();
            if (predOptName == optname) break;
         }
        
         if (decOpt!=0) {
            decOpt->SetValue(hasNot?"0":"1");
            paramParsed = true;
         } 
         else {
            if (hasNot) {
               fLogger << kFATAL << "negating a non-boolean variable " << decOpt->GetName()
                       << ", please check the opions for method " << GetName() << Endl;
            }
         }
      }

      if (!paramParsed && LooseOptionCheckingEnabled()) {
         // loose options specification, loops through the possible string 
         // values any parameter can have not applicable for boolean or floats
         decOptIt.Reset();
         while (OptionBase * decOpt = (OptionBase*) decOptIt()) {
            if (decOpt->Type()=="float" || decOpt->Type()=="bool" ) continue;
            TString sT;
            if (decOpt->HasPreDefinedVal() && decOpt->IsPreDefinedVal(s) ) {
               paramParsed = decOpt->SetValue(s);
               break;
            }
         }
      }
   
      if (!paramParsed) {
         fLogger << kFATAL << "can not interpret option \"" << s << "\" for method " 
                 << GetName() << ", please check" << Endl;
      } 
   }
   PrintOptions();
}

//_______________________________________________________________________
void TMVA::MethodBase::Init()
{
   // default initialisation called by all constructors
   fIsOK          = kTRUE;
   fNvar          = Data().GetNVariables();
   fMeanS         = -1; // it is nice to have them "initialized". Every method
   fMeanB         = -1; // but "MethodCuts" sets them later
   fRmsS          = -1;
   fRmsB          = -1;

   fNbins         = NBIN_HIST_PLOT;
   fNbinsH        = NBIN_HIST_HIGH;

   fRanking       = NULL;

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

   fXminNorm[0]   = 0;
   fXminNorm[1]   = 0;
   fXminNorm[2]   = 0;
   fXmaxNorm[0]   = 0;
   fXmaxNorm[1]   = 0;
   fXmaxNorm[2]   = 0;

   fSignalReferenceCut = 0.5;

   fPreprocessingType  = Types::kSignal;

   // temporary until the move to DataSet is complete
   fInputVars = new vector<TString>;
   for(UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++)
      fInputVars->push_back(Data().GetInternalVarName(ivar));

   // define "this" pointer
   ResetThisBase();
}

//_______________________________________________________________________
void TMVA::MethodBase::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   // here the options valid for ALL MVA methods are declared.
   // know options: Preprocess=None,Decorrelated,PCA  to use decorrelated variables 
   //                                                 instead of the original ones
   //               PreprocessType=Signal,Background  which decorrelation matrix to use
   //                                                 in the method. Only the Likelihood
   //                                                 Method can make proper use of independent
   //                                                 transformations of signal and background
   //               V  for Verbose output (!V) for non verbos
   //               H  for Help 

   DeclareOptionRef(fUseDecorr, "D", "use-decorrelated-variables flag (for backward compatibility)");

   DeclareOptionRef(fPreprocessingString="None", "Preprocess", "Variable Decorrelation Method");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("Decorrelate"));
   AddPreDefVal(TString("PCA"));

   DeclareOptionRef(fPreprocessingTypeString="Signal", "PreprocessType", "Use signal or background for Preprocess");
   AddPreDefVal(TString("Signal"));
   AddPreDefVal(TString("Background"));

   DeclareOptionRef(fVerbose, "V","verbose flag");
   DeclareOptionRef(fHelp, "H","help flag");
}

//_______________________________________________________________________
void TMVA::MethodBase::ProcessOptions() 
{
   // the option string is decoded, for availabel options see "DeclareOptions"

   if      (fPreprocessingString == "None")         fPreprocessingMethod = Types::kNone;
   else if (fPreprocessingString == "Decorrelate" ) fPreprocessingMethod = Types::kDecorrelated;
   else if (fPreprocessingString == "PCA" )         fPreprocessingMethod = Types::kPCA;
   else {
      fLogger << kFATAL << "<ProcessOptions> preprocess parameter '" 
              << fPreprocessingString << "' unknown." << Endl;
   }

   // for backward compatibility
   if ((fPreprocessingMethod == Types::kNone) && fUseDecorr) fPreprocessingMethod = Types::kDecorrelated;

   if      (fPreprocessingTypeString == "Signal")      fPreprocessingType = Types::kSignal;
   else if (fPreprocessingTypeString == "Background" ) fPreprocessingType = Types::kBackground;
   else {
      fLogger << kFATAL << "<ProcessOptions> preprocess type '" 
              << fPreprocessingTypeString << "' unknown." << Endl;
   }

   if (Verbose()) fLogger.SetMinType( kVERBOSE );


   if( GetPreprocessingMethod() == Types::kDecorrelated ) {
      Types::EPreprocessingMethod c = Types::kDecorrelated;
      Data().EnablePreprocess(Types::kDecorrelated);
      if( Data().Preprocess(Types::kDecorrelated) ) {
         // local copy of the variable ranges 
         // used for normalization
         for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
            SetXmin(ivar, Data().GetXmin(ivar, c), c);
            SetXmax(ivar, Data().GetXmax(ivar, c), c);
         }
      }
   }

   if( GetPreprocessingMethod() == Types::kPCA ) {
      Types::EPreprocessingMethod c = Types::kPCA;
      Data().EnablePreprocess(Types::kPCA);
      if( Data().Preprocess(Types::kPCA) ) {
         // local copy of the variable ranges 
         // used for normalization
         for (UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
            SetXmin(ivar, Data().GetXmin(ivar, c), c);
            SetXmax(ivar, Data().GetXmax(ivar, c), c);
         }
      }
   }



}

//_______________________________________________________________________
void TMVA::MethodBase::TrainMethod() 
{ 
   // all histograms should be created in the method's subdirectory
   BaseDir()->cd();

   Train();
   WriteStateToFile();
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteStateToStream(std::ostream& o) const 
{
   // general method used in writing the header of the weight files where
   // the used variables, preprocessing type etc. is specified

   o << "#GEN -*-*-*-*-*-*-*-*-*-*-*- general info -*-*-*-*-*-*-*-*-*-*-*-" << endl << endl;
   o << "Method : " << GetMethodName() << endl;
   o << "Creator: " << gSystem->GetUserInfo()->fUser << endl;
   o << "Date   : "; TDatime *d = new TDatime; o << d->AsString() << endl; delete d;
   o << "Host   : " << gSystem->GetBuildNode() << endl;
   o << "Dir    : " << gSystem->Getenv("PWD") << endl;
   o << "Training events: " << Data().GetNEvtTrain() << endl;
   o << endl;

   // First write all options
   o << endl << "#OPT -*-*-*-*-*-*-*-*-*-*-*-*- options -*-*-*-*-*-*-*-*-*-*-*-*-" << endl << endl;
   WriteOptionsToStream(o);
   o << endl;
      
   // Second write variable info
   o << endl << "#VAR -*-*-*-*-*-*-*-*-*-*-*-* variables *-*-*-*-*-*-*-*-*-*-*-*-" << endl << endl;
   Data().WriteVarsToStream(o, GetPreprocessingMethod()); 
   o << endl;

   // Third write decorrelation matrix if available
   if (GetPreprocessingMethod() != Types::kNone) {
      o << endl << "#MAT -*-*-*-*-*-*-*-*-* decorrelation matrix -*-*-*-*-*-*-*-*-*-" << endl;
      Data().WriteCorrMatToStream(o); 
      o << endl;
   }

   // Fourth, write weights
   o << endl << "#WGT -*-*-*-*-*-*-*-*-*-*-*-*- weights -*-*-*-*-*-*-*-*-*-*-*-*-" << endl << endl;
   WriteWeightsToStream(o);

   // write additional monitoring histograms to main target file
   WriteMonitoringHistosToFile();
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteStateToFile() const
{ 
   // Function to write options and weights to file

   if (GetWeightFileType()==kTEXT) {

      // get the filename
      TString fname(GetWeightFileName());
      fLogger << kINFO << "creating weight file: " << BC_blue << fname << EC__ << Endl;

      ofstream fout( fname );
      if (!fout.good()) { // file not found --> Error
         fLogger << kFATAL << "<WriteStateToFile> "
                 << "unable to open output weight file: " << fname << Endl;
      }

      WriteStateToStream(fout);

      fout.close();
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::ReadStateFromFile() 
{ 
   // Function to write options and weights to file

   // get the filename
   TString fname(GetWeightFileName());

   fLogger << kINFO << "reading weight file: " << BC_blue << fname << EC__ << Endl;

   if (GetWeightFileType()==kTEXT) {

      ifstream fin( fname );
      if (!fin.good()) { // file not found --> Error
         fLogger << kFATAL << "<ReadStateFromFile> "
                 << "unable to open input weight file: " << fname << Endl;
      }

      ReadStateFromStream(fin);
      fin.close();
   }
}

//_______________________________________________________________________
void TMVA::MethodBase::ReadStateFromStream( std::istream& fin )
{     
   // read the header from the weight files of the different MVA methods

   char buf[512];
   
   // first read the method name
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("Method")) fin.getline(buf,512);
   TString ls(buf);
   Int_t idx1 = ls.First(':')+2; Int_t idx2 = ls.Index(' ',idx1)-idx1; if (idx2<0) idx2=ls.Length();
   this->SetMethodName(ls(idx1,idx2));
   
   // now the question is whether to read the variables first or the options (well, of course the order 
   // of writing them needs to agree) the option "Decorrelation" is needed to decide if the variables we 
   // read are decorrelated or not the variables are needed by some methods (TMLP) to build the NN 
   // which is done in ProcessOptions so for the time being we first Read and Parse the options then 
   // we read the variables, and then we process the options
   
   // now read all options
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("#OPT")) fin.getline(buf,512);
   

   ReadOptionsFromStream(fin);
   ParseOptions(Verbose());

   // Now read variable info
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("#VAR")) {
      fin.getline(buf,512);
   }
   Data().ReadVarsFromStream(fin, GetPreprocessingMethod());

   // now we process the options
   ProcessOptions();

   // Now read decorrelation matrix if available
   if ( GetPreprocessingMethod() != Types::kNone ) {
      fin.getline(buf,512);
      while (!TString(buf).BeginsWith("#MAT")) fin.getline(buf,512);
      Data().ReadCorrMatFromStream(fin);
   }

   // now the local min and max array can and need to be set
   for (Int_t corr=0; corr!=Types::kMaxPreprocessingMethod; corr++) {
      if (0 != fXminNorm[corr]) delete fXminNorm[corr]; 
      if (0 != fXmaxNorm[corr]) delete fXmaxNorm[corr]; 
      fXminNorm[corr] = new Double_t[Data().GetNVariables()];
      fXmaxNorm[corr] = new Double_t[Data().GetNVariables()];
      Types::EPreprocessingMethod c = (Types::EPreprocessingMethod) corr;
      for(UInt_t ivar=0; ivar<Data().GetNVariables(); ivar++) {
         SetXmin(ivar, Data().GetXmin(ivar, c), c);
         SetXmax(ivar, Data().GetXmax(ivar, c), c);
      }
   }


   // Now read weights
   fin.getline(buf,512);
   while (!TString(buf).BeginsWith("#WGT")) fin.getline(buf,512);
   fin.getline(buf,512);

   ReadWeightsFromStream(fin);   
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::GetEventValNormalized(Int_t ivar) const 
{ 
   // return the normalized event variable (normalized to interval [0,1]
   return Tools::NormVariable( Data().Event().GetVal(ivar), 
                               GetXmin(ivar, GetPreprocessingMethod()),
                               GetXmax(ivar, GetPreprocessingMethod()));
}

//_______________________________________________________________________
TDirectory * TMVA::MethodBase::BaseDir( void ) const
{
   // returns the ROOT directory where info/histograms etc of the 
   // corresponding MVA method are stored

   if (fBaseDir != 0) return fBaseDir;

   TDirectory* dir = 0;

   TObject * o = Data().BaseRootDir()->FindObject(GetMethodTitle());
   if (o!=0 && o->InheritsFrom("TDirectory")) dir = (TDirectory*)o;
   if (dir != 0) return dir;

   return Data().BaseRootDir()->mkdir(GetMethodTitle());
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
   TString weightFileName =  fFileDir + "/" + fJobName + "_" + fMethodTitle + suffix + "." + fFileExtension;
   if (GetWeightFileType()==kROOT) weightFileName += ".root";
   if (GetWeightFileType()==kTEXT) weightFileName += ".txt";
   return weightFileName;

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
   return TMVA::Tools::NormVariable( x, GetXmin( var ), GetXmax( var ) );
}

//_______________________________________________________________________
Double_t TMVA::MethodBase::Norm( Int_t ivar, Double_t x ) const
{
   // renormalises variable with respect to its min and max
   return TMVA::Tools::NormVariable( x, GetXmin( ivar ), GetXmax( ivar ) );
}

//_______________________________________________________________________
void TMVA::MethodBase::TestInit(TTree* theTestTree)
{
   // initialisation of MVA testing 
   if (theTestTree == 0)
      theTestTree = Data().GetTestTree(); // sets theTestTree to the TestTree in the DataSet,
   // decorrelation properly taken care of

   //  fTestTree       = theTestTree;
   fHistS_plotbin  = fHistB_plotbin = 0;
   fHistS_highbin  = fHistB_highbin = 0;
   fEffS           = fEffB = fEffBvsS = fRejBvsS = 0;
   fGraphS         = fGraphB = 0;
   fCutOrientation = kPositive;
   fSplS           = fSplB = 0;
   fSplRefS        = fSplRefB = 0;


   // sanity checks: tree must exist, and theVar must be in tree
   if (0 == theTestTree) {
      fLogger << kFALSE << "<TestInit> test tree has zero pointer " << Endl;
      fIsOK = kFALSE;
   }
   if ( 0 == theTestTree->FindBranch( GetTestvarName() ) && !(GetMethodName().Contains("Cuts"))) {
      fLogger << kFALSE << "<TestInit> test variable " << GetTestvarName()
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
      fLogger << kFALSE << "<PrepareEvaluationTree> sanity check failed" << Endl;
   }

   // read the coefficients
   this->ReadStateFromFile();

   // use timer
   TMVA::Timer timer( testTree->GetEntries(), GetName(), kTRUE );
   Data().BaseRootDir()->cd();
   Double_t myMVA = 0;
   TBranch *newBranch = testTree->Branch( GetTestvarName(), &myMVA, GetTestvarName() + "/D", 128000 );
   for (Int_t ievt=0; ievt<testTree->GetEntries(); ievt++) {
      if ((Int_t)ievt%100 == 0) timer.DrawProgressBar( ievt );
      ReadTestEvent(ievt);
      newBranch->SetAddress(&myMVA); // only when the tree changed, but we don't know when that is
      myMVA = GetMvaValue();
      newBranch->Fill();
   }

   Data().BaseRootDir()->Write("",TObject::kOverwrite);

   fLogger << kINFO << "elapsed time for evaluation of "
           << testTree->GetEntries() <<  " events: "
           << timer.GetElapsedTime() << "       " << Endl;

   newBranch->ResetAddress();
}

//_______________________________________________________________________
void TMVA::MethodBase::Test( TTree *theTestTree )
{
   // test the method - not much is done here... mainly furthor initialisation

   // If Empty tree: sets theTestTree to the TestTree in the DataSet,
   // decorrelation properly taken care of
   if (theTestTree == 0) theTestTree = Data().GetTestTree();

   // basic statistics operations are made in base class
   // note: cannot directly modify private class members
   Double_t meanS, meanB, rmsS, rmsB, xmin, xmax;

   TMVA::Tools::ComputeStat( theTestTree, GetTestvarName(), meanS, meanB, rmsS, rmsB, xmin, xmax );

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
   fHistS_plotbin = TMVA::Tools::projNormTH1F( theTestTree, GetTestvarName(),
                                               GetTestvarName() + "_S",
                                               fNbins, fXmin, fXmax, "weight*(type == 1)" );
   fHistB_plotbin = TMVA::Tools::projNormTH1F( theTestTree, GetTestvarName(),
                                               GetTestvarName() + "_B",
                                               fNbins, fXmin, fXmax, "weight*(type == 0)" );

   // need histograms with even more bins for efficiency calculation and integration
   fHistS_highbin = TMVA::Tools::projNormTH1F( theTestTree, GetTestvarName(),
                                               GetTestvarName() + "_S_high",
                                               fNbinsH, fXmin, fXmax, "weight*(type == 1)" );
   fHistB_highbin = TMVA::Tools::projNormTH1F( theTestTree, GetTestvarName(),
                                               GetTestvarName() + "_B_high",
                                               fNbinsH, fXmin, fXmax, "weight*(type == 0)" );

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
      fLogger << kFALSE << "<GetEfficiency> wrong number of arguments"
              << " in string: " << theString
              << " | required format, e.g., Efficiency:0.05" << Endl;
      return -1;
   }
   // that will be the value of the efficiency retured (does not affect
   // the efficiency-vs-bkg plot which is done anyway.
   Float_t effBref  = atof( ((TObjString*)list->At(1))->GetString() );

   if (DEBUG_TMVA_MethodBase)
      fLogger << kINFO << "<GetEfficiency> compute eff(S) at eff(B) = " << effBref << Endl;

   // sanity check
   if (fHistS_highbin->GetNbinsX() != fHistB_highbin->GetNbinsX() ||
       fHistS_plotbin->GetNbinsX() != fHistB_plotbin->GetNbinsX()) {
      fLogger << kWARNING << "<GetEfficiency> binning mismatch between signal and background histos" << Endl;
      fIsOK = kFALSE;
      return -1.0;
   }

   // create histograms

   // first, get efficiency histograms for signal and background
   Double_t xmin = fHistS_highbin->GetXaxis()->GetXmin();
   Double_t xmax = fHistS_highbin->GetXaxis()->GetXmax();

   // first round ? --> create histograms
   Bool_t firstPass = kFALSE;
   if (NULL == fEffS && NULL == fEffB) firstPass = kTRUE;

   if (firstPass) {

      fEffS = new TH1F( GetTestvarName() + "_effS", GetTestvarName() + " (signal)",     fNbinsH, xmin, xmax );
      fEffB = new TH1F( GetTestvarName() + "_effB", GetTestvarName() + " (background)", fNbinsH, xmin, xmax );

      // sign if cut
      Int_t sign = (fCutOrientation == kPositive) ? +1 : -1;

      // this method is unbinned
      Int_t    theType;
      Double_t theVal;
      TBranch* brType = theTree->GetBranch("type");
      TBranch* brVal  = theTree->GetBranch(GetTestvarName());
      if (brVal == 0) {
         fLogger << kFALSE << "could not find variable " 
                 << GetTestvarName() << " in tree " << theTree->GetName() << Endl;
      }
      brType->SetAddress(&theType);
      brVal ->SetAddress(&theVal );

      for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {

         // read the tree
         brType->GetEntry(ievt);
         brVal ->GetEntry(ievt);
         // select histogram depending on if sig or bgd
         TH1* theHist = (theType==1?fEffS:fEffB);
         for (Int_t bin=1; bin<=fNbinsH; bin++)
            if (sign*theVal >= sign*theHist->GetBinLowEdge( bin )) theHist->AddBinContent( bin );
      }
      
      // renormalize to maximum
      fEffS->Scale( 1.0/(fEffS->GetMaximum() > 0 ? fEffS->GetMaximum() : 1) );
      fEffB->Scale( 1.0/(fEffB->GetMaximum() > 0 ? fEffB->GetMaximum() : 1) );

      // now create efficiency curve: background versus signal
      fEffBvsS = new TH1F( GetTestvarName() + "_effBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      fRejBvsS = new TH1F( GetTestvarName() + "_rejBvsS", GetTestvarName() + "", fNbins, 0, 1 );
      // use root finder
      // spline background efficiency plot
      // note that there is a bin shift when going from a TH1F object to a TGraph :-(
      if (Use_Splines_for_Eff_) {
         fGraphS   = new TGraph( fEffS );
         fGraphB   = new TGraph( fEffB );
         fSplRefS  = new TMVA::TSpline1( "spline2_signal",     fGraphS );
         fSplRefB  = new TMVA::TSpline1( "spline2_background", fGraphB );

         // verify spline sanity
         fLogger << kVERBOSE << "<GetEfficiency> verify signal and background eff. splines" << Endl;
         TMVA::Tools::CheckSplines( fEffS, fSplRefS );
         TMVA::Tools::CheckSplines( fEffB, fSplRefB );
      }

      // make the background-vs-signal efficiency plot

      // create root finder
      // reset static "this" pointer before calling external function
      ResetThisBase();
      TMVA::RootFinder rootFinder(&IGetEffForRoot, fXmin, fXmax );

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
Double_t TMVA::MethodBase::GetTrainingEfficiency( TString theString)
{
   // fill background efficiency (resp. rejection) versus signal efficiency plots
   // returns signal efficiency at background efficiency indicated in theString

   // parse input string for required background efficiency
   TList*  list  = TMVA::Tools::ParseFormatLine( theString );
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

   if (DEBUG_TMVA_MethodBase)
      fLogger << kINFO << "<GetTrainingEfficiency> compute eff(S) at eff(B) = " 
              << effBref << Endl;

   // sanity check
   if (fHistS_highbin->GetNbinsX() != fHistB_highbin->GetNbinsX() ||
       fHistS_plotbin->GetNbinsX() != fHistB_plotbin->GetNbinsX()) {
      fLogger << kFATAL << "<GetTrainingEfficiency> binning mismatch between signal and background histos"
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

         TH1* theHist = (Data().Event().IsSignal() ? fTrainEffS : fTrainEffB);
 
         Double_t theVal = this->GetMvaValue();

         for (Int_t bin=1; bin<=fNbinsH; bin++)
            if (sign*theVal > sign*theHist->GetBinCenter( bin )) theHist->AddBinContent( bin );
      }

      // renormalize to maximum
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
         fSplTrainRefS  = new TMVA::TSpline1( "spline2_signal",     fGraphTrainS );
         fSplTrainRefB  = new TMVA::TSpline1( "spline2_background", fGraphTrainB );

         // verify spline sanity
         fLogger << kVERBOSE << "<GetEfficiency> verify signal and background eff. splines" << Endl;

         TMVA::Tools::CheckSplines( fTrainEffS, fSplTrainRefS );
         TMVA::Tools::CheckSplines( fTrainEffB, fSplTrainRefB );
      }

      // make the background-vs-signal efficiency plot

      // create root finder
      // reset static "this" pointer before calling external function
      ResetThisBase();
      TMVA::RootFinder rootFinder(&IGetEffForRoot, fXmin, fXmax );

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
      fSplTrainEffBvsS   = new TMVA::TSpline1( "effBvsS", fGraphTrainEffBvsS );
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
Double_t TMVA::MethodBase::GetSignificance( void )
{
   // compute significance of mean difference
   // significance = |<S> - <B>|/Sqrt(RMS_S2 + RMS_B2)
   Double_t rms = sqrt( fRmsS*fRmsS + fRmsB*fRmsB );

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

   fLogger << kVERBOSE << "get optimal significance ..." << Endl;
  
   Double_t optimal_significance(0);    
   Double_t effS(0),effB(0),significance(0);
   TH1F *temp_histogram = new TH1F("temp", "temp", fNbinsH, fXmin, fXmax );

   if (SignalEvents <= 0 || BackgroundEvents <= 0) {
      fLogger << kFATAL << "<GetOptimalSignificance> "
              << "number of signal or background events is <= 0 ==> abort"
              << Endl;
   }

   fLogger << kINFO << "using ratio SignalEvents/BackgroundEvents = "
           << SignalEvents/BackgroundEvents << Endl;
    
   if ((fEffS == 0) || (fEffB == 0)) {
      fLogger << kWARNING << "efficiency histograms empty !" << Endl;
      fLogger << kWARNING << "no optimal cut found, return 0" << Endl;
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
  
   fLogger << kINFO << "optimal cut at      : " << optimal_significance << Endl;
   fLogger << kINFO << "optimal significance: " << optimal_significance_value << Endl;
  
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

   Double_t x;
   TBranch * br = theTree->GetBranch(GetTestvarName());
   for (Int_t ievt=0; ievt<theTree->GetEntries(); ievt++) {
      Data().ReadEvent(theTree,ievt);
      br->SetAddress(&x);
      br->GetEvent(ievt);
      Double_t s = fSplS->GetVal( x );
      Double_t b = fSplB->GetVal( x );
      Double_t aBhat = 0;
      if (b + s > 0) aBhat = b/(b + s);

      if (Data().Event().IsSignal()) { // this is signal
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
void TMVA::MethodBase::Statistics( TMVA::Types::ETreeType treeType, const TString& theVarName,
                                   Double_t& meanS, Double_t& meanB,
                                   Double_t& rmsS,  Double_t& rmsB,
                                   Double_t& xmin,  Double_t& xmax,
                                   Bool_t    norm )
{
   // calculates rms,mean, xmin, xmax of the event variable
   // this can be either done for the variables as they are or for
   // normalised variables (in the range of 0-1) if "norm" is set to kTRUE

   Long64_t entries = ( (treeType == TMVA::Types::kTesting ) ? Data().GetNEvtTest() :
                        (treeType == TMVA::Types::kTraining) ? Data().GetNEvtTrain() : -1 );

   // sanity check
   if (entries <=0) 
      fLogger << kFATAL << "<CalculateEstimator> wrong tree type: " << treeType << Endl;

   // index of the wanted variable
   UInt_t varIndex = Data().FindVar( theVarName );

   // first fill signal and background in arrays before analysis
   Double_t* varVecS  = new Double_t[entries];
   Double_t* varVecB  = new Double_t[entries];
   xmin               = +1e20;
   xmax               = -1e20;
   Long64_t nEventsS  = -1;
   Long64_t nEventsB  = -1;

   // loop over all training events 
   for (Int_t i = 0; i < entries; i++) {

      if (treeType == TMVA::Types::kTesting )
         ReadTestEvent(i);
      else
         ReadTrainingEvent(i);
      
      Double_t theVar = (norm) ? GetEventValNormalized(varIndex) : GetEventVal(varIndex);

      if (Data().Event().IsSignal()) varVecS[++nEventsS] = theVar;
      else                           varVecB[++nEventsB] = theVar;

      xmin = TMath::Min( xmin, theVar );
      xmax = TMath::Max( xmax, theVar );
   }
   ++nEventsS;
   ++nEventsB;

   // basic statistics
   meanS = TMath::Mean( nEventsS, varVecS );
   meanB = TMath::Mean( nEventsB, varVecB );
   rmsS  = TMath::RMS ( nEventsS, varVecS );
   rmsB  = TMath::RMS ( nEventsB, varVecB );

   delete [] varVecS;
   delete [] varVecB;
}

//_______________________________________________________________________
void TMVA::MethodBase::WriteEvaluationHistosToFile( TDirectory* targetDir )
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

//______________________________________________________________________
void TMVA::MethodBase::PrintOptions() const 
{
   // prints out the options set in the options string and the defaults

   fLogger << kINFO << "the following options are set:" << Endl;
   TListIter optIt( & ListOfOptions() );
   fLogger << kINFO << "by User:" << Endl;
   fLogger << kINFO << "--------" << Endl;
   while (OptionBase * opt = (OptionBase *) optIt()) {
      if (opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; }
   }
   optIt.Reset();
   fLogger << kINFO << "default:" << Endl;
   fLogger << kINFO << "--------" << Endl;
   while (OptionBase * opt = (OptionBase *) optIt()) {
      if (!opt->IsSet()) { fLogger << kINFO << "    "; opt->Print(fLogger); fLogger << Endl; }
   }
}

//______________________________________________________________________
void TMVA::MethodBase::WriteOptionsToStream(ostream& o) const 
{
   // write options to output stream (e.g. in writing the MVA weight files

   TListIter optIt( & ListOfOptions() );
   o << "# Set by User:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) if (opt->IsSet()) { opt->Print(o); o << endl; }
   optIt.Reset();
   o << "# Default:" << endl;
   while (OptionBase * opt = (OptionBase *) optIt()) if (!opt->IsSet()) { opt->Print(o); o << endl; }
   o << "##" << endl;
}

//______________________________________________________________________
void TMVA::MethodBase::ReadOptionsFromStream(istream& istr)
{
   // read option back from the weight file

   fOptions = "";
   char buf[512];
   istr.getline(buf,512);
   TString stropt, strval;
   while (istr.good() && !istr.eof() && !(buf[0]=='#' && buf[1]=='#')) { // if line starts with ## return
      char *p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512); // reading the next line
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> stropt >> strval;
      stropt.ReplaceAll(':','=');
      strval.ReplaceAll("\"","");
      if (fOptions.Length()!=0) fOptions += ":";
      fOptions += stropt;
      fOptions += strval;
      istr.getline(buf,512); // reading the next line
   }
}

//_______________________________________________________________________
void  TMVA::MethodBase::WriteMonitoringHistosToFile( void ) const
{
   // write special monitoring histograms to file - not implemented for this method
   fLogger << kINFO << "no monitoring histograms written" << Endl;
}
