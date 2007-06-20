// @(#)root/tmva $Id: MethodFDA.cxx,v 1.16 2007/06/15 22:01:32 andreas.hoecker Exp $    
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodFDA                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker  <Andreas.Hocker@cern.ch> - CERN, Switzerland             *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005-2006:                                                       *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Function discriminant analysis (FDA). This simple classifier         //
// fits any user-defined TFormula (via option configuration string) to  //
// the training data by requiring a formula response of 1 (0) to signal //
// (background) events. The parameter fitting is done via the abstract  //
// class FitterBase, featuring Monte Carlo sampling, Genetic            //
// Algorithm, Simulated Annealing, MINUIT and combinations of these.    //
//_______________________________________________________________________

#include "Riostream.h"
#include "TList.h"
#include "TFormula.h"
#include "TString.h"
#include "TObjString.h"
#include "TRandom.h"

#include "TMVA/MethodFDA.h"
#include "TMVA/Tools.h"
#include "TMVA/Interval.h"
#include "TMVA/Timer.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/SimulatedAnnealingFitter.h"
#include "TMVA/MinuitFitter.h"
#include "TMVA/MCFitter.h"

ClassImp(TMVA::MethodFDA)

//_______________________________________________________________________
TMVA::MethodFDA::MethodFDA( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : MethodBase( jobName, methodTitle, theData, theOption, theTargetDir ), 
     IFitterTarget()
{
   // standard constructor
   InitFDA();

   // interpretation of configuration option string
   DeclareOptions();
   ParseOptions();
   ProcessOptions();
}

//_______________________________________________________________________
TMVA::MethodFDA::MethodFDA( DataSet& theData, 
                            TString theWeightFile,  
                            TDirectory* theTargetDir )
   : MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor from weight file
   InitFDA();

   DeclareOptions();
}

//_______________________________________________________________________
void TMVA::MethodFDA::InitFDA( void )
{
   // default initialisation
   SetMethodName( "FDA" );
   SetMethodType( Types::kFDA );
   SetTestvarName();

   fNPars    = 0;
   fFormula  = 0;
   fBestPars.clear();

   fEventsSig.clear();
   fEventsBkg.clear();

   fSumOfWeightsSig = 0;
   fSumOfWeightsBkg = 0;
}

//_______________________________________________________________________
void TMVA::MethodFDA::DeclareOptions() 
{
   // define the options (their key words) that can be set in the option string 
   //
   // format of function string:
   //    "x0*(0)+((1)/x1)**(2)..."
   // where "[i]" are the parameters, and "xi" the input variables
   //
   // format of parameter string:
   //    "(-1.2,3.4);(-2.3,4.55);..."
   // where the numbers in "(a,b)" correspond to the a=min, b=max parameter ranges;
   // each parameter defined in the function string must have a corresponding range
   //
   DeclareOptionRef( fFormulaStringP  = "", "Formula",   "The discrimination formula" );
   DeclareOptionRef( fParRangeStringP = "", "ParRanges", "Parameter ranges" );

   // fitter
   DeclareOptionRef( fFitMethod = "MINUIT", "FitMethod", "Optimisation Method");
   AddPreDefVal(TString("MC"));
   AddPreDefVal(TString("GA"));
   AddPreDefVal(TString("SA"));
   AddPreDefVal(TString("MINUIT"));

   DeclareOptionRef( fConverger = "None", "Converger", "FitMethod uses Converger to improve result");
   AddPreDefVal(TString("None"));
   AddPreDefVal(TString("MINUIT"));
}

//_______________________________________________________________________
void TMVA::MethodFDA::ProcessOptions() 
{
   // the option string is decoded, for availabel options see "DeclareOptions"
   MethodBase::ProcessOptions();

   // clean up first
   ClearAll();

   // process transient strings
   fFormulaStringT  = fFormulaStringP;
   fParRangeStringT = fParRangeStringP;

   // interpret parameter string   
   fParRangeStringT.ReplaceAll( " ", "" );
   fNPars = fParRangeStringT.CountChar( ')' );

   TList* parList = Tools::ParseFormatLine( fParRangeStringT, ";" );
   if (parList->GetSize() != fNPars) {
      fLogger << kFATAL << "<ProcessOptions> Mismatch in parameter string: " 
              << "the number of parameters: " << fNPars << " != ranges defined: " 
              << parList->GetSize() << "; the format of the \"ParRanges\" string "
              << "must be: \"(-1.2,3.4);(-2.3,4.55);...\", "
              << "where the numbers in \"(a,b)\" correspond to the a=min, b=max parameter ranges; "
              << "each parameter defined in the function string must have a corresponding rang."
              << Endl;
   }

   fParRange.resize( fNPars );
   for (Int_t ipar=0; ipar<fNPars; ipar++) fParRange[ipar] = 0;

   for (Int_t ipar=0; ipar<fNPars; ipar++) {
      // parse (a,b)
      TString str = ((TObjString*)parList->At(ipar))->GetString();
      Ssiz_t istr = str.First( ',' );
      TString pminS(str(1,istr-1));
      TString pmaxS(str(istr+1,str.Length()-2-istr));
      Float_t pmin = atof(pminS.Data());
      Float_t pmax = atof(pmaxS.Data());

      // sanity check
      if (pmin > pmax) fLogger << kFATAL << "<ProcessOptions> max > min in interval for parameter: [" 
                               << ipar << "] : [" << pmin  << ", " << pmax << "] " << Endl;

      fParRange[ipar] = new Interval( pmin, pmax );
   }
   
   // intepret formula string

   // replace the parameters "(i)" by the TFormula style "[i]"
   for (Int_t ipar=0; ipar<fNPars; ipar++) {
      fFormulaStringT.ReplaceAll( Form("(%i)",ipar), Form("[%i]",ipar) );
   }

   // sanity check, there should be no "(i)", with 'i' a number anymore
   for (Int_t ipar=fNPars; ipar<1000; ipar++) {
      if (fFormulaStringT.Contains( Form("(%i)",ipar) ))
         fLogger << kFATAL 
                 << "<ProcessOptions> Formula contains expression: \"" << Form("(%i)",ipar) << "\", "
                 << "which cannot be attributed to a parameter; " 
                 << "it may be that the number of variable ranges given via \"ParRanges\" "
                 << "does not match the number of parameters in the formula expression, please verify!"
                 << Endl;
   }

   // write the variables "xi" as additional parameters "[npar+i]"
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      fFormulaStringT.ReplaceAll( Form("x%i",ivar), Form("[%i]",ivar+fNPars) );
   }

   // sanity check, there should be no "xi", with 'i' a number anymore
   for (Int_t ivar=GetNvar(); ivar<1000; ivar++) {
      if (fFormulaStringT.Contains( Form("x%i",ivar) ))
         fLogger << kFATAL 
                 << "<ProcessOptions> Formula contains expression: \"" << Form("x%i",ivar) << "\", "
                 << "which cannot be attributed to an input variable" << Endl;
   }
   
   fLogger << "User-defined formula string       : \"" << fFormulaStringP << "\"" << Endl;
   fLogger << "TFormula-compatible formula string: \"" << fFormulaStringT << "\"" << Endl;
   fLogger << "Creating and compiling formula" << Endl;
   
   // create TF1
   fFormula = new TFormula( "FDA_Formula", fFormulaStringT );
   fFormula->Optimize();

   // is formula correct ?
   if (fFormula->Compile() != 0)
      fLogger << kFATAL << "<ProcessOptions> Formula expression could not be properly compiled" << Endl;

   // other sanity checks
   if (fFormula->GetNpar() > fNPars + GetNvar())
      fLogger << kFATAL << "<ProcessOptions> Dubious number of parameters in formula expression: " 
              << fFormula->GetNpar() << " - compared to maximum allowed: " << fNPars + GetNvar() << Endl;

   fConvergerFitter = (IFitterTarget*)this;
   if (fConverger == "MINUIT") {
      fConvergerFitter = new MinuitFitter( *this, Form("%s_Converger_Minuit", GetName()), fParRange, GetOptions() );
      SetOptions(dynamic_cast<Configurable*>(fConvergerFitter)->GetOptions());
   }

   if      (fFitMethod == "MC")     
      fFitter = new MCFitter( *fConvergerFitter, Form("%s_Fitter_MC", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "GA")     
      fFitter = new GeneticFitter( *fConvergerFitter, Form("%s_Fitter_GA", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "SA")     
      fFitter = new SimulatedAnnealingFitter( *fConvergerFitter, Form("%s_Fitter_SA", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "MINUIT") 
      fFitter = new MinuitFitter( *fConvergerFitter, Form("%s_Fitter_Minuit", GetName()), fParRange, GetOptions() );
   else {
      fLogger << kFATAL << "<Train> Do not understand fit method:" << fFitMethod << Endl;
   }
   
   fFitter->CheckForUnusedOptions();
}

//_______________________________________________________________________
TMVA::MethodFDA::~MethodFDA( void )
{
   // destructor
   ClearAll();
}

//_______________________________________________________________________
void TMVA::MethodFDA::ClearAll( void )
{
   // delete and clear all class members
   for (UInt_t ipar=0; ipar<fParRange.size(); ipar++) {
      if (fParRange[ipar] != 0) { delete fParRange[ipar]; fParRange[ipar] = 0; }
   }
   fParRange.clear(); 
   
   if (fFormula  != 0) { delete fFormula; fFormula = 0; }
   fBestPars.clear();
}

//_______________________________________________________________________
void TMVA::MethodFDA::Train( void )
{
   // FDA training 

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;

   // cache training events
   fSumOfWeightsSig = 0;
   fSumOfWeightsBkg = 0;

   for (Int_t ievt=0; ievt<Data().GetNEvtTrain(); ievt++) {

      // read the training event 
      ReadTrainingEvent(ievt);

      // true event copy
      Event*  ev = new Event( GetEvent() );
      Float_t w  = ev->GetWeight();

      if (ev->IsSignal()) { fEventsSig.push_back( ev ); fSumOfWeightsSig += w; }
      else                { fEventsBkg.push_back( ev ); fSumOfWeightsBkg += w; }
   }

   // sanity check
   if (fSumOfWeightsSig <= 0 || fSumOfWeightsBkg <= 0) {
      fLogger << kFATAL << "<Train> Troubles in sum of weights: " 
              << fSumOfWeightsSig << " (S) : " << fSumOfWeightsBkg << " (B)" << Endl;
   }

   // starting values (not used by all fitters)
   fBestPars.clear();
   for (std::vector<Interval*>::const_iterator parIt = fParRange.begin(); parIt != fParRange.end(); parIt++) {
      fBestPars.push_back( (*parIt)->GetMean() );
   }

   // execute the fit
   Double_t estimator = fFitter->Run( fBestPars );
      
   // print results
   PrintResults( fFitMethod, fBestPars, estimator );

   // free cache 
   std::vector<const Event*>::const_iterator itev;
   for (itev = fEventsSig.begin(); itev != fEventsSig.end(); itev++) delete *itev;
   for (itev = fEventsBkg.begin(); itev != fEventsBkg.end(); itev++) delete *itev;

   fEventsSig.clear();
   fEventsBkg.clear();

   if (fConverger == "MINUIT") delete fConvergerFitter;
   delete fFitter; fFitter = 0;
}

//_______________________________________________________________________
void TMVA::MethodFDA::PrintResults( const TString& fitter, std::vector<Double_t>& pars, const Double_t estimator ) const
{
   // display fit parameters
   // check maximum length of variable name
   fLogger << kINFO;
   fLogger << "Results for parameter fit using \"" << fitter << "\" fitter:" << Endl;
   vector<TString>  parNames;
   for (UInt_t ipar=0; ipar<pars.size(); ipar++) parNames.push_back( Form("Par(%i)",ipar ) );
   Tools::FormattedOutput( pars, parNames, "Parameter" , "Fit result", fLogger, "%g" );   
   fLogger << "Discriminator expression: \"" << fFormulaStringP << "\"" << Endl;
   fLogger << "Value of estimator at minimum: " << estimator << Endl;
}

//_______________________________________________________________________
Double_t TMVA::MethodFDA::EstimatorFunction( std::vector<Double_t>& pars )
{
   // compute estimator for given parameter set (to be minimised)

   // species-specific stuff
   const std::vector<const Event*>* eventVecs[] = { &fEventsSig, &fEventsBkg };
   const Double_t sumOfWeights[]                = { fSumOfWeightsSig, fSumOfWeightsBkg };
   const Double_t desiredVal[]                  = { 1, 0 };
   Double_t estimator[]                         = { 0, 0 };
   std::vector<const Event*>::const_iterator itev;

   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      // loop over specific events
      for (itev = eventVecs[itype]->begin(); itev != eventVecs[itype]->end(); itev++) {

         // read the training event 
         Double_t result    = InterpretFormula( **itev, pars );
         Double_t deviation = (result - desiredVal[itype])*(result - desiredVal[itype]);

         estimator[itype] += deviation * (*itev)->GetWeight();
      }
      estimator[itype] /= sumOfWeights[itype];
   }

   // return value is sum over normalised signal and background contributions
   return estimator[0] + estimator[1];
}

//_______________________________________________________________________
Double_t TMVA::MethodFDA::InterpretFormula( const Event& event, std::vector<Double_t>& pars )
{
   // formula interpretation
   for (UInt_t ipar=0; ipar<pars.size(); ipar++) fFormula->SetParameter( ipar, pars[ipar] );
   for (Int_t ivar=0;  ivar<GetNvar();   ivar++) fFormula->SetParameter( fNPars+ivar, event.GetVal(ivar) );

   return fFormula->Eval( 0 );
}

//_______________________________________________________________________
Double_t TMVA::MethodFDA::GetMvaValue()
{
   // returns MVA value for given event

   return InterpretFormula( GetEvent(), fBestPars );
}

//_______________________________________________________________________
void  TMVA::MethodFDA::WriteWeightsToStream( ostream& o ) const
{  
   // write the weight from the training to a file (stream)

   // save fitted function parameters
   o << fNPars << endl;
   for (Int_t ipar=0; ipar<fNPars; ipar++) o << fBestPars[ipar] << endl;
}
  
//_______________________________________________________________________
void  TMVA::MethodFDA::ReadWeightsFromStream( istream& istr )
{
   // read back the training results from a file (stream)

   // retrieve best function parameters
   istr >> fNPars;
   fBestPars.clear();
   fBestPars.resize( fNPars );
   for (Int_t ipar=0; ipar<fNPars; ipar++) istr >> fBestPars[ipar];
}

//_______________________________________________________________________
void TMVA::MethodFDA::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write FDA-specific classifier response
   fout << "   double              fParameter[" << fNPars << "];" << endl;
   fout << "};" << endl;
   fout << "" << endl;
   fout << "inline void " << className << "::Initialize() " << endl;
   fout << "{" << endl;
   for (Int_t ipar=0; ipar<fNPars; ipar++) {
      fout << "   fParameter[" << ipar << "] = " << fBestPars[ipar] << ";" << endl;
   }
   fout << "}" << endl;
   fout << endl;
   fout << "inline double " << className << "::GetMvaValue__( const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   // interpret the formula" << endl;

   // replace parameters
   TString str = fFormulaStringT;
   for (Int_t ipar=0; ipar<fNPars; ipar++) {
      str.ReplaceAll( Form("[%i]", ipar), Form("fParameter[%i]", ipar) );
   }
   
   // replace input variables
   for (Int_t ivar=0; ivar<GetNvar(); ivar++) {
      str.ReplaceAll( Form("[%i]", ivar+fNPars), Form("inputValues[%i]", ivar) );
   }

   fout << "   double retval = " << str << ";" << endl;
   fout << endl;
   fout << "   return retval; " << endl;
   fout << "}" << endl;
   fout << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   // nothing to clear" << endl;
   fout << "}" << endl;
}

//_______________________________________________________________________
void TMVA::MethodFDA::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The function discriminant analysis (FDA) is a classifier suitable " << Endl;
   fLogger << "to solve linear or simple nonlinear discrimination problems." << Endl; 
   fLogger << Endl;
   fLogger << "The user provides the desired function with adjustable parameters" << Endl;
   fLogger << "via the configuration option string, and FDA fits the parameters to" << Endl;
   fLogger << "it, requiring the signal (background) function value to be as close" << Endl;
   fLogger << "as possible to 1 (0). Its advantage over the more involved and" << Endl;
   fLogger << "automatic nonlinear discriminators is the simplicity and transparency " << Endl;
   fLogger << "of the discrimination expression. A shortcoming is that FDA will" << Endl;
   fLogger << "underperform for involved problems with complicated, phase space" << Endl;
   fLogger << "dependent nonlinear correlations." << Endl;
   fLogger << Endl;
   fLogger << "Please consult the users manual for the format of the formula string" << Endl;
   fLogger << "and the allowed parameter ranges:" << Endl;
   fLogger << "http://tmva.sourceforge.net/docu/TMVAUsersGuide.pdf" << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The FDA performance depends on the complexity and fidelity of the" << Endl;
   fLogger << "user-defined discriminator function. As a general rule, it should" << Endl;
   fLogger << "be able to reproduce the discrimination power of any linear" << Endl;
   fLogger << "discriminant analysis. To reach into the nonlinear domain, it is" << Endl;
   fLogger << "useful to inspect the correlation profiles of the input variables," << Endl;
   fLogger << "and add quadratic and higher polynomial terms between variables as" << Endl;
   fLogger << "necessary. Comparison with more involved nonlinear classifiers can" << Endl;
   fLogger << "be used as a guide." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Depending on the function used, the choice of \"FitMethod\" is" << Endl;
   fLogger << "crucial for getting valuable solutions with FDA. As a guideline it" << Endl;
   fLogger << "is recommended to start with \"FitMethod=MINUIT\". When more complex" << Endl;
   fLogger << "functions are used where MINUIT does not converge to reasonable" << Endl;
   fLogger << "results, the user should switch to non-gradient FitMethods such" << Endl;
   fLogger << "as GeneticAlgorithm (GA) or Monte Carlo (MC). It might prove to be" << Endl;
   fLogger << "useful to combine GA (or MC) with MINUIT by setting the option" << Endl;
   fLogger << "\"Converger=MINUIT\". GA (MC) will then set the starting parameters" << Endl;
   fLogger << "for MINUIT such that the basic quality of GA (MC) of finding global" << Endl;
   fLogger << "minima is combined with the efficacy of MINUIT of finding local" << Endl;
   fLogger << "minima." << Endl;
}
