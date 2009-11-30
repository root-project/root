// @(#)root/tmva $Id$    
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSeedDistance                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
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
/* Begin_Html
   This method is experimental only. It does not show any improvements
   compared to any of the traditional methods.
End_Html */
//_______________________________________________________________________

#include <sstream>

#include "TList.h"
#include "TFormula.h"
#include "TString.h"
#include "TObjString.h"
#include "TRandom3.h"
#include "TMath.h"

#include "TMVA/ClassifierFactory.h"
#include "TMVA/MethodSeedDistance.h"
#include "TMVA/Tools.h"
#include "TMVA/Interval.h"
#include "TMVA/Timer.h"
#include "TMVA/GeneticFitter.h"
#include "TMVA/SimulatedAnnealingFitter.h"
#include "TMVA/MinuitFitter.h"
#include "TMVA/MCFitter.h"
#include "TMVA/MetricEuler.h"
#include "TMVA/MetricManhattan.h"
#include "TMVA/SeedDistance.h"

REGISTER_METHOD(SeedDistance)

ClassImp(TMVA::MethodSeedDistance)

//_______________________________________________________________________
TMVA::MethodSeedDistance::MethodSeedDistance( const TString& jobName,
                                              const TString& methodTitle,
                                              DataSetInfo& theData, 
                                              const TString& theOption,
                                              TDirectory* theTargetDir ) :
   TMVA::MethodBase( jobName, Types::kSeedDistance, methodTitle, theData, theOption, theTargetDir ), 
   IFitterTarget(),
   fSeedRangeStringP(""),
   fSeedRangeStringT(""),
   fScalingFactor(1),
   fMetric(0),
   fSeedDistance(0),
   fSeeds(),
   fMetricPars(),
   fPars(),
   fDataSeeds(0),
   fBackSeeds(0),
   fMetricType(""),
   fPow2Estimator(kTRUE),
   fNPars(0),
   fParRange(),
   fFitMethod(""),
   fConverger(""),
   fFitter(0),
   fIntermediateFitter(0),
   fEventsSig(),
   fEventsBkg(),
   fSumOfWeightsSig(0),
   fSumOfWeightsBkg(0)
{
   // standard constructor
}

//_______________________________________________________________________
TMVA::MethodSeedDistance::MethodSeedDistance( DataSetInfo& theData, 
                                              const TString& theWeightFile,  
                                              TDirectory* theTargetDir ) :
   TMVA::MethodBase( Types::kSeedDistance, theData, theWeightFile, theTargetDir ),
   IFitterTarget(),
   fSeedRangeStringP(""),
   fSeedRangeStringT(""),
   fScalingFactor(1),
   fMetric(0),
   fSeedDistance(0),
   fSeeds(),
   fMetricPars(),
   fPars(),
   fDataSeeds(0),
   fBackSeeds(0),
   fMetricType(""),
   fPow2Estimator(kTRUE),
   fNPars(0),
   fParRange(),
   fFitMethod(""),
   fConverger(""),
   fFitter(0),
   fIntermediateFitter(0),
   fEventsSig(),
   fEventsBkg(),
   fSumOfWeightsSig(0),
   fSumOfWeightsBkg(0)
{
   // constructor from weight file
}

//_______________________________________________________________________
Bool_t TMVA::MethodSeedDistance::HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ )
{
   // SeedDistance can handle classification with 2 classes
   if( type == Types::kClassification && numberClasses == 2 ) return kTRUE;
   return kFALSE;
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::Init( void )
{
   // default initialisation
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::DeclareOptions() 
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
   DeclareOptionRef( fSeedRangeStringP = "", "SeedRanges", "Range intervals confining the variables for the seeds" );
   DeclareOptionRef( fDataSeeds = 1, "DataSeeds", "Number of used data seeds" );
   DeclareOptionRef( fBackSeeds = 1, "BackSeeds", "Number of used background seeds" );
   DeclareOptionRef( fMetricType = "Euler", "Metric", "Type of metric used (Euler, Manhattan)" );
   AddPreDefVal(TString("Euler"));
   AddPreDefVal(TString("Manhattan"));

   DeclareOptionRef( fPow2Estimator = false, "Pow2Estimator", "Squared deviation from desired result (true) or number of correct classifications (false) as estimator" );
   DeclareOptionRef( fScalingFactor = true, "Scaling", "Produces an additional free parameter for a Seed which scales the calculated distance" );

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
void TMVA::MethodSeedDistance::ProcessOptions() 
{
   // the option string is decoded, for availabel options see "DeclareOptions"
   // clean up first
   ClearAll();

   if (IgnoreEventsWithNegWeightsInTraining()) {
      Log() << kFATAL << "Mechanism to ignore events with negative weights in training not yet available for method: "
            << GetMethodTypeName() 
            << " --> please remove \"IgnoreNegWeightsInTraining\" option from booking string."
            << Endl;
   }

   // process transient strings
   //   fFormulaStringT  = fFormulaStringP;
   fSeedRangeStringT = fSeedRangeStringP;

   // interpret parameter string   
   fSeedRangeStringT.ReplaceAll( " ", "" );
   fNPars = fSeedRangeStringT.CountChar( ')' );
   //   fNPars = 4;

   //   Log() << kINFO << "rangestring " << fSeedRangeStringT << Endl;
   //   Log() << kINFO << "rangestring number ) " << fNPars << Endl;

   TList* parList = gTools().ParseFormatLine( fSeedRangeStringT, ";" );
   //   if (parList->GetSize()*2 != fNPars) {
   //      Log() << kFATAL << "<ProcessOptions> Mismatch in parameter string: " 
   //              << "the number of parameters: " << fNPars << " != ranges defined: " 
   //              << parList->GetSize() << "; the format of the \"SeedRanges\" string "
   //              << "must be: \"(-1.2,3.4);(-2.3,4.55);...\", "
   //              << "where the numbers in \"(a,b)\" correspond to the a=min, b=max parameter ranges; "
   //              << "each parameter defined in the function string must have a corresponding rang."
   //              << Endl;
   //   }

   fParRange.resize( fNPars );
   for (Int_t ipar=0; ipar<fNPars; ipar++) fParRange[ipar] = 0;

   for (Int_t ipar=0; ipar<fNPars; ipar++) {
      // parse (a,b)
      TString str = ((TObjString*)parList->At(ipar))->GetString();
      Ssiz_t istr = str.First( ',' );
      TString pminS(str(1,istr-1));
      TString pmaxS(str(istr+1,str.Length()-2-istr));
      std::stringstream st;
      st.precision( 16 );
      st << std::scientific << pminS.Data();
      Float_t pmin;
      st >> pmin;
      st << std::scientific << pmaxS.Data();
      Float_t pmax;
      st >> pmax;

      // sanity check
      if (pmin > pmax) Log() << kFATAL << "<ProcessOptions> max > min in interval for parameter: [" 
                               << ipar << "] : [" << pmin  << ", " << pmax << "] " << Endl;

      fParRange[ipar] = new Interval( pmin, pmax );
   }

   delete parList;

   if( fScalingFactor ){
      fParRange.push_back( new Interval( 0.0, 1.0 ) );
   }
   
   
   for( Int_t i = 0; i< fDataSeeds+fBackSeeds; i++ ){
      fSeeds.push_back( std::vector< Double_t >() );
      for(std::vector<TMVA::Interval*>::const_iterator parIt = fParRange.begin(); parIt != fParRange.end(); parIt++) {
         fSeeds[i].push_back( (*parIt)->GetMean() );
      }
   }

   std::vector<Interval*>::iterator maxpos;
   for( Int_t i = 1; i< fDataSeeds+fBackSeeds; i++ ){
      maxpos = fParRange.begin();
      for( Int_t j=0; j< fNPars; j++ ){
         maxpos++;
      }
      if( fScalingFactor ){
         maxpos++;
      }
      fParRange.insert( fParRange.end(), fParRange.begin(), maxpos );
   }

   for( Int_t i = 0; i < fNPars; i++) {
      fMetricPars.push_back( 0.5 );
      fParRange.push_back( new Interval( 0.0, 1.0 ) );
   }
   
   if( fMetricType == "Euler" )     fMetric = new MetricEuler();
   if( fMetricType == "Manhattan" ) fMetric = new MetricManhattan();

   fMetric->SetParameters( &fMetricPars );
   fSeedDistance = new SeedDistance( *fMetric, fSeeds );

   fIntermediateFitter = (TMVA::IFitterTarget*)this;
   if (fConverger == "MINUIT")
      fIntermediateFitter = new TMVA::MinuitFitter( *this, Form("%s_MINUIT", GetName()), fParRange, GetOptions() );
   if      (fFitMethod == "MC")     fFitter = new TMVA::MCFitter                ( *fIntermediateFitter, Form("%sFitter_MC", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "GA")     fFitter = new TMVA::GeneticFitter           ( *fIntermediateFitter, Form("%sFitter_GA", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "SA")     fFitter = new TMVA::SimulatedAnnealingFitter( *fIntermediateFitter, Form("%sFitter_SA", GetName()), fParRange, GetOptions() );
   else if (fFitMethod == "MINUIT") fFitter = new TMVA::MinuitFitter            ( *fIntermediateFitter, Form("%sFitter_MINUIT", GetName()), fParRange, GetOptions() );
   else {
      Log() << kFATAL << "<Train> Do not understand fit method: " << fFitMethod << Endl;
   }
   
   fFitter->CheckForUnusedOptions();
   
}

//_______________________________________________________________________
TMVA::MethodSeedDistance::~MethodSeedDistance( void )
{
   // destructor
   ClearAll();
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::ClearAll( void )
{
   // reset all parameters of the method
   std::map< Interval*, Int_t > delmap;
    
   for (UInt_t ipar=0; ipar<fParRange.size(); ipar++) {
      delmap[fParRange[ipar]] = ipar;
      fParRange[ipar] = 0;
   }
   for( std::map< Interval*, Int_t >::iterator it = delmap.begin(); it != delmap.end(); it++ ){
      delete it->first;
   }
   fParRange.clear(); 

   fMetricPars.clear();

   fPars.clear();
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::Train( void )
{
   // FDA training 

   // cache training events
   fSumOfWeightsSig = 0;
   fSumOfWeightsBkg = 0;

   for (Int_t ievt=0; ievt<Data()->GetNEvents(); ievt++) {

      const Event*  ev = Data()->GetEvent(ievt);
      Float_t w  = ev->GetWeight();

      if (ev->IsSignal()) { fEventsSig.push_back( ev ); fSumOfWeightsSig += w; }
      else                { fEventsBkg.push_back( ev ); fSumOfWeightsBkg += w; }
   }

   // sanity check
   if (fSumOfWeightsSig <= 0 || fSumOfWeightsBkg <= 0) {
      Log() << kFATAL << "<Train> Troubles in sum of weights: " 
              << fSumOfWeightsSig << " (S) : " << fSumOfWeightsBkg << " (B)" << Endl;
   }

   // starting values (not used by all fitters)
   fPars.clear();

   MakeListFromStructure( fPars, fSeeds, fMetricPars );

   // execute the fit
//   Double_t estimator = fFitter->Run( fPars );
   Double_t estimator = fFitter->Run( fPars );

   MakeStructureFromList( fPars, fSeeds, fMetricPars );

   // print results
   PrintResults( fFitMethod, fPars, estimator );

   // free cache 
   std::vector<const Event*>::const_iterator itev;
   for (itev = fEventsSig.begin(); itev != fEventsSig.end(); itev++) delete *itev;
   for (itev = fEventsBkg.begin(); itev != fEventsBkg.end(); itev++) delete *itev;

   fEventsSig.clear();
   fEventsBkg.clear();

   if (fConverger == "MINUIT") delete fIntermediateFitter;
   delete fFitter; fFitter = 0;
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::PrintResults( const TString& fitter, std::vector<Double_t>& , const Double_t estimator ) const
{
   //MakeStructureFromList( pars, fSeeds, fMetricPars );

   // display fit parameters
   // check maximum length of variable name
   Log() << kINFO;
   Log() << "Results for distance to seed method using fitter: \"" << fitter << Endl;
   Log() << "Value of estimator at minimum: " << estimator << Endl;

   // print seeds
   Log() << kINFO << "Number of Seeds: " << fSeeds.size() << Endl;
   for( Int_t i = 0; i< (Int_t)fSeeds.size(); i++ ){
      if( i < fDataSeeds ){
         Log() << kINFO << "Seed " << i << " -- DATA" << Endl;
      }else{
         Log() << kINFO << "Seed " << i << " -- BACKGROUND" << Endl;
      }
      for( Int_t j = 0; j< (Int_t)fSeeds[i].size(); j++ ){
         if( fScalingFactor && j >= (Int_t)fSeeds[i].size()-1 ){
            Log() << kINFO << "   scaling factor " << ": " << fSeeds[i][j] << Endl;
         }else{
            Log() << kINFO << "   dimension " << j << ": " << fSeeds[i][j] << Endl;
         }
      }
   }
   
   // print metric parameters
   Log() << kINFO << Endl;
   Log() << kINFO << "Metric: " << fMetricType << " with " << fMetricPars.size() << " parameters" << Endl;
   for( Int_t i = 0; i< (Int_t)fMetricPars.size(); i++ ){
      Log() << kINFO << "   par " << i << ": " << fMetricPars[i] << Endl;
   }

}

//_______________________________________________________________________
Double_t TMVA::MethodSeedDistance::EstimatorFunction( std::vector<Double_t>& pars )
{
   // compute estimator for given parameter set (to be minimised)

   MakeStructureFromList( pars, fSeeds, fMetricPars );
   std::vector< Double_t > point;
   Double_t looksLike = 0.0;
   
   // species-specific stuff
   const std::vector<const Event*>* eventVecs[] = { &fEventsSig, &fEventsBkg };
   const Double_t sumOfWeights[]                = { fSumOfWeightsSig, fSumOfWeightsBkg };
   const Double_t desiredVal[]                  = { 1, 0 };
   Double_t estimator[]                         = { 0, 0 };
   std::vector<const Event*>::const_iterator itev;

   Double_t distData;
   Double_t distBack;
   Double_t deviation;
   
   // loop over species
   for (Int_t itype=0; itype<2; itype++) {

      // loop over specific events
      for (itev = eventVecs[itype]->begin(); itev != eventVecs[itype]->end(); itev++) {
         point.clear();
         for (UInt_t ivar=0;  ivar<GetNvar();   ivar++) point.push_back( (**itev).GetValue(ivar) );

         std::vector< Double_t >& distances = fSeedDistance->GetDistances( point );
         
         distData = distances[0];
         for( Int_t i=1; i< fDataSeeds; i++ ){
            distData = TMath::Min( distData, distances[i] );
         }
         distBack = distances[fDataSeeds];
         for( Int_t i=fDataSeeds; i< fDataSeeds+fBackSeeds; i++ ){
            distBack = TMath::Min( distBack, distances[i] );
         }
         
         if( !fPow2Estimator ){
            if( distData < distBack ){ 
               deviation = 1-desiredVal[itype];
            }else{
               deviation = desiredVal[itype];
            }
         }else{
            looksLike = distBack/(distData+distBack);
            deviation = (looksLike - desiredVal[itype])*(looksLike - desiredVal[itype]);
         }

         estimator[itype] += deviation * (*itev)->GetWeight();
      }
      estimator[itype] /= sumOfWeights[itype];
   }

   // return value is sum over normalised signal and background contributions
   return estimator[0] + estimator[1];
}

//_______________________________________________________________________
Double_t TMVA::MethodSeedDistance::GetMvaValue( Double_t* err )
{
   // returns MVA value for given event
   std::vector< Double_t > point;
   const Event* ev = GetEvent();

   // cannot determine error
   if (err != 0) *err = -1;

   Double_t distData;
   Double_t distBack;

   point.clear();
   for (UInt_t ivar=0;  ivar<GetNvar();   ivar++) point.push_back( ev->GetValue(ivar) );

   std::vector< Double_t >& distances = fSeedDistance->GetDistances( point );

   distData = distances[0];
   for( Int_t i=1; i< fDataSeeds; i++ ){
      distData = TMath::Min( distData, distances[i] );
   }
   distBack = distances[fDataSeeds];
   for( Int_t i=fDataSeeds; i< fDataSeeds+fBackSeeds; i++ ){
      distBack = TMath::Min( distBack, distances[i] );
   }
   

   if( distData+distBack == 0 ){
      Log() << kINFO << "backgroundseed=dataseed";
      return 0.0;
   }
   Double_t looksLike = distBack/(distData+distBack);

   return looksLike;
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::AddWeightsXMLTo( void* /*parent*/ ) const 
{
   Log() << kFATAL << "Please implement writing of weights as XML" << Endl;
}
 
//_______________________________________________________________________
void  TMVA::MethodSeedDistance::ReadWeightsFromStream( istream& istr )
{
   // read back the training results from a file (stream)

   Int_t size;
   Double_t val;
   istr >> size;
//   Log() << kINFO << size << " ";
   fSeeds.clear();
   for( Int_t i = 0; i<size; i++ ){
      fSeeds.push_back( std::vector< Double_t >() );
      Int_t subSize;
      istr >> subSize;
//      Log() << kINFO << subSize << " ";
      for( Int_t j = 0; j<subSize; j++ ){
         istr >> val;
//         Log() << kINFO << val << " ";
         fSeeds[i].push_back( val );
      }
   }

   istr >> fDataSeeds;
   istr >> fBackSeeds;
   istr >> fScalingFactor;

   istr >> fMetricType;
   istr >> size;
//   Log() << kINFO << size << " ";
   fMetricPars.clear();
   for( Int_t i = 0; i<size; i++ ){
      istr >> val;
//      Log() << kINFO << val << " ";
      fMetricPars.push_back( val );
   }

   if( fMetricType == "Euler" ) fMetric = new MetricEuler();
   else if( fMetricType == "Manhattan" ) fMetric = new MetricManhattan();
   else{
      Log() << kFATAL << "unknown metric" << Endl;
   }
   fMetric->SetParameters( &fMetricPars );
   fSeedDistance = new SeedDistance( *fMetric, fSeeds );
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::MakeClassSpecific( std::ostream& fout, const TString& /*className*/ ) const
{
   //    fout << "Bool_t                 fScalingFactor = " << fScalingFactor << ";" << endl;
   //    fout << "IMetric*               fMetric = new Metric" << fMetricType << "();" << endl;
   //    fout << "SeedDistance*          fSeedDistance;" << endl;
   //    fout << "std::vector< std::vector< Double_t > > fSeeds;" << endl;
   //    fout << "std::vector<Double_t>  fMetricPars;" << endl;
   //    fout << "Int_t                  fDataSeeds = " << fDataSeeds << ";" << endl;
   //    fout << "Int_t                  fBackSeeds = " << fBackSeeds << ";" << endl;
   //    fout << "TString                fMetricType = \"" << fMetricType << "\";" << endl;
   //    fout << "Int_t                  fNPars = " << fNPars << ";" << endl;
   fout << "not implemented for class" << std::endl;
}


//_______________________________________________________________________
void TMVA::MethodSeedDistance::MakeListFromStructure( std::vector<Double_t>& linear, 
                                  std::vector< std::vector< Double_t > >& seeds,
                                  std::vector<Double_t>& metricParams )
{
   // linear: / /-seed1-//-seed-2//...//-seed n-/ /metricParams/ /
   linear.clear();
   for( std::vector< std::vector< Double_t > >::iterator itSeed = seeds.begin(); itSeed != seeds.end(); itSeed++ ){
      linear.insert( linear.end(), (*itSeed).begin(), (*itSeed).end() );
   }
   linear.insert( linear.end(), metricParams.begin(), metricParams.end() );
}

//_______________________________________________________________________
void TMVA::MethodSeedDistance::MakeStructureFromList( std::vector<Double_t>& linear, 
                                  std::vector< std::vector< Double_t > >& seeds,
                                  std::vector<Double_t>& metricParams )
{
   // makes the structure from the list
   std::vector<Double_t>::iterator loc = linear.begin();
   for( std::vector< std::vector<Double_t> >::iterator itSeed = seeds.begin(); itSeed != seeds.end(); itSeed++ ){
      for( std::vector<Double_t>::iterator it = (*itSeed).begin(); it != (*itSeed).end(); it++ ){
         (*it) = (*loc);
         loc++;
      }
   }
   for( std::vector<Double_t>::iterator it = metricParams.begin(); it != metricParams.end(); it++ ){
      (*it) = (*loc);
      loc++;
   }
}


//_______________________________________________________________________
void TMVA::MethodSeedDistance::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Short description:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance optimisation:" << gTools().Color("reset") << Endl;
   Log() << Endl;
   Log() << Endl;
   Log() << gTools().Color("bold") << "--- Performance tuning via configuration options:" << gTools().Color("reset") << Endl;
   Log() << Endl;
}
