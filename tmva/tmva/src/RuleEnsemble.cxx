// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RuleEnsemble                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class generating an ensemble of rules                                   *
 *      Input:  a forest of decision trees                                        *
 *      Output: an ensemble of rules                                              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, GER  *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::RuleEnsemble
\ingroup TMVA

*/
#include "TMVA/RuleEnsemble.h"

#include "TMVA/DataSetInfo.h"
#include "TMVA/DecisionTree.h"
#include "TMVA/Event.h"
#include "TMVA/MethodBase.h"
#include "TMVA/MethodRuleFit.h"
#include "TMVA/MsgLogger.h"
#include "TMVA/RuleFit.h"
#include "TMVA/Tools.h"
#include "TMVA/Types.h"

#include "TRandom3.h"
#include "TH1F.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <list>

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::RuleEnsemble::RuleEnsemble( RuleFit *rf )
   : fLearningModel   ( kFull )
   , fImportanceCut   ( 0 )
   , fLinQuantile     ( 0.025 ) // default quantile for killing outliers in linear terms
   , fOffset          ( 0 )
   , fAverageSupport  ( 0.8 )
   , fAverageRuleSigma( 0.4 )  // default value - used if only linear model is chosen
   , fRuleFSig        ( 0 )
   , fRuleNCave       ( 0 )
   , fRuleNCsig       ( 0 )
   , fRuleMinDist     ( 1e-3 ) // closest allowed 'distance' between two rules
   , fNRulesGenerated ( 0 )
   , fEvent           ( 0 )
   , fEventCacheOK    ( true )
   , fRuleMapOK       ( true )
   , fRuleMapInd0     ( 0 )
   , fRuleMapInd1     ( 0 )
   , fRuleMapEvents   ( 0 )
   , fLogger( new MsgLogger("RuleFit") )
{
   Initialize( rf );
}

////////////////////////////////////////////////////////////////////////////////
/// copy constructor

TMVA::RuleEnsemble::RuleEnsemble( const RuleEnsemble& other )
   : fAverageSupport   ( 1 )
   , fEvent(0)
   , fRuleMapEvents(0)
   , fRuleFit(0)
   , fLogger( new MsgLogger("RuleFit") )
{
   Copy( other );
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::RuleEnsemble::RuleEnsemble()
   : fLearningModel     ( kFull )
   , fImportanceCut   ( 0 )
   , fLinQuantile     ( 0.025 ) // default quantile for killing outliers in linear terms
   , fOffset          ( 0 )
   , fImportanceRef   ( 1.0 )
   , fAverageSupport  ( 0.8 )
   , fAverageRuleSigma( 0.4 )  // default value - used if only linear model is chosen
   , fRuleFSig        ( 0 )
   , fRuleNCave       ( 0 )
   , fRuleNCsig       ( 0 )
   , fRuleMinDist     ( 1e-3 ) // closest allowed 'distance' between two rules
   , fNRulesGenerated ( 0 )
   , fEvent           ( 0 )
   , fEventCacheOK    ( true )
   , fRuleMapOK       ( true )
   , fRuleMapInd0     ( 0 )
   , fRuleMapInd1     ( 0 )
   , fRuleMapEvents   ( 0 )
   , fRuleFit         ( 0 )
   , fLogger( new MsgLogger("RuleFit") )
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::RuleEnsemble::~RuleEnsemble()
{
   for ( std::vector<Rule *>::iterator itrRule = fRules.begin(); itrRule != fRules.end(); ++itrRule ) {
      delete *itrRule;
   }
   // NOTE: Should not delete the histos fLinPDFB/S since they are delete elsewhere
   delete fLogger;
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes all member variables with default values

void TMVA::RuleEnsemble::Initialize( const RuleFit *rf )
{
   SetAverageRuleSigma(0.4); // default value - used if only linear model is chosen
   fRuleFit = rf;
   UInt_t nvars =  GetMethodBase()->GetNvar();
   fVarImportance.clear();
   fLinPDFB.clear();
   fLinPDFS.clear();
   //
   fVarImportance.resize( nvars,0.0 );
   fLinPDFB.resize( nvars,0 );
   fLinPDFS.resize( nvars,0 );
   fImportanceRef = 1.0;
   for (UInt_t i=0; i<nvars; i++) { // a priori all linear terms are equally valid
      fLinTermOK.push_back(kTRUE);
   }
}

////////////////////////////////////////////////////////////////////////////////

void TMVA::RuleEnsemble::SetMsgType( EMsgType t ) {
   fLogger->SetMinType(t);
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get a pointer to the original MethodRuleFit.

const TMVA::MethodRuleFit*  TMVA::RuleEnsemble::GetMethodRuleFit() const
{
   return ( fRuleFit==0 ? 0:fRuleFit->GetMethodRuleFit());
}

////////////////////////////////////////////////////////////////////////////////
///
/// Get a pointer to the original MethodRuleFit.

const TMVA::MethodBase*  TMVA::RuleEnsemble::GetMethodBase() const
{
   return ( fRuleFit==0 ? 0:fRuleFit->GetMethodBase());
}

////////////////////////////////////////////////////////////////////////////////
/// create model

void TMVA::RuleEnsemble::MakeModel()
{
   MakeRules( fRuleFit->GetForest() );

   MakeLinearTerms();

   MakeRuleMap();

   CalcRuleSupport();

   RuleStatistics();

   PrintRuleGen();
}

////////////////////////////////////////////////////////////////////////////////
///
/// Calculates sqrt(Sum(a_i^2)), i=1..N (NOTE do not include a0)

Double_t TMVA::RuleEnsemble::CoefficientRadius()
{
   Int_t ncoeffs = fRules.size();
   if (ncoeffs<1) return 0;
   //
   Double_t sum2=0;
   Double_t val;
   for (Int_t i=0; i<ncoeffs; i++) {
      val = fRules[i]->GetCoefficient();
      sum2 += val*val;
   }
   return sum2;
}

////////////////////////////////////////////////////////////////////////////////
/// reset all rule coefficients

void TMVA::RuleEnsemble::ResetCoefficients()
{
   fOffset = 0.0;
   UInt_t nrules = fRules.size();
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(0.0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set all rule coefficients

void TMVA::RuleEnsemble::SetCoefficients( const std::vector< Double_t > & v )
{
   UInt_t nrules = fRules.size();
   if (v.size()!=nrules) {
      Log() << kFATAL << "<SetCoefficients> - BUG TRAP - input vector wrong size! It is = " << v.size()
            << " when it should be = " << nrules << Endl;
   }
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(v[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Retrieve all rule coefficients

void TMVA::RuleEnsemble::GetCoefficients( std::vector< Double_t > & v )
{
   UInt_t nrules = fRules.size();
   v.resize(nrules);
   if (nrules==0) return;
   //
   for (UInt_t i=0; i<nrules; i++) {
      v[i] = (fRules[i]->GetCoefficient());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// get list of training events from the rule fitter

const std::vector<const TMVA::Event*>* TMVA::RuleEnsemble::GetTrainingEvents()  const
{
   return &(fRuleFit->GetTrainingEvents());
}

////////////////////////////////////////////////////////////////////////////////
/// get the training event from the rule fitter

const TMVA::Event * TMVA::RuleEnsemble::GetTrainingEvent(UInt_t i) const
{
   return fRuleFit->GetTrainingEvent(i);
}

////////////////////////////////////////////////////////////////////////////////
/// remove rules that behave similar

void TMVA::RuleEnsemble::RemoveSimilarRules()
{
   Log() << kVERBOSE << "Removing similar rules; distance = " << fRuleMinDist << Endl;

   UInt_t nrulesIn = fRules.size();
   TMVA::Rule *first, *second;
   std::vector< Char_t > removeMe( nrulesIn,false );  // <--- stores boolean

   Int_t nrem = 0;
   Int_t remind=-1;
   Double_t r;

   for (UInt_t i=0; i<nrulesIn; i++) {
      if (!removeMe[i]) {
         first = fRules[i];
         for (UInt_t k=i+1; k<nrulesIn; k++) {
            if (!removeMe[k]) {
               second = fRules[k];
               Bool_t equal = first->Equal(*second,kTRUE,fRuleMinDist);
               if (equal) {
                  r = gRandom->Rndm();
                  remind = (r>0.5 ? k:i); // randomly select rule
               }
               else {
                  remind = -1;
               }

               if (remind>-1) {
                  if (!removeMe[remind]) {
                     removeMe[remind] = true;
                     nrem++;
                  }
               }
            }
         }
      }
   }
   UInt_t ind = 0;
   Rule *theRule;
   for (UInt_t i=0; i<nrulesIn; i++) {
      if (removeMe[i]) {
         theRule = fRules[ind];
         fRules.erase( fRules.begin() + ind );
         delete theRule;
         ind--;
      }
      ind++;
   }
   UInt_t nrulesOut = fRules.size();
   Log() << kVERBOSE << "Removed " << nrulesIn - nrulesOut << " out of " << nrulesIn << " rules" << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// cleanup rules

void TMVA::RuleEnsemble::CleanupRules()
{
   UInt_t nrules   = fRules.size();
   if (nrules==0) return;
   Log() << kVERBOSE << "Removing rules with relative importance < " << fImportanceCut << Endl;
   if (fImportanceCut<=0) return;
   //
   // Mark rules to be removed
   //
   Rule *therule;
   Int_t ind=0;
   for (UInt_t i=0; i<nrules; i++) {
      if (fRules[ind]->GetRelImportance()<fImportanceCut) {
         therule = fRules[ind];
         fRules.erase( fRules.begin() + ind );
         delete therule;
         ind--;
      }
      ind++;
   }
   Log() << kINFO << "Removed " << nrules-ind << " out of a total of " << nrules
         << " rules with importance < " << fImportanceCut << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// cleanup linear model

void TMVA::RuleEnsemble::CleanupLinear()
{
   UInt_t nlin = fLinNorm.size();
   if (nlin==0) return;
   Log() << kVERBOSE << "Removing linear terms with relative importance < " << fImportanceCut << Endl;
   //
   fLinTermOK.clear();
   for (UInt_t i=0; i<nlin; i++) {
      fLinTermOK.push_back( (fLinImportance[i]/fImportanceRef > fImportanceCut) );
   }
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the support for all rules

void TMVA::RuleEnsemble::CalcRuleSupport()
{
   Log() << kVERBOSE << "Evaluating Rule support" << Endl;
   Double_t s,t,stot,ttot,ssb;
   Double_t ssig, sbkg, ssum;
   Int_t indrule=0;
   stot = 0;
   ttot = 0;
   // reset to default values
   SetAverageRuleSigma(0.4);
   const std::vector<const Event *> *events = GetTrainingEvents();
   Double_t nrules = static_cast<Double_t>(fRules.size());
   Double_t ew;
   //
   if ((nrules>0) && (events->size()>0)) {
      for ( std::vector< Rule * >::iterator itrRule=fRules.begin(); itrRule!=fRules.end(); ++itrRule ) {
         s=0.0;
         ssig=0.0;
         sbkg=0.0;
         for ( std::vector<const Event * >::const_iterator itrEvent=events->begin(); itrEvent!=events->end(); ++itrEvent ) {
            if ((*itrRule)->EvalEvent( *(*itrEvent) )) {
               ew = (*itrEvent)->GetWeight();
               s += ew;
               if (GetMethodRuleFit()->DataInfo().IsSignal(*itrEvent)) ssig += ew;
               else                         sbkg += ew;
            }
         }
         //
         s = s/fRuleFit->GetNEveEff();
         t = s*(1.0-s);
         t = (t<0 ? 0:sqrt(t));
         stot += s;
         ttot += t;
         ssum = ssig+sbkg;
         ssb = (ssum>0 ? Double_t(ssig)/Double_t(ssig+sbkg) : 0.0 );
         (*itrRule)->SetSupport(s);
         (*itrRule)->SetNorm(t);
         (*itrRule)->SetSSB( ssb );
         (*itrRule)->SetSSBNeve(Double_t(ssig+sbkg));
         indrule++;
      }
      fAverageSupport   = stot/nrules;
      fAverageRuleSigma = TMath::Sqrt(fAverageSupport*(1.0-fAverageSupport));
      Log() << kVERBOSE << "Standard deviation of support = " << fAverageRuleSigma << Endl;
      Log() << kVERBOSE << "Average rule support          = " << fAverageSupport   << Endl;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the importance of each rule

void TMVA::RuleEnsemble::CalcImportance()
{
   Double_t maxRuleImp = CalcRuleImportance();
   Double_t maxLinImp  = CalcLinImportance();
   Double_t maxImp = (maxRuleImp>maxLinImp ? maxRuleImp : maxLinImp);
   SetImportanceRef( maxImp );
}

////////////////////////////////////////////////////////////////////////////////
/// set reference importance

void TMVA::RuleEnsemble::SetImportanceRef(Double_t impref)
{
   for ( UInt_t i=0; i<fRules.size(); i++ ) {
      fRules[i]->SetImportanceRef(impref);
   }
   fImportanceRef = impref;
}
////////////////////////////////////////////////////////////////////////////////
/// calculate importance of each rule

Double_t TMVA::RuleEnsemble::CalcRuleImportance()
{
   Double_t maxImp=-1.0;
   Double_t imp;
   Int_t nrules = fRules.size();
   for ( int i=0; i<nrules; i++ ) {
      fRules[i]->CalcImportance();
      imp = fRules[i]->GetImportance();
      if (imp>maxImp) maxImp = imp;
   }
   for ( Int_t i=0; i<nrules; i++ ) {
      fRules[i]->SetImportanceRef(maxImp);
   }

   return maxImp;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the linear importance for each rule

Double_t TMVA::RuleEnsemble::CalcLinImportance()
{
   Double_t maxImp=-1.0;
   UInt_t nvars = fLinCoefficients.size();
   fLinImportance.resize(nvars,0.0);
   if (!DoLinear()) return maxImp;
   //
   // The linear importance is:
   // I = |b_x|*sigma(x)
   // Note that the coefficients in fLinCoefficients are for the normalized x
   // => b'_x * x' = b'_x * sigma(r)*x/sigma(x)
   // => b_x = b'_x*sigma(r)/sigma(x)
   // => I = |b'_x|*sigma(r)
   //
   Double_t imp;
   for ( UInt_t i=0; i<nvars; i++ ) {
      imp = fAverageRuleSigma*TMath::Abs(fLinCoefficients[i]);
      fLinImportance[i] = imp;
      if (imp>maxImp) maxImp = imp;
   }
   return maxImp;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculates variable importance using eq (35) in RuleFit paper by Friedman et.al

void TMVA::RuleEnsemble::CalcVarImportance()
{
   Log() << kVERBOSE << "Compute variable importance" << Endl;
   Double_t rimp;
   UInt_t nrules = fRules.size();
   if (GetMethodBase()==0) Log() << kFATAL << "RuleEnsemble::CalcVarImportance() - should not be here!" << Endl;
   UInt_t nvars  = GetMethodBase()->GetNvar();
   UInt_t nvarsUsed;
   Double_t rimpN;
   fVarImportance.resize(nvars,0);
   // rules
   if (DoRules()) {
      for ( UInt_t ind=0; ind<nrules; ind++ ) {
         rimp = fRules[ind]->GetImportance();
         nvarsUsed = fRules[ind]->GetNumVarsUsed();
         if (nvarsUsed<1)
            Log() << kFATAL << "<CalcVarImportance> Variables for importance calc!!!??? A BUG!" << Endl;
         rimpN = (nvarsUsed > 0 ? rimp/nvarsUsed:0.0);
         for ( UInt_t iv=0; iv<nvars; iv++ ) {
            if (fRules[ind]->ContainsVariable(iv)) {
               fVarImportance[iv] += rimpN;
            }
         }
      }
   }
   // linear terms
   if (DoLinear()) {
      for ( UInt_t iv=0; iv<fLinTermOK.size(); iv++ ) {
         if (fLinTermOK[iv]) fVarImportance[iv] += fLinImportance[iv];
      }
   }
   //
   // Make variable importance relative the strongest variable
   //
   Double_t maximp = 0.0;
   for ( UInt_t iv=0; iv<nvars; iv++ ) {
      if ( fVarImportance[iv] > maximp ) maximp = fVarImportance[iv];
   }
   if (maximp>0) {
      for ( UInt_t iv=0; iv<nvars; iv++ ) {
         fVarImportance[iv] *= 1.0/maximp;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set rules
///
/// first clear all

void TMVA::RuleEnsemble::SetRules( const std::vector<Rule *> & rules )
{
   DeleteRules();
   //
   fRules.resize(rules.size());
   for (UInt_t i=0; i<fRules.size(); i++) {
      fRules[i] = rules[i];
   }
   fEventCacheOK = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Makes rules from the given decision tree.
/// First node in all rules is ALWAYS the root node.

void TMVA::RuleEnsemble::MakeRules( const std::vector< const DecisionTree *> & forest )
{
   fRules.clear();
   if (!DoRules()) return;
   //
   Int_t nrulesCheck=0;
   Int_t nrules;
   Int_t nendn;
   Double_t sumnendn=0;
   Double_t sumn2=0;
   //
   // UInt_t prevs;
   UInt_t ntrees = forest.size();
   for ( UInt_t ind=0; ind<ntrees; ind++ ) {
      // prevs = fRules.size();
      MakeRulesFromTree( forest[ind] );
      nrules = CalcNRules( forest[ind] );
      nendn = (nrules/2) + 1;
      sumnendn += nendn;
      sumn2    += nendn*nendn;
      nrulesCheck += nrules;
   }
   Double_t nmean = (ntrees>0) ? sumnendn/ntrees : 0;
   Double_t nsigm = TMath::Sqrt( gTools().ComputeVariance(sumn2,sumnendn,ntrees) );
   Double_t ndev = 2.0*(nmean-2.0-nsigm)/(nmean-2.0+nsigm);
   //
   Log() << kVERBOSE << "Average number of end nodes per tree   = " << nmean << Endl;
   if (ntrees>1) Log() << kVERBOSE << "sigma of ditto ( ~= mean-2 ?)          = "
                       << nsigm
                       << Endl;
   Log() << kVERBOSE << "Deviation from exponential model       = " << ndev      << Endl;
   Log() << kVERBOSE << "Corresponds to L (eq. 13, RuleFit ppr) = " << nmean << Endl;
   // a BUG trap
   if (nrulesCheck != static_cast<Int_t>(fRules.size())) {
      Log() << kFATAL
            << "BUG! number of generated and possible rules do not match! N(rules) =  " << fRules.size()
            << " != " << nrulesCheck << Endl;
   }
   Log() << kVERBOSE << "Number of generated rules: " << fRules.size() << Endl;

   // save initial number of rules
   fNRulesGenerated = fRules.size();

   RemoveSimilarRules();

   ResetCoefficients();

}

////////////////////////////////////////////////////////////////////////////////
/// Make the linear terms as in eq 25, ref 2
/// For this the b and (1-b) quantiles are needed

void TMVA::RuleEnsemble::MakeLinearTerms()
{
   if (!DoLinear()) return;

   const std::vector<const Event *> *events = GetTrainingEvents();
   UInt_t neve  = events->size();
   UInt_t nvars = ((*events)[0])->GetNVariables(); // Event -> GetNVariables();
   Double_t val,ew;
   typedef std::pair< Double_t, Int_t> dataType;
   typedef std::pair< Double_t, dataType > dataPoint;

   std::vector< std::vector<dataPoint> > vardata(nvars);
   std::vector< Double_t > varsum(nvars,0.0);
   std::vector< Double_t > varsum2(nvars,0.0);
   // first find stats of all variables
   // vardata[v][i].first         -> value of var <v> in event <i>
   // vardata[v][i].second.first  -> the event weight
   // vardata[v][i].second.second -> the event type
   for (UInt_t i=0; i<neve; i++) {
      ew   = ((*events)[i])->GetWeight();
      for (UInt_t v=0; v<nvars; v++) {
         val = ((*events)[i])->GetValue(v);
         vardata[v].push_back( dataPoint( val, dataType(ew,((*events)[i])->GetClass()) ) );
      }
   }
   //
   fLinDP.clear();
   fLinDM.clear();
   fLinCoefficients.clear();
   fLinNorm.clear();
   fLinDP.resize(nvars,0);
   fLinDM.resize(nvars,0);
   fLinCoefficients.resize(nvars,0);
   fLinNorm.resize(nvars,0);

   Double_t averageWeight = neve ? fRuleFit->GetNEveEff()/static_cast<Double_t>(neve) : 0;
   // sort and find limits
   Double_t stdl;

   // find normalisation given in ref 2 after eq 26
   Double_t lx;
   Double_t nquant;
   Double_t neff;
   UInt_t   indquantM;
   UInt_t   indquantP;

   MethodBase *fMB=const_cast<MethodBase *>(fRuleFit->GetMethodBase());

   for (UInt_t v=0; v<nvars; v++) {
      varsum[v] = 0;
      varsum2[v] = 0;
      //
      std::sort( vardata[v].begin(),vardata[v].end() );
      nquant = fLinQuantile*fRuleFit->GetNEveEff(); // quantile = 0.025
      neff=0;
      UInt_t ie=0;
      // first scan for lower quantile (including weights)
      while ( (ie<neve) && (neff<nquant) ) {
         neff += vardata[v][ie].second.first;
         ie++;
      }
      indquantM = (ie==0 ? 0:ie-1);
      // now for upper quantile
      ie = neve;
      neff=0;
      while ( (ie>0) && (neff<nquant) ) {
         ie--;
         neff += vardata[v][ie].second.first;
      }
      indquantP = (ie==neve ? ie=neve-1:ie);
      //
      fLinDM[v] = vardata[v][indquantM].first; // delta-
      fLinDP[v] = vardata[v][indquantP].first; // delta+

      if(!fMB->IsSilentFile())
      {
            if (fLinPDFB[v]) delete fLinPDFB[v];
            if (fLinPDFS[v]) delete fLinPDFS[v];
            fLinPDFB[v] = new TH1F(Form("bkgvar%d",v),"bkg temphist",40,fLinDM[v],fLinDP[v]);
            fLinPDFS[v] = new TH1F(Form("sigvar%d",v),"sig temphist",40,fLinDM[v],fLinDP[v]);
            fLinPDFB[v]->Sumw2();
            fLinPDFS[v]->Sumw2();
      }
      //
      Int_t type;
      const Double_t w = 1.0/fRuleFit->GetNEveEff();
      for (ie=0; ie<neve; ie++) {
         val  = vardata[v][ie].first;
         ew   = vardata[v][ie].second.first;
         type = vardata[v][ie].second.second;
         lx = TMath::Min( fLinDP[v], TMath::Max( fLinDM[v], val ) );
         varsum[v] += ew*lx;
         varsum2[v] += ew*lx*lx;
         if(!fMB->IsSilentFile())
         {
             if (type==1) fLinPDFS[v]->Fill(lx,w*ew);
             else         fLinPDFB[v]->Fill(lx,w*ew);
         }
      }
      //
      // Get normalization.
      //
      stdl = TMath::Sqrt( (varsum2[v] - (varsum[v]*varsum[v]/fRuleFit->GetNEveEff()))/(fRuleFit->GetNEveEff()-averageWeight) );
      fLinNorm[v] = CalcLinNorm(stdl);
   }
   // Save PDFs - for debugging purpose
   if(!fMB->IsSilentFile())
   {
        for (UInt_t v=0; v<nvars; v++) {
            fLinPDFS[v]->Write();
            fLinPDFB[v]->Write();
        }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This function returns Pr( y = 1 | x ) for the linear terms.

Double_t TMVA::RuleEnsemble::PdfLinear( Double_t & nsig, Double_t & ntot  ) const
{
   UInt_t nvars=fLinDP.size();

   Double_t fstot=0;
   Double_t fbtot=0;
   nsig = 0;
   ntot = nvars;
   for (UInt_t v=0; v<nvars; v++) {
      Double_t val = fEventLinearVal[v];
      Int_t bin = fLinPDFS[v]->FindBin(val);
      fstot += fLinPDFS[v]->GetBinContent(bin);
      fbtot += fLinPDFB[v]->GetBinContent(bin);
   }
   if (nvars<1) return 0;
   ntot = (fstot+fbtot)/Double_t(nvars);
   nsig = (fstot)/Double_t(nvars);
   return fstot/(fstot+fbtot);
}

////////////////////////////////////////////////////////////////////////////////
/// This function returns Pr( y = 1 | x ) for rules.
/// The probability returned is normalized against the number of rules which are actually passed

Double_t TMVA::RuleEnsemble::PdfRule( Double_t & nsig, Double_t & ntot  ) const
{
   Double_t sump  = 0;
   Double_t sumok = 0;
   Double_t sumz  = 0;
   Double_t ssb;
   Double_t neve;
   //
   UInt_t nrules = fRules.size();
   for (UInt_t ir=0; ir<nrules; ir++) {
      if (fEventRuleVal[ir]>0) {
         ssb = fEventRuleVal[ir]*GetRulesConst(ir)->GetSSB(); // S/(S+B) is evaluated in CalcRuleSupport() using ALL training events
         neve = GetRulesConst(ir)->GetSSBNeve(); // number of events accepted by the rule
         sump  += ssb*neve; // number of signal events
         sumok += neve; // total number of events passed
      }
      else sumz += 1.0; // all events
   }

   nsig = sump;
   ntot = sumok;
   //
   if (ntot>0) return nsig/ntot;
   return 0.0;
}

////////////////////////////////////////////////////////////////////////////////
/// We want to estimate F* = argmin Eyx( L(y,F(x) ), min wrt F(x)
/// F(x) = FL(x) + FR(x) , linear and rule part

Double_t TMVA::RuleEnsemble::FStar( const Event & e )
{
   SetEvent(e);
   UpdateEventVal();
   return FStar();
}

////////////////////////////////////////////////////////////////////////////////
/// We want to estimate F* = argmin Eyx( L(y,F(x) ), min wrt F(x)
/// F(x) = FL(x) + FR(x) , linear and rule part

Double_t TMVA::RuleEnsemble::FStar() const
{
   Double_t p=0;
   Double_t nrs=0, nrt=0;
   Double_t nls=0, nlt=0;
   Double_t nt;
   Double_t pr=0;
   Double_t pl=0;

   // first calculate Pr(y=1|X) for rules and linear terms
   if (DoLinear()) pl = PdfLinear(nls, nlt);
   if (DoRules())  pr = PdfRule(nrs, nrt);
   // nr(l)t=0 or 1
   if ((nlt>0) && (nrt>0)) nt=2.0;
   else                    nt=1.0;
   p = (pl+pr)/nt;
   return 2.0*p-1.0;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate various statistics for this rule

void TMVA::RuleEnsemble::RuleResponseStats()
{
   // TODO: NOT YET UPDATED FOR WEIGHTS
   const std::vector<const Event *> *events = GetTrainingEvents();
   const UInt_t neve   = events->size();
   const UInt_t nvars  = GetMethodBase()->GetNvar();
   const UInt_t nrules = fRules.size();
   const Event *eveData;
   // Flags
   Bool_t sigRule;
   Bool_t sigTag;
   Bool_t bkgTag;
   // Bool_t noTag;
   Bool_t sigTrue;
   Bool_t tagged;
   // Counters
   Int_t nsig=0;
   Int_t nbkg=0;
   Int_t ntag=0;
   Int_t nss=0;
   Int_t nsb=0;
   Int_t nbb=0;
   Int_t nbs=0;
   std::vector<Int_t> varcnt;
   // Clear vectors
   fRulePSS.clear();
   fRulePSB.clear();
   fRulePBS.clear();
   fRulePBB.clear();
   fRulePTag.clear();
   //
   varcnt.resize(nvars,0);
   fRuleVarFrac.clear();
   fRuleVarFrac.resize(nvars,0);
   //
   for ( UInt_t i=0; i<nrules; i++ ) {
      for ( UInt_t v=0; v<nvars; v++) {
         if (fRules[i]->ContainsVariable(v)) varcnt[v]++; // count how often a variable occurs
      }
      sigRule = fRules[i]->IsSignalRule();
      if (sigRule) { // rule is a signal rule (ie s/(s+b)>0.5)
         nsig++;
      }
      else {
         nbkg++;
      }
      // reset counters
      nss=0;
      nsb=0;
      nbs=0;
      nbb=0;
      ntag=0;
      // loop over all events
      for (UInt_t e=0; e<neve; e++) {
         eveData = (*events)[e];
         tagged  = fRules[i]->EvalEvent(*eveData);
         sigTag = (tagged && sigRule);        // it's tagged as a signal
         bkgTag = (tagged && (!sigRule));     // ... as bkg
         // noTag = !(sigTag || bkgTag);         // ... not tagged
         sigTrue = (eveData->GetClass() == 0);       // true if event is true signal
         if (tagged) {
            ntag++;
            if (sigTag && sigTrue)  nss++;
            if (sigTag && !sigTrue) nsb++;
            if (bkgTag && sigTrue)  nbs++;
            if (bkgTag && !sigTrue) nbb++;
         }
      }
      // Fill tagging probabilities
      if (ntag>0 && neve > 0) { // should always be the case, but let's make sure and keep coverity quiet
         fRulePTag.push_back(Double_t(ntag)/Double_t(neve));
         fRulePSS.push_back(Double_t(nss)/Double_t(ntag));
         fRulePSB.push_back(Double_t(nsb)/Double_t(ntag));
         fRulePBS.push_back(Double_t(nbs)/Double_t(ntag));
         fRulePBB.push_back(Double_t(nbb)/Double_t(ntag));
      }
      //
   }
   fRuleFSig = (nsig>0) ? static_cast<Double_t>(nsig)/static_cast<Double_t>(nsig+nbkg) : 0;
   for ( UInt_t v=0; v<nvars; v++) {
      fRuleVarFrac[v] =  (nrules>0) ? Double_t(varcnt[v])/Double_t(nrules) : 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// calculate various statistics for this rule

void TMVA::RuleEnsemble::RuleStatistics()
{
   const UInt_t nrules = fRules.size();
   Double_t nc;
   Double_t sumNc =0;
   Double_t sumNc2=0;
   for ( UInt_t i=0; i<nrules; i++ ) {
      nc = static_cast<Double_t>(fRules[i]->GetNcuts());
      sumNc  += nc;
      sumNc2 += nc*nc;
   }
   fRuleNCave = 0.0;
   fRuleNCsig = 0.0;
   if (nrules>0) {
      fRuleNCave = sumNc/nrules;
      fRuleNCsig = TMath::Sqrt(gTools().ComputeVariance(sumNc2,sumNc,nrules));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// print rule generation info

void TMVA::RuleEnsemble::PrintRuleGen() const
{
   Log() << kHEADER << "-------------------RULE ENSEMBLE SUMMARY------------------------"  << Endl;
   const MethodRuleFit *mrf = GetMethodRuleFit();
   if (mrf) Log() << kINFO << "Tree training method               : " << (mrf->UseBoost() ? "AdaBoost":"Random") << Endl;
   Log() << kINFO << "Number of events per tree          : " << fRuleFit->GetNTreeSample()    << Endl;
   Log() << kINFO << "Number of trees                    : " << fRuleFit->GetForest().size() << Endl;
   Log() << kINFO << "Number of generated rules          : " << fNRulesGenerated << Endl;
   Log() << kINFO << "Idem, after cleanup                : " << fRules.size() << Endl;
   Log() << kINFO << "Average number of cuts per rule    : " << Form("%8.2f",fRuleNCave) << Endl;
   Log() << kINFO << "Spread in number of cuts per rules : " << Form("%8.2f",fRuleNCsig) << Endl;
   Log() << kVERBOSE << "Complexity                         : " << Form("%8.2f",fRuleNCave*fRuleNCsig) << Endl;
   Log() << kINFO << "----------------------------------------------------------------"  << Endl;
   Log() << kINFO << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// print function

void TMVA::RuleEnsemble::Print() const
{
   const EMsgType kmtype=kINFO;
   const Bool_t   isDebug = (fLogger->GetMinType()<=kDEBUG);
   //
   Log() << kmtype << Endl;
   Log() << kmtype << "================================================================" << Endl;
   Log() << kmtype << "                          M o d e l                             " << Endl;
   Log() << kmtype << "================================================================" << Endl;

   Int_t ind;
   const UInt_t nvars =  GetMethodBase()->GetNvar();
   const Int_t nrules = fRules.size();
   const Int_t printN = TMath::Min(10,nrules); //nrules+1;
   Int_t maxL = 0;
   for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
      if (GetMethodBase()->GetInputLabel(iv).Length() > maxL) maxL = GetMethodBase()->GetInputLabel(iv).Length();
   }
   //
   if (isDebug) {
      Log() << kDEBUG << "Variable importance:" << Endl;
      for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
         Log() << kDEBUG << std::setw(maxL) << GetMethodBase()->GetInputLabel(iv)
               << std::resetiosflags(std::ios::right)
               << " : " << Form(" %3.3f",fVarImportance[iv]) << Endl;
      }
   }
   //
   Log() << kHEADER << "Offset (a0) = " << fOffset << Endl;
   //
   if (DoLinear()) {
      if (fLinNorm.size() > 0) {
         Log() << kmtype << "------------------------------------" << Endl;
         Log() << kmtype << "Linear model (weights unnormalised)" << Endl;
         Log() << kmtype << "------------------------------------" << Endl;
         Log() << kmtype << std::setw(maxL) << "Variable"
               << std::resetiosflags(std::ios::right) << " : "
               << std::setw(11) << " Weights"
               << std::resetiosflags(std::ios::right) << " : "
               << "Importance"
               << std::resetiosflags(std::ios::right)
               << Endl;
         Log() << kmtype << "------------------------------------" << Endl;
         for ( UInt_t i=0; i<fLinNorm.size(); i++ ) {
            Log() << kmtype << std::setw(std::max(maxL,8)) << GetMethodBase()->GetInputLabel(i);
            if (fLinTermOK[i]) {
               Log() << kmtype
                     << std::resetiosflags(std::ios::right)
                     << " : " << Form(" %10.3e",fLinCoefficients[i]*fLinNorm[i])
                     << " : " << Form(" %3.3f",fLinImportance[i]/fImportanceRef) << Endl;
            }
            else {
               Log() << kmtype << "-> importance below threshold = "
                     << Form(" %3.3f",fLinImportance[i]/fImportanceRef) << Endl;
            }
         }
         Log() << kmtype << "------------------------------------" << Endl;
      }
   }
   else Log() << kmtype << "Linear terms were disabled" << Endl;

   if ((!DoRules()) || (nrules==0)) {
      if (!DoRules()) {
         Log() << kmtype << "Rule terms were disabled" << Endl;
      }
      else {
         Log() << kmtype << "Even though rules were included in the model, none passed! " << nrules << Endl;
      }
   }
   else {
      Log() << kmtype << "Number of rules = " << nrules << Endl;
      if (isDebug) {
         Log() << kmtype << "N(cuts) in rules, average = " << fRuleNCave << Endl;
         Log() << kmtype << "                      RMS = " << fRuleNCsig << Endl;
         Log() << kmtype << "Fraction of signal rules = " << fRuleFSig << Endl;
         Log() << kmtype << "Fraction of rules containing a variable (%):" << Endl;
         for ( UInt_t v=0; v<nvars; v++) {
            Log() << kmtype << "   " << std::setw(maxL) << GetMethodBase()->GetInputLabel(v);
            Log() << kmtype << Form(" = %2.2f",fRuleVarFrac[v]*100.0) << " %" << Endl;
         }
      }
      //
      // Print out all rules sorted in importance
      //
      std::list< std::pair<double,int> > sortedImp;
      for (Int_t i=0; i<nrules; i++) {
         sortedImp.push_back( std::pair<double,int>( fRules[i]->GetImportance(),i ) );
      }
      sortedImp.sort();
      //
      Log() << kmtype << "Printing the first " << printN << " rules, ordered in importance." << Endl;
      int pind=0;
      for ( std::list< std::pair<double,int> >::reverse_iterator itpair = sortedImp.rbegin();
            itpair != sortedImp.rend(); ++itpair ) {
         ind = itpair->second;
         //    if (pind==0) impref =
         //         Log() << kmtype << "Rule #" <<
         //         Log() << kmtype << *fRules[ind] << Endl;
         fRules[ind]->PrintLogger(Form("Rule %4d : ",pind+1));
         pind++;
         if (pind==printN) {
            if (nrules==printN) {
               Log() << kmtype << "All rules printed" << Endl;
            }
            else {
               Log() << kmtype << "Skipping the next " << nrules-printN << " rules" << Endl;
            }
            break;
         }
      }
   }
   Log() << kmtype << "================================================================" << Endl;
   Log() << kmtype << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// write rules to stream

void TMVA::RuleEnsemble::PrintRaw( std::ostream & os ) const
{
   Int_t dp = os.precision();
   UInt_t nrules = fRules.size();
   //   std::sort(fRules.begin(),fRules.end());
   //
   os << "ImportanceCut= "    << fImportanceCut << std::endl;
   os << "LinQuantile= "      << fLinQuantile   << std::endl;
   os << "AverageSupport= "   << fAverageSupport << std::endl;
   os << "AverageRuleSigma= " << fAverageRuleSigma << std::endl;
   os << "Offset= "           << fOffset << std::endl;
   os << "NRules= "           << nrules << std::endl;
   for (UInt_t i=0; i<nrules; i++){
      os << "***Rule " << i << std::endl;
      (fRules[i])->PrintRaw(os);
   }
   UInt_t nlinear = fLinNorm.size();
   //
   os << "NLinear= " << fLinTermOK.size() << std::endl;
   for (UInt_t i=0; i<nlinear; i++) {
      os << "***Linear " << i << std::endl;
      os << std::setprecision(10) << (fLinTermOK[i] ? 1:0) << " "
         << fLinCoefficients[i] << " "
         << fLinNorm[i] << " "
         << fLinDM[i] << " "
         << fLinDP[i] << " "
         << fLinImportance[i] << " " << std::endl;
   }
   os << std::setprecision(dp);
}

////////////////////////////////////////////////////////////////////////////////
/// write rules to XML

void* TMVA::RuleEnsemble::AddXMLTo(void* parent) const
{
   void* re = gTools().AddChild( parent, "Weights" ); // this is the "RuleEnsemble"

   UInt_t nrules  = fRules.size();
   UInt_t nlinear = fLinNorm.size();
   gTools().AddAttr( re, "NRules",           nrules );
   gTools().AddAttr( re, "NLinear",          nlinear );
   gTools().AddAttr( re, "LearningModel",    (int)fLearningModel );
   gTools().AddAttr( re, "ImportanceCut",    fImportanceCut );
   gTools().AddAttr( re, "LinQuantile",      fLinQuantile );
   gTools().AddAttr( re, "AverageSupport",   fAverageSupport );
   gTools().AddAttr( re, "AverageRuleSigma", fAverageRuleSigma );
   gTools().AddAttr( re, "Offset",           fOffset );
   for (UInt_t i=0; i<nrules; i++) fRules[i]->AddXMLTo(re);

   for (UInt_t i=0; i<nlinear; i++) {
      void* lin = gTools().AddChild( re, "Linear" );
      gTools().AddAttr( lin, "OK",         (fLinTermOK[i] ? 1:0) );
      gTools().AddAttr( lin, "Coeff",      fLinCoefficients[i] );
      gTools().AddAttr( lin, "Norm",       fLinNorm[i] );
      gTools().AddAttr( lin, "DM",         fLinDM[i] );
      gTools().AddAttr( lin, "DP",         fLinDP[i] );
      gTools().AddAttr( lin, "Importance", fLinImportance[i] );
   }
   return re;
}

////////////////////////////////////////////////////////////////////////////////
/// read rules from XML

void TMVA::RuleEnsemble::ReadFromXML( void* wghtnode )
{
   UInt_t nrules, nlinear;
   gTools().ReadAttr( wghtnode, "NRules",   nrules );
   gTools().ReadAttr( wghtnode, "NLinear",  nlinear );
   Int_t iLearningModel;
   gTools().ReadAttr( wghtnode, "LearningModel",     iLearningModel );
   fLearningModel =  (ELearningModel) iLearningModel;
   gTools().ReadAttr( wghtnode, "ImportanceCut",     fImportanceCut );
   gTools().ReadAttr( wghtnode, "LinQuantile",       fLinQuantile );
   gTools().ReadAttr( wghtnode, "AverageSupport",    fAverageSupport );
   gTools().ReadAttr( wghtnode, "AverageRuleSigma",  fAverageRuleSigma );
   gTools().ReadAttr( wghtnode, "Offset",            fOffset );

   // read rules
   DeleteRules();

   UInt_t i = 0;
   fRules.resize( nrules  );
   void* ch = gTools().GetChild( wghtnode );
   for (i=0; i<nrules; i++) {
      fRules[i] = new Rule();
      fRules[i]->SetRuleEnsemble( this );
      fRules[i]->ReadFromXML( ch );

      ch = gTools().GetNextChild(ch);
   }

   // read linear classifier (Fisher)
   fLinNorm        .resize( nlinear );
   fLinTermOK      .resize( nlinear );
   fLinCoefficients.resize( nlinear );
   fLinDP          .resize( nlinear );
   fLinDM          .resize( nlinear );
   fLinImportance  .resize( nlinear );

   Int_t iok;
   i=0;
   while(ch) {
      gTools().ReadAttr( ch, "OK",         iok );
      fLinTermOK[i] = (iok == 1);
      gTools().ReadAttr( ch, "Coeff",      fLinCoefficients[i]  );
      gTools().ReadAttr( ch, "Norm",       fLinNorm[i]          );
      gTools().ReadAttr( ch, "DM",         fLinDM[i]            );
      gTools().ReadAttr( ch, "DP",         fLinDP[i]            );
      gTools().ReadAttr( ch, "Importance", fLinImportance[i]    );

      i++;
      ch = gTools().GetNextChild(ch);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// read rule ensemble from stream

void TMVA::RuleEnsemble::ReadRaw( std::istream & istr )
{
   UInt_t nrules;
   //
   std::string dummy;
   Int_t idum;
   //
   // First block is general stuff
   //
   istr >> dummy >> fImportanceCut;
   istr >> dummy >> fLinQuantile;
   istr >> dummy >> fAverageSupport;
   istr >> dummy >> fAverageRuleSigma;
   istr >> dummy >> fOffset;
   istr >> dummy >> nrules;
   //
   // Now read in the rules
   //
   DeleteRules();
   //
   for (UInt_t i=0; i<nrules; i++){
      istr >> dummy >> idum; // read line  "***Rule <ind>"
      fRules.push_back( new Rule() );
      (fRules.back())->SetRuleEnsemble( this );
      (fRules.back())->ReadRaw(istr);
   }
   //
   // and now the linear terms
   //
   UInt_t nlinear;
   //
   // coverity[tainted_data_argument]
   istr >> dummy >> nlinear;
   //
   fLinNorm        .resize( nlinear );
   fLinTermOK      .resize( nlinear );
   fLinCoefficients.resize( nlinear );
   fLinDP          .resize( nlinear );
   fLinDM          .resize( nlinear );
   fLinImportance  .resize( nlinear );
   //

   Int_t iok;
   for (UInt_t i=0; i<nlinear; i++) {
      istr >> dummy >> idum;
      istr >> iok;
      fLinTermOK[i] = (iok==1);
      istr >> fLinCoefficients[i];
      istr >> fLinNorm[i];
      istr >> fLinDM[i];
      istr >> fLinDP[i];
      istr >> fLinImportance[i];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// copy function

void TMVA::RuleEnsemble::Copy( const RuleEnsemble & other )
{
   if(this != &other) {
      fRuleFit           = other.GetRuleFit();
      fRuleMinDist       = other.GetRuleMinDist();
      fOffset            = other.GetOffset();
      fRules             = other.GetRulesConst();
      fImportanceCut     = other.GetImportanceCut();
      fVarImportance     = other.GetVarImportance();
      fLearningModel     = other.GetLearningModel();
      fLinQuantile       = other.GetLinQuantile();
      fRuleNCsig         = other.fRuleNCsig;
      fAverageRuleSigma  = other.fAverageRuleSigma;
      fEventCacheOK      = other.fEventCacheOK;
      fImportanceRef     = other.fImportanceRef;
      fNRulesGenerated   = other.fNRulesGenerated;
      fRuleFSig          = other.fRuleFSig;
      fRuleMapInd0       = other.fRuleMapInd0;
      fRuleMapInd1       = other.fRuleMapInd1;
      fRuleMapOK         = other.fRuleMapOK;
      fRuleNCave         = other.fRuleNCave;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// calculate the number of rules

Int_t TMVA::RuleEnsemble::CalcNRules( const DecisionTree *dtree )
{
   if (dtree==0) return 0;
   Node *node = dtree->GetRoot();
   Int_t nendnodes = 0;
   FindNEndNodes( node, nendnodes );
   return 2*(nendnodes-1);
}

////////////////////////////////////////////////////////////////////////////////
/// find the number of leaf nodes

void TMVA::RuleEnsemble::FindNEndNodes( const Node *node, Int_t & nendnodes )
{
   if (node==0) return;
   if ((node->GetRight()==0) && (node->GetLeft()==0)) {
      ++nendnodes;
      return;
   }
   const Node *nodeR = node->GetRight();
   const Node *nodeL = node->GetLeft();
   FindNEndNodes( nodeR, nendnodes );
   FindNEndNodes( nodeL, nendnodes );
}

////////////////////////////////////////////////////////////////////////////////
/// create rules from the decision tree structure

void TMVA::RuleEnsemble::MakeRulesFromTree( const DecisionTree *dtree )
{
   Node *node = dtree->GetRoot();
   AddRule( node );
}

////////////////////////////////////////////////////////////////////////////////
/// add a new rule to the tree

void TMVA::RuleEnsemble::AddRule( const Node *node )
{
   if (node==0) return;
   if (node->GetParent()==0) { // it's a root node, don't make a rule
      AddRule( node->GetRight() );
      AddRule( node->GetLeft() );
   }
   else {
      Rule *rule = MakeTheRule(node);
      if (rule) {
         fRules.push_back( rule );
         AddRule( node->GetRight() );
         AddRule( node->GetLeft() );
      }
      else {
         Log() << kFATAL << "<AddRule> - ERROR failed in creating a rule! BUG!" << Endl;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Make a Rule from a given Node.
/// The root node (ie no parent) does not generate a Rule.
/// The first node in a rule is always the root node => fNodes.size()>=2
/// Each node corresponds to a cut and the cut value is given by the parent node.

TMVA::Rule *TMVA::RuleEnsemble::MakeTheRule( const Node *node )
{
   if (node==0) {
      Log() << kFATAL << "<MakeTheRule> Input node is NULL. Should not happen. BUG!" << Endl;
      return 0;
   }

   if (node->GetParent()==0) { // a root node - ignore
      return 0;
   }
   //
   std::vector< const Node * > nodeVec;
   const Node *parent = node;
   //
   // Make list with the input node at the end:
   // <root node> <node1> <node2> ... <node given as argument>
   //
   nodeVec.push_back( node );
   while (parent!=0) {
      parent = parent->GetParent();
      if (!parent) continue;
      const DecisionTreeNode* dtn = dynamic_cast<const DecisionTreeNode*>(parent);
      if (dtn && dtn->GetSelector()>=0)
         nodeVec.insert( nodeVec.begin(), parent );

   }
   if (nodeVec.size()<2) {
      Log() << kFATAL << "<MakeTheRule> BUG! Inconsistent Rule!" << Endl;
      return 0;
   }
   Rule *rule = new Rule( this, nodeVec );
   rule->SetMsgType( Log().GetMinType() );
   return rule;
}

////////////////////////////////////////////////////////////////////////////////
/// Makes rule map for all events

void TMVA::RuleEnsemble::MakeRuleMap(const std::vector<const Event *> *events, UInt_t ifirst, UInt_t ilast)
{
   Log() << kVERBOSE << "Making Rule map for all events" << Endl;
   // make rule response map
   if (events==0) events = GetTrainingEvents();
   if ((ifirst==0) || (ilast==0) || (ifirst>ilast)) {
      ifirst = 0;
      ilast  = events->size()-1;
   }
   // check if identical to previous call
   if ((events!=fRuleMapEvents) ||
       (ifirst!=fRuleMapInd0) ||
       (ilast !=fRuleMapInd1)) {
      fRuleMapOK = kFALSE;
   }
   //
   if (fRuleMapOK) {
      Log() << kVERBOSE << "<MakeRuleMap> Map is already valid" << Endl;
      return;  // already cached
   }
   fRuleMapEvents = events;
   fRuleMapInd0   = ifirst;
   fRuleMapInd1   = ilast;
   // check number of rules
   UInt_t nrules = GetNRules();
   if (nrules==0) {
      Log() << kVERBOSE << "No rules found in MakeRuleMap()" << Endl;
      fRuleMapOK = kTRUE;
      return;
   }
   //
   // init map
   //
   std::vector<UInt_t> ruleind;
   fRuleMap.clear();
   for (UInt_t i=ifirst; i<=ilast; i++) {
      ruleind.clear();
      fRuleMap.push_back( ruleind );
      for (UInt_t r=0; r<nrules; r++) {
         if (fRules[r]->EvalEvent(*((*events)[i]))) {
            fRuleMap.back().push_back(r); // save only rules that are accepted
         }
      }
   }
   fRuleMapOK = kTRUE;
   Log() << kVERBOSE << "Made rule map for event# " << ifirst << " : " << ilast << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// std::ostream operator

std::ostream& TMVA::operator<< ( std::ostream& os, const RuleEnsemble & rules )
{
   os << "DON'T USE THIS - TO BE REMOVED" << std::endl;
   rules.Print();
   return os;
}
