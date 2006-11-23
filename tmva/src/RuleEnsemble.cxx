// @(#)root/tmva $Id: RuleEnsemble.cxx,v 1.7 2006/11/20 15:35:28 brun Exp $
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
 *      Helge Voss         <Helge.Voss@cern.ch>         - MPI-KP Heidelberg, Ger. *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      Iowa State U.                                                             *
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <algorithm>
#include <list>

#include "Riostream.h"
#include "TRandom.h"
#include "TH1F.h"
#include "TMVA/RuleEnsemble.h"
#include "TMVA/RuleFit.h"
#include "TMVA/MethodRuleFit.h"

//_______________________________________________________________________
TMVA::RuleEnsemble::RuleEnsemble( RuleFit *rf )
   : fLearningModel    ( kFull )
   , fAverageRuleSigma ( 0.4 ) // default value - used if only linear model is chosen
   , fMaxRuleDist      ( 1e-3 ) // closest allowed 'distance' between two rules
   , fLogger( "RuleEnsemble" )
{
   // constructor
   Initialize( rf );
}

//_______________________________________________________________________
TMVA::RuleEnsemble::RuleEnsemble( const RuleEnsemble& other )
   : fLogger( "RuleEnsemble" )
{
   // copy constructor
   Copy( other );
}

//_______________________________________________________________________
TMVA::RuleEnsemble::RuleEnsemble()
   : fLogger( "RuleEnsemble" )
{
   // constructor
}

//_______________________________________________________________________
TMVA::RuleEnsemble::~RuleEnsemble()
{
   // destructor
   for ( std::vector< TMVA::Rule *>::iterator itrRule = fRules.begin(); itrRule != fRules.end(); itrRule++ ) {
      delete *itrRule;
   }
   // NOTE: Should not delete the histos fLinPDFB/S since they are delete elsewhere
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::Initialize( RuleFit *rf )
{
   // Initializes all member variables with default values

   fAverageRuleSigma = 0.4; // default value - used if only linear model is chosen
   fRuleFit = rf;
   UInt_t nvars =  GetMethodRuleFit()->GetNvar();
   fVarImportance.resize( nvars );
   fLinPDFB.resize( nvars,0 );
   fLinPDFS.resize( nvars,0 );
   fImportanceRef = 1.0;
   for (UInt_t i=0; i<nvars; i++) { // a priori all linear terms are equally valid
      fLinTermOK.push_back(kTRUE);
   }
}

//_______________________________________________________________________
const TMVA::MethodRuleFit*  TMVA::RuleEnsemble::GetMethodRuleFit() const
{
   //
   // Get a pointer to the original MethodRuleFit.
   //
   return ( fRuleFit==0 ? 0:fRuleFit->GetMethodRuleFit());
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::MakeModel()
{
   // create model
   if (DoRules())
      MakeRules( fRuleFit->GetForest() );
   if (DoLinear())
      MakeLinearTerms();
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::CoefficientRadius()
{
   //
   // Calculates sqrt(Sum(a_i^2)), i=1..N (NOTE do not include a0)
   //
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

//_______________________________________________________________________
void TMVA::RuleEnsemble::ResetCoefficients()
{
   // reset all rule coefficients

   fOffset = 0.0;
   UInt_t nrules = fRules.size();
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(0.0);
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::SetCoefficients( const std::vector< Double_t > & v )
{
   // set all rule coefficients

   UInt_t nrules = fRules.size();
   if (v.size()!=nrules) {
      fLogger << kFATAL << "<SetCoefficients> - BUG TRAP - input vector worng size! It is = " << v.size()
              << " when it should be = " << nrules << Endl;
   }
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(v[i]);
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::GetCoefficients( std::vector< Double_t > & v )
{
   // Retrieve all rule coefficients

   UInt_t nrules = fRules.size();
   v.resize(nrules);
   if (nrules==0) return;
   //
   for (UInt_t i=0; i<nrules; i++) {
      v[i] = (fRules[i]->GetCoefficient());
   }
}

//_______________________________________________________________________
const std::vector<const TMVA::Event *> *TMVA::RuleEnsemble::GetTrainingEvents()  const
{ 
   // get list of training events from the rule fitter

   return &(fRuleFit->GetTrainingEvents());
}

//_______________________________________________________________________
const std::vector< Int_t > *TMVA::RuleEnsemble::GetSubsampleEvents() const
{
   // get list of events for the subsamples from the rule fitter
   return &(fRuleFit->GetSubsampleEvents());
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const
{
   // get list of events for the subsample sub from the rule fitter
   fRuleFit->GetSubsampleEvents(sub,ibeg,iend);
}
//_______________________________________________________________________
UInt_t TMVA::RuleEnsemble::GetNSubsamples() const
{
   // get the number of subsamples from the rule fitter
   return fRuleFit->GetNSubsamples();
}

//_______________________________________________________________________
const TMVA::Event * TMVA::RuleEnsemble::GetTrainingEvent(UInt_t i) const
{
   // get the training event from the rule fitter
   return fRuleFit->GetTrainingEvent(i);
}

//_______________________________________________________________________
const TMVA::Event * TMVA::RuleEnsemble::GetTrainingEvent(UInt_t i, UInt_t isub)  const
{
   // get one training event for one subsample from the rule fitter
   return fRuleFit->GetTrainingEvent(i,isub);
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::SetRulesNCuts()
{
   // set the number of nodes to the cut array
 
   std::vector<Int_t> nodes;
   fRulesNCuts.clear();
   for (UInt_t i=0; i<fRules.size(); i++) {
      fRules[i]->GetEffectiveRule( nodes );
      fRulesNCuts.push_back( Int_t(nodes.size())-1 );
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::RemoveSimpleRules()
{
   // remove all simple rules

   fLogger << kINFO << "removing simple rules" << Endl;
   UInt_t nrulesIn = fRules.size();
   std::vector<bool> removeMe( nrulesIn,false );
   //
   for (UInt_t i=0; i<nrulesIn; i++) {
      if (fRules[i]->IsSimpleRule()) {
         //         removeMe[i] = kTRUE;
      }
   }
   UInt_t ind = 0;
   TMVA::Rule *theRule;
   for (UInt_t i=0; i<nrulesIn; i++) {
      if (removeMe[i]) {
         theRule = fRules[ind];
         fRules.erase( std::vector<TMVA::Rule *>::iterator(&fRules[ind]) );
         delete theRule;
         ind--;
      } else {
      }
      ind++;
   }
   UInt_t nrulesOut = fRules.size();
   fLogger << kINFO << "removed " << nrulesIn - nrulesOut << " out of " << nrulesIn<< " rules" << Endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::RemoveSimilarRules()
{
   // remove rules that behave similar 
   
   fLogger << kINFO << "removing similar rules; distance = " << fMaxRuleDist << Endl;

   UInt_t nrulesIn = fRules.size();
   TMVA::Rule *first, *second;
   std::vector<bool> removeMe( nrulesIn,false );

   Int_t nrem = 0;
   Int_t remind=-1;
   Double_t r;

   for (UInt_t i=0; i<nrulesIn; i++) {
      if (!removeMe[i]) {
         first = fRules[i];
         for (UInt_t k=i+1; k<nrulesIn; k++) {
            if (!removeMe[k]) {
               second = fRules[k];
               Bool_t equal = first->Equal(*second,kTRUE,fMaxRuleDist);
               if (equal) {
                  r = gRandom->Rndm();
                  remind = (r>0.5 ? k:i); // randomly select rule
               } else {
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
   TMVA::Rule *theRule;
   for (UInt_t i=0; i<nrulesIn; i++) {
      if (removeMe[i]) {
         theRule = fRules[ind];
         fRules.erase( std::vector<TMVA::Rule *>::iterator(&fRules[ind]) );
         delete theRule;
         ind--;
      } else {
      }
      ind++;
   }
   UInt_t nrulesOut = fRules.size();
   fLogger << kINFO << "removed " << nrulesIn - nrulesOut << " out of " << nrulesIn<< " rules" << Endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CleanupRules()
{
   // cleanup rules

   UInt_t nrules   = fRules.size();
   if (nrules==0) return;
   fLogger << kINFO << "removing rules with relative importance < " << fImportanceCut << Endl;
   if (fImportanceCut<=0) return;
   //
   // Mark rules to be removed
   //
   TMVA::Rule *therule;
   Int_t ind=0;
   for (UInt_t i=0; i<nrules; i++) {
      if (fRules[ind]->GetRelImportance()<fImportanceCut) {
         therule = fRules[ind];
         fRules.erase( std::vector<Rule *>::iterator(&fRules[ind]) );
         fRulesNCuts.erase( std::vector<Int_t>::iterator(&fRulesNCuts[ind]) ); // PHASE OUT
         delete therule;
         ind--;
      } 
      ind++;
   }
   fLogger << kINFO << "removed in total " << nrules-ind << " out of " << nrules << " rules" << Endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CleanupLinear()
{
   // cleanup linear model

   UInt_t nlin = fLinNorm.size();
   if (nlin==0) return;
   fLogger << kINFO << "removing linear terms with relative importance < " << fImportanceCut << Endl;
   if (fImportanceCut<=0) return;
   //
   fLinTermOK.clear();
   for (UInt_t i=0; i<nlin; i++) {
      fLinTermOK.push_back( (fLinImportance[i]/fImportanceRef > fImportanceCut) );
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CalcRuleSupport()
{
   // calculate the support for all rules

   Double_t seve, s,t,stot,ttot;
   Double_t ssig, sbkg;
   Int_t indrule=0;
   stot = 0;
   ttot = 0;
   const std::vector<const TMVA::Event *> *events = GetTrainingEvents();
   Double_t nevent=static_cast<Double_t>(events->size());
   Double_t nrules=static_cast<Double_t>(fRules.size());
   if (nevent>0) {

      for ( std::vector< TMVA::Rule * >::iterator itrRule=fRules.begin(); itrRule!=fRules.end(); itrRule++ ) {
         s=0.0;
         ssig=0;
         sbkg=0;
         for ( std::vector<const TMVA::Event * >::const_iterator itrEvent=events->begin(); itrEvent!=events->end(); itrEvent++ ) {
            seve = (*itrRule)->EvalEvent( *(*itrEvent) );
            if (seve>0) {
               s++;
               if ((*itrEvent)->IsSignal()) ssig++;
               else                         sbkg++;
            }
         }
         //
         s = s/nevent;
         t = s*(1.0-s);
         t = (t<0 ? 0:sqrt(t));
         stot += s;
         ttot += t;
         (*itrRule)->SetSupport(s);
         (*itrRule)->SetSigma(t);
         (*itrRule)->SetNorm(t);
         (*itrRule)->SetSSB(Double_t(ssig)/Double_t(ssig+sbkg));
         (*itrRule)->SetSSBNeve(Double_t(ssig+sbkg));
         indrule++;
      }
      fAverageSupport   = stot/nrules;
      fAverageRuleSigma = sqrt(fAverageSupport*(1.0-fAverageSupport));
      fLogger << kINFO << "standard deviation of support = " << fAverageRuleSigma << Endl;
      fLogger << kINFO << "average rule support          = " << fAverageSupport   << Endl;
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CalcImportance()
{
   // calculate the importance of each rule

   Double_t maxRuleImp = CalcRuleImportance();
   Double_t maxLinImp  = CalcLinImportance();
   Double_t maxImp = (maxRuleImp>maxLinImp ? maxRuleImp : maxLinImp);

   for ( UInt_t i=0; i<fRules.size(); i++ ) {
      fRules[i]->SetImportanceRef(maxImp);
   }
   fImportanceRef = maxImp;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::CalcRuleImportance()
{
   // calculate importance of each rule

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

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::CalcLinImportance()
{
   // calculate the linear importance for each rule

   Double_t maxImp=-1.0;
   UInt_t nvars = fLinCoefficients.size();
   fLinImportance.resize(nvars,0.0);
   if (!DoLinear()) return maxImp;
   //
   Double_t imp;
   for ( UInt_t i=0; i<nvars; i++ ) {
      imp = fAverageRuleSigma*TMath::Abs(fLinCoefficients[i]);
      fLinImportance[i] = imp;
      if (imp>maxImp) maxImp = imp;
   }
   return maxImp;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CalcVarImportance()
{
   //
   // Calculates variable importance using eq (35) in RuleFit paper by Friedman et.al
   //
   Double_t rimp;
   UInt_t nrules = fRules.size();
   UInt_t nvars  = GetMethodRuleFit()->GetNvar();
   UInt_t nvarsUsed;
   Double_t rimpN;
   for ( UInt_t iv=0; iv<nvars; iv++ ) {
      fVarImportance[iv] = 0;
   }
   // rules
   if (DoRules()) {
      for ( UInt_t ind=0; ind<nrules; ind++ ) {
         rimp = fRules[ind]->GetImportance();
         nvarsUsed = fRules[ind]->GetNumVarsUsed();
         if (nvarsUsed<1)
            fLogger << kFATAL << "<CalcVarImportance> variables for importance calc!!!??? A BUG!" << Endl;
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

//_______________________________________________________________________
void TMVA::RuleEnsemble::MakeRules( const std::vector< const TMVA::DecisionTree *> & forest )
{
   //
   // Makes rules from the given decision tree.
   // First node in all rules is ALWAYS the root node.
   //
   fRules.clear();
   Int_t nrulesCheck=0;
   //
   for ( UInt_t ind=0; ind<forest.size(); ind++ ) {
      MakeRulesFromTree( forest[ind] );
      nrulesCheck += CalcNRules( forest[ind] );
   }
   fLogger << kINFO << "number of possible rules  = " << nrulesCheck << Endl;
   fLogger << kINFO << "number of generated rules = " << fRules.size() << Endl;

   RemoveSimpleRules();

   RemoveSimilarRules();

   SetRulesNCuts();

   ResetCoefficients();

   CalcRuleSupport();
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::MakeLinearTerms()
{
   //
   // Make the linear terms as in eq 25, ref 2
   // For this the b and (1-b) quatiles are needed
   //
   const std::vector<const TMVA::Event *> *events = GetTrainingEvents();
   UInt_t neve  = events->size();
   UInt_t nvars = ((*events)[0])->GetNVars(); // Event -> GetNVars();
   Double_t val;
   typedef std::pair< Double_t, Int_t> dataType;
   std::vector< std::vector<dataType> > vardata(nvars);
   std::vector< Double_t > varsum(nvars,0.0);
   std::vector< Double_t > varsum2(nvars,0.0);
   // first find stats of all variables
   for (UInt_t i=0; i<neve; i++) {
      for (UInt_t v=0; v<nvars; v++) {
         val = ((*events)[i])->GetVal(v); // Event -> GetVal(v);
         vardata[v].push_back( dataType(val,((*events)[i])->Type()) );
      }
   }
   //
   fLinDP.resize(nvars,0);
   fLinDM.resize(nvars,0);
   fLinCoefficients.resize(nvars,0);
   fLinNorm.resize(nvars,0);

   // sort and find limits
   UInt_t ndata;
   UInt_t nquant;
   Double_t stdl;

   // find normalisation given in ref 2 after eq 26
   Double_t lx;
   for (UInt_t v=0; v<nvars; v++) {
      varsum[v] = 0;
      varsum2[v] = 0;
      //
      std::sort( vardata[v].begin(),vardata[v].end() );
      ndata = vardata[v].size();
      nquant = UInt_t(0.025*Double_t(ndata)); // quantile = 0.025
      fLinDM[v] = vardata[v][nquant].first;         // delta-
      fLinDP[v] = vardata[v][ndata-nquant-1].first; // delta+
      if (fLinPDFB[v]) delete fLinPDFB[v];
      if (fLinPDFS[v]) delete fLinPDFS[v];
      fLinPDFB[v] = new TH1F(Form("bkgvar%d",v),"bkg temphist",40,fLinDM[v],fLinDP[v]);
      fLinPDFS[v] = new TH1F(Form("sigvar%d",v),"sig temphist",40,fLinDM[v],fLinDP[v]);
      fLinPDFB[v]->Sumw2();
      fLinPDFS[v]->Sumw2();
      //
      Int_t type;
      const Double_t w = 1.0/Double_t(neve); // TODO: introduce event weights!
      for (UInt_t i=0; i<neve; i++) {
         val = vardata[v][i].first;
         type = vardata[v][i].second;
         lx = TMath::Min( fLinDP[v], TMath::Max( fLinDM[v], val ) );
         varsum[v] += lx;
         varsum2[v] += lx*lx;
         if (type==1) fLinPDFS[v]->Fill(lx,w);
         else         fLinPDFB[v]->Fill(lx,w);
      }
      stdl = sqrt( (varsum2[v] - (varsum[v]*varsum[v]/Double_t(neve)))/Double_t(neve-1) );
      fLinNorm[v] = ( stdl>0 ? fAverageRuleSigma/stdl : 1.0 ); // norm
   }
   // Save PDFs - for debugging purpose
   for (UInt_t v=0; v<nvars; v++) {
      fLinPDFS[v]->Write();
      fLinPDFB[v]->Write();
   }
}


//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfLinear( Double_t & nsig, Double_t & ntot  ) const
{
   //
   // This function returns Pr( y = 1 | x ) for the linear terms.
   //
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

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfRule( Double_t & nsig, Double_t & ntot  ) const
{
   //
   // This function returns Pr( y = 1 | x ) for rules.
   // The probability returned is normalized against the number of rules which are actually passed
   //
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
      } else sumz += 1.0; // all events
   }

   nsig = sump;
   ntot = sumok;
   //
   if (ntot>0) return nsig/ntot;
   return 0.0;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::FStar( const TMVA::Event & e )
{
   //
   // We want to estimate F* = argmin Eyx( L(y,F(x) ), min wrt F(x)
   // F(x) = FL(x) + FR(x) , linear and rule part
   // 
   //
   SetEvent(e);
   UpdateEventVal();
   return FStar();
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::FStar() const
{
   //
   // We want to estimate F* = argmin Eyx( L(y,F(x) ), min wrt F(x)
   // F(x) = FL(x) + FR(x) , linear and rule part
   // 
   //
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

//_____________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalEvent() const
{
   // evaluate current event

   Int_t nrules = fRules.size();
   Double_t rval=fOffset;
   Double_t linear=0;
   //
   // evaluate all rules
   // normally it should NOT use the normalized rules - the flag should be kFALSE
   //
   if (DoRules()) {
      for ( Int_t i=0; i<nrules; i++ ) {
         rval += fRules[i]->GetCoefficient() * fEventRuleVal[i]; //TRUE);
      }
   }
   //
   // Include linear part - the call below incorporates both coefficient and normalisation (fLinNorm)
   //
   if (DoLinear()) linear = EvalLinEvent();
   rval +=linear;

   return rval;
}

//_____________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalEvent(const TMVA::Event & e)
{
   // evaluate event e
   SetEvent(e);
   UpdateEventVal();
   return EvalEvent();
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEventRaw( UInt_t vind, const TMVA::Event & e)
{
   // evaluate the event linearly (not normalized)

   Double_t val  = e.GetVal(vind);
   Double_t rval = TMath::Min( fLinDP[vind], TMath::Max( fLinDM[vind], val ) );
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEvent( UInt_t vind, Bool_t norm ) const
{
   // evaluate the event linearly normalized
   Double_t rval=0;
   rval = fEventLinearVal[vind];
   if (norm) rval*=fLinNorm[vind];
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEvent() const
{
   // evaluate event linearly

   Double_t rval=0;
   for (UInt_t v=0; v<fLinTermOK.size(); v++) {
      if (fLinTermOK[v])
         rval += fLinCoefficients[v]*fEventLinearVal[v]*fLinNorm[v];
   }
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEvent( const TMVA::Event& e )
{
   // evaluate event linearly

   SetEvent(e);
   UpdateEventVal();
   return EvalLinEvent();
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::RuleStatistics()
{
   // calculate various statistics for this rule

   const UInt_t nrules = fRules.size();
   const std::vector<const TMVA::Event *> *events = GetTrainingEvents();
   const TMVA::Event *eveData;
   const UInt_t neve  = events->size();
   // Flags
   Bool_t sigRule;
   Bool_t sigTag;
   Bool_t bkgTag;
   Bool_t noTag;
   Bool_t sigTrue;
   Bool_t tagged;
   Double_t re;
   // Counters
   Int_t nsig=0;
   Int_t nbkg=0;
   Int_t ntag=0;
   Int_t nss=0;
   Int_t nsb=0;
   Int_t nbb=0;
   Int_t nbs=0;
   // Clear vectors
   fRulePSS.clear();
   fRulePSB.clear();
   fRulePBS.clear();
   fRulePBB.clear();
   fRulePTag.clear();
   //

   for ( UInt_t i=0; i<nrules; i++ ) {
      sigRule = fRules[i]->IsSignalRule();
      if (sigRule) { // rule is a signal rule (ie s/(s+b)>0.5)
         nsig++;
      } else {
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
         re = fRules[i]->EvalEvent(*eveData); // returns 0 (not accepted) or 1 (accepted)
         tagged  = (re>0.5);                  // event is tagged
         sigTag = (tagged && sigRule);        // it's tagged as a signal
         bkgTag = (tagged && (!sigRule));     // ... as bkg
         noTag = !(sigTag || bkgTag);         // ... not tagged
         sigTrue = eveData->IsSignal();       // true if event is true signal
         if (tagged) {
            ntag++;
            if (sigTag && sigTrue)  nss++;
            if (sigTag && !sigTrue) nsb++;
            if (bkgTag && sigTrue)  nbs++;
            if (bkgTag && !sigTrue) nbb++;
         }
      }
      // Fill tagging probabilities
      fRulePTag.push_back(Double_t(ntag)/Double_t(neve));
      fRulePSS.push_back(Double_t(nss)/Double_t(ntag));
      fRulePSB.push_back(Double_t(nsb)/Double_t(ntag));
      fRulePBS.push_back(Double_t(nbs)/Double_t(ntag));
      fRulePBB.push_back(Double_t(nbb)/Double_t(ntag));
      //
   }
   fRuleFSig = static_cast<Double_t>(nsig)/static_cast<Double_t>(nsig+nbkg);
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::Print( ostream& os ) const
{
   // print function

   fLogger << kINFO << "===============================" << Endl;
   fLogger << kINFO << "          RuleEnsemble " << Endl;
   fLogger << kINFO << "===============================" << Endl;

   Int_t ind;
   const Int_t nrules = fRules.size();
   const Int_t printN = 10; //nrules+1;
   Int_t maxL = 0;
   for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
      if (GetMethodRuleFit()->GetInputExp(iv).Length() > maxL) maxL = GetMethodRuleFit()->GetInputExp(iv).Length();
   }
   //
   os << "Offset (a0)     = " << fOffset << std::endl;
   //
   os << "Variable importance:" << std::endl;
   for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
      os << setw(maxL) << GetMethodRuleFit()->GetInputExp(iv) //(*(fRuleFit->GetInputVars()))[iv]
         << resetiosflags(ios::right) 
         << " : " << Form(" %3.3f",fVarImportance[iv]) << std::endl;
   }
   if (DoLinear()) {
      if (fLinNorm.size() > 0) {
         os << "Results from linear terms:" << std::endl;
         os << setw(maxL) << "Variable"
            << resetiosflags(ios::right) << " : "
            << setw(11) << " Weights"
            << resetiosflags(ios::right) << " : "
            << "Importance"
            << resetiosflags(ios::right)
            << std::endl;
         for ( UInt_t i=0; i<fLinNorm.size(); i++ ) {
            os << setw(std::max(maxL,8)) << GetMethodRuleFit()->GetInputExp(i)
               << resetiosflags(ios::right) 
               << " : " << Form(" %10.3e",fLinCoefficients[i])
               << " : " << Form(" %3.3f",fLinImportance[i]/fImportanceRef) << std::endl;
         }
      }
   } 
   else os << "Linear terms were disabled" << std::endl;

   if (DoRules()) {
      os << std::endl;
      os << "Number of rules          = " << nrules << std::endl;
      os << "Fraction of signal rules = " << fRuleFSig << std::endl;
      //
      // Print out all rules
      //
      std::list< std::pair<double,int> > sortedImp;
      for (Int_t i=0; i<nrules; i++) {
         sortedImp.push_back( std::pair<double,int>( fRules[i]->GetImportance(),i ) );
      }
      sortedImp.sort();
      //
      os << "--- RuleEnsemble : Printing the first " << printN << " rules, ordered in importance." << std::endl;
      int pind=0;
      for ( std::list< std::pair<double,int> >::reverse_iterator itpair = sortedImp.rbegin();
            itpair != sortedImp.rend(); itpair++ ) {
         ind = itpair->second;
         //    if (pind==0) impref = 
         os << std::endl;
         os << "Rule #" << Form("%4d",pind+1) << std::endl;
         os << *fRules[ind];
         pind++;
         if (pind==printN) {
            os << std::endl;
            fLogger << kINFO << "skipping the next " << nrules-printN-1 << " rules" << Endl;
            break;
         }
      }
   }
}
//_______________________________________________________________________
void TMVA::RuleEnsemble::PrintRaw( ostream & os ) const
{
   // write rules to stream
   UInt_t nrules = fRules.size();
   //   std::sort(fRules.begin(),fRules.end());
   //
   os << "ImportanceCut= " <<  fImportanceCut << endl;
   os << "AverageSupport= " << fAverageSupport << endl;
   os << "AverageRuleSigma= " << fAverageRuleSigma << endl;
   os << "Offset= "   << fOffset << endl;
   os << "NRules= "   << nrules << endl; 
   for (UInt_t i=0; i<nrules; i++){
      os << "***Rule " << i << endl;
      (fRules[i])->PrintRaw(os);
   }
   UInt_t nlinear = fLinNorm.size();
   //
   os << "NLinear= " << fLinTermOK.size() << endl;
   for (UInt_t i=0; i<nlinear; i++) {
      os << "***Linear " << i << endl;
      os << (fLinTermOK[i] ? 1:0) << " "
         << fLinCoefficients[i] << " "
         << fLinNorm[i] << " "
         << fLinDM[i] << " "
         << fLinDP[i] << " "
         << fLinImportance[i] << " " << endl;
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::ReadRaw( istream & istr )
{
   // read rule ensemble from stream
   UInt_t nrules;
   //
   string dummy;
   Int_t idum;
   //   Double_t ddum;
   //
   // First block is general stuff
   //
   istr >> dummy >> fImportanceCut;
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
   istr >> dummy >> nlinear;
   //
   fLinTermOK.resize(nlinear);
   fLinCoefficients.resize(nlinear);
   fLinNorm.resize(nlinear);
   fLinDP.resize(nlinear);
   fLinDM.resize(nlinear);
   fLinImportance.resize(nlinear);
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

//_______________________________________________________________________
void TMVA::RuleEnsemble::Copy( const RuleEnsemble & other )
{
   // copy function
   if(this != &other) {
      fRuleFit           = other.GetRuleFit();
      fMaxRuleDist       = other.GetMaxRuleDist();
      fOffset            = other.GetOffset();
      fRules             = other.GetRulesConst();
      fImportanceCut     = other.GetImportanceCut();
      fVarImportance     = other.GetVarImportance();
      fLearningModel     = other.GetLearningModel();
   }
}

//_______________________________________________________________________
Int_t TMVA::RuleEnsemble::CalcNRules( const TMVA::DecisionTree *dtree )
{
   // calculate the number of rules
   if (dtree==0) return 0;
   TMVA::Node *node = dtree->GetRoot();
   Int_t nendnodes = 0;
   FindNEndNodes( node, nendnodes );
   return 2*(nendnodes-1);
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::FindNEndNodes( const TMVA::Node *node, Int_t & nendnodes )
{
   // find the number of leaf nodes

   if (node==0) return;
   if (dynamic_cast<const TMVA::DecisionTreeNode*>(node)->GetSelector()<0) {
      ++nendnodes;
      return;
   }
   const TMVA::Node *nodeR = node->GetRight();
   const TMVA::Node *nodeL = node->GetLeft();
   FindNEndNodes( nodeR, nendnodes );
   FindNEndNodes( nodeL, nendnodes );
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::MakeRulesFromTree( const TMVA::DecisionTree *dtree )
{
   // create rules from the decsision tree structure 
   TMVA::Node *node = dtree->GetRoot();
   AddRule( node );
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::AddRule( const TMVA::Node *node )
{
   // add a new rule to the tree

   if (node==0) return;
   if (node->GetParent()==0) { // it's a root node, don't make a rule
      AddRule( node->GetRight() );
      AddRule( node->GetLeft() );
   } 
   else {
      TMVA::Rule *rule = MakeTheRule(node);
      if (rule) {
         fRules.push_back( rule );
         AddRule( node->GetRight() );
         AddRule( node->GetLeft() );
      } 
      else {
         fLogger << kFATAL << "<AddRule> - ERROR failed in creating a rule! BUG!" << Endl;
      }
   }
}

//_______________________________________________________________________
TMVA::Rule *TMVA::RuleEnsemble::MakeTheRule( const TMVA::Node *node )
{
   //
   // Make a Rule from a given Node.
   // The root node (ie no parent) does not generate a Rule.
   // The first node in a rule is always the root node => fNodes.size()>=2
   // Each node corresponds to a cut and the cut value is given by the parent node.
   //
   //
   if (node==0) {
      fLogger << kFATAL << "<MakeTheRule> input node is NULL. Should not happen. BUG!" << Endl;
      return 0;
   }

   if (node->GetParent()==0) { // a root node - ignore
      fLogger << kFATAL << "<MakeTheRule> a parent where we should not have a parent" << Endl;
      return 0;
   }
   //
   std::vector< const TMVA::Node * > nodeVec;
   std::vector< Int_t > cutTypes;
   const TMVA::Node *parent = node;
   const TMVA::Node *prev;
   nodeVec.push_back( node );
   //
   // Make list with the input node at the end
   //
   while (parent!=0) {
      parent = parent->GetParent();
      if (parent) {
         if (dynamic_cast<const TMVA::DecisionTreeNode*>(parent)->GetSelector()>=0)
            nodeVec.insert( nodeVec.begin(), parent );
      }
   }
   if (nodeVec.size()<2) {
      fLogger << kFATAL << "inconsistent Rule! -> BUG!!!" << Endl;
      return 0;
   }
   for (UInt_t i=1; i<nodeVec.size(); i++) {
      prev = nodeVec[i-1];
      if (prev->GetRight() == nodeVec[i]) {
         cutTypes.push_back(1);
      } 
      else if (prev->GetLeft() == nodeVec[i]) {
         cutTypes.push_back(-1);
      } 
      else {
         fLogger << kFATAL << "<MakeTheRule> BUG! Should not be here - an end-node before the end." << Endl;
         cutTypes.push_back(0);
      }
   }
   cutTypes.push_back(0); // end of 
   return new TMVA::Rule( this, nodeVec, cutTypes );
}

//_______________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::RuleEnsemble & rules )
{
   // ostream operator
   rules.Print( os );
   return os;
}
