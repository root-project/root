// @(#)root/tmva $Id: RuleEnsemble.cxx,v 1.1 2006/10/09 15:55:02 brun Exp $
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
 *      MPI-KP Heidelberg, Germany                                                * 
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

//ClassImp(TMVA::RuleEnsemble)

//_______________________________________________________________________
TMVA::RuleEnsemble::RuleEnsemble( RuleFit *rf )
{
   fAverageRuleSigma = 0.4; // default value - used if only linear model is chosen
   fDoLinear = kTRUE;
   fLearningModel = kFull;
   fMaxRuleDist = 1e-3; // closest allowed 'distance' between two rules
   Initialize( rf );
}

//_______________________________________________________________________
TMVA::RuleEnsemble::~RuleEnsemble()
{
   for ( std::vector< TMVA::Rule *>::iterator itrRule = fRules.begin(); itrRule != fRules.end(); itrRule++ ) {
      delete *itrRule;
   }
   // NOTE: Should not delete the histos fLinPDFB/S since they are delete elsewhere
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::Initialize( RuleFit *rf )
{
   fAverageRuleSigma = 0.4; // default value - used if only linear model is chosen
   fRuleFit = rf;
   UInt_t nvars =  rf->GetInputVars()->size();
   fVarImportance.resize( nvars );
   fLinPDFB.resize( nvars,0 );
   fLinPDFS.resize( nvars,0 );
   fImportanceRef = 1.0;
   for (UInt_t i=0; i<nvars; i++) { // a priori all linear terms are equally valid
      fLinTermOK.push_back(kTRUE);
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::MakeModel()
{
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
   fOffset = 0.0;
   UInt_t nrules = fRules.size();
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(0.0);
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::SetCoefficients( const std::vector< Double_t > & v )
{
   UInt_t nrules = fRules.size();
   if (v.size()!=nrules) {
      std::cout << "--- RuleEnsemble::SetCoefficients() - BUG TRAP - input vector worng size! It is = " << v.size()
                << " when it should be = " << nrules << std::endl;
      exit(1);
   }
   for (UInt_t i=0; i<nrules; i++) {
      fRules[i]->SetCoefficient(v[i]);
   }
}

//_______________________________________________________________________
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

//_______________________________________________________________________
const std::vector<const TMVA::Event *> *TMVA::RuleEnsemble::GetTrainingEvents()  const
{ 
   return &(fRuleFit->GetTrainingEvents());
}
//_______________________________________________________________________
const std::vector< Int_t > *TMVA::RuleEnsemble::GetSubsampleEvents() const
{
   return &(fRuleFit->GetSubsampleEvents());
}
//_______________________________________________________________________
void TMVA::RuleEnsemble::GetSubsampleEvents(UInt_t sub, UInt_t & ibeg, UInt_t & iend) const
{
   fRuleFit->GetSubsampleEvents(sub,ibeg,iend);
}
//_______________________________________________________________________
const UInt_t TMVA::RuleEnsemble::GetNSubsamples() const
{
   return fRuleFit->GetNSubsamples();
}
//_______________________________________________________________________
const TMVA::Event * TMVA::RuleEnsemble::GetTrainingEvent(UInt_t i) const
{
   return fRuleFit->GetTrainingEvent(i);
}
//_______________________________________________________________________
const TMVA::Event * TMVA::RuleEnsemble::GetTrainingEvent(UInt_t i, UInt_t isub)  const
{
   return fRuleFit->GetTrainingEvent(i,isub);
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::SetRulesNCuts()
{
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
   std::cout << "--- RuleEnsemble : *** Removing simple rules ***" << std::endl;
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
   std::cout << "--- RuleEnsemble : Removed " << nrulesIn - nrulesOut << " out of " << nrulesIn<< " rules" << std::endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::RemoveSimilarRules()
{
   std::cout << "--- RuleEnsemble : *** Removing similar rules ***" << std::endl;
   std::cout << "--- RuleEnsemble :     Distance = " << fMaxRuleDist << std::endl;
   UInt_t nrulesIn = fRules.size();
   TMVA::Rule *first, *second;
   std::vector<bool> removeMe( nrulesIn,false );
   //
   Int_t nrem = 0;
   Int_t remind=-1;
   //   Int_t cutop;
   //
   Double_t r;
   //   TRandom rnd();

   for (UInt_t i=0; i<nrulesIn; i++) {
      if (!removeMe[i]) {
         first = fRules[i];
         for (UInt_t k=i+1; k<nrulesIn; k++) {
            if (!removeMe[k]) {
               second = fRules[k];
               //               std::cout<< "Comparing rules: " << i << " and " << k << std::endl;
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
   std::cout << "--- RuleEnsemble : Removed " << nrulesIn - nrulesOut << " out of " << nrulesIn<< " rules" << std::endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CleanupRules()
{
   UInt_t nrules   = fRules.size();
   if (nrules==0) return;
   std::cout << "--- RuleEnsemble : *** Removing Rules with relative importance < " << fImportanceCut << " ***" << std::endl;
   if (fImportanceCut<=0) return;
   //
   // Mark rules to be removed
   //
   TMVA::Rule *therule;
   Int_t ind=0;
   for ( UInt_t i=0; i<nrules; i++ ) {
      if (fRules[ind]->GetRelImportance()<fImportanceCut) {
         therule = fRules[ind];
         fRules.erase( std::vector<Rule *>::iterator(&fRules[ind]) );
         fRulesNCuts.erase( std::vector<Int_t>::iterator(&fRulesNCuts[ind]) ); // PHASE OUT
         delete therule;
         ind--;
      } else {
         //         std::cout << "retained" << std::endl;
      }
      ind++;
   }
   std::cout << "--- RuleEnsemble : Removed in total " << nrules-ind << " out of " << nrules << " rules" << std::endl;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CleanupLinear()
{
   UInt_t nlin   = fLinNorm.size();
   if (nlin==0) return;
   std::cout << "--- RuleEnsemble : *** Removing linear terms with relative importance < " << fImportanceCut << " ***" << std::endl;
   if (fImportanceCut<=0) return;
   //
   fLinTermOK.clear();
   for (UInt_t i=0; i<nlin; i++) {
      fLinTermOK.push_back( (fLinImportance[i]/fImportanceRef > fImportanceCut) );
   }
   //   std::cout << "--- RuleEnsemble : Removed in total " << nlin-fLinTermOK.size() << " out of " << nlin << " linear terms" << std::endl; TODO!!!
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CalcRuleSupport()
{
   Double_t seve, s,t,stot,ttot;
   Int_t indrule=0;
   Bool_t printme;
   stot = 0;
   ttot = 0;
   const std::vector<const TMVA::Event *> *events = GetTrainingEvents();
   Double_t nevent=static_cast<Double_t>(events->size());
   Double_t nrules=static_cast<Double_t>(fRules.size());
   if (nevent>0) {

      for ( std::vector< TMVA::Rule * >::iterator itrRule=fRules.begin(); itrRule!=fRules.end(); itrRule++ ) {
         s=0.0;
         printme = ((fRulesNCuts[indrule]==2) &&
                    ((*itrRule)->GetNumVars()==1) &&
                    ((*itrRule)->ContainsVariable(3)) );
         for ( std::vector<const TMVA::Event * >::const_iterator itrEvent=events->begin(); itrEvent!=events->end(); itrEvent++ ) {
            seve = (*itrRule)->EvalEvent( *(*itrEvent) );
            s += seve;
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
         indrule++;
      }
      fAverageSupport   = stot/nrules;
      fAverageRuleSigma = sqrt(fAverageSupport*(1.0-fAverageSupport));
      std::cout << "--- RuleEnsemble : Standard deviation of support = " << fAverageRuleSigma << std::endl;
      std::cout << "--- RuleEnsemble : Average rule support          = " << fAverageSupport   << std::endl;
   }
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::CalcImportance()
{
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
   Double_t maxImp=-1.0;
   UInt_t nvars = fLinCoefficients.size();
   fLinImportance.resize(nvars,0.0);
   if (!DoLinear()) return maxImp;
   //
   Double_t imp;
   for ( UInt_t i=0; i<nvars; i++ ) {
      imp = TMath::Abs(fLinCoefficients[i]);//*fLinNorm[i]/fAverageRuleSigma;
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
   UInt_t nvars  = fRuleFit->GetInputVars()->size();
   UInt_t nvarsUsed;
   Double_t rimpN;
   for ( UInt_t iv=0; iv<nvars; iv++ ) {
      fVarImportance[iv] = 0;
   }
   // rules
   if (DoRules()) {
      for ( UInt_t ind=0; ind<nrules; ind++ ) {
         rimp = fRules[ind]->GetImportance();
         nvarsUsed = fRules[ind]->GetNumVars();
         if (nvarsUsed<1) std::cout << "ERROR: No variables for importance calc!!!??? A BUG!" << std::endl;
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
   std::cout << "--- RuleEnsemble : Number of possible rules  = " << nrulesCheck << std::endl;
   std::cout << "--- RuleEnsemble : Number of generated rules = " << fRules.size() << std::endl;

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
      //
      Int_t type;
      for (UInt_t i=0; i<neve; i++) {
         val = vardata[v][i].first;
         type = vardata[v][i].second;
         lx = TMath::Min( fLinDP[v], TMath::Max( fLinDM[v], val ) );
         varsum[v] += lx;
         varsum2[v] += lx*lx;
         if (type==1) fLinPDFS[v]->Fill(lx);
         else         fLinPDFB[v]->Fill(lx);
      }
      stdl = sqrt( (varsum2[v] - (varsum[v]*varsum[v]/Double_t(neve)))/Double_t(neve-1) );
      fLinNorm[v] = ( stdl>0 ? fAverageRuleSigma/stdl : 1.0 ); // norm
   }
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfLinVar( UInt_t var, const TMVA::Event& e, Double_t & fs, Double_t & fb ) const
{
   Double_t val = EvalLinEvent( var, e, kFALSE );
   Int_t bin = fLinPDFS[var]->FindBin(val);
   fs = fLinPDFS[var]->GetBinContent(bin);
   fb = fLinPDFB[var]->GetBinContent(bin);
   if (fs+fb<=0) return 0;
   return fs/(fs+fb);
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfLinVar( const TMVA::Event& e, Int_t & nsig, Int_t & ntot  ) const
{
   //
   // This function returns Pr( y = 1 | x ) for the linear terms.
   //
   UInt_t nvars=fLinDP.size();

   Double_t fstot=0;
   Double_t fbtot=0;
   Double_t fs,fb;
   nsig = 0;
   ntot = nvars;
   for (UInt_t v=0; v<nvars; v++) {
      PdfLinVar(v,e,fs,fb);
      fstot += fs;
      fbtot += fb;
      if (fs>fb) nsig++;
   }
   if (nvars<1) return 0;
   return fstot/(fstot+fbtot);
   //   return Double_t(nsig)/Double_t(nvars);
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfRule( const TMVA::Rule *rule, const TMVA::Event& e ) const
{
   if (rule==0) {
      std::cout << "--- RuleEnsemble : WARNING in PdfRule - null ptr to Rule!" << std::endl;
      return 0;
   }
   Double_t ssb = rule->EvalEventSB(e);
   return ssb; //(ssb>0 ? (ssb>0.5 ? +1:-1) : 0);
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::PdfRule( const TMVA::Event& e, Int_t & nsig, Int_t & ntot  ) const
{
   //
   // This function returns Pr( y = 1 | x ) for rules.
   // The probability returned is normalized against the number of rules which are actually passed
   //
   Int_t nyp = 0;
   Int_t nyn = 0;
   Int_t nyz = 0;
   Double_t fr;
   Double_t sumf=0;
   //
   UInt_t nrules = fRules.size();
   for (UInt_t ir=0; ir<nrules; ir++) {
      fr = PdfRule( GetRulesConst(ir), e );
      //      std::cout << "PdfRule(ir="<< ir << ")::ssb = " << fr << std::endl;
      if      (fr>0.5) nyp += 1;
      else if (fr>0.0) nyn += 1;
      else             nyz += 1;
      sumf += fr;
   }

   nsig = nyp;
   ntot = nyp+nyn;
   //
   if (nyp+nyn==0) return 0.0;
   //   std::cout << "nrules = " << nrules << " => nyp/n/z = " << nyp << " : " << nyn << " : " << nyz << std::endl;
   //std::cout << "PdfRule:: no +1 or -1 events; nrules = " << nrules << std::endl;
   //   if ((nyn>0) && (nyp>0)) std::cout << "nrules = " << nrules << " => nyp/n/z = " << nyp << " : " << nyn << " : " << nyz << std::endl;
   //   return static_cast<Double_t>(nsig)/static_cast<Double_t>(ntot);
   return sumf/static_cast<Double_t>(ntot);
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::FStar( const TMVA::Event& e ) const
{
   //
   // We want to estimate F* = argmin Eyx( L(y,F(x) ), min wrt F(x)
   // F(x) = FL(x) + FR(x) , linear and rule part
   // 
   //
   Double_t p=0;
   Int_t nrs, nrt;
   Int_t nls, nlt;
   //   Int_t nt;
   Double_t pr=0;
   Double_t pl=0;

   // first calculate Pr(y=1|X) for rules and linear terms
   if (DoLinear()) pl = PdfLinVar(e, nls, nlt);
   if (DoRules())  pr = PdfRule(e, nrs, nrt);

//    if (fDoLinear) {
//       pr = PdfRule(e, nrs, nrt);
//       //      Double_t pl = 
//       pl = PdfLinVar(e, nls, nlt);
//       nt = nrt+nlt;
//       //      if (nt<1) p = 0.0;
//       //      else      p = Double_t(nls+nrs)/Double_t(nt);
//       p = pr+pl;
//       //      std::cout << "ns = " << nls+nrs << "  ; nt = " << nt << std::endl;
      
//       //      std::cout << "P(r), P(l) = " << pr << " , " << pl << std::endl;
//       //      p = pr*pl;
//       //      p = pr+pl - 2*pr*pl;
//       //      p = PdfRule(e)*PdfLinVar(e);
//    } else {
//       p = PdfRule(e, nrs, nrt);
//    }
   //   std::cout << " F* = " << 2.0*p-1.0 << std::endl;
   p = pr+pl;
   return 2.0*p-1.0;
}

//_____________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalEvent( const TMVA::Event& e ) const
{
   //
   Int_t nrules = fRules.size();
   Double_t rval=fOffset;
   Double_t linear=0;
   //
   // evaluate all rules
   // normally it should NOT use the normalized rules - the flag should be kFALSE
   //
   if (DoRules()) {
      for ( Int_t i=0; i<nrules; i++ ) {
         rval += fRules[i]->EvalEvent(e,kFALSE); //TRUE);
      }
   }
   //
   // Include linear part - the call below incorporates both coefficient and normalisation (fLinNorm)
   //
   if (DoLinear()) linear = EvalLinEvent(e);
   rval +=linear;

   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEvent( UInt_t vind, const TMVA::Event& e, Bool_t norm ) const
{
   Double_t rval=0;
   Double_t val = e.GetVal(vind);
   rval = TMath::Min( fLinDP[vind], TMath::Max( fLinDM[vind], val ) );
   if (norm) rval*=fLinNorm[vind];
   //   if (norm) {
//       std::cout << "rval = " << rval << "  norm = " << fLinNorm[vind]
//                 << "  dm = " << fLinDM[vind]
//                 << "  dp = " << fLinDP[vind] << std::endl;
//   }
   return rval;
}

//_______________________________________________________________________
Double_t TMVA::RuleEnsemble::EvalLinEvent( const TMVA::Event& e ) const
{
   Double_t rval=0;
   for (UInt_t v=0; v<fLinTermOK.size(); v++) {
      if (fLinTermOK[v])
         rval += fLinCoefficients[v]*EvalLinEvent( v, e, kTRUE ); // normalise with fLinNorm
   }
   return rval;
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::RuleStatistics()
{
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
   std::cout << "--- ===============================" << std::endl;
   std::cout << "---           RuleEnsemble " << std::endl;
   std::cout << "--- ===============================" << std::endl;

   Int_t ind;
   const Int_t nrules = fRules.size();
   const Int_t printN = 10; //nrules+1;
   Int_t maxL = 0;
   for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
      if ((*(fRuleFit->GetInputVars()))[iv].Length() > maxL) maxL = (*(fRuleFit->GetInputVars()))[iv].Length();
   }
   //
   os << "---" << std::endl;
   os << "--- Offset (a0)     = " << fOffset << std::endl;
   //
   os << "--- " << std::endl;
   os << "--- Variable importance:" << std::endl;
   for (UInt_t iv = 0; iv<fVarImportance.size(); iv++) {
      os << "--- " << setw(maxL) << (*(fRuleFit->GetInputVars()))[iv]
         << resetiosflags(ios::right) 
         << " : " << Form(" %3.3f",fVarImportance[iv]) << std::endl;
   }
   os << "---" << std::endl;
   if (DoLinear()) {
      os << "--- Results from linear terms:" << std::endl;
      os << "--- " << setw(maxL) << "Variable"
         << resetiosflags(ios::right) << " : "
         << setw(11) << " Weights"
         << resetiosflags(ios::right) << " : "
         << "Importance"
         << resetiosflags(ios::right)
         << std::endl;
      for ( UInt_t i=0; i<fLinNorm.size(); i++ ) {
         os << "--- " << setw(std::max(maxL,8)) << (*(fRuleFit->GetInputVars()))[i]
            << resetiosflags(ios::right) 
            << " : " << Form(" %10.3e",fLinCoefficients[i])
            << " : " << Form(" %3.3f",fLinImportance[i]/fImportanceRef) << std::endl;
      }
   } else {
      os << "--- Linear terms were disabled" << std::endl;
   }
   if (DoRules()) {
      os << "--- " << std::endl;
      os << "--- Number of rules          = " << nrules << std::endl;
      os << "--- Fraction of signal rules = " << fRuleFSig << std::endl;
      os << "--- " << std::endl;
      //
      // Print out all rules
      //
      std::list< std::pair<double,int> > sortedImp;
      for ( Int_t i=0; i<nrules; i++ ) {
         sortedImp.push_back( std::pair<double,int>( fRules[i]->GetImportance(),i ) );
      }
      sortedImp.sort();
      //
      os << "--- RuleEnsemble: Printing the first " << printN << " rules, ordered in importance." << std::endl;
      int pind=0;
      for ( std::list< std::pair<double,int> >::reverse_iterator itpair = sortedImp.rbegin();
            itpair != sortedImp.rend(); itpair++ ) {
         ind = itpair->second;
         //    if (pind==0) impref = 
         os << "---" << std::endl;
         os << "--- Rule #" << Form("%4d",pind+1) << std::endl;
         os << *fRules[ind];
         pind++;
         if (pind==printN) {
            os << "---" << std::endl;
            std::cout << "--- RuleEnsemble: Skipping the next " << nrules-printN-1 << " rules <<<" << std::endl;
            break;
         }
      }
   }
   os << "---" << std::endl;
}
//_______________________________________________________________________
void TMVA::RuleEnsemble::PrintRaw( ostream & os ) const
{
   // write rules to stream
   UInt_t nrules = fRules.size();
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
   for (UInt_t i=0; i<nrules; i++){
      istr >> dummy >> idum; // read line  "***Rule <ind>"
      fRules.push_back( new Rule() );
      (fRules.back())->ReadRaw(istr); // NEED TO SET INPUT VARS!
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
   fRuleFit           = other.GetRuleFit();
   fMaxRuleDist       = other.GetMaxRuleDist();
   fOffset            = other.GetOffset();
   fRules             = other.GetRulesConst();
   fImportanceCut     = other.GetImportanceCut();
   fVarImportance     = other.GetVarImportance();
   fLearningModel     = other.GetLearningModel();
}

//_______________________________________________________________________
Int_t TMVA::RuleEnsemble::CalcNRules( const TMVA::DecisionTree *dtree )
{
   if (dtree==0) return 0;
   TMVA::Node *node = dtree->GetRoot();
   Int_t nendnodes = 0;
   FindNEndNodes( node, nendnodes );
   return 2*(nendnodes-1);
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::FindNEndNodes( const TMVA::Node *node, Int_t & nendnodes )
{
   if (node==0) return;
   if (node->GetSelector()<0) {
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
   TMVA::Node *node = dtree->GetRoot();
   AddRule( node );
}

//_______________________________________________________________________
void TMVA::RuleEnsemble::AddRule( const TMVA::Node *node )
{
   if (node==0) return;
   if (node->GetParent()==0) { // it's a root node, don't make a rule
      AddRule( node->GetRight() );
      AddRule( node->GetLeft() );
   } else {
      TMVA::Rule *rule = MakeTheRule(node);
      if (rule) {
         fRules.push_back( rule );
         AddRule( node->GetRight() );
         AddRule( node->GetLeft() );
      } else {
         std::cout << "--- RuleEnsemble : AddRule() - ERROR failed in creating a rule! BUG!" << std::endl;
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
      std::cout << "--- RuleEnsemble::MakeTheRule() - ERROR - Input node is NULL. Should not happen. BUG!" << std::endl;
      return 0;
   }
//    if (node->GetSelector()<0) { // no selector - ignore
//       std::cout << "--- RuleEnsemble::MakeTheRule() - ERROR - selector = "
//                 << node->GetSelector() << std::endl;
//       return 0;
//    }
   if (node->GetParent()==0) { // a root node - ignore
      std::cout << "--- RuleEnsemble::MakeTheRule() - ERROR - a parent where we should not have a parent" << std::endl;
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
         if (parent->GetSelector()>=0)
            nodeVec.insert( nodeVec.begin(), parent );
      }
   }
   if (nodeVec.size()<2) {
      std::cout << "--- RuleEnsemble : ERROR - Inconsistent Rule! -> BUG!!!" << std::endl;
      return 0;
   }
   for (UInt_t i=1; i<nodeVec.size(); i++) {
      prev = nodeVec[i-1];
      if (prev->GetRight() == nodeVec[i]) {
         cutTypes.push_back(1);
      } else if (prev->GetLeft() == nodeVec[i]) {
         cutTypes.push_back(-1);
      } else {
         std::cout << "RuleEnsemble::MakeTheRule() - BUG! Should not be here - an end-node before the end." << std::endl;
         cutTypes.push_back(0);
      }
   }
   cutTypes.push_back(0); // end of 
   return new TMVA::Rule( this, nodeVec, cutTypes, fRuleFit->GetInputVars() );
}

//_______________________________________________________________________
ostream& TMVA::operator<< ( ostream& os, const TMVA::RuleEnsemble & rules )
{
   rules.Print( os );
   return os;
}
