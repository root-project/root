// @(#)root/tmva $Id: RuleFit.cxx,v 1.6 2006/11/20 15:35:28 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Fredrik Tegenfeldt, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Rule                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      A class describung a 'rule'                                               *
 *      Each internal node of a tree defines a rule from all the parental nodes.  *
 *      A rule with 0 or 1 nodes in the list is a root rule -> corresponds to a0. *
 *      Input: a decision tree (in the constructor)                               *
 *             its coefficient                                                    *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Fredrik Tegenfeldt <Fredrik.Tegenfeldt@cern.ch> - Iowa State U., USA      *
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

#include <iostream>
#include <algorithm>

#include "TKey.h"

#include "TMVA/RuleFit.h"
#include "TMVA/MethodRuleFit.h"

ClassImp(TMVA::RuleFit)

//_______________________________________________________________________
TMVA::RuleFit::RuleFit( const TMVA::MethodRuleFit *rfbase,
                        const std::vector< TMVA::DecisionTree *> & forest,
                        const std::vector<TMVA::Event *> & trainingEvents,
                        Double_t samplefrac )
   : fLogger( "RuleFit" )
{
   // constructor
   Initialise( rfbase, forest, trainingEvents, samplefrac );
}

//_______________________________________________________________________
TMVA::RuleFit::RuleFit()
   : fLogger( "RuleFit" )
{
   // default constructor
}

//_______________________________________________________________________
TMVA::RuleFit::~RuleFit()
{
   // destructor
}

//_______________________________________________________________________
void TMVA::RuleFit::Initialise(  const TMVA::MethodRuleFit *rfbase,
                                 const std::vector< TMVA::DecisionTree *> & forest,
                                 const std::vector< TMVA::Event *> & events,
                                 Double_t sampfrac )
{
   // initialize the parameters of the RuleFit method
   fMethodRuleFit = rfbase;
   std::vector< TMVA::DecisionTree *>::const_iterator itrDtree=forest.begin();
   for (; itrDtree!=forest.end(); ++itrDtree ) fForest.push_back( *itrDtree );

   ForestStatistics();

   SetTrainingEvents( events, sampfrac );

   // Initialize RuleEnsemble
   fRuleEnsemble.Initialize( this );

   // Make the model - Rule + Linear (if fDoLinear is true)
   fRuleEnsemble.MakeModel();

   // Initialize RuleFitParams
   fRuleFitParams.SetRuleFit( this );
}

//_______________________________________________________________________
void TMVA::RuleFit::Copy( const TMVA::RuleFit& other )
{
   // copy method
   if(this != &other) {
      fMethodRuleFit   = other.GetMethodRuleFit();
      fTrainingEvents  = other.GetTrainingEvents();
      fSubsampleEvents = other.GetSubsampleEvents();
   
      fForest       = other.GetForest();
      fRuleEnsemble = other.GetRuleEnsemble();
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::SetMsgType( EMsgType t )
{
   // set the current message type to that of mlog for this class and all other subtools
   fLogger.SetMinType(t);
   fRuleEnsemble.SetMsgType(t);
   fRuleFitParams.SetMsgType(t);
}
//_______________________________________________________________________
void TMVA::RuleFit::ForestStatistics()
{
   // summary of statistics of all trees
   // * end-nodes: average and spread
   UInt_t ntrees = fForest.size();
   if (ntrees==0) return;
   Double_t nt   = Double_t(ntrees);
   const TMVA::DecisionTree *tree;
   Double_t sumn2 = 0;
   Double_t sumn  = 0;
   Double_t nd;
   for (UInt_t i=0; i<ntrees; i++) {
      tree = fForest[i];
      nd = Double_t(tree->GetNNodes());
      sumn  += nd;
      sumn2 += nd*nd;
   }
   Double_t sig = TMath::Sqrt((sumn2 - (sumn*sumn/nt))/(nt-1));
   fLogger << kVERBOSE << "nodes in trees: average & std dev = " << sumn/nt << " , " << sig << Endl;
}

//_______________________________________________________________________
void TMVA::RuleFit::FitCoefficients()
{
   //
   // Fit the coefficients for the rule ensemble
   //
   fLogger << kVERBOSE << "fitting rule/linear terms" << Endl;
   fRuleFitParams.MakeGDPath();
}

//_______________________________________________________________________
void TMVA::RuleFit::CalcImportance()
{
   // calculates the importance of each rule

   fLogger << kVERBOSE << "calculating importance" << Endl;
   fRuleEnsemble.CalcImportance();
   fRuleEnsemble.CleanupRules();
   fRuleEnsemble.CleanupLinear();
   fRuleEnsemble.CalcVarImportance();
   fLogger << kVERBOSE << "filling rule statistics" << Endl;
   fRuleEnsemble.RuleStatistics();
   fLogger << kVERBOSE << "done filling rule statistics" << Endl;
}

//_______________________________________________________________________
Double_t TMVA::RuleFit::EvalEvent( const TMVA::Event& e )
{
   // evaluate single event

   return fRuleEnsemble.EvalEvent( e );
}

//_______________________________________________________________________
void TMVA::RuleFit::SetTrainingEvents( const std::vector<TMVA::Event *>& el, Double_t sampfrac )
{
   // set the training events randomly

   UInt_t neve = el.size();
   if (neve==0) fLogger << kWARNING << "an empty sample of training events was given" << Endl;

   // copy vector
   fTrainingEvents.clear();
   for (UInt_t i=0; i<el.size(); i++) {
      fTrainingEvents.push_back(static_cast< const TMVA::Event *>(el[i]));
   }

   // Re-shuffle the vector, ie, recreate it in a random order
   std::random_shuffle( fTrainingEvents.begin(), fTrainingEvents.end() );

   // Divide into subsamples
   Int_t istep = static_cast<Int_t>(neve*sampfrac);
   // Any change to nsub -> see also MethodRuleFit::MakeForest()
   Int_t nsub  = static_cast<Int_t>(1.0/sampfrac);

   for (Int_t s=0; s<nsub; s++) {
      fSubsampleEvents.push_back( s*istep );
   }
   fSubsampleEvents.push_back( neve ); // last index is the total length
   fLogger << kINFO << "created " << nsub << " training samples of in total "
           << fTrainingEvents.size() << " events" << Endl;
}

//_______________________________________________________________________
void TMVA::RuleFit::GetSubsampleEvents(Int_t sub, UInt_t& ibeg, UInt_t& iend) const
{
   // get the events for subsample sub

   Int_t nsub = GetNSubsamples();
   if (nsub==0) {
      fLogger << kFATAL << "<GetSubsampleEvents> - wrong size, not properly initialised! BUG!!!" << Endl;
   }
   ibeg = 0;
   iend = 0;
   if (sub<0) {
      ibeg = 0;
      iend = fTrainingEvents.size() - 1;
   }
   if (sub<nsub) {
      ibeg = fSubsampleEvents[sub];
      iend = fSubsampleEvents[sub+1]-1;
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::FillCut(TH2F* h2,const TMVA::Rule *rule,Int_t vind)
{
   // Fill cut

   if (rule==0) return;
   if (h2==0) return;
   //
   Double_t rmin,  rmax;
   Bool_t   dormin,dormax;
   Bool_t ruleHasVar = rule->GetRuleCut()->GetCutRange(vind,rmin,rmax,dormin,dormax);
   if (!ruleHasVar) return;
   //
   Int_t firstbin = h2->GetBin(1,1,1);
   Int_t lastbin = h2->GetBin(h2->GetNbinsX(),1,1);
   Int_t binmin=(dormin ? h2->FindBin(rmin,0.5):firstbin);
   Int_t binmax=(dormax ? h2->FindBin(rmax,0.5):lastbin);
   Int_t fbin;
   Double_t xbinw = h2->GetBinWidth(firstbin);
   Double_t fbmin = h2->GetBinLowEdge(binmin-firstbin+1);
   Double_t lbmax = h2->GetBinLowEdge(binmax-firstbin+1)+xbinw;
   Double_t fbfrac = (dormin ? ((fbmin+xbinw-rmin)/xbinw):1.0);
   Double_t lbfrac = (dormax ? ((rmax-lbmax+xbinw)/xbinw):1.0);
   Double_t f;
   Double_t xc;

   for (Int_t bin = binmin; bin<binmax+1; bin++) {
      fbin = bin-firstbin+1;
      if (bin==binmin) {
         f = fbfrac;
      } else if (bin==binmax) {
         f = lbfrac;
      } else {
         f = 1.0;
      }
      xc = h2->GetBinCenter(fbin);
//       Double_t coef = rule->GetCoefficient();
//       Double_t rr   = (ruleHasVar ? 1.0:0.0);
//       Double_t supp = rule->GetSupport();
      Double_t imp  = rule->GetImportance();
      //
      //      h2->Fill(xc,0.5,TMath::Abs(coef)*TMath::Abs(rr-supp)*f);
      h2->Fill(xc,0.5,TMath::Abs(imp)*f);
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::FillLin(TH2F* h2,Int_t vind)
{
   // fill lin

   if (h2==0) return;
   if (!fRuleEnsemble.DoLinear()) return;
   //
   Int_t firstbin = 1;
   Int_t lastbin = h2->GetNbinsX();
   Double_t xc;
   Double_t imp = fRuleEnsemble.GetLinImportance(vind);
   for (Int_t bin = firstbin; bin<lastbin+1; bin++) {
      xc = h2->GetBinCenter(bin);
      h2->Fill(xc,0.5,imp);
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::FillCorr(TH2F* h2,const TMVA::Rule *rule,Int_t vx, Int_t vy)
{
   // fill correlation
   if (rule==0) return;
   if (h2==0) return;
   Double_t ruleimp  = rule->GetImportance();
   if (!(ruleimp>0)) return;
   if (ruleimp<fRuleEnsemble.GetImportanceCut()) return;
   //
   Double_t rxmin,   rxmax,   rymin,   rymax;
   Bool_t   dorxmin, dorxmax, dorymin, dorymax;
   //
   // Get range in rule for X and Y
   //
   Bool_t ruleHasVarX = rule->GetRuleCut()->GetCutRange(vx,rxmin,rxmax,dorxmin,dorxmax);
   Bool_t ruleHasVarY = rule->GetRuleCut()->GetCutRange(vy,rymin,rymax,dorymin,dorymax);
   if (!(ruleHasVarX || ruleHasVarY)) return;
   // min max of varX and varY in hist
   Double_t vxmin = (dorxmin ? rxmin:h2->GetXaxis()->GetXmin());
   Double_t vxmax = (dorxmax ? rxmax:h2->GetXaxis()->GetXmax());
   Double_t vymin = (dorymin ? rymin:h2->GetYaxis()->GetXmin());
   Double_t vymax = (dorymax ? rymax:h2->GetYaxis()->GetXmax());
   // min max bin in X and Y
   Int_t binxmin= h2->GetXaxis()->FindBin(vxmin);
   Int_t binxmax= h2->GetXaxis()->FindBin(vxmax);
   Int_t binymin= h2->GetYaxis()->FindBin(vymin);
   Int_t binymax= h2->GetYaxis()->FindBin(vymax);
   // bin widths
   Double_t xbinw = h2->GetXaxis()->GetBinWidth(binxmin);
   Double_t ybinw = h2->GetYaxis()->GetBinWidth(binxmin);
   Double_t xbinmin = h2->GetXaxis()->GetBinLowEdge(binxmin);
   Double_t xbinmax = h2->GetXaxis()->GetBinLowEdge(binxmax)+xbinw;
   Double_t ybinmin = h2->GetYaxis()->GetBinLowEdge(binymin);
   Double_t ybinmax = h2->GetYaxis()->GetBinLowEdge(binymax)+ybinw;
   // fraction of edges
   Double_t fxbinmin = (dorxmin ? ((xbinmin+xbinw-vxmin)/xbinw):1.0);
   Double_t fxbinmax = (dorxmax ? ((vxmax-xbinmax+xbinw)/xbinw):1.0);
   Double_t fybinmin = (dorymin ? ((ybinmin+ybinw-vymin)/ybinw):1.0);
   Double_t fybinmax = (dorymax ? ((vymax-ybinmax+ybinw)/ybinw):1.0);
   //
   Double_t fx,fy;
   Double_t xc,yc;
   // fill histo
   for (Int_t binx = binxmin; binx<binxmax+1; binx++) {
      if (binx==binxmin) {
         fx = fxbinmin;
      } else if (binx==binxmax) {
         fx = fxbinmax;
      } else {
         fx = 1.0;
      }
      xc = h2->GetXaxis()->GetBinCenter(binx);
      for (Int_t biny = binymin; biny<binymax+1; biny++) {
         if (biny==binymin) {
            fy = fybinmin;
         } else if (biny==binymax) {
            fy = fybinmax;
         } else {
            fy = 1.0;
         }
         yc = h2->GetYaxis()->GetBinCenter(biny);
         h2->Fill(xc,yc,TMath::Abs(ruleimp)*fx*fy);
      }
   }
}

//_______________________________________________________________________
void TMVA::RuleFit::FillVisHistCut(const Rule * rule, std::vector<TH2F *> & hlist)
{
   // help routine to MakeVisHists() - fills for all variables
   //   if (rule==0) return;
   Int_t nhists = hlist.size();
   Int_t nvar   = fMethodRuleFit->GetNvar();
   if (nhists!=nvar) fLogger << kFATAL << "BUG TRAP: number of hists is not equal the number of variables!" << Endl;
   //
   std::vector<Int_t> vindex;
   TString hstr;
   // not a nice way to do a check...
   for (Int_t ih=0; ih<nhists; ih++) {
      hstr = hlist[ih]->GetTitle();
      for (Int_t iv=0; iv<nvar; iv++) {
         if (fMethodRuleFit->GetInputExp(iv) == hstr)
            vindex.push_back(iv);
      }
   }
   //
   for (Int_t iv=0; iv<nvar; iv++) {
      if (rule) {
         if (rule->ContainsVariable(vindex[iv])) {
            FillCut(hlist[iv],rule,vindex[iv]);
         }
      }
      FillLin(hlist[iv],vindex[iv]);
   }
}
//_______________________________________________________________________
void TMVA::RuleFit::FillVisHistCorr(const Rule * rule, std::vector<TH2F *> & hlist)
{
   // help routine to MakeVisHists() - fills for all correlation plots
   if (rule==0) return;
   Double_t ruleimp  = rule->GetImportance();
   if (!(ruleimp>0)) return;
   if (ruleimp<fRuleEnsemble.GetImportanceCut()) return;
   //
   Int_t nhists = hlist.size();
   Int_t nvar   = fMethodRuleFit->GetNvar();
   Int_t ncorr  = (nvar*(nvar+1)/2)-nvar;
   if (nhists!=ncorr) fLogger << kERROR << "BUG TRAP: number of corr hists is not correct! ncorr = "
                              << ncorr << " nvar = " << nvar << Endl;
   //
   std::vector< std::pair<Int_t,Int_t> > vindex;
   TString hstr, var1, var2;
   Int_t iv1=0,iv2=0;
   // not a nice way to do a check...
   for (Int_t ih=0; ih<nhists; ih++) {
      hstr = hlist[ih]->GetName();
      if (GetCorrVars( hstr, var1, var2 )) {
         iv1 = fMethodRuleFit->Data().FindVar( var1 );
         iv2 = fMethodRuleFit->Data().FindVar( var2 );
         vindex.push_back( std::pair<Int_t,Int_t>(iv2,iv1) ); // pair X, Y
      } else {
         fLogger << kERROR << "BUG TRAP: should not be here - failed getting var1 and var2" << Endl;
      }
   }
   //
   for (Int_t ih=0; ih<nhists; ih++) {
      if ( (rule->ContainsVariable(vindex[ih].first)) ||
           (rule->ContainsVariable(vindex[ih].second)) ) {
         FillCorr(hlist[ih],rule,vindex[ih].first,vindex[ih].second);
      }
   }
}
//_______________________________________________________________________
Bool_t TMVA::RuleFit::GetCorrVars(TString & title, TString & var1, TString & var2)
{
   // get first and second variables from title
   var1="";
   var2="";
   if(!title.BeginsWith("scat_")) return kFALSE;

   TString titleCopy = title(5,title.Length());
   if(titleCopy.Index("_RF2D")>=0) titleCopy.Remove(titleCopy.Index("_RF2D"));

   Int_t splitPos = titleCopy.Index("_vs_");
   if(splitPos>=0) { // there is a _vs_ in the string
      var1 = titleCopy(0,splitPos);
      var2 = titleCopy(splitPos+4, titleCopy.Length());
      return kTRUE;
   } else {
      var1 = titleCopy;
      return kFALSE;
   }
}
//_______________________________________________________________________
void TMVA::RuleFit::MakeVisHists()
{
   // this will create histograms visualizing the rule ensemble

   const TString directories[3] = { "InputVariables_NoTransform",
                                    "InputVariables_DecorrTransform",
                                    "InputVariables_PCATransform" };

   const TString corrDirName = "CorrelationPlots";   
   //   const TString outfname[TMVAGlob::kNumOfMethods] = { "variables",
   //                                                       "variables_decorr",
   //                                                       "variables_pca" };
   
   TDirectory* localDir = fMethodRuleFit->Data().BaseRootDir();
   TDirectory* methodDir = fMethodRuleFit->GetMethodBaseDir();
   TDirectory* varDir = 0;
   TDirectory* corrDir=0;
   TString varDirName;
   //
   Bool_t done=(localDir==0);
   Int_t type=0;
   if (done) fLogger << kWARNING << "no basedir - BUG??" << Endl;
   while (!done) {
      varDir = (TDirectory*)localDir->Get( directories[type] );
      type++;
      done = ((varDir!=0) || (type>2));
   }
   if (varDir==0) {
      fLogger << kWARNING << "no input variable directory found - bug?" << Endl;
      return;
   }
   if (methodDir==0) {
      fLogger << kWARNING << "no rulefit method directory found - bug?" << Endl;
      return;
   }
   varDirName = varDir->GetName();
   varDir->cd();
   //
   // get correlation plot directory
   corrDir = (TDirectory *)varDir->Get(corrDirName);
   if (corrDir==0) fLogger << kWARNING << "No correlation directory found : " << corrDirName << Endl;
   // how many plots are in the var directory?
   Int_t noPlots = ((varDir->GetListOfKeys())->GetEntries()) / 2;
   fLogger << kDEBUG << "got number of plots = " << noPlots << Endl;
 
   // loop over all objects in directory
   std::vector<TH2F *> h1Vector;
   std::vector<TH2F *> h2CorrVector;
   TIter next(varDir->GetListOfKeys());
   TKey *key;
   while ((key = (TKey*)next())) {
      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH1F")) continue;
      TH1F *sig = (TH1F*)key->ReadObj();
      TString hname= sig->GetName();
      fLogger << kDEBUG << "got histogram : " << hname << Endl;

      // check for all signal histograms
      if (hname.Contains("__S")){ // found a new signal plot
         TString htitle = sig->GetTitle();
         htitle.ReplaceAll("signal","");
         TString newname = hname;
         newname.ReplaceAll("__S","__RF");
         //         TString newtitle = "RuleFit: " + htitle;
         methodDir->cd();
         TH2F *newhist = new TH2F(newname,htitle,sig->GetNbinsX(),sig->GetXaxis()->GetXmin(),sig->GetXaxis()->GetXmax(),
                                  1,sig->GetYaxis()->GetXmin(),sig->GetYaxis()->GetXmax());
         newhist->Write();
         varDir->cd();
         h1Vector.push_back( newhist );
      }
   }
   //
   corrDir->cd();
   TString var1,var2;
   TIter nextCorr(corrDir->GetListOfKeys());
   while ((key = (TKey*)nextCorr())) {
      // make sure, that we only look at histograms
      TClass *cl = gROOT->GetClass(key->GetClassName());
      if (!cl->InheritsFrom("TH2F")) continue;
      TH2F *sig = (TH2F*)key->ReadObj();
      TString hname= sig->GetName();

      // check for all signal histograms
      if ((hname.Contains("scat_")) && (hname.Contains("_sig_"))) {
         fLogger << kDEBUG << "got histogram : " << hname << Endl;
         TString htitle = sig->GetTitle();
         htitle.ReplaceAll("(signal)_","");
         TString newname = hname;
         newname.ReplaceAll("_sig_","_RF2D_");
         //         TString newtitle = "RuleFit: " + htitle;
         methodDir->cd();
         const Int_t rebin=2;
         TH2F *newhist = new TH2F(newname,htitle,
                                  sig->GetNbinsX()/rebin,sig->GetXaxis()->GetXmin(),sig->GetXaxis()->GetXmax(),
                                  sig->GetNbinsY()/rebin,sig->GetYaxis()->GetXmin(),sig->GetYaxis()->GetXmax());
         newhist->Write();
         if (GetCorrVars( newname, var1, var2 )) {
            Int_t iv1 = fMethodRuleFit->Data().FindVar(var1);
            Int_t iv2 = fMethodRuleFit->Data().FindVar(var2);
            if (iv1<0) {
               sig->GetYaxis()->SetTitle(var1);
            } else {
               sig->GetYaxis()->SetTitle(fMethodRuleFit->GetInputExp(iv1));
            }
            if (iv2<0) {
               sig->GetXaxis()->SetTitle(var2);
            } else {
               sig->GetXaxis()->SetTitle(fMethodRuleFit->GetInputExp(iv2));
            }
         }
         corrDir->cd();
         h2CorrVector.push_back( newhist );
      }
   }

   varDir->cd();
   // fill rules
   UInt_t nrules = fRuleEnsemble.GetNRules();
   const Rule *rule;
   for (UInt_t i=0; i<nrules; i++) {
      rule = fRuleEnsemble.GetRulesConst(i);
      FillVisHistCut(rule, h1Vector);
   }
   // fill linear terms
   FillVisHistCut(0, h1Vector);

   corrDir->cd();
   // fill rules
   for (UInt_t i=0; i<nrules; i++) {
      rule = fRuleEnsemble.GetRulesConst(i);
      FillVisHistCorr(rule, h2CorrVector);
   }
}
