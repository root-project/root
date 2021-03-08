// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Rtypes.h"

#if !defined(R__ALPHA) && !defined(R__SOLARIS) && !defined(R__ACC) && !defined(R__FBSD)
NamespaceImp(RooStats)
#endif

#include "TTree.h"
#include "TBranch.h"

#include "RooUniform.h"
#include "RooProdPdf.h"
#include "RooExtendPdf.h"
#include "RooSimultaneous.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/RooStatsUtils.h"
#include <typeinfo>

using namespace std;

namespace RooStats {

   RooStatsConfig& GetGlobalRooStatsConfig() {
      static RooStatsConfig theConfig;
      return theConfig;
   }

   Double_t AsimovSignificance(Double_t s, Double_t b, Double_t sigma_b ) {
   // Asimov significance
   // formula [10] and [20] from  https://www.pp.rhul.ac.uk/~cowan/stat/notes/medsigNote.pdf
      // case we have a sigma_b
      double sb2 = sigma_b*sigma_b;
      // formula below has a large error when sigma_b becomes zero
      // better to use the approximation for sigma_b=0 for very small values
      double r = sb2/b;
      if (r > 1.E-12) {
         double bpsb2 = b + sb2;
         double b2 = b*b;
         double spb = s+b; 
         double za2 = 2.*( (spb)* std::log( ( spb)*(bpsb2)/(b2+ spb*sb2) ) -
                           (b2/sb2) * std::log(1. + ( sb2 * s)/(b * bpsb2) ) );
         return sqrt(za2); 

      }
      // case when the background (b) is known
      double za2 = 2.*( (s+b) * std::log(1. + s/b) -s );
      return std::sqrt(za2); 
   }

   /// Use an offset in NLL calculations.
   void UseNLLOffset(bool on) {
      GetGlobalRooStatsConfig().useLikelihoodOffset = on;
   }

   /// Test of RooStats should by default offset NLL calculations.
   bool IsNLLOffset() {
      return GetGlobalRooStatsConfig().useLikelihoodOffset;
   }

   void FactorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints) {
   // utility function to factorize constraint terms from a pdf
   // (from G. Petrucciani)
      const std::type_info & id = typeid(pdf);
      if (id == typeid(RooProdPdf)) {
         RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
         RooArgList list(prod->pdfList());
         for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            FactorizePdf(observables, *pdfi, obsTerms, constraints);
         }
      } else if (id == typeid(RooExtendPdf)) {
         TIterator *iter = pdf.serverIterator();
         // extract underlying pdf which is extended; first server is the pdf; second server is the number of events variable
         RooAbsPdf *updf = dynamic_cast<RooAbsPdf *>(iter->Next());
         assert(updf != nullptr);
         delete iter;
         FactorizePdf(observables, *updf, obsTerms, constraints);
      } else if (id == typeid(RooSimultaneous)) {    //|| id == typeid(RooSimultaneousOpt)) {
         RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf);
         assert(sim != nullptr);
         RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().clone(sim->indexCat().GetName());
         for (int ic = 0, nc = cat->numBins((const char *)nullptr); ic < nc; ++ic) {
            cat->setBin(ic);
            RooAbsPdf* catPdf = sim->getPdf(cat->getCurrentLabel());
            // it is possible that a pdf is not defined for every category
            if (catPdf != nullptr) FactorizePdf(observables, *catPdf, obsTerms, constraints);
         }
         delete cat;
      } else if (pdf.dependsOn(observables)) {
         if (!obsTerms.contains(pdf)) obsTerms.add(pdf);
      } else {
         if (!constraints.contains(pdf)) constraints.add(pdf);
      }
   }


   void FactorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints) {
      // utility function to factorize constraint terms from a pdf
      // (from G. Petrucciani)
      if (!model.GetObservables() ) {
         oocoutE((TObject*)nullptr,InputArguments) << "RooStatsUtils::FactorizePdf - invalid input model: missing observables" << endl;
         return;
      }
      return FactorizePdf(*model.GetObservables(), pdf, obsTerms, constraints);
   }


   RooAbsPdf * MakeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name) {
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf
      RooArgList obsTerms, constraints;
      FactorizePdf(observables, pdf, obsTerms, constraints);
      if(constraints.getSize() == 0) {
         oocoutW((TObject *)nullptr, Eval) << "RooStatsUtils::MakeNuisancePdf - no constraints found on nuisance parameters in the input model" << endl;
         return nullptr;
      } 
      return new RooProdPdf(name,"", constraints);
   }

   RooAbsPdf * MakeNuisancePdf(const RooStats::ModelConfig &model, const char *name) {
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf
      if (!model.GetPdf() || !model.GetObservables() ) {
         oocoutE((TObject*)nullptr, InputArguments) << "RooStatsUtils::MakeNuisancePdf - invalid input model: missing pdf and/or observables" << endl;
         return nullptr;
      }
      return MakeNuisancePdf(*model.GetPdf(), *model.GetObservables(), name);
   }

   RooAbsPdf * StripConstraints(RooAbsPdf &pdf, const RooArgSet &observables) {
      const std::type_info & id = typeid(pdf);

      if (id == typeid(RooProdPdf)) {

         RooProdPdf *prod = dynamic_cast<RooProdPdf *>(&pdf);
         RooArgList list(prod->pdfList()); RooArgList newList;

         for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            RooAbsPdf *newPdfi = StripConstraints(*pdfi, observables);
            if(newPdfi != nullptr) newList.add(*newPdfi);
         }

         if(newList.getSize() == 0) return nullptr; // only constraints in product
         // return single component (no longer a product)
         else if(newList.getSize() == 1) return dynamic_cast<RooAbsPdf *>(newList.at(0)->clone(TString::Format("%s_unconstrained",
                                                                                                               newList.at(0)->GetName())));
         else return new RooProdPdf(TString::Format("%s_unconstrained", prod->GetName()).Data(),
            TString::Format("%s without constraints", prod->GetTitle()).Data(), newList);

      } else if (id == typeid(RooExtendPdf)) {

         TIterator *iter = pdf.serverIterator();
         // extract underlying pdf which is extended; first server is the pdf; second server is the number of events variable
         RooAbsPdf *uPdf = dynamic_cast<RooAbsPdf *>(iter->Next());
         RooAbsReal *extended_term = dynamic_cast<RooAbsReal *>(iter->Next());
         assert(uPdf != nullptr); assert(extended_term != nullptr); assert(iter->Next() == nullptr);
         delete iter;

         RooAbsPdf *newUPdf = StripConstraints(*uPdf, observables);
         if(newUPdf == nullptr) return nullptr; // only constraints in underlying pdf
         else return new RooExtendPdf(TString::Format("%s_unconstrained", pdf.GetName()).Data(),
            TString::Format("%s without constraints", pdf.GetTitle()).Data(), *newUPdf, *extended_term);

      } else if (id == typeid(RooSimultaneous)) {    //|| id == typeid(RooSimultaneousOpt)) {

         RooSimultaneous *sim  = dynamic_cast<RooSimultaneous *>(&pdf); assert(sim != nullptr);
         RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().Clone(); assert(cat != nullptr);
         RooArgList pdfList;

         for (int ic = 0, nc = cat->numBins((const char *)nullptr); ic < nc; ++ic) {
            cat->setBin(ic);
            RooAbsPdf* catPdf = sim->getPdf(cat->getCurrentLabel());
            RooAbsPdf* newPdf = nullptr;
            // it is possible that a pdf is not defined for every category
            if (catPdf != nullptr) newPdf = StripConstraints(*catPdf, observables);
            if (newPdf == nullptr) { delete cat; return nullptr; } // all channels must have observables
            pdfList.add(*newPdf);
         }

         return new RooSimultaneous(TString::Format("%s_unconstrained", sim->GetName()).Data(),
            TString::Format("%s without constraints", sim->GetTitle()).Data(), pdfList, *cat);

      } else if (pdf.dependsOn(observables)) {
         return (RooAbsPdf *) pdf.clone(TString::Format("%s_unconstrained", pdf.GetName()).Data());
      }

      return nullptr; // just  a constraint term
   }

   RooAbsPdf * MakeUnconstrainedPdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name) {
      // make a clone pdf without all constraint terms in a common pdf
      RooAbsPdf * unconstrainedPdf = StripConstraints(pdf, observables);
      if(!unconstrainedPdf) {
         oocoutE((TObject *)nullptr, InputArguments) << "RooStats::MakeUnconstrainedPdf - invalid observable list passed (observables not found in original pdf) or invalid pdf passed (without observables)" << endl;
         return nullptr;
      }
      if(name != nullptr) unconstrainedPdf->SetName(name);
      return unconstrainedPdf;
   }

   RooAbsPdf * MakeUnconstrainedPdf(const RooStats::ModelConfig &model, const char *name) {
      // make a clone pdf without all constraint terms in a common pdf
      if(!model.GetPdf() || !model.GetObservables()) {
         oocoutE((TObject *)nullptr, InputArguments) << "RooStatsUtils::MakeUnconstrainedPdf - invalid input model: missing pdf and/or observables" << endl;
         return nullptr;
      }
      return MakeUnconstrainedPdf(*model.GetPdf(), *model.GetObservables(), name);
   }

   // Helper class for GetAsTTree
   class BranchStore {
      public:
         std::map<TString, Double_t> fVarVals;
         double fInval;
         TTree *fTree;

         BranchStore(const vector <TString> &params = vector <TString>(), double _inval = -999.) : fTree(nullptr) {
            fInval = _inval;
            for(unsigned int i = 0;i<params.size();i++)
               fVarVals[params[i]] = _inval;
         }

         ~BranchStore() {
            if (fTree) {
               for(std::map<TString, Double_t>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
                  TBranch *br = fTree->GetBranch( it->first );
                  if (br) br->ResetAddress();
               }
            }
         }

         void AssignToTTree(TTree &myTree) {
            fTree = &myTree;
            for(std::map<TString, Double_t>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
               const TString& name = it->first;
               myTree.Branch( name, &fVarVals[name], TString::Format("%s/D", name.Data()));
            }
         }
         void ResetValues() {
            for(std::map<TString, Double_t>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
               const TString& name = it->first;
               fVarVals[name] = fInval;
            }
         }
   };

   BranchStore* CreateBranchStore(const RooDataSet& data) {
      if (data.numEntries() == 0) {
         return new BranchStore;
      }
      vector <TString> V;
      const RooArgSet* aset = data.get(0);
      RooAbsArg *arg(nullptr);
      TIterator *it = aset->createIterator();
      for(;(arg = dynamic_cast<RooAbsArg*>(it->Next()));) {
         RooRealVar *rvar = dynamic_cast<RooRealVar*>(arg);
         if (rvar == nullptr)
            continue;
         V.push_back(rvar->GetName());
         if (rvar->hasAsymError()) {
            V.push_back(TString::Format("%s_errlo", rvar->GetName()));
            V.push_back(TString::Format("%s_errhi", rvar->GetName()));
         }
         else if (rvar->hasError()) {
            V.push_back(TString::Format("%s_err", rvar->GetName()));
         }
      }
      delete it;
      return new BranchStore(V);
   }

   void FillTree(TTree &myTree, const RooDataSet &data) {
      BranchStore *bs = CreateBranchStore(data);
      bs->AssignToTTree(myTree);

      for(int entry = 0;entry<data.numEntries();entry++) {
         bs->ResetValues();
         const RooArgSet* aset = data.get(entry);
         RooAbsArg *arg(nullptr);
         RooLinkedListIter it = aset->iterator();
         for(;(arg = dynamic_cast<RooAbsArg*>(it.Next()));) {
            RooRealVar *rvar = dynamic_cast<RooRealVar*>(arg);
            if (rvar == nullptr)
               continue;
            bs->fVarVals[rvar->GetName()] = rvar->getValV();
            if (rvar->hasAsymError()) {
               bs->fVarVals[TString::Format("%s_errlo", rvar->GetName())] = rvar->getAsymErrorLo();
               bs->fVarVals[TString::Format("%s_errhi", rvar->GetName())] = rvar->getAsymErrorHi();
            }
            else if (rvar->hasError()) {
               bs->fVarVals[TString::Format("%s_err", rvar->GetName())] = rvar->getError();
            }
         }
         myTree.Fill();
      }

      delete bs;
   }

   TTree * GetAsTTree(TString name, TString desc, const RooDataSet& data) {
      TTree* myTree = new TTree(name, desc);
      FillTree(*myTree, data);
      return myTree;
   }


   // useful function to print in one line the content of a set with their values
   void PrintListContent(const RooArgList & l, std::ostream & os ) {
      bool first = true;
      os << "( ";
      for (int i = 0; i< l.getSize(); ++i) {
         if (first) {
            first=kFALSE ;
         } else {
            os << ", " ;
         }
         l[i].printName(os);
         os << " = ";
         l[i].printValue(os);
      }
      os << ")\n";
   }
}
