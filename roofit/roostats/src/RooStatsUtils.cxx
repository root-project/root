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

#include "RooArgSet.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooUniform.h"
#include "RooProdPdf.h"
#include "RooExtendPdf.h"
#include "RooSimultaneous.h"
#include "RooStats/ModelConfig.h"
#include "RooStats/RooStatsUtils.h"

using namespace std;


namespace {
     template<class listT, class stringT> void getParameterNames(const listT* l,std::vector<stringT>& names){
       // extract the parameter names from a list
       if(!l) return;
       for (auto const *obj : *l) {
         names.push_back(obj->GetName());
       }
     }
     void getArgs(RooWorkspace* ws, const std::vector<TString> names, RooArgSet& args){
       for(const auto& p:names){
         RooRealVar* v = ws->var(p.Data());
         if(v){
           args.add(*v);
         }
       }
     }
   }

namespace RooStats {

   RooStatsConfig& GetGlobalRooStatsConfig() {
      static RooStatsConfig theConfig;
      return theConfig;
   }

   double AsimovSignificance(double s, double b, double sigma_b ) {
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
      if (auto prod = dynamic_cast<RooProdPdf *>(&pdf)) {
         RooArgList list(prod->pdfList());
         for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            FactorizePdf(observables, *pdfi, obsTerms, constraints);
         }
      } else if (dynamic_cast<RooExtendPdf *>(&pdf)) {
         // extract underlying pdf which is extended; first server is the pdf; second server is the number of events variable
         auto iter = pdf.servers().begin();
         assert(iter != pdf.servers().end());
         assert(dynamic_cast<RooAbsPdf*>(*iter));
         FactorizePdf(observables, static_cast<RooAbsPdf&>(**iter), obsTerms, constraints);
      } else if (auto sim = dynamic_cast<RooSimultaneous *>(&pdf)) {  //|| dynamic_cast<RooSimultaneousOpt>(&pdf)) {
         assert(sim != 0);
         RooAbsCategoryLValue *cat = (RooAbsCategoryLValue *) sim->indexCat().clone(sim->indexCat().GetName());
         for (int ic = 0, nc = cat->numBins((const char *)0); ic < nc; ++ic) {
            cat->setBin(ic);
            RooAbsPdf* catPdf = sim->getPdf(cat->getCurrentLabel());
            // it is possible that a pdf is not defined for every category
            if (catPdf != 0) FactorizePdf(observables, *catPdf, obsTerms, constraints);
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
         oocoutE(nullptr,InputArguments) << "RooStatsUtils::FactorizePdf - invalid input model: missing observables" << endl;
         return;
      }
      return FactorizePdf(*model.GetObservables(), pdf, obsTerms, constraints);
   }


   RooAbsPdf * MakeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name) {
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf
      RooArgList obsTerms, constraints;
      FactorizePdf(observables, pdf, obsTerms, constraints);
      if(constraints.empty()) {
         oocoutW(nullptr, Eval) << "RooStatsUtils::MakeNuisancePdf - no constraints found on nuisance parameters in the input model" << endl;
         return 0;
      }
      return new RooProdPdf(name,"", constraints);
   }

   RooAbsPdf * MakeNuisancePdf(const RooStats::ModelConfig &model, const char *name) {
      // make a nuisance pdf by factorizing out all constraint terms in a common pdf
      if (!model.GetPdf() || !model.GetObservables() ) {
         oocoutE(nullptr, InputArguments) << "RooStatsUtils::MakeNuisancePdf - invalid input model: missing pdf and/or observables" << endl;
         return 0;
      }
      return MakeNuisancePdf(*model.GetPdf(), *model.GetObservables(), name);
   }

   RooAbsPdf * StripConstraints(RooAbsPdf &pdf, const RooArgSet &observables) {

      if (auto prod = dynamic_cast<RooProdPdf *>(&pdf)) {

         RooArgList list(prod->pdfList()); RooArgList newList;

         for (int i = 0, n = list.getSize(); i < n; ++i) {
            RooAbsPdf *pdfi = (RooAbsPdf *) list.at(i);
            RooAbsPdf *newPdfi = StripConstraints(*pdfi, observables);
            if(newPdfi != nullptr) newList.add(*newPdfi);
         }

         if(newList.empty()) return nullptr; // only constraints in product
         // return single component (no longer a product)
         else if(newList.getSize() == 1) return dynamic_cast<RooAbsPdf *>(newList.at(0)->clone(TString::Format("%s_unconstrained",
                                                                                                               newList.at(0)->GetName())));
         else return new RooProdPdf(TString::Format("%s_unconstrained", prod->GetName()).Data(),
            TString::Format("%s without constraints", prod->GetTitle()).Data(), newList);

      } else if (dynamic_cast<RooExtendPdf*>(&pdf)) {

         auto iter = pdf.servers().begin();
         // extract underlying pdf which is extended; first server is the pdf; second server is the number of events variable
         auto uPdf = dynamic_cast<RooAbsPdf *>(*(iter++));
         auto extended_term = dynamic_cast<RooAbsReal *>(*(iter++));
         assert(uPdf != nullptr);
         assert(extended_term != nullptr);
         assert(iter == pdf.servers().end());

         RooAbsPdf *newUPdf = StripConstraints(*uPdf, observables);
         if(newUPdf == nullptr) return nullptr; // only constraints in underlying pdf
         else return new RooExtendPdf(TString::Format("%s_unconstrained", pdf.GetName()).Data(),
            TString::Format("%s without constraints", pdf.GetTitle()).Data(), *newUPdf, *extended_term);

      } else if (auto sim = dynamic_cast<RooSimultaneous *>(&pdf)) {  //|| dynamic_cast<RooSimultaneousOpt *>(&pdf)) {

         assert(sim != nullptr);
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
         oocoutE(nullptr, InputArguments) << "RooStats::MakeUnconstrainedPdf - invalid observable list passed (observables not found in original pdf) or invalid pdf passed (without observables)" << endl;
         return nullptr;
      }
      if(name != nullptr) unconstrainedPdf->SetName(name);
      return unconstrainedPdf;
   }

   RooAbsPdf * MakeUnconstrainedPdf(const RooStats::ModelConfig &model, const char *name) {
      // make a clone pdf without all constraint terms in a common pdf
      if(!model.GetPdf() || !model.GetObservables()) {
         oocoutE(nullptr, InputArguments) << "RooStatsUtils::MakeUnconstrainedPdf - invalid input model: missing pdf and/or observables" << endl;
         return nullptr;
      }
      return MakeUnconstrainedPdf(*model.GetPdf(), *model.GetObservables(), name);
   }

   // Helper class for GetAsTTree
   class BranchStore {
      public:
         std::map<TString, double> fVarVals;
         double fInval;
         TTree *fTree;

         BranchStore(const vector <TString> &params = vector <TString>(), double _inval = -999.) : fTree(0) {
            fInval = _inval;
            for(unsigned int i = 0;i<params.size();i++)
               fVarVals[params[i]] = _inval;
         }

         ~BranchStore() {
            if (fTree) {
               for(std::map<TString, double>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
                  TBranch *br = fTree->GetBranch( it->first );
                  if (br) br->ResetAddress();
               }
            }
         }

         void AssignToTTree(TTree &myTree) {
            fTree = &myTree;
            for(std::map<TString, double>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
               const TString& name = it->first;
               myTree.Branch( name, &fVarVals[name], TString::Format("%s/D", name.Data()));
            }
         }
         void ResetValues() {
            for(std::map<TString, double>::iterator it = fVarVals.begin();it!=fVarVals.end();++it) {
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
      for (auto *rvar : dynamic_range_cast<RooRealVar *>(* data.get(0))) {
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
      return new BranchStore(V);
   }

   void FillTree(TTree &myTree, const RooDataSet &data) {
      BranchStore *bs = CreateBranchStore(data);
      bs->AssignToTTree(myTree);

      for(int entry = 0;entry<data.numEntries();entry++) {
         bs->ResetValues(); 
         for (auto const *rvar : dynamic_range_cast<RooRealVar *>(*data.get(entry))) {
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
            first=false ;
         } else {
            os << ", " ;
         }
         l[i].printName(os);
         os << " = ";
         l[i].printValue(os);
      }
      os << ")\n";
   }

   // clone a workspace, copying all needed components and discarding all others
   // start off with the old workspace
   RooWorkspace* MakeCleanWorkspace(RooWorkspace *oldWS, const char *newName,
                                   bool copySnapshots, const char *mcname,
                                   const char *newmcname) {
      auto objects = oldWS->allGenericObjects();
      RooStats::ModelConfig *oldMC =
          dynamic_cast<RooStats::ModelConfig *>(oldWS->obj(mcname));
      auto data = oldWS->allData();
      for (auto it : objects) {
        if (!oldMC) {
          oldMC = dynamic_cast<RooStats::ModelConfig *>(it);
        }
      }
      if (!oldMC)
        throw std::runtime_error("unable to retrieve ModelConfig");

      RooAbsPdf *origPdf = oldMC->GetPdf();

      // start off with the old modelconfig
      std::vector<TString> poilist;
      std::vector<TString> nplist;
      std::vector<TString> obslist;
      std::vector<TString> globobslist;
      RooAbsPdf *pdf = nullptr;
      if (oldMC) {
        pdf = oldMC->GetPdf();
        ::getParameterNames(oldMC->GetParametersOfInterest(), poilist);
        ::getParameterNames(oldMC->GetNuisanceParameters(), nplist);
        ::getParameterNames(oldMC->GetObservables(), obslist);
        ::getParameterNames(oldMC->GetGlobalObservables(), globobslist);
      }
      if (!pdf) {
        if (origPdf)
          pdf = origPdf;
      }
      if (!pdf) {
        return nullptr;
      }

      // create them anew
      RooWorkspace *newWS = new RooWorkspace(newName ? newName : oldWS->GetName());
      newWS->autoImportClassCode(true);
      RooStats::ModelConfig *newMC = new RooStats::ModelConfig(newmcname, newWS);

      // Copy snapshots
      if (copySnapshots) {
        for (auto *snap : oldWS->getSnapshots()) {
          RooArgSet *snapClone = static_cast<RooArgSet *>(snap)->snapshot();
          snapClone->setName(snap->GetName());
          newWS->getSnapshots().Add(snapClone);
        }
      }

      newWS->import(*pdf, RooFit::RecycleConflictNodes());
      RooAbsPdf *newPdf = newWS->pdf(pdf->GetName());
      newMC->SetPdf(*newPdf);

      for (auto d : data) {
        newWS->import(*d);
      }

      RooArgSet poiset;
      ::getArgs(newWS, poilist, poiset);
      RooArgSet npset;
      ::getArgs(newWS, nplist, npset);
      RooArgSet obsset;
      ::getArgs(newWS, obslist, obsset);
      RooArgSet globobsset;
      ::getArgs(newWS, globobslist, globobsset);

      newMC->SetParametersOfInterest(poiset);
      newMC->SetNuisanceParameters(npset);
      newMC->SetObservables(obsset);
      newMC->SetGlobalObservables(globobsset);
      newWS->import(*newMC);

      return newWS;
  }

}
