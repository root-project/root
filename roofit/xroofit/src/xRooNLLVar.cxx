/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class ROOT::Experimental::XRooFit::xRooNLLVar
\ingroup xroofit

This xRooNLLVar object has several special methods, e.g. for fitting and toy dataset generation.

 */

#include "RVersion.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#define protected public
#endif

#include "RooFitResult.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 33, 00)
#include "RooNLLVar.h"
#endif

#ifdef protected
#undef protected
#endif

#include "xRooFit/xRooFit.h"

#include "RooCmdArg.h"
#include "RooAbsPdf.h"
#include "RooAbsData.h"

#include "RooConstraintSum.h"
#include "RooSimultaneous.h"
#include "RooAbsCategoryLValue.h"
#include "TPRegexp.h"
#include "TEfficiency.h"

#include "RooRealVar.h"
#include "Math/ProbFunc.h"
#include "RooRandom.h"

#include "TPad.h"
#include "TSystem.h"

#include "coutCapture.h"

#include <chrono>

#include "Math/GenAlgoOptions.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#define private public
#define GETWS(a) a->_myws
#define GETWSSETS(w) w->_namedSets
#else
#define GETWS(a) a->workspace()
#define GETWSSETS(w) w->sets()
#endif
#include "RooWorkspace.h"
#ifdef private
#undef private
#endif

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#define protected public
#endif
#include "RooStats/HypoTestResult.h"
#ifdef protected
#undef protected
#endif

#include "TMultiGraph.h"
#include "TCanvas.h"
#include "TArrow.h"
#include "RooStringVar.h"
#include "TDirectory.h"
#include "TStyle.h"
#include "TH1D.h"
#include "TLegend.h"
#include "RooCategory.h"
#include "TTree.h"
#include "TGraph2D.h"

#include "RooGaussian.h"
#include "RooPoisson.h"

#include "TROOT.h"
#include "TKey.h"
#include "TRegexp.h"
#include "TStopwatch.h"

BEGIN_XROOFIT_NAMESPACE

std::set<int> xRooNLLVar::xRooHypoPoint::allowedStatusCodes = {0};

class AutoRestorer {
public:
   AutoRestorer(const RooAbsCollection &s, xRooNLLVar *nll = nullptr) : fSnap(s.snapshot()), fNll(nll)
   {
      fPars.add(s);
      if (fNll) {
         // if (!fNll->kReuseNLL) fOldNll = *fNll;
         fOldData = fNll->getData();
         fOldName = fNll->get()->GetName();
         fOldTitle = fNll->get()->getStringAttribute("fitresultTitle");
      }
   }
   ~AutoRestorer()
   {
      ((RooAbsCollection &)fPars) = *fSnap;
      if (fNll) {
         // commented out code was attempt to speed up things avoid unnecessarily reinitializing things over and over
         //            if (!fNll->kReuseNLL) {
         //                // can be faster just by putting back in old nll
         //                fNll->std::shared_ptr<RooAbsReal>::operator=(fOldNll);
         //                fNll->fData = fOldData.first;
         //                fNll->fGlobs = fOldData.second;
         //            } else {
         //                fNll->setData(fOldData);
         //                fNll->get()->SetName(fOldName);
         //                fNll->get()->setStringAttribute("fitresultTitle", (fOldTitle == "") ? nullptr : fOldTitle);
         //            }
         fNll->fGlobs = fOldData.second; // will mean globs matching checks are skipped in setData
         fNll->setData(fOldData);
         fNll->get()->SetName(fOldName);
         fNll->get()->setStringAttribute("fitresultTitle", (fOldTitle == "") ? nullptr : fOldTitle);
      }
   }
   RooArgSet fPars;
   std::unique_ptr<RooAbsCollection> fSnap;
   xRooNLLVar *fNll = nullptr;
   // std::shared_ptr<RooAbsReal> fOldNll;
   std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> fOldData;
   TString fOldName, fOldTitle;
};

xRooNLLVar::~xRooNLLVar() {}

xRooNLLVar::xRooNLLVar(RooAbsPdf &pdf, const std::pair<RooAbsData *, const RooAbsCollection *> &data,
                       const RooLinkedList &nllOpts)
   : xRooNLLVar(std::shared_ptr<RooAbsPdf>(&pdf, [](RooAbsPdf *) {}),
                std::make_pair(std::shared_ptr<RooAbsData>(data.first, [](RooAbsData *) {}),
                               std::shared_ptr<const RooAbsCollection>(data.second, [](const RooAbsCollection *) {})),
                nllOpts)
{
}

xRooNLLVar::xRooNLLVar(const std::shared_ptr<RooAbsPdf> &pdf,
                       const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &data,
                       const RooLinkedList &opts)
   : fPdf(pdf), fData(data.first), fGlobs(data.second)
{

   RooMsgService::instance().getStream(RooFit::INFO).removeTopic(RooFit::NumIntegration);

   fOpts = std::shared_ptr<RooLinkedList>(new RooLinkedList, [](RooLinkedList *l) {
      if (l)
         l->Delete();
      delete l;
   });
   fOpts->SetName("");

   // we *must* take global observables from the model even if they are included in the dataset
   // this is because the way xRooNLLVar is coded up it assumes the globs in the funcVars *ARE*
   // part of the model
   fOpts->Add(RooFit::GlobalObservablesSource("model").Clone(nullptr));

   for (int i = 0; i < opts.GetSize(); i++) {
      if (strlen(opts.At(i)->GetName()) == 0)
         continue; // skipping "none" cmds
      if (strcmp(opts.At(i)->GetName(), "GlobalObservables") == 0) {
         // will skip here to add with the obs from the function below
         // must match global observables
         auto gl = dynamic_cast<RooCmdArg *>(opts.At(i))->getSet(0);
         if (!fGlobs || !fGlobs->equals(*gl)) {
            throw std::runtime_error("GlobalObservables mismatch");
         }
      } else if (strcmp(opts.At(i)->GetName(), "Hesse") == 0) {
         fitConfig()->SetParabErrors(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0)); // controls hesse
      } else if (strcmp(opts.At(i)->GetName(), "Minos") == 0) {
         fitConfig()->SetMinosErrors(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0)); // controls minos
      } else if (strcmp(opts.At(i)->GetName(), "Strategy") == 0) {
         fitConfig()->MinimizerOptions().SetStrategy(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0));
      } else if (strcmp(opts.At(i)->GetName(), "StrategySequence") == 0) {
         fitConfigOptions()->SetNamedValue("StrategySequence", dynamic_cast<RooCmdArg *>(opts.At(i))->getString(0));
      } else if (strcmp(opts.At(i)->GetName(), "Tolerance") == 0) {
         fitConfig()->MinimizerOptions().SetTolerance(dynamic_cast<RooCmdArg *>(opts.At(i))->getDouble(0));
      } else if (strcmp(opts.At(i)->GetName(), "MaxCalls") == 0) {
         fitConfig()->MinimizerOptions().SetMaxFunctionCalls(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0));
      } else if (strcmp(opts.At(i)->GetName(), "MaxIterations") == 0) {
         fitConfig()->MinimizerOptions().SetMaxIterations(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0));
      } else if (strcmp(opts.At(i)->GetName(), "PrintLevel") == 0) {
         fitConfig()->MinimizerOptions().SetPrintLevel(dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0));
      } else {
         if (strcmp(opts.At(i)->GetName(), "Optimize") == 0) {
            // this flag will trigger constOptimizeTestStatistic to be called on the nll in createNLL method
            // we should ensure that the fitconfig setting is consistent with it ...
            fitConfigOptions()->SetValue("OptimizeConst", dynamic_cast<RooCmdArg *>(opts.At(i))->getInt(0));
         }
         fOpts->Add(opts.At(i)->Clone(nullptr)); // nullptr needed because accessing Clone via TObject base class puts
                                                 // "" instead, so doesnt copy names
      }
   }
   if (fGlobs) {
      // add global observables opt with function obs
      auto _vars = std::unique_ptr<RooArgSet>(fPdf->getVariables());
      if (auto extCon = dynamic_cast<RooCmdArg *>(fOpts->find("ExternalConstraints"))) {
         for (auto con : *extCon->getSet(0)) {
            _vars->add(*std::unique_ptr<RooArgSet>(con->getVariables()));
         }
      }
      auto _funcGlobs = std::unique_ptr<RooArgSet>(dynamic_cast<RooArgSet *>(_vars->selectCommon(*fGlobs)));
      fOpts->Add(RooFit::GlobalObservables(*_funcGlobs).Clone());
   }

   if (auto flag = dynamic_cast<RooCmdArg *>(fOpts->find("ReuseNLL"))) {
      kReuseNLL = flag->getInt(0);
   }

   // if fit range specified, and pdf is a RooSimultaneous, may need to 'reduce' the model if some of the pdfs are in
   // range and others are not
   if (auto range = dynamic_cast<RooCmdArg *>(fOpts->find("RangeWithName"))) {
      TString rangeName = range->getString(0);

      // reduce the data here for convenience, not really necessary because will happen inside RooNLLVar but still
      // fData.reset( fData->reduce(RooFit::SelectVars(*fData->get()),RooFit::CutRange(rangeName)) );

      if (auto s = dynamic_cast<RooSimultaneous *>(fPdf.get()); s) {
         auto &_cat = const_cast<RooAbsCategoryLValue &>(s->indexCat());
         std::vector<TString> chanPatterns;
         TStringToken pattern(rangeName, ",");
         bool hasRange(false);
         std::string noneCatRanges;
         while (pattern.NextToken()) {
            chanPatterns.emplace_back(pattern);
            if (_cat.hasRange(chanPatterns.back())) {
               hasRange = true;
            } else {
               if (!noneCatRanges.empty())
                  noneCatRanges += ",";
               noneCatRanges += chanPatterns.back();
            }
         }
         if (hasRange) {
            // must remove the ranges that referred to selections on channel category
            // otherwise RooFit will incorrectly evaluate the NLL (it creates a partition for each range given in the
            // list, which all end up being equal) the NLL would become scaled by the number of ranges given
            if (noneCatRanges.empty()) {
               fOpts->Remove(range);
               SafeDelete(range);
            } else {
               range->setString(0, noneCatRanges.c_str());
            }
            // must reduce because category var has one of the ranges
            auto newPdf =
               std::make_shared<RooSimultaneous>(TString::Format("%s_reduced", s->GetName()), "Reduced model", _cat);
            for (auto &c : _cat) {
               auto _pdf = s->getPdf(c.first.c_str());
               if (!_pdf)
                  continue;
               _cat.setIndex(c.second);
               bool matchAny = false;
               for (auto &p : chanPatterns) {
                  if (_cat.hasRange(p) && _cat.inRange(p)) {
                     matchAny = true;
                     break;
                  }
               }
               if (matchAny) {
                  newPdf->addPdf(*_pdf, c.first.c_str());
               }
            }
            fPdf = newPdf;
         }
      }
   }

   //    if (fGlobs) {
   //        // must check GlobalObservables is in the list
   //    }
   //
   //    if (auto globs = dynamic_cast<RooCmdArg*>(fOpts->find("GlobalObservables"))) {
   //        // first remove any obs the pdf doesnt depend on
   //        auto _vars = std::unique_ptr<RooAbsCollection>( fPdf->getVariables() );
   //        auto _funcGlobs = std::unique_ptr<RooAbsCollection>(_vars->selectCommon(*globs->getSet(0)));
   //        fGlobs.reset( std::unique_ptr<RooAbsCollection>(globs->getSet(0)->selectCommon(*_funcGlobs))->snapshot() );
   //        globs->setSet(0,dynamic_cast<const RooArgSet&>(*_funcGlobs)); // globs in linked list has its own argset
   //        but args need to live as long as the func
   //        /*RooArgSet toRemove;
   //        for(auto a : *globs->getSet(0)) {
   //            if (!_vars->find(*a)) toRemove.add(*a);
   //        }
   //        const_cast<RooArgSet*>(globs->getSet(0))->remove(toRemove);
   //        fGlobs.reset( globs->getSet(0)->snapshot() );
   //        fGlobs->setAttribAll("Constant",true);
   //        const_cast<RooArgSet*>(globs->getSet(0))->replace(*fGlobs);*/
   //    }
}

xRooNLLVar::xRooNLLVar(const std::shared_ptr<RooAbsPdf> &pdf, const std::shared_ptr<RooAbsData> &data,
                       const RooLinkedList &opts)
   : xRooNLLVar(
        pdf,
        std::make_pair(data, std::shared_ptr<const RooAbsCollection>(
                                (opts.find("GlobalObservables"))
                                   ? dynamic_cast<RooCmdArg *>(opts.find("GlobalObservables"))->getSet(0)->snapshot()
                                   : nullptr)),
        opts)
{
}

void xRooNLLVar::Print(Option_t *)
{
   std::cout << "PDF: ";
   if (fPdf) {
      fPdf->Print();
   } else {
      std::cout << "<null>" << std::endl;
   }
   std::cout << "Data: ";
   if (fData) {
      fData->Print();
   } else {
      std::cout << "<null>" << std::endl;
   }
   std::cout << "NLL Options: " << std::endl;
   for (int i = 0; i < fOpts->GetSize(); i++) {
      auto c = dynamic_cast<RooCmdArg *>(fOpts->At(i));
      if (!c)
         continue;
      std::cout << " " << c->GetName() << " : ";
      if (c->getString(0)) {
         std::cout << c->getString(0);
      } else if (c->getSet(0) && !c->getSet(0)->empty()) {
         std::cout << (c->getSet(0)->contentsString());
      } else {
         std::cout << c->getInt(0);
      }
      std::cout << std::endl;
   }
   if (fFitConfig) {
      std::cout << "Fit Config: " << std::endl;
      std::cout << "  UseParabErrors: " << (fFitConfig->ParabErrors() ? "True" : "False")
                << "  [toggles HESSE algorithm]" << std::endl;
      std::cout << "  MinimizerOptions: " << std::endl;
      fFitConfig->MinimizerOptions().Print();
   }
   std::cout << "Last Rebuild Log Output: " << fFuncCreationLog << std::endl;
}

void xRooNLLVar::reinitialize()
{
   TString oldName = "";
   if (std::shared_ptr<RooAbsReal>::get())
      oldName = std::shared_ptr<RooAbsReal>::get()->GetName();
   if (fPdf) {
      cout_redirect c(fFuncCreationLog);
      // need to find all RooRealSumPdf nodes and mark them binned or unbinned as required
      RooArgSet s;
      fPdf->treeNodeServerList(&s, nullptr, true, false);
      s.add(*fPdf); // ensure include self in case fitting a RooRealSumPdf
      bool isBinned = false;
      bool hasBinned = false; // if no binned option then 'auto bin' ...
      if (auto a = dynamic_cast<RooCmdArg *>(fOpts->find("Binned")); a) {
         hasBinned = true;
         isBinned = a->getInt(0);
      }
      std::map<RooAbsArg *, bool> origValues;
      if (hasBinned) {
         for (auto a : s) {
            if (a->InheritsFrom("RooRealSumPdf")) {
               // since RooNLLVar will assume binBoundaries available (not null), we should check bin boundaries
               // available
               bool setBinned = false;
               if (isBinned) {
                  std::unique_ptr<RooArgSet> obs(a->getObservables(fData->get()));
                  if (obs->size() == 1) { // RooNLLVar requires exactly 1 obs
                     auto *var = static_cast<RooRealVar *>(obs->first());
                     std::unique_ptr<std::list<double>> boundaries{dynamic_cast<RooAbsReal *>(a)->binBoundaries(
                        *var, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())};
                     if (boundaries) {
                        if (!std::shared_ptr<RooAbsReal>::get()) {
                           Info("xRooNLLVar", "%s will be evaluated as a Binned PDF (%d bins)", a->GetName(),
                                int(boundaries->size() - 1));
                        }
                        setBinned = true;
                     }
                  }
               }
               origValues[a] = a->getAttribute("BinnedLikelihood");
               a->setAttribute("BinnedLikelihood", setBinned);
            }
         }
      }
      std::map<RooAbsPdf *, std::string> normRanges;
      if (auto range = dynamic_cast<RooCmdArg *>(fOpts->find("RangeWithName"))) {
         TString rangeName = range->getString(0);
         if (auto sr = dynamic_cast<RooCmdArg *>(fOpts->find("SplitRange"));
             sr && sr->getInt(0) && dynamic_cast<RooSimultaneous *>(fPdf.get())) {
            // doing split range ... need to loop over categories of simpdf and apply range to each
            auto simPdf = dynamic_cast<RooSimultaneous *>(fPdf.get());
            for (auto cat : simPdf->indexCat()) {
               auto subpdf = simPdf->getPdf(cat.first.c_str());
               if (!subpdf)
                  continue; // state not in pdf
               TString srangeName(rangeName);
               srangeName.ReplaceAll(",", "_" + cat.first + ",");
               srangeName += "_" + cat.first;
               RooArgSet ss;
               subpdf->treeNodeServerList(&ss, nullptr, true, false);
               ss.add(*subpdf);
               for (auto a : ss) {
                  if (a->InheritsFrom("RooAddPdf")) {
                     auto p = dynamic_cast<RooAbsPdf *>(a);
                     normRanges[p] = p->normRange() ? p->normRange() : "";
                     p->setNormRange(srangeName);
                  }
               }
            }
         } else {
            // set range on all AddPdfs before creating - needed in cases where coefs are present and need fractioning
            // based on fit range bugfix needed: roofit needs to propagate the normRange to AddPdfs child nodes (used in
            // createExpectedEventsFunc)
            for (auto a : s) {
               if (a->InheritsFrom("RooAddPdf")) {
                  auto p = dynamic_cast<RooAbsPdf *>(a);
                  normRanges[p] = p->normRange() ? p->normRange() : "";
                  p->setNormRange(rangeName);
               }
            }
         }
      }
      // before creating, clear away caches if any if pdf is in ws
      if (GETWS(fPdf)) {
         std::set<std::string> setNames;
         for (auto &a : GETWSSETS(GETWS(fPdf))) {
            if (TString(a.first.c_str()).BeginsWith("CACHE_")) {
               setNames.insert(a.first);
            }
         }
         for (auto &a : setNames) {
            GETWS(fPdf)->removeSet(a.c_str());
         }
      }
      std::set<std::string> attribs;
      if (std::shared_ptr<RooAbsReal>::get())
         attribs = std::shared_ptr<RooAbsReal>::get()->attributes();
      this->reset(std::unique_ptr<RooAbsReal>{fPdf->createNLL(*fData, *fOpts)}.release());
      std::shared_ptr<RooAbsReal>::get()->SetName(TString::Format("nll_%s/%s", fPdf->GetName(), fData->GetName()));
      // RooFit only swaps in what it calls parameters, this misses out the RooConstVars which we treat as pars as well
      // so swap those in ... question: is recursiveRedirectServers usage in RooAbsOptTestStatic (and here) a memory
      // leak?? where do the replaced servers get deleted??

      for (auto &[k, v] : normRanges)
         k->setNormRange(v == "" ? nullptr : v.c_str());

      for (auto &a : attribs)
         std::shared_ptr<RooAbsReal>::get()->setAttribute(a.c_str());
      // create parent on next line to avoid triggering workspace initialization code in constructor of xRooNode
      if (GETWS(fPdf)) {
         xRooNode(*GETWS(fPdf), std::make_shared<xRooNode>()).sterilize();
      } // there seems to be a nasty bug somewhere that can make the cache become invalid, so clear it here
      if (oldName != "")
         std::shared_ptr<RooAbsReal>::get()->SetName(oldName);
      if (!origValues.empty()) {
         // need to evaluate NOW so that slaves are created while the BinnedLikelihood settings are in place
         std::shared_ptr<RooAbsReal>::get()->getVal();
         for (auto &[o, v] : origValues)
            o->setAttribute("BinnedLikelihood", v);
      }
   }

   fFuncVars = std::unique_ptr<RooArgSet>{std::shared_ptr<RooAbsReal>::get()->getVariables()};
   if (fGlobs) {
      fFuncGlobs.reset(fFuncVars->selectCommon(*fGlobs));
      fFuncGlobs->setAttribAll("Constant", true);
   }
   fConstVars.reset(fFuncVars->selectByAttrib("Constant", true)); // will check if any of these have floated
}

std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>>
xRooNLLVar::generate(bool expected, int seed)
{
   if (!fPdf)
      return std::pair(nullptr, nullptr);
   auto fr = std::make_shared<RooFitResult>(TUUID().AsString());
   fr->setFinalParList(RooArgList());
   RooArgList l;
   l.add((fFuncVars) ? *fFuncVars : *std::unique_ptr<RooAbsCollection>(fPdf->getParameters(*fData)));
   fr->setConstParList(l);
   const_cast<RooArgList &>(fr->constPars()).setAttribAll("global", false);
   if (fGlobs)
      std::unique_ptr<RooAbsCollection>(fr->constPars().selectCommon(*fGlobs))->setAttribAll("global", true);
   return xRooFit::generateFrom(*fPdf, *fr, expected, seed);
}

xRooNLLVar::xRooFitResult::xRooFitResult(const RooFitResult &fr) : xRooFitResult(std::make_shared<xRooNode>(fr)) {}

xRooNLLVar::xRooFitResult::xRooFitResult(const std::shared_ptr<xRooNode> &in, const std::shared_ptr<xRooNLLVar> &nll)
   : std::shared_ptr<const RooFitResult>(std::dynamic_pointer_cast<const RooFitResult>(in->fComp)),
     fNode(in),
     fNll(nll),
     fCfits(std::make_shared<std::map<std::string, xRooFitResult>>())
{
}
const RooFitResult *xRooNLLVar::xRooFitResult::operator->() const
{
   return fNode->get<RooFitResult>();
}
// xRooNLLVar::xRooFitResult::operator std::shared_ptr<const RooFitResult>() const { return
// std::dynamic_pointer_cast<const RooFitResult>(fNode->fComp); }
xRooNLLVar::xRooFitResult::operator const RooFitResult *() const
{
   return fNode->get<const RooFitResult>();
}
void xRooNLLVar::xRooFitResult::Draw(Option_t *opt)
{
   fNode->Draw(opt);
}

xRooNLLVar::xRooFitResult xRooNLLVar::xRooFitResult::cfit(const char *poiValues, const char *alias)
{

   // create a hypoPoint with ufit equal to this fit
   // and poi equal to given poi
   if (!fNll)
      throw std::runtime_error("xRooFitResult::cfit: Cannot create cfit without nll");

   // see if fit already done
   if (alias) {
      if (auto res = fCfits->find(alias); res != fCfits->end()) {
         return res->second;
      }
   }
   if (auto res = fCfits->find(poiValues); res != fCfits->end()) {
      return res->second;
   }

   AutoRestorer s(*fNll->fFuncVars);
   *fNll->fFuncVars = get()->floatParsFinal();
   fNll->fFuncVars->assignValueOnly(get()->constPars());
   std::unique_ptr<RooAbsCollection>(fNll->fFuncVars->selectCommon(get()->floatParsFinal()))
      ->setAttribAll("Constant", false);
   std::unique_ptr<RooAbsCollection>(fNll->fFuncVars->selectCommon(get()->constPars()))->setAttribAll("Constant", true);

   auto hp = fNll->hypoPoint(poiValues, std::numeric_limits<double>::quiet_NaN(), xRooFit::Asymptotics::Unknown);
   hp.fUfit = *this;
   auto out = xRooNLLVar::xRooFitResult(std::make_shared<xRooNode>(hp.cfit_null(), fNode->fParent), fNll);
   fCfits->insert(std::pair((alias) ? alias : poiValues, out));
   return out;
}
xRooNLLVar::xRooFitResult xRooNLLVar::xRooFitResult::ifit(const char *np, bool up, bool prefit)
{
   RooRealVar *npVar = dynamic_cast<RooRealVar *>((prefit ? get()->floatParsInit() : get()->floatParsFinal()).find(np));
   if (!npVar)
      throw std::runtime_error("xRooFitResult::ifit: par not found");
   return cfit(TString::Format("%s=%f", np, npVar->getVal() + (up ? npVar->getErrorHi() : npVar->getErrorLo())));
}
double xRooNLLVar::xRooFitResult::impact(const char *poi, const char *np, bool up, bool prefit, bool covApprox)
{
   if (!covApprox) {
      // get the ifit and get the difference between the postFit poi values
      RooRealVar *poiHat = dynamic_cast<RooRealVar *>((get()->floatParsFinal()).find(poi));
      if (!poiHat)
         throw std::runtime_error("xRooFitResult::impact: poi not found");
      auto _ifit = ifit(np, up, prefit);
      if (!_ifit)
         throw std::runtime_error("xRooFitResult::impact: null ifit");
      if (_ifit->status() != 0)
         fNode->Warning("impact", "ifit status code is %d", _ifit->status());
      return _ifit->floatParsFinal().getRealValue(poi) - poiHat->getVal();
   } else {
      // estimate impact from the covariance matrix ....
      int iPoi = get()->floatParsFinal().index(poi);
      int iNp = get()->floatParsFinal().index(np);
      if (iPoi == -1)
         throw std::runtime_error("xRooFitResult::impact: poi not found");
      if (iNp == -1)
         throw std::runtime_error("xRooFitResult::impact: np not found");
      RooRealVar *npVar =
         dynamic_cast<RooRealVar *>((prefit ? get()->floatParsInit() : get()->floatParsFinal()).find(np));
      return get()->covarianceMatrix()(iPoi, iNp) / (up ? npVar->getErrorHi() : npVar->getErrorLo());
   }
   return std::numeric_limits<double>::quiet_NaN();
}

double xRooNLLVar::xRooFitResult::conditionalError(const char *poi, const char *nps, bool up, bool covApprox)
{
   // run a fit with given NPs held constant, return quadrature difference

   TString npNames;
   RooArgList vars;
   RooAbsArg *poiVar = nullptr;
   for (auto p : get()->floatParsFinal()) {
      if (strcmp(p->GetName(), poi) == 0) {
         vars.add(*p);
         poiVar = p;
         continue;
      }
      TStringToken pattern(nps, ",");
      bool matches = false;
      while (pattern.NextToken()) {
         TString s(pattern);
         if ((p->getStringAttribute("group") && s == p->getStringAttribute("group")) ||
             TString(p->GetName()).Contains(TRegexp(s, true)) || p->getAttribute(s)) {
            matches = true;
            break;
         }
      }
      if (matches) {
         if (npNames.Length())
            npNames += ",";
         npNames += p->GetName();
      } else {
         vars.add(*p); // keeping in reduced cov matrix
      }
   }
   if (!poiVar) {
      throw std::runtime_error(TString::Format("Could not find poi: %s", poi));
   }
   if (npNames == "") {
      fNode->Warning("conditionalError", "No parameters selected by: %s", nps);
      return (up) ? static_cast<RooRealVar *>(poiVar)->getErrorHi() : static_cast<RooRealVar *>(poiVar)->getErrorLo();
   }

   if (covApprox) {
      int idx = vars.index(poi);
      return sqrt(get()->conditionalCovarianceMatrix(vars)(idx, idx));
   }

   auto _cfit = cfit(npNames.Data(), nps);

   auto _poi = _cfit->floatParsFinal().find(poi);

   return (up) ? static_cast<RooRealVar *>(_poi)->getErrorHi() : static_cast<RooRealVar *>(_poi)->getErrorLo();
}

RooArgList xRooNLLVar::xRooFitResult::ranknp(const char *poi, bool up, bool prefit, double approxThreshold)
{

   RooRealVar *poiHat = dynamic_cast<RooRealVar *>((get()->floatParsFinal()).find(poi));
   if (!poiHat)
      throw std::runtime_error("xRooFitResult::ranknp: poi not found");

   std::vector<std::pair<std::string, double>> ranks;
   // first do with the covariance approximation, since that's always available
   for (auto par : get()->floatParsFinal()) {
      if (par == poiHat)
         continue;
      ranks.emplace_back(std::pair(par->GetName(), impact(poi, par->GetName(), up, prefit, true)));
   }

   std::sort(ranks.begin(), ranks.end(), [](auto &left, auto &right) {
      if (std::isnan(left.second) && !std::isnan(right.second))
         return false;
      if (!std::isnan(left.second) && std::isnan(right.second))
         return true;
      return fabs(left.second) > fabs(right.second);
   });

   // now redo the ones above the threshold
   for (auto &[n, v] : ranks) {
      if (v >= approxThreshold) {
         try {
            v = impact(poi, n.c_str(), up, prefit);
         } catch (...) {
            v = std::numeric_limits<double>::quiet_NaN();
         };
      }
   }

   // resort
   std::sort(ranks.begin(), ranks.end(), [](auto &left, auto &right) {
      if (std::isnan(left.second) && !std::isnan(right.second))
         return false;
      if (!std::isnan(left.second) && std::isnan(right.second))
         return true;
      return fabs(left.second) > fabs(right.second);
   });

   RooArgList out;
   out.setName("rankings");
   for (auto &[n, v] : ranks) {
      out.addClone(*get()->floatParsFinal().find(n.c_str()));
      auto vv = static_cast<RooRealVar *>(out.at(out.size() - 1));
      vv->setVal(v);
      vv->removeError();
      vv->removeRange();
   }
   return out;
}

xRooNLLVar::xRooFitResult xRooNLLVar::minimize(const std::shared_ptr<ROOT::Fit::FitConfig> &_config)
{
   auto &nll = *get();
   auto out = xRooFit::minimize(nll, (_config) ? _config : fitConfig(), fOpts);
   // add any pars that are const here that aren't in constPars list because they may have been
   // const-optimized and their values cached with the dataset, so if subsequently floated the
   // nll wont evaluate correctly
   // fConstVars.reset( fFuncVars->selectByAttrib("Constant",true) );

   // before returning, flag which of the constPars were actually global observables
   if (out) {
      const_cast<RooArgList &>(out->constPars()).setAttribAll("global", false);
      if (fGlobs)
         std::unique_ptr<RooAbsCollection>(out->constPars().selectCommon(*fGlobs))->setAttribAll("global", true);
   }
   return xRooFitResult(std::make_shared<xRooNode>(out, fPdf), std::make_shared<xRooNLLVar>(*this));
}

std::shared_ptr<ROOT::Fit::FitConfig> xRooNLLVar::fitConfig()
{
   if (!fFitConfig)
      fFitConfig = xRooFit::createFitConfig();
   return fFitConfig;
}

ROOT::Math::IOptions *xRooNLLVar::fitConfigOptions()
{
   if (auto conf = fitConfig(); conf)
      return const_cast<ROOT::Math::IOptions *>(conf->MinimizerOptions().ExtraOptions());
   return nullptr;
}

double xRooNLLVar::getEntryVal(size_t entry) const
{
   auto _data = data();
   if (!_data)
      return 0;
   if (size_t(_data->numEntries()) <= entry)
      return 0;
   auto _pdf = pdf();
   *std::unique_ptr<RooAbsCollection>(_pdf->getObservables(_data)) = *_data->get(entry);
   // if (auto s = dynamic_cast<RooSimultaneous*>(_pdf.get());s) return
   // -_data->weight()*s->getPdf(s->indexCat().getLabel())->getLogVal(_data->get());
   return -_data->weight() * _pdf->getLogVal(_data->get());
}

std::set<std::string> xRooNLLVar::binnedChannels() const
{
   std::set<std::string> out;

   auto binnedOpt = dynamic_cast<RooCmdArg *>(fOpts->find("Binned")); // the binned option, if explicitly specified

   if (auto s = dynamic_cast<RooSimultaneous *>(pdf().get())) {
      xRooNode simPdf(*s);
      bool allChannels = true;
      for (auto c : simPdf.bins()) {
         // see if there's a RooRealSumPdf in the channel - if there is, if it has BinnedLikelihood set
         // then assume is a BinnedLikelihood channel
         RooArgSet nodes;
         c->get<RooAbsArg>()->treeNodeServerList(&nodes, nullptr, true, false);
         bool isBinned = false;
         for (auto a : nodes) {
            if (a->InheritsFrom("RooRealSumPdf") &&
                ((binnedOpt && binnedOpt->getInt(0)) || (!binnedOpt && a->getAttribute("BinnedLikelihood")))) {
               TString chanName(c->GetName());
               out.insert(chanName(chanName.Index("=") + 1, chanName.Length()).Data());
               isBinned = true;
               break;
            }
         }
         if (!isBinned) {
            allChannels = false;
         }
      }
      if (allChannels) {
         out.clear();
         out.insert("*");
      }
   } else {
      RooArgSet nodes;
      pdf()->treeNodeServerList(&nodes, nullptr, true, false);
      for (auto a : nodes) {
         if (a->InheritsFrom("RooRealSumPdf") &&
             ((binnedOpt && binnedOpt->getInt(0)) || (!binnedOpt && a->getAttribute("BinnedLikelihood")))) {
            out.insert("*");
            break;
         }
      }
   }
   return out;
}

double xRooNLLVar::getEntryBinWidth(size_t entry) const
{

   auto _data = data();
   if (!_data)
      return 0;
   if (size_t(_data->numEntries()) <= entry)
      return 0;
   auto _pdf = pdf().get();
   std::unique_ptr<RooAbsCollection> _robs(_pdf->getObservables(_data->get()));
   *_robs = *_data->get(entry); // only set robs
   if (auto s = dynamic_cast<RooSimultaneous *>(_pdf); s) {
      _pdf = s->getPdf(s->indexCat().getCurrentLabel());
   }
   double volume = 1.;
   for (auto o : *_robs) {

      if (auto a = dynamic_cast<RooAbsRealLValue *>(o);
          a && _pdf->dependsOn(*a)) { // dependsOn check needed until ParamHistFunc binBoundaries method fixed
         std::unique_ptr<std::list<double>> bins(
            _pdf->binBoundaries(*a, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()));
         if (bins) {
            double lowEdge = -std::numeric_limits<double>::infinity();
            for (auto b : *bins) {
               if (b > a->getVal()) {
                  volume *= (b - lowEdge);
                  break;
               }
               lowEdge = b;
            }
         }
      }
   }

   return volume;
}

double xRooNLLVar::saturatedConstraintTermVal() const
{
   // for each global observable in the dataset, determine which constraint term is associated to it
   // and given its type, add the necessary saturated term...

   double out = 0;

   if (!fGlobs)
      return 0;

   auto cTerm = constraintTerm();
   if (!cTerm)
      return 0;

   for (auto c : cTerm->list()) {
      if (std::string(c->ClassName()) == "RooAbsPdf" || std::string(c->ClassName()).find("RooNormalizedPdf")!=std::string::npos) {
         // in ROOT 6.32 the constraintTerm is full of RooNormalizedPdfs which aren't public
         // became public in 6.34, hence now also check for RooNormalizedPdf explicitly
         // in this case use the first server
         c = c->servers()[0];
      }
      if (auto gaus = dynamic_cast<RooGaussian *>(c)) {
         auto v = dynamic_cast<RooAbsReal *>(fGlobs->find(gaus->getX().GetName()));
         if (!v) {
            v = dynamic_cast<RooAbsReal *>(fGlobs->find(
               gaus->getMean().GetName())); // shouldn't really happen but does for at least ws made by pyhf
         }
         if (!v)
            continue;
         out -= std::log(ROOT::Math::gaussian_pdf(v->getVal(), gaus->getSigma().getVal(), v->getVal()));
      } else if (auto pois = dynamic_cast<RooPoisson *>(c)) {
         auto v = dynamic_cast<RooAbsReal *>(fGlobs->find(pois->getX().GetName()));
         if (!v)
            continue;
         out -= std::log(TMath::Poisson(v->getVal(), v->getVal()));
      }
   }

   return out;
}

double xRooNLLVar::ndof() const
{
   return data()->numEntries() + (fFuncGlobs ? fFuncGlobs->size() : 0) -
          std::unique_ptr<RooAbsCollection>(pars()->selectByAttrib("Constant", false))->size();
}

double xRooNLLVar::pgof() const
{
   // note that if evaluating this for a single channel, until 6.30 is available if you are using Binned mode the pdf
   // will need to be part of a Simultaneous
   return TMath::Prob(2. * (get()->getVal() - saturatedVal()), ndof());
}

double xRooNLLVar::mainTermNdof() const
{
   // need to count number of floating unconstrained parameters
   // which are floating parameters not featured in the constraintTerm
   std::unique_ptr<RooAbsCollection> _floats(pars()->selectByAttrib("Constant", false));
   if (auto _constraintTerm = constraintTerm()) {
      _floats->remove(*std::unique_ptr<RooAbsCollection>(_constraintTerm->getVariables()));
   }
   return data()->numEntries() - _floats->size();
}

double xRooNLLVar::mainTermVal() const
{
   // using totVal - constraintTerm while new evalbackend causes mainTerm() to return nullptr
   return get()->getVal() - constraintTermVal();
}

double xRooNLLVar::constraintTermVal() const
{
   if (auto _constraintTerm = constraintTerm()) {
      return _constraintTerm->getVal();
   }
   return 0;
}

double xRooNLLVar::mainTermPgof() const
{
   return TMath::Prob(2. * (mainTermVal() - saturatedMainTermVal()), mainTermNdof());
}

double xRooNLLVar::saturatedVal() const
{
   return saturatedMainTermVal() + saturatedConstraintTermVal();
}

double xRooNLLVar::saturatedMainTermVal() const
{

   // Use this term to create a goodness-of-fit metric, which is approx chi2 distributed with numEntries (data) d.o.f:
   // prob = TMath::Prob( 2.*(nll.mainTerm()->getVal() - nll.saturatedNllTerm()), nll.data()->numEntries() )

   // note that need to construct nll with explicit Binned(1 or 0) option otherwise will pick up nll eval
   // from attributes in model already, so many get binned mainTerm eval when thinking not binned because didnt specify
   // Binned(1)

   auto _data = data();
   if (!_data)
      return std::numeric_limits<double>::quiet_NaN();

   std::set<std::string> _binnedChannels = binnedChannels();

   // for binned case each entry is: -(-N + Nlog(N) - TMath::LnGamma(N+1))
   // for unbinned case each entry is: -(N*log(N/(sumN*binW))) = -N*logN + N*log(sumN) + N*log(binW)
   // but unbinned gets extendedTerm = sumN - sumN*log(sumN)
   // so resulting sum is just sumN - sum[ N*logN - N*log(binW) ]
   // which is the same as the binned case without the LnGamma part and with the extra sum[N*log(binW)] part

   const RooAbsCategoryLValue *cat = (dynamic_cast<RooSimultaneous *>(pdf().get()))
                                        ? &dynamic_cast<RooSimultaneous *>(pdf().get())->indexCat()
                                        : nullptr;

   double out = _data->sumEntries();
   for (int i = 0; i < _data->numEntries(); i++) {
      _data->get(i);
      double w = _data->weight();
      if (w == 0)
         continue;
      out -= w * std::log(w);
      if (_binnedChannels.count("*")) {
         out += TMath::LnGamma(w + 1);
      } else if (_binnedChannels.empty()) {
         out += w * std::log(getEntryBinWidth(i));
      } else if (cat) {
         // need to determine which channel we are in for this entry to decide if binned or unbinned active
         if (_binnedChannels.count(_data->get()->getCatLabel(cat->GetName()))) {
            out += TMath::LnGamma(w + 1);
         } else {
            out += w * std::log(getEntryBinWidth(i));
         }
      } else {
         throw std::runtime_error("Cannot determine category of RooSimultaneous pdf");
      }
   }

   out += simTermVal();

   return out;
}

std::shared_ptr<RooArgSet> xRooNLLVar::pars(bool stripGlobalObs) const
{
   auto out = std::shared_ptr<RooArgSet>(get()->getVariables());
   if (stripGlobalObs && fGlobs) {
      out->remove(*fGlobs, true, true);
   }
   return out;
}

TObject *
xRooNLLVar::Scan(const char *scanPars, const std::vector<std::vector<double>> &coords, const RooArgList &profilePars)
{
   return Scan(*std::unique_ptr<RooAbsCollection>(get()->getVariables()->selectByName(scanPars)), coords, profilePars);
}

TObject *xRooNLLVar::Scan(const RooArgList &scanPars, const std::vector<std::vector<double>> &coords,
                          const RooArgList &profilePars)
{

   if (scanPars.size() > 2 || scanPars.empty())
      return nullptr;

   TGraph2D *out2d = (scanPars.size() == 2) ? new TGraph2D() : nullptr;
   TGraph *out1d = (out2d) ? nullptr : new TGraph();
   TNamed *out = (out2d) ? static_cast<TNamed *>(out2d) : static_cast<TNamed *>(out1d);
   out->SetName(get()->GetName());
   out->SetTitle(TString::Format("%s;%s%s%s", get()->GetTitle(), scanPars.first()->GetTitle(), out2d ? ";" : "",
                                 out2d ? scanPars.at(1)->GetTitle() : ""));

   std::unique_ptr<RooAbsCollection> funcVars(get()->getVariables());
   AutoRestorer snap(*funcVars);

   for (auto &coord : coords) {
      if (coord.size() != scanPars.size()) {
         throw std::runtime_error("Invalid coordinate");
      }
      for (size_t i = 0; i < coord.size(); i++) {
         static_cast<RooAbsRealLValue &>(scanPars[i]).setVal(coord[i]);
      }

      if (profilePars.empty()) {
         // just evaluate
         if (out2d) {
            out2d->SetPoint(out2d->GetN(), coord[0], coord[1], get()->getVal());
         } else {
            out1d->SetPoint(out1d->GetN(), coord[0], get()->getVal());
         }
      }
   }

   return out;
}

void xRooNLLVar::Draw(Option_t *opt)
{
   TString sOpt(opt);

   auto _pars = pars();

   if (sOpt == "sensitivity") {

      // will make a plot of DeltaNLL
   }

   if (sOpt == "floating") {
      // start scanning floating pars
      auto floats = std::unique_ptr<RooAbsCollection>(_pars->selectByAttrib("Constant", false));
      TVirtualPad *pad = gPad;
      if (!pad) {
         TCanvas::MakeDefCanvas();
         pad = gPad;
      }
      TMultiGraph *gr = new TMultiGraph;
      gr->SetName("multigraph");
      gr->SetTitle(TString::Format("%s;Normalized Parameter Value;#Delta NLL", get()->GetTitle()));
      /*((TPad*)pad)->DivideSquare(floats->size());
      int i=0;
      for(auto a : *floats) {
          i++;
          pad->cd(i);
          Draw(a->GetName());
      }*/
      return;
   }

   RooArgList vars;
   TStringToken pattern(sOpt, ":");
   while (pattern.NextToken()) {
      TString s(pattern);
      if (auto a = _pars->find(s); a)
         vars.add(*a);
   }

   if (vars.size() == 1) {
      TGraph *out = new TGraph;
      out->SetBit(kCanDelete);
      TGraph *bad = new TGraph;
      bad->SetBit(kCanDelete);
      bad->SetMarkerColor(kRed);
      bad->SetMarkerStyle(5);
      TMultiGraph *gr = (gPad) ? dynamic_cast<TMultiGraph *>(gPad->GetPrimitive("multigraph")) : nullptr;
      bool normRange = false;
      if (!gr) {
         gr = new TMultiGraph;
         gr->Add(out, "LP");
         gr->SetBit(kCanDelete);
      } else {
         normRange = true;
      }
      out->SetName(get()->GetName());
      gr->SetTitle(TString::Format("%s;%s;#Delta NLL", get()->GetTitle(), vars.at(0)->GetTitle()));
      // scan outwards from current value towards limits
      auto v = dynamic_cast<RooRealVar *>(vars.at(0));
      double low = v->getVal();
      double high = low;
      double step = (v->getMax() - v->getMin()) / 100;
      double init = v->getVal();
      double initVal = func()->getVal();
      // double xscale = (normRange) ? (2.*(v->getMax() - v->getMin())) : 1.;
      auto currTime = std::chrono::steady_clock::now();
      while (out->GetN() < 100 && (low > v->getMin() || high < v->getMax())) {
         if (out->GetN() == 0) {
            out->SetPoint(out->GetN(), low, 0);
            low -= step;
            high += step;
            if (!normRange) {
               gr->Draw("A");
               gPad->SetGrid();
            }
            continue;
         }
         if (low > v->getMin()) {
            v->setVal(low);
            auto _v = func()->getVal();
            if (std::isnan(_v) || std::isinf(_v)) {
               if (bad->GetN() == 0)
                  gr->Add(bad, "P");
               bad->SetPoint(bad->GetN(), low, out->GetPointY(0));
            } else {
               out->SetPoint(out->GetN(), low, _v - initVal);
            }
            low -= step;
         }
         if (high < v->getMax()) {
            v->setVal(high);
            auto _v = func()->getVal();
            if (std::isnan(_v) || std::isinf(_v)) {
               if (bad->GetN() == 0)
                  gr->Add(bad, "P");
               bad->SetPoint(bad->GetN(), high, out->GetPointY(0));
            } else {
               out->SetPoint(out->GetN(), high, _v - initVal);
            }
            high += step;
         }
         out->Sort();
         // should only do processEvents once every second in case using x11 (which is slow)
         gPad->Modified();
         if (std::chrono::steady_clock::now() - currTime > std::chrono::seconds(1)) {
            currTime = std::chrono::steady_clock::now();
            gPad->Update();
            gSystem->ProcessEvents();
         }
      }
      // add arrow to show current par value
      TArrow a;
      a.DrawArrow(init, 0, init, -0.1);
      gPad->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
      gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
      gSystem->ProcessEvents();
      v->setVal(init);
   } else {
      Error("Draw", "Name a parameter to scan over: Draw(<name>) , choose from: %s",
            _pars->empty() ? "" : _pars->contentsString().c_str());
   }
}

std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> xRooNLLVar::getData() const
{
   return std::make_pair(fData, fGlobs);
}

bool xRooNLLVar::setData(const xRooNode &data)
{
   if (data.fComp && !data.get<RooAbsData>()) {
      return false;
   }
   return setData(std::dynamic_pointer_cast<RooAbsData>(data.fComp),
                  std::shared_ptr<const RooAbsCollection>(data.globs().argList().snapshot()));
}

bool xRooNLLVar::setData(const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &_data)
{

   if (fData == _data.first && fGlobs == _data.second)
      return true;

   auto _globs = fGlobs; // done to keep globs alive while NLL might still be alive.

   auto _dglobs = (_data.second) ? _data.second
                                 : std::shared_ptr<const RooAbsCollection>(_data.first->getGlobalObservables(),
                                                                           [](const RooAbsCollection *) {});

   if (fGlobs && !(fGlobs->empty() && !_dglobs) && _data.first &&
       fGlobs != _dglobs) { // second condition allows for no globs being a nullptr, third allow globs to remain if
                            // nullifying data
      if (!_dglobs)
         throw std::runtime_error("Missing globs");
      // ignore 'extra' globs
      RooArgSet s;
      s.add(*fGlobs);
      std::unique_ptr<RooAbsCollection> _actualGlobs(fPdf->getObservables(s));
      RooArgSet s2;
      s2.add(*_dglobs);
      std::unique_ptr<RooAbsCollection> _actualGlobs2(fPdf->getObservables(s2));
      if (!_actualGlobs->equals(*_actualGlobs2)) {
         RooArgSet rC;
         rC.add(*_actualGlobs2);
         rC.remove(*std::unique_ptr<RooAbsCollection>(rC.selectCommon(*_actualGlobs)));
         TString r = (!rC.empty()) ? rC.contentsString() : "";
         RooArgSet lC;
         lC.add(*_actualGlobs);
         lC.remove(*std::unique_ptr<RooAbsCollection>(lC.selectCommon(*_actualGlobs2)));
         TString l = (!lC.empty()) ? lC.contentsString() : "";
         throw std::runtime_error(TString::Format("globs mismatch: adding %s removing %s", r.Data(), l.Data()));
      }
      fGlobs = _dglobs;
   }

   if (!std::shared_ptr<RooAbsReal>::get()) {
      fData = _data.first;
      return true; // not loaded yet so nothing to do
   }

   try {
      if (!kReuseNLL                                                        /*|| !mainTerm()*/
          /*|| mainTerm()->operMode() == RooAbsTestStatistic::MPMaster*/) { // lost access to RooAbsTestStatistic
                                                                            // in 6.34, but MP-mode will still throw
                                                                            // exception, so we will still catch it
         // happens when using MP need to rebuild the nll instead
         // also happens if there's no mainTerm(), which is the case in 6.32 where RooNLLVar is partially deprecated
         AutoRestorer snap(*fFuncVars);
         // ensure the const state is back where it was at nll construction time;
         fFuncVars->setAttribAll("Constant", false);
         fConstVars->setAttribAll("Constant", true);
         std::shared_ptr<RooAbsData> __data = fData; // do this just to keep fData alive while killing previous NLLVar
                                                     // (can't kill data while NLL constructed with it)
         fData = _data.first;
         reinitialize();
         return true;
      }
      bool out = false;
      if (_data.first) {
         // replace in all terms
         out = get()->setData(*_data.first, false /* clone data */);
         //         get()->setValueDirty();
         //         if (_data.first->getGlobalObservables()) {
         //            // replace in all terms
         //            out = get()->setData(*_data.first, false);
         //            get()->setValueDirty();
         //         } else {
         //            // replace just in mainTerm ... note to self: why not just replace in all like above? should
         //            test! auto _mainTerm = mainTerm(); out = _mainTerm->setData(*_data.first, false /* clone data?
         //            */); _mainTerm->setValueDirty();
         //         }
      } else {
         reset();
      }
      fData = _data.first;
      return out;
   } catch (std::runtime_error &) {
      // happens when using MP need to rebuild the nll instead
      // also happens if there's no mainTerm(), which is the case in 6.32 where RooNLLVar is partially deprecated
      AutoRestorer snap(*fFuncVars);
      // ensure the const state is back where it was at nll construction time;
      fFuncVars->setAttribAll("Constant", false);
      fConstVars->setAttribAll("Constant", true);
      std::shared_ptr<RooAbsData> __data = fData; // do this just to keep fData alive while killing previous NLLVar
                                                  // (can't kill data while NLL constructed with it)
      fData = _data.first;
      reinitialize();
      return true;
   }
   throw std::runtime_error("Unable to setData");
}

std::shared_ptr<RooAbsReal> xRooNLLVar::func() const
{
   if (!(*this)) {
      const_cast<xRooNLLVar *>(this)->reinitialize();
   } else if (auto f = std::unique_ptr<RooAbsCollection>(fConstVars->selectByAttrib("Constant", false)); !f->empty()) {
      // have to reinitialize if const par values have changed - const optimization forces this
      // TODO: currently changes to globs also triggers this since the vars includes globs (vars are the non-obs pars)
      // std::cout << "Reinitializing because of change of const parameters:" << f->contentsString() << std::endl;
      const_cast<xRooNLLVar *>(this)->reinitialize();

      // note ... it may be sufficient here to do:
      // nll.constOptimizeTestStatistic(RooAbsArg::ConfigChange, constOptimize>1 /* do tracking too if >1 */); //
      // trigger a re-evaluate of which nodes to cache-and-track nll.constOptimizeTestStatistic(RooAbsArg::ValueChange,
      // constOptimize>1); // update the cache values -- is this needed??
      // this forces the optimization to be redone
      // for now leave as a reinitialize though, until had a chance to test this properly
   }
   if (fGlobs && fFuncGlobs) {
      *fFuncGlobs = *fGlobs;
      fFuncGlobs->setAttribAll("Constant", true);
   }
   return *this;
}

void xRooNLLVar::AddOption(const RooCmdArg &opt)
{
   fOpts->Add(opt.Clone(nullptr));
   if (std::shared_ptr<RooAbsReal>::get()) {
      reinitialize(); // do this way to keep name of nll if user set
   } else {
      reset(); // will trigger reinitialize
   }
}

RooAbsData *xRooNLLVar::data() const
{
   return fData.get();
   /*
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 33, 00)
   auto _nll = mainTerm();
   if (!_nll)
      return fData.get();
   RooAbsData *out = &static_cast<RooAbsOptTestStatistic*>(_nll)->data();
#else
   RooAbsData* out = nullptr; // new backends not conducive to having a reference to a RooAbsData in them (they use
buffers instead) #endif if (!out) return fData.get(); return out;
    */
}

/*
RooAbsReal *xRooNLLVar::mainTerm() const
{
   return nullptr;
   // the main term is the "other term" in a RooAddition alongside a ConstraintSum
   // if can't find the ConstraintSum, just return the function

   RooAbsArg* _func = func().get();
   if(!_func->InheritsFrom("RooAddition")) {
      _func = nullptr;
      // happens with new 6.32 backend, where the top-level function is an EvaluatorWrapper
      for (auto s : func()->servers()) {
         if(s->InheritsFrom("RooAddition")) {
            _func = s; break;
         }
      }
      if(!_func) {
         return func().get();
      }
   }
   std::set<RooAbsArg*> others,constraints;
   for (auto s : _func->servers()) {
      if(s->InheritsFrom("RooConstraintSum")) {
         constraints.insert(s);
      } else {
         others.insert(s);
      }
   }
   if(constraints.size()==1 && others.size()==1) {
      return static_cast<RooAbsReal*>(*others.begin());
   }
   return nullptr; // failed to find the right term?


}
 */

double xRooNLLVar::extendedTermVal() const
{
   // returns Nexp - Nobs*log(Nexp)
   return fPdf->extendedTerm(fData->sumEntries(), fData->get());
}

double xRooNLLVar::simTermVal() const
{
   // comes from the _simCount code inside RooNLLVar
   // is this actually only appropriate if the roosimultaneous is not extended?
   // i.e. then this term represents the probability the entry belongs to a given state, and given
   // all the states are normalized to 1, this probability is assumed to just be 1/N_states
   if (auto s = dynamic_cast<RooSimultaneous *>(fPdf.get()); s) {
      return fData->sumEntries() * log(1.0 * (s->servers().size() - 1)); // one of the servers is the cat
   }
   return 0;
}

double xRooNLLVar::binnedDataTermVal() const
{
   // this is only relevant if BinnedLikelihood active
   // = sum[ N_i! ] since LnGamma(N_i+1) ~= N_i!
   // need to also subtract off sum[ N_i*log(width_i) ] in order to have formula: binnedLL = unbinnedLL + binnedDataTerm
   // note this is 0 if all the bin widths are 1
   double out = 0;
   for (int i = 0; i < fData->numEntries(); i++) {
      fData->get(i);
      out += TMath::LnGamma(fData->weight() + 1) - fData->weight() * std::log(getEntryBinWidth(i));
   }

   return out;
}

RooConstraintSum *xRooNLLVar::constraintTerm() const
{
   auto _func = func();
   if (auto a = dynamic_cast<RooConstraintSum *>(_func.get()); a)
      return a;
   for (auto s : _func->servers()) {
      if (auto a = dynamic_cast<RooConstraintSum *>(s); a)
         return a;
      // allow one more depth to support 6.32 (where sum is hidden inside the first server)
      for (auto s2 : s->servers()) {
         if (auto a2 = dynamic_cast<RooConstraintSum *>(s2); a2)
            return a2;
      }
   }
   return nullptr;
}

/*xRooNLLVar::operator RooAbsReal &() const {
    // this works in c++ but not in python
    std::cout << "implicit conversion" << std::endl;
    return *fFunc;
}*/

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::getVal(const char *what)
{
   TString sWhat(what);
   sWhat.ToLower();
   bool doTS = sWhat.Contains("ts");
   bool doCLs = sWhat.Contains("pcls");
   bool doNull = sWhat.Contains("pnull");
   bool doAlt = sWhat.Contains("palt");
   double nSigma = (sWhat.Contains("exp"))
                      ? (TString(sWhat(sWhat.Index("exp") + 3, sWhat.Index(" ", sWhat.Index("exp")) == -1
                                                                  ? sWhat.Length()
                                                                  : sWhat.Index(" ", sWhat.Index("exp"))))
                            .Atof())
                      : std::numeric_limits<double>::quiet_NaN();

   bool toys = sWhat.Contains("toys");

   // bool asymp = sWhat.Contains("asymp");

   bool readOnly = sWhat.Contains("readonly");

   if (!readOnly) {
      if (toys) {
         sigma_mu(); // means we will be able to evaluate the asymptotic values too
      }
      // only add toys if actually required
      if (getVal(sWhat + " readonly").second != 0) {
         if (sWhat.Contains("toys=")) {
            // extract number of toys required ... format is "nullToys.altToysFraction" if altToysFraction=0 then use
            // same for both, unless explicitly set (i.e. N.0) then means we want no alt toys
            // e.g. if doing just pnull significance
            TString toyNum = sWhat(sWhat.Index("toys=") + 5, sWhat.Length());
            size_t nToys = toyNum.Atoi();
            size_t nToysAlt = (toyNum.Atof() - nToys)*nToys;
            if (nToysAlt == 0 && !toyNum.Contains('.'))
               nToysAlt = nToys;
            if (nullToys.size() < nToys) {
               addNullToys(nToys - nullToys.size());
            }
            if (altToys.size() < nToysAlt) {
               addAltToys(nToysAlt - altToys.size());
            }
         } else if (doCLs && toys) {
            // auto toy-generating for limits .. do in blocks of 100
            addCLsToys(100, 0, 0.05, nSigma);
         } else if(toys) {
            throw std::runtime_error("Auto-generating toys for anything other than CLs not yet supported, please specify number of toys with 'toys=N' ");
         }
      }
   }

   struct RestoreNll {
      RestoreNll(std::shared_ptr<xRooNLLVar> &v, bool r) : rr(r), var(v)
      {
         if (rr && var && var->get()) {
            _readOnly = var->get()->getAttribute("readOnly");
            var->get()->setAttribute("readOnly", rr);
         } else {
            rr = false;
         }
      };
      ~RestoreNll()
      {
         if (rr)
            var->get()->setAttribute("readOnly", _readOnly);
      };

      bool rr = false;
      bool _readOnly = false;

      std::shared_ptr<xRooNLLVar> &var;
   };

   RestoreNll rest(nllVar, readOnly);

   if (doTS)
      return (toys) ? ts_toys(nSigma) : ts_asymp(nSigma);
   if (doNull)
      return (toys) ? pNull_toys(nSigma) : pNull_asymp(nSigma);
   if (doAlt)
      return (toys) ? pAlt_toys(nSigma) : pAlt_asymp(nSigma);
   if (doCLs)
      return (toys) ? pCLs_toys(nSigma) : pCLs_asymp(nSigma);

   throw std::runtime_error(std::string("Unknown: ") + what);
}

RooArgList xRooNLLVar::xRooHypoPoint::poi() const
{
   RooArgList out;
   out.setName("poi");
   out.add(*std::unique_ptr<RooAbsCollection>(coords->selectByAttrib("poi", true)));
   return out;
}

RooArgList xRooNLLVar::xRooHypoPoint::alt_poi() const
{
   RooArgList out;
   out.setName("alt_poi");
   out.addClone(*std::unique_ptr<RooAbsCollection>(coords->selectByAttrib("poi", true)));
   for (auto a : out) {
      auto v = dynamic_cast<RooAbsRealLValue *>(a);
      if (!v)
         continue;
      if (auto s = a->getStringAttribute("altVal"); s && strlen(s)) {
         v->setVal(TString(s).Atof());
      } else {
         v->setVal(std::numeric_limits<double>::quiet_NaN());
      }
   }
   return out;
}

int xRooNLLVar::xRooHypoPoint::status() const
{
   auto &me = const_cast<xRooHypoPoint &>(*this);
   int out = 0;
   if (me.ufit(true) && !allowedStatusCodes.count(me.ufit(true)->status()))
      out += 1;
   if (me.cfit_null(true) && !allowedStatusCodes.count(me.cfit_null(true)->status()))
      out += 1 << 1;
   if (me.cfit_alt(true) && !allowedStatusCodes.count(me.cfit_alt(true)->status()))
      out += 1 << 2;
   if (me.asimov(true))
      out += me.asimov(true)->status() << 3;
   return out;
}

void xRooNLLVar::xRooHypoPoint::Print(Option_t *) const
{
   auto _poi = const_cast<xRooHypoPoint *>(this)->poi();
   auto _alt_poi = const_cast<xRooHypoPoint *>(this)->alt_poi();
   std::cout << "POI: " << _poi.contentsString() << " , null: ";
   bool first = true;
   for (auto a : _poi) {
      auto v = dynamic_cast<RooAbsReal *>(a);
      if (!a)
         continue;
      if (!first)
         std::cout << ",";
      std::cout << v->getVal();
      first = false;
   }
   std::cout << " , alt: ";
   first = true;
   bool any_alt = false;
   for (auto a : _alt_poi) {
      auto v = dynamic_cast<RooAbsReal *>(a);
      if (!a)
         continue;
      if (!first)
         std::cout << ",";
      std::cout << v->getVal();
      first = false;
      if (!std::isnan(v->getVal()))
         any_alt = true;
   }
   std::cout << " , pllType: " << fPllType << std::endl;

   std::cout << " -        ufit: ";
   if (fUfit) {
      std::cout << fUfit->GetName() << " " << fUfit->minNll() << " (status=" << fUfit->status() << ") (";
      first = true;
      for (auto a : _poi) {
         auto v = dynamic_cast<RooRealVar *>(fUfit->floatParsFinal().find(a->GetName()));
         if (!v)
            continue;
         if (!first)
            std::cout << ",";
         std::cout << v->GetName() << "_hat: " << v->getVal() << " +/- " << v->getError();
         first = false;
      }
      std::cout << ")" << std::endl;
   } else {
      std::cout << "Not calculated" << std::endl;
   }
   std::cout << " -   cfit_null: ";
   if (fNull_cfit) {
      std::cout << fNull_cfit->GetName() << " " << fNull_cfit->minNll() << " (status=" << fNull_cfit->status() << ")";
   } else {
      std::cout << "Not calculated";
   }
   if (any_alt) {
      std::cout << std::endl << " -    cfit_alt: ";
      if (fAlt_cfit) {
         std::cout << fAlt_cfit->GetName() << " " << fAlt_cfit->minNll() << " (status=" << fAlt_cfit->status() << ")"
                   << std::endl;
      } else {
         std::cout << "Not calculated" << std::endl;
      }
      std::cout << " sigma_mu: ";
      const_cast<xRooHypoPoint *>(this)->asimov(true); // will trigger construction of fAsimov hypoPoint if possible
      if (!fAsimov || !fAsimov->fUfit || !fAsimov->fNull_cfit) {
         std::cout << "Not calculated";
      } else {
         std::cout << const_cast<xRooHypoPoint *>(this)->sigma_mu().first << " +/- "
                   << const_cast<xRooHypoPoint *>(this)->sigma_mu().second;
      }
      if (fAsimov) {
         std::cout << std::endl;
         std::cout << "   - asimov ufit: ";
         if (fAsimov->fUfit) {
            std::cout << fAsimov->fUfit->GetName() << " " << fAsimov->fUfit->minNll()
                      << " (status=" << fAsimov->fUfit->status() << ")";
         } else {
            std::cout << "Not calculated";
         }
         std::cout << std::endl << "   - asimov cfit_null: ";
         if (fAsimov->fNull_cfit) {
            std::cout << fAsimov->fNull_cfit->GetName() << " " << fAsimov->fNull_cfit->minNll()
                      << " (status=" << fAsimov->fNull_cfit->status() << ")";
         } else {
            std::cout << "Not calculated";
         }
      }
      std::cout << std::endl;
   } else {
      std::cout << std::endl;
   }
   if (fLbound_cfit) {
      std::cout << " - cfit_lbound: " << fLbound_cfit->GetName() << " " << fLbound_cfit->minNll()
                << " (status=" << fLbound_cfit->status() << ")";
   }
   if (fGenFit)
      std::cout << " -      gfit: " << fGenFit->GetName() << std::endl;
   if (!nullToys.empty() || !altToys.empty()) {
      std::cout << " *   null toys: " << nullToys.size();
      size_t firstToy = 0;
      while (firstToy < nullToys.size() && std::isnan(std::get<1>(nullToys[firstToy])))
         firstToy++;
      if (firstToy > 0)
         std::cout << " [ of which " << firstToy << " are bad]";
      std::cout << " , alt toys: " << altToys.size();
      firstToy = 0;
      while (firstToy < altToys.size() && std::isnan(std::get<1>(altToys[firstToy])))
         firstToy++;
      if (firstToy > 0)
         std::cout << " [ of which " << firstToy << " are bad]";
      std::cout << std::endl;
   }
   // std::cout << " nllVar: " << nllVar << std::endl;
}

RooRealVar &xRooNLLVar::xRooHypoPoint::mu_hat()
{
   if (ufit()) {
      auto var = dynamic_cast<RooRealVar *>(ufit()->floatParsFinal().find(fPOIName()));
      if (var) {
         return *var;
      } else {
         throw std::runtime_error(TString::Format("Cannot find POI: %s", fPOIName()));
      }
   }
   throw std::runtime_error("Unconditional fit unavailable");
}

std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> xRooNLLVar::xRooHypoPoint::data()
{
   if (fData.first)
      return fData;
   if (fGenFit && isExpected) {
      // std::cout << "Generating asimov" << std::endl;poi().Print("v");
      fData = xRooFit::generateFrom(*nllVar->fPdf, *fGenFit, true);
   }
   return fData;
}

xRooNLLVar::xRooHypoPoint::xRooHypoPoint(std::shared_ptr<RooStats::HypoTestResult> htr, const RooAbsCollection *_coords)
   : hypoTestResult(htr)
{
   if (hypoTestResult) {
      // load the pllType
      fPllType =
         xRooFit::Asymptotics::PLLType(hypoTestResult->GetFitInfo()->getGlobalObservables()->getCatIndex("pllType"));
      isExpected = hypoTestResult->GetFitInfo()->getGlobalObservables()->getRealValue("isExpected");

      // load the toys
      auto toys = hypoTestResult->GetNullDetailedOutput();
      if (toys) {
         // load coords from the nullDist globs list
         if (toys->getGlobalObservables()) {
            coords = std::shared_ptr<RooAbsCollection>(toys->getGlobalObservables()->snapshot());
         }
         for (int i = 0; i < toys->numEntries(); i++) {
            auto toy = toys->get(i);
            nullToys.emplace_back(
               std::make_tuple(int(toy->getRealValue("seed")), toy->getRealValue("ts"), toys->weight()));
         }
      }
      toys = hypoTestResult->GetAltDetailedOutput();
      if (toys) {
         for (int i = 0; i < toys->numEntries(); i++) {
            auto toy = toys->get(i);
            altToys.emplace_back(
               std::make_tuple(int(toy->getRealValue("seed")), toy->getRealValue("ts"), toys->weight()));
         }
      }
   }
   if (!coords && _coords)
      coords.reset(_coords->snapshot());
}

std::shared_ptr<xRooNLLVar::xRooHypoPoint> xRooNLLVar::xRooHypoPoint::asimov(bool readOnly)
{

   if (!fAsimov && (nllVar || hypoTestResult)) {
      auto theFit = (!fData.first && fGenFit && !isExpected)
                       ? fGenFit
                       : cfit_alt(readOnly); // first condition allows genFit to be used as the altFit *if* the data is
                                             // entirely absent, provided not expected data because we postpone data
                                             // creation til later in that case (see below)
      if (!theFit || allowedStatusCodes.find(theFit->status()) == allowedStatusCodes.end())
         return fAsimov;
      fAsimov = std::make_shared<xRooHypoPoint>(*this);
      fAsimov->coords.reset(fAsimov->coords->snapshot()); // create a copy so can remove the physical range below
      fAsimov->hypoTestResult.reset();
      fAsimov->fPllType = xRooFit::Asymptotics::TwoSided;
      for (auto p : fAsimov->poi()) {
         // dynamic_cast<RooRealVar *>(p)->removeRange("physical"); -- can't use this as will modify shared property
         if (auto v = dynamic_cast<RooRealVar *>(p)) {
            v->deleteSharedProperties(); // effectively removes all custom ranges
         }
      }

      fAsimov->nullToys.clear();
      fAsimov->altToys.clear();
      fAsimov->fUfit = retrieveFit(3);
      fAsimov->fNull_cfit = retrieveFit(4);
      fAsimov->fAlt_cfit.reset();
      fAsimov->fData =
         std::make_pair(nullptr, nullptr); // postpone generating expected data until we definitely need it
      fAsimov->fGenFit = theFit;
      fAsimov->isExpected = true;
   }

   return fAsimov;
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pNull_asymp(double nSigma)
{
   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair<double, double>(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   auto _sigma_mu = sigma_mu();
   double nom = xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fNullVal(), _sigma_mu.first,
                                             first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   return std::pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pAlt_asymp(double nSigma)
{
   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair<double, double>(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   auto _sigma_mu = sigma_mu();
   double nom = xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fAltVal(), _sigma_mu.first,
                                             first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));

   return std::pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pCLs_asymp(double nSigma)
{
   if (fNullVal() == fAltVal())
      return std::pair<double, double>(1, 0); // by construction

   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair<double, double>(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);

   auto _ts_asymp = ts_asymp(nSigma);
   auto _sigma_mu = sigma_mu();
   double nom1 = xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first, fNullVal(), fNullVal(), _sigma_mu.first,
                                              first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up1 =
      xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first + _ts_asymp.second, fNullVal(), fNullVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down1 =
      xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first - _ts_asymp.second, fNullVal(), fNullVal(),
                                   _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double nom2 = xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first, fNullVal(), fAltVal(), _sigma_mu.first,
                                              first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up2 =
      xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first + _ts_asymp.second, fNullVal(), fAltVal(), _sigma_mu.first,
                                   first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down2 =
      xRooFit::Asymptotics::PValue(fPllType, _ts_asymp.first - _ts_asymp.second, fNullVal(), fAltVal(), _sigma_mu.first,
                                   first_poi->getMin("physical"), first_poi->getMax("physical"));

   auto nom = (nom1 == 0) ? 0 : nom1 / nom2;
   auto up = (up1 == 0) ? 0 : up1 / up2;
   auto down = (down1 == 0) ? 0 : down1 / down2;

   return std::pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::ts_asymp(double nSigma)
{
   if (std::isnan(nSigma))
      return pll();
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   auto _sigma_mu = sigma_mu();
   if (!first_poi || (!std::isnan(nSigma) && std::isnan(_sigma_mu.first)))
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   double nom = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                        _sigma_mu.first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                       _sigma_mu.first + _sigma_mu.second, first_poi->getMin("physical"),
                                       first_poi->getMax("physical"));
   double down = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                         _sigma_mu.first - _sigma_mu.second, first_poi->getMin("physical"),
                                         first_poi->getMax("physical"));
   return std::pair<double, double>(nom, std::max(std::abs(nom - up), std::abs(nom - down)));
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::ts_toys(double nSigma)
{
   if (std::isnan(nSigma))
      return pll();
   // nans should appear in the alt toys first ... so loop until past nans
   size_t firstToy = 0;
   while (firstToy < altToys.size() && std::isnan(std::get<1>(altToys[firstToy])))
      firstToy++;
   if (firstToy >= altToys.size())
      return std::pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
   int targetIdx =
      (altToys.size() - firstToy) * ROOT::Math::gaussian_cdf(nSigma) + firstToy; // TODO: Account for weights
   return std::pair(std::get<1>(altToys[targetIdx]), (std::get<1>(altToys[std::min(int(altToys.size()), targetIdx)]) -
                                                      std::get<1>(altToys[std::max(0, targetIdx)])) /
                                                        2.);
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pll(bool readOnly)
{
   auto _ufit = ufit(readOnly);
   if (!_ufit) {
      if (hypoTestResult)
         return std::pair<double, double>(hypoTestResult->GetTestStatisticData(), 0);
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   }
   if (allowedStatusCodes.find(_ufit->status()) == allowedStatusCodes.end()) {
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   }
   if (auto _first_poi = dynamic_cast<RooRealVar *>(poi().first());
       _first_poi && _first_poi->getMin("physical") > _first_poi->getMin() &&
       mu_hat().getVal() < _first_poi->getMin("physical")) {
      // replace _ufit with fit "boundary" conditional fit
      _ufit = cfit_lbound(readOnly);
      if (!_ufit) {
         return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
      }
   }
   auto cFactor = (fPllType == xRooFit::Asymptotics::TwoSided)
                     ? 1.
                     : xRooFit::Asymptotics::CompatFactor(fPllType, fNullVal(), mu_hat().getVal());
   if (cFactor == 0)
      return std::pair<double, double>(0, 0);
   if (!cfit_null(readOnly) || allowedStatusCodes.find(cfit_null(readOnly)->status()) == allowedStatusCodes.end())
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   // std::cout << cfit->minNll() << ":" << cfit->edm() << " " << ufit->minNll() << ":" << ufit->edm() << std::endl;
   return std::pair<double, double>(2. * cFactor * (cfit_null(readOnly)->minNll() - _ufit->minNll()),
                                    2. * cFactor * sqrt(pow(cfit_null(readOnly)->edm(), 2) + pow(_ufit->edm(), 2)));
   // return 2.*cFactor*(cfit->minNll()+cfit->edm() - ufit->minNll()+ufit->edm());
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::retrieveFit(int type)
{
   if (!hypoTestResult)
      return nullptr;
   // see if can retrieve from that ....
   if (auto fits = hypoTestResult->GetFitInfo()) {
      for (int i = 0; i < fits->numEntries(); i++) {
         auto fit = fits->get(i);
         if (fit->getCatIndex("type") != type)
            continue;
         // found ufit ... construct
         std::string _name =
            fits->getGlobalObservables()->getStringValue(TString::Format("%s.name", fit->getCatLabel("type")));
         // see if can retrieve from any open file ....
         TDirectory *tmp = gDirectory;
         for (auto file : *gROOT->GetListOfFiles()) {
            if (auto k = static_cast<TDirectory *>(file)->FindKeyAny(_name.c_str())) {
               // use pre-retrieved fits if available
               xRooFit::StoredFitResult *storedFr =
                  k->GetMotherDir()->GetList()
                     ? dynamic_cast<xRooFit::StoredFitResult *>(k->GetMotherDir()->GetList()->FindObject(k->GetName()))
                     : nullptr;
               if (auto cachedFit = (storedFr) ? storedFr->fr.get() : k->ReadObject<RooFitResult>(); cachedFit) {
                  if (!storedFr) {
                     storedFr = new xRooFit::StoredFitResult(cachedFit);
                     k->GetMotherDir()->Add(storedFr);
                  }
                  gDirectory = tmp; // one of the above calls moves to key's directory ... i didn't check which
                  return storedFr->fr;
               }
            }
         }
         auto rfit = std::make_shared<RooFitResult>(_name.c_str(), TUUID(_name.c_str()).GetTime().AsString());
         rfit->setStatus(fit->getRealValue("status"));
         rfit->setMinNLL(fit->getRealValue("minNll"));
         rfit->setEDM(fit->getRealValue("edm"));
         if (type == 0) {
            std::unique_ptr<RooAbsCollection> par_hats(
               hypoTestResult->GetFitInfo()->getGlobalObservables()->selectByName(coords->contentsString().c_str()));
            par_hats->setName("floatParsFinal");
            rfit->setFinalParList(*par_hats);
         } else {
            rfit->setFinalParList(RooArgList());
         }
         rfit->setConstParList(RooArgList());
         rfit->setInitParList(RooArgList());
         TMatrixDSym cov(0);
         rfit->setCovarianceMatrix(cov);
         rfit->setCovQual(fit->getRealValue("covQual"));
         return rfit;
      }
   }
   return nullptr;
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::ufit(bool readOnly)
{
   if (fUfit)
      return fUfit;
   if (auto rfit = retrieveFit(0)) {
      return fUfit = rfit;
   }
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   if (!fData.first) {
      if (!readOnly && isExpected && fGenFit) {
         // can try to do a readOnly in case can load from cache
         bool tmp = nllVar->get()->getAttribute("readOnly");
         nllVar->get()->setAttribute("readOnly");
         auto out = ufit(true);
         nllVar->get()->setAttribute("readOnly", tmp);
         if (out) {
            // retrieve from cache worked, no need to generate dataset
            return out;
         } else if (!tmp) { // don't need to setData if doing a readOnly fit
            nllVar->setData(data());
         }
      }
   } else if (!nllVar->get()->getAttribute("readOnly")) { // don't need to setData if doing a readOnly fit
      nllVar->setData(fData);
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   std::unique_ptr<RooAbsCollection>(nllVar->fFuncVars->selectCommon(poi()))
      ->setAttribAll("Constant", false); // float the poi
   if (fGenFit) {
      // make initial guess same as pars we generated with
      nllVar->fFuncVars->assignValueOnly(fGenFit->constPars());
      nllVar->fFuncVars->assignValueOnly(fGenFit->floatParsFinal());
      // rename nll so if caching fit results will cache into subdir
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
      if (!isExpected)
         nllVar->get()->SetName(TString::Format("%s/%s", nllVar->get()->GetName(), fData.first->GetName()));

   } else if (!std::isnan(fAltVal())) {
      // guess data given is expected to align with alt value, unless initVal attribute specified
      for (auto _poiCoord : poi()) {
         auto _poi = dynamic_cast<RooRealVar *>(nllVar->fFuncVars->find(_poiCoord->GetName()));
         if (_poi) {
            _poi->setVal(_poi->getStringAttribute("initVal") ? TString(_poi->getStringAttribute("initVal")).Atof()
                                                             : fAltVal());
         }
      }
   }
   return (fUfit = nllVar->minimize());
}

std::string collectionContents(const RooAbsCollection &coll)
{
   std::string out;
   for (auto &c : coll) {
      if (!out.empty())
         out += ",";
      out += c->GetName();
      if (auto v = dynamic_cast<RooAbsReal *>(c); v) {
         out += TString::Format("=%g", v->getVal());
      } else if (auto cc = dynamic_cast<RooAbsCategory *>(c); cc) {
         out += TString::Format("=%s", cc->getLabel());
      } else if (auto s = dynamic_cast<RooStringVar *>(c); v) {
         out += TString::Format("=%s", s->getVal());
      }
   }
   return out;
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::cfit_null(bool readOnly)
{
   if (fNull_cfit)
      return fNull_cfit;
   if (auto rfit = retrieveFit(1)) {
      return fNull_cfit = rfit;
   }
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   if (!fData.first) {
      if (!readOnly && isExpected && fGenFit) {
         // can try to do a readOnly in case can load from cache
         bool tmp = nllVar->get()->getAttribute("readOnly");
         nllVar->get()->setAttribute("readOnly");
         auto out = cfit_null(true);
         nllVar->get()->setAttribute("readOnly", tmp);
         if (out) {
            // retrieve from cache worked, no need to generate dataset
            return out;
         } else if (!tmp) { // don't need to setData if doing a readOnly fit
            nllVar->setData(data());
         }
      }
   } else if (!nllVar->get()->getAttribute("readOnly")) { // don't need to setData if doing a readOnly fit
      nllVar->setData(fData);
   }
   if (fUfit) {
      // move to ufit coords before evaluating
      *nllVar->fFuncVars = fUfit->floatParsFinal();
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   if (fPOIName()) {
      nllVar->fFuncVars->find(fPOIName())
         ->setStringAttribute("altVal", (!std::isnan(fAltVal())) ? TString::Format("%g", fAltVal()) : nullptr);
   }
   if (fGenFit) {
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
      if (!isExpected)
         nllVar->get()->SetName(TString::Format("%s/%s", nllVar->get()->GetName(), fData.first->GetName()));
   }
   nllVar->get()->setStringAttribute("fitresultTitle", collectionContents(poi()).c_str());
   return (fNull_cfit = nllVar->minimize());
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::cfit_lbound(bool readOnly)
{
   auto _first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!_first_poi)
      return nullptr;
   if (_first_poi->getMin("physical") <= _first_poi->getMin())
      return nullptr;
   if (fLbound_cfit)
      return fLbound_cfit;
   if (auto rfit = retrieveFit(6)) {
      return fLbound_cfit = rfit;
   }
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   if (!fData.first) {
      if (!readOnly && isExpected && fGenFit) {
         // can try to do a readOnly in case can load from cache
         bool tmp = nllVar->get()->getAttribute("readOnly");
         nllVar->get()->setAttribute("readOnly");
         auto out = cfit_lbound(true);
         nllVar->get()->setAttribute("readOnly", tmp);
         if (out) {
            // retrieve from cache worked, no need to generate dataset
            return out;
         } else if (!tmp) { // don't need to setData if doing a readOnly fit
            nllVar->setData(data());
         }
      }
   } else if (!nllVar->get()->getAttribute("readOnly")) { // don't need to setData if doing a readOnly fit
      nllVar->setData(fData);
   }
   if (fUfit) {
      // move to ufit coords before evaluating
      *nllVar->fFuncVars = fUfit->floatParsFinal();
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   nllVar->fFuncVars->setRealValue(_first_poi->GetName(), _first_poi->getMin("physical"));
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   if (fPOIName()) {
      nllVar->fFuncVars->find(fPOIName())
         ->setStringAttribute("altVal", (!std::isnan(fAltVal())) ? TString::Format("%g", fAltVal()) : nullptr);
   }
   if (fGenFit) {
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
      if (!isExpected)
         nllVar->get()->SetName(TString::Format("%s/%s", nllVar->get()->GetName(), fData.first->GetName()));
   }
   nllVar->get()->setStringAttribute(
      "fitresultTitle",
      collectionContents(*std::unique_ptr<RooAbsCollection>(nllVar->fFuncVars->selectCommon(poi()))).c_str());
   return (fLbound_cfit = nllVar->minimize());
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::cfit_alt(bool readOnly)
{
   if (std::isnan(fAltVal()))
      return nullptr;
   if (fAlt_cfit)
      return fAlt_cfit;
   if (auto rfit = retrieveFit(2)) {
      return fAlt_cfit = rfit;
   }
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   if (!fData.first) {
      if (!readOnly && isExpected && fGenFit) {
         // can try to do a readOnly in case can load from cache
         bool tmp = nllVar->get()->getAttribute("readOnly");
         nllVar->get()->setAttribute("readOnly");
         auto out = cfit_alt(true);
         nllVar->get()->setAttribute("readOnly", tmp);
         if (out) {
            // retrieve from cache worked, no need to generate dataset
            return out;
         } else if (!tmp) { // don't need to setData if doing a readOnly fit
            nllVar->setData(data());
         }
      }
   } else if (!nllVar->get()->getAttribute("readOnly")) { // don't need to setData if doing a readOnly fit
      nllVar->setData(fData);
   }
   if (fUfit) {
      // move to ufit coords before evaluating
      *nllVar->fFuncVars = fUfit->floatParsFinal();
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   *nllVar->fFuncVars = alt_poi();
   if (fGenFit) {
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
      if (!isExpected)
         nllVar->get()->SetName(TString::Format("%s/%s", nllVar->get()->GetName(), fData.first->GetName()));
   }
   nllVar->get()->setStringAttribute("fitresultTitle", collectionContents(alt_poi()).c_str());
   return (fAlt_cfit = nllVar->minimize());
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::sigma_mu(bool readOnly)
{

   auto asi = asimov(readOnly);

   if (!asi) {
      return std::pair<double, double>(std::numeric_limits<double>::quiet_NaN(), 0);
   }

   auto out = asi->pll(readOnly);
   return std::pair<double, double>(std::abs(fNullVal() - fAltVal()) / sqrt(out.first),
                                    out.second * 0.5 * std::abs(fNullVal() - fAltVal()) /
                                       (out.first * sqrt(out.first)));
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pX_toys(bool alt, double nSigma)
{
   auto _ts = ts_toys(nSigma);
   if (std::isnan(_ts.first))
      return _ts;
   if (fPllType != xRooFit::Asymptotics::Uncapped && _ts.first == 0)
      return std::pair<double, double>(1, 0); // don't need toys to compute this point!

   TEfficiency eff("", "", 1, 0, 1);

   auto &_theToys = (alt) ? altToys : nullToys;

   if (_theToys.empty()) {
      return std::pair(0.5, std::numeric_limits<double>::infinity());
   }

   // loop over toys, count how many are > ts value
   // nans (mean bad ts evaluations) will count towards uncertainty
   int nans = 0;
   double result = 0;
   double result_err_up = 0;
   double result_err_down = 0;
   for (auto &toy : _theToys) {
      if (std::isnan(std::get<1>(toy))) {
         nans++;
      } else {
         bool res = std::get<1>(toy) >= _ts.first;
         if (std::get<2>(toy) != 1) {
            eff.FillWeighted(res, 0.5, std::get<2>(toy));
         } else {
            eff.Fill(res, 0.5);
         }
         if (res)
            result += std::get<2>(toy);
         if (std::get<1>(toy) >= _ts.first - _ts.second)
            result_err_up += std::get<2>(toy);
         if (std::get<1>(toy) >= _ts.first - _ts.second)
            result_err_down += std::get<2>(toy);
      }
   }
   // symmetrize the error
   result_err_up -= result;
   result_err_down -= result;
   double result_err = std::max(std::abs(result_err_up), std::abs(result_err_down));
   // assume the nans would "add" to the p-value, conservative scenario
   result_err += nans;
   result_err /= _theToys.size();

   // don't include the nans for the central value though
   result /= (_theToys.size() - nans);

   // add to the result_err (in quadrature) the uncert due to limited stats
   result_err = sqrt(result_err * result_err + eff.GetEfficiencyErrorUp(1) * eff.GetEfficiencyErrorUp(1));
   return std::pair<double, double>(result, result_err);
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pNull_toys(double nSigma)
{
   return pX_toys(false, nSigma);
}

xRooNLLVar::xValueWithError xRooNLLVar::xRooHypoPoint::pAlt_toys(double nSigma)
{
   if (!std::isnan(nSigma)) {
      return std::pair<double, double>(ROOT::Math::gaussian_cdf(nSigma), 0); // by construction
   }
   return pX_toys(true, nSigma);
}

xRooNLLVar::xRooHypoPoint xRooNLLVar::xRooHypoPoint::generateNull(int seed)
{
   xRooHypoPoint out;
   out.coords = coords;
   out.fPllType = fPllType; // out.fPOIName = fPOIName; out.fNullVal=fNullVal; out.fAltVal = fAltVal;
   out.nllVar = nllVar;
   if (!nllVar)
      return out;
   auto _cfit = cfit_null();
   if (!_cfit)
      return out;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   //*nllVar->fFuncVars = cfit_null()->floatParsFinal();
   //*nllVar->fFuncVars = cfit_null()->constPars();
   out.fData = xRooFit::generateFrom(*nllVar->fPdf, *_cfit, false, seed); // nllVar->generate(false,seed);
   out.fGenFit = _cfit;
   return out;
}

xRooNLLVar::xRooHypoPoint xRooNLLVar::xRooHypoPoint::generateAlt(int seed)
{
   xRooHypoPoint out;
   out.coords = coords;
   out.fPllType = fPllType; // out.fPOIName = fPOIName; out.fNullVal=fNullVal; out.fAltVal = fAltVal;
   out.nllVar = nllVar;
   if (!nllVar)
      return out;
   if (!cfit_alt())
      return out;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   //*nllVar->fFuncVars = cfit_alt()->floatParsFinal();
   //*nllVar->fFuncVars = cfit_alt()->constPars();
   out.fData =
      xRooFit::generateFrom(*nllVar->fPdf, *cfit_alt(), false, seed); // out.data = nllVar->generate(false,seed);
   out.fGenFit = cfit_alt();
   return out;
}

size_t xRooNLLVar::xRooHypoPoint::addToys(bool alt, int nToys, int initialSeed, double target, double target_nSigma,
                                          bool targetCLs, double relErrThreshold, size_t maxToys)
{
   if ((alt && !cfit_alt()) || (!alt && !cfit_null())) {
      throw std::runtime_error("Cannot add toys, invalid conditional fit");
   }

   auto condition = [&]() { // returns true if need more toys
      if (std::isnan(target))
         return false;
      auto obs = targetCLs ? pCLs_toys(target_nSigma) : (alt ? pAlt_toys(target_nSigma) : pNull_toys(target_nSigma));
      if (!std::isnan(obs.first)) {
         double diff = (target < 0) ? obs.first : std::abs(obs.first - target);
         double err = obs.second;
         if (err > 1e-4 && diff <= relErrThreshold * obs.second) {
            // return true; // more toys needed
            if (targetCLs) {
               // decide which type we'd want to generate and update alt flag
               auto pNull = pNull_toys(target_nSigma);
               auto pAlt = pAlt_toys(target_nSigma);
               // std::cout << obs.first << " +/- " << obs.second << ": " << pNull.first << " +/- " << pNull.second << "
               // , " << pAlt.first << " +/- " << pAlt.second << std::endl;
               alt = (pAlt.second * pNull.first > pNull.second * pAlt.first);
               if ((alt ? pAlt.second : pNull.second) < 1e-4)
                  return false; // stop if error gets too small
            }
            return true;
         }
      }
      return false;
   };

   if (!std::isnan(target) && std::isnan(ts_toys(target_nSigma).first)) {
      if (std::isnan(target_nSigma)) {
         throw std::runtime_error("Cannot target obs p-value because ts value unavailable");
      }
      if (targetCLs && pCLs_toys(target_nSigma).second == 0) {
         // this happens if the mu_test=mu_alt ... no toys needed
         return 0;
      }

      // try generating 100 alt toys
      Info("addToys", "First generating 100 alt toys in order to determine expected ts value");
      addToys(true, 100, initialSeed);
      // if still null then exit
      if (std::isnan(ts_toys(target_nSigma).first)) {
         throw std::runtime_error("Unable to determine expected ts value");
      }
   }

   size_t nans = 0;
   float lastTime = 0;
   int lasti = 0;
   auto g = gROOT->Get<TGraph>("toyTime");
   if (!g) {
      g = new TGraph;
      g->SetNameTitle("toyTime", "Time per toy;Toy;time [s]");
      gROOT->Add(g);
   }
   g->Set(0);
   TStopwatch s2;
   s2.Start();
   TStopwatch s;
   s.Start();

   size_t toysAdded(0);
   size_t altToysAdded(0);
   if (initialSeed) {
      RooRandom::randomGenerator()->SetSeed(initialSeed);
   }
   do {
      auto &toys = (alt) ? altToys : nullToys;
      if (toys.size() >= maxToys) {
         // cannot generate more toys, reached limit already
         break;
      }
      // don't generate toys if reached target
      if (!std::isnan(target) && !condition()) {
         break;
      }
      auto currVal = std::isnan(target) ? std::pair(0., 0.)
                                        : (targetCLs ? pCLs_toys(target_nSigma)
                                                     : (alt ? pAlt_toys(target_nSigma) : pNull_toys(target_nSigma)));
      size_t nnToys = std::min(size_t(nToys), (maxToys - toys.size()));

      for (size_t i = 0; i < nnToys; i++) {
         int seed = RooRandom::randomGenerator()->Integer(std::numeric_limits<uint32_t>::max());
         auto toy = ((alt) ? generateAlt(seed) : generateNull(seed));
         TDirectory *tmp = gDirectory;
         gDirectory = nullptr; // disables any saving of fit results for toys
         toys.push_back(std::make_tuple(seed, toy.pll().first, 1.));
         gDirectory = tmp;
         (alt ? altToysAdded : toysAdded)++;
         if (std::isnan(std::get<1>(toys.back())))
            nans++;
         g->SetPoint(g->GetN(), g->GetN(), s.RealTime() - lastTime); // stops the clock
         lastTime = s.RealTime();
         if (s.RealTime() > 10) {
            std::cout << "\r"
                      << TString::Format("Generated %d/%d %s hypothesis toys [%.2f toys/s]",
                                         int(alt ? altToysAdded : toysAdded), int(nnToys), alt ? "alt" : "null",
                                         double(altToysAdded + toysAdded - lasti) / s.RealTime());
            if (!std::isnan(target)) {
               std::cout << " [current=" << currVal.first << "+/-" << currVal.second << " target=" << target
                         << " nSigma=" << target_nSigma << "]";
            }
            std::cout << "..." << std::flush;
            lasti = altToysAdded + toysAdded;
            s.Reset();
            if(!gROOT->IsBatch()) {
               Draw();
               if (gPad) {
                  gPad->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
                  gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
                  gSystem->ProcessEvents();
               }
            }
            s.Start();
            // std::cout << "Generated " << i << "/" << nToys << (alt ? " alt " : " null ") << " hypothesis toys " ..."
            // << std::endl;
         }
         s.Continue();
      }
      // sort the toys ... put nans first - do by setting all as negative inf (is that still necessary with the custom
      // sort below??)
      for (auto &t : toys) {
         if (std::isnan(std::get<1>(t)))
            std::get<1>(t) = -std::numeric_limits<double>::infinity();
      }
      std::sort(toys.begin(), toys.end(),
                [](const decltype(nullToys)::value_type &a, const decltype(nullToys)::value_type &b) -> bool {
                   if (std::isnan(std::get<1>(a)))
                      return true;
                   if (std::isnan(std::get<1>(b)))
                      return false;
                   return std::get<1>(a) < std::get<1>(b);
                });
      for (auto &t : toys) {
         if (std::isinf(std::get<1>(t)))
            std::get<1>(t) = std::numeric_limits<double>::quiet_NaN();
      }
      if (std::isnan(target)) {
         break; // no more toys if not doing a target
      }
      // if(condition()) {
      //    Info("addToys","Generating more toys to determine p-value ... currently: %f +/-
      //    %f",pNull_toys(target_nSigma).first,pNull_toys(target_nSigma).second);
      // }
   } while (condition());
   if (lasti) {
      std::cout << "\r"
                << "Finished Generating ";
      if (toysAdded) {
         std::cout << toysAdded << " null ";
      }
      if (altToysAdded) {
         std::cout << altToysAdded << " alt ";
      }
      std::cout << "toys " << TString::Format("[%.2f toys/s overall]", double(toysAdded + altToysAdded) / s2.RealTime())
                << std::endl;
      if(!gROOT->IsBatch()) {
         Draw();
         if (gPad) {
            gPad->Update();
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 30, 00)
            gPad->GetCanvas()->ResetUpdated(); // stops previous canvas being replaced in a jupyter notebook
#endif
            gSystem->ProcessEvents();
         }
      }
   }

   if (nans > 0) {
      std::cout << "Warning: " << nans << " toys were bad" << std::endl;
   }
   return toysAdded;
}

void xRooNLLVar::xRooHypoPoint::addNullToys(int nToys, int seed, double target, double target_nSigma)
{
   addToys(false, nToys, seed, target, target_nSigma);
}
void xRooNLLVar::xRooHypoPoint::addAltToys(int nToys, int seed, double target, double target_nSigma)
{
   addToys(true, nToys, seed, target, target_nSigma);
}

void xRooNLLVar::xRooHypoPoint::addCLsToys(int nToys, int seed, double target, double target_nSigma)
{
   addToys(false, nToys, seed, target, target_nSigma, true);
   return;
   //
   //   auto condition = [&](bool doingAlt=false) { // returns true if need more toys
   //      if(std::isnan(target)) return false;
   //      auto pval = pCLs_toys(target_nSigma);
   //      if (!std::isnan(pval.first)) {
   //         double diff = std::abs(pval.first - target);
   //         double err = pval.second;
   //         if (err > 1e-4 && diff <= 2 * pval.second) {
   //            return true; // more toys needed
   //            // decide which type we'd want to generate
   //            // if it matches the type we are generating, then return true
   //            auto pNull = pNull_toys(target_nSigma);
   //            auto pAlt = pAlt_toys(target_nSigma);
   //            if ((doingAlt ? pAlt.second : pNull.second) < 1e-4) return false; // stop if error gets too small
   //            bool doAlt = (pAlt.second * pNull.first > pNull.second * pAlt.first);
   //            return doAlt == doingAlt;
   //         }
   //      }
   //      return false;
   //   };
   //   while(condition()) {
   //      bool doAlt = false;
   //      double relErrThreshold = 2;
   //      if(nullToys.size()<size_t(nToys)) {
   //         addToys(false,nToys);continue;
   //      } else if(altToys.size()<size_t(nToys)) {
   //         addToys(true,nToys);continue;
   //      } else {
   //         // see which have bigger errors ... generate more of that ...
   //         auto pNull = pNull_toys(target_nSigma);
   //         auto pAlt = pAlt_toys(target_nSigma);
   //         doAlt = (pAlt.second*pNull.first > pNull.second*pAlt.first);
   //         if( (doAlt ? pAlt.second : pNull.second) < 1e-4 ) break; // stop if error gets too small
   //         auto pCLs = pCLs_toys(target_nSigma);
   //         relErrThreshold = (doAlt) ? (pNull.second/pNull.first) : (pAlt.second/pAlt.first);
   //         relErrThreshold = std::min(2.,std::abs(relErrThreshold));
   //         std::cout << "Current pCLs = " << pCLs.first << " +/- " << pCLs.second
   //                   << " (pNull = " << pNull.first << " +/- " << pNull.second
   //                  << " , pAlt = " << pAlt.first << " +/- " << pAlt.second << ") ... generating more " << (doAlt ?
   //                  "alt" : "null") << " toys " << relErrThreshold  << std::endl;
   //
   //      }
   //      if( addToys(doAlt, nToys/*, seed, -1, target_nSigma,relErrThreshold*/) == 0) {
   //         break; // no toys got added, so stop looping
   //      }
   //   }
}

xRooNLLVar::xRooHypoPoint
xRooNLLVar::hypoPoint(const char *poiValues, double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   xRooHypoPoint out;
   // out.fPOIName = parName; out.fNullVal = value; out.fAltVal = alt_value;

   if (!fFuncVars) {
      reinitialize();
   }
   AutoRestorer snap(*fFuncVars);

   out.nllVar = std::make_shared<xRooNLLVar>(*this);
   out.fData = getData();

   TStringToken pattern(poiValues, ",");
   TString poiNames;
   while (pattern.NextToken()) {
      TString s = pattern.Data();
      TString cName = s;
      double val = std::numeric_limits<double>::quiet_NaN();
      auto i = s.Index("=");
      if (i != -1) {
         cName = s(0, i);
         TString cVal = s(i + 1, s.Length());
         if (!cVal.IsFloat())
            throw std::runtime_error("poiValues must contain value");
         val = cVal.Atof();
      }
      auto v = dynamic_cast<RooRealVar *>(fFuncVars->find(cName));
      if (!v)
         throw std::runtime_error("Cannot find poi");
      if (!std::isnan(val))
         v->setVal(val);
      v->setConstant(); // because will select constants as coords
      if (poiNames != "") {
         poiNames += ",";
      }
      poiNames += cName;
   }
   if (poiNames == "") {
      throw std::runtime_error("No poi");
   }
   if (!std::isnan(alt_value)) {
      std::unique_ptr<RooAbsCollection> thePoi(fFuncVars->selectByName(poiNames));
      for (auto b : *thePoi) {
         if (!static_cast<RooRealVar *>(b)->hasRange("physical")) {
            static_cast<RooRealVar *>(b)->setRange("physical", 0, std::numeric_limits<double>::infinity());
         }
      }
   }
   auto _snap = std::unique_ptr<RooAbsCollection>(fFuncVars->selectByAttrib("Constant", true))->snapshot();
   _snap->setAttribAll("poi", false);
   std::unique_ptr<RooAbsCollection> _poi(_snap->selectByName(poiNames));
   _poi->setAttribAll("poi", true);
   if (std::isnan(alt_value)) {
      for (auto a : *_poi)
         a->setStringAttribute("altVal", nullptr);
   } else {
      for (auto a : *_poi)
         a->setStringAttribute("altVal", TString::Format("%g", alt_value));
   }
   if (fGlobs)
      _snap->remove(*fGlobs, true, true);
   out.coords.reset(_snap);

   auto _type = pllType;
   if (_type == xRooFit::Asymptotics::Unknown) {
      // decide based on values
      if (std::isnan(alt_value)) {
         _type = xRooFit::Asymptotics::TwoSided;
      } else if (dynamic_cast<RooRealVar *>(_poi->first())->getVal() >= alt_value) {
         _type = xRooFit::Asymptotics::OneSidedPositive;
      } else {
         _type = xRooFit::Asymptotics::Uncapped;
      }
   }

   out.fPllType = _type;

   return out;
}

xRooNLLVar::xRooHypoPoint
xRooNLLVar::hypoPoint(double value, double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   if (!fFuncVars) {
      reinitialize();
   }
   std::unique_ptr<RooAbsCollection> _poi(fFuncVars->selectByAttrib("poi", true));
   if (_poi->empty()) {
      throw std::runtime_error("No POI specified in model");
   } else if (_poi->size() != 1) {
      throw std::runtime_error("Multiple POI specified in model");
   }
   return hypoPoint(_poi->first()->GetName(), value, alt_value, pllType);
}

xRooNLLVar::xRooHypoPoint
xRooNLLVar::hypoPoint(const char *parName, double value, double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   return hypoPoint(TString::Format("%s=%f", parName, value), alt_value, pllType);
}

void xRooNLLVar::xRooHypoPoint::Draw(Option_t *opt)
{

   if (!nllVar && !hypoTestResult)
      return;

   TString sOpt(opt);
   sOpt.ToLower();
   bool hasSame = sOpt.Contains("same");
   sOpt.ReplaceAll("same", "");

   TVirtualPad *pad = gPad;

   TH1 *hAxis = nullptr;

   auto clearPad = []() {
      gPad->Clear();
      if (gPad->GetNumber() == 0) {
         gPad->SetBottomMargin(gStyle->GetPadBottomMargin());
         gPad->SetTopMargin(gStyle->GetPadTopMargin());
         gPad->SetLeftMargin(gStyle->GetPadLeftMargin());
         gPad->SetRightMargin(gStyle->GetPadRightMargin());
      }
   };

   if (!hasSame || !pad) {
      if (!pad) {
         TCanvas::MakeDefCanvas();
         pad = gPad;
      }
      clearPad();
   } else {
      // get the histogram representing the axes
      hAxis = dynamic_cast<TH1 *>(pad->GetPrimitive(".axis"));
      if (!hAxis) {
         for (auto o : *pad->GetListOfPrimitives()) {
            if (hAxis = dynamic_cast<TH1 *>(o); hAxis)
               break;
         }
      }
   }

   // get min and max values
   double _min = std::numeric_limits<double>::quiet_NaN();
   double _max = -std::numeric_limits<double>::quiet_NaN();

   for (auto &p : nullToys) {
      if (std::get<2>(p) == 0)
         continue;
      if (std::isnan(std::get<1>(p)))
         continue;
      _min = std::min(std::get<1>(p), _min);
      _max = std::max(std::get<1>(p), _max);
   }
   for (auto &p : altToys) {
      if (std::get<2>(p) == 0)
         continue;
      if (std::isnan(std::get<1>(p)))
         continue;
      _min = std::min(std::get<1>(p), _min);
      _max = std::max(std::get<1>(p), _max);
   }

   auto obs = pll();
   if (!std::isnan(obs.first)) {
      _min = std::min(obs.first - std::abs(obs.first) * 0.1, _min);
      _max = std::max(obs.first + std::abs(obs.first) * 0.1, _max);
   }
   // these are used down below to add obs p-values to legend, but up here because can trigger fits that create asimov
   auto pNull = pNull_toys();
   auto pAlt = pAlt_toys();
   auto pNullA = pNull_asymp();
   auto pAltA = pAlt_asymp();
   sigma_mu(true);
   auto asi = (fAsimov && fAsimov->fUfit && fAsimov->fNull_cfit) ? fAsimov->pll().first
                                                                 : std::numeric_limits<double>::quiet_NaN();
   if (!std::isnan(asi) && asi > 0 && fPllType != xRooFit::Asymptotics::Unknown) {
      // can calculate asymptotic distributions,
      _min = std::min(asi - std::abs(asi), _min);
      _max = std::max(asi + std::abs(asi), _max);
   }
   if (_min > 0)
      _min = 0;

   auto _poi = dynamic_cast<RooRealVar *>(poi().first());

   auto makeHist = [&](bool isAlt) {
      TString title;
      auto h = new TH1D((isAlt) ? "alt_toys" : "null_toys", "", 100, _min, _max + (_max - _min) * 0.01);
      h->SetDirectory(nullptr);
      size_t nBadOrZero = 0;
      for (auto &p : (isAlt) ? altToys : nullToys) {
         double w = std::isnan(std::get<1>(p)) ? 0 : std::get<2>(p);
         if (w == 0)
            nBadOrZero++;
         if (!std::isnan(std::get<1>(p)))
            h->Fill(std::get<1>(p), w);
      }
      if (h->GetEntries() > 0)
         h->Scale(1. / h->Integral(0, h->GetNbinsX() + 1));

      // add POI values to identify hypos
      //        for(auto p : *fPOI) {
      //            if (auto v = dynamic_cast<RooRealVar*>(p)) {
      //                if (auto v2 = dynamic_cast<RooRealVar*>(fAltPoint->fCoords->find(*v)); v2 &&
      //                v2->getVal()!=v->getVal()) {
      //                    // found point that differs in poi and altpoint value, so print my coords value for this
      //                    title += TString::Format("%s' = %g,
      //                    ",v->GetTitle(),dynamic_cast<RooRealVar*>(fCoords->find(*v))->getVal());
      //                }
      //            }
      //        }
      if (fPOIName())
         title += TString::Format("%s' = %g", fPOIName(), (isAlt) ? fAltVal() : fNullVal());
      title += TString::Format(" , N_{toys}=%d", int((isAlt) ? altToys.size() : nullToys.size()));
      if (nBadOrZero > 0)
         title += TString::Format(" (N_{bad/0}=%d)", int(nBadOrZero));
      title += ";";
      title += tsTitle();
      title += TString::Format(";Probability Mass");
      h->SetTitle(title);
      h->SetLineColor(isAlt ? kRed : kBlue);
      h->SetLineWidth(2);
      h->SetMarkerSize(0);
      h->SetBit(kCanDelete);
      return h;
   };

   auto nullHist = makeHist(false);
   auto altHist = makeHist(true);

   TLegend *l = nullptr;
   auto h = (nullHist->GetEntries()) ? nullHist : altHist;
   if (!hasSame) {
      gPad->SetLogy();
      auto axis = static_cast<TH1 *>(h->Clone(".axis"));
      axis->SetBit(kCanDelete);
      axis->SetStats(false);
      axis->Reset("ICES");
      axis->SetTitle(TString::Format("%s HypoPoint", collectionContents(poi()).c_str()));
      axis->SetLineWidth(0);
      axis->Draw(""); // h->Draw("axis"); cant use axis option if want title drawn
      axis->SetMinimum(1e-7);
      axis->GetYaxis()->SetRangeUser(1e-7, 10);
      axis->SetMaximum(h->GetMaximum());
      hAxis = axis;
      l = new TLegend(0.4, 0.7, 1. - gPad->GetRightMargin(), 1. - gPad->GetTopMargin());
      l->SetName("legend");
      l->SetFillStyle(0);
      l->SetBorderSize(0);
      l->SetBit(kCanDelete);
      l->Draw();
      l->ConvertNDCtoPad();
   } else {
      for (auto o : *gPad->GetListOfPrimitives()) {
         l = dynamic_cast<TLegend *>(o);
         if (l)
            break;
      }
   }

   if (h->GetEntries() > 0) {
      h->Draw("esame");
   } else {
      h->Draw("axissame"); // for unknown reason if second histogram empty it still draws with two weird bars???
   }
   h = altHist;
   if (h->GetEntries() > 0) {
      h->Draw("esame");
   } else {
      h->Draw("axissame"); // for unknown reason if second histogram empty it still draws with two weird bars???
   }

   if (l) {
      l->AddEntry(nullHist);
      l->AddEntry(altHist);
   }

   if (fAsimov && fAsimov->fUfit && fAsimov->fNull_cfit && !std::isnan(sigma_mu().first) && !std::isnan(fAltVal())) {
      auto hh = static_cast<TH1 *>(nullHist->Clone("null_asymp"));
      hh->SetBit(kCanDelete);
      hh->SetStats(false);
      hh->SetLineStyle(2);
      hh->Reset();
      for (int i = 1; i <= hh->GetNbinsX(); i++) {
         hh->SetBinContent(
            i, xRooFit::Asymptotics::PValue(fPllType, hh->GetBinLowEdge(i), fNullVal(), fNullVal(), sigma_mu().first,
                                            _poi->getMin("physical"), _poi->getMax("physical")) -
                  xRooFit::Asymptotics::PValue(fPllType, hh->GetBinLowEdge(i + 1), fNullVal(), fNullVal(),
                                               sigma_mu().first, _poi->getMin("physical"), _poi->getMax("physical")));
      }
      hh->Draw("lsame");
      hh = static_cast<TH1 *>(altHist->Clone("alt_asymp"));
      hh->SetBit(kCanDelete);
      hh->SetStats(false);
      hh->SetLineStyle(2);
      hh->Reset();
      for (int i = 1; i <= hh->GetNbinsX(); i++) {
         hh->SetBinContent(
            i, xRooFit::Asymptotics::PValue(fPllType, hh->GetBinLowEdge(i), fNullVal(), fAltVal(), sigma_mu().first,
                                            _poi->getMin("physical"), _poi->getMax("physical")) -
                  xRooFit::Asymptotics::PValue(fPllType, hh->GetBinLowEdge(i + 1), fNullVal(), fAltVal(),
                                               sigma_mu().first, _poi->getMin("physical"), _poi->getMax("physical")));
      }
      hh->Draw("lsame");
   }

   // draw observed points
   TLine ll;
   ll.SetLineStyle(1);
   ll.SetLineWidth(3);
   // for(auto p : fObs) {
   auto tl = ll.DrawLine(obs.first, hAxis->GetMinimum(), obs.first, 0.1);
   auto label = TString::Format("obs ts = %.4f", obs.first);
   if (obs.second)
      label += TString::Format(" #pm %.4f", obs.second);

   l->AddEntry(tl, label, "l");
   label = "";
   if (nullHist->GetEntries() || altHist->GetEntries()) {
      auto pCLs = pCLs_toys();
      label += " p_{toy}=(";
      label += (std::isnan(pNull.first)) ? "-" : TString::Format("%.4f #pm %.4f", pNull.first, pNull.second);
      label += (std::isnan(pAlt.first)) ? ",-" : TString::Format(",%.4f #pm %.4f", pAlt.first, pAlt.second);
      label += (std::isnan(pCLs.first)) ? ",-)" : TString::Format(",%.4f #pm %.4f)", pCLs.first, pCLs.second);
   }
   if (label.Length() > 0)
      l->AddEntry("", label, "");
   label = "";
   if (!std::isnan(pNullA.first) || !std::isnan(pAltA.first)) {
      auto pCLs = pCLs_asymp();
      label += " p_{asymp}=(";
      label += (std::isnan(pNullA.first)) ? "-" : TString::Format("%.4f #pm %.4f", pNullA.first, pNullA.second);
      label += (std::isnan(pAltA.first)) ? ",-" : TString::Format(",%.4f #pm %.4f", pAltA.first, pAltA.second);
      label += (std::isnan(pCLs.first)) ? ",-)" : TString::Format(",%.4f #pm %.4f)", pCLs.first, pCLs.second);
   }
   if (label.Length() > 0)
      l->AddEntry("", label, "");

   if (auto ax = dynamic_cast<TH1 *>(gPad->GetPrimitive(".axis")))
      ax->GetYaxis()->SetRangeUser(1e-7, 1);
}

TString xRooNLLVar::xRooHypoPoint::tsTitle(bool inWords) const
{
   auto v = dynamic_cast<RooRealVar *>(poi().empty() ? nullptr : poi().first());
   if (fPllType == xRooFit::Asymptotics::OneSidedPositive) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity()) {
         return (inWords) ? TString::Format("Lower-Bound One-Sided Limit PLR")
                          : TString::Format("#tilde{q}_{%s=%g}", v->GetTitle(), v->getVal());
      } else if (v) {
         return (inWords) ? TString::Format("One-Sided Limit PLR")
                          : TString::Format("q_{%s=%g}", v->GetTitle(), v->getVal());
      } else {
         return "q";
      }
   } else if (fPllType == xRooFit::Asymptotics::TwoSided) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity()) {
         return (inWords) ? TString::Format("Lower-Bound PLR")
                          : TString::Format("#tilde{t}_{%s=%g}", v->GetTitle(), v->getVal());
      } else if (v) {
         return (inWords) ? TString::Format("-2log[L(%s,#hat{#hat{#theta}})/L(#hat{%s},#hat{#theta})]", v->GetTitle(),
                                            v->GetTitle())
                          : TString::Format("t_{%s=%g}", v->GetTitle(), v->getVal());
      } else
         return "t";
   } else if (fPllType == xRooFit::Asymptotics::OneSidedNegative) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity()) {
         return (inWords) ? TString::Format("Lower-Bound One-Sided Discovery PLR")
                          : TString::Format("#tilde{r}_{%s=%g}", v->GetTitle(), v->getVal());
      } else if (v) {
         return (inWords) ? TString::Format("One-Sided Discovery PLR")
                          : TString::Format("r_{%s=%g}", v->GetTitle(), v->getVal());
      } else {
         return "r";
      }
   } else if (fPllType == xRooFit::Asymptotics::Uncapped) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity()) {
         return (inWords) ? TString::Format("Lower-Bound Uncapped PLR")
                          : TString::Format("#tilde{u}_{%s=%g}", v->GetTitle(), v->getVal());
      } else if (v) {
         return (inWords) ? TString::Format("Uncapped PLR") : TString::Format("u_{%s=%g}", v->GetTitle(), v->getVal());
      } else {
         return "u";
      }
   } else {
      return "Test Statistic";
   }
}

const char *xRooNLLVar::xRooHypoPoint::fPOIName()
{
   return (poi().empty()) ? nullptr : (poi().first())->GetName();
}
double xRooNLLVar::xRooHypoPoint::fNullVal()
{
   auto first_poi = dynamic_cast<RooAbsReal *>(poi().first());
   return (first_poi == nullptr) ? std::numeric_limits<double>::quiet_NaN() : first_poi->getVal();
}
double xRooNLLVar::xRooHypoPoint::fAltVal()
{
   auto _alt_poi = alt_poi(); // need to keep alive as alt_poi owns its contents
   auto first_poi = dynamic_cast<RooAbsReal *>(_alt_poi.first());
   return (first_poi == nullptr) ? std::numeric_limits<double>::quiet_NaN() : first_poi->getVal();
}

xRooNLLVar::xRooHypoSpace xRooNLLVar::hypoSpace(const char *parName, int nPoints, double low, double high,
                                                double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   if (nPoints < 0) {
      // catches case where pyROOT has converted TestStatistic enum to int
      int tsType = nPoints;
      double alt_val = std::numeric_limits<double>::quiet_NaN();
      if (tsType == xRooFit::TestStatistic::qmutilde || tsType == xRooFit::TestStatistic::qmu) {
         alt_val = 0;
      } else if (tsType == xRooFit::TestStatistic::q0 || tsType == xRooFit::TestStatistic::uncappedq0) {
         alt_val = 1;
      }

      auto out = hypoSpace(parName, pllType, alt_val);

      // TODO: things like the physical range and alt value can't be stored on the poi
      // because if they change they will change for all hypoSpaces at once, so cannot have
      // two hypoSpace with e.g. different physical ranges.
      // the hypoSpace should make a copy of them at some point
      for (auto p : out.poi()) {
         if (tsType == xRooFit::TestStatistic::qmutilde) {
            dynamic_cast<RooRealVar *>(p)->setRange("physical", 0, std::numeric_limits<double>::infinity());
            Info("xRooNLLVar::hypoSpace", "Setting physical range of %s to [0,inf]", p->GetName());
         } else if (dynamic_cast<RooRealVar *>(p)->hasRange("physical")) {
            dynamic_cast<RooRealVar *>(p)->removeRange("physical");
            Info("xRooNLLVar::hypoSpace", "Removing physical range of %s",
                 p->GetName());
         }
      }

      // ensure pll type is set explicitly if known at this point
      if (tsType == xRooFit::TestStatistic::qmutilde || tsType == xRooFit::TestStatistic::qmu) {
         out.fTestStatType = xRooFit::Asymptotics::OneSidedPositive;
      } else if (tsType == xRooFit::TestStatistic::uncappedq0) {
         out.fTestStatType = xRooFit::Asymptotics::Uncapped;
      } else if (tsType == xRooFit::TestStatistic::q0) {
         out.fTestStatType = xRooFit::Asymptotics::OneSidedNegative;
      }

      // in this case the arguments are shifted over by one
      if (int(low + 0.5) > 0) {
         out.AddPoints(parName, int(low + 0.5), high, alt_value);
      } else {
         if (!std::isnan(high) && !std::isnan(alt_value) && !(std::isinf(high) && std::isinf(alt_value))) {
            for (auto p : out.poi()) {
               dynamic_cast<RooRealVar *>(p)->setRange("scan", high, alt_value);
            }
         }
      }
      return out;
   }

   xRooNLLVar::xRooHypoSpace hs = hypoSpace(parName, pllType, alt_value);
   if (nPoints > 0)
      hs.AddPoints(parName, nPoints, low, high);
   else {
      if (!std::isnan(low) && !std::isnan(high) && !(std::isinf(low) && std::isinf(high))) {
         for (auto p : hs.poi()) {
            dynamic_cast<RooRealVar *>(p)->setRange("scan", low, high);
         }
      }
   }
   return hs;
}

xRooNLLVar::xRooHypoSpace xRooNLLVar::hypoSpace(int nPoints, double low, double high, double alt_value,
                                                const xRooFit::Asymptotics::PLLType &pllType)
{
   auto _poi = std::unique_ptr<RooAbsCollection>(
      std::unique_ptr<RooAbsCollection>(pdf()->getVariables())->selectByAttrib("poi", true));
   if (_poi->empty())
      throw std::runtime_error("You must specify a POI for the hypoSpace");
   return hypoSpace(_poi->first()->GetName(), nPoints, low, high, alt_value, pllType);
}

xRooNLLVar::xRooHypoSpace
xRooNLLVar::hypoSpace(const char *parName, const xRooFit::Asymptotics::PLLType &pllType, double alt_value)
{
   xRooNLLVar::xRooHypoSpace s(parName, parName);

   s.AddModel(pdf());
   if (strlen(parName)) {
      std::unique_ptr<RooAbsCollection> axes(s.pars()->selectByName(parName));
      if (axes->empty())
         throw std::runtime_error("parameter not found");
      axes->setAttribAll("axis", true);
   }
   /*if (std::unique_ptr<RooAbsCollection>(s.pars()->selectByAttrib("poi", true))->empty()) {
      throw std::runtime_error("You must specify at least one POI for the hypoSpace");
   }*/
   s.fNlls[s.fPdfs.begin()->second] = std::make_shared<xRooNLLVar>(*this);
   s.fTestStatType = pllType;

   for (auto poi : s.poi()) {
      poi->setStringAttribute("altVal", std::isnan(alt_value) ? nullptr : TString::Format("%f", alt_value));
   }

   return s;
}

RooStats::HypoTestResult xRooNLLVar::xRooHypoPoint::result()
{
   if (hypoTestResult) {
      return *hypoTestResult;
   }
   RooStats::HypoTestResult out;
   out.SetBackgroundAsAlt(true);
   out.SetName(TUUID().AsString());
   out.SetTitle(TString::Format("%s HypoPoint", collectionContents(poi()).c_str()));

   bool setReadonly = false;
   if (nllVar && !nllVar->get()->getAttribute("readOnly")) {
      setReadonly = true;
      nllVar->get()->setAttribute("readOnly");
   }

   auto ts_obs = ts_asymp();

   out.SetTestStatisticData(ts_obs.first);

   // build a ds to hold all fits ... store coords in the globs list of the nullDist
   // also need to store at least mu_hat value(s)
   RooArgList fitDetails;
   RooArgList fitMeta;
   fitMeta.addClone(RooCategory(
      "pllType", "test statistic type",
      {{"TwoSided", 0}, {"OneSidedPositive", 1}, {"OneSidedNegative", 2}, {"Uncapped", 3}, {"Unknown", 4}}));
   if (ufit()) {
      fitMeta.addClone(ufit()->floatParsFinal());
   }
   fitMeta.setCatIndex("pllType", int(fPllType));
   fitMeta.addClone(RooRealVar("isExpected", "isExpected", int(isExpected)));
   fitDetails.addClone(RooCategory("type", "fit type",
                                   {{"ufit", 0},
                                    {"cfit_null", 1},
                                    {"cfit_alt", 2},
                                    {"asimov_ufit", 3},
                                    {"asimov_cfit_null", 4},
                                    {"gen", 5},
                                    {"cfit_lbound", 6}}));
   // fitDetails.addClone(RooStringVar("name", "Fit Name", "")); -- not supported properly in ROOT yet
   fitDetails.addClone(RooRealVar("status", "status", 0));
   fitDetails.addClone(RooRealVar("covQual", "covQual", 0));
   fitDetails.addClone(RooRealVar("minNll", "minNll", 0));
   fitDetails.addClone(RooRealVar("edm", "edm", 0));
   auto fitDS = new RooDataSet("fits", "fit summary data", fitDetails);
   // fitDS->convertToTreeStore(); // strings not stored properly in vector store, so do convert! - not needed since
   // string var storage not properly supported - storing in globs list instead

   for (int i = 0; i < 7; i++) {
      std::shared_ptr<const RooFitResult> fit;
      switch (i) {
      case 0: fit = ufit(); break;
      case 1: fit = cfit_null(); break;
      case 2: fit = cfit_alt(); break;
      case 3: fit = asimov() ? asimov()->ufit(true) : nullptr; break;
      case 4: fit = asimov() ? asimov()->cfit_null(true) : nullptr; break;
      case 5: fit = fGenFit; break;
      case 6: fit = cfit_lbound(); break;
      }
      if (fit) {
         fitDetails.setCatIndex("type", i);
         fitMeta.addClone(RooStringVar(TString::Format("%s.name", fitDetails.getCatLabel("type")),
                                       fitDetails.getCatLabel("type"), fit->GetName()));
         // fitDetails.setStringValue("name",fit->GetName());
         fitDetails.setRealValue("status", fit->status());
         fitDetails.setRealValue("minNll", fit->minNll());
         fitDetails.setRealValue("edm", fit->edm());
         fitDetails.setRealValue("covQual", fit->covQual());
         fitDS->add(fitDetails);
      }
   }
   fitDS->setGlobalObservables(fitMeta);

   out.SetFitInfo(fitDS);

   RooArgList nullDetails;
   RooArgList nullMeta;
   nullMeta.addClone(*coords);
   nullDetails.addClone(RooRealVar("seed", "Toy Seed", 0));
   nullDetails.addClone(RooRealVar("ts", "test statistic value", 0));
   nullDetails.addClone(RooRealVar("weight", "weight", 1));
   auto nullToyDS = new RooDataSet("nullToys", "nullToys", nullDetails, RooFit::WeightVar("weight"));
   nullToyDS->setGlobalObservables(nullMeta);
   if (!nullToys.empty()) {

      std::vector<double> values;
      std::vector<double> weights;
      values.reserve(nullToys.size());
      weights.reserve(nullToys.size());

      for (auto &t : nullToys) {
         values.push_back(std::get<1>(t));
         weights.push_back(std::get<2>(t));
         nullDetails.setRealValue("seed", std::get<0>(t));
         nullDetails.setRealValue("ts", std::get<1>(t));
         nullToyDS->add(nullDetails, std::get<2>(t));
      }
      out.SetNullDistribution(new RooStats::SamplingDistribution("null", "Null dist", values, weights, tsTitle()));
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.fNullPValue = pNull_toys().first; // technically set above
      out.fNullPValueError =
         pNull_toys().second; // overrides binomial error used in SamplingDistribution::IntegralAndError
#else
      out.SetNullPValue(pNull_toys().first); // technically set above
      out.SetNullPValueError(
         pNull_toys().second); // overrides binomial error used in SamplingDistribution::IntegralAndError
#endif
   } else {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.fNullPValue = pNull_asymp().first;
      out.fNullPValueError = pNull_asymp().second;
#else
      out.SetNullPValue(pNull_asymp().first);
      out.SetNullPValueError(pNull_asymp().second);
#endif
   }
   out.SetNullDetailedOutput(nullToyDS);

   if (!altToys.empty()) {
      std::vector<double> values;
      std::vector<double> weights;
      values.reserve(altToys.size());
      weights.reserve(altToys.size());
      RooArgList altDetails;
      RooArgList altMeta;
      altDetails.addClone(RooRealVar("seed", "Toy Seed", 0));
      altDetails.addClone(RooRealVar("ts", "test statistic value", 0));
      altDetails.addClone(RooRealVar("weight", "weight", 1));
      auto altToyDS = new RooDataSet("altToys", "altToys", altDetails, RooFit::WeightVar("weight"));
      altToyDS->setGlobalObservables(altMeta);
      for (auto &t : altToys) {
         values.push_back(std::get<1>(t));
         weights.push_back(std::get<2>(t));
         altDetails.setRealValue("seed", std::get<0>(t));
         altDetails.setRealValue("ts", std::get<1>(t));
         altToyDS->add(altDetails, std::get<2>(t));
      }
      out.SetAltDistribution(new RooStats::SamplingDistribution("alt", "Alt dist", values, weights, tsTitle()));
      out.SetAltDetailedOutput(altToyDS);
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.fAlternatePValue = pAlt_toys().first; // technically set above
      out.fAlternatePValueError =
         pAlt_toys().second; // overrides binomial error used in SamplingDistribution::IntegralAndError
#else
      out.SetAltPValue(pAlt_toys().first); // technically set above
      out.SetAltPValueError(
         pAlt_toys().second); // overrides binomial error used in SamplingDistribution::IntegralAndError
#endif

   } else {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.fAlternatePValue = pAlt_asymp().first;
      out.fAlternatePValueError = pAlt_asymp().second;
#else
      out.SetAltPValue(pAlt_asymp().first);
      out.SetAltPValueError(pAlt_asymp().second);
#endif
   }

   if (setReadonly) {
      nllVar->get()->setAttribute("readOnly", false);
   }

   return out;
}

std::string cling::printValue(const xRooNLLVar::xValueWithError *v)
{
   if (!v)
      return "xValueWithError: nullptr\n";
   return Form("%f +/- %f", v->first, v->second);
}
std::string cling::printValue(const std::map<std::string, xRooNLLVar::xValueWithError> *m)
{
   if (!m)
      return "nullptr\n";
   std::string out = "{\n";
   for (auto [k, v] : *m) {
      out += "\"" + k + "\" => " + printValue(&v) + "\n";
   }
   out += "}\n";
   return out;
}

END_XROOFIT_NAMESPACE
