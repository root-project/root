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

#include "RVersion.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#define protected public
#endif
#include "RooFitResult.h"
#include "RooNLLVar.h"
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


BEGIN_XROOFIT_NAMESPACE

std::set<int> xRooNLLVar::xRooHypoPoint::allowedStatusCodes = {0};

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
            if (_cat.hasRange(chanPatterns.back()))
               hasRange = true;
            else {
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
};

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
   if (fPdf)
      fPdf->Print();
   else
      std::cout << "<null>" << std::endl;
   std::cout << "Data: ";
   if (fData)
      fData->Print();
   else
      std::cout << "<null>" << std::endl;
   std::cout << "NLL Options: " << std::endl;
   for (int i = 0; i < fOpts->GetSize(); i++) {
      auto c = dynamic_cast<RooCmdArg *>(fOpts->At(i));
      if (!c)
         continue;
      std::cout << " " << c->GetName() << " : ";
      if (c->getString(0))
         std::cout << c->getString(0);
      else if (c->getSet(0) && !c->getSet(0)->empty())
         std::cout << (c->getSet(0)->contentsString());
      else
         std::cout << c->getInt(0);
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
                     std::unique_ptr<std::list<Double_t>> boundaries{dynamic_cast<RooAbsReal *>(a)->binBoundaries(
                        *var, -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity())};
                     if (boundaries) {
                        if (!std::shared_ptr<RooAbsReal>::get())
                           Info("xRooNLLVar", "%s will be evaluated as a Binned PDF (%d bins)", a->GetName(),
                                int(boundaries->size() - 1));
                        setBinned = true;
                     }
                  }
               }
               origValues[a] = a->getAttribute("BinnedLikelihood");
               a->setAttribute("BinnedLikelihood", setBinned);
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
      // RooFit only swaps in what it calls parameters, this misses out the RooConstVars which we treat as pars as well
      // so swap those in ... question: is recursiveRedirectServers usage in RooAbsOptTestStatic (and here) a memory
      // leak?? where do the replaced servers get deleted??

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
   const_cast<RooArgList&>(fr->constPars()).setAttribAll("global", false);
   if (fGlobs)
      std::unique_ptr<RooAbsCollection>(fr->constPars().selectCommon(*fGlobs))->setAttribAll("global", true);
   return xRooFit::generateFrom(*fPdf, fr, expected, seed);
}

xRooNLLVar::xRooFitResult::xRooFitResult(const std::shared_ptr<xRooNode> &in)
   : std::shared_ptr<const RooFitResult>(std::dynamic_pointer_cast<const RooFitResult>(in->fComp)), fNode(in)
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

xRooNLLVar::xRooFitResult xRooNLLVar::minimize(const std::shared_ptr<ROOT::Fit::FitConfig> &_config)
{
   auto &nll = *get();
   auto out = xRooFit::minimize(nll, (_config) ? _config : fitConfig());
   // add any pars that are const here that aren't in constPars list because they may have been
   // const-optimized and their values cached with the dataset, so if subsequently floated the
   // nll wont evaluate correctly
   // fConstVars.reset( fFuncVars->selectByAttrib("Constant",true) );

   // if saving fits, check the nllOpts have been saved as well ...

   if (out && !nll.getAttribute("readOnly")) {
      if (strlen(fOpts->GetName()) == 0)
         fOpts->SetName(TUUID().AsString());
      auto cacheDir = gDirectory;
      if (cacheDir && cacheDir->IsWritable()) {
         // save a copy of fit result to relevant dir
         if (!cacheDir->GetDirectory(nll.GetName()))
            cacheDir->mkdir(nll.GetName());
         if (auto dir = cacheDir->GetDirectory(nll.GetName()); dir) {
            if (!dir->FindKey(fOpts->GetName())) {
               dir->WriteObject(fOpts.get(), fOpts->GetName());
            }
         }
      }
   }

   // before returning, flag which of the constPars were actually global observables
   if (out) {
      const_cast<RooArgList&>(out->constPars()).setAttribAll("global", false);
      if (fGlobs)
         std::unique_ptr<RooAbsCollection>(out->constPars().selectCommon(*fGlobs))->setAttribAll("global", true);
   }
   return xRooFitResult(std::make_shared<xRooNode>(out, fPdf));
}

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

double xRooNLLVar::getEntryVal(size_t entry)
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

std::shared_ptr<RooArgSet> xRooNLLVar::pars(bool stripGlobalObs)
{
   auto out = std::shared_ptr<RooArgSet>(get()->getVariables());
   if (stripGlobalObs && fGlobs) {
      out->remove(*fGlobs, true, true);
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

Bool_t xRooNLLVar::setData(const xRooNode &data)
{
   if (data.fComp && !data.get<RooAbsData>()) {
      return false;
   }
   return setData(std::dynamic_pointer_cast<RooAbsData>(data.fComp),
                  std::shared_ptr<const RooAbsCollection>(data.globs().argList().snapshot()));
}

Bool_t xRooNLLVar::setData(const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &_data)
{

   if (fData == _data.first && fGlobs == _data.second)
      return true;

   auto _globs = fGlobs; // done to keep globs alive while NLL might still be alive.

   if (fGlobs && !(fGlobs->empty() && !_data.second) &&
       _data.first) { // second condition allows for no globs being a nullptr, third allow globs to remain if nullifying
                      // data
      if (!_data.second)
         throw std::runtime_error("Missing globs");
      // ignore 'extra' globs
      RooArgSet s;
      s.add(*fGlobs);
      std::unique_ptr<RooAbsCollection> _actualGlobs(fPdf->getObservables(s));
      RooArgSet s2;
      s2.add(*_data.second);
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
      fGlobs = _data.second;
   }

   if (!std::shared_ptr<RooAbsReal>::get()) {
      fData = _data.first;
      return true; // not loaded yet so nothing to do
   }

   try {
      if (!kReuseNLL || nllTerm()->operMode() == RooAbsTestStatistic::MPMaster) {
         throw std::runtime_error("not supported");
      }
      bool out = false;
      if (_data.first)
         out = nllTerm()->setData(*_data.first, false /* clone data? */);
      else
         reset();
      fData = _data.first;
      return out;
   } catch (std::runtime_error &) {
      // happens when using MP need to rebuild the nll instead
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
   if (std::shared_ptr<RooAbsReal>::get())
      reinitialize(); // do this way to keep name of nll if user set
   else
      reset(); // will trigger reinitialize
}

RooAbsData *xRooNLLVar::data() const
{
   auto _nll = nllTerm();
   if (!_nll)
      return fData.get();
   RooAbsData *out = &_nll->data();
   if (!out)
      return fData.get();
   return out;
}

RooNLLVar *xRooNLLVar::nllTerm() const
{
   auto _func = func();
   if (auto a = dynamic_cast<RooNLLVar *>(_func.get()); a)
      return a;
   for (auto s : _func->servers()) {
      if (auto a = dynamic_cast<RooNLLVar *>(s); a)
         return a;
   }
   return nullptr;
}

double xRooNLLVar::extendedTerm() const
{
   // returns Nexp - Nobs*log(Nexp)
   return fPdf->extendedTerm(fData->sumEntries(), fData->get());
}

double xRooNLLVar::simTerm() const
{
   if (auto s = dynamic_cast<RooSimultaneous *>(fPdf.get()); s) {
      return fData->sumEntries() * log(1.0 * (s->servers().size() - 1)); // one of the servers is the cat
   }
   return 0;
}

double xRooNLLVar::binnedDataTerm() const
{
   // this is only relevant if BinnedLikelihood active
   double out = 0;
   for (int i = 0; i < fData->numEntries(); i++) {
      fData->get(i);
      out += TMath::LnGamma(fData->weight() + 1);
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
   }
   return nullptr;
}

/*xRooNLLVar::operator RooAbsReal &() const {
    // this works in c++ but not in python
    std::cout << "implicit conversion" << std::endl;
    return *fFunc;
}*/

std::pair<double, double> xRooNLLVar::xRooHypoPoint::getVal(const char *what)
{
   TString sWhat(what);
   sWhat.ToLower();
   bool doTS = sWhat.Contains("ts");
   bool doCLs = sWhat.Contains("cls");
   bool doNull = sWhat.Contains("null");
   bool doAlt = sWhat.Contains("alt");
   double nSigma = (sWhat.Contains("exp"))
                      ? (TString(sWhat(sWhat.Index("exp") + 3, sWhat.Index(" ", sWhat.Index("exp")) == -1
                                                                  ? sWhat.Length()
                                                                  : sWhat.Index(" ", sWhat.Index("exp"))))
                            .Atof())
                      : std::numeric_limits<double>::quiet_NaN();

   bool toys = sWhat.Contains("toys");
   // bool asymp = sWhat.Contains("asymp");

   bool readOnly = sWhat.Contains("readonly");

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

RooArgList xRooNLLVar::xRooHypoPoint::poi()
{
   RooArgList out;
   out.setName("poi");
   out.add(*std::unique_ptr<RooAbsCollection>(coords->selectByAttrib("poi", true)));
   return out;
}

RooArgList xRooNLLVar::xRooHypoPoint::alt_poi()
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

void xRooNLLVar::xRooHypoPoint::Print()
{
   std::cout << "POI: " << poi().contentsString() << " , null: " << dynamic_cast<RooAbsReal *>(poi().first())->getVal()
             << " , alt: " << dynamic_cast<RooAbsReal *>(alt_poi().first())->getVal();
   std::cout << " , pllType: " << fPllType << std::endl;

   std::cout << " -        ufit: ";
   if (fUfit) {
      std::cout << fUfit->minNll() << " (status=" << fUfit->status() << ") (" << mu_hat().GetName()
                << "_hat: " << mu_hat().getVal() << " +/- " << mu_hat().getError() << ")" << std::endl;
   } else {
      std::cout << "Not calculated" << std::endl;
   }
   std::cout << " -   null cfit: ";
   if (fNull_cfit) {
      std::cout << fNull_cfit->GetName() << " " << fNull_cfit->minNll() << " (status=" << fNull_cfit->status() << ")";
   } else {
      std::cout << "Not calculated";
   }
   if (!std::isnan(dynamic_cast<RooAbsReal *>(alt_poi().first())->getVal())) {
      std::cout << std::endl << " -    alt cfit: ";
      if (fAlt_cfit) {
         std::cout << fAlt_cfit->GetName() << " " << fAlt_cfit->minNll() << " (status=" << fAlt_cfit->status() << ")"
                   << std::endl;
      } else {
         std::cout << "Not calculated" << std::endl;
      }
      std::cout << " sigma_mu: ";
      if (!fAsimov || !fAsimov->fUfit || !fAsimov->fNull_cfit) {
         std::cout << "Not calculated";
      } else {
         std::cout << sigma_mu().first << " +/- " << sigma_mu().second;
      }
      if (fAsimov) {
         std::cout << std::endl;
         std::cout << "   - asimov ufit: ";
         if (fAsimov->fUfit)
            std::cout << fAsimov->fUfit->GetName() << " " << fAsimov->fUfit->minNll()
                      << " (status=" << fAsimov->fUfit->status() << ")";
         else
            std::cout << "Not calculated";
         std::cout << std::endl << "   - asimov null cfit: ";
         if (fAsimov->fNull_cfit)
            std::cout << fAsimov->fNull_cfit->GetName() << " " << fAsimov->fNull_cfit->minNll()
                      << " (status=" << fAsimov->fNull_cfit->status() << ")";
         else
            std::cout << "Not calculated";
      }
      std::cout << std::endl;
   } else {
      std::cout << std::endl;
   }
   if (fGenFit)
      std::cout << " -      genFit: " << fGenFit->GetName() << std::endl;
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
      if (var)
         return *var;
      else
         throw std::runtime_error("Cannot find POI");
   }
   throw std::runtime_error("Unconditional fit unavailable");
}

std::shared_ptr<xRooNLLVar::xRooHypoPoint> xRooNLLVar::xRooHypoPoint::asimov(bool readOnly)
{

   if (!fAsimov && nllVar) {
      if (!nllVar->fFuncVars)
         nllVar->reinitialize();
      AutoRestorer snap(*nllVar->fFuncVars);
      auto theFit = (!data.first && fGenFit) ? fGenFit : cfit_alt(readOnly);
      if (!theFit || allowedStatusCodes.find(theFit->status()) == allowedStatusCodes.end())
         return fAsimov;
      *nllVar->fFuncVars = theFit->floatParsFinal();
      *nllVar->fFuncVars = theFit->constPars();
      auto asimov = nllVar->generate(true);
      fAsimov = std::make_shared<xRooHypoPoint>(*this);
      fAsimov->fPllType = xRooFit::Asymptotics::TwoSided;
      fAsimov->fUfit.reset();
      fAsimov->fNull_cfit.reset();
      fAsimov->fAlt_cfit.reset();
      fAsimov->data = asimov;
      fAsimov->fGenFit = theFit;
      fAsimov->isExpected = true;
   }

   return fAsimov;
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pNull_asymp(double nSigma)
{
   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
   double nom = xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fNullVal(), sigma_mu().first,
                                             first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   return std::pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pAlt_asymp(double nSigma)
{
   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);

   double nom = xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fAltVal(), sigma_mu().first,
                                             first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));

   return std::pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pCLs_asymp(double nSigma)
{
   if (fNullVal() == fAltVal())
      return std::pair(1, 0); // by construction
   if (fPllType != xRooFit::Asymptotics::Uncapped && ts_asymp(nSigma).first == 0)
      return std::pair(1, 0);
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi)
      return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);

   double nom1 =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fNullVal(), sigma_mu().first,
                                   first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up1 =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down1 =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fNullVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double nom2 = xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first, fNullVal(), fAltVal(), sigma_mu().first,
                                              first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up2 =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first + ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double down2 =
      xRooFit::Asymptotics::PValue(fPllType, ts_asymp(nSigma).first - ts_asymp(nSigma).second, fNullVal(), fAltVal(),
                                   sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));

   auto nom = (nom1 == 0) ? 0 : nom1 / nom2;
   auto up = (up1 == 0) ? 0 : up1 / up2;
   auto down = (down1 == 0) ? 0 : down1 / down2;

   return std::make_pair(nom, std::max(std::abs(up - nom), std::abs(down - nom)));
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::ts_asymp(double nSigma)
{
   auto first_poi = dynamic_cast<RooRealVar *>(poi().first());
   if (!first_poi || (!std::isnan(nSigma) && std::isnan(sigma_mu().first)))
      return std::pair(std::numeric_limits<double>::quiet_NaN(), 0);
   if (std::isnan(nSigma))
      return pll();
   double nom = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                        sigma_mu().first, first_poi->getMin("physical"), first_poi->getMax("physical"));
   double up = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                       sigma_mu().first + sigma_mu().second, first_poi->getMin("physical"),
                                       first_poi->getMax("physical"));
   double down = xRooFit::Asymptotics::k(fPllType, ROOT::Math::gaussian_cdf(nSigma), fNullVal(), fAltVal(),
                                         sigma_mu().first - sigma_mu().second, first_poi->getMin("physical"),
                                         first_poi->getMax("physical"));
   return std::pair<double, double>(nom, std::max(std::abs(nom - up), std::abs(nom - down)));
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::ts_toys(double nSigma)
{
   if (std::isnan(nSigma))
      return pll();
   // nans should appear in the alt toys first ... so loop until past nans
   size_t firstToy = 0;
   while (firstToy < altToys.size() && std::isnan(std::get<1>(altToys[firstToy])))
      firstToy++;
   if (firstToy >= altToys.size())
      return std::make_pair(std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN());
   int targetIdx =
      (altToys.size() - firstToy) * ROOT::Math::gaussian_cdf(nSigma) + firstToy; // TODO: Account for weights
   return std::make_pair(
      std::get<1>(altToys[targetIdx]),
      (std::get<1>(altToys[std::min(int(altToys.size()), targetIdx)]) - std::get<1>(altToys[std::max(0, targetIdx)])) /
         2.);
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pll(bool readOnly)
{
   if (!ufit(readOnly) || allowedStatusCodes.find(ufit(readOnly)->status()) == allowedStatusCodes.end())
      return std::make_pair(std::numeric_limits<double>::quiet_NaN(), 0);
   auto cFactor = xRooFit::Asymptotics::CompatFactor(fPllType, fNullVal(), mu_hat().getVal());
   if (cFactor == 0)
      return std::make_pair(0, 0);
   if (!cfit_null(readOnly) || allowedStatusCodes.find(cfit_null(readOnly)->status()) == allowedStatusCodes.end())
      return std::make_pair(std::numeric_limits<double>::quiet_NaN(), 0);
   // std::cout << cfit->minNll() << ":" << cfit->edm() << " " << ufit->minNll() << ":" << ufit->edm() << std::endl;
   return std::make_pair(2. * cFactor * (cfit_null(readOnly)->minNll() - ufit(readOnly)->minNll()),
                         2. * cFactor * sqrt(pow(cfit_null(readOnly)->edm(), 2) + pow(ufit(readOnly)->edm(), 2)));
   // return 2.*cFactor*(cfit->minNll()+cfit->edm() - ufit->minNll()+ufit->edm());
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::ufit(bool readOnly)
{
   if (fUfit)
      return fUfit;
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   nllVar->setData(data);
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
   } else if (!std::isnan(fAltVal())) {
      // guess data given is expected to align with alt value
      nllVar->fFuncVars->setRealValue(fPOIName(), fAltVal());
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
      if (auto v = dynamic_cast<RooAbsReal *>(c); v)
         out += TString::Format("=%g", v->getVal());
      else if (auto cc = dynamic_cast<RooAbsCategory *>(c); cc)
         out += TString::Format("=%s", cc->getLabel());
      else if (auto s = dynamic_cast<RooStringVar *>(c); v)
         out += TString::Format("=%s", s->getVal());
   }
   return out;
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::cfit_null(bool readOnly)
{
   if (fNull_cfit)
      return fNull_cfit;
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   nllVar->setData(data);
   if (fUfit) {
      // move to ufit coords before evaluating
      *nllVar->fFuncVars = fUfit->floatParsFinal();
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   nllVar->fFuncVars->find(fPOIName())
      ->setStringAttribute("altVal", (!std::isnan(fAltVal())) ? TString::Format("%g", fAltVal()) : nullptr);
   if (fGenFit)
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
   nllVar->get()->setStringAttribute("fitresultTitle", collectionContents(poi()).c_str());
   return (fNull_cfit = nllVar->minimize());
}

std::shared_ptr<const RooFitResult> xRooNLLVar::xRooHypoPoint::cfit_alt(bool readOnly)
{
   if (std::isnan(fAltVal()))
      return nullptr;
   if (fAlt_cfit)
      return fAlt_cfit;
   if (!nllVar || (readOnly && nllVar->get() && !nllVar->get()->getAttribute("readOnly")))
      return nullptr;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   AutoRestorer snap(*nllVar->fFuncVars, nllVar.get());
   nllVar->setData(data);
   if (fUfit) {
      // move to ufit coords before evaluating
      *nllVar->fFuncVars = fUfit->floatParsFinal();
   }
   nllVar->fFuncVars->setAttribAll("Constant", false);
   *nllVar->fFuncVars = *coords; // will reconst the coords
   if (nllVar->fFuncGlobs)
      nllVar->fFuncGlobs->setAttribAll("Constant", true);
   *nllVar->fFuncVars = alt_poi();
   if (fGenFit)
      nllVar->get()->SetName(
         TString::Format("%s/%s_%s", nllVar->get()->GetName(), fGenFit->GetName(), (isExpected) ? "asimov" : "toys"));
   nllVar->get()->setStringAttribute("fitresultTitle", collectionContents(alt_poi()).c_str());
   return (fAlt_cfit = nllVar->minimize());
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::sigma_mu(bool readOnly)
{

   if (!asimov(readOnly)) {
      return std::make_pair(std::numeric_limits<double>::quiet_NaN(), 0);
   }

   auto out = asimov(readOnly)->pll(readOnly);
   return std::make_pair(std::abs(fNullVal() - fAltVal()) / sqrt(out.first),
                         out.second * 0.5 * std::abs(fNullVal() - fAltVal()) / (out.first * sqrt(out.first)));
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pX_toys(bool alt, double nSigma)
{
   auto _ts = ts_toys(nSigma);
   if (std::isnan(_ts.first))
      return _ts;

   TEfficiency eff("", "", 1, 0, 1);

   auto &_theToys = (alt) ? altToys : nullToys;

   // loop over toys, count how many are > ts value
   // nans (mean bad ts evaluations) will count towards uncertainty
   int nans = 0;
   double result = 0;
   double result_err_up = 0;
   double result_err_down = 0;
   for (auto &toy : _theToys) {
      if (std::isnan(std::get<1>(toy)))
         nans++;
      else {
         bool res = std::get<1>(toy) >= _ts.first;
         if (std::get<2>(toy) != 1)
            eff.FillWeighted(res, 0.5, std::get<2>(toy));
         else
            eff.Fill(res, 0.5);
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
   return std::make_pair(result, result_err);
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pNull_toys(double nSigma)
{
   return pX_toys(false, nSigma);
}

std::pair<double, double> xRooNLLVar::xRooHypoPoint::pAlt_toys(double nSigma)
{
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
   if (!cfit_null())
      return out;
   if (!nllVar->fFuncVars)
      nllVar->reinitialize();
   //*nllVar->fFuncVars = cfit_null()->floatParsFinal();
   //*nllVar->fFuncVars = cfit_null()->constPars();
   out.data = xRooFit::generateFrom(*nllVar->fPdf, cfit_null(), false, seed); // nllVar->generate(false,seed);
   out.fGenFit = cfit_null();
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
   out.data = xRooFit::generateFrom(*nllVar->fPdf, cfit_alt(), false, seed); // out.data = nllVar->generate(false,seed);
   out.fGenFit = cfit_alt();
   return out;
}

void xRooNLLVar::xRooHypoPoint::addToys(bool alt, int nToys, int initialSeed)
{
   if ((alt && !cfit_alt()) || (!alt && !cfit_null())) {
      throw std::runtime_error("Cannot add toys, invalid conditional fit");
   }
   auto &toys = (alt) ? altToys : nullToys;
   int nans = 0;
   std::vector<float> times(nToys);
   float lastTime = 0;
   int lasti = -1;
   TStopwatch s2;
   s2.Start();
   TStopwatch s;
   s.Start();
   for (auto i = 0; i < nToys; i++) {
      if (i == 0 && initialSeed != 0)
         RooRandom::randomGenerator()->SetSeed(initialSeed);
      int seed = RooRandom::randomGenerator()->Integer(std::numeric_limits<uint32_t>::max());
      toys.push_back(std::make_tuple(seed, ((alt) ? generateAlt(seed) : generateNull(seed)).pll().first, 1.));
      if (std::isnan(std::get<1>(toys.back())))
         nans++;
      times[i] = s.RealTime() - lastTime; // stops the clock
      lastTime = s.RealTime();
      if (s.RealTime() > 10) {
         std::cout << "\r"
                   << TString::Format("Generated %d/%d %s hypothesis toys [%.2f toys/s]...", i + 1, nToys,
                                      alt ? "alt" : "null", double(i - lasti) / s.RealTime())
                   << std::flush;
         lasti = i;
         s.Reset();
         s.Start();
         // std::cout << "Generated " << i << "/" << nToys << (alt ? " alt " : " null ") << " hypothesis toys " ..." <<
         // std::endl;
      }
      s.Continue();
   }
   if (lasti)
      std::cout << "\r"
                << TString::Format("Generated %d/%d %s hypothesis toys [%.2f toys/s overall]...Done!", nToys, nToys,
                                   alt ? "alt" : "null", double(nToys) / s2.RealTime())
                << std::endl;
   auto g = gDirectory->Get<TGraph>("toyTime");
   if (!g) {
      g = new TGraph;
      g->SetNameTitle("toyTime", "Time per toy;Toy;time [s]");
      gDirectory->Add(g);
   }
   g->Set(times.size());
   for (size_t i = 0; i < times.size(); i++)
      g->SetPoint(i, i, times[i]);
   // sort the toys ... put nans first - do by setting all as negative inf
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
   if (nans > 0)
      std::cout << "Warning: " << nans << " toys were bad" << std::endl;
}

void xRooNLLVar::xRooHypoPoint::addNullToys(int nToys, int seed)
{
   addToys(false, nToys, seed);
}
void xRooNLLVar::xRooHypoPoint::addAltToys(int nToys, int seed)
{
   addToys(true, nToys, seed);
}

xRooNLLVar::xRooHypoPoint
xRooNLLVar::hypoPoint(const char *parName, double value, double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   xRooHypoPoint out;
   // out.fPOIName = parName; out.fNullVal = value; out.fAltVal = alt_value;

   if (!fFuncVars) {
      reinitialize();
   }

   out.nllVar = std::make_shared<xRooNLLVar>(*this);
   out.data = getData();

   auto poi = dynamic_cast<RooRealVar *>(fFuncVars->find(parName));
   if (!poi)
      return out;
   AutoRestorer snap((RooArgSet(*poi)));
   poi->setVal(value);
   poi->setConstant();
   auto _snap = std::unique_ptr<RooAbsCollection>(fFuncVars->selectByAttrib("Constant", true))->snapshot();
   _snap->find(poi->GetName())->setAttribute("poi", true);
   if (std::isnan(alt_value))
      _snap->find(poi->GetName())->setStringAttribute("altVal", nullptr);
   else
      _snap->find(poi->GetName())->setStringAttribute("altVal", TString::Format("%g", alt_value));
   if (fGlobs)
      _snap->remove(*fGlobs, true, true);
   out.coords.reset(_snap);

   auto _type = pllType;
   if (_type == xRooFit::Asymptotics::Unknown) {
      // decide based on values
      if (std::isnan(alt_value))
         _type = xRooFit::Asymptotics::TwoSided;
      else if (value >= alt_value)
         _type = xRooFit::Asymptotics::OneSidedPositive;
      else
         _type = xRooFit::Asymptotics::Uncapped;
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

void xRooNLLVar::xRooHypoPoint::Draw(Option_t *opt)
{

   if (!nllVar)
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
      hAxis = dynamic_cast<TH1 *>(pad->GetPrimitive("axis"));
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
      h->SetDirectory(0);
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
      title += TString::Format("%s' = %g", fPOIName(), (isAlt) ? fAltVal() : fNullVal());
      title += TString::Format(" , N_{toys}=%lu", (isAlt) ? altToys.size() : nullToys.size());
      if (nBadOrZero > 0)
         title += TString::Format(" (N_{bad/0}=%lu)", nBadOrZero);
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
   auto h = nullHist;
   if (!hasSame) {
      gPad->SetLogy();
      auto axis = (TH1 *)h->Clone(".axis");
      axis->Reset("ICES");
      axis->SetMinimum(1e-7);
      axis->SetMaximum(h->GetMaximum());
      axis->SetTitle(TString::Format("HypoPoint"));
      axis->SetLineWidth(0);
      axis->Draw(""); // h->Draw("axis"); cant use axis option if want title drawn
      hAxis = axis;
      l = new TLegend(0.4, 0.7, 1. - gPad->GetRightMargin(), 1. - gPad->GetTopMargin());
      l->SetName("legend");
      l->SetFillStyle(0);
      l->SetBorderSize(0);
      l->SetBit(kCanDelete);
      l->Draw();
   } else {
      for (auto o : *gPad->GetListOfPrimitives()) {
         l = dynamic_cast<TLegend *>(o);
         if (l)
            break;
      }
   }

   if (h->GetEntries() > 0)
      h->Draw("histesame");
   else
      h->Draw("axissame"); // for unknown reason if second histogram empty it still draws with two weird bars???
   h = altHist;
   if (h->GetEntries() > 0)
      h->Draw("histesame");
   else
      h->Draw("axissame"); // for unknown reason if second histogram empty it still draws with two weird bars???

   if (l) {
      l->AddEntry(nullHist);
      l->AddEntry(altHist);
   }

   if (fAsimov && fAsimov->fUfit && fAsimov->fNull_cfit && !std::isnan(sigma_mu().first) && !std::isnan(fAltVal())) {
      auto hh = (TH1 *)nullHist->Clone("null_asymp");
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
      hh = (TH1 *)altHist->Clone("alt_asymp");
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
   ll.SetLineStyle(2);
   // for(auto p : fObs) {
   auto tl = ll.DrawLine(pll().first, hAxis->GetMinimum(), pll().first, 0.1);
   auto label = TString::Format("obs ts = %.4f", pll().first);
   if (pll().second)
      label += TString::Format(" #pm %.4f", pll().second);
   auto pNull = pNull_toys();
   auto pAlt = pAlt_toys();

   auto pNullA = pNull_asymp();
   auto pAltA = pAlt_asymp();

   l->AddEntry(tl, label, "l");
   label = "";
   if (!std::isnan(pNull.first) || !std::isnan(pAlt.first)) {
      auto pCLs = pCLs_toys();
      label += " p_{toy}=(";
      label += (std::isnan(pNull.first)) ? "-" : TString::Format("%.4f #pm %.4f", pNull.first, pNull.second);
      label += (std::isnan(pAlt.first)) ? ",-" : TString::Format(",%.4f #pm %.4f", pAlt.first, pAlt.second);
      label += (std::isnan(pCLs.first)) ? ",-)" : TString::Format(",%.4f #pm %.4f", pCLs.first, pCLs.second);
   }
   if (label.Length() > 0)
      l->AddEntry("", label, "");
   label = "";
   if (!std::isnan(pNullA.first) || !std::isnan(pAltA.first)) {
      auto pCLs = pCLs_asymp();
      label += " p_{asymp}=(";
      label += (std::isnan(pNullA.first)) ? "-" : TString::Format("%.4f #pm %.4f", pNullA.first, pNullA.second);
      label += (std::isnan(pAltA.first)) ? ",-" : TString::Format(",%.4f #pm %.4f", pAltA.first, pAltA.second);
      label += (std::isnan(pCLs.first)) ? ",-)" : TString::Format(",%.4f #pm %.4f", pCLs.first, pCLs.second);
   }
   if (label.Length() > 0)
      l->AddEntry("", label, "");

   //}
}

TString xRooNLLVar::xRooHypoPoint::tsTitle()
{
   auto v = dynamic_cast<RooRealVar *>(poi().empty() ? nullptr : poi().first());
   if (fPllType == xRooFit::Asymptotics::OneSidedPositive) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
         return TString::Format("#tilde{q}_{%s=%g}", v->GetTitle(), v->getVal());
      else if (v)
         return TString::Format("q_{%s=%g}", v->GetTitle(), v->getVal());
      else
         return "q";
   } else if (fPllType == xRooFit::Asymptotics::TwoSided) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
         return TString::Format("#tilde{t}_{%s=%g}", v->GetTitle(), v->getVal());
      else if (v)
         return TString::Format("t_{%s=%g}", v->GetTitle(), v->getVal());
      else
         return "t";
   } else if (fPllType == xRooFit::Asymptotics::OneSidedNegative) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
         return TString::Format("#tilde{r}_{%s=%g}", v->GetTitle(), v->getVal());
      else if (v)
         return TString::Format("r_{%s=%g}", v->GetTitle(), v->getVal());
      else
         return "r";
   } else if (fPllType == xRooFit::Asymptotics::Uncapped) {
      if (v && v->hasRange("physical") && v->getMin("physical") != -std::numeric_limits<double>::infinity())
         return TString::Format("#tilde{s}_{%s=%g}", v->GetTitle(), v->getVal());
      else if (v)
         return TString::Format("s_{%s=%g}", v->GetTitle(), v->getVal());
      else
         return "s";
   } else {
      return "Test Statistic";
   }
}

const char *xRooNLLVar::xRooHypoPoint::fPOIName()
{
   return (poi().first())->GetName();
}
double xRooNLLVar::xRooHypoPoint::fNullVal()
{
   return dynamic_cast<RooAbsReal *>(poi().first())->getVal();
}
double xRooNLLVar::xRooHypoPoint::fAltVal()
{
   return dynamic_cast<RooAbsReal *>(alt_poi().first())->getVal();
}

xRooNLLVar::xRooHypoSpace xRooNLLVar::hypoSpace(const char *parName, int nPoints, double low, double high,
                                                double alt_value, const xRooFit::Asymptotics::PLLType &pllType)
{
   xRooNLLVar::xRooHypoSpace hs = hypoSpace(parName, pllType);
   hs.poi().first()->setStringAttribute("altVal", std::isnan(alt_value) ? nullptr : TString::Format("%f", alt_value));
   if (nPoints > 0)
      hs.AddPoints(parName, nPoints, low, high);
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

xRooNLLVar::xRooHypoSpace xRooNLLVar::hypoSpace(const char *parName, const xRooFit::Asymptotics::PLLType &pllType)
{
   xRooNLLVar::xRooHypoSpace s(parName, parName);

   s.AddModel(pdf());
   if (strlen(parName)) {
      auto poi = s.pars()->find(parName);
      if (!poi)
         throw std::runtime_error("parameter not found");
      s.pars()->setAttribAll("poi", false);
      poi->setAttribute("poi", true);
   } else if (std::unique_ptr<RooAbsCollection>(s.pars()->selectByAttrib("poi", true))->empty()) {
      throw std::runtime_error("You must specify a POI for the hypoSpace");
   }
   s.fNlls[s.fPdfs.begin()->second] = std::make_shared<xRooNLLVar>(*this);
   s.fTestStatType = pllType;
   return s;
}

RooStats::HypoTestResult xRooNLLVar::xRooHypoPoint::result()
{
   RooStats::HypoTestResult out;
   out.SetBackgroundAsAlt(true);

   bool setReadonly = false;
   if (nllVar && !nllVar->get()->getAttribute("readOnly")) {
      setReadonly = true;
      nllVar->get()->setAttribute("readOnly");
   }

   auto ts_obs = ts_asymp();

   out.SetTestStatisticData(ts_obs.first);
   RooArgList nullDetails;
   if (!nullToys.empty()) {

      std::vector<double> values;
      std::vector<double> weights;
      values.reserve(nullToys.size());
      weights.reserve(nullToys.size());
      size_t badToys = 0;
      for (auto &t : nullToys) {
         if (std::isnan(std::get<1>(t))) {
            badToys++;
         } else {
            values.push_back(std::get<1>(t));
            weights.push_back(std::get<2>(t));
         }
      }
      nullDetails.addClone(RooRealVar("badToys", "Number of bad Toys", badToys));

      out.SetNullDistribution(new RooStats::SamplingDistribution("null", "Null dist", values, weights, tsTitle()));
      out.SetNullDetailedOutput(new RooDataSet("nullDetails", "nullDetails", nullDetails));
      out.GetNullDetailedOutput()->add(nullDetails);
   } else {
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
      out.fNullPValue = pNull_asymp().first;
      out.fNullPValueError = pNull_asymp().second;
#else
      out.SetNullPValue(pNull_asymp().first);
      out.SetNullPValueError(pNull_asymp().second);
#endif
   }

   RooArgList altDetails;
   if (!altToys.empty()) {
      std::vector<double> values;
      std::vector<double> weights;
      values.reserve(nullToys.size());
      weights.reserve(nullToys.size());
      size_t badToys = 0;
      for (auto &t : nullToys) {
         if (std::isnan(std::get<1>(t))) {
            badToys++;
         } else {
            values.push_back(std::get<1>(t));
            weights.push_back(std::get<2>(t));
         }
      }
      nullDetails.addClone(RooRealVar("badToys", "Number of bad Toys", badToys));

      out.SetAltDistribution(new RooStats::SamplingDistribution("alt", "Alt dist", values, weights, tsTitle()));
      out.SetAltDetailedOutput(new RooDataSet("altDetails", "altDetails", altDetails));
      out.GetAltDetailedOutput()->add(altDetails);
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

END_XROOFIT_NAMESPACE
