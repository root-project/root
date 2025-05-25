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

// #define private public
// #include "Minuit2/Minuit2Minimizer.h"
// #undef private
// #include "Minuit2/FunctionMinimum.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
#define protected public
#endif
#include "RooFitResult.h"
#ifdef protected
#undef protected
#endif

#include "xRooFit/xRooFit.h"

#include "RooDataSet.h"
#include "RooSimultaneous.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "RooAbsPdf.h"
#include "TUUID.h"
#include "RooProdPdf.h"
#include "RooGamma.h"
#include "RooPoisson.h"
#include "RooGaussian.h"
#include "RooBifurGauss.h"
#include "RooLognormal.h"
#include "RooBinning.h"
#include "RooUniformBinning.h"

#include "RooStats/AsymptoticCalculator.h"
#include "Math/GenAlgoOptions.h"
#include "Math/Minimizer.h"
#include "RooMinimizer.h"
#include "coutCapture.h"

#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TKey.h"
#include "TPRegexp.h"
#include "RooStringVar.h"

#include "RooRealProxy.h"
#include "RooSuperCategory.h"

#include "xRooFitVersion.h"

#include <csignal>
#include "TROOT.h"
#include "TBrowser.h"

BEGIN_XROOFIT_NAMESPACE

std::shared_ptr<RooLinkedList> xRooFit::sDefaultNLLOptions = nullptr;
std::shared_ptr<ROOT::Fit::FitConfig> xRooFit::sDefaultFitConfig = nullptr;

const char *xRooFit::GetVersion()
{
   return GIT_COMMIT_HASH;
}
const char *xRooFit::GetVersionDate()
{
   return GIT_COMMIT_DATE;
}

RooCmdArg xRooFit::ReuseNLL(bool flag)
{
   return RooCmdArg("ReuseNLL", flag, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
}

RooCmdArg xRooFit::Tolerance(double val)
{
   return RooCmdArg("Tolerance", 0, 0, val);
}

RooCmdArg xRooFit::StrategySequence(const char *val)
{
   return RooCmdArg("StrategySequence", 0, 0, 0, 0, val);
}

RooCmdArg xRooFit::MaxIterations(int val)
{
   return RooCmdArg("MaxIterations", val);
}

xRooNLLVar xRooFit::createNLL(const std::shared_ptr<RooAbsPdf> pdf, const std::shared_ptr<RooAbsData> data,
                              const RooLinkedList &nllOpts)
{
   return xRooNLLVar(pdf, data, nllOpts);
}

xRooNLLVar xRooFit::createNLL(RooAbsPdf &pdf, RooAbsData *data, const RooLinkedList &nllOpts)
{
   return createNLL(std::shared_ptr<RooAbsPdf>(&pdf, [](RooAbsPdf *) {}),
                    std::shared_ptr<RooAbsData>(data, [](RooAbsData *) {}), nllOpts);
}

xRooNLLVar xRooFit::createNLL(RooAbsPdf &pdf, RooAbsData *data, const RooCmdArg &arg1, const RooCmdArg &arg2,
                              const RooCmdArg &arg3, const RooCmdArg &arg4, const RooCmdArg &arg5,
                              const RooCmdArg &arg6, const RooCmdArg &arg7, const RooCmdArg &arg8)
{

   RooLinkedList l;
   l.Add((TObject *)&arg1);
   l.Add((TObject *)&arg2);
   l.Add((TObject *)&arg3);
   l.Add((TObject *)&arg4);
   l.Add((TObject *)&arg5);
   l.Add((TObject *)&arg6);
   l.Add((TObject *)&arg7);
   l.Add((TObject *)&arg8);
   return createNLL(pdf, data, l);
}

std::shared_ptr<const RooFitResult>
xRooFit::fitTo(RooAbsPdf &pdf,
               const std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> &data,
               const RooLinkedList &nllOpts, const ROOT::Fit::FitConfig &fitConf)
{
   return xRooNLLVar(std::shared_ptr<RooAbsPdf>(&pdf, [](RooAbsPdf *) {}), data, nllOpts)
      .minimize(std::shared_ptr<ROOT::Fit::FitConfig>(const_cast<ROOT::Fit::FitConfig *>(&fitConf),
                                                      [](ROOT::Fit::FitConfig *) {}));
}

std::shared_ptr<const RooFitResult> xRooFit::fitTo(RooAbsPdf &pdf,
                                                   const std::pair<RooAbsData *, const RooAbsCollection *> &data,
                                                   const RooLinkedList &nllOpts, const ROOT::Fit::FitConfig &fitConf)
{
   return xRooNLLVar(pdf, data, nllOpts)
      .minimize(std::shared_ptr<ROOT::Fit::FitConfig>(const_cast<ROOT::Fit::FitConfig *>(&fitConf),
                                                      [](ROOT::Fit::FitConfig *) {}));
}

std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>>
xRooFit::generateFrom(RooAbsPdf &pdf, const RooFitResult &_fr, bool expected, int seed)
{

   std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooAbsCollection>> out;

   auto fr = &_fr;
   if (!fr)
      return out;

   auto _allVars = std::unique_ptr<RooAbsCollection>(pdf.getVariables());
   auto _snap = std::unique_ptr<RooAbsCollection>(_allVars->snapshot());
   *_allVars = fr->constPars();
   *_allVars = fr->floatParsFinal();

   // determine globs from fr constPars
   auto _globs = std::unique_ptr<RooAbsCollection>(fr->constPars().selectByAttrib("global", true));

   // bool doBinned = false;
   // RooAbsPdf::GenSpec** gs = nullptr;

   if (seed == 0)
      seed = RooRandom::randomGenerator()->Integer(std::numeric_limits<uint32_t>::max());
   RooRandom::randomGenerator()->SetSeed(seed);

   TString uuid = TUUID().AsString();

   std::function<std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooArgSet>>(RooAbsPdf *)> genSubPdf;

   genSubPdf = [&](RooAbsPdf *_pdf) {
      std::pair<std::shared_ptr<RooAbsData>, std::shared_ptr<const RooArgSet>> _out;
      // std::unique_ptr<RooArgSet> _obs(_pdf->getParameters(*pars)); // using this "trick" to get observables can
      // produce 'error' msg because of RooProdPdf trying to setup partial integrals
      std::unique_ptr<RooArgSet> _obs(_pdf->getVariables());
      _obs->remove(fr->constPars(), true, true);
      _obs->remove(fr->floatParsFinal(), true, true); // use this instead

      if (!_globs->empty()) {
         RooArgSet *toy_gobs = new RooArgSet(uuid + "_globs");
         // ensure we use the gobs from the model ...
         RooArgSet t;
         t.add(*_globs);
         std::unique_ptr<RooArgSet> globs(_pdf->getObservables(t));
         globs->snapshot(*toy_gobs);
         if (!toy_gobs->empty() &&
             !dynamic_cast<RooSimultaneous *>(
                _pdf)) { // if was simPdf will call genSubPdf on each subpdf so no need to generate here
            if (!expected) {
               *toy_gobs = *std::unique_ptr<RooDataSet>(_pdf->generate(*globs, 1))->get();
            } else {
               // loop over pdfs in top-level prod-pdf,
               auto pp = dynamic_cast<RooProdPdf *>(_pdf);
               if (pp) {
                  for (auto thePdf : pp->pdfList()) {
                     auto gob = std::unique_ptr<RooArgSet>(thePdf->getObservables(*globs));
                     if (gob->empty())
                        continue;
                     if (gob->size() > 1) {
                        Warning("generate", "%s contains multiple global obs: %s", thePdf->GetName(),
                                gob->contentsString().c_str());
                        continue;
                     }
                     RooRealVar &rrv = dynamic_cast<RooRealVar &>(*gob->first());
                     std::unique_ptr<RooArgSet> cpars(thePdf->getParameters(*globs));

                     bool foundServer = false;
                     // note : this will work only for this type of constraints
                     // expressed as RooPoisson, RooGaussian, RooLognormal, RooGamma
                     // SimpleGaussianConstraint is CMS's own version of a RooGaussian, which also works.
                     TClass *cClass = thePdf->IsA();
                     if (cClass != RooGaussian::Class() && cClass != RooPoisson::Class() &&
                         cClass != RooGamma::Class() && cClass != RooLognormal::Class() &&
                         cClass != RooBifurGauss::Class() &&
                         !(cClass && strcmp(cClass->GetName(), "SimpleGaussianConstraint") == 0)) {
                        TString className = (cClass) ? cClass->GetName() : "undefined";
                        oocoutW((TObject *)nullptr, Generation)
                           << "AsymptoticCalculator::MakeAsimovData:constraint term " << thePdf->GetName()
                           << " of type " << className << " is a non-supported type - result might be not correct "
                           << std::endl;
                     }

                     // in case of a Poisson constraint make sure the rounding is not set
                     if (cClass == RooPoisson::Class()) {
                        RooPoisson *pois = dynamic_cast<RooPoisson *>(thePdf);
                        assert(pois);
                        pois->setNoRounding(true);
                     }

                     // look at server of the constraint term and check if the global observable is part of the server
                     RooAbsArg *arg = thePdf->findServer(rrv);
                     if (!arg) {
                        // special case is for the Gamma where one might define the global observable n and you have a
                        // Gamma(b, n+1, ...._ in this case n+1 is the server and we don;t have a direct dependency, but
                        // we want to set n to the b value so in case of the Gamma ignore this test
                        if (cClass != RooGamma::Class()) {
                           oocoutE((TObject *)nullptr, Generation)
                              << "AsymptoticCalculator::MakeAsimovData:constraint term " << thePdf->GetName()
                              << " has no direct dependence on global observable- cannot generate it " << std::endl;
                           continue;
                        }
                     }

                     // loop on the server of the constraint term
                     // need to treat the Gamma as a special case
                     // the mode of the Gamma is (k-1)*theta where theta is the inverse of the rate parameter.
                     // we assume that the global observable is defined as ngobs = k-1 and the theta parameter has the
                     // name theta otherwise we use other procedure which might be wrong
                     RooAbsReal *thetaGamma = nullptr;
                     if (cClass == RooGamma::Class()) {
                        for (RooAbsArg *a2 : thePdf->servers()) {
                           if (TString(a2->GetName()).Contains("theta")) {
                              thetaGamma = dynamic_cast<RooAbsReal *>(a2);
                              break;
                           }
                        }
                        if (thetaGamma == nullptr) {
                           oocoutI((TObject *)nullptr, Generation)
                              << "AsymptoticCalculator::MakeAsimovData:constraint term " << thePdf->GetName()
                              << " is a Gamma distribution and no server named theta is found. Assume that the Gamma "
                                 "scale is  1 "
                              << std::endl;
                        }
                     }
                     for (RooAbsArg *a2 : thePdf->servers()) {
                        RooAbsReal *rrv2 = dynamic_cast<RooAbsReal *>(a2);
                        if (rrv2 && !rrv2->dependsOn(*gob) &&
                            (!rrv2->isConstant() || !rrv2->InheritsFrom("RooConstVar"))) {

                           // found server not depending on the gob
                           if (foundServer) {
                              oocoutE((TObject *)nullptr, Generation)
                                 << "AsymptoticCalculator::MakeAsimovData:constraint term " << thePdf->GetName()
                                 << " constraint term has more server depending on nuisance- cannot generate it "
                                 << std::endl;
                              foundServer = false;
                              break;
                           }
                           if (thetaGamma && thetaGamma->getVal() > 0) {
                              rrv.setVal(rrv2->getVal() / thetaGamma->getVal());
                           } else {
                              rrv.setVal(rrv2->getVal());
                           }
                           foundServer = true;
                        }
                     }

                     if (!foundServer) {
                        oocoutE((TObject *)nullptr, Generation)
                           << "AsymptoticCalculator::MakeAsimovData - can't find nuisance for constraint term - global "
                              "observables will not be set to Asimov value "
                           << thePdf->GetName() << std::endl;
                        std::cerr << "Parameters: " << std::endl;
                        cpars->Print("V");
                        std::cerr << "Observables: " << std::endl;
                        gob->Print("V");
                     }
                  }
               } else {
                  Error("generate", "Cannot generate global observables, pdf is: %s::%s", _pdf->ClassName(),
                        _pdf->GetName());
               }
               *toy_gobs = *globs;
            }
         }
         _out.second.reset(toy_gobs);
      } // end of globs generation

      RooRealVar w("weightVar", "weightVar", 1);
      if (auto s = dynamic_cast<RooSimultaneous *>(_pdf)) {
         // do subpdf's individually
         _obs->add(w);
         _out.first = std::make_unique<RooDataSet>(
            uuid, TString::Format("%s %s", _pdf->GetTitle(), (expected) ? "Expected" : "Toy"), *_obs,
            RooFit::WeightVar("weightVar"));

         for (auto &c : s->indexCat()) {
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 22, 00)
            std::string cLabel = c.first;
#else
            std::string cLabel = c->GetName();
#endif
            auto p = s->getPdf(cLabel.c_str());
            if (!p)
               continue;
            auto toy = genSubPdf(p);
            if (toy.second && _out.second)
               *const_cast<RooArgSet *>(_out.second.get()) = *toy.second;
            _obs->setCatLabel(s->indexCat().GetName(), cLabel.c_str());
            for (int i = 0; i < toy.first->numEntries(); i++) {
               *_obs = *toy.first->get(i);
               _out.first->add(*_obs, toy.first->weight());
            }
         }
         return _out;
      }

      std::map<RooRealVar *, std::shared_ptr<RooAbsBinning>> binnings;

      for (auto &o : *_obs) {
         auto r = dynamic_cast<RooRealVar *>(o);
         if (!r)
            continue;
         if (auto res = _pdf->binBoundaries(*r, -std::numeric_limits<double>::infinity(),
                                            std::numeric_limits<double>::infinity())) {
            binnings[r] = std::shared_ptr<RooAbsBinning>(r->getBinning().clone(r->getBinning().GetName()));

            std::vector<double> boundaries;
            boundaries.reserve(res->size());
            for (auto &rr : *res) {
               if (boundaries.empty() || std::abs(boundaries.back() - rr) > 1e-3 ||
                   std::abs(boundaries.back() - rr) > 1e-5 * boundaries.back())
                  boundaries.push_back(rr);
            } // sometimes get virtual duplicates of boundaries
            r->setBinning(RooBinning(boundaries.size() - 1, &boundaries[0]));
            delete res;
         } else if (r->numBins(r->getBinning().GetName()) == 0 && expected) {
            // no bins ... in order to generate expected we need to have some bins
            binnings[r] = std::shared_ptr<RooAbsBinning>(r->getBinning().clone(r->getBinning().GetName()));
            r->setBinning(RooUniformBinning(r->getMin(), r->getMax(), 100));
         }
      }

      // now can generate
      if (_obs->empty()) {
         // no observables, create a single dataset with 1 entry ... why 1 entry??
         _obs->add(w);
         RooArgSet _tmp;
         _tmp.add(w);
         _out.first = std::make_unique<RooDataSet>("", "Toy", _tmp, RooFit::WeightVar("weightVar"));
         _out.first->add(_tmp);
      } else {
         if (_pdf->canBeExtended()) {
            _out.first =
               std::unique_ptr<RooDataSet>{_pdf->generate(*_obs, RooFit::Extended(), RooFit::ExpectedData(expected))};
         } else {
            if (expected) {
               // use AsymptoticCalculator because generate expected not working correctly on unextended pdf?
               // TODO: Can the above code for expected globs be used instead, or what about replace above code with
               // ObsToExpected?
               _out.first.reset(RooStats::AsymptoticCalculator::GenerateAsimovData(*_pdf, *_obs));
            } else {
               _out.first = std::unique_ptr<RooDataSet>{_pdf->generate(*_obs, RooFit::ExpectedData(expected))};
            }
         }
      }
      _out.first->SetName(TUUID().AsString());

      for (auto &b : binnings) {
         auto v = b.first;
         auto binning = b.second;
         v->setBinning(*binning);
         // range of variable in dataset may be less than in the workspace
         // if e.g. generate for a specific channel. So need to expand ranges to match
         auto x = dynamic_cast<RooRealVar *>(_out.first->get()->find(v->GetName()));
         auto r = x->getRange();
         if (r.first > binning->lowBound())
            x->setMin(binning->lowBound());
         if (r.second < binning->highBound())
            x->setMax(binning->highBound());
      }
      return _out;
   };

   out = genSubPdf(&pdf);
   out.first->SetName(expected ? (TString(fr->GetName()) + "_asimov") : uuid);

   // from now on we store the globs in the dataset
   if (out.second) {
      out.first->setGlobalObservables(*out.second);
      out.second.reset();
   }

#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 26, 00)
   // store fitResult name on the weightVar
   if (auto w = dynamic_cast<RooDataSet *>(out.first.get())->weightVar()) {
      w->setStringAttribute("fitResult", fr->GetName());
      w->setAttribute("expected", expected);
   }
#endif

   *_allVars = *_snap;

   // June2023: Added this because found that generation was otherwise getting progressively slower
   // the RooAbsPdf::generate does a clone, and it seems that the RooCacheManager of the original pdf
   // is getting polluted on each generate call, causing it to grow larger and therefore the clone of it
   // to take longer and longer. So sterilize to clear the caches of all components
#if ROOT_VERSION_CODE < ROOT_VERSION(6, 27, 00)
   auto _ws = pdf._myws;
#else
   auto _ws = pdf.workspace();
#endif
   if (_ws) {
      // do explicitly rather than via xRooNode sterilize method because don't want to invoke the constructor
      // workspace tweaking features (which sets poi etc etc)
      for (auto obj : _ws->components()) {
         for (int i = 0; i < obj->numCaches(); i++) {
            if (auto cache = dynamic_cast<RooObjCacheManager *>(obj->getCache(i))) {
               cache->reset();
            }
         }
         if (RooAbsPdf *p = dynamic_cast<RooAbsPdf *>(obj); p) {
            p->setNormRange(p->normRange());
         }
         obj->setValueDirty();
      }
      // xRooNode(pdf.workspace()).sterilize();
   }

   return out;
}

std::shared_ptr<RooLinkedList> xRooFit::createNLLOptions()
{
   auto out = std::shared_ptr<RooLinkedList>(new RooLinkedList, [](RooLinkedList *l) {
      l->Delete();
      delete l;
   });
   for (auto opt : *defaultNLLOptions()) {
      out->Add(opt->Clone(nullptr)); // nullptr needed because accessing Clone via TObject base class puts
                                     // "" instead, so doesnt copy names
   }
   return out;
}

std::shared_ptr<RooLinkedList> xRooFit::defaultNLLOptions()
{
   if (sDefaultNLLOptions)
      return sDefaultNLLOptions;
   sDefaultNLLOptions = std::shared_ptr<RooLinkedList>(new RooLinkedList, [](RooLinkedList *l) {
      l->Delete();
      delete l;
   });
   sDefaultNLLOptions->Add(RooFit::Offset().Clone());
   // disable const-optimization at the construction step ... can happen in the minimization though
   sDefaultNLLOptions->Add(RooFit::Optimize(0).Clone());
   return sDefaultNLLOptions;
}

std::shared_ptr<ROOT::Fit::FitConfig> xRooFit::createFitConfig()
{
   return std::make_shared<ROOT::Fit::FitConfig>(*defaultFitConfig());
}

std::shared_ptr<ROOT::Fit::FitConfig> xRooFit::defaultFitConfig()
{
   if (sDefaultFitConfig)
      return sDefaultFitConfig;
   sDefaultFitConfig = std::make_shared<ROOT::Fit::FitConfig>();
   auto &fitConfig = *sDefaultFitConfig;
   fitConfig.SetParabErrors(true); // will use to run hesse after fit
   fitConfig.MinimizerOptions().SetMinimizerType("Minuit2");
   fitConfig.MinimizerOptions().SetErrorDef(0.5); // ensures errors are +/- 1 sigma ..IMPORTANT
   fitConfig.SetParabErrors(true);                // runs HESSE
   fitConfig.SetMinosErrors(true); // computes asymmetric errors on any parameter with the "minos" attribute set
   fitConfig.MinimizerOptions().SetMaxFunctionCalls(
      -1); // calls per iteration. if left as 0 will set automatically to 500*nPars below
   fitConfig.MinimizerOptions().SetMaxIterations(-1); // if left as 0 will set automatically to 500*nPars
   fitConfig.MinimizerOptions().SetStrategy(-1);      // will start at front of StrategySequence (given below)
   // fitConfig.MinimizerOptions().SetTolerance(
   //         1); // default is 0.01 (i think) but roominimizer uses 1 as default - use specify with
   //         ROOT::Math::MinimizerOptions::SetDefaultTolerance(..)
   fitConfig.MinimizerOptions().SetPrintLevel(-2);
   fitConfig.MinimizerOptions().SetExtraOptions(ROOT::Math::GenAlgoOptions());
   // have to const cast to set extra options
   auto extraOpts = const_cast<ROOT::Math::IOptions *>(fitConfig.MinimizerOptions().ExtraOptions());
   extraOpts->SetValue("OptimizeConst", 2); // if 0 will disable constant term optimization and cache-and-track of the
                                            // NLL. 1 = just caching, 2 = cache and track
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 29, 00)
   extraOpts->SetValue("StrategySequence", "0s01s12s2s3m");
   extraOpts->SetValue("HesseStrategySequence", "23");
#else
   extraOpts->SetValue("StrategySequence", "0s01s12s2m");
   extraOpts->SetValue("HesseStrategySequence", "2");
#endif
   extraOpts->SetValue(
      "HesseStrategy",
      -1); // when hesse is run after minimization, will use this strategy. -1 means start at begin of strat sequence
   extraOpts->SetValue("LogSize", 0); // length of log to capture and save
   extraOpts->SetValue("BoundaryCheck",
                       0.); // if non-zero, warn if any post-fit value is close to boundary (e.g. 0.01 = within 1%)
   extraOpts->SetValue("TrackProgress", 30);               // seconds between output to log of evaluation progress
   extraOpts->SetValue("xRooFitVersion", GIT_COMMIT_HASH); // not really options but here for logging purposes
   // extraOpts->SetValue("ROOTVersion",ROOT_VERSION_CODE); - not needed as should by part of the ROOT TFile definition

   // extraOpts->SetValue("HessianStepTolerance",0.);
   // extraOpts->SetValue("HessianG2Tolerance",0.);

   return sDefaultFitConfig;
}

ROOT::Math::IOptions *xRooFit::defaultFitConfigOptions()
{
   return const_cast<ROOT::Math::IOptions *>(defaultFitConfig()->MinimizerOptions().ExtraOptions());
}

class ProgressMonitor : public RooAbsReal {
public:
   void (*oldHandlerr)(int) = nullptr;
   static ProgressMonitor *me;
   static bool fInterrupt;
   static void interruptHandler(int signum)
   {
      if (signum == SIGINT) {
         std::cout << "Minimization interrupted ... will exit as soon as possible" << std::endl;
         // TODO: create a global mutex for this
         fInterrupt = true;
      } else {
         if (me)
            me->oldHandlerr(signum);
      }
   };
   ProgressMonitor(RooAbsReal &f, int interval = 30)
      : RooAbsReal(Form("progress_%s", f.GetName()), ""),
        oldHandlerr(signal(SIGINT, interruptHandler)),
        fFunc("func", "func", this, f),
        fInterval(interval)
   {
      s.Start();

      me = this;
      vars.reset(std::unique_ptr<RooAbsCollection>(f.getVariables())->selectByAttrib("Constant", false));
   }
   ~ProgressMonitor() override
   {
      if (oldHandlerr) {
         signal(SIGINT, oldHandlerr);
      }
      if (me == this)
         me = nullptr;
   };
   ProgressMonitor(const ProgressMonitor &other, const char *name = nullptr)
      : RooAbsReal(other, name), fFunc("func", this, other.fFunc), fInterval(other.fInterval)
   {
   }
   TObject *clone(const char *newname) const override { return new ProgressMonitor(*this, newname); }

   // required forwarding methods for RooEvaluatorWrapper in 6.32 onwards
   double defaultErrorLevel() const override { return fFunc->defaultErrorLevel(); }
   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool stripDisconnected) const override
   {
      return fFunc->getParameters(observables, outputSet, stripDisconnected);
   }
   bool setData(RooAbsData &data, bool cloneData) override { return fFunc->setData(data, cloneData); }
   double getValV(const RooArgSet *) const override { return evaluate(); }
   void applyWeightSquared(bool flag) override { fFunc->applyWeightSquared(flag); }
   void printMultiline(std::ostream &os, Int_t contents, bool verbose = false, TString indent = "") const override
   {
      fFunc->printMultiline(os, contents, verbose, indent);
   }
   void constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt) override
   {
      fFunc->constOptimizeTestStatistic(opcode, doAlsoTrackingOpt);
   }

   double evaluate() const override
   {
      if (fInterrupt) {
         throw std::runtime_error("Keyboard interrupt");
         return std::numeric_limits<double>::quiet_NaN();
      }
      double out = fFunc;
      if (prevMin == std::numeric_limits<double>::infinity()) {
         prevMin = out;
         prevPars.addClone(*vars);
      }
      if (!std::isnan(out)) {
         if (out < minVal) {
            if (minPars.empty())
               minPars.addClone(*vars);
            minPars = *vars;
         }
         minVal = std::min(minVal, out);
      }
      counter++;
      if (s.RealTime() > fInterval) {
         double evalRate = (counter - prevCounter) / s.RealTime();
         s.Reset();
         std::stringstream sout;

         sout << (counter) << ") (" << evalRate << "Hz) " << TDatime().AsString();
         if (!fState.empty())
            sout << " : " << fState;
         if (counter2) {
            // doing a hesse step, estimate progress based on evaluations
            int nRequired = prevPars.size();
            if (nRequired > 1) {
               nRequired *= nRequired;
               if (fState == "Hesse3") {
                  nRequired *= 4;
               }
               sout << " (~" << int(100.0 * (counter - counter2) / nRequired) << "%)";
            }
         }
         sout << " : " << minVal << " Delta = " << (minVal - prevMin);
         if (minVal < prevMin) {
            sout << " : ";
            // compare minPars and prevPars, print biggest deltas
            std::vector<std::pair<double, std::string>> parDeltas;
            parDeltas.reserve(minPars.size());
            for (auto p : minPars) {
               parDeltas.emplace_back(std::pair<double, std::string>(
                  dynamic_cast<RooRealVar *>(p)->getVal() - prevPars.getRealValue(p->GetName()), p->GetName()));
            }
            std::sort(parDeltas.begin(), parDeltas.end(),
                      [](auto &left, auto &right) { return std::abs(left.first) > std::abs(right.first); });
            int i;
            for (i = 0; i < std::min(3, int(parDeltas.size())); i++) {
               if (parDeltas.at(i).first == 0)
                  break;
               if (i != 0)
                  sout << ",";
               sout << parDeltas.at(i).second << (parDeltas.at(i).first >= 0 ? "+" : "-") << "="
                    << std::abs(parDeltas.at(i).first) << "(" << minPars.getRealValue(parDeltas.at(i).second.c_str())
                    << ")";
            }
            if (i < int(parDeltas.size()) && parDeltas.at(i).first != 0)
               sout << " ...";
            prevPars.assignFast(minPars);
         }

         if (gROOT->FromPopUp() && gROOT->GetListOfBrowsers()->At(0)) {
            auto browser = dynamic_cast<TBrowser *>(gROOT->GetListOfBrowsers()->At(0));
            std::string status = sout.str();
            int col = 0;
            while (col < 4) {
               std::string status_part;
               if (status.find(" : ") != std::string::npos) {
                  status_part = status.substr(0, status.find(" : "));
                  status = status.substr(status.find(" : ") + 3);
               } else {
                  status_part = status;
                  status = "";
               }
               browser->SetStatusText(status_part.c_str(), col);
               col++;
            }
            gSystem->ProcessEvents();
         }
         std::cerr << sout.str() << std::endl;

         prevMin = minVal;
         prevCounter = counter;
      } else {
         s.Continue();
      }
      return out;
   }

   std::string fState;
   mutable int counter = 0;
   int counter2 = 0; // used to estimate progress of a Hesse calculation

private:
   RooRealProxy fFunc;
   mutable double minVal = std::numeric_limits<double>::infinity();
   mutable double prevMin = std::numeric_limits<double>::infinity();
   mutable RooArgList minPars;
   mutable RooArgList prevPars;
   mutable int prevCounter = 0;
   mutable int fInterval = 0; // time in seconds before next report
   mutable TStopwatch s;
   std::shared_ptr<RooAbsCollection> vars;
};
bool ProgressMonitor::fInterrupt = false;
ProgressMonitor *ProgressMonitor::me = nullptr;

xRooFit::StoredFitResult::StoredFitResult(RooFitResult *_fr) : TNamed(*_fr)
{
   fr.reset(_fr);
}

xRooFit::StoredFitResult::StoredFitResult(const std::shared_ptr<RooFitResult> &_fr) : TNamed(*_fr), fr(_fr) {}

std::shared_ptr<const RooFitResult> xRooFit::minimize(RooAbsReal &nll,
                                                      const std::shared_ptr<ROOT::Fit::FitConfig> &_fitConfig,
                                                      const std::shared_ptr<RooLinkedList> &nllOpts)
{

   auto myFitConfig = _fitConfig ? _fitConfig : createFitConfig();
   auto &fitConfig = *myFitConfig;

   auto _nll = &nll;

   TString resultTitle = nll.getStringAttribute("fitresultTitle");
   TString fitName = TUUID().AsString();
   if (resultTitle == "")
      resultTitle = TUUID(fitName).GetTime().AsString();

   // extract any user pars from the nll too
   RooArgList fUserPars;
   if (nll.getStringAttribute("userPars")) {
      TStringToken st(nll.getStringAttribute("userPars"), ",");
      while (st.NextToken()) {
         TString parName = st;
         TString parVal = nll.getStringAttribute(parName);
         if (parVal.IsFloat()) {
            fUserPars.addClone(RooRealVar(parName, parName, parVal.Atof()));
         } else {
            fUserPars.addClone(RooStringVar(parName, parName, parVal));
         }
      }
   }

   auto _nllVars = std::unique_ptr<RooAbsCollection>(_nll->getVariables());

   std::unique_ptr<RooAbsCollection> constPars(_nllVars->selectByAttrib("Constant", true));
   constPars->add(fUserPars, true); // add here so checked for when loading from cache
   std::unique_ptr<RooAbsCollection> floatPars(_nllVars->selectByAttrib("Constant", false));

   int _progress = 0;
   double boundaryCheck = 0;
   std::string s;
   std::string hs;
   int logSize = 0;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 29, 00)
   int hesseStrategy = 3; // uses most precise hesse settings (step sizes and g2 tolerances)
#else
   int hesseStrategy = 2; // uses most precise hesse settings (step sizes and g2 tolerances)
#endif
   if (fitConfig.MinimizerOptions().ExtraOptions()) {
      fitConfig.MinimizerOptions().ExtraOptions()->GetNamedValue("StrategySequence", s);
      fitConfig.MinimizerOptions().ExtraOptions()->GetIntValue("TrackProgress", _progress);
      fitConfig.MinimizerOptions().ExtraOptions()->GetRealValue("BoundaryCheck", boundaryCheck);
      fitConfig.MinimizerOptions().ExtraOptions()->GetIntValue("LogSize", logSize);
      fitConfig.MinimizerOptions().ExtraOptions()->GetIntValue("HesseStrategy", hesseStrategy);
      fitConfig.MinimizerOptions().ExtraOptions()->GetNamedValue("HesseStrategySequence", hs);
   }
   TString m_strategy = s;
   TString m_hessestrategy = hs;

   // if fit caching enabled, try to locate a valid fitResult
   // must have matching constPars
   TDirectory *cacheDir = gDirectory;

   if (cacheDir) {
      if (auto nllDir = cacheDir->GetDirectory(nll.GetName()); nllDir) {
         if (auto keys = nllDir->GetListOfKeys(); keys) {
            for (auto &&k : *keys) {
               auto cl = TClass::GetClass((static_cast<TKey *>(k))->GetClassName());
               if (cl->InheritsFrom("RooFitResult")) {
                  StoredFitResult *storedFr =
                     nllDir->GetList() ? dynamic_cast<StoredFitResult *>(nllDir->GetList()->FindObject(k->GetName()))
                                       : nullptr;
                  if (auto cachedFit =
                         (storedFr) ? storedFr->fr.get() : dynamic_cast<TKey *>(k)->ReadObject<RooFitResult>();
                      cachedFit) {
                     if (!storedFr) {
                        storedFr = new StoredFitResult(cachedFit);
                        nllDir->Add(storedFr);
                        // std::cout << "Loaded " << nllDir->GetPath() << "/" << k->GetName() << " : " << k->GetTitle()
                        // << std::endl;
                     }
                     bool match = true;
                     if (!cachedFit->floatParsFinal().equals(*floatPars)) {
                        match = false;
                     } else {
                        for (auto &p : *constPars) {
                           auto v = dynamic_cast<RooAbsReal *>(p);
                           if (!v) {
                              if (auto c = dynamic_cast<RooAbsCategory *>(p)) {
                                 if (auto _p =
                                        dynamic_cast<RooAbsCategory *>(cachedFit->constPars().find(p->GetName()));
                                     _p && !_p->getAttribute("global") &&
                                     _p->getCurrentIndex() != c->getCurrentIndex()) {
                                    match = false;
                                    break;
                                 }
                              } else {
                                 match = false;
                                 break;
                              }
                           };
                           if (auto _p = dynamic_cast<RooAbsReal *>(cachedFit->constPars().find(p->GetName())); _p) {
                              // note: do not need global observable values to match (globals currently added to
                              // constPars list)
                              if (!_p->getAttribute("global") && std::abs(_p->getVal() - v->getVal()) > 1e-12) {
                                 match = false;
                                 break;
                              }
                           }
                        }
                     }
                     if (match) {
                        return storedFr->fr;
                        // return std::shared_ptr<RooFitResult>(cachedFit,[](RooFitResult*){}); // dir owns the
                        // fitResult - this means dir needs to stay open for fits to be valid return
                        // std::make_shared<RooFitResult>(*cachedFit); // return a copy ... dir doesn't need to stay
                        // open, but fit result isn't shared
                     } else {
                        // delete cachedFit;
                     }
                  }
               }
            }
         }
      }
   }

   if (nll.getAttribute("readOnly"))
      return nullptr;

   int printLevel = fitConfig.MinimizerOptions().PrintLevel();
   RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
   if (printLevel < 0)
      RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

   // check how many parameters we have ... if 0 parameters then we wont run a fit, we just evaluate nll and return ...
   if (floatPars->empty() || fitConfig.MinimizerOptions().MaxFunctionCalls() == 1) {
      std::shared_ptr<RooFitResult> result;
      RooArgList parsList;
      parsList.add(*floatPars);
      // construct an empty fit result ...
      result = std::make_shared<RooFitResult>(); // if put name here fitresult gets added to dir, we don't want that
      result->SetName(TUUID().AsString());
      result->SetTitle(resultTitle);
      result->setFinalParList(parsList);
      result->setInitParList(parsList);
      result->setConstParList(dynamic_cast<RooArgSet &>(*constPars)); /* RooFitResult takes a snapshot */
      TMatrixDSym d;
      d.ResizeTo(parsList.size(), parsList.size());
      result->setCovarianceMatrix(d);
      result->setCovQual(floatPars->empty() ? 3 : -1);
      result->setMinNLL(_nll->getVal());
      result->setEDM(0);
      result->setStatus(floatPars->empty() ? 0 : 1);

      std::vector<std::pair<std::string, int>> statusHistory;
      statusHistory.emplace_back(std::make_pair("EVAL", result->status()));
      result->setStatusHistory(statusHistory);

      if (cacheDir && cacheDir->IsWritable()) {
         // save a copy of fit result to relevant dir
         if (!cacheDir->GetDirectory(nll.GetName()))
            cacheDir->mkdir(nll.GetName());
         if (auto dir = cacheDir->GetDirectory(nll.GetName()); dir) {
            // save NLL opts if was given one, unless already present
            if (nllOpts) {
               if (strlen(nllOpts->GetName()) == 0) {
                  nllOpts->SetName(TUUID().AsString());
               }
               if (!dir->FindKey(nllOpts->GetName())) {
                  dir->WriteObject(nllOpts.get(), nllOpts->GetName());
               }
            }
            dir->WriteObject(result.get(), result->GetName());
         }
      }

      if (printLevel < 0)
         RooMsgService::instance().setGlobalKillBelow(msglevel);
      return result;
   }

   std::shared_ptr<RooFitResult> out;

   // check if any floatPars are categorical .. if so, need to a "discrete minimization" over the permutations
   RooArgSet floatCats;
   for (auto p : *floatPars) {
      if (p->isCategory()) {
         floatCats.add(*p);
      }
   }
   if (!floatCats.empty()) {
      RooSuperCategory allCats("floatCats", "Floating categorical parameters", floatCats);
      std::unique_ptr<RooAbsCollection> _snap(floatCats.snapshot());
      floatCats.setAttribAll("Constant");

      std::shared_ptr<const RooFitResult> bestFr;
      for (auto c : allCats) {
         allCats.setIndex(c.second);
         Info("minimize", "Minimizing with discrete %s", c.first.c_str());
         auto fr = minimize(nll, _fitConfig, nllOpts);
         if (!fr) {
            Warning("minimize", "Minimization with discrete %s failed", c.first.c_str());
            continue;
         }
         if (!bestFr || fr->minNll() < bestFr->minNll()) {
            bestFr = fr;
         }
      }

      floatCats.setAttribAll("Constant", false);

      if (!bestFr)
         return out;

      // create a copy of the fit result, give it a new uuid, and move the const categories into the float area
      out = std::make_shared<RooFitResult>(*bestFr);
      const_cast<RooArgList &>(out->floatParsFinal())
         .addClone(*std::unique_ptr<RooAbsCollection>(out->constPars().selectCommon(floatCats)));
      const_cast<RooArgList &>(out->floatParsInit()).addClone(*_snap);
      const_cast<RooArgList &>(out->constPars()).remove(floatCats);
      out->SetName(TUUID().AsString());
   }

   bool restore = !fitConfig.UpdateAfterFit();
   bool minos = fitConfig.MinosErrors();
   std::string logs;
   if (!out) {
      int strategy = fitConfig.MinimizerOptions().Strategy();
      // Note: AsymptoticCalculator enforces not less than 1 on tolerance - should we do so too?
      if (_progress) {
         _nll = new ProgressMonitor(*_nll, _progress);
         ProgressMonitor::fInterrupt = false;
      }
      auto logger = (logSize > 0) ? std::make_unique<cout_redirect>(logs, logSize) : nullptr;
      RooMinimizer _minimizer(*_nll);
      _minimizer.fitter()->Config() = fitConfig;
      //      if(fitConfig.MinimizerOptions().ExtraOptions()) {
      //         //for loading hesse options
      //         double a;
      //         if(fitConfig.MinimizerOptions().ExtraOptions()->GetValue("HessianStepTolerance",a)) {
      //            ROOT::Math::MinimizerOptions::Default("Minuit2").SetValue("HessianStepTolerance",a);
      //         }
      //         if(fitConfig.MinimizerOptions().ExtraOptions()->GetValue("HessianG2Tolerance",a)) {
      //            ROOT::Math::MinimizerOptions::Default("Minuit2").SetValue("HessianG2Tolerance",a);
      //         }
      //      }

      bool autoMaxCalls = (_minimizer.fitter()->Config().MinimizerOptions().MaxFunctionCalls() == 0);
      if (autoMaxCalls) {
         _minimizer.fitter()->Config().MinimizerOptions().SetMaxFunctionCalls(
            500 * floatPars->size() * floatPars->size()); // hesse requires O(N^2) function calls
      }
      if (_minimizer.fitter()->Config().MinimizerOptions().MaxIterations() == 0) {
         _minimizer.fitter()->Config().MinimizerOptions().SetMaxIterations(500 * floatPars->size());
      }

      bool hesse = _minimizer.fitter()->Config().ParabErrors();
      _minimizer.fitter()->Config().SetParabErrors(
         false); // turn "off" so can run hesse as a separate step, appearing in status
      _minimizer.fitter()->Config().SetMinosErrors(false);
      _minimizer.fitter()->Config().SetUpdateAfterFit(true); // note: seems to always take effect

      std::vector<std::pair<std::string, int>> statusHistory;

      // gCurrentSampler = this;
      // gOldHandlerr = signal(SIGINT,toyInterruptHandlerr);

      TString actualFirstMinimizer = _minimizer.fitter()->Config().MinimizerType();

      int status = 0;

      int constOptimize = 2;
      _minimizer.fitter()->Config().MinimizerOptions().ExtraOptions()->GetValue("OptimizeConst", constOptimize);
      if (constOptimize) {
         _minimizer.optimizeConst(constOptimize);
         // for safety force a refresh of the cache (and tracking) in the nll
         // DO NOT do a ConfigChange ... this is just a deactivate-reactivate of caching
         // but it seems like doing this breaks the const optimization and function is badly behaved
         // so once its turned on never turn it off.
         // nll.constOptimizeTestStatistic(RooAbsArg::ConfigChange, constOptimize>1 /* do tracking too if >1 */); //
         // trigger a re-evaluate of which nodes to cache-and-track
         // the next line seems safe to do but wont bother doing it because not bothering with above
         // need to understand why turning the cache off and on again breaks it??
         // nll.constOptimizeTestStatistic(RooAbsArg::ValueChange, constOptimize>1); // update the cache values -- is
         // this needed??
      } else {
         // disable const optimization
         // warning - if the nll was previously activated then it seems like deactivating may break it.
         nll.constOptimizeTestStatistic(RooAbsArg::DeActivate);
      }

      int sIdx = -1;
      TString minim = _minimizer.fitter()->Config().MinimizerType();
      TString algo = _minimizer.fitter()->Config().MinimizerAlgoType();
      if (minim == "Minuit2") {
         if (strategy == -1) {
            sIdx = 0;
         } else {
            sIdx = m_strategy.Index('0' + strategy);
         }
         if (sIdx == -1) {
            Warning("minimize", "Strategy %d not specified in StrategySequence %s ... defaulting to start of sequence",
                    strategy, m_strategy.Data());
            sIdx = 0;
         }
      } else if (minim == "Minuit")
         sIdx = m_strategy.Index('m');

      int tries = 0;
      int maxtries = 4;
      bool first = true;
      while (tries < maxtries && sIdx != -1) {
         if (m_strategy(sIdx) == 'm') {
            minim = "Minuit";
            algo = "migradImproved";
         } else if (m_strategy(sIdx) == 's') {
            algo = "Scan";
         } else if (m_strategy(sIdx) == 'h') {
            break; // jumping straight to a hesse evaluation
         } else {
            strategy = int(m_strategy(sIdx) - '0');
            _minimizer.setStrategy(strategy);
            minim = "Minuit2";
            algo = "Migrad";
         }
         if (auto fff = dynamic_cast<ProgressMonitor *>(_nll); fff) {
            fff->fState = minim + algo + std::to_string(_minimizer.fitter()->Config().MinimizerOptions().Strategy());
         }
         try {
            status = _minimizer.minimize(minim, algo);
         } catch (const std::exception &e) {
            std::cerr << "Exception while minimizing: " << e.what() << std::endl;
         }
         if (first && actualFirstMinimizer != _minimizer.fitter()->Config().MinimizerType())
            actualFirstMinimizer = _minimizer.fitter()->Config().MinimizerType();
         first = false;
         tries++;

         if (auto fff = dynamic_cast<ProgressMonitor *>(_nll); fff && fff->fInterrupt) {
            delete _nll;
            throw std::runtime_error("Keyboard interrupt while minimizing");
         }

         // RooMinimizer loses the useful status code, so here we will override it
         status = _minimizer.fitter()
                     ->Result()
                     .Status(); // note: Minuit failure is status code 4, minuit2 that is edm above max
         minim = _minimizer.fitter()->Config().MinimizerType(); // may have changed value
         statusHistory.emplace_back(_minimizer.fitter()->Config().MinimizerType() +
                                       _minimizer.fitter()->Config().MinimizerAlgoType() +
                                       std::to_string(_minimizer.fitter()->Config().MinimizerOptions().Strategy()),
                                    status);
         if (status % 1000 == 0)
            break; // fit was good

         if (status == 4 && minim != "Minuit") {
            if (printLevel >= -1) {
               Warning("fitTo", "%s Hit max function calls of %d", fitName.Data(),
                       _minimizer.fitter()->Config().MinimizerOptions().MaxFunctionCalls());
            }
            if (autoMaxCalls) {
               if (printLevel >= -1)
                  Warning("fitTo", "will try doubling this");
               _minimizer.fitter()->Config().MinimizerOptions().SetMaxFunctionCalls(
                  _minimizer.fitter()->Config().MinimizerOptions().MaxFunctionCalls() * 2);
               _minimizer.fitter()->Config().MinimizerOptions().SetMaxIterations(
                  _minimizer.fitter()->Config().MinimizerOptions().MaxIterations() * 2);
               continue;
            }
         }

         // NOTE: minuit2 seems to distort the tolerance in a weird way, so that tol becomes 1000 times smaller than
         // specified Also note that if fits are failing because of edm over max, it can be a good idea to activate the
         // Offset option when building nll
         if (printLevel >= -1) {
            Warning("fitTo", "%s %s%s Status=%d (edm=%f, tol=%f, strat=%d), tries=#%d...", fitName.Data(),
                    _minimizer.fitter()->Config().MinimizerType().c_str(),
                    _minimizer.fitter()->Config().MinimizerAlgoType().c_str(), status,
                    _minimizer.fitter()->Result().Edm(), _minimizer.fitter()->Config().MinimizerOptions().Tolerance(),
                    _minimizer.fitter()->Config().MinimizerOptions().Strategy(), tries);
         }

         // decide what to do next based on strategy sequence
         if (sIdx == m_strategy.Length() - 1) {
            break; // done
         }

         tries--;
         sIdx++;
      }

      /* Minuit2 status codes:
       * status = 0    : OK
            status = 1    : Covariance was made pos defined
             status = 2    : Hesse is invalid
             status = 3    : Edm is above max
             status = 4    : Reached call limit
             status = 5    : Any other failure

        For Minuit its basically 0 is OK, 4 is failure, I think?
       */

      if (printLevel >= -1 && status != 0) {
         Warning("fitTo", "%s final status is %d", fitName.Data(), status);
      }

      // currently dont have a way to access the covariance "dcovar" which is a metric from iterative
      // covariance method that is used by minuit2 to say if the covariance is accurate or not
      // See MinimumError.h: IsAccurate if Dcovar < 0.1
      // Note that if strategy>=2 or (strategy=1 and Dcovar>0.05) then hesse will be forced to be run (see
      // VariadicMetricBuilder) So only in Strategy=0 can you skip hesse (even if SetParabErrors false).

      int miniStrat = _minimizer.fitter()->Config().MinimizerOptions().Strategy();
      double dCovar = std::numeric_limits<double>::quiet_NaN();
      // if(auto _minuit2 = dynamic_cast<ROOT::Minuit2::Minuit2Minimizer*>(_minimizer.fitter()->GetMinimizer());
      // _minuit2 && _minuit2->fMinimum) {
      //    dCovar = _minuit2->fMinimum->Error().Dcovar();
      // }

      // only do hesse if was a valid min and not full accurate cov matrix already (can happen if e.g. ran strat2)
      if (hesse && m_hessestrategy.Length() != 0 &&
          (m_strategy(sIdx) == 'h' || (_minimizer.fitter()->Result().IsValid()))) {

         // Note: minima where the covariance was made posdef are deemed 'valid' ...

         // remove limits on pars before calculation - CURRENTLY HAS NO EFFECT, minuit still holds the state as
         // transformed interesting note: error on pars before hesse can be significantly smaller than after hesse ...
         // what is the pre-hesse error corresponding to? - corresponds to approximation of covariance matrix calculated
         // with iterative method
         /*auto parSettings = _minimizer.fitter()->Config().ParamsSettings();
         for (auto &ss : _minimizer.fitter()->Config().ParamsSettings()) {
            ss.RemoveLimits();
         }

         for(auto f : *floatPars) {
            auto v = dynamic_cast<RooRealVar*>(f);
            if(v->hasRange(nullptr)) v->setRange("backup",v->getMin(),v->getMax());
            v->removeRange();
         }*/

         // std::cout << "nIterations = " << _minimizer.fitter()->GetMinimizer()->NIterations() << std::endl;
         // std::cout << "covQual before hesse = " << _minimizer.fitter()->GetMinimizer()->CovMatrixStatus() <<
         // std::endl;
         sIdx = -1;
         if (hesseStrategy == -1) {
            sIdx = 0;
         } else {
            sIdx = m_hessestrategy.Index('0' + hesseStrategy);
         }
         if (sIdx == -1) {
            Warning("minimize",
                    "HesseStrategy %d not specified in HesseStrategySequence %s ... defaulting to start of sequence",
                    hesseStrategy, m_hessestrategy.Data());
            sIdx = 0;
         }
         while (sIdx != -1) {
            hesseStrategy = int(m_hessestrategy(sIdx) - '0');

            if (strategy == 2 && hesseStrategy == 2) {
               // don't repeat hesse if strategy=2 and hesseStrategy=2, and the matrix was valid
               if (_minimizer.fitter()->GetMinimizer()->CovMatrixStatus() == 3) {
                  break;
               }
               if (sIdx >= m_hessestrategy.Length() - 1) {
                  break; // run out of strategies to try, stop
               }
               sIdx++;
               continue;
            }

            _minimizer.fitter()->Config().MinimizerOptions().SetStrategy(hesseStrategy);
            // const_cast<ROOT::Math::IOptions*>(_minimizer.fitter()->Config().MinimizerOptions().ExtraOptions())->SetValue("HessianStepTolerance",0.1);
            // const_cast<ROOT::Math::IOptions*>(_minimizer.fitter()->Config().MinimizerOptions().ExtraOptions())->SetValue("HessianG2Tolerance",0.02);

            if (auto fff = dynamic_cast<ProgressMonitor *>(_nll); fff) {
               fff->fState = TString::Format("Hesse%d", _minimizer.fitter()->Config().MinimizerOptions().Strategy());
               fff->counter2 = fff->counter;
            }

            //_nll->getVal(); // for reasons I dont understand, if nll evaluated before hesse call the edm is smaller? -
            // and also becomes WRONG :-S

            // auto _status = (_minimizer.fitter()->CalculateHessErrors()) ? _minimizer.fitter()->Result().Status() :
            // -1;
            auto _status = _minimizer.hesse(); // note: I have seen that you can get 'full covariance quality' without
                                               // running hesse ... is that expected?
            // note: hesse status will be -1 if hesse failed (no covariance matrix)
            // otherwise the status appears to be whatever was the status before
            // note that hesse succeeds even if the cov matrix it calculates is forced pos def. Failure is only
            // if it cannot calculate a cov matrix at all.
            if (_status != -1)
               _status = 0; // mark as hesse succeeded, although need to look at covQual to see if was any good

            /*for(auto f : *floatPars) {
               auto v = dynamic_cast<RooRealVar*>(f);
               if(v->hasRange("backup")) {
                  v->setRange(v->getMin(),v->getMax());
                  v->removeRange("backup");
               }
            }
            _minimizer.fitter()->Config().SetParamsSettings(parSettings);*/

            /*for (auto &ss : _minimizer.fitter()->Config().ParamsSettings()) {
               if( ss.HasLowerLimit() || ss.HasUpperLimit() ) std::cout << ss.Name() << " limit restored " <<
            ss.LowerLimit() << " - " << ss.UpperLimit() << std::endl;
            }*/

            statusHistory.push_back(std::pair<std::string, int>(
               TString::Format("Hesse%d", _minimizer.fitter()->Config().MinimizerOptions().Strategy()), _status));

            if (auto fff = dynamic_cast<ProgressMonitor *>(_nll); fff && fff->fInterrupt) {
               delete _nll;
               throw std::runtime_error("Keyboard interrupt while hesse calculating");
            }
            if ((_status != 0 || _minimizer.fitter()->GetMinimizer()->CovMatrixStatus() != 3) && status == 0 &&
                printLevel >= -1) {
               Warning("fitTo", "%s hesse status is %d, covQual=%d", fitName.Data(), _status,
                       _minimizer.fitter()->GetMinimizer()->CovMatrixStatus());
            }

            if (sIdx >= m_hessestrategy.Length() - 1) {
               break; // run out of strategies to try, stop
            }

            if (_status == 0 && _minimizer.fitter()->GetMinimizer()->CovMatrixStatus() == 3) {
               // covariance is valid!
               break;
            } else if (_status == 0) {
               // set the statusHistory to the cov status, since that's more informative
               statusHistory.back().second = _minimizer.fitter()->GetMinimizer()->CovMatrixStatus();
            }
            sIdx++;
         } // end of hesse attempt loop
      }

      // call minos if requested on any parameters
      if (status == 0 && minos) {
         if (std::unique_ptr<RooAbsCollection> mpars(floatPars->selectByAttrib("minos", true)); !mpars->empty()) {
            if (auto fff = dynamic_cast<ProgressMonitor *>(_nll); fff) {
               fff->fState = "Minos";
               fff->counter2 = 0;
            }
            auto _status = _minimizer.minos(*mpars);
            statusHistory.push_back(std::pair("Minos", _status));
         }
      }

      // DO NOT DO THIS - seems to mess with the NLL function in a way that breaks the cache - reactivating wont fix
      // if(constOptimize) { _minimizer.optimizeConst(0); } // doing this because saw happens in RooAbsPdf::minimizeNLL
      // method

      // signal(SIGINT,gOldHandlerr);
      out = std::unique_ptr<RooFitResult>{_minimizer.save(fitName, resultTitle)};

      // if status is 0 (min succeeded) but the covQual isn't fully accurate but requested hesse, reflect that in the
      // status
      if (out->status() == 0 && out->covQual() != 3 && hesse) {
         if (out->covQual() == 2) { // was made posdef
            out->setStatus(1);      // indicates covariance made pos-def
         } else { // anything else indicates either hessian is approximate or something else wrong (e.g. not pos-def
                  // return from strat3)
            out->setStatus(2); // hesse invalid
         }
      }

      if(miniStrat < _minimizer.fitter()->Config().MinimizerOptions().Strategy() && hesse && out->edm() > _minimizer.fitter()->Config().MinimizerOptions().Tolerance()*1e-3 && out->status() != 3) {
         // hesse may have updated edm by using a better strategy than used in the minimization
         // so print a warning about this
         std::cerr << "Warning: post-Hesse edm greater than allowed by tolerance. Consider increasing minimization strategy" << std::endl;
         // Dec24: As this is a new warning, will not update status code for now, so edm will be large
         // but in the future we should probably update the code to 3 so that users don't miss this warning.
         // out->setStatus(3); // edm above max
      }

      out->setStatusHistory(statusHistory);

      // userPars wont have been added to the RooFitResult by RooMinimizer
      const_cast<RooArgList &>(out->constPars()).addClone(fUserPars, true);

      if (!std::isnan(dCovar)) {
         const_cast<RooArgList &>(out->constPars())
            .addClone(RooRealVar(".dCovar", "dCovar from minimization", dCovar), true);
      }

      if (boundaryCheck) {
         // check if any of the parameters are at their limits (potentially a problem with fit)
         // or their errors go over their limits (just a warning)
         int limit_status = 0;
         std::string listpars;
         for (auto *v : dynamic_range_cast<RooRealVar *>(*floatPars)) {
            if (!v)
               continue;
            double vRange = v->getMax() - v->getMin();
            if (v->getMin() > v->getVal() - vRange * boundaryCheck ||
                v->getMax() < v->getVal() + vRange * boundaryCheck) {
               // within 0.01% of edge

               // check if nll actually lower 'at' the boundary, if it is, refine the best fit to the limit value
               auto tmp = v->getVal();
               v->setVal(v->getMin());
               double boundary_nll = _nll->getVal();
               if (boundary_nll <= out->minNll()) {
                  static_cast<RooRealVar *>(out->floatParsFinal().find(v->GetName()))->setVal(v->getMin());
                  out->setMinNLL(boundary_nll);
                  // Info("fit","Corrected %s onto minimum @ %g",v->GetName(),v->getMin());
               } else {
                  // not better, so restore value
                  v->setVal(tmp);
               }

               // if has a 'physical' range specified, don't warn if near the limit
               if (v->hasRange("physical"))
                  limit_status = 900;
               listpars += v->GetName();
               listpars += ",";
            } else if (hesse &&
                       (v->getMin() > v->getVal() - v->getError() || v->getMax() < v->getVal() + v->getError())) {
               if (printLevel >= 0) {
                  Info("minimize", "PARLIM: %s (%f +/- %f) range (%f - %f)", v->GetName(), v->getVal(), v->getError(),
                       v->getMin(), v->getMax());
               }
               limit_status = 9000;
            }
         }
         if (limit_status == 900) {
            if (printLevel >= 0) {
               Warning("minimize", "BOUNDCHK: Parameters within %g%% limit in fit result: %s", boundaryCheck * 100,
                       listpars.c_str());
            }
         } else if (limit_status > 0) {
            if (printLevel >= 0)
               Warning("minimize", "BOUNDCHK: Parameters near limit in fit result");
         }

         // store the limit check result
         statusHistory.emplace_back("BOUNDCHK", limit_status);
         out->setStatusHistory(statusHistory);
         out->setStatus(out->status() + limit_status);
      }

      //        // automatic parameter range adjustment based on errors
      //        for(auto a : *floatPars) {
      //            RooRealVar *v = dynamic_cast<RooRealVar *>(a);
      //            if(v->getMin() > v->getVal() - 3.*v->getError()) {
      //                v->setMin(v->getVal() - 3.1*v->getError());
      //            }
      //            if(v->getMax() < v->getVal() + 3.*v->getError()) {
      //                v->setMax(v->getVal() + 3.1*v->getError());
      //            }
      //            // also make sure the range isn't too big (fits can struggle)
      //            if(v->getMin() < v->getVal() - 10.*v->getError()) {
      //                v->setMin(v->getVal() - 9.9*v->getError());
      //            }
      //            if(v->getMax() > v->getVal() + 10.*v->getError()) {
      //                v->setMax(v->getVal() + 9.9*v->getError());
      //            }
      //        }

      if (printLevel < 0)
         RooMsgService::instance().setGlobalKillBelow(msglevel);

      // before returning we will override _minLL with the actual NLL value ... offsetting could have messed up the
      // value
      out->setMinNLL(_nll->getVal());

      // ensure no asymm errors on any pars unless had minuitMinos
      for (auto o : out->floatParsFinal()) {
         if (auto v = dynamic_cast<RooRealVar *>(o);
             v && !v->getAttribute("minos") && !v->getAttribute("xminos") && !v->getAttribute("xMinos"))
            v->removeAsymError();
      }

      // minimizer may have slightly altered the fitConfig (e.g. unavailable minimizer etc) so update for that ...
      if (fitConfig.MinimizerOptions().MinimizerType() != actualFirstMinimizer) {
         fitConfig.MinimizerOptions().SetMinimizerType(actualFirstMinimizer);
      }

      if (_progress) {
         delete _nll;
      }
   }

   if(out && out->status() == 0 && minos) {
      // call minos if requested on any parameters
      for (auto label : {"xminos", "xMinos"}) {
         std::unique_ptr<RooAbsCollection> pars(floatPars->selectByAttrib(label, true));
         for (auto p : *pars) {
            Info("minimize", "Computing xminos error for %s", p->GetName());
            xRooFit::minos(nll, *out, p->GetName(), myFitConfig);
         }
         if (!pars->empty())
            *floatPars = out->floatParsFinal(); // put values back to best fit
      }
   }

   if (restore) {
      *floatPars = out->floatParsInit();
   }

   if (out && !logs.empty()) {
      // save logs to StringVar in constPars list
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 28, 00)
      const_cast<RooArgList &>(out->constPars()).addOwned(std::make_unique<RooStringVar>(".log", "log", logs.c_str()));
#else
      const_cast<RooArgList &>(out->constPars()).addOwned(*new RooStringVar(".log", "log", logs.c_str()));
#endif
   }

   if (out && cacheDir && cacheDir->IsWritable()) {
      // std::cout << "Saving " << out->GetName() << " " << out->GetTitle() << " to " << nll.GetName() << std::endl;
      //  save a copy of fit result to relevant dir
      if (!cacheDir->GetDirectory(nll.GetName()))
         cacheDir->mkdir(nll.GetName());
      if (auto dir = cacheDir->GetDirectory(nll.GetName()); dir) {
         // save NLL opts if was given one, unless already present
         if (nllOpts) {
            if (strlen(nllOpts->GetName()) == 0) {
               nllOpts->SetName(TUUID().AsString());
            }
            if (!dir->FindKey(nllOpts->GetName())) {
               dir->WriteObject(nllOpts.get(), nllOpts->GetName());
            }
         }

         // also save the fitConfig ... unless one with same name already present
         std::string configName;
         if (!fitConfig.MinimizerOptions().ExtraOptions()->GetValue("Name", configName)) {
            auto extraOpts = const_cast<ROOT::Math::IOptions *>(fitConfig.MinimizerOptions().ExtraOptions());
            configName = TUUID().AsString();
            extraOpts->SetValue("Name", configName.data());
         }
         if (!dir->GetKey(configName.data())) {
            dir->WriteObject(&fitConfig, configName.data());
         }
         // add the fitConfig name into the fit result before writing, so can retrieve in future
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 28, 00)
         const_cast<RooArgList &>(out->constPars())
            .addOwned(std::make_unique<RooStringVar>(".fitConfigName", "fitConfigName", configName.c_str()));
#else
         const_cast<RooArgList &>(out->constPars())
            .addOwned(*new RooStringVar(".fitConfigName", "fitConfigName", configName.c_str()));
#endif
         dir->WriteObject(out.get(), out->GetName());
         auto sfr = new StoredFitResult(out);
         dir->Add(sfr);
         return sfr->fr;
         // return std::shared_ptr<const RooFitResult>(out, [](const RooFitResult*){}); // disowned shared_ptr
      }
   }

   return out;
}

// calculate asymmetric errors, if required, on the named parameter that was floating in the fit
// returns status code. 0 = all good, 1 = failure, ...
int xRooFit::minos(RooAbsReal &nll, const RooFitResult &ufit, const char *parName,
                   const std::shared_ptr<ROOT::Fit::FitConfig> &_fitConfig)
{

   auto par = dynamic_cast<RooRealVar *>(std::unique_ptr<RooArgSet>(nll.getVariables())->find(parName));
   if (!par)
      return 1;

   auto par_hat = dynamic_cast<RooRealVar *>(ufit.floatParsFinal().find(parName));
   if (!par_hat)
      return 1;

   auto myFitConfig = _fitConfig ? _fitConfig : createFitConfig();
   auto &fitConfig = *myFitConfig;

   bool pErrs = fitConfig.ParabErrors();
   fitConfig.SetParabErrors(false);
   double mErrs = fitConfig.MinosErrors();
   fitConfig.SetMinosErrors(false);

   double val_best = par_hat->getVal();
   double val_err = (par_hat->hasError() ? par_hat->getError() : -1);
   double orig_err = val_err;
   double nll_min = ufit.minNll();

   int status = 0;

   bool isConst = par->isConstant();
   par->setConstant(true);

   auto findValue = [&](double val_guess, double N_sigma = 1, double precision = 0.002, int printLevel = 0) {
      double tmu;
      int nrItr = 0;
      double sigma_guess = std::abs((val_guess - val_best) / N_sigma);
      double val_pre =
         val_guess -
         10 * precision * sigma_guess; // this is just to set value st. guarantees will do at least one iteration
      bool lastOverflow = false;
      bool lastUnderflow = false;
      while (std::abs(val_pre - val_guess) > precision * sigma_guess) {
         val_pre = val_guess;
         if (val_guess > 0 && par->getMax() < val_guess)
            par->setMax(2 * val_guess);
         if (val_guess < 0 && par->getMin() > val_guess)
            par->setMin(2 * val_guess);
         par->setVal(val_guess);
         // std::cout << "Guessing " << val_guess << std::endl;
         auto result = xRooFit::minimize(nll, myFitConfig);
         if (!result) {
            status = 1;
            return std::numeric_limits<double>::quiet_NaN();
         }
         double nll_val = result->minNll();
         status += result->status() * 10;
         tmu = 2 * (nll_val - nll_min);
         sigma_guess = std::abs(val_guess - val_best) / sqrt(tmu);

         if (tmu <= 0) {
            // found an alternative or improved minima
            std::cout << "Warning: Alternative best-fit of " << par->GetName() << " @ " << val_guess << " vs "
                      << val_best << " (delta=" << tmu / 2. << ")" << std::endl;
            double new_guess = val_guess + (val_guess - val_best);
            val_best = val_guess;
            val_guess = new_guess;
            sigma_guess = std::abs((val_guess - val_best) / N_sigma);
            val_pre = val_guess - 10 * precision * sigma_guess;
            status = (status / 10) * 10 + 1;
            continue;
         }

         double corr = /*damping_factor**/ (val_pre - val_best - N_sigma * sigma_guess);

         // subtract off the difference in the new and damped correction
         val_guess -= corr;

         if (printLevel > 1) {
            // cout << "nPars:          " << nPars << std::endl;
            // cout << "NLL:            " << nll->GetName() << " = " << nll->getVal() << endl;
            // cout << "delta(NLL):     " << nll->getVal()-nll_min << endl;
            std::cout << "NLL min: " << nll_min << std::endl;
            std::cout << "N_sigma*sigma(pre):   " << std::abs(val_pre - val_best) << std::endl;
            std::cout << "sigma(guess):   " << sigma_guess << std::endl;
            std::cout << "par(guess):     " << val_guess + corr << std::endl;
            std::cout << "true val:       " << val_best << std::endl;
            std::cout << "tmu:            " << tmu << std::endl;
            std::cout << "Precision:      " << sigma_guess * precision << std::endl;
            std::cout << "Correction:     " << (-corr < 0 ? " " : "") << -corr << std::endl;
            std::cout << "N_sigma*sigma(guess): " << std::abs(val_guess - val_best) << std::endl;
            std::cout << std::endl;
         }
         if (val_guess > par->getMax()) {
            if (lastOverflow) {
               val_guess = par->getMin();
               break;
            }
            lastOverflow = true;
            lastUnderflow = false;
            val_guess = par->getMax() - 1e-12;
         } else if (val_guess < par->getMin()) {
            if (lastUnderflow) {
               val_guess = par->getMin();
               break;
            }
            lastOverflow = false;
            lastUnderflow = true;
            val_guess = par->getMin() + 1e-12;
         } else {
            lastUnderflow = false;
            lastOverflow = false;
         }

         nrItr++;
         if (nrItr > 25) {
            status = (status / 10) * 10 + 3;
            break;
         }
      }

      if (lastOverflow) {
         // msg().Error("findSigma","%s at upper limit of %g .. error may be underestimated
         // (t=%g)",par->GetName(),par->getMax(),tmu);
         status = (status / 10) * 10 + 2;
      } else if (lastUnderflow) {
         // msg().Error("findSigma","%s at lower limit of %g .. error may be underestimated
         // (t=%g)",par->GetName(),par->getMin(),tmu);
         status = (status / 10) * 10 + 2;
      }

      if (printLevel > 1)
         std::cout << "Found sigma for nll " << nll.GetName() << ": " << (val_guess - val_best) / N_sigma << std::endl;
      if (printLevel > 1)
         std::cout << "Finished in " << nrItr << " iterations." << std::endl;
      if (printLevel > 1)
         std::cout << std::endl;
      return (val_guess - val_best) / N_sigma;
   };

   // determine if asym error defined by temporarily setting error to nan ... will then return non-nan if defined

   par_hat->setError(std::numeric_limits<double>::quiet_NaN());
   double lo = par_hat->getErrorLo();
   double hi = par_hat->getErrorHi();
   if (std::isnan(hi)) {
      hi = findValue(val_best + val_err, 1) + val_best -
           par_hat->getVal(); // put error wrt par_hat value, even if found better min
      if (hi > val_err)
         val_err = hi; // in case val_err was severe underestimate, don't want to waste time being too 'near' min
   }
   if (std::isnan(lo)) {
      lo = -findValue(val_best - val_err, -1) + val_best -
           par_hat->getVal(); // put error wrt par_hat value, even if found better min
   }
   dynamic_cast<RooRealVar *>(ufit.floatParsFinal().find(parName))->setAsymError(lo, hi);
   par_hat->setError(orig_err);

   fitConfig.SetParabErrors(pErrs);
   fitConfig.SetMinosErrors(mErrs);
   par->setConstant(isConst);

   std::vector<std::pair<std::string, int>> statusHistory;
   for (unsigned int i = 0; i < ufit.numStatusHistory(); i++) {
      statusHistory.emplace_back(ufit.statusLabelHistory(i), ufit.statusCodeHistory(i));
   }
   statusHistory.emplace_back(TString::Format("xMinos:%s", parName), status);
   const_cast<RooFitResult &>(ufit).setStatusHistory(statusHistory);
   const_cast<RooFitResult &>(ufit).setStatus(ufit.status() + status);

   return status;
}

TCanvas *
xRooFit::hypoTest(RooWorkspace &w, int nToysNull, int /*nToysAlt*/, const xRooFit::Asymptotics::PLLType &pllType)
{
   TCanvas *out = nullptr;

   // 1. Determine pdf: use top-level, if more than 1 then exit and tell user they need to flag
   RooAbsPdf *model = nullptr;
   std::deque<RooAbsArg *> topPdfs;
   int flagCount = 0;
   for (auto p : w.allPdfs()) {
      if (p->hasClients())
         continue;
      flagCount += p->getAttribute("hypoTest");
      if (p->getAttribute("hypoTest")) {
         topPdfs.push_front(p);
      } else {
         topPdfs.push_back(p);
      }
   }
   if (topPdfs.empty()) {
      Error("hypoTest", "Cannot find top-level pdf in workspace");
      return nullptr;
   } else if (topPdfs.size() > 1) {
      // should be one flagged
      if (flagCount == 0) {
         Error("hypoTest", "Multiple top-level pdfs. Flag which one to test with "
                           "w->pdf(\"pdfName\")->setAttribute(\"hypoTest\",true)");
         return out;
      } else if (flagCount != 1) {
         Error("hypoTest", "Multiple top-level pdfs flagged for hypoTest -- pick one.");
         return out;
      }
   }
   model = dynamic_cast<RooAbsPdf *>(topPdfs.front());

   Info("hypoTest", "Using PDF: %s", model->GetName());

   double CL = 0.95; // TODO: make configurable

   // 2. Determine the data (including globs). if more than 1 then exit and tell user they need to flag
   RooAbsData *obsData = nullptr;
   std::shared_ptr<RooArgSet> obsGlobs = nullptr;

   for (auto p : w.allData()) {
      if (obsData) {
         Error("hypoTest", "Multiple datasets in workspace. Flag which one to test with "
                           "w->data(\"dataName\")->setAttribute(\"hypoTest\",true)");
         return out;
      }
      obsData = p;
   }

   if (!obsData) {
      Error("hypoTest", "No data -- cannot determine observables");
      return nullptr;
   }

   Info("hypoTest", "Using Dataset: %s", obsData->GetName());

   {
      auto _globs = xRooNode(w).datasets()[obsData->GetName()]->globs(); // keep alive because may own the globs
      obsGlobs = std::make_shared<RooArgSet>();
      obsGlobs->addClone(_globs.argList());
      Info("hypoTest", "Using Globs: %s", (obsGlobs->empty()) ? " <NONE>" : obsGlobs->contentsString().c_str());
   }

   // 3. Determine the POI and args - look for model pars with "hypoPoints" binning, if none then cannot scan
   //  args are const, poi are floating - exception is if only one then assume it is the POI
   auto _vars = std::unique_ptr<RooArgSet>(model->getVariables());
   RooArgSet poi;
   RooArgSet args;
   for (auto _v : *_vars) {
      if (auto v = dynamic_cast<RooRealVar *>(_v); v && v->hasBinning("hypoPoints")) {
         poi.add(*v);
      }
   }
   if (poi.size() > 1) {
      auto _const = std::unique_ptr<RooAbsCollection>(poi.selectByAttrib("Constant", true));
      args.add(*_const);
      poi.remove(*_const);
   }
   if (!args.empty()) {
      Info("hypoTest", "Using Arguments: %s", args.contentsString().c_str());
   }
   if (poi.empty()) {
      Error("hypoTest", "No POI detected: add the hypoPoints binning to at least one non-const model parameter e.g.:\n "
                        "w->var(\"mu\")->setBinning(RooUniformBinning(0.5,10.5,10),\"hypoPoints\"))");
      return nullptr;
   }

   Info("hypoTest", "Using Parameters of Interest: %s", poi.contentsString().c_str());

   out = TCanvas::MakeDefCanvas();

   // should check if exist in workspace
   auto nllOpts = createNLLOptions();
   auto fitConfig = createFitConfig();

   xRooNLLVar nll(*model, std::make_pair(obsData, obsGlobs.get()), *nllOpts);
   nll.SetFitConfig(fitConfig);

   if (poi.size() == 1) {
      auto mu = dynamic_cast<RooRealVar *>(poi.first());

      double altVal = (mu->getStringAttribute("altVal")) ? TString(mu->getStringAttribute("altVal")).Atof()
                                                         : std::numeric_limits<double>::quiet_NaN();

      if (std::isnan(altVal) && mu->hasRange("physical")) {
         // use the smallest absolute value for the altValue
         altVal = mu->getMin("physical");
         Info("hypoTest", "No altVal specified - using min of given physical range = %g", altVal);
      } else {
         if (!std::isnan(altVal)) {
            Info("hypoTest", "alt hypo: %g - CLs activated", altVal);
         } else {
            Info("hypoTest", "No altVal found - to specify setStringAttribute(\"altVal\",\"<value>\") on POI or set "
                             "the physical range");
         }
      }
      bool doCLs = !std::isnan(altVal) && std::abs(mu->getMin("hypoPoints")) > altVal &&
                   std::abs(mu->getMax("hypoPoints")) > altVal;

      const char *sCL = (doCLs) ? "CLs" : "null";
      Info("hypoTest", "%s testing active", sCL);

      auto obs_ts = new TGraphErrors;
      obs_ts->SetNameTitle("obs_ts", TString::Format("Observed TestStat;%s", mu->GetTitle()));
      auto obs_pcls = new TGraphErrors;
      obs_pcls->SetNameTitle(TString::Format("obs_p%s", sCL),
                             TString::Format("Observed p_{%s};%s", sCL, mu->GetTitle()));
      auto obs_cls = new TGraphErrors;
      obs_cls->SetNameTitle(TString::Format("obs_%s", sCL), TString::Format("Observed %s;%s", sCL, mu->GetTitle()));

      std::vector<int> expSig = {-2, -1, 0, 1, 2};
      if (std::isnan(altVal))
         expSig.clear();
      std::map<int, TGraphErrors> exp_pcls;
      std::map<int, TGraphErrors> exp_cls;
      for (auto &s : expSig) {
         exp_pcls[s].SetNameTitle(TString::Format("exp%d_p%s", s, sCL),
                                  TString::Format("Expected (%d#sigma) p_{%s};%s", s, sCL, mu->GetTitle()));
         exp_cls[s].SetNameTitle(TString::Format("exp%d_%s", s, sCL),
                                 TString::Format("Expected (%d#sigma) %s;%s", s, sCL, mu->GetTitle()));
      }

      auto getLimit = [CL](TGraphErrors &pValues) {
         double _out = std::numeric_limits<double>::quiet_NaN();
         bool lastAbove = false;
         for (int i = 0; i < pValues.GetN(); i++) {
            bool thisAbove = pValues.GetPointY(i) >= (1. - CL);
            if (i != 0 && thisAbove != lastAbove) {
               // crossed over ... find limit by interpolation
               // using linear interpolation so far
               _out = pValues.GetPointX(i - 1) + (pValues.GetPointX(i) - pValues.GetPointX(i - 1)) *
                                                    ((1. - CL) - pValues.GetPointY(i - 1)) /
                                                    (pValues.GetPointY(i) - pValues.GetPointY(i - 1));
            }
            lastAbove = thisAbove;
         }
         return _out;
      };

      auto testPoint = [&](double testVal) {
         auto hp = nll.hypoPoint(mu->GetName(), testVal, altVal, pllType);
         obs_ts->SetPoint(obs_ts->GetN(), testVal, hp.pll().first);
         obs_ts->SetPointError(obs_ts->GetN() - 1, 0, hp.pll().second);

         if (nToysNull > 0) {
         }

         obs_pcls->SetPoint(obs_pcls->GetN(), testVal, (doCLs) ? hp.pCLs_asymp().first : hp.pNull_asymp().first);
         obs_pcls->SetPointError(obs_pcls->GetN() - 1, 0, (doCLs) ? hp.pCLs_asymp().second : hp.pNull_asymp().second);
         for (auto &s : expSig) {
            exp_pcls[s].SetPoint(exp_pcls[s].GetN(), testVal,
                                 (doCLs) ? hp.pCLs_asymp(s).first : hp.pNull_asymp(s).first);
         }
         if (doCLs) {
            Info("hypoTest", "%s=%g: %s=%g sigma_mu=%g %s=%g", mu->GetName(), testVal, obs_ts->GetName(),
                 obs_ts->GetPointY(obs_ts->GetN() - 1), hp.sigma_mu().first, obs_pcls->GetName(),
                 obs_pcls->GetPointY(obs_pcls->GetN() - 1));
         } else {
            Info("hypoTest", "%s=%g: %s=%g %s=%g", mu->GetName(), testVal, obs_ts->GetName(),
                 obs_ts->GetPointY(obs_ts->GetN() - 1), obs_pcls->GetName(), obs_pcls->GetPointY(obs_pcls->GetN() - 1));
         }
      };

      if (mu->getBins("hypoPoints") <= 0) {
         // autoTesting
         // evaluate min and max points
         testPoint(mu->getMin("hypoPoints"));
         testPoint(mu->getMax("hypoPoints"));
         testPoint((mu->getMax("hypoPoints") + mu->getMin("hypoPoints")) / 2.);

         while (std::abs(obs_pcls->GetPointY(obs_pcls->GetN() - 1) - (1. - CL)) > 0.01) {
            obs_pcls->Sort();
            double nextTest = getLimit(*obs_pcls);
            if (std::isnan(nextTest))
               break;
            testPoint(nextTest);
         }
         for (auto s : expSig) {
            while (std::abs(exp_pcls[s].GetPointY(exp_pcls[s].GetN() - 1) - (1. - CL)) > 0.01) {
               exp_pcls[s].Sort();
               double nextTest = getLimit(exp_pcls[s]);
               if (std::isnan(nextTest))
                  break;
               testPoint(nextTest);
            }
         }
         obs_ts->Sort();
         obs_pcls->Sort();
         for (auto &s : expSig)
            exp_pcls[s].Sort();

      } else {
         for (int i = 0; i <= mu->getBins("hypoPoints"); i++) {
            testPoint((i == mu->getBins("hypoPoints")) ? mu->getBinning("hypoPoints").binHigh(i - 1)
                                                       : mu->getBinning("hypoPoints").binLow(i));
         }
      }

      obs_cls->SetPoint(obs_cls->GetN(), getLimit(*obs_pcls), 0.05);
      for (auto &s : expSig) {
         exp_cls[s].SetPoint(exp_cls[s].GetN(), getLimit(exp_pcls[s]), 0.05);
      }

      // if more than two hypoPoints, visualize as bands
      if (exp_pcls[2].GetN() > 1) {
         TGraph *band2 = new TGraph;
         band2->SetNameTitle(".pCLs_2sigma", "2 sigma band");
         TGraph *band2up = new TGraph;
         band2up->SetNameTitle(".pCLs_2sigma_upUncert", "");
         TGraph *band2down = new TGraph;
         band2down->SetNameTitle(".pCLs_2sigma_downUncert", "");
         band2->SetFillColor(kYellow);
         band2up->SetFillColor(kYellow);
         band2down->SetFillColor(kYellow);
         band2up->SetFillStyle(3005);
         band2down->SetFillStyle(3005);
         for (int i = 0; i < exp_pcls[2].GetN(); i++) {
            band2->SetPoint(band2->GetN(), exp_pcls[2].GetPointX(i),
                            exp_pcls[2].GetPointY(i) - exp_pcls[2].GetErrorYlow(i));
            band2up->SetPoint(band2up->GetN(), exp_pcls[2].GetPointX(i),
                              exp_pcls[2].GetPointY(i) + exp_pcls[2].GetErrorYhigh(i));
         }
         for (int i = exp_pcls[2].GetN() - 1; i >= 0; i--) {
            band2up->SetPoint(band2up->GetN(), exp_pcls[2].GetPointX(i),
                              exp_pcls[2].GetPointY(i) - exp_pcls[2].GetErrorYlow(i));
         }
         for (int i = 0; i < exp_pcls[-2].GetN(); i++) {
            band2down->SetPoint(band2down->GetN(), exp_pcls[-2].GetPointX(i),
                                exp_pcls[-2].GetPointY(i) + exp_pcls[-2].GetErrorYhigh(i));
         }
         for (int i = exp_pcls[-2].GetN() - 1; i >= 0; i--) {
            band2->SetPoint(band2->GetN(), exp_pcls[-2].GetPointX(i),
                            exp_pcls[-2].GetPointY(i) + exp_pcls[-2].GetErrorYhigh(i));
            band2down->SetPoint(band2down->GetN(), exp_pcls[-2].GetPointX(i),
                                exp_pcls[-2].GetPointY(i) - exp_pcls[-2].GetErrorYlow(i));
         }
         band2->SetBit(kCanDelete);
         band2up->SetBit(kCanDelete);
         band2down->SetBit(kCanDelete);
         auto ax = static_cast<TNamed *>(band2->Clone(".axis"));
         ax->SetTitle(TString::Format("Hypothesis Test;%s", mu->GetTitle()));
         ax->Draw("AF");
         band2->Draw("F");
         band2up->Draw("F");
         band2down->Draw("F");
      }

      if (exp_pcls[1].GetN() > 1) {
         TGraph *band2 = new TGraph;
         band2->SetNameTitle(".pCLs_1sigma", "1 sigma band");
         TGraph *band2up = new TGraph;
         band2up->SetNameTitle(".pCLs_1sigma_upUncert", "");
         TGraph *band2down = new TGraph;
         band2down->SetNameTitle(".pCLs_1sigma_downUncert", "");
         band2->SetFillColor(kGreen);
         band2up->SetFillColor(kGreen);
         band2down->SetFillColor(kGreen);
         band2up->SetFillStyle(3005);
         band2down->SetFillStyle(3005);
         for (int i = 0; i < exp_pcls[1].GetN(); i++) {
            band2->SetPoint(band2->GetN(), exp_pcls[1].GetPointX(i),
                            exp_pcls[1].GetPointY(i) - exp_pcls[1].GetErrorYlow(i));
            band2up->SetPoint(band2up->GetN(), exp_pcls[1].GetPointX(i),
                              exp_pcls[1].GetPointY(i) + exp_pcls[1].GetErrorYhigh(i));
         }
         for (int i = exp_pcls[1].GetN() - 1; i >= 0; i--) {
            band2up->SetPoint(band2up->GetN(), exp_pcls[1].GetPointX(i),
                              exp_pcls[1].GetPointY(i) - exp_pcls[1].GetErrorYlow(i));
         }
         for (int i = 0; i < exp_pcls[-1].GetN(); i++) {
            band2down->SetPoint(band2down->GetN(), exp_pcls[-1].GetPointX(i),
                                exp_pcls[-1].GetPointY(i) + exp_pcls[-1].GetErrorYhigh(i));
         }
         for (int i = exp_pcls[-1].GetN() - 1; i >= 0; i--) {
            band2->SetPoint(band2->GetN(), exp_pcls[-1].GetPointX(i),
                            exp_pcls[-1].GetPointY(i) + exp_pcls[-1].GetErrorYhigh(i));
            band2down->SetPoint(band2down->GetN(), exp_pcls[-1].GetPointX(i),
                                exp_pcls[-1].GetPointY(i) - exp_pcls[-1].GetErrorYlow(i));
         }
         band2->SetBit(kCanDelete);
         band2up->SetBit(kCanDelete);
         band2down->SetBit(kCanDelete);
         band2->Draw("F");
         band2up->Draw("F");
         band2down->Draw("F");
      }

      TObject *expPlot = nullptr;
      if (exp_cls[0].GetN() > 0) {
         exp_pcls[0].SetLineStyle(2);
         exp_pcls[0].SetFillColor(kGreen);
         exp_pcls[0].SetMarkerStyle(0);
         expPlot = exp_pcls[0].DrawClone("L");
      }
      obs_pcls->SetBit(kCanDelete);
      obs_pcls->Draw(gPad->GetListOfPrimitives()->IsEmpty() ? "ALP" : "LP");

      obs_ts->SetLineColor(kRed);
      obs_ts->SetMarkerColor(kRed);
      obs_ts->SetBit(kCanDelete);
      obs_ts->Draw("LP");

      auto l = new TLegend(0.5, 0.6, 1. - gPad->GetRightMargin(), 1. - gPad->GetTopMargin());
      l->SetName("legend");
      l->AddEntry(obs_ts, obs_ts->GetTitle(), "LPE");
      l->AddEntry(obs_pcls, obs_pcls->GetTitle(), "LPE");
      if (expPlot)
         l->AddEntry(expPlot, "Expected", "LFE");
      l->SetBit(kCanDelete);
      l->Draw();

      obs_cls->SetMarkerStyle(29);
      obs_cls->SetEditable(false);
      obs_cls->Draw("LP");
      for (auto s : expSig) {
         exp_cls[s].SetMarkerStyle(29);
         exp_cls[s].SetEditable(false);
         exp_cls[s].DrawClone("LP");
      }
   }

   if (out)
      out->RedrawAxis();

   return out;
}

double round_to_digits(double value, int digits)
{
   if (value == 0.0)
      return 0.0;
   double factor = pow(10.0, digits - ceil(log10(std::abs(value))));
   return std::round(value * factor) / factor;
}
double round_to_decimal(double value, int decimal_places)
{
   const double multiplier = std::pow(10.0, decimal_places);
   return std::round(value * multiplier) / multiplier;
}

// rounds error to 1 or 2 sig fig and round value to match that precision
std::pair<double, double> xRooFit::matchPrecision(const std::pair<double, double> &in)
{
   auto out = in;
   if (!std::isinf(out.second)) {
      auto tmp = out.second;
      out.second = round_to_digits(out.second, 2);
      int expo = (out.second == 0) ? 0 : (int)std::floor(std::log10(std::abs(out.second)));
      if (TString::Format("%e", out.second)(0) != '1') {
         out.second = round_to_digits(tmp, 1);
         out.first = (expo >= 0) ? round(out.first) : round_to_decimal(out.first, -expo);
      } else if (out.second != 0) {
         out.first = (expo >= 0) ? round(out.first) : round_to_decimal(out.first, -expo + 1);
      }
   }
   return out;
}

END_XROOFIT_NAMESPACE
