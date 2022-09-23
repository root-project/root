/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFitHS3/JSONInterface.h>

#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooBinWidthFunction.h>
#include <RooProdPdf.h>
#include <RooPoisson.h>
#include <RooGaussian.h>
#include <RooProduct.h>
#include <RooWorkspace.h>

#include <TH1.h>

#include <stack>

#include "static_execute.h"

using RooFit::Experimental::JSONNode;

namespace {
inline void collectNames(const JSONNode &n, std::vector<std::string> &names)
{
   for (const auto &c : n.children()) {
      names.push_back(RooJSONFactoryWSTool::name(c));
   }
}

inline void stackError(const JSONNode &n, std::vector<double> &sumW, std::vector<double> &sumW2)
{
   if (!n.is_map())
      return;
   if (!n.has_child("counts"))
      throw "no counts given";
   if (!n["counts"].is_seq())
      throw "counts are not in list form";
   if (!n.has_child("errors"))
      throw "no errors given";
   if (!n["errors"].is_seq())
      throw "errors are not in list form";
   if (n["counts"].num_children() != n["errors"].num_children()) {
      throw "inconsistent bin numbers";
   }
   const size_t nbins = n["counts"].num_children();
   for (size_t ibin = 0; ibin < nbins; ++ibin) {
      double w = n["counts"][ibin].val_float();
      double e = n["errors"][ibin].val_float();
      if (ibin < sumW.size())
         sumW[ibin] += w;
      else
         sumW.push_back(w);
      if (ibin < sumW2.size())
         sumW2[ibin] += e * e;
      else
         sumW2.push_back(e * e);
   }
}

std::vector<std::string> getVarnames(const RooHistFunc *hf)
{
   const RooDataHist &dh = hf->dataHist();
   RooArgList vars(*dh.get());
   return RooJSONFactoryWSTool::names(&vars);
}

std::unique_ptr<TH1> histFunc2TH1(const RooHistFunc *hf)
{
   if (!hf)
      RooJSONFactoryWSTool::error("null pointer passed to histFunc2TH1");
   const RooDataHist &dh = hf->dataHist();
   RooArgSet *vars = hf->getVariables();
   auto varnames = RooJSONFactoryWSTool::names(vars);
   std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(vars).c_str())};
   hist->SetDirectory(nullptr);
   auto volumes = dh.binVolumes(0, dh.numEntries());
   for (size_t i = 0; i < volumes.size(); ++i) {
      hist->SetBinContent(i + 1, hist->GetBinContent(i + 1) / volumes[i]);
      hist->SetBinError(i + 1, sqrt(hist->GetBinContent(i + 1)));
   }
   return hist;
}

template <class T>
T *findClient(RooAbsArg *gamma)
{
   for (const auto &client : gamma->clients()) {
      if (client->InheritsFrom(T::Class())) {
         return static_cast<T *>(client);
      } else {
         T *c = findClient<T>(client);
         if (c)
            return c;
      }
   }
   return nullptr;
}

RooRealVar *getNP(RooJSONFactoryWSTool *tool, const char *parname)
{
   RooRealVar *par = tool->workspace()->var(parname);
   if (!tool->workspace()->var(parname)) {
      par = (RooRealVar *)tool->workspace()->factory(std::string(parname) + "[0.,-5,5]");
   }
   if (par) {
      par->setAttribute("np");
   }
   std::string globname = std::string("nom_") + parname;
   RooRealVar *nom = tool->workspace()->var(globname);
   if (!nom) {
      nom = (RooRealVar *)tool->workspace()->factory(globname + "[0.]");
   }
   if (nom) {
      nom->setAttribute("glob");
      nom->setRange(-5, 5);
      nom->setConstant(true);
   }
   std::string constrname = std::string("sigma_") + parname;
   RooRealVar *sigma = tool->workspace()->var(constrname);
   if (!sigma) {
      sigma = (RooRealVar *)tool->workspace()->factory(constrname + "[1.]");
   }
   if (sigma) {
      sigma->setRange(sigma->getVal(), sigma->getVal());
      sigma->setConstant(true);
   }
   if (!par)
      RooJSONFactoryWSTool::error(TString::Format("unable to find nuisance parameter '%s'", parname));
   return par;
}
RooAbsPdf *getConstraint(RooJSONFactoryWSTool *tool, const std::string &sysname)
{
   RooAbsPdf *pdf = tool->workspace()->pdf((sysname + "_constraint").c_str());
   if (!pdf) {
      pdf = (RooAbsPdf *)(tool->workspace()->factory(
         TString::Format("RooGaussian::%s_constraint(alpha_%s,nom_alpha_%s,sigma_alpha_%s)", sysname.c_str(),
                         sysname.c_str(), sysname.c_str(), sysname.c_str())
            .Data()));
   }
   if (!pdf) {
      RooJSONFactoryWSTool::error(TString::Format("unable to find constraint term '%s'", sysname.c_str()));
   }
   return pdf;
}

class RooHistogramFactory : public RooJSONFactoryWSTool::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
      if (prefix.size() > 0)
         name = prefix + name;
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      try {
         std::stack<std::unique_ptr<RooAbsArg>> ownedArgsStack;
         RooArgSet shapeElems;
         RooArgSet normElems;
         RooArgSet varlist;
         tool->getObservables(p["data"], prefix, varlist);

         auto getBinnedData = [&tool, &p, &varlist](std::string const &binnedDataName) -> RooDataHist & {
            auto *dh = dynamic_cast<RooDataHist *>(tool->workspace()->embeddedData(binnedDataName));
            if (!dh) {
               auto dhForImport = tool->readBinnedData(p["data"], binnedDataName, varlist);
               tool->workspace()->import(*dhForImport, RooFit::Silence(true), RooFit::Embedded());
               dh = static_cast<RooDataHist *>(tool->workspace()->embeddedData(dhForImport->GetName()));
            }
            return *dh;
         };

         RooDataHist &dh = getBinnedData(name);
         auto hf = std::make_unique<RooHistFunc>(("hist_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(),
                                                 *(dh.get()), dh);
         ownedArgsStack.push(std::make_unique<RooBinWidthFunction>(
            TString::Format("%s_binWidth", (!prefix.empty() ? prefix : name).c_str()).Data(),
            TString::Format("%s_binWidth", (!prefix.empty() ? prefix : name).c_str()).Data(), *hf, true));
         shapeElems.add(*ownedArgsStack.top());

         if (p.has_child("statError") && p["statError"].val_bool()) {
            RooAbsArg *phf = tool->getScopeObject("mcstat");
            if (phf) {
               shapeElems.add(*phf);
            } else {
               RooJSONFactoryWSTool::error("function '" + name +
                                           "' has 'statError' active, but no element called 'mcstat' in scope!");
            }
         }

         if (p.has_child("normFactors")) {
            for (const auto &nf : p["normFactors"].children()) {
               std::string nfname(RooJSONFactoryWSTool::name(nf));
               RooAbsReal *r = tool->workspace()->var(nfname);
               if (r) {
                  normElems.add(*r);
               } else {
                  normElems.add(*static_cast<RooRealVar *>(tool->workspace()->factory(nfname + "[1.]")));
               }
            }
         }

         if (p.has_child("overallSystematics")) {
            RooArgList nps;
            std::vector<double> low;
            std::vector<double> high;
            for (const auto &sys : p["overallSystematics"].children()) {
               std::string sysname(RooJSONFactoryWSTool::name(sys));
               std::string parname(sys.has_child("parameter") ? RooJSONFactoryWSTool::name(sys["parameter"])
                                                              : "alpha_" + sysname);
               RooRealVar *par = ::getNP(tool, parname.c_str());
               if (par) {
                  nps.add(*par);
                  low.push_back(sys["low"].val_float());
                  high.push_back(sys["high"].val_float());
               } else {
                  RooJSONFactoryWSTool::error("overall systematic '" + sysname + "' doesn't have a valid parameter!");
               }
            }
            auto v = std::make_unique<RooStats::HistFactory::FlexibleInterpVar>(
               ("overallSys_" + name).c_str(), ("overallSys_" + name).c_str(), nps, 1., low, high);
            v->setAllInterpCodes(4); // default HistFactory interpCode
            normElems.add(*v);
            ownedArgsStack.push(std::move(v));
         }

         if (p.has_child("histogramSystematics")) {
            RooArgList nps;
            RooArgList low;
            RooArgList high;
            for (const auto &sys : p["histogramSystematics"].children()) {
               std::string sysname(RooJSONFactoryWSTool::name(sys));
               std::string parname(sys.has_child("parameter") ? RooJSONFactoryWSTool::name(sys["parameter"])
                                                              : "alpha_" + sysname);
               RooAbsReal *par = ::getNP(tool, parname.c_str());
               nps.add(*par);
               RooDataHist &dh_low = getBinnedData(sysname + "Low_" + name);
               ownedArgsStack.push(std::make_unique<RooHistFunc>(
                  (sysname + "Low_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(), *(dh_low.get()), dh_low));
               low.add(*ownedArgsStack.top());
               RooDataHist &dh_high = getBinnedData(sysname + "High_" + name);
               ownedArgsStack.push(std::make_unique<RooHistFunc>((sysname + "High_" + name).c_str(),
                                                                 RooJSONFactoryWSTool::name(p).c_str(),
                                                                 *(dh_high.get()), dh_high));
               high.add(*ownedArgsStack.top());
            }
            auto v = std::make_unique<PiecewiseInterpolation>(("histoSys_" + name).c_str(),
                                                              ("histoSys_" + name).c_str(), *hf, low, high, nps, false);
            v->setAllInterpCodes(4); // default interpCode for HistFactory
            shapeElems.add(*v);
            ownedArgsStack.push(std::move(v));
         } else {
            shapeElems.add(*hf);
            ownedArgsStack.push(std::move(hf));
         }

         if (p.has_child("shapeSystematics")) {
            for (const auto &sys : p["shapeSystematics"].children()) {
               std::string sysname(RooJSONFactoryWSTool::name(sys));
               std::string funcName = prefix + sysname + "_ShapeSys";
               RooAbsArg *phf = tool->getScopeObject(funcName);
               if (!phf) {
                  RooJSONFactoryWSTool::error("PHF '" + funcName +
                                              "' should have been created but cannot be found in scope.");
               }
               shapeElems.add(*phf);
            }
         }

         RooProduct shape(name.c_str(), (name + "_shape").c_str(), shapeElems);
         tool->workspace()->import(shape, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         if (normElems.size() > 0) {
            RooProduct norm((name + "_norm").c_str(), (name + "_norm").c_str(), normElems);
            tool->workspace()->import(norm, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         } else {
            tool->workspace()->factory("RooConstVar::" + name + "_norm(1.)");
         }
      } catch (const std::runtime_error &e) {
         RooJSONFactoryWSTool::error("function '" + name +
                                     "' is of histogram type, but 'data' is not a valid definition. " + e.what() + ".");
      }
      return true;
   }
};

class RooRealSumPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   std::unique_ptr<ParamHistFunc> createPHF(const std::string &sysname, const std::string &phfname,
                                            const std::vector<double> &vals, RooWorkspace &w, RooArgList &constraints,
                                            const RooArgSet &observables, const std::string &constraintType,
                                            RooArgList &gammas, double gamma_min, double gamma_max) const
   {
      RooArgList ownedComponents;

      std::string funcParams = "gamma_" + sysname;
      gammas.add(ParamHistFunc::createParamSet(w, funcParams.c_str(), observables, gamma_min, gamma_max));
      auto phf = std::make_unique<ParamHistFunc>(phfname.c_str(), phfname.c_str(), observables, gammas);
      for (auto &g : gammas) {
         g->setAttribute("np");
      }

      if (constraintType == "Gauss") {
         for (size_t i = 0; i < vals.size(); ++i) {
            TString nomname = TString::Format("nom_%s", gammas[i].GetName());
            TString poisname = TString::Format("%s_constraint", gammas[i].GetName());
            TString sname = TString::Format("%s_sigma", gammas[i].GetName());
            auto nom = std::make_unique<RooRealVar>(nomname.Data(), nomname.Data(), 1);
            nom->setAttribute("glob");
            nom->setConstant(true);
            nom->setRange(0, std::max(10., gamma_max));
            auto sigma = std::make_unique<RooConstVar>(sname.Data(), sname.Data(), vals[i]);
            auto g = static_cast<RooRealVar *>(gammas.at(i));
            auto gaus = std::make_unique<RooGaussian>(poisname.Data(), poisname.Data(), *nom, *g, *sigma);
            gaus->addOwnedComponents(std::move(nom), std::move(sigma));
            constraints.add(*gaus, true);
            ownedComponents.addOwned(std::move(gaus), true);
         }
      } else if (constraintType == "Poisson") {
         for (size_t i = 0; i < vals.size(); ++i) {
            double tau_float = vals[i];
            TString tname = TString::Format("%s_tau", gammas[i].GetName());
            TString nomname = TString::Format("nom_%s", gammas[i].GetName());
            TString prodname = TString::Format("%s_poisMean", gammas[i].GetName());
            TString poisname = TString::Format("%s_constraint", gammas[i].GetName());
            auto tau = std::make_unique<RooConstVar>(tname.Data(), tname.Data(), tau_float);
            auto nom = std::make_unique<RooRealVar>(nomname.Data(), nomname.Data(), tau_float);
            nom->setAttribute("glob");
            nom->setConstant(true);
            nom->setMin(0);
            RooArgSet elems{gammas[i], *tau};
            auto prod = std::make_unique<RooProduct>(prodname.Data(), prodname.Data(), elems);
            auto pois = std::make_unique<RooPoisson>(poisname.Data(), poisname.Data(), *nom, *prod);
            pois->addOwnedComponents(std::move(tau), std::move(nom), std::move(prod));
            pois->setNoRounding(true);
            constraints.add(*pois, true);
            ownedComponents.addOwned(std::move(pois), true);
         }
      } else {
         RooJSONFactoryWSTool::error("unknown constraint type " + constraintType);
      }
      for (auto &g : gammas) {
         for (auto client : g->clients()) {
            if (client->InheritsFrom(RooAbsPdf::Class()) && !constraints.find(*client)) {
               constraints.add(*client);
            }
         }
      }
      phf->recursiveRedirectServers(observables);
      // Transfer ownership of gammas and owned constraints to the ParamHistFunc
      phf->addOwnedComponents(std::move(ownedComponents));

      return phf;
   }

   std::unique_ptr<ParamHistFunc> createPHFMCStat(const std::string &name, const std::vector<double> &sumW,
                                                  const std::vector<double> &sumW2, RooWorkspace &w,
                                                  RooArgList &constraints, const RooArgSet &observables,
                                                  double statErrorThreshold, const std::string &statErrorType) const
   {
      if (sumW.empty())
         return nullptr;

      RooArgList gammas;
      std::string phfname = std::string("mc_stat_") + name;
      std::string sysname = std::string("stat_") + name;
      std::vector<double> vals(sumW.size());
      std::vector<double> errs(sumW.size());

      for (size_t i = 0; i < sumW.size(); ++i) {
         errs[i] = sqrt(sumW2[i]) / sumW[i];
         if (statErrorType == "Gauss") {
            vals[i] = std::max(errs[i], 0.); // avoid negative sigma. This NP will be set constant anyway later
         } else if (statErrorType == "Poisson") {
            vals[i] = sumW[i] * sumW[i] / sumW2[i];
         }
      }

      auto phf = createPHF(sysname, phfname, vals, w, constraints, observables, statErrorType, gammas, 0, 10);

      // set constant NPs which are below the MC stat threshold, and remove them from the np list
      for (size_t i = 0; i < sumW.size(); ++i) {
         auto g = static_cast<RooRealVar *>(gammas.at(i));
         g->setError(errs[i]);
         if (errs[i] < statErrorThreshold) {
            g->setConstant(true); // all negative errs are set constant
         }
      }

      return phf;
   }

   std::unique_ptr<ParamHistFunc> createPHFShapeSys(const JSONNode &p, const std::string &phfname, RooWorkspace &w,
                                                    RooArgList &constraints, const RooArgSet &observables) const
   {
      std::string sysname(RooJSONFactoryWSTool::name(p));
      std::vector<double> vals;
      for (const auto &v : p["vals"].children()) {
         vals.push_back(v.val_float());
      }
      RooArgList gammas;
      return createPHF(sysname, phfname, vals, w, constraints, observables, p["constraint"].val(), gammas, 0, 1000);
   }

   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooArgList funcs;
      RooArgList coefs;
      RooArgList constraints;
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
      }
      std::vector<std::string> usesStatError;
      double statErrorThreshold = 0;
      std::string statErrorType = "Poisson";
      if (p.has_child("statError")) {
         auto &staterr = p["statError"];
         if (staterr.has_child("relThreshold"))
            statErrorThreshold = staterr["relThreshold"].val_float();
         if (staterr.has_child("constraint"))
            statErrorType = staterr["constraint"].val();
      }
      std::vector<double> sumW;
      std::vector<double> sumW2;
      std::vector<double> dummy;
      std::vector<std::string> sysnames;
      std::vector<std::string> funcnames;
      std::vector<std::string> coefnames;
      RooArgSet observables;
      if (p.has_child("observables")) {
         tool->getObservables(p, name, observables);
         tool->setScopeObservables(observables);
      }
      for (const auto &comp : p["samples"].children()) {
         std::string fname(RooJSONFactoryWSTool::name(comp));
         auto &def = comp.is_container() ? comp : p["functions"][fname.c_str()];
         std::string fprefix = RooJSONFactoryWSTool::genPrefix(def, true);
         if (def["type"].val() == "hist-sample") {
            try {
               if (observables.empty()) {
                  tool->getObservables(comp["data"], fprefix, observables);
               }
               if (def.has_child("overallSystematics"))
                  ::collectNames(def["overallSystematics"], sysnames);
               if (def.has_child("histogramSystematics"))
                  ::collectNames(def["histogramSystematics"], sysnames);
               if (def.has_child("shapeSystematics")) { // ShapeSys are special case. Create PHFs here if needed
                  std::vector<std::string> shapeSysNames;
                  ::collectNames(def["shapeSystematics"], shapeSysNames);
                  for (auto &sysname : shapeSysNames) {
                     std::string phfname = name + "_" + sysname + "_ShapeSys";
                     auto phf = tool->getScopeObject(phfname);
                     if (!phf) {
                        auto newphf = createPHFShapeSys(def["shapeSystematics"][sysname], phfname, *(tool->workspace()),
                                                        constraints, observables);
                        tool->workspace()->import(*newphf, RooFit::RecycleConflictNodes(), RooFit::Silence(true));
                        tool->setScopeObject(phfname, tool->workspace()->function(phfname.c_str()));
                     }
                  }
               }
            } catch (const char *s) {
               RooJSONFactoryWSTool::error("function '" + name + "' unable to collect observables from function " +
                                           fname + ". " + s);
            }
            try {
               if (comp["statError"].val_bool()) {
                  ::stackError(def["data"], sumW, sumW2);
               }
            } catch (const char *s) {
               RooJSONFactoryWSTool::error("function '" + name + "' unable to sum statError from function " + fname +
                                           ". " + s);
            }
         }
         funcnames.push_back(fprefix + fname);
         coefnames.push_back(fprefix + fname + "_norm");
      }

      auto phf = createPHFMCStat(name, sumW, sumW2, *(tool->workspace()), constraints, observables, statErrorThreshold,
                                 statErrorType);
      if (phf) {
         tool->workspace()->import(*phf, RooFit::RecycleConflictNodes(), RooFit::Silence(true));
         tool->setScopeObject("mcstat", tool->workspace()->function(phf->GetName()));
      }

      tool->importFunctions(p["samples"]);
      for (const auto &fname : funcnames) {
         RooAbsReal *func = tool->request<RooAbsReal>(fname.c_str(), name);
         funcs.add(*func);
      }
      for (const auto &coefname : coefnames) {
         RooAbsReal *coef = tool->request<RooAbsReal>(coefname.c_str(), name);
         coefs.add(*coef);
      }
      for (auto sysname : sysnames) {
         RooAbsPdf *pdf = ::getConstraint(tool, sysname.c_str());
         constraints.add(*pdf);
      }
      if (constraints.empty()) {
         RooRealSumPdf sum(name.c_str(), name.c_str(), funcs, coefs, true);
         sum.setAttribute("BinnedLikelihood");
         tool->workspace()->import(sum, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      } else {
         RooRealSumPdf sum((name + "_model").c_str(), name.c_str(), funcs, coefs, true);
         sum.setAttribute("BinnedLikelihood");
         tool->workspace()->import(sum, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         RooArgList lhelems;
         lhelems.add(sum);
         RooProdPdf prod(name.c_str(), name.c_str(), RooArgSet(constraints), RooFit::Conditional(lhelems, observables));
         tool->workspace()->import(prod, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      }

      tool->clearScope();

      return true;
   }
};

} // namespace

namespace {
class FlexibleInterpVarStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "interpolation0d";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooStats::HistFactory::FlexibleInterpVar *fip =
         static_cast<const RooStats::HistFactory::FlexibleInterpVar *>(func);
      elem["type"] << key();
      auto &vars = elem["vars"];
      vars.set_seq();
      for (const auto &v : fip->variables()) {
         vars.append_child() << v->GetName();
      }
      elem["nom"] << fip->nominal();
      elem["high"] << fip->high();
      elem["low"] << fip->low();
      return true;
   }
};

class PiecewiseInterpolationStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "interpolation";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const PiecewiseInterpolation *pip = static_cast<const PiecewiseInterpolation *>(func);
      elem["type"] << key();
      elem["interpolationCodes"] << pip->interpolationCodes();
      auto &vars = elem["vars"];
      vars.set_seq();
      for (const auto &v : pip->paramList()) {
         vars.append_child() << v->GetName();
      }

      auto &nom = elem["nom"];
      nom << pip->nominalHist()->GetName();

      auto &high = elem["high"];
      high.set_seq();
      for (const auto &v : pip->highList()) {
         high.append_child() << v->GetName();
      }

      auto &low = elem["low"];
      low.set_seq();
      for (const auto &v : pip->lowList()) {
         low.append_child() << v->GetName();
      }
      return true;
   }
};
} // namespace

namespace {
class PiecewiseInterpolationFactory : public RooJSONFactoryWSTool::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("vars")) {
         RooJSONFactoryWSTool::error("no vars of '" + name + "'");
      }
      if (!p.has_child("high")) {
         RooJSONFactoryWSTool::error("no high variations of '" + name + "'");
      }
      if (!p.has_child("low")) {
         RooJSONFactoryWSTool::error("no low variations of '" + name + "'");
      }
      if (!p.has_child("nom")) {
         RooJSONFactoryWSTool::error("no nominal variation of '" + name + "'");
      }

      std::string nomname(p["nom"].val());
      RooAbsReal *nominal = tool->request<RooAbsReal>(nomname, name);

      RooArgList vars;
      for (const auto &d : p["vars"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooRealVar *obj = tool->request<RooRealVar>(objname, name);
         vars.add(*obj);
      }

      RooArgList high;
      for (const auto &d : p["high"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooAbsReal *obj = tool->request<RooAbsReal>(objname, name);
         high.add(*obj);
      }

      RooArgList low;
      for (const auto &d : p["low"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooAbsReal *obj = tool->request<RooAbsReal>(objname, name);
         low.add(*obj);
      }

      PiecewiseInterpolation pip(name.c_str(), name.c_str(), *nominal, low, high, vars);

      if (p.has_child("interpolationCodes")) {
         for (size_t i = 0; i < vars.size(); ++i) {
            pip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), p["interpolationCodes"][i].val_int(), true);
         }
      }

      tool->workspace()->import(pip, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};
} // namespace

namespace {
class HistFactoryStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   void collectElements(RooArgSet &elems, RooProduct *prod) const
   {
      for (const auto &e : prod->components()) {
         if (e->InheritsFrom(RooProduct::Class())) {
            collectElements(elems, (RooProduct *)e);
         } else {
            elems.add(*e);
         }
      }
   }
   bool tryExport(const RooProdPdf *prodpdf, JSONNode &elem) const
   {
      std::string chname(prodpdf->GetName());
      if (chname.find("model_") == 0) {
         chname = chname.substr(6);
      }
      RooRealSumPdf *sumpdf = nullptr;
      for (const auto &v : prodpdf->pdfList()) {
         if (v->InheritsFrom(RooRealSumPdf::Class())) {
            sumpdf = static_cast<RooRealSumPdf *>(v);
         }
      }
      if (!sumpdf)
         return false;
      for (const auto &sample : sumpdf->funcList()) {
         if (!sample->InheritsFrom(RooProduct::Class()) && !sample->InheritsFrom(RooRealSumPdf::Class()))
            return false;
      }

      bool has_poisson_constraints = false;
      bool has_gauss_constraints = false;
      std::map<int, double> tot_yield;
      std::map<int, double> tot_yield2;
      std::map<int, double> rel_errors;
      std::map<std::string, std::unique_ptr<TH1>> bb_histograms;
      std::map<std::string, std::unique_ptr<TH1>> nonbb_histograms;
      std::vector<std::string> varnames;

      for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
         const auto func = sumpdf->funcList().at(sampleidx);
         const auto coef = sumpdf->coefList().at(sampleidx);
         std::string samplename = func->GetName();
         if (samplename.find("L_x_") == 0)
            samplename = samplename.substr(4);
         auto end = samplename.find("_" + chname);
         if (end < samplename.size())
            samplename = samplename.substr(0, end);

         RooArgSet elems;
         if (func->InheritsFrom(RooProduct::Class())) {
            collectElements(elems, (RooProduct *)func);
         } else {
            elems.add(*func);
         }
         if (coef->InheritsFrom(RooProduct::Class())) {
            collectElements(elems, (RooProduct *)coef);
         } else {
            elems.add(*coef);
         }
         std::unique_ptr<TH1> hist;
         std::vector<ParamHistFunc *> phfs;
         PiecewiseInterpolation *pip = nullptr;
         std::vector<const RooAbsArg *> norms;

         RooStats::HistFactory::FlexibleInterpVar *fip = nullptr;
         for (const auto &e : elems) {
            if (e->InheritsFrom(RooConstVar::Class())) {
               if (((RooConstVar *)e)->getVal() == 1.)
                  continue;
               norms.push_back(e);
            } else if (e->InheritsFrom(RooRealVar::Class())) {
               norms.push_back(e);
            } else if (e->InheritsFrom(RooHistFunc::Class())) {
               const RooHistFunc *hf = static_cast<const RooHistFunc *>(e);
               if (varnames.empty()) {
                  varnames = getVarnames(hf);
               }
               if (!hist) {
                  hist = histFunc2TH1(hf);
               }
            } else if (e->InheritsFrom(RooStats::HistFactory::FlexibleInterpVar::Class())) {
               fip = static_cast<RooStats::HistFactory::FlexibleInterpVar *>(e);
            } else if (e->InheritsFrom(PiecewiseInterpolation::Class())) {
               pip = static_cast<PiecewiseInterpolation *>(e);
            } else if (e->InheritsFrom(ParamHistFunc::Class())) {
               phfs.push_back((ParamHistFunc *)e);
            }
         }
         if (pip) {
            if (!hist && pip->nominalHist()->InheritsFrom(RooHistFunc::Class())) {
               hist = histFunc2TH1(static_cast<const RooHistFunc *>(pip->nominalHist()));
            }
            if (varnames.empty() && pip->nominalHist()->InheritsFrom(RooHistFunc::Class())) {
               varnames = getVarnames(dynamic_cast<const RooHistFunc *>(pip->nominalHist()));
            }
         }
         if (!hist) {
            return false;
         }

         elem["name"] << chname;
         elem["type"] << key();

         auto &samples = elem["samples"];
         samples.set_map();
         auto &s = samples[samplename];
         s.set_map();
         s["type"] << "hist-sample";

         for (const auto &norm : norms) {
            auto &nfs = s["normFactors"];
            nfs.set_seq();
            nfs.append_child() << norm->GetName();
         }

         if (pip) {
            auto &systs = s["histogramSystematics"];
            systs.set_map();
            for (size_t i = 0; i < pip->paramList().size(); ++i) {
               std::string sysname(pip->paramList().at(i)->GetName());
               if (sysname.find("alpha_") == 0) {
                  sysname = sysname.substr(6);
               }
               auto &sys = systs[sysname];
               sys.set_map();
               auto &dataLow = sys["dataLow"];
               if (pip->lowList().at(i)->InheritsFrom(RooHistFunc::Class())) {
                  auto histLow = histFunc2TH1(static_cast<RooHistFunc *>(pip->lowList().at(i)));
                  RooJSONFactoryWSTool::exportHistogram(*histLow, dataLow, varnames, 0, false, false);
               }
               auto &dataHigh = sys["dataHigh"];
               if (pip->highList().at(i)->InheritsFrom(RooHistFunc::Class())) {
                  auto histHigh = histFunc2TH1(static_cast<RooHistFunc *>(pip->highList().at(i)));
                  RooJSONFactoryWSTool::exportHistogram(*histHigh, dataHigh, varnames, 0, false, false);
               }
            }
         }

         if (fip) {
            auto &systs = s["overallSystematics"];
            systs.set_map();
            for (size_t i = 0; i < fip->variables().size(); ++i) {
               std::string sysname(fip->variables().at(i)->GetName());
               if (sysname.find("alpha_") == 0) {
                  sysname = sysname.substr(6);
               }
               auto &sys = systs[sysname];
               sys.set_map();
               sys["low"] << fip->low()[i];
               sys["high"] << fip->high()[i];
            }
         }
         bool has_mc_stat = false;
         for (auto phf : phfs) {
            if (TString(phf->GetName()).BeginsWith("mc_stat_")) { // MC stat uncertainty
               has_mc_stat = true;
               s["statError"] << 1;
               int idx = 0;
               for (const auto &g : phf->paramList()) {
                  ++idx;
                  RooPoisson *constraint_p = findClient<RooPoisson>(g);
                  RooGaussian *constraint_g = findClient<RooGaussian>(g);
                  if (tot_yield.find(idx) == tot_yield.end()) {
                     tot_yield[idx] = 0;
                     tot_yield2[idx] = 0;
                  }
                  tot_yield[idx] += hist->GetBinContent(idx);
                  tot_yield2[idx] += (hist->GetBinContent(idx) * hist->GetBinContent(idx));
                  if (constraint_p) {
                     double erel = 1. / std::sqrt(constraint_p->getX().getVal());
                     rel_errors[idx] = erel;
                     has_poisson_constraints = true;
                  } else if (constraint_g) {
                     double erel = constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                     rel_errors[idx] = erel;
                     has_gauss_constraints = true;
                  }
               }
               bb_histograms[samplename] = std::move(hist);
            } else { // other ShapeSys
               auto &shapesysts = s["shapeSystematics"];
               shapesysts.set_map();
               // Getting the name of the syst is tricky.
               TString sysName(phf->GetName());
               sysName.Remove(sysName.Index("_ShapeSys"));
               sysName.Remove(0, chname.size() + 1);
               auto &sys = shapesysts[sysName.Data()];
               sys.set_map();
               auto &cstrts = sys["vals"];
               cstrts.set_seq();
               bool is_poisson = false;
               for (const auto &g : phf->paramList()) {
                  RooPoisson *constraint_p = findClient<RooPoisson>(g);
                  RooGaussian *constraint_g = findClient<RooGaussian>(g);
                  if (constraint_p) {
                     is_poisson = true;
                     cstrts.append_child() << constraint_p->getX().getVal();
                  } else if (constraint_g) {
                     is_poisson = false;
                     cstrts.append_child() << constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                  }
               }
               if (is_poisson) {
                  sys["constraint"] << "Poisson";
               } else {
                  sys["constraint"] << "Gauss";
               }
            }
         }
         if (!has_mc_stat) {
            nonbb_histograms[samplename] = std::move(hist);
            s["statError"] << 0;
         }
         auto &ns = s["namespaces"];
         ns.set_seq();
         ns.append_child() << chname;
      }

      auto &samples = elem["samples"];
      for (const auto &hist : nonbb_histograms) {
         auto &s = samples[hist.first];
         auto &data = s["data"];
         RooJSONFactoryWSTool::writeObservables(*hist.second, elem, varnames);
         RooJSONFactoryWSTool::exportHistogram(*hist.second, data, varnames, 0, false, false);
      }
      for (const auto &hist : bb_histograms) {
         auto &s = samples[hist.first];
         for (auto bin : rel_errors) {
            // reverse engineering the correct partial error
            // the (arbitrary) convention used here is that all samples should have the same relative error
            const int i = bin.first;
            const double relerr_tot = bin.second;
            const double count = hist.second->GetBinContent(i);
            hist.second->SetBinError(i, relerr_tot * tot_yield[i] / sqrt(tot_yield2[i]) * count);
         }
         auto &data = s["data"];
         RooJSONFactoryWSTool::writeObservables(*hist.second, elem, varnames);
         RooJSONFactoryWSTool::exportHistogram(*hist.second, data, varnames, 0, false, true);
      }
      auto &statError = elem["statError"];
      statError.set_map();
      if (has_poisson_constraints) {
         statError["constraint"] << "Poisson";
      } else if (has_gauss_constraints) {
         statError["constraint"] << "Gauss";
      }
      return true;
   }

   std::string const &key() const override
   {
      static const std::string keystring = "histfactory";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *p, JSONNode &elem) const override
   {
      const RooProdPdf *prodpdf = static_cast<const RooProdPdf *>(p);
      if (tryExport(prodpdf, elem)) {
         return true;
      }
      return false;
   }
};

STATIC_EXECUTE(

   using Tool = RooJSONFactoryWSTool;

   Tool::registerImporter<RooRealSumPdfFactory>("histfactory", true);
   Tool::registerImporter<RooHistogramFactory>("hist-sample", true);
   Tool::registerImporter<PiecewiseInterpolationFactory>("interpolation", true);
   Tool::registerExporter<FlexibleInterpVarStreamer>(RooStats::HistFactory::FlexibleInterpVar::Class(), true);
   Tool::registerExporter<PiecewiseInterpolationStreamer>(PiecewiseInterpolation::Class(), true);
   Tool::registerExporter<HistFactoryStreamer>(RooProdPdf::Class(), true);

)

} // namespace
