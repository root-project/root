#include <RooFitHS3/RooJSONFactoryWSTool.h>

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
#include <RooProduct.h>
#include <RooWorkspace.h>

#include <TH1.h>

#include "JSONInterface.h"
#include "static_execute.h"

using RooFit::Detail::JSONNode;

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

TH1 *histFunc2TH1(const RooHistFunc *hf)
{
   const RooDataHist &dh = hf->dataHist();
   RooArgList vars(*dh.get());
   auto varnames = RooJSONFactoryWSTool::names(&vars);
   TH1 *hist = hf->createHistogram(RooJSONFactoryWSTool::concat(&vars).c_str());
   hist->SetDirectory(NULL);
   auto volumes = dh.binVolumes(0, dh.numEntries());
   for (size_t i = 0; i < volumes.size(); ++i) {
      hist->SetBinContent(i + 1, hist->GetBinContent(i + 1) / volumes[i]);
      hist->SetBinError(i + 1, sqrt(hist->GetBinContent(i + 1)));
   }
   return hist;
}

RooPoisson *findPoissonClient(RooAbsArg *gamma)
{
   for (const auto &client : gamma->clients()) {
      if (client->InheritsFrom(RooPoisson::Class())) {
         return (RooPoisson *)client;
      } else {
         RooPoisson *p = findPoissonClient(client);
         if (p)
            return p;
      }
   }
   return NULL;
}

RooRealVar *getNP(RooJSONFactoryWSTool *tool, const char *parname)
{
   RooRealVar *par = tool->workspace()->var(parname);
   if (!tool->workspace()->var(parname)) {
      par = (RooRealVar *)tool->workspace()->factory(TString::Format("%s[0.,-5,5]", parname).Data());
   }
   if (par) {
      par->setAttribute("np");
   }
   TString globname = TString::Format("nom_%s", parname);
   RooRealVar *nom = tool->workspace()->var(globname.Data());
   if (!nom) {
      nom = (RooRealVar *)tool->workspace()->factory((globname + "[0.]").Data());
   }
   if (nom) {
      nom->setAttribute("glob");
      nom->setRange(-5, 5);
      nom->setConstant(true);
   }
   TString constrname = TString::Format("sigma_%s", parname);
   RooRealVar *sigma = tool->workspace()->var(constrname.Data());
   if (!sigma) {
      sigma = (RooRealVar *)tool->workspace()->factory((constrname + "[1.]").Data());
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
   RooAbsPdf *pdf = tool->workspace()->pdf(sysname.c_str());
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
   virtual bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
      if (prefix.size() > 0)
         name = prefix + name;
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      try {
         std::vector<RooAbsArg *> tmp;
         RooArgSet shapeElems;
         RooArgSet normElems;
         auto varlist = tool->getObservables(p["data"], prefix);
         RooDataHist *dh = tool->readBinnedData(p["data"], name, varlist);
         RooHistFunc *hf =
            new RooHistFunc(("hist_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(), *(dh->get()), *dh);
         RooBinWidthFunction *binning =
            new RooBinWidthFunction(TString::Format("%s_binWidth", name.c_str()).Data(),
                                    TString::Format("%s_binWidth", name.c_str()).Data(), *hf, true);
         shapeElems.add(*binning);
         tmp.push_back(binning);

         if (p["statError"].val_bool()) {
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
               RooAbsReal *r = tool->workspace()->var(nfname.c_str());
               if (r) {
                  normElems.add(*r);
               } else {
                  normElems.add(
                     *(RooRealVar *)tool->workspace()->factory(TString::Format("%s[1.]", nfname.c_str()).Data()));
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
            RooStats::HistFactory::FlexibleInterpVar *v = new RooStats::HistFactory::FlexibleInterpVar(
               ("overallSys_" + name).c_str(), ("overallSys_" + name).c_str(), nps, 1., low, high);
            normElems.add(*v);
            tmp.push_back(v);
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
               RooDataHist *dh_low = tool->readBinnedData(sys["dataLow"], sysname + "Low_" + name, varlist);
               RooHistFunc *hf_low = new RooHistFunc((sysname + "Low_" + name).c_str(),
                                                     RooJSONFactoryWSTool::name(p).c_str(), *(dh_low->get()), *dh_low);
               low.add(*hf_low);
               tmp.push_back(hf_low);
               RooDataHist *dh_high = tool->readBinnedData(sys["dataHigh"], sysname + "High_" + name, varlist);
               RooHistFunc *hf_high =
                  new RooHistFunc((sysname + "High_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(),
                                  *(dh_high->get()), *dh_high);
               high.add(*hf_high);
               tmp.push_back(hf_high);
            }
            PiecewiseInterpolation *v = new PiecewiseInterpolation(
               ("histoSys_" + name).c_str(), ("histoSys_" + name).c_str(), *hf, nps, low, high, false);
            shapeElems.add(*v);
            tmp.push_back(v);
         } else {
            shapeElems.add(*hf);
            tmp.push_back(hf);
         }
         RooProduct shape(name.c_str(), (name + "_shape").c_str(), shapeElems);
         tool->workspace()->import(shape, RooFit::RecycleConflictNodes(true));
         if (normElems.size() > 0) {
            RooProduct norm((name + "_norm").c_str(), (name + "_norm").c_str(), normElems);
            tool->workspace()->import(norm, RooFit::RecycleConflictNodes(true));
         } else {
            tool->workspace()->factory(("RooConstVar::" + name + "_norm(1.)").c_str());
         }
         for (auto e : tmp) {
            delete e;
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
   ParamHistFunc *createPHF(const std::string &name, const std::vector<double> sumW, const std::vector<double> &sumW2,
                            RooArgList &nps, RooArgList &constraints, const RooArgSet &observables,
                            double statErrorThreshold) const
   {
      RooArgList gammas;
      for (size_t i = 0; i < sumW.size(); ++i) {
         TString gname = TString::Format("gamma_stat_%s_bin_%d", name.c_str(), (int)i);
         TString tname = TString::Format("tau_stat_%s_bin_%d", name.c_str(), (int)i);
         TString prodname = TString::Format("nExp_stat_%s_bin_%d", name.c_str(), (int)i);
         TString poisname = TString::Format("Constraint_stat_%s_bin_%d", name.c_str(), (int)i);
         double err = sqrt(sumW2[i]) / sumW[i];
         if (err > 0) {
            double tauCV = 1. / (err * err);
            RooRealVar *g = new RooRealVar(gname.Data(), gname.Data(), 1.);
            g->setAttribute("np");
            if (err < statErrorThreshold) {
               g->setConstant(true);
            } else {
               g->setConstant(false);
            }
            RooRealVar *tau = new RooRealVar(tname.Data(), tname.Data(), tauCV);
            //           tau->setAttribute("glob");
            tau->setConstant(true);
            tau->setRange(tau->getVal(), tau->getVal());
            RooArgSet elems;
            elems.add(*g);
            elems.add(*tau);
            g->setError(err);
            g->setMin(1. - 10 * err);
            g->setMax(1. + 10 * err);
            RooProduct *prod = new RooProduct(prodname.Data(), prodname.Data(), elems);
            RooPoisson *pois = new RooPoisson(poisname.Data(), poisname.Data(), *tau, *prod);
            pois->setNoRounding(true);
            gammas.add(*g, true);
            nps.add(*g);
            constraints.add(*pois, true);
         } else {
            RooRealVar *g = new RooRealVar(gname.Data(), gname.Data(), 1.);
            g->setConstant(true);
            gammas.add(*g, true);
         }
      }
      if (gammas.size() > 0) {
         ParamHistFunc *phf =
            new ParamHistFunc(TString::Format("%s_mcstat", name.c_str()), "staterror", observables, gammas);
         phf->recursiveRedirectServers(observables);
         return phf;
      } else {
         return NULL;
      }
   }

   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooArgList funcs;
      RooArgList coefs;
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
      }
      RooArgList constraints;
      RooArgList nps;
      std::vector<std::string> usesStatError;
      double statErrorThreshold = 0;
      if (p.has_child("statError")) {
         auto &staterr = p["statError"];
         if (staterr.has_child("relThreshold"))
            statErrorThreshold = staterr["relThreshold"].val_float();
      }
      std::vector<double> sumW;
      std::vector<double> sumW2;
      std::vector<double> dummy;
      std::vector<std::string> sysnames;
      std::vector<std::string> funcnames;
      std::vector<std::string> coefnames;
      RooArgSet observables;
      if (p.has_child("observables")) {
         observables.add(tool->getObservables(p, name));
         tool->setScopeObservables(observables);
      }
      for (const auto &comp : p["samples"].children()) {
         std::string fname(RooJSONFactoryWSTool::name(comp));
         auto &def = comp.is_container() ? comp : p["functions"][fname.c_str()];
         std::string fprefix = RooJSONFactoryWSTool::genPrefix(def, true);
         if (def["type"].val() == "histogram") {
            try {
               if (observables.size() == 0) {
                  observables.add(tool->getObservables(comp["data"], fprefix));
               }
               if (def.has_child("overallSystematics"))
                  ::collectNames(def["overallSystematics"], sysnames);
               if (def.has_child("histogramSystematics"))
                  ::collectNames(def["histogramSystematics"], sysnames);
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

      ParamHistFunc *phf = createPHF(name, sumW, sumW2, nps, constraints, observables, statErrorThreshold);
      if (phf) {
         tool->workspace()->import(*phf, RooFit::RecycleConflictNodes());
         tool->setScopeObject("mcstat", tool->workspace()->function(phf->GetName()));
         delete phf;
      }

      tool->importFunctions(p["samples"]);
      for (const auto &fname : funcnames) {
         RooAbsReal *func = tool->workspace()->function(fname.c_str());
         if (!func) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + fname + "' of '" + name + "'");
         }
         funcs.add(*func);
      }
      for (const auto &coefname : coefnames) {
         RooAbsReal *coef = tool->workspace()->function(coefname.c_str());
         if (!coef) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + coefname + "' of '" + name + "'");
         }
         coefs.add(*coef);
      }

      for (auto &np : nps) {
         for (auto client : np->clients()) {
            if (client->InheritsFrom(RooAbsPdf::Class()) && !constraints.find(*client)) {
               constraints.add(*client);
            }
         }
      }
      for (auto sysname : sysnames) {
         RooAbsPdf *pdf = ::getConstraint(tool, sysname.c_str());
         constraints.add(*pdf);
      }
      if (constraints.getSize() == 0) {
         RooRealSumPdf sum(name.c_str(), name.c_str(), funcs, coefs, true);
         tool->workspace()->import(sum, RooFit::RecycleConflictNodes(true));
      } else {
         RooRealSumPdf sum((name + "_model").c_str(), name.c_str(), funcs, coefs, true);
         tool->workspace()->import(sum, RooFit::RecycleConflictNodes(true));
         RooArgList lhelems;
         lhelems.add(sum);
         RooProdPdf prod(name.c_str(), name.c_str(), constraints, RooFit::Conditional(lhelems, observables));
         tool->workspace()->import(prod, RooFit::RecycleConflictNodes(true));
      }

      tool->clearScope();

      return true;
   }
};

} // namespace

namespace {
class FlexibleInterpVarStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooStats::HistFactory::FlexibleInterpVar *fip =
         static_cast<const RooStats::HistFactory::FlexibleInterpVar *>(func);
      elem["type"] << "interpolation0d";
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
      elem["name"] << chname;
      RooRealSumPdf *sumpdf = NULL;
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
      // this seems to be ok
      elem["type"] << "histfactory";
      auto &samples = elem["samples"];
      samples.set_map();

      for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
         const auto func = sumpdf->funcList().at(sampleidx);
         const auto coef = sumpdf->coefList().at(sampleidx);
         std::string samplename = func->GetName();
         if (samplename.find("L_x_") == 0)
            samplename = samplename.substr(4);
         auto end = samplename.find("_" + chname);
         if (end < samplename.size())
            samplename = samplename.substr(0, end);
         auto &s = samples[samplename];
         s.set_map();
         s["type"] << "histogram";
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
         TH1 *hist = NULL;
         ParamHistFunc *phf = NULL;
         PiecewiseInterpolation *pip = NULL;
         RooStats::HistFactory::FlexibleInterpVar *fip = NULL;
         std::vector<std::string> varnames;
         for (const auto &e : elems) {
            if (e->InheritsFrom(RooConstVar::Class())) {
               if (((RooConstVar *)e)->getVal() == 1.)
                  continue;
               auto &norms = s["normFactors"];
               norms.set_seq();
               norms.append_child() << e->GetName();
            } else if (e->InheritsFrom(RooRealVar::Class())) {
               auto &norms = s["normFactors"];
               norms.set_seq();
               norms.append_child() << e->GetName();
            } else if (e->InheritsFrom(RooHistFunc::Class())) {
               const RooHistFunc *hf = static_cast<const RooHistFunc *>(e);
               if (varnames.size() == 0) {
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
               phf = (ParamHistFunc *)e;
            }
         }
         if (pip) {
            auto &systs = s["histogramSystematics"];
            systs.set_map();
            if (!hist) {
               hist = histFunc2TH1(dynamic_cast<const RooHistFunc *>(pip->nominalHist()));
            }
            if (varnames.size() == 0) {
               varnames = getVarnames(dynamic_cast<const RooHistFunc *>(pip->nominalHist()));
            }
            for (size_t i = 0; i < pip->paramList().size(); ++i) {
               std::string sysname(pip->paramList().at(i)->GetName());
               if (sysname.find("alpha_") == 0) {
                  sysname = sysname.substr(6);
               }
               auto &sys = systs[sysname];
               sys.set_map();
               auto &dataLow = sys["dataLow"];
               TH1 *histLow = histFunc2TH1(dynamic_cast<RooHistFunc *>(pip->lowList().at(i)));
               RooJSONFactoryWSTool::exportHistogram(*histLow, dataLow, varnames, 0, false, false);
               delete histLow;
               auto &dataHigh = sys["dataHigh"];
               TH1 *histHigh = histFunc2TH1(dynamic_cast<RooHistFunc *>(pip->highList().at(i)));
               RooJSONFactoryWSTool::exportHistogram(*histHigh, dataHigh, varnames, 0, false, false);
               delete histHigh;
            }
         }
         if (!hist) {
            RooJSONFactoryWSTool::error(
               TString::Format("cannot find histogram in HistFactory-style pdf '%s'", prodpdf->GetName()));
            return false;
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
         if (phf) {
            s["statError"] << 1;
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               ++idx;
               RooPoisson *constraint = findPoissonClient(g);
               if (constraint) {
                  double erel = 1. / sqrt(constraint->getX().getVal());
                  hist->SetBinError(idx, erel * hist->GetBinContent(idx));
               } else {
                  hist->SetBinError(idx, 0);
               }
            }
         } else {
            s["statError"] << 0;
         }
         if (hist) {
            auto &data = s["data"];
            RooJSONFactoryWSTool::writeObservables(*hist, elem, varnames);
            RooJSONFactoryWSTool::exportHistogram(*hist, data, varnames, 0, false, phf);
            delete hist;
         }
      }
      return true;
   }

   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *p, JSONNode &elem) const override
   {
      const RooProdPdf *prodpdf = static_cast<const RooProdPdf *>(p);
      if (tryExport(prodpdf, elem)) {
         return true;
      }
      elem["type"] << "pdfprod";
      auto &factors = elem["factors"];
      factors.set_seq();
      for (const auto &v : prodpdf->pdfList()) {
         factors.append_child() << v->GetName();
      }
      return true;
   }
};

STATIC_EXECUTE(

   RooJSONFactoryWSTool::registerImporter("histfactory", new RooRealSumPdfFactory());
   RooJSONFactoryWSTool::registerImporter("histogram", new RooHistogramFactory());
   RooJSONFactoryWSTool::registerExporter(RooStats::HistFactory::FlexibleInterpVar::Class(),
                                          new FlexibleInterpVarStreamer());
   RooJSONFactoryWSTool::registerExporter(RooProdPdf::Class(), new HistFactoryStreamer());

)

} // namespace
