// Tests for the auxiliary measurements (constraint terms) of HistFactory models in HS3
// Authors: Carsten D. Burgard, DESY/CMS, 11/2026

// A classical HistFactory workspace is written to HS3 in every layout of the auxiliary measurements that the flags
// RooJSONFactoryWSTool::Config::exportArrayDist and ::exportConstraintData select, and read back. The model must stay
// the same: the same likelihood, the same toys, and the same document when it is written again, also if the layout
// changes between the exports.

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFit/Detail/JSONInterface.h>

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooFit/ModelConfig.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooLognormal.h>
#include <RooMinimizer.h>
#include <RooPoisson.h>
#include <RooProdPdf.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TH1D.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace {

// The four combinations of the two export flags. The HS3 documentation shows three layouts for auxiliary
// measurements: the terms in "aux_distributions" (chapter Likelihoods), each constraint paired with its own data set,
// and one array distribution paired with one unbinned data set (both in chapter Auxiliary measurements).
struct Style {
   const char *name;
   bool arrays;         // config().exportArrayDist
   bool constraintData; // config().exportConstraintData
};

const std::vector<Style> &styles()
{
   static const std::vector<Style> out{
      {"ArrayAux", true, false},  // default: array distributions in "aux_distributions"
      {"ArrayData", true, true},  // array distributions paired with unbinned data sets
      {"PlainData", false, true}, // every constraint paired with its own data set
      {"PlainAux", false, false}, // every constraint in "aux_distributions"
   };
   return out;
}

std::string exportWith(RooWorkspace &ws, Style const &style)
{
   auto &config = RooJSONFactoryWSTool::config();
   const bool oldArrays = config.exportArrayDist;
   const bool oldData = config.exportConstraintData;
   config.exportArrayDist = style.arrays;
   config.exportConstraintData = style.constraintData;
   std::string out;
   try {
      out = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   } catch (...) {
      config.exportArrayDist = oldArrays;
      config.exportConstraintData = oldData;
      throw;
   }
   config.exportArrayDist = oldArrays;
   config.exportConstraintData = oldData;
   return out;
}

std::unique_ptr<RooWorkspace> importJSON(std::string const &json, const char *name = "ws")
{
   auto ws = std::make_unique<RooWorkspace>(name);
   RooJSONFactoryWSTool{*ws}.importJSONfromString(json);
   return ws;
}

// A classical HistFactory workspace with three channels whose sets of constrained parameters overlap partially:
//
//   parameter            SR   CR1  CR2   constraint
//   Lumi                 x    x    x     Gaussian
//   alpha_systA          x         x     Gaussian
//   alpha_systB          x    x          Gaussian
//   alpha_systC          x    x    x     Gaussian
//   alpha_shapeS         x               Gaussian (histosys)
//   gamma_stat_SR_*      x               Poisson
//   gamma_stat_CR1_*          x          Gaussian
//   gamma_stat_CR2_*               x     Poisson
//   gamma_bkg2shape_*    x               Poisson (shapesys)
//   gamma_cr1shape_*          x          Gaussian (shapesys)
//   mu                   x         x     free
//   k_bkg1                    x          free
std::unique_ptr<RooWorkspace> makeHistFactoryWorkspace(std::string const &tag)
{
   using namespace RooStats::HistFactory;

   const std::string inputFile = "TestHS3AuxiliaryMeasurements_" + tag + ".root";
   struct Hist {
      const char *name;
      std::vector<double> contents;
      std::vector<double> errors;
   };
   // clang-format off
   const std::vector<Hist> hists{
      {"SR_data",        {62., 95.},        {}},
      {"SR_signal",      {10., 25.},        {0.5, 1.0}},
      {"SR_signal_up",   {11., 28.},        {}},
      {"SR_signal_dn",   {9.5, 22.},        {}},
      {"SR_bkg1",        {30., 40.},        {2.0, 3.0}},
      {"SR_bkg2",        {20., 25.},        {1.5, 2.5}},
      {"SR_bkg2_shape",  {0.10, 0.20},      {}},
      {"CR1_data",       {210., 148.},      {}},
      {"CR1_bkg1",       {150., 100.},      {6.0, 5.0}},
      {"CR1_bkg2",       {50., 60.},        {3.0, 4.0}},
      {"CR1_bkg2_shape", {0.05, 0.08},      {}},
      {"CR2_data",       {85., 71., 40.},   {}},
      {"CR2_bkg2",       {80., 60., 30.},   {4.0, 3.5, 2.5}},
      {"CR2_signal",     {3., 6., 9.},      {0.3, 0.4, 0.5}},
   };
   // clang-format on
   {
      TFile file(inputFile.c_str(), "RECREATE");
      for (auto const &h : hists) {
         const int n = h.contents.size();
         TH1D hist{h.name, h.name, n, 0., static_cast<double>(n)};
         for (int i = 0; i < n; ++i) {
            hist.SetBinContent(i + 1, h.contents[i]);
            hist.SetBinError(i + 1, h.errors.empty() ? 0. : h.errors[i]);
         }
         file.WriteTObject(&hist);
      }
   }

   Measurement meas(("meas_" + tag).c_str(), "meas");
   meas.SetOutputFilePrefix("TestHS3AuxiliaryMeasurements_" + tag);
   meas.SetPOI("mu");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.05);

   Channel sr("SR");
   sr.SetData("SR_data", inputFile);
   sr.SetStatErrorConfig(0.005, "Poisson");
   {
      Sample signal("signal", "SR_signal", inputFile);
      signal.AddNormFactor("mu", 1, 0, 5);
      signal.AddOverallSys("systA", 0.92, 1.08);
      signal.AddHistoSys("shapeS", "SR_signal_dn", inputFile, "", "SR_signal_up", inputFile, "");
      sr.AddSample(signal);
      Sample bkg1("bkg1", "SR_bkg1", inputFile);
      bkg1.AddOverallSys("systA", 0.97, 1.03);
      bkg1.AddOverallSys("systB", 0.90, 1.10);
      bkg1.ActivateStatError();
      sr.AddSample(bkg1);
      Sample bkg2("bkg2", "SR_bkg2", inputFile);
      bkg2.AddOverallSys("systC", 0.95, 1.05);
      bkg2.ActivateStatError();
      bkg2.AddShapeSys("bkg2shape", Constraint::Poisson, "SR_bkg2_shape", inputFile);
      sr.AddSample(bkg2);
   }
   meas.AddChannel(sr);

   Channel cr1("CR1");
   cr1.SetData("CR1_data", inputFile);
   cr1.SetStatErrorConfig(0.005, "Gaussian");
   {
      Sample bkg1("bkg1", "CR1_bkg1", inputFile);
      bkg1.AddNormFactor("k_bkg1", 1, 0, 3);
      bkg1.AddOverallSys("systB", 0.93, 1.07);
      bkg1.ActivateStatError();
      cr1.AddSample(bkg1);
      Sample bkg2("bkg2", "CR1_bkg2", inputFile);
      bkg2.AddOverallSys("systC", 0.96, 1.04);
      bkg2.AddShapeSys("cr1shape", Constraint::Gaussian, "CR1_bkg2_shape", inputFile);
      cr1.AddSample(bkg2);
   }
   meas.AddChannel(cr1);

   Channel cr2("CR2");
   cr2.SetData("CR2_data", inputFile);
   cr2.SetStatErrorConfig(0.005, "Poisson");
   {
      Sample bkg2("bkg2", "CR2_bkg2", inputFile);
      bkg2.AddOverallSys("systC", 0.94, 1.06);
      bkg2.AddOverallSys("systA", 0.98, 1.02);
      bkg2.ActivateStatError();
      cr2.AddSample(bkg2);
      Sample signal("signal", "CR2_signal", inputFile);
      signal.AddNormFactor("mu", 1, 0, 5);
      signal.AddOverallSys("systA", 0.92, 1.08);
      cr2.AddSample(signal);
   }
   meas.AddChannel(cr2);

   meas.CollectHistograms();
   return std::unique_ptr<RooWorkspace>(MakeModelAndMeasurementFast(meas));
}

RooStats::ModelConfig *modelConfig(RooWorkspace &ws)
{
   for (TObject *obj : ws.allGenericObjects()) {
      if (auto *mc = dynamic_cast<RooStats::ModelConfig *>(obj)) {
         return mc;
      }
   }
   return nullptr;
}

std::string section(std::string const &json, const char *key)
{
   auto tree = RooFit::Detail::JSONTree::create(json);
   if (!tree->rootnode().has_child(key)) {
      return "";
   }
   std::stringstream ss;
   tree->rootnode()[key].writeJSON(ss);
   return ss.str();
}

// ---------------------------------------------------------------------------------------------------------------
// Models

enum class ModelKind {
   Base,                // classical HistFactory: Gaussian and Poisson constraints
   LogNormalNormFactor, // Base, plus by hand a log-normal constraint (median, shape) on the normfactor k_bkg1 of CR1
   LogNormalStandard    // the same with the RooLognormal in the standard parametrization (mu, sigma) of HS3
};

const char *toString(ModelKind kind)
{
   switch (kind) {
   case ModelKind::Base: return "Base";
   case ModelKind::LogNormalNormFactor: return "LogNormalNormFactor";
   case ModelKind::LogNormalStandard: return "LogNormalStandard";
   }
   return "";
}

// The Base model with a RooLognormal constraint on the free normfactor "k_bkg1" added as a factor of the channel CR1,
// in a fresh workspace. This is not what HistFactory builds, but the only way to get all three constraint types into
// channel products that the exporter recognizes: Measurement::AddLogNormSyst() creates a function of the parameter
// with a Gaussian constraint, which can't be exported (see the test NormSysWithFunctionAsParameter).
std::unique_ptr<RooWorkspace> addLogNormalConstraint(RooWorkspace &base, bool standard)
{
   auto *mc = modelConfig(base);
   auto &sim = dynamic_cast<RooSimultaneous &>(*mc->GetPdf());
   auto &cat = dynamic_cast<RooCategory &>(const_cast<RooAbsCategoryLValue &>(sim.indexCat()));

   RooRealVar nom{"nom_k_bkg1", "nom_k_bkg1", 1.0, 0.0, 10.0};
   nom.setConstant(true);
   // In the standard parametrization, the parameter is the mean of log(x).
   RooLognormal constraint{"k_bkg1Constraint",
                           "k_bkg1Constraint",
                           nom,
                           *base.var("k_bkg1"),
                           RooFit::RooConst(standard ? 0.2 : 1.2),
                           standard};

   RooSimultaneous newSim{"simPdf", "simPdf", cat};
   std::vector<std::unique_ptr<RooProdPdf>> owned;
   for (auto const &state : cat) {
      RooAbsPdf *channel = sim.getPdf(state.first.c_str());
      if (state.first == "CR1") {
         RooArgList factors{static_cast<RooProdPdf *>(channel)->pdfList()};
         factors.add(constraint);
         owned.push_back(std::make_unique<RooProdPdf>(channel->GetName(), channel->GetTitle(), factors));
         channel = owned.back().get();
      }
      newSim.addPdf(*channel, state.first.c_str());
   }

   auto ws = std::make_unique<RooWorkspace>("combined");
   ws->import(newSim, RooFit::Silence());
   ws->import(*base.data("obsData"), RooFit::Silence());
   ws->import(*base.data("asimovData"), RooFit::Silence());
   RooStats::ModelConfig newMc{"ModelConfig", ws.get()};
   newMc.SetPdf(*ws->pdf("simPdf"));
   auto names = [](RooArgSet const *set, std::string extra = "") {
      std::string out = extra;
      for (RooAbsArg *arg : *set) {
         out += (out.empty() ? "" : ",") + std::string(arg->GetName());
      }
      return out;
   };
   newMc.SetParametersOfInterest(ws->argSet(names(mc->GetParametersOfInterest())));
   newMc.SetNuisanceParameters(ws->argSet(names(mc->GetNuisanceParameters())));
   newMc.SetObservables(ws->argSet(names(mc->GetObservables())));
   newMc.SetGlobalObservables(ws->argSet(names(mc->GetGlobalObservables(), "nom_k_bkg1")));
   ws->import(newMc);
   return ws;
}

std::unique_ptr<RooWorkspace> makeWorkspace(ModelKind kind)
{
   static int counter = 0;
   auto base = makeHistFactoryWorkspace("m" + std::to_string(counter++));
   return kind == ModelKind::Base ? std::move(base)
                                  : addLogNormalConstraint(*base, kind == ModelKind::LogNormalStandard);
}

// ---------------------------------------------------------------------------------------------------------------
// Helpers

RooStats::ModelConfig *findModelConfig(RooWorkspace &ws, std::string const &name)
{
   if (auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws.obj(name))) {
      return mc;
   }
   return nullptr;
}

std::vector<std::string> modelConfigNames(RooWorkspace &ws)
{
   std::vector<std::string> out;
   for (TObject *obj : ws.allGenericObjects()) {
      if (dynamic_cast<RooStats::ModelConfig *>(obj)) {
         out.push_back(obj->GetName());
      }
   }
   std::sort(out.begin(), out.end());
   return out;
}

std::set<std::string> namesOf(RooAbsCollection const *coll)
{
   std::set<std::string> out;
   if (coll) {
      for (RooAbsArg *arg : *coll) {
         out.insert(arg->GetName());
      }
   }
   return out;
}

std::string join(std::set<std::string> const &names)
{
   std::string out;
   for (auto const &n : names) {
      out += (out.empty() ? "" : ",") + n;
   }
   return out;
}

// What a variable "is": its class, whether it is constant, and its value.
std::string nature(RooAbsArg const *arg)
{
   if (!arg) {
      return "<missing>";
   }
   std::stringstream ss;
   ss << arg->ClassName();
   if (auto const *real = dynamic_cast<RooAbsReal const *>(arg)) {
      ss << (arg->isConstant() ? " const" : " floating") << " value=" << real->getVal();
   }
   return ss.str();
}

// The class, constness, value and range of all variables and constants of the workspace that have a name (literal
// constants are named after their value).
std::map<std::string, std::string> describeVariables(RooWorkspace &ws)
{
   std::map<std::string, std::string> out;
   for (RooAbsArg *arg : ws.components()) {
      auto *var = dynamic_cast<RooRealVar *>(arg);
      auto *constVar = dynamic_cast<RooConstVar *>(arg);
      if (!var && !constVar) {
         continue;
      }
      std::stringstream ss;
      // The current value of an observable is no property of the model.
      if (std::string(arg->GetName()).rfind("obs_x_", 0) == 0) {
         ss << arg->ClassName();
      } else {
         ss << nature(arg);
      }
      if (var) {
         ss << " range=[" << var->getMin() << "," << var->getMax() << "]";
      }
      out[arg->GetName()] = ss.str();
   }
   return out;
}

// The differences between two maps, at most `maxLines` of them.
std::string diff(std::map<std::string, std::string> const &expected, std::map<std::string, std::string> const &actual,
                 std::size_t maxLines = 12)
{
   std::vector<std::string> lines;
   for (auto const &[key, value] : expected) {
      auto found = actual.find(key);
      if (found == actual.end()) {
         lines.push_back("  missing: " + key + " (" + value + ")");
      } else if (found->second != value) {
         lines.push_back("  " + key + ": expected {" + value + "}, got {" + found->second + "}");
      }
   }
   for (auto const &[key, value] : actual) {
      if (!expected.count(key)) {
         lines.push_back("  extra: " + key + " (" + value + ")");
      }
   }
   std::string out;
   for (std::size_t i = 0; i < lines.size() && i < maxLines; ++i) {
      out += lines[i] + "\n";
   }
   if (lines.size() > maxLines) {
      out += "  ... and " + std::to_string(lines.size() - maxLines) + " more\n";
   }
   return out;
}

// The roles of the servers of all constraint pdfs: which object is "x", which the "mean".
std::map<std::string, std::string> describeConstraints(RooWorkspace &ws)
{
   std::map<std::string, std::string> out;
   for (RooAbsArg *arg : ws.allPdfs()) {
      std::stringstream ss;
      if (auto *gauss = dynamic_cast<RooGaussian *>(arg)) {
         // A Gaussian is symmetric in x and mean: which of the two is the observable is no property of the model.
         std::string first = gauss->getX().GetName();
         std::string second = gauss->getMean().GetName();
         if (second < first) {
            std::swap(first, second);
         }
         ss << "RooGaussian {x,mean}={" << first << "," << second << "} sigma=" << gauss->getSigma().ClassName() << ":"
            << gauss->getSigma().getVal();
      } else if (auto *pois = dynamic_cast<RooPoisson *>(arg)) {
         ss << "RooPoisson x=" << pois->getX().GetName() << " mean=" << pois->getMean().GetName()
            << " noRounding=" << pois->getNoRounding();
      } else if (auto *logn = dynamic_cast<RooLognormal *>(arg)) {
         ss << "RooLognormal x=" << logn->getX().GetName() << " m0=" << logn->getMedian().GetName();
      } else {
         continue;
      }
      out[arg->GetName()] = ss.str();
   }
   return out;
}

// The same without the names of the pdfs, which change if the constraints are components of an array.
std::map<std::string, std::string> constraintSignatures(RooWorkspace &ws)
{
   std::map<std::string, std::string> out;
   for (auto const &[name, description] : describeConstraints(ws)) {
      out[description] = "";
   }
   return out;
}

void setPoint(RooWorkspace &ws, RooStats::ModelConfig &mc, bool displaced)
{
   for (auto *var : static_range_cast<RooRealVar *>(*mc.GetNuisanceParameters())) {
      const std::string name = var->GetName();
      RooRealVar *v = ws.var(name);
      if (name.rfind("alpha_", 0) == 0) {
         v->setVal(displaced ? 0.3 : 0.0);
      } else if (name.rfind("gamma_", 0) == 0) {
         v->setVal(displaced ? 1.04 : 1.0);
      } else if (name == "Lumi") {
         v->setVal(displaced ? 1.02 : 1.0);
      }
   }
   if (RooRealVar *k = ws.var("k_bkg1")) {
      k->setVal(displaced ? 1.1 : 1.0);
   }
}

struct Scan {
   std::vector<double> fixed;     // NLL at the nominal nuisance parameters
   std::vector<double> displaced; // NLL at displaced nuisance parameters
   std::vector<double> profiled;  // NLL minimized over the nuisance parameters
   std::vector<int> status;
};

const std::vector<double> muValues{0.4, 1.0, 1.6, 2.5};

Scan likelihoodScan(RooWorkspace &ws, RooStats::ModelConfig &mc, RooAbsData &data, bool profile)
{
   Scan out;
   RooRealVar &mu = *ws.var("mu");
   const bool wasConstant = mu.isConstant();
   std::unique_ptr<RooAbsReal> nll{
      mc.GetPdf()->createNLL(data, RooFit::GlobalObservables(*mc.GetGlobalObservables()), RooFit::Offset(false))};
   // The constness of the parameters as the workspace has it: the scan must not depend on fixing things by hand.
   for (double value : muValues) {
      mu.setVal(value);
      setPoint(ws, mc, false);
      out.fixed.push_back(nll->getVal());
      setPoint(ws, mc, true);
      out.displaced.push_back(nll->getVal());
   }
   if (profile) {
      for (double value : muValues) {
         setPoint(ws, mc, false);
         mu.setVal(value);
         mu.setConstant(true);
         RooMinimizer minimizer{*nll};
         minimizer.setPrintLevel(-1);
         minimizer.setStrategy(1);
         out.status.push_back(minimizer.minimize("Minuit2", "migrad"));
         out.profiled.push_back(nll->getVal());
      }
   }
   mu.setConstant(wasConstant);
   mu.setVal(1.0);
   setPoint(ws, mc, false);
   return out;
}

// A data set as text, one line per entry.
std::vector<std::string> describeData(RooAbsData const &data)
{
   std::vector<std::string> out;
   for (int i = 0; i < data.numEntries(); ++i) {
      RooArgSet const *row = data.get(i);
      std::map<std::string, std::string> sorted;
      for (RooAbsArg *arg : *row) {
         std::stringstream ss;
         ss.precision(15);
         if (auto *cat = dynamic_cast<RooAbsCategory *>(arg)) {
            ss << cat->getCurrentLabel();
         } else {
            ss << static_cast<RooAbsReal *>(arg)->getVal();
         }
         sorted[arg->GetName()] = ss.str();
      }
      std::stringstream line;
      line.precision(15);
      for (auto const &[name, value] : sorted) {
         line << name << "=" << value << " ";
      }
      line << "w=" << data.weight();
      out.push_back(line.str());
   }
   return out;
}

struct Toys {
   std::vector<std::string> observables; // toy data of the observables, per entry
   std::vector<std::string> globals;     // toy values of the global observables
   double sumEntries = 0.;
   double expected = 0.;
   std::map<std::string, double> perChannel;
};

// The yield per channel and bin of a data set, as text. The order of the entries doesn't matter.
std::vector<std::string> binnedYields(RooAbsData const &data)
{
   std::map<std::string, double> yields;
   for (int i = 0; i < data.numEntries(); ++i) {
      RooArgSet const *row = data.get(i);
      std::string channel;
      for (RooAbsArg *arg : *row) {
         if (auto *cat = dynamic_cast<RooAbsCategory *>(arg)) {
            channel = cat->getCurrentLabel();
         }
      }
      auto *obs = dynamic_cast<RooRealVar *>(row->find(("obs_x_" + channel).c_str()));
      std::stringstream key;
      key << channel << " bin " << (obs ? obs->getBin() : -1);
      yields[key.str()] += data.weight();
   }
   std::vector<std::string> out;
   for (auto const &[key, value] : yields) {
      std::stringstream ss;
      ss.precision(15);
      ss << key << ": " << value;
      out.push_back(ss.str());
   }
   return out;
}

Toys generateToys(RooWorkspace &ws, RooStats::ModelConfig &mc, int seed)
{
   Toys out;
   ws.var("mu")->setVal(1.0);
   setPoint(ws, mc, false);
   RooAbsPdf &pdf = *mc.GetPdf();
   RooRandom::randomGenerator()->SetSeed(seed);
   std::unique_ptr<RooAbsData> data{pdf.generate(*mc.GetObservables(), RooFit::Extended())};
   out.observables = binnedYields(*data);
   out.sumEntries = data->sumEntries();
   out.expected = pdf.expectedEvents(*mc.GetObservables());
   for (int i = 0; i < data->numEntries(); ++i) {
      for (RooAbsArg *arg : *data->get(i)) {
         if (auto *cat = dynamic_cast<RooAbsCategory *>(arg)) {
            out.perChannel[cat->getCurrentLabel()] += data->weight();
         }
      }
   }
   RooRandom::randomGenerator()->SetSeed(seed + 1);
   auto &sim = dynamic_cast<RooSimultaneous &>(pdf);
   std::unique_ptr<RooAbsData> globals{sim.generateSimGlobal(*mc.GetGlobalObservables(), 1)};
   out.globals = describeData(*globals);
   return out;
}

std::string typeOf(JSONNode const &node)
{
   return node.has_child("type") ? node["type"].val() : "";
}

std::vector<std::string> strings(JSONNode const &node, const char *key)
{
   std::vector<std::string> out;
   if (node.has_child(key)) {
      for (auto const &child : node[key].children()) {
         out.push_back(child.is_null() ? "<null>" : child.val());
      }
   }
   return out;
}

// The names of the constraint distributions that the modifiers of the HistFactory distributions name.
std::set<std::string> namedConstraints(JSONNode const &root)
{
   std::set<std::string> out;
   for (auto const &dist : root["distributions"].children()) {
      if (typeOf(dist) != "histfactory_dist") {
         continue;
      }
      for (auto const &sample : dist["samples"].children()) {
         for (auto const &modifier : sample["modifiers"].children()) {
            for (auto const &name : strings(modifier, "constraints")) {
               if (name != "<null>") {
                  out.insert(name);
               }
            }
         }
      }
   }
   return out;
}

const std::set<std::string> constraintTypes{"gaussian_dist", "poisson_dist", "poisson_continuous_dist",
                                            "lognormal_dist"};

using Param = std::tuple<ModelKind, Style>;

std::string paramName(testing::TestParamInfo<Param> const &info)
{
   return std::string(toString(std::get<0>(info.param))) + "_" + std::get<1>(info.param).name;
}

class AuxMeasurements : public testing::TestWithParam<Param> {
protected:
   void SetUp() override
   {
      kind = std::get<0>(GetParam());
      style = std::get<1>(GetParam());
      original = makeWorkspace(kind);
      ASSERT_NE(original, nullptr);
      ASSERT_NO_THROW(json = exportWith(*original, style));
   }

   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::ERROR};
   ModelKind kind = ModelKind::Base;
   Style style{};
   std::unique_ptr<RooWorkspace> original;
   std::string json;
};

// ---------------------------------------------------------------------------------------------------------------
// 1. The flags select the layout of the auxiliary measurements.

TEST_P(AuxMeasurements, Layout)
{
   auto tree = JSONTree::create(json);
   JSONNode const &root = tree->rootnode();

   std::map<std::string, int> arraysByType;
   std::map<std::string, JSONNode const *> distributions;
   std::set<std::string> standaloneConstraints;
   for (auto const &dist : root["distributions"].children()) {
      const std::string name = dist["name"].val();
      distributions[name] = &dist;
      if (typeOf(dist) == "array_dist") {
         ++arraysByType[dist["component_type"].val()];
      } else if (constraintTypes.count(typeOf(dist))) {
         standaloneConstraints.insert(name);
      }
   }

   const std::set<std::string> named = namedConstraints(root);
   EXPECT_FALSE(named.empty());
   const std::size_t nTypes = kind == ModelKind::Base ? 2 : 3;

   if (style.arrays) {
      // Exactly one array distribution per constraint type in use, and nothing else.
      EXPECT_EQ(arraysByType.size(), nTypes) << "array distributions by component type";
      EXPECT_LE(arraysByType.size(), 3u);
      for (auto const &[type, count] : arraysByType) {
         EXPECT_EQ(count, 1) << "array distributions with component_type " << type;
      }
      EXPECT_TRUE(standaloneConstraints.empty())
         << "constraints written on their own although arrays are enabled: " << join(standaloneConstraints);
      for (auto const &name : named) {
         EXPECT_EQ(typeOf(*distributions[name]), "array_dist") << "modifier constraint '" << name << "'";
      }
   } else {
      EXPECT_TRUE(arraysByType.empty()) << "array distributions although they are switched off";
      for (auto const &name : named) {
         EXPECT_TRUE(standaloneConstraints.count(name)) << "modifier constraint '" << name << "'";
      }
   }

   // Every constraint term enters each likelihood exactly once, in the place that the flag selects.
   std::set<std::string> allTerms = named;
   allTerms.insert(standaloneConstraints.begin(), standaloneConstraints.end());
   for (auto const &[name, dist] : distributions) {
      if (typeOf(*dist) == "array_dist") {
         allTerms.insert(name);
      }
   }
   std::map<std::string, JSONNode const *> data;
   for (auto const &d : root["data"].children()) {
      data[d["name"].val()] = &d;
   }
   ASSERT_GT(root["likelihoods"].num_children(), 0u);
   for (auto const &likelihood : root["likelihoods"].children()) {
      const std::string lhName = likelihood["name"].val();
      const auto dists = strings(likelihood, "distributions");
      const auto lhData = strings(likelihood, "data");
      const auto aux = strings(likelihood, "aux_distributions");
      ASSERT_EQ(dists.size(), lhData.size());
      for (auto const &term : allTerms) {
         const auto inDists = std::count(dists.begin(), dists.end(), term);
         const auto inAux = std::count(aux.begin(), aux.end(), term);
         EXPECT_EQ(inDists + inAux, 1) << "term '" << term << "' in likelihood '" << lhName << "'";
         if (style.constraintData) {
            EXPECT_EQ(inDists, 1) << "term '" << term << "' should be paired with data in '" << lhName << "'";
         } else {
            EXPECT_EQ(inAux, 1) << "term '" << term << "' should be in aux_distributions of '" << lhName << "'";
         }
      }
      if (style.constraintData) {
         EXPECT_TRUE(aux.empty()) << "aux_distributions of '" << lhName << "' with exportConstraintData";
         std::set<std::string> wrongType;
         for (std::size_t i = 0; i < dists.size(); ++i) {
            if (!allTerms.count(dists[i])) {
               continue;
            }
            auto found = data.find(lhData[i]);
            ASSERT_NE(found, data.end()) << "data '" << lhData[i] << "' of term '" << dists[i] << "'";
            JSONNode const &d = *found->second;
            JSONNode const &dist = *distributions[dists[i]];
            if (style.arrays) {
               // One unbinned data set with a single entry, with one axis per component in the order of the components.
               // An array with a single component may be paired with point data instead.
               if (dist["components"].num_children() == 1 && typeOf(d) == "point") {
                  continue;
               }
               EXPECT_EQ(typeOf(d), "unbinned") << lhData[i];
               ASSERT_EQ(d["entries"].num_children(), 1u) << lhData[i];
               ASSERT_EQ(d["axes"].num_children(), dist["components"].num_children()) << lhData[i];
               ASSERT_EQ(d["entries"].child(0).num_children(), dist["components"].num_children()) << lhData[i];
               for (std::size_t j = 0; j < d["axes"].num_children(); ++j) {
                  EXPECT_EQ(d["axes"].child(j)["name"].val(), dist["components"].child(j)["x"].val()) << lhData[i];
               }
            } else {
               // The HS3 documentation pairs every constraint with a data set of type "point".
               if (typeOf(d) != "point") {
                  wrongType.insert(lhData[i] + ":" + typeOf(d));
               }
            }
         }
         EXPECT_TRUE(wrongType.empty()) << wrongType.size()
                                        << " data sets of single constraints are not of type \"point\", e.g. "
                                        << *wrongType.begin();
      } else {
         for (auto const &[name, d] : data) {
            EXPECT_FALSE(name.size() > 5 && name.compare(name.size() - 5, 5, "_data") == 0 &&
                         allTerms.count(name.substr(0, name.size() - 5)))
               << "data set '" << name << "' although the terms are in aux_distributions";
         }
      }
   }
}

// ---------------------------------------------------------------------------------------------------------------
// 2. The re-imported workspace is the original one.

TEST_P(AuxMeasurements, RoundTripWorkspace)
{
   std::unique_ptr<RooWorkspace> imported;
   ASSERT_NO_THROW(imported = importJSON(json));

   // The document is a fixed point: exporting the imported workspace again gives the same document.
   std::string again;
   ASSERT_NO_THROW(again = exportWith(*imported, style));
   if (again != json) {
      std::string differing;
      for (const char *key : {"metadata", "distributions", "functions", "data", "likelihoods", "domains",
                              "parameter_points", "analyses", "misc"}) {
         if (section(again, key) != section(json, key)) {
            differing += std::string(differing.empty() ? "" : ", ") + key;
         }
      }
      ADD_FAILURE() << "second export differs from the first in: " << differing;
   }

   // The ModelConfig.
   EXPECT_EQ(modelConfigNames(*imported), modelConfigNames(*original));
   auto *mcOrig = modelConfig(*original);
   ASSERT_NE(mcOrig, nullptr);
   auto *mc = findModelConfig(*imported, mcOrig->GetName());
   ASSERT_NE(mc, nullptr) << "no ModelConfig '" << mcOrig->GetName()
                          << "', there are: " << testing::PrintToString(modelConfigNames(*imported));
   ASSERT_NE(mc->GetPdf(), nullptr);
   EXPECT_STREQ(mc->GetPdf()->GetName(), mcOrig->GetPdf()->GetName());
   EXPECT_STREQ(mc->GetPdf()->ClassName(), mcOrig->GetPdf()->ClassName());
   EXPECT_EQ(join(namesOf(mc->GetParametersOfInterest())), join(namesOf(mcOrig->GetParametersOfInterest())));
   EXPECT_EQ(join(namesOf(mc->GetNuisanceParameters())), join(namesOf(mcOrig->GetNuisanceParameters())));
   EXPECT_EQ(join(namesOf(mc->GetGlobalObservables())), join(namesOf(mcOrig->GetGlobalObservables())));
   EXPECT_EQ(join(namesOf(mc->GetObservables())), join(namesOf(mcOrig->GetObservables())));
   EXPECT_TRUE(!mc->GetExternalConstraints() || mc->GetExternalConstraints()->empty())
      << join(namesOf(mc->GetExternalConstraints()));

   // The data sets.
   std::set<std::string> dataOrig;
   std::set<std::string> dataNew;
   for (RooAbsData *d : original->allData()) {
      dataOrig.insert(d->GetName());
   }
   for (RooAbsData *d : imported->allData()) {
      dataNew.insert(d->GetName());
   }
   EXPECT_EQ(join(dataNew), join(dataOrig));

   // The variables: class, constness, value and range.
   const std::string variables = diff(describeVariables(*original), describeVariables(*imported));
   EXPECT_TRUE(variables.empty()) << "variables differ:\n" << variables;

   // The constraint terms: same pdfs with the same roles of their servers.
   // The names of the constraint pdfs need not survive: the components of an array have none.
   const std::string constraints = diff(constraintSignatures(*original), constraintSignatures(*imported), 6);
   EXPECT_TRUE(constraints.empty()) << "roles in the constraint pdfs differ:\n" << constraints;
}

TEST_P(AuxMeasurements, LikelihoodScan)
{
   std::unique_ptr<RooWorkspace> imported;
   ASSERT_NO_THROW(imported = importJSON(json));
   auto *mcOrig = modelConfig(*original);
   auto *mc = findModelConfig(*imported, mcOrig->GetName());
   if (!mc) {
      // Reported by RoundTripWorkspace; scan what there is.
      mc = modelConfig(*imported);
   }
   ASSERT_NE(mc, nullptr);
   ASSERT_NE(mc->GetGlobalObservables(), nullptr);
   ASSERT_NE(imported->data("obsData"), nullptr);

   const Scan expected = likelihoodScan(*original, *mcOrig, *original->data("obsData"), true);
   const Scan actual = likelihoodScan(*imported, *mc, *imported->data("obsData"), true);

   for (std::size_t i = 0; i < muValues.size(); ++i) {
      SCOPED_TRACE("mu = " + std::to_string(muValues[i]));
      ASSERT_TRUE(std::isfinite(expected.fixed[i]));
      // Differences of the NLL, which is what a scan uses.
      const double tolerance = 1e-9 * std::abs(expected.fixed[i]);
      EXPECT_NEAR(actual.fixed[i] - actual.fixed[1], expected.fixed[i] - expected.fixed[1], tolerance) << "fixed";
      EXPECT_NEAR(actual.displaced[i] - actual.fixed[1], expected.displaced[i] - expected.fixed[1], tolerance)
         << "displaced nuisance parameters";
      EXPECT_EQ(actual.status[i], 0);
      EXPECT_EQ(expected.status[i], 0);
      EXPECT_NEAR(actual.profiled[i] - actual.profiled[1], expected.profiled[i] - expected.profiled[1], 1e-4)
         << "profiled";
   }
   // The absolute value: an identical model has the same constant terms.
   EXPECT_NEAR(actual.fixed[1], expected.fixed[1], 1e-9 * std::abs(expected.fixed[1])) << "absolute NLL";
}

// ---------------------------------------------------------------------------------------------------------------
// 4. Toys.

// One toy: data for the observables and values for the global observables, sampled from the model of `ws`.
struct Toy {
   std::unique_ptr<RooAbsData> data;
   std::map<std::string, double> globals;
};

Toy sampleToy(RooWorkspace &ws, RooStats::ModelConfig &mc)
{
   Toy out;
   ws.var("mu")->setVal(1.0);
   setPoint(ws, mc, false);
   auto &sim = dynamic_cast<RooSimultaneous &>(*mc.GetPdf());
   out.data = std::unique_ptr<RooAbsData>{sim.generate(*mc.GetObservables(), RooFit::Extended())};
   std::unique_ptr<RooAbsData> globals{sim.generateSimGlobal(*mc.GetGlobalObservables(), 1)};
   for (RooAbsArg *arg : *globals->get(0)) {
      out.globals[arg->GetName()] = static_cast<RooAbsReal *>(arg)->getVal();
   }
   return out;
}

// The NLL of the model of `ws` for a toy, at the nominal and at displaced parameters. The global observables are
// restored afterwards.
std::vector<double> evaluateToy(RooWorkspace &ws, RooStats::ModelConfig &mc, Toy const &toy)
{
   std::map<std::string, double> saved;
   for (auto const &[name, value] : toy.globals) {
      saved[name] = ws.var(name)->getVal();
      ws.var(name)->setVal(value);
   }
   std::unique_ptr<RooAbsReal> nll{
      mc.GetPdf()->createNLL(*toy.data, RooFit::GlobalObservables(*mc.GetGlobalObservables()), RooFit::Offset(false))};
   std::vector<double> out;
   for (double mu : {1.0, 1.7}) {
      ws.var("mu")->setVal(mu);
      setPoint(ws, mc, false);
      out.push_back(nll->getVal());
      setPoint(ws, mc, true);
      out.push_back(nll->getVal());
   }
   ws.var("mu")->setVal(1.0);
   setPoint(ws, mc, false);
   for (auto const &[name, value] : saved) {
      ws.var(name)->setVal(value);
   }
   return out;
}

// Mean and variance of the sampled global observables and of the yields per channel and bin, over many toys.
struct Moments {
   std::map<std::string, double> sum;
   std::map<std::string, double> sum2;
   int n = 0;
   double mean(std::string const &key) const { return sum.at(key) / n; }
   double variance(std::string const &key) const { return sum2.at(key) / n - mean(key) * mean(key); }
};

Moments sampleMoments(RooWorkspace &ws, RooStats::ModelConfig &mc, int nGlobals, int nData, int seed)
{
   Moments out;
   ws.var("mu")->setVal(1.0);
   setPoint(ws, mc, false);
   auto &sim = dynamic_cast<RooSimultaneous &>(*mc.GetPdf());
   RooRandom::randomGenerator()->SetSeed(seed);
   std::unique_ptr<RooAbsData> globals{sim.generateSimGlobal(*mc.GetGlobalObservables(), nGlobals)};
   for (int i = 0; i < globals->numEntries(); ++i) {
      for (RooAbsArg *arg : *globals->get(i)) {
         const double value = static_cast<RooAbsReal *>(arg)->getVal();
         out.sum[arg->GetName()] += value;
         out.sum2[arg->GetName()] += value * value;
      }
   }
   // The yields are rescaled such that both kinds of quantities are averages over `nGlobals` toys.
   const double scale = static_cast<double>(nGlobals) / nData;
   for (int i = 0; i < nData; ++i) {
      std::unique_ptr<RooAbsData> data{sim.generate(*mc.GetObservables(), RooFit::Extended())};
      std::map<std::string, double> yields;
      for (int j = 0; j < data->numEntries(); ++j) {
         RooArgSet const *row = data->get(j);
         std::string channel;
         for (RooAbsArg *arg : *row) {
            if (auto *cat = dynamic_cast<RooAbsCategory *>(arg)) {
               channel = cat->getCurrentLabel();
            }
         }
         auto *obs = dynamic_cast<RooRealVar *>(row->find(("obs_x_" + channel).c_str()));
         yields["yield " + channel + " bin " + std::to_string(obs ? obs->getBin() : -1)] += data->weight();
      }
      for (auto const &[key, value] : yields) {
         out.sum[key] += scale * value;
         out.sum2[key] += scale * value * value;
      }
   }
   out.n = nGlobals;
   return out;
}

TEST_P(AuxMeasurements, Toys)
{
   std::unique_ptr<RooWorkspace> imported;
   ASSERT_NO_THROW(imported = importJSON(json));
   auto *mcOrig = modelConfig(*original);
   auto *mc = findModelConfig(*imported, mcOrig->GetName());
   if (!mc) {
      mc = modelConfig(*imported);
   }
   ASSERT_NE(mc, nullptr);

   Toys toys;
   ASSERT_NO_THROW(toys = generateToys(*imported, *mc, 4357));

   // Valid: every channel has events, about as many as expected in total.
   EXPECT_EQ(toys.perChannel.size(), 3u);
   for (auto const &[channel, yield] : toys.perChannel) {
      EXPECT_GT(yield, 0.) << channel;
   }
   EXPECT_GT(toys.expected, 0.);
   EXPECT_NEAR(toys.sumEntries, toys.expected, 6. * std::sqrt(toys.expected));
   ASSERT_EQ(toys.globals.size(), 1u);
   EXPECT_EQ(toys.globals[0].find("nan"), std::string::npos) << toys.globals[0];

   // Identical for all representations: the documents describe the same RooFit objects, so with the same seed even
   // the random numbers are the same. Compare to the default representation.
   {
      auto reference = importJSON(exportWith(*original, styles()[0]));
      auto *mcRef = findModelConfig(*reference, mcOrig->GetName());
      if (!mcRef) {
         mcRef = modelConfig(*reference);
      }
      ASSERT_NE(mcRef, nullptr);
      const Toys ref = generateToys(*reference, *mcRef, 4357);
      EXPECT_EQ(toys.observables, ref.observables) << "toy data differ from the " << styles()[0].name << " style";
      EXPECT_EQ(toys.globals, ref.globals) << "toy global observables differ from the " << styles()[0].name << " style";
   }

   // The same model as the original one. The order in which the random numbers are used may differ, so the toys are
   // not compared one by one. Instead:
   // (a) A toy sampled from either model gives the same likelihood in both models.
   RooRandom::randomGenerator()->SetSeed(1234);
   for (int i = 0; i < 3; ++i) {
      for (bool fromOriginal : {true, false}) {
         SCOPED_TRACE("toy " + std::to_string(i) +
                      (fromOriginal ? " sampled from the original" : " sampled from the import"));
         Toy toy;
         ASSERT_NO_THROW(toy = fromOriginal ? sampleToy(*original, *mcOrig) : sampleToy(*imported, *mc));
         const std::vector<double> expected = evaluateToy(*original, *mcOrig, toy);
         const std::vector<double> actual = evaluateToy(*imported, *mc, toy);
         for (std::size_t j = 0; j < expected.size(); ++j) {
            ASSERT_TRUE(std::isfinite(expected[j])) << j;
            EXPECT_NEAR(actual[j], expected[j], 1e-9 * std::abs(expected[j])) << "evaluation " << j;
         }
      }
   }
   // (b) In the limit of many toys, both models sample the same distributions: the means and variances of the global
   // observables and of the yields agree within their statistical uncertainties. The two samples are independent.
   const int nGlobals = 2000;
   const int nData = 50;
   const Moments expected = sampleMoments(*original, *mcOrig, nGlobals, nData, 111);
   const Moments actual = sampleMoments(*imported, *mc, nGlobals, nData, 222);
   ASSERT_EQ(testing::PrintToString(namesOf(mc->GetGlobalObservables())),
             testing::PrintToString(namesOf(mcOrig->GetGlobalObservables())));
   for (auto const &[key, unused] : expected.sum) {
      ASSERT_TRUE(actual.sum.count(key)) << key;
      const bool isYield = key.rfind("yield", 0) == 0;
      const int n = isYield ? nData : nGlobals;
      const double variance = expected.variance(key) * (isYield ? static_cast<double>(nGlobals) / nData : 1.);
      ASSERT_GT(variance, 0.) << key;
      // 5 standard deviations of the difference of two means, and of two variances (Gaussian approximation).
      EXPECT_NEAR(actual.mean(key), expected.mean(key), 5. * std::sqrt(2. * variance / n)) << "mean of " << key;
      if (!isYield) {
         EXPECT_NEAR(actual.variance(key), expected.variance(key), 5. * variance * std::sqrt(4. / n))
            << "variance of " << key;
      }
   }
}

INSTANTIATE_TEST_SUITE_P(HS3, AuxMeasurements,
                         testing::Combine(testing::Values(ModelKind::Base, ModelKind::LogNormalNormFactor,
                                                          ModelKind::LogNormalStandard),
                                          testing::ValuesIn(styles())),
                         paramName);

// ---------------------------------------------------------------------------------------------------------------
// 3. The nature of the global observables is stable when the representation changes between exports.

// The layout of a document without the names of the constraints, which need not survive an array: the types of the
// distributions and data sets with their multiplicity, the number of terms of each likelihood, and the names of the
// array distributions.
std::string distributionTypes(std::string const &json);
std::string layoutSummary(std::string const &json)
{
   auto tree = JSONTree::create(json);
   JSONNode const &root = tree->rootnode();
   std::stringstream ss;
   ss << "distributions: " << distributionTypes(json) << "; arrays:";
   for (auto const &dist : root["distributions"].children()) {
      if (typeOf(dist) == "array_dist") {
         ss << " " << dist["name"].val() << "(" << dist["component_type"].val() << ", "
            << dist["components"].num_children() << ")";
      }
   }
   std::map<std::string, int> dataTypes;
   for (auto const &d : root["data"].children()) {
      ++dataTypes[typeOf(d)];
   }
   ss << "; data:";
   for (auto const &[type, count] : dataTypes) {
      ss << " " << count << " " << type;
   }
   for (auto const &likelihood : root["likelihoods"].children()) {
      ss << "; " << likelihood["name"].val() << ": " << strings(likelihood, "distributions").size() << " terms, "
         << strings(likelihood, "aux_distributions").size() << " aux";
   }
   ss << "; functions: " << (root.has_child("functions") ? root["functions"].num_children() : 0);
   return ss.str();
}

std::string distributionTypes(std::string const &json)
{
   auto tree = JSONTree::create(json);
   std::map<std::string, int> counts;
   for (auto const &dist : tree->rootnode()["distributions"].children()) {
      ++counts[typeOf(dist)];
   }
   std::string out;
   for (auto const &[type, count] : counts) {
      out += (out.empty() ? "" : ", ") + std::to_string(count) + " " + type;
   }
   return out;
}

class AuxMeasurementsChain : public testing::TestWithParam<ModelKind> {};

TEST_P(AuxMeasurementsChain, GlobalObservablesKeepTheirNature)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl{RooFit::ERROR};
   auto original = makeWorkspace(GetParam());
   auto *mcOrig = modelConfig(*original);
   ASSERT_NE(mcOrig, nullptr);

   // The global observables, and the other constants of the constraint terms (tau, sigma).
   auto natures = [&](RooWorkspace &ws) {
      std::map<std::string, std::string> out;
      for (RooAbsArg *arg : *mcOrig->GetGlobalObservables()) {
         out[arg->GetName()] = nature(ws.arg(arg->GetName()));
      }
      for (RooAbsArg *arg : original->components()) {
         const std::string name = arg->GetName();
         auto endsWith = [&](const char *suffix) {
            const std::string s = suffix;
            return name.size() > s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0;
         };
         if (endsWith("_tau") || endsWith("_sigma")) {
            out[name] = nature(ws.arg(name));
         }
      }
      return out;
   };
   auto globalSet = [&](RooWorkspace &ws) {
      auto *mc = findModelConfig(ws, mcOrig->GetName());
      if (!mc) {
         mc = modelConfig(ws);
      }
      return mc ? join(namesOf(mc->GetGlobalObservables())) : std::string("<no ModelConfig>");
   };
   const auto expected = natures(*original);
   const std::string expectedSet = globalSet(*original);
   const Scan reference = likelihoodScan(*original, *mcOrig, *original->data("obsData"), false);

   // Every pair of representations: export as `first`, import, export as `second`, import, export as `first`.
   for (auto const &first : styles()) {
      std::string json1;
      ASSERT_NO_THROW(json1 = exportWith(*original, first)) << first.name;
      std::unique_ptr<RooWorkspace> ws1;
      ASSERT_NO_THROW(ws1 = importJSON(json1)) << first.name;
      {
         SCOPED_TRACE(std::string("after ") + first.name);
         const std::string d = diff(expected, natures(*ws1));
         EXPECT_TRUE(d.empty()) << d;
         EXPECT_EQ(globalSet(*ws1), expectedSet);
      }
      for (auto const &second : styles()) {
         SCOPED_TRACE(std::string(first.name) + " -> " + second.name);
         std::string json2;
         ASSERT_NO_THROW(json2 = exportWith(*ws1, second));
         // The same document as a direct export of the original in the second representation.
         std::string direct;
         ASSERT_NO_THROW(direct = exportWith(*original, second));
         EXPECT_EQ(layoutSummary(json2), layoutSummary(direct))
            << "the layout differs from a direct export as " << second.name;
         // Array distributions have no names for their components, so the documents are even identical.
         if (first.arrays && second.arrays) {
            EXPECT_TRUE(json2 == direct) << "the document differs from a direct export as " << second.name;
         }
         std::unique_ptr<RooWorkspace> ws2;
         ASSERT_NO_THROW(ws2 = importJSON(json2));
         const std::string d = diff(expected, natures(*ws2));
         EXPECT_TRUE(d.empty()) << d;
         EXPECT_EQ(globalSet(*ws2), expectedSet);
         std::string back;
         ASSERT_NO_THROW(back = exportWith(*ws2, first));
         EXPECT_EQ(layoutSummary(back), layoutSummary(json1))
            << "the export as " << first.name << " changed after a detour through " << second.name;
         // The likelihood after two hops.
         auto *mc2 = findModelConfig(*ws2, mcOrig->GetName());
         ASSERT_NE(mc2, nullptr);
         ASSERT_NE(ws2->data("obsData"), nullptr);
         const Scan scan = likelihoodScan(*ws2, *mc2, *ws2->data("obsData"), false);
         for (std::size_t i = 0; i < muValues.size(); ++i) {
            EXPECT_NEAR(scan.fixed[i], reference.fixed[i], 1e-9 * std::abs(reference.fixed[i])) << "mu=" << muValues[i];
            EXPECT_NEAR(scan.displaced[i], reference.displaced[i], 1e-9 * std::abs(reference.displaced[i]))
               << "mu=" << muValues[i];
         }
      }
   }
}

INSTANTIATE_TEST_SUITE_P(HS3, AuxMeasurementsChain,
                         testing::Values(ModelKind::Base, ModelKind::LogNormalNormFactor, ModelKind::LogNormalStandard),
                         [](testing::TestParamInfo<ModelKind> const &paramInfo) { return toString(paramInfo.param); });

} // namespace
