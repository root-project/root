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
#include <RooFitHS3/HistFactoryJSONTool.h>
#include <RooFit/Detail/JSONInterface.h>

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/Sample.h"

#include "Domains.h"

#include "TH1.h"

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace {

bool checkRegularBins(const TAxis &ax)
{
   double w = ax.GetXmax() - ax.GetXmin();
   double bw = w / ax.GetNbins();
   for (int i = 0; i <= ax.GetNbins(); ++i) {
      if (std::abs(ax.GetBinUpEdge(i) - (ax.GetXmin() + (bw * i))) > w * 1e-6)
         return false;
   }
   return true;
}

<<<<<<< HEAD
inline void writeAxis(JSONNode &bounds, const TAxis &ax)
{
   bool regular = (!ax.IsVariableBinSize()) || checkRegularBins(ax);
   if (regular) {
      bounds.set_map();
      bounds["nbins"] << ax.GetNbins();
      bounds["min"] << ax.GetXmin();
      bounds["max"] << ax.GetXmax();
   } else {
      bounds.set_seq();
      for (int i = 0; i <= ax.GetNbins(); ++i) {
         bounds.append_child() << ax.GetBinUpEdge(i);
      }
   }
}
||||||| 758899052b
   s.set_map();
=======
inline void writeAxis(JSONNode &axis, const TAxis &ax)
{
   bool regular = (!ax.IsVariableBinSize()) || checkRegularBins(ax);
   axis.set_map();
   if (regular) {
      axis["nbins"] << ax.GetNbins();
      axis["min"] << ax.GetXmin();
      axis["max"] << ax.GetXmax();
   } else {
      auto &bounds = axis["bounds"];
      bounds.set_seq();
      for (int i = 0; i <= ax.GetNbins(); ++i) {
         bounds.append_child() << ax.GetBinUpEdge(i);
      }
   }
}
>>>>>>> master

<<<<<<< HEAD
std::vector<std::string> getObsnames(RooStats::HistFactory::Channel const &c)
{
   std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};
   obsnames.resize(c.GetData().GetHisto()->GetDimension());
   return obsnames;
}

void writeObservables(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames)
{
   auto &observables = n["axes"];
   auto &x = RooJSONFactoryWSTool::appendNamedChild(observables, varnames[0]);
   writeAxis(x, *h.GetXaxis());
   if (h.GetDimension() > 1) {
      auto &y = RooJSONFactoryWSTool::appendNamedChild(observables, varnames[1]);
      writeAxis(y, *(h.GetYaxis()));
      if (h.GetDimension() > 2) {
         auto &z = RooJSONFactoryWSTool::appendNamedChild(observables, varnames[2]);
         writeAxis(z, *(h.GetZaxis()));
      }
   }
}

void exportHistogram(const TH1 &histo, JSONNode &node, const std::vector<std::string> &varnames,
                     const TH1 *errH = nullptr, bool doWriteObservables = true, bool writeErrors = true)
{
   node.set_map();
   auto &weights = node["contents"].set_seq();
   JSONNode *errors = nullptr;
   if (writeErrors) {
      errors = &node["errors"].set_seq();
   }
   if (doWriteObservables) {
      writeObservables(histo, node, varnames);
   }
   const int nBins = histo.GetNbinsX() * histo.GetNbinsY() * histo.GetNbinsZ();
   for (int i = 1; i <= nBins; ++i) {
      const double val = histo.GetBinContent(i);
      weights.append_child() << val;
      if (writeErrors) {
         const double err = errH ? val * errH->GetBinContent(i) : histo.GetBinError(i);
         errors->append_child() << err;
      }
   }
}

void exportSample(const RooStats::HistFactory::Sample &sample, JSONNode &channelNode,
                  std::vector<std::string> const &obsnames)
{
   auto &s = RooJSONFactoryWSTool::appendNamedChild(channelNode["samples"], sample.GetName());

   if (!sample.GetOverallSysList().empty()) {
      auto &modifiers = s["modifiers"];
||||||| 758899052b
   if (sample.GetOverallSysList().size() > 0) {
      auto &overallSys = s["overallSystematics"];
      overallSys.set_map();
=======
std::vector<std::string> getObsnames(RooStats::HistFactory::Channel const &c)
{
   std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};
   obsnames.resize(c.GetData().GetHisto()->GetDimension());
   return obsnames;
}

void writeObservables(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames)
{
   // axes need to be ordered, so this is a sequence and not a map
   auto &observables = n["axes"].set_seq();
   auto &x = observables.append_child().set_map();
   x["name"] << varnames[0];
   writeAxis(x, *h.GetXaxis());
   if (h.GetDimension() > 1) {
      auto &y = observables.append_child().set_map();
      y["name"] << varnames[1];
      writeAxis(y, *(h.GetYaxis()));
      if (h.GetDimension() > 2) {
         auto &z = observables.append_child().set_map();
         z["name"] << varnames[2];
         writeAxis(z, *(h.GetZaxis()));
      }
   }
}

void exportHistogram(const TH1 &histo, JSONNode &node, const std::vector<std::string> &varnames,
                     const TH1 *errH = nullptr, bool doWriteObservables = true, bool writeErrors = true)
{
   node.set_map();
   auto &weights = node["contents"].set_seq();
   JSONNode *errors = nullptr;
   if (writeErrors) {
      errors = &node["errors"].set_seq();
   }
   if (doWriteObservables) {
      writeObservables(histo, node, varnames);
   }
   const int nBins = histo.GetNbinsX() * histo.GetNbinsY() * histo.GetNbinsZ();
   for (int i = 1; i <= nBins; ++i) {
      const double val = histo.GetBinContent(i);
      weights.append_child() << val;
      if (writeErrors) {
         const double err = errH ? val * errH->GetBinContent(i) : histo.GetBinError(i);
         errors->append_child() << err;
      }
   }
}

void exportSample(const RooStats::HistFactory::Sample &sample, JSONNode &channelNode,
                  std::vector<std::string> const &obsnames)
{
   auto &s = RooJSONFactoryWSTool::appendNamedChild(channelNode["samples"], sample.GetName());

   if (!sample.GetOverallSysList().empty()) {
      auto &modifiers = s["modifiers"];
>>>>>>> master
      for (const auto &sys : sample.GetOverallSysList()) {
         auto &node = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.GetName());
         node["type"] << "normsys";
         auto &data = node["data"];
         data.set_map();
         data["lo"] << sys.GetLow();
         data["hi"] << sys.GetHigh();
      }
   }

   if (!sample.GetNormFactorList().empty()) {
      auto &modifiers = s["modifiers"];
      for (const auto &nf : sample.GetNormFactorList()) {
         RooJSONFactoryWSTool::appendNamedChild(modifiers, nf.GetName())["type"] << "normfactor";
      }
      auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, "Lumi");
      mod["type"] << "normfactor";
      mod["constraint_name"] << "lumiConstraint";
   }

   if (!sample.GetHistoSysList().empty()) {
      auto &modifiers = s["modifiers"];
      for (size_t i = 0; i < sample.GetHistoSysList().size(); ++i) {
         auto &sys = sample.GetHistoSysList()[i];
<<<<<<< HEAD
         auto &node = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.GetName());
         node["type"] << "histosys";
         auto &data = node["data"];
         data.set_map();
         exportHistogram(*(sys.GetHistoLow()), data["lo"], obsnames, nullptr, false);
         exportHistogram(*(sys.GetHistoHigh()), data["hi"], obsnames, nullptr, false);
||||||| 758899052b
         auto &node = histoSys[sys.GetName()];
         node.set_map();
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()), node["dataLow"], obsnames);
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()), node["dataHigh"], obsnames);
=======
         auto &node = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.GetName());
         node["type"] << "histosys";
         auto &data = node["data"].set_map();
         exportHistogram(*(sys.GetHistoLow()), data["lo"], obsnames, nullptr, false);
         exportHistogram(*(sys.GetHistoHigh()), data["hi"], obsnames, nullptr, false);
>>>>>>> master
      }
   }

   auto &tags = s["dict"].set_map();
   tags["normalizeByTheory"] << sample.GetNormalizeByTheory();

   if (sample.GetStatError().GetActivate()) {
      RooStats::HistFactory::JSONTool::activateStatError(s);
   }

   auto &data = s["data"];
<<<<<<< HEAD
   TH1 const *errH = sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto()
                        ? sample.GetStatError().GetErrorHist()
                        : nullptr;

   if (!channelNode.has_child("axes")) {
      writeObservables(*sample.GetHisto(), channelNode, obsnames);
   }
   exportHistogram(*sample.GetHisto(), data, obsnames, errH, false);
||||||| 758899052b
   RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(), data, obsnames,
                                         sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto()
                                            ? sample.GetStatError().GetErrorHist()
                                            : nullptr);
=======
   const bool useStatError = sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto();
   TH1 const *errH = useStatError ? sample.GetStatError().GetErrorHist() : nullptr;

   if (!channelNode.has_child("axes")) {
      writeObservables(*sample.GetHisto(), channelNode, obsnames);
   }
   exportHistogram(*sample.GetHisto(), data, obsnames, errH, false);
>>>>>>> master
}

void exportChannel(const RooStats::HistFactory::Channel &c, JSONNode &ch)
{
<<<<<<< HEAD
   ch.set_map();
   ch["name"] << "model_" + c.GetName();
   ch["type"] << "histfactory_dist";
||||||| 758899052b
   ch.set_map();
   ch["type"] << "histfactory";
=======
   ch["type"] << "histfactory_dist";
>>>>>>> master

   auto &staterr = ch["statError"].set_map();
   staterr["relThreshold"] << c.GetStatErrorConfig().GetRelErrorThreshold();
   staterr["constraint"] << RooStats::HistFactory::Constraint::Name(c.GetStatErrorConfig().GetConstraintType());

   const std::vector<std::string> obsnames = getObsnames(c);

   for (const auto &s : c.GetSamples()) {
      exportSample(s, ch, obsnames);
   }
}

void exportMeasurement(RooStats::HistFactory::Measurement &measurement, JSONNode &n,
                       RooFit::JSONIO::Detail::Domains &domains)
{
   using namespace RooStats::HistFactory;

   for (const auto &ch : measurement.GetChannels()) {
      if (!ch.CheckHistograms())
         throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
   }

   // collect information
   std::map<std::string, RooStats::HistFactory::Constraint::Type> constraints;
   std::map<std::string, NormFactor> normfactors;
   for (const auto &ch : measurement.GetChannels()) {
      for (const auto &s : ch.GetSamples()) {
         for (const auto &sys : s.GetOverallSysList()) {
            constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
         }
         for (const auto &sys : s.GetHistoSysList()) {
            constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
         }
         for (const auto &sys : s.GetShapeSysList()) {
            constraints[sys.GetName()] = sys.GetConstraintType();
         }
         for (const auto &norm : s.GetNormFactorList()) {
            normfactors[norm.GetName()] = norm;
         }
      }
   }

   // preprocess functions
   if (!measurement.GetFunctionObjects().empty()) {
      auto &funclist = n["functions"];
      for (const auto &func : measurement.GetFunctionObjects()) {
         auto &f = RooJSONFactoryWSTool::appendNamedChild(funclist, func.GetName());
         f["name"] << func.GetName();
         f["expression"] << func.GetExpression();
         f["dependents"] << func.GetDependents();
         f["command"] << func.GetCommand();
      }
   }

<<<<<<< HEAD
   auto &pdflist = n["distributions"];

   auto &analysisNode = RooJSONFactoryWSTool::appendNamedChild(n["analyses"], "simPdf");
   analysisNode.set_map();
   analysisNode["InterpolationScheme"] << measurement.GetInterpolationScheme();
   analysisNode["domains"].set_seq().append_child() << "default_domain";

   auto &analysisPois = analysisNode["pois"].set_seq();

   auto &analysisObservables = analysisNode["observables"].set_seq();

   for (const auto &poi : measurement.GetPOIList()) {
      analysisPois.append_child() << poi;
||||||| 758899052b
   // the simpdf
   auto &sim = pdflist[measurement.GetName()];
   sim.set_map();
   sim["type"] << "simultaneous";
   sim["index"] << "channelCat";
   auto &simdict = sim["dict"];
   simdict.set_map();
   simdict["InterpolationScheme"] << measurement.GetInterpolationScheme();
   auto &simtags = sim["tags"];
   simtags.set_seq();
   simtags.append_child() << "toplevel";
   auto &ch = sim["channels"];
   ch.set_map();
   for (const auto &c : measurement.GetChannels()) {
      auto pdfName = std::string("model_") + c.GetName();
      ch[c.GetName()] << pdfName;
      exportChannel(c, pdflist[pdfName]);
=======
   auto &pdflist = n["distributions"];

   auto &analysisNode = RooJSONFactoryWSTool::appendNamedChild(n["analyses"], "simPdf");
   analysisNode["domains"].set_seq().append_child() << "default_domain";

   auto &analysisPois = analysisNode["parameters_of_interest"].set_seq();

   for (const auto &poi : measurement.GetPOIList()) {
      analysisPois.append_child() << poi;
>>>>>>> master
   }

   analysisNode["likelihood"] << measurement.GetName();

   auto &likelihoodNode = RooJSONFactoryWSTool::appendNamedChild(n["likelihoods"], measurement.GetName());
   likelihoodNode["distributions"].set_seq();
   likelihoodNode["data"].set_seq();

   // the simpdf
   for (const auto &c : measurement.GetChannels()) {
<<<<<<< HEAD

      auto pdfName = std::string("model_") + c.GetName();

      likelihoodNode["distributions"].append_child() << pdfName;
      likelihoodNode["data"].append_child() << std::string("obsData_") + c.GetName();
      exportChannel(c, RooJSONFactoryWSTool::appendNamedChild(pdflist, pdfName));
   }

   struct VariableInfo {
      double val = 0.0;
      double minVal = -5.0;
      double maxVal = 5.0;
      bool isConstant = false;
   };
   std::unordered_map<std::string, VariableInfo> variables;

   for (const auto &channel : measurement.GetChannels()) {
      for (const auto &sample : channel.GetSamples()) {
         for (const auto &norm : sample.GetNormFactorList()) {
            auto &info = variables[norm.GetName()];
            info.val = norm.GetVal();
            info.minVal = norm.GetLow();
            info.maxVal = norm.GetHigh();
||||||| 758899052b
      for (const auto &s : c.GetSamples()) {
         for (const auto &norm : s.GetNormFactorList()) {
            if (!varlist.has_child(norm.GetName())) {
               auto &v = varlist[norm.GetName()];
               v.set_map();
               v["value"] << norm.GetVal();
               v["min"] << norm.GetLow();
               v["max"] << norm.GetHigh();
            }
=======

      auto pdfName = std::string("model_") + c.GetName();

      likelihoodNode["distributions"].append_child() << pdfName;
      likelihoodNode["data"].append_child() << std::string("obsData_") + c.GetName();
      exportChannel(c, RooJSONFactoryWSTool::appendNamedChild(pdflist, pdfName));
   }

   struct VariableInfo {
      double val = 0.0;
      double minVal = -5.0;
      double maxVal = 5.0;
      bool isConstant = false;
      bool writeDomain = true;
   };
   std::unordered_map<std::string, VariableInfo> variables;

   for (const auto &channel : measurement.GetChannels()) {
      for (const auto &sample : channel.GetSamples()) {
         for (const auto &norm : sample.GetNormFactorList()) {
            auto &info = variables[norm.GetName()];
            info.val = norm.GetVal();
            info.minVal = norm.GetLow();
            info.maxVal = norm.GetHigh();
>>>>>>> master
         }
         for (const auto &sys : sample.GetOverallSysList()) {
            variables[std::string("alpha_") + sys.GetName()] = VariableInfo{};
         }
      }
   }
   for (const auto &sys : measurement.GetConstantParams()) {
<<<<<<< HEAD
      variables[std::string("alpha_") + sys].isConstant = true;
||||||| 758899052b
      std::string parname = "alpha_" + sys;
      if (!varlist.has_child(parname)) {
         auto &v = varlist[parname];
         v.set_map();
      }
      varlist[parname]["const"] << true;
=======
      auto &info = variables[sys];
      info.isConstant = true;
      bool isGamma = sys.find("gamma_") != std::string::npos;
      // Gammas are 1.0 by default, alphas are 0.0
      info.val = isGamma ? 1.0 : 0.0;
      // For the gamma parameters, HistFactory will figure out the ranges
      // itself based on the template bin contents and errors.
      info.writeDomain = !isGamma;
>>>>>>> master
   }

<<<<<<< HEAD
   // the lumi variables
   {
      double nominal = measurement.GetLumi();
      double error = measurement.GetLumi() * measurement.GetLumiRelErr();

      auto &info1 = variables["Lumi"];
      info1.val = nominal;
      info1.minVal = 0;
      info1.maxVal = 10 * nominal;
      info1.isConstant = true;

      auto &info2 = variables["nominalLumi"];
      info2.val = nominal;
      info2.minVal = 0;
      info2.maxVal = nominal + 10 * error;
      info2.isConstant = true;
   }

   JSONNode &varlist = RooJSONFactoryWSTool::makeVariablesNode(n);
   for (auto const &item : variables) {
      std::string const &parname = item.first;
      VariableInfo const &info = item.second;

      auto &v = RooJSONFactoryWSTool::appendNamedChild(varlist, parname);
      v["value"] << info.val;
      if (info.isConstant)
         v["const"] << true;
      domains.readVariable(parname.c_str(), info.minVal, info.maxVal);
||||||| 758899052b
   for (const auto &poi : measurement.GetPOIList()) {
      if (!varlist[poi].has_child("tags")) {
         auto &tags = varlist[poi]["tags"];
         tags.set_seq();
      }
      varlist[poi]["tags"].append_child() << "poi";
=======
   // the lumi variables
   {
      double nominal = measurement.GetLumi();
      double error = measurement.GetLumi() * measurement.GetLumiRelErr();

      auto &info1 = variables["Lumi"];
      info1.val = nominal;
      info1.minVal = 0;
      info1.maxVal = 10 * nominal;
      info1.isConstant = true;

      auto &info2 = variables["nominalLumi"];
      info2.val = nominal;
      info2.minVal = 0;
      info2.maxVal = nominal + 10 * error;
      info2.isConstant = true;
   }

   JSONNode &varlist = RooJSONFactoryWSTool::makeVariablesNode(n);
   for (auto const &item : variables) {
      std::string const &parname = item.first;
      VariableInfo const &info = item.second;

      auto &v = RooJSONFactoryWSTool::appendNamedChild(varlist, parname);
      v["value"] << info.val;
      if (info.isConstant)
         v["const"] << true;
      if (info.writeDomain) {
         domains.readVariable(parname.c_str(), info.minVal, info.maxVal);
      }
>>>>>>> master
   }

   // the data
<<<<<<< HEAD
   auto &child1 = RooJSONFactoryWSTool::appendNamedChild(n.get("misc", "ROOT_internal", "combined_datas"), "obsData");
   auto &child2 =
      RooJSONFactoryWSTool::appendNamedChild(n.get("misc", "ROOT_internal", "combined_distributions"), "simPdf");
||||||| 758899052b
   auto &datalist = n["data"];
   datalist.set_map();
   auto &obsdata = datalist["obsData"];
   obsdata.set_map();
   obsdata["index"] << "channelCat";
   for (const auto &c : measurement.GetChannels()) {
      const std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};
=======
   auto &child1 = n.get("misc", "ROOT_internal", "combined_datas").set_map()["obsData"].set_map();
   auto &child2 = n.get("misc", "ROOT_internal", "combined_distributions").set_map()["simPdf"].set_map();
>>>>>>> master

<<<<<<< HEAD
   child1["index_cat"] << "channelCat";
   auto &labels1 = child1["labels"].set_seq();
   auto &indices1 = child1["indices"].set_seq();

   child2["index_cat"] << "channelCat";
   auto &labels2 = child2["labels"].set_seq();
   auto &indices2 = child2["indices"].set_seq();
   auto &pdfs2 = child2["distributions"].set_seq();

   std::vector<std::string> channelNames;
   for (const auto &c : measurement.GetChannels()) {
      labels1.append_child() << c.GetName();
      indices1.append_child() << int(channelNames.size());
      labels2.append_child() << c.GetName();
      indices2.append_child() << int(channelNames.size());
      pdfs2.append_child() << (std::string("model_") + c.GetName());

      JSONNode &dataOutput = RooJSONFactoryWSTool::appendNamedChild(n["data"], std::string("obsData_") + c.GetName());
      dataOutput["type"] << "binned";

      const std::vector<std::string> obsnames = getObsnames(c);

      for (auto const &obsname : obsnames) {
         analysisObservables.append_child() << obsname;
      }

      exportHistogram(*c.GetData().GetHisto(), dataOutput, obsnames);
      channelNames.push_back(c.GetName());
||||||| 758899052b
      auto &chdata = obsdata[c.GetName()];
      RooJSONFactoryWSTool::exportHistogram(*c.GetData().GetHisto(), chdata, obsnames);
=======
   child1["index_cat"] << "channelCat";
   auto &labels1 = child1["labels"].set_seq();
   auto &indices1 = child1["indices"].set_seq();

   child2["index_cat"] << "channelCat";
   auto &labels2 = child2["labels"].set_seq();
   auto &indices2 = child2["indices"].set_seq();
   auto &pdfs2 = child2["distributions"].set_seq();

   std::vector<std::string> channelNames;
   for (const auto &c : measurement.GetChannels()) {
      labels1.append_child() << c.GetName();
      indices1.append_child() << int(channelNames.size());
      labels2.append_child() << c.GetName();
      indices2.append_child() << int(channelNames.size());
      pdfs2.append_child() << (std::string("model_") + c.GetName());

      JSONNode &dataOutput = RooJSONFactoryWSTool::appendNamedChild(n["data"], std::string("obsData_") + c.GetName());
      dataOutput["type"] << "binned";

      exportHistogram(*c.GetData().GetHisto(), dataOutput, getObsnames(c));
      channelNames.push_back(c.GetName());
>>>>>>> master
   }
<<<<<<< HEAD

   RooJSONFactoryWSTool::writeCombinedDataName(n, "simPdf", "obsData");

   // Finally write lumi constraint
   auto &lumiConstraint = RooJSONFactoryWSTool::appendNamedChild(pdflist, "lumiConstraint");
   lumiConstraint["mean"] << "nominalLumi";
   lumiConstraint["name"] << "lumiConstraint";
   lumiConstraint["sigma"] << std::to_string(measurement.GetLumi() * measurement.GetLumiRelErr());
   lumiConstraint["type"] << "gaussian_dist";
   lumiConstraint["x"] << "Lumi";
||||||| 758899052b
=======

   auto &modelConfigAux = RooJSONFactoryWSTool::getRooFitInternal(n, "ModelConfigs", "simPdf").set_map();
   modelConfigAux["combined_data_name"] << "obsData";
   modelConfigAux["pdfName"] << "simPdf";
   modelConfigAux["mcName"] << "ModelConfig";

   // Finally write lumi constraint
   auto &lumiConstraint = RooJSONFactoryWSTool::appendNamedChild(pdflist, "lumiConstraint");
   lumiConstraint["mean"] << "nominalLumi";
   lumiConstraint["sigma"] << (measurement.GetLumi() * measurement.GetLumiRelErr());
   lumiConstraint["type"] << "gaussian_dist";
   lumiConstraint["x"] << "Lumi";
>>>>>>> master
}

} // namespace

void RooStats::HistFactory::JSONTool::PrintJSON(std::ostream &os)
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooJSONFactoryWSTool::createNewJSONTree();
   auto &n = tree->rootnode();
   RooFit::JSONIO::Detail::Domains domains;
   exportMeasurement(_measurement, n, domains);
   domains.writeJSON(n["domains"]);
   n.writeJSON(os);
}
void RooStats::HistFactory::JSONTool::PrintJSON(std::string const &filename)
{
   std::ofstream out(filename);
   this->PrintJSON(out);
}

void RooStats::HistFactory::JSONTool::PrintYAML(std::ostream &os)
{
<<<<<<< HEAD
   TRYMLTree p;
   auto &n = p.rootnode();
   n.set_map();
   RooFit::JSONIO::Detail::Domains domains;
   exportMeasurement(_measurement, n, domains);
   domains.writeJSON(n["domains"]);
||||||| 758899052b
   TRYMLTree p;
   auto &n = p.rootnode();
   n.set_map();
   exportMeasurement(_measurement, n);
=======
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooJSONFactoryWSTool::createNewJSONTree();
   auto &n = tree->rootnode().set_map();
   RooFit::JSONIO::Detail::Domains domains;
   exportMeasurement(_measurement, n, domains);
   domains.writeJSON(n["domains"]);
>>>>>>> master
   n.writeYML(os);
}

void RooStats::HistFactory::JSONTool::PrintYAML(std::string const &filename)
{
   std::ofstream out(filename);
   this->PrintYAML(out);
}
<<<<<<< HEAD

void RooStats::HistFactory::JSONTool::activateStatError(JSONNode &sampleNode)
{
   auto &node = sampleNode["modifiers"].set_seq().append_child().set_map();
   node["type"] << "staterror";
   node["name"] << "mcstat";
}
||||||| 758899052b
=======

void RooStats::HistFactory::JSONTool::activateStatError(JSONNode &sampleNode)
{
   auto &node = RooJSONFactoryWSTool::appendNamedChild(sampleNode["modifiers"], "mcstat");
   node["type"] << "staterror";
}
>>>>>>> master
