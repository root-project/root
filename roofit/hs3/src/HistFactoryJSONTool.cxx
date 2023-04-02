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

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace {

std::vector<std::string> getObsnames(RooStats::HistFactory::Channel const &c)
{
   std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};
   obsnames.resize(c.GetData().GetHisto()->GetDimension());
   return obsnames;
}

void exportSample(const RooStats::HistFactory::Sample &sample, JSONNode &channelNode,
                  std::vector<std::string> const &obsnames)
{
   auto &s = RooJSONFactoryWSTool::appendNamedChild(channelNode["samples"], sample.GetName());

   if (!sample.GetOverallSysList().empty()) {
      auto &modifiers = s["modifiers"];
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
         auto &node = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.GetName());
         node["type"] << "histosys";
         auto &data = node["data"];
         data.set_map();
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()), data["lo"], obsnames, nullptr, false);
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()), data["hi"], obsnames, nullptr, false);
      }
   }

   auto &tags = s["dict"];
   tags.set_map();
   tags["normalizeByTheory"] << sample.GetNormalizeByTheory();

   if (sample.GetStatError().GetActivate()) {
      RooStats::HistFactory::JSONTool::activateStatError(s);
   }

   auto &data = s["data"];
   TH1 const *errH = sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto()
                        ? sample.GetStatError().GetErrorHist()
                        : nullptr;

   if (!channelNode.has_child("axes")) {
      RooJSONFactoryWSTool::writeObservables(*sample.GetHisto(), channelNode, obsnames);
   }
   RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(), data, obsnames, errH, false);
}

void exportChannel(const RooStats::HistFactory::Channel &c, JSONNode &ch)
{
   ch.set_map();
   ch["name"] << "model_" + c.GetName();
   ch["type"] << "histfactory_dist";

   auto &staterr = ch["statError"];
   staterr.set_map();
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

   auto &pdflist = n["distributions"];

   auto &analysisNode = RooJSONFactoryWSTool::appendNamedChild(n["analyses"], "simPdf");
   analysisNode.set_map();
   analysisNode["InterpolationScheme"] << measurement.GetInterpolationScheme();
   auto &analysisDomains = analysisNode["domains"];
   analysisDomains.set_seq();
   analysisDomains.append_child() << "default_domain";

   auto &analysisPois = analysisNode["pois"];
   analysisPois.set_seq();

   auto &analysisObservables = analysisNode["observables"];
   analysisObservables.set_seq();

   for (const auto &poi : measurement.GetPOIList()) {
      analysisPois.append_child() << poi;
   }

   analysisNode["likelihood"] << measurement.GetName();

   auto &likelihoodNode = RooJSONFactoryWSTool::appendNamedChild(n["likelihoods"], measurement.GetName());
   likelihoodNode["distributions"].set_seq();
   likelihoodNode["data"].set_seq();

   // the simpdf
   for (const auto &c : measurement.GetChannels()) {

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
         }
         for (const auto &sys : sample.GetOverallSysList()) {
            variables[std::string("alpha_") + sys.GetName()] = VariableInfo{};
         }
      }
   }
   for (const auto &sys : measurement.GetConstantParams()) {
      variables[std::string("alpha_") + sys].isConstant = true;
   }

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
   }

   // the data
   auto &child1 = RooJSONFactoryWSTool::appendNamedChild(n.get("misc", "ROOT_internal", "combined_datas"), "obsData");
   auto &child2 =
      RooJSONFactoryWSTool::appendNamedChild(n.get("misc", "ROOT_internal", "combined_distributions"), "simPdf");

   child1["index_cat"] << "channelCat";
   auto &labels1 = child1["labels"];
   labels1.set_seq();
   auto &indices1 = child1["indices"];
   indices1.set_seq();

   child2["index_cat"] << "channelCat";
   auto &labels2 = child2["labels"];
   labels2.set_seq();
   auto &indices2 = child2["indices"];
   indices2.set_seq();
   auto &pdfs2 = child2["distributions"];
   pdfs2.set_seq();

   std::vector<std::string> channelNames;
   for (const auto &c : measurement.GetChannels()) {
      labels1.append_child() << c.GetName();
      indices1.append_child() << int(channelNames.size());
      labels2.append_child() << c.GetName();
      indices2.append_child() << int(channelNames.size());
      pdfs2.append_child() << (std::string("model_") + c.GetName());

      JSONNode &dataOutput = RooJSONFactoryWSTool::appendNamedChild(n["data"], std::string("obsData_") + c.GetName());

      const std::vector<std::string> obsnames = getObsnames(c);

      for (auto const &obsname : obsnames) {
         analysisObservables.append_child() << obsname;
      }

      RooJSONFactoryWSTool::exportHistogram(*c.GetData().GetHisto(), dataOutput, obsnames);
      channelNames.push_back(c.GetName());
   }

   RooJSONFactoryWSTool::writeCombinedDataName(n, "simPdf", "obsData");

   // Finally write lumi constraint
   auto &lumiConstraint = RooJSONFactoryWSTool::appendNamedChild(pdflist, "lumiConstraint");
   lumiConstraint["mean"] << "nominalLumi";
   lumiConstraint["name"] << "lumiConstraint";
   lumiConstraint["sigma"] << std::to_string(measurement.GetLumi() * measurement.GetLumiRelErr());
   lumiConstraint["type"] << "gaussian_dist";
   lumiConstraint["x"] << "Lumi";
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
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooJSONFactoryWSTool::createNewJSONTree();
   auto &n = tree->rootnode();
   n.set_map();
   RooFit::JSONIO::Detail::Domains domains;
   exportMeasurement(_measurement, n, domains);
   domains.writeJSON(n["domains"]);
   n.writeYML(os);
}

void RooStats::HistFactory::JSONTool::PrintYAML(std::string const &filename)
{
   std::ofstream out(filename);
   this->PrintYAML(out);
}

void RooStats::HistFactory::JSONTool::activateStatError(JSONNode &sampleNode)
{
   auto &modifiers = sampleNode["modifiers"];
   modifiers.set_seq();
   auto &node = modifiers.append_child();
   node.set_map();
   node["type"] << "staterror";
   node["name"] << "mcstat";
}
