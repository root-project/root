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

void exportSample(const RooStats::HistFactory::Sample &sample, JSONNode &s)
{
   const std::vector<std::string> obsnames{"obs_x_" + sample.GetChannelName(), "obs_y_" + sample.GetChannelName(),
                                           "obs_z_" + sample.GetChannelName()};

   s.set_map();
   s["name"] << sample.GetName();

   if (!sample.GetOverallSysList().empty()) {
      auto &modifiers = s["modifiers"];
      modifiers.set_seq();
      for (const auto &sys : sample.GetOverallSysList()) {
         auto &node = modifiers.append_child();
         node.set_map();
         node["type"] << "normsys";
         node["name"] << sys.GetName();
         auto &data = node["data"];
         data.set_map();
         data["lo"] << sys.GetLow();
         data["hi"] << sys.GetHigh();
      }
   }

   if (!sample.GetNormFactorList().empty()) {
      auto &modifiers = s["modifiers"];
      modifiers.set_seq();
      for (const auto &nf : sample.GetNormFactorList()) {
         auto &node = modifiers.append_child();
         node.set_map();
         node["type"] << "normfactor";
         node["name"] << nf.GetName();
      }
   }

   if (!sample.GetHistoSysList().empty()) {
      auto &modifiers = s["modifiers"];
      modifiers.set_seq();
      for (size_t i = 0; i < sample.GetHistoSysList().size(); ++i) {
         auto &sys = sample.GetHistoSysList()[i];
         auto &node = modifiers.append_child();
         node.set_map();
         node["type"] << "histosys";
         node["name"] << sys.GetName();
         auto &data = node["data"];
         data.set_map();
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()), data["lo"], obsnames);
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()), data["hi"], obsnames);
      }
   }

   auto &tags = s["dict"];
   tags.set_map();
   tags["normalizeByTheory"] << sample.GetNormalizeByTheory();

   if (sample.GetStatError().GetActivate()) {
      RooStats::HistFactory::JSONTool::activateStatError(s);
   }

   auto &data = s["data"];
   RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(), data, obsnames,
                                         sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto()
                                            ? sample.GetStatError().GetErrorHist()
                                            : nullptr);
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

   auto &samples = ch["samples"];
   samples.set_seq();
   for (const auto &s : c.GetSamples()) {
      auto &sample = samples.append_child();
      exportSample(s, sample);
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

   auto &analysisNode = RooJSONFactoryWSTool::appendNamedChild(n["analyses"], measurement.GetName());
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
   for (const auto &c : measurement.GetChannels()) {
      JSONNode &dataOutput = RooJSONFactoryWSTool::appendNamedChild(n["data"], std::string("obsData_") + c.GetName());

      const std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};

      for (int i = 0; i < c.GetData().GetHisto()->GetDimension(); ++i) {
         analysisObservables.append_child() << obsnames[i];
      }

      RooJSONFactoryWSTool::exportHistogram(*c.GetData().GetHisto(), dataOutput, obsnames);
   }

   RooJSONFactoryWSTool::writeCombinedDataName(n, measurement.GetName(), "obsData");
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

#ifdef ROOFIT_HS3_WITH_RYML
void RooStats::HistFactory::JSONTool::PrintYAML(std::ostream &os)
{
   TRYMLTree p;
   auto &n = p.rootnode();
   n.set_map();
   RooFit::JSONIO::Detail::Domains domains;
   exportMeasurement(_measurement, n, domains);
   domains.writeJSON(n["domains"]);
   n.writeYML(os);
}
#else
void RooStats::HistFactory::JSONTool::PrintYAML(std::ostream & /*os*/)
{
   std::cerr << "YAML export only support with rapidyaml!" << std::endl;
}
#endif

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
