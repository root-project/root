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

   if (sample.GetOverallSysList().size() > 0) {
      auto &overallSys = s["overallSystematics"];
      overallSys.set_map();
      for (const auto &sys : sample.GetOverallSysList()) {
         auto &node = overallSys[sys.GetName()];
         node.set_map();
         node["low"] << sys.GetLow();
         node["high"] << sys.GetHigh();
      }
   }

   if (sample.GetNormFactorList().size() > 0) {
      s["normFactors"].fill_seq(sample.GetNormFactorList(), [](auto const &x) { return x.GetName(); });
   }

   if (sample.GetHistoSysList().size() > 0) {
      auto &histoSys = s["histogramSystematics"];
      histoSys.set_map();
      for (size_t i = 0; i < sample.GetHistoSysList().size(); ++i) {
         auto &sys = sample.GetHistoSysList()[i];
         auto &node = histoSys[sys.GetName()];
         node.set_map();
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()), node["dataLow"], obsnames);
         RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()), node["dataHigh"], obsnames);
      }
   }

   auto &tags = s["dict"];
   tags.set_map();
   tags["normalizeByTheory"] << sample.GetNormalizeByTheory();

   s["statError"] << sample.GetStatError().GetActivate();

   auto &data = s["data"];
   RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(), data, obsnames,
                                         sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto()
                                            ? sample.GetStatError().GetErrorHist()
                                            : nullptr);
}

void exportChannel(const RooStats::HistFactory::Channel &c, JSONNode &ch)
{
   ch.set_map();
   ch["type"] << "histfactory";

   auto &staterr = ch["statError"];
   staterr.set_map();
   staterr["relThreshold"] << c.GetStatErrorConfig().GetRelErrorThreshold();
   staterr["constraint"] << RooStats::HistFactory::Constraint::Name(c.GetStatErrorConfig().GetConstraintType());

   auto &samples = ch["samples"];
   samples.set_map();
   for (const auto &s : c.GetSamples()) {
      auto &sample = samples[s.GetName()];
      exportSample(s, sample);
      auto &ns = sample["namespaces"];
      ns.set_seq();
      ns.append_child() << c.GetName();
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
      funclist.set_map();
      for (const auto &func : measurement.GetFunctionObjects()) {
         auto &f = funclist[func.GetName()];
         f.set_map();
         f["name"] << func.GetName();
         f["expression"] << func.GetExpression();
         f["dependents"] << func.GetDependents();
         f["command"] << func.GetCommand();
      }
   }

   auto &pdflist = n["distributions"];
   pdflist.set_map();

   auto &likelihoodlist = n["likelihoods"];
   likelihoodlist.set_map();

   auto &analysislist = n["analyses"];
   analysislist.set_map();

   auto &analysisNode = analysislist[measurement.GetName()];
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

   auto &likelihoods = analysisNode["likelihoods"];
   likelihoods.set_seq();

   // the simpdf
   for (const auto &c : measurement.GetChannels()) {

      auto &likelihoodNode = likelihoodlist[c.GetName()];
      auto pdfName = std::string("model_") + c.GetName();
      likelihoodNode.set_map();

      likelihoodNode["dist"] << pdfName;
      likelihoodNode["obs"] << std::string("obsData_") + c.GetName();
      likelihoods.append_child() << c.GetName();
      exportChannel(c, pdflist[pdfName]);
   }

   // the variables
   JSONNode &varlist = RooJSONFactoryWSTool::makeVariablesNode(n);
   for (const auto &channel : measurement.GetChannels()) {
      for (const auto &sample : channel.GetSamples()) {
         for (const auto &norm : sample.GetNormFactorList()) {
            if (!varlist.has_child(norm.GetName())) {
               auto &v = varlist[norm.GetName()];
               v.set_map();
               v["value"] << norm.GetVal();
               domains.readVariable(norm.GetName().c_str(), norm.GetLow(), norm.GetHigh());
            }
         }
         for (const auto &sys : sample.GetOverallSysList()) {
            std::string parname("alpha_");
            parname += sys.GetName();
            if (!varlist.has_child(parname)) {
               auto &v = varlist[parname];
               v.set_map();
               v["value"] << 0.;
               domains.readVariable(parname.c_str(), -5., 5.);
            }
         }
      }
   }
   for (const auto &sys : measurement.GetConstantParams()) {
      std::string parname = "alpha_" + sys;
      if (!varlist.has_child(parname)) {
         auto &v = varlist[parname];
         v.set_map();
      }
      varlist[parname]["const"] << true;
   }

   // the data
   auto &datalist = n["data"];
   datalist.set_map();

   for (const auto &c : measurement.GetChannels()) {
      const std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};

      for (int i = 0; i < c.GetData().GetHisto()->GetDimension(); ++i) {
         analysisObservables.append_child() << obsnames[i];
      }

      RooJSONFactoryWSTool::exportHistogram(*c.GetData().GetHisto(), datalist[std::string("obsData_") + c.GetName()],
                                            obsnames);
   }
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
