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

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/Sample.h"

#ifdef ROOFIT_HS3_WITH_RYML
#include "RYMLParser.h"
typedef TRYMLTree tree_t;
#else
#include "JSONParser.h"
typedef TJSONTree tree_t;
#endif

using RooFit::Experimental::JSONNode;

namespace {

void exportSample(const RooStats::HistFactory::Sample &sample, JSONNode &s)
{
   const std::vector<std::string> obsnames{"obs_x_" + sample.GetChannelName(), "obs_y_" + sample.GetChannelName(),
                                           "obs_z_" + sample.GetChannelName()};

   s.set_map();
   s["type"] << "hist-sample";

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
      auto &normFactors = s["normFactors"];
      normFactors.set_seq();
      for (auto &sys : sample.GetNormFactorList()) {
         normFactors.append_child() << sys.GetName();
      }
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

void exportMeasurement(RooStats::HistFactory::Measurement &measurement, JSONNode &n)
{
   using namespace RooStats::HistFactory;

   for (const auto &ch : measurement.GetChannels()) {
      if (!ch.CheckHistograms())
         throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
   }

   auto &pdflist = n["pdfs"];
   pdflist.set_map();

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
      auto &thisch = ch[c.GetName()];
      exportChannel(c, thisch);
   }

   // the variables
   auto &varlist = n["variables"];
   varlist.set_map();
   for (const auto &c : measurement.GetChannels()) {
      for (const auto &s : c.GetSamples()) {
         for (const auto &norm : s.GetNormFactorList()) {
            if (!varlist.has_child(norm.GetName())) {
               auto &v = varlist[norm.GetName()];
               v.set_map();
               v["value"] << norm.GetVal();
               v["min"] << norm.GetLow();
               v["max"] << norm.GetHigh();
               if (norm.GetConst()) {
                  v["const"] << true;
               }
            }
         }
         for (const auto &sys : s.GetOverallSysList()) {
            std::string parname("alpha_");
            parname += sys.GetName();
            if (!varlist.has_child(parname)) {
               auto &v = varlist[parname];
               v.set_map();
               v["value"] << 0.;
               v["min"] << -5.;
               v["max"] << 5.;
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

   for (const auto &poi : measurement.GetPOIList()) {
      if (!varlist[poi].has_child("tags")) {
         auto &tags = varlist[poi]["tags"];
         tags.set_seq();
      }
      varlist[poi]["tags"].append_child() << "poi";
   }

   // the data
   auto &datalist = n["data"];
   datalist.set_map();
   auto &obsdata = datalist["obsData"];
   obsdata.set_map();
   obsdata["index"] << "channelCat";
   for (const auto &c : measurement.GetChannels()) {
      const std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};

      auto &chdata = obsdata[c.GetName()];
      RooJSONFactoryWSTool::exportHistogram(*c.GetData().GetHisto(), chdata, obsnames);
   }
}

} // namespace

void RooStats::HistFactory::JSONTool::PrintJSON(std::ostream &os)
{
   tree_t p;
   auto &n = p.rootnode();
   n.set_map();
   exportMeasurement(_measurement, n);
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
   exportMeasurement(_measurement, n);
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
