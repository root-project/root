/// \cond ROOFIT_INTERNAL

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

/** \class RooStats::HistFactory::JSONTool
 * \ingroup HistFactory
The RooStats::HistFactory::JSONTool can be used to export a HistFactory
measurement to HS3 JSON. It is not part of the public user interface, but a
pretty useful tool for unit test, validating if a measurement object can be
directly translated to HS3 without going over RooFit. If this translation turns
out to be important for users, it can be considered in the future to make the
class part of the public interface.
*/

#include "./JSONTool.h"

#include <RooFit/Detail/JSONInterface.h>

#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/Channel.h>
#include <RooStats/HistFactory/Sample.h>

#include <TH1.h>
#include <TROOT.h>

using RooFit::Detail::JSONNode;

namespace {

JSONNode &appendNamedChild(JSONNode &node, std::string const &name)
{
   static constexpr bool useListsInsteadOfDicts = true;

   if (!useListsInsteadOfDicts) {
      return node.set_map()[name].set_map();
   }
   JSONNode &child = node.set_seq().append_child().set_map();
   child["name"] << name;
   return child;
}

class Domains {
public:
   void readVariable(const char *name, double min, double max);

   void writeJSON(RooFit::Detail::JSONNode &) const;

private:
   class ProductDomain {
   public:
      void readVariable(const char *name, double min, double max);

      void writeJSON(RooFit::Detail::JSONNode &) const;

   private:
      struct ProductDomainElement {
         double min = 0.0;
         double max = 0.0;
      };

      std::map<std::string, ProductDomainElement> _map;
   };

   std::map<std::string, ProductDomain> _map;
};

void Domains::readVariable(const char *name, double min, double max)
{
   _map["default_domain"].readVariable(name, min, max);
}

void Domains::writeJSON(RooFit::Detail::JSONNode &node) const
{
   for (auto const &domain : _map) {
      domain.second.writeJSON(appendNamedChild(node, domain.first));
   }
}
void Domains::ProductDomain::readVariable(const char *name, double min, double max)
{
   auto &elem = _map[name];

   elem.min = min;
   elem.max = max;
}
void Domains::ProductDomain::writeJSON(RooFit::Detail::JSONNode &node) const
{
   node.set_map();
   node["type"] << "product_domain";

   auto &variablesNode = node["axes"];

   for (auto const &item : _map) {
      auto const &elem = item.second;
      RooFit::Detail::JSONNode &varnode = appendNamedChild(variablesNode, item.first);
      varnode["min"] << elem.min;
      varnode["max"] << elem.max;
   }
}

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

inline void writeAxis(JSONNode &axis, const TAxis &ax)
{
   bool regular = (!ax.IsVariableBinSize()) || checkRegularBins(ax);
   axis.set_map();
   if (regular) {
      axis["nbins"] << ax.GetNbins();
      axis["min"] << ax.GetXmin();
      axis["max"] << ax.GetXmax();
   } else {
      auto &edges = axis["edges"];
      edges.set_seq();
      for (int i = 0; i <= ax.GetNbins(); ++i) {
         edges.append_child() << ax.GetBinUpEdge(i);
      }
   }
}

std::vector<std::string> getObsnames(RooStats::HistFactory::Channel const &c)
{
   std::vector<std::string> obsnames{"obs_x_" + c.GetName(), "obs_y_" + c.GetName(), "obs_z_" + c.GetName()};
   obsnames.resize(c.GetData().GetHisto()->GetDimension());
   return obsnames;
}

void writeObservables(const TH1 &h, JSONNode &node, const std::vector<std::string> &varnames)
{
   // axes need to be ordered, so this is a sequence and not a map
   auto &observables = node["axes"].set_seq();
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

void exportSimpleHistogram(const TH1 &histo, JSONNode &node)
{
   node.set_seq();
   const int nBins = histo.GetNbinsX() * histo.GetNbinsY() * histo.GetNbinsZ();
   for (int i = 1; i <= nBins; ++i) {
      const double val = histo.GetBinContent(i);
      node.append_child() << val;
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
   auto &s = appendNamedChild(channelNode["samples"], sample.GetName());

   if (!sample.GetOverallSysList().empty()) {
      auto &modifiers = s["modifiers"].set_seq();
      for (const auto &sys : sample.GetOverallSysList()) {
         auto &node = modifiers.append_child().set_map();
         node["name"] << sys.GetName();
         node["type"] << "normsys";
         auto &data = node["data"];
         data.set_map();
         data["lo"] << sys.GetLow();
         data["hi"] << sys.GetHigh();
      }
   }

   if (!sample.GetNormFactorList().empty()) {
      auto &modifiers = s["modifiers"].set_seq();
      for (const auto &nf : sample.GetNormFactorList()) {
         auto &mod = modifiers.append_child().set_map();
         mod["name"] << nf.GetName();
         mod["type"] << "normfactor";
      }
      auto &mod = modifiers.append_child().set_map();
      mod["name"] << "Lumi";
      mod["type"] << "normfactor";
      mod["constraint_name"] << "lumiConstraint";
   }

   if (!sample.GetHistoSysList().empty()) {
      auto &modifiers = s["modifiers"].set_seq();
      for (size_t i = 0; i < sample.GetHistoSysList().size(); ++i) {
         auto &sys = sample.GetHistoSysList()[i];
         auto &node = modifiers.append_child().set_map();
         node["name"] << sys.GetName();
         node["type"] << "histosys";
         auto &data = node["data"].set_map();
         exportSimpleHistogram(*sys.GetHistoLow(), data["lo"].set_map()["contents"]);
         exportSimpleHistogram(*sys.GetHistoHigh(), data["hi"].set_map()["contents"]);
      }
   }

   if (!sample.GetShapeSysList().empty()) {
      auto &modifiers = s["modifiers"].set_seq();
      for (size_t i = 0; i < sample.GetShapeSysList().size(); ++i) {
         auto &sys = sample.GetShapeSysList()[i];
         auto &node = modifiers.append_child().set_map();
         node["name"] << sys.GetName();
         node["type"] << "shapesys";
         if (sys.GetConstraintType() == RooStats::HistFactory::Constraint::Gaussian)
            node["constraint"] << "Gauss";
         if (sys.GetConstraintType() == RooStats::HistFactory::Constraint::Poisson)
            node["constraint"] << "Poisson";
         auto &data = node["data"].set_map();
         exportSimpleHistogram(*sys.GetErrorHist(), data["vals"]);
      }
   }

   auto &tags = s["dict"].set_map();
   tags["normalizeByTheory"] << sample.GetNormalizeByTheory();

   if (sample.GetStatError().GetActivate()) {
      RooStats::HistFactory::JSONTool::activateStatError(s);
   }

   auto &data = s["data"];
   const bool useStatError = sample.GetStatError().GetActivate() && sample.GetStatError().GetUseHisto();
   TH1 const *errH = useStatError ? sample.GetStatError().GetErrorHist() : nullptr;

   if (!channelNode.has_child("axes")) {
      writeObservables(*sample.GetHisto(), channelNode, obsnames);
   }
   exportHistogram(*sample.GetHisto(), data, obsnames, errH, false);
}

void exportChannel(const RooStats::HistFactory::Channel &c, JSONNode &ch)
{
   ch["type"] << "histfactory_dist";

   auto &staterr = ch["statError"].set_map();
   staterr["relThreshold"] << c.GetStatErrorConfig().GetRelErrorThreshold();
   staterr["constraint"] << RooStats::HistFactory::Constraint::Name(c.GetStatErrorConfig().GetConstraintType());

   const std::vector<std::string> obsnames = getObsnames(c);

   for (const auto &s : c.GetSamples()) {
      exportSample(s, ch, obsnames);
   }
}

void setAttribute(JSONNode &rootnode, const std::string &obj, const std::string &attrib)
{
   auto node = &rootnode.get("misc", "ROOT_internal", "attributes").set_map()[obj].set_map();
   auto &tags = (*node)["tags"];
   tags.set_seq();
   tags.append_child() << attrib;
}

void exportMeasurement(RooStats::HistFactory::Measurement &measurement, JSONNode &rootnode, Domains &domains)
{
   using namespace RooStats::HistFactory;

   for (const auto &ch : measurement.GetChannels()) {
      if (!ch.CheckHistograms())
         throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
   }

   // preprocess functions
   if (!measurement.GetFunctionObjects().empty()) {
      auto &funclist = rootnode["functions"];
      for (const auto &func : measurement.GetFunctionObjects()) {
         auto &f = appendNamedChild(funclist, func.GetName());
         f["name"] << func.GetName();
         f["expression"] << func.GetExpression();
         f["dependents"] << func.GetDependents();
         f["command"] << func.GetCommand();
      }
   }

   auto &pdflist = rootnode["distributions"];

   auto &analysisNode = appendNamedChild(rootnode["analyses"], "simPdf");
   analysisNode["domains"].set_seq().append_child() << "default_domain";

   auto &analysisPois = analysisNode["parameters_of_interest"].set_seq();

   for (const auto &poi : measurement.GetPOIList()) {
      analysisPois.append_child() << poi;
   }

   analysisNode["likelihood"] << measurement.GetName();

   auto &likelihoodNode = appendNamedChild(rootnode["likelihoods"], measurement.GetName());
   likelihoodNode["distributions"].set_seq();
   likelihoodNode["data"].set_seq();

   // the simpdf
   for (const auto &c : measurement.GetChannels()) {

      auto pdfName = std::string("model_") + c.GetName();
      auto realSumPdfName = c.GetName() + std::string("_model");

      likelihoodNode["distributions"].append_child() << pdfName;
      likelihoodNode["data"].append_child() << std::string("obsData_") + c.GetName();
      exportChannel(c, appendNamedChild(pdflist, pdfName));
      setAttribute(rootnode, realSumPdfName, "BinnedLikelihood");
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
         }
         for (const auto &sys : sample.GetOverallSysList()) {
            variables[std::string("alpha_") + sys.GetName()] = VariableInfo{};
         }
      }
   }
   for (const auto &sys : measurement.GetConstantParams()) {
      auto &info = variables[sys];
      info.isConstant = true;
      bool isGamma = sys.find("gamma_") != std::string::npos;
      // Gammas are 1.0 by default, alphas are 0.0
      info.val = isGamma ? 1.0 : 0.0;
      // For the gamma parameters, HistFactory will figure out the ranges
      // itself based on the template bin contents and errors.
      info.writeDomain = !isGamma;
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

   JSONNode &varlist = appendNamedChild(rootnode["parameter_points"], "default_values")["parameters"];
   for (auto const &item : variables) {
      std::string const &parname = item.first;
      VariableInfo const &info = item.second;

      auto &v = appendNamedChild(varlist, parname);
      v["value"] << info.val;
      if (info.isConstant)
         v["const"] << true;
      if (info.writeDomain) {
         domains.readVariable(parname.c_str(), info.minVal, info.maxVal);
      }
   }

   // the data
   auto &child1 = rootnode.get("misc", "ROOT_internal", "combined_datasets").set_map()["obsData"].set_map();
   auto &child2 = rootnode.get("misc", "ROOT_internal", "combined_distributions").set_map()["simPdf"].set_map();

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

      JSONNode &dataOutput = appendNamedChild(rootnode["data"], std::string("obsData_") + c.GetName());
      dataOutput["type"] << "binned";

      exportHistogram(*c.GetData().GetHisto(), dataOutput, getObsnames(c));
      channelNames.push_back(c.GetName());
   }

   auto &modelConfigAux = rootnode.get("misc", "ROOT_internal", "ModelConfigs", "simPdf").set_map();
   modelConfigAux["combined_data_name"] << "obsData";
   modelConfigAux["pdfName"] << "simPdf";
   modelConfigAux["mcName"] << "ModelConfig";

   // Finally write lumi constraint
   auto &lumiConstraint = appendNamedChild(pdflist, "lumiConstraint");
   lumiConstraint["mean"] << "nominalLumi";
   lumiConstraint["sigma"] << (measurement.GetLumi() * measurement.GetLumiRelErr());
   lumiConstraint["type"] << "gaussian_dist";
   lumiConstraint["x"] << "Lumi";
}

std::unique_ptr<RooFit::Detail::JSONTree> createNewJSONTree()
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create();
   JSONNode &n = tree->rootnode();
   n.set_map();
   auto &metadata = n["metadata"].set_map();

   // add the mandatory hs3 version number
   metadata["hs3_version"] << "0.1.90";

   // Add information about the ROOT version that was used to generate this file
   auto &rootInfo = appendNamedChild(metadata["packages"], "ROOT");
   std::string versionName = gROOT->GetVersion();
   // We want to consistently use dots such that the version name can be easily
   // digested automatically.
   std::replace(versionName.begin(), versionName.end(), '/', '.');
   rootInfo["version"] << versionName;

   return tree;
}

} // namespace

void RooStats::HistFactory::JSONTool::PrintJSON(std::ostream &os)
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = createNewJSONTree();
   auto &rootnode = tree->rootnode();
   Domains domains;
   exportMeasurement(_measurement, rootnode, domains);
   domains.writeJSON(rootnode["domains"]);
   rootnode.writeJSON(os);
}
void RooStats::HistFactory::JSONTool::PrintJSON(std::string const &filename)
{
   std::ofstream out(filename);
   this->PrintJSON(out);
}

void RooStats::HistFactory::JSONTool::PrintYAML(std::ostream &os)
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = createNewJSONTree();
   auto &rootnode = tree->rootnode().set_map();
   Domains domains;
   exportMeasurement(_measurement, rootnode, domains);
   domains.writeJSON(rootnode["domains"]);
   rootnode.writeYML(os);
}

void RooStats::HistFactory::JSONTool::PrintYAML(std::string const &filename)
{
   std::ofstream out(filename);
   this->PrintYAML(out);
}

void RooStats::HistFactory::JSONTool::activateStatError(JSONNode &sampleNode)
{
   auto &node = sampleNode["modifiers"].set_seq().append_child().set_map();
   node["name"] << "mcstat";
   node["type"] << "staterror";
}

/// \endcond
