/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "HS3Document.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace RooFit {
namespace HS3 {
namespace {

std::string at(const char *section, std::size_t index, std::string const &name)
{
   return std::string(section) + "[" + std::to_string(index) + "](" + name + ")";
}

class Validator {
public:
   Validator(HS3Document const &doc, Diagnostics &diagnostics, ValidateOptions const &options)
      : _doc{doc}, _diagnostics{diagnostics}, _options{options}
   {
   }

   void run()
   {
      collectNames();
      checkFixedAxes();
      checkCategoricalAxes();
      checkLikelihoods();
      checkAnalyses();
      checkData();
      checkCombinedData();
      for (std::size_t i = 0; i < _doc.distributions.size(); ++i) {
         auto const &dist = _doc.distributions[i];
         if (auto const *histFactory = std::get_if<HistFactoryDist>(&dist.body)) {
            checkHistFactory(*histFactory, dist.name, at("distributions", i, dist.name));
         }
      }
      for (auto const &item : _doc.rooFit.combinedDistributions) {
         for (auto const &ref : item.second.distributions) {
            requireDistribution(ref.name, "misc.ROOT_internal.combined_distributions." + item.first, "distribution");
         }
      }
      for (auto const &item : _doc.rooFit.channelConstraints) {
         requireDistribution(item.first, "misc.ROOT_internal.histfactory_constraints", "distribution");
         for (auto const &ref : item.second) {
            requireDistribution(ref.name, "misc.ROOT_internal.histfactory_constraints." + item.first, "constraint");
         }
      }
   }

   void error(std::string const &path, std::string const &message)
   {
      _diagnostics.push_back({Diagnostic::Severity::Error, path, message});
   }

   void warning(std::string const &path, std::string const &message)
   {
      _diagnostics.push_back({Diagnostic::Severity::Warning, path, message});
   }

   // An error in documents that follow the current standard, which older versions of RooFit didn't care about.
   void strictError(std::string const &path, std::string const &message)
   {
      _diagnostics.push_back(
         {_options.strict ? Diagnostic::Severity::Error : Diagnostic::Severity::Warning, path, message});
   }

   void checkHistFactory(HistFactoryDist const &dist, std::string const &name, std::string const &path);

private:
   bool isExternal(std::string const &name) const { return _options.isExternal && _options.isExternal(name); }

   char const *sectionOf(std::string const &name) const
   {
      auto found = _sections.find(name);
      return found != _sections.end() ? found->second.front() : nullptr;
   }

   bool isDefinedIn(std::string const &name, char const *section) const
   {
      auto found = _sections.find(name);
      if (found == _sections.end()) {
         return false;
      }
      return std::any_of(found->second.begin(), found->second.end(),
                         [&](char const *s) { return std::string(s) == section; });
   }

   void requireDistribution(std::string const &name, std::string const &path, std::string const &what)
   {
      // RooFit refers to a RooSimultaneous by name, like the likelihood of a ModelConfig without data does.
      if (!isDefinedIn(name, "distributions") && !_doc.rooFit.combinedDistributions.count(name) && !isExternal(name)) {
         error(path, "the " + what + " '" + name + "' does not exist");
      }
   }

   void addParameter(std::string const &name) { _parameters.insert(name); }

   // The number of bins of an axis, after reporting an invalid binning.
   std::size_t checkAxis(BinnedAxis const &axis, std::string const &path);
   std::size_t checkAxis(CategoricalAxis const &axis, std::string const &path);
   template <class... Axis>
   std::size_t checkAxis(std::variant<Axis...> const &axis, std::string const &path)
   {
      return std::visit([&](auto const &a) { return checkAxis(a, path); }, axis);
   }
   // The number of bins of all axes, or nothing if it can't be stored.
   template <class Axes>
   std::optional<std::size_t> numBins(Axes const &axes, std::string const &path);

   void collectNames();
   void checkFixedAxes();
   void checkCategoricalAxes();
   void checkLikelihoods();
   void checkAnalyses();
   void checkData();
   void checkCombinedData();

   HS3Document const &_doc;
   Diagnostics &_diagnostics;
   ValidateOptions const &_options;
   std::map<std::string, std::vector<char const *>> _sections; ///< The sections where a name is defined.
   std::set<std::string> _parameters;
};

// HS3 requires the names of all objects to be unique within a document, including the parameters. RooFit writes such
// documents, e.g. for a range and a pdf of one name, so only ambiguous clashes in current documents are errors.
void Validator::collectNames()
{
   auto check = [&](char const *section, std::string const &name) {
      auto &sections = _sections[name];
      sections.push_back(section);
      if (sections.size() == 1) {
         return;
      }
      std::string const firstSection = sections.front();
      // RooFit used to give analyses and likelihoods the name of the pdf or the analysis. Accept this for backwards
      // compatibility: analyses are not referenced by name, and likelihoods only from analyses, so there is no
      // ambiguity.
      auto isLegacyNamed = [](std::string const &s) { return s == "likelihoods" || s == "analyses"; };
      // Within one section, and between distributions and functions (both can be referenced by name), a clash is
      // ambiguous. Other clashes are produced by RooFit itself (a range, snapshot or data set named like a pdf).
      auto isReferable = [](std::string const &s) { return s == "distributions" || s == "functions"; };
      const bool ambiguous = firstSection == section || (isReferable(firstSection) && isReferable(section));
      if (!_options.strict || !ambiguous || isLegacyNamed(firstSection) || isLegacyNamed(section)) {
         warning("", "the name '" + name + "' is used in \"" + firstSection + "\" and in \"" + section +
                        "\", which is not allowed by the HS3 standard.");
         return;
      }
      error("", "duplicate name '" + name + "': it is defined in \"" + firstSection + "\" and in \"" + section +
                   "\", but names must be unique within the document");
   };
   auto checkSection = [&](char const *section, auto const &objects) {
      for (auto const &object : objects) {
         check(section, object.name);
      }
   };
   checkSection("distributions", _doc.distributions);
   checkSection("functions", _doc.functions);
   checkSection("data", _doc.data);
   checkSection("domains", _doc.domains);
   checkSection("parameter_points", _doc.parameterPoints);
   checkSection("likelihoods", _doc.likelihoods);
   checkSection("analyses", _doc.analyses);

   // The parameters are the axes of the domains, the entries of the parameter points, and the parameters of
   // HistFactory modifiers, which are created on the fly.
   for (auto const &domain : _doc.domains) {
      for (auto const &axis : domain.axes) {
         addParameter(std::visit([](auto const &a) -> std::string const & { return a.name; }, axis));
      }
   }
   for (auto const &point : _doc.parameterPoints) {
      for (auto const &parameter : point.parameters) {
         addParameter(parameter.name);
      }
   }
   for (auto const &item : _doc.rooFit.combinedDistributions) {
      addParameter(item.second.indexCat);
   }
   for (auto const &dist : _doc.distributions) {
      auto const *histFactory = std::get_if<HistFactoryDist>(&dist.body);
      if (!histFactory) {
         continue;
      }
      for (auto const &sample : histFactory->samples) {
         for (auto const &modifier : sample.modifiers) {
            std::visit(overloaded{[&](NormFactor const &m) { addParameter(m.parameter.name); },
                                  [&](NormSys const &m) { addParameter(m.parameter.name); },
                                  [&](HistoSys const &m) { addParameter(m.parameter.name); },
                                  [&](ShapeSys const &m) {
                                     for (auto const &p : m.parameters) {
                                        addParameter(p.name);
                                     }
                                  },
                                  [&](StatError const &m) {
                                     for (auto const &p : m.parameters) {
                                        addParameter(p.name);
                                     }
                                  },
                                  [](CustomModifier const &) {}},
                       modifier);
         }
      }
   }
   for (std::string const &parameter : _parameters) {
      if (char const *section = sectionOf(parameter)) {
         strictError("", "the name '" + parameter + "' is used in \"" + section +
                            "\" and for a parameter, which is not allowed by the HS3 standard.");
      }
   }
}

// A fixed axis says that the parameter only takes a single value. No other domain can give it a range or another
// value, and no parameter point another value.
void Validator::checkFixedAxes()
{
   struct FixedAxis {
      double value = 0.0;
      std::string const *domain = nullptr;
   };
   std::map<std::string, FixedAxis> fixedAxes;
   auto forEachAxis = [&](auto &&fn) {
      for (std::size_t i = 0; i < _doc.domains.size(); ++i) {
         for (auto const &axis : _doc.domains[i].axes) {
            // A categorical axis is passed as null.
            fn(i, std::visit([](auto const &a) -> std::string const & { return a.name; }, axis),
               std::get_if<DomainAxis>(&axis));
         }
      }
   };
   forEachAxis([&](std::size_t i, std::string const &, DomainAxis const *continuous) {
      if (!continuous || !continuous->fixedValue) {
         return;
      }
      DomainAxis const &axis = *continuous;
      Domain const &domain = _doc.domains[i];
      if (!std::isfinite(*axis.fixedValue)) {
         error(at("domains", i, domain.name),
               "the single value of the parameter '" + axis.name + "' is not finite, which JSON can't represent");
      }
      auto inserted = fixedAxes.insert({axis.name, FixedAxis{*axis.fixedValue, &domain.name}});
      if (!inserted.second && inserted.first->second.value != *axis.fixedValue) {
         error(at("domains", i, domain.name),
               "the parameter '" + axis.name + "' has different single values in the domains '" +
                  *inserted.first->second.domain + "' and '" + domain.name + "'");
      }
   });
   if (fixedAxes.empty()) {
      return;
   }
   forEachAxis([&](std::size_t i, std::string const &name, DomainAxis const *continuous) {
      auto found = fixedAxes.find(name);
      if (found != fixedAxes.end() && !(continuous && continuous->fixedValue)) {
         Domain const &domain = _doc.domains[i];
         error(at("domains", i, domain.name),
               "the parameter '" + name + "' has the single value " + std::to_string(found->second.value) +
                  " in the domain '" + *found->second.domain + "', but " + (continuous ? "a range" : "categories") +
                  " in the domain '" + domain.name + "'");
      }
   });
   for (std::size_t i = 0; i < _doc.parameterPoints.size(); ++i) {
      ParameterPoint const &point = _doc.parameterPoints[i];
      for (ParameterValue const &parameter : point.parameters) {
         auto found = fixedAxes.find(parameter.name);
         double const *value = parameter.value ? std::get_if<double>(&*parameter.value) : nullptr;
         if (value && found != fixedAxes.end() && *value != found->second.value) {
            error(at("parameter_points", i, point.name),
                  "the parameter point '" + point.name + "' gives the parameter '" + parameter.name + "' the value " +
                     std::to_string(*value) + ", but its domain only has the value " +
                     std::to_string(found->second.value));
         }
      }
   }
}

// A categorical axis (HS3 #108) declares a category. All domains that list it must give it the same categories, and
// none may list it as continuous.
void Validator::checkCategoricalAxes()
{
   struct Declaration {
      std::vector<std::string> const *categories = nullptr;
      std::string const *domain = nullptr;
   };
   std::map<std::string, Declaration> categorical;
   for (std::size_t i = 0; i < _doc.domains.size(); ++i) {
      Domain const &domain = _doc.domains[i];
      for (auto const &axis : domain.axes) {
         auto const *cat = std::get_if<CategoricalAxis>(&axis);
         if (!cat) {
            continue;
         }
         auto inserted = categorical.insert({cat->name, Declaration{&cat->categories, &domain.name}});
         if (!inserted.second && *inserted.first->second.categories != cat->categories) {
            error(at("domains", i, domain.name), "the categorical axis '" + cat->name + "' is listed in the domains '" +
                                                    *inserted.first->second.domain + "' and '" + domain.name +
                                                    "' with different categories");
         }
      }
   }
   for (std::size_t i = 0; i < _doc.domains.size(); ++i) {
      Domain const &domain = _doc.domains[i];
      for (auto const &axis : domain.axes) {
         auto const *continuous = std::get_if<DomainAxis>(&axis);
         if (!continuous || continuous->fixedValue) {
            continue; // fixed axes are checked by checkFixedAxes()
         }
         auto found = categorical.find(continuous->name);
         if (found != categorical.end()) {
            error(at("domains", i, domain.name),
                  "the axis '" + continuous->name + "' is listed in a domain as categorical and as continuous");
         }
      }
   }
}

// The columns of combined data sets are categorical axes of the domains, or categories that exist already, and the keys
// consist of their categories. Every sub-data set is a data set of the document that belongs to one combined data set.
void Validator::checkCombinedData()
{
   std::map<std::string, std::vector<std::string> const *> categorical;
   for (Domain const &domain : _doc.domains) {
      for (auto const &axis : domain.axes) {
         if (auto const *cat = std::get_if<CategoricalAxis>(&axis)) {
            categorical.emplace(cat->name, &cat->categories);
         }
      }
   }
   std::set<std::string> combinedNames;
   for (auto const &data : _doc.data) {
      if (std::holds_alternative<CombinedData>(data.body)) {
         combinedNames.insert(data.name);
      }
   }
   std::set<std::string> usedComponents;
   for (std::size_t i = 0; i < _doc.data.size(); ++i) {
      auto const *combined = std::get_if<CombinedData>(&_doc.data[i].body);
      if (!combined) {
         continue;
      }
      std::string const path = at("data", i, _doc.data[i].name);
      std::string const context = "the combined data set '" + _doc.data[i].name + "'";
      std::vector<std::vector<std::string> const *> columnCategories;
      std::set<std::string> columns;
      for (std::string const &column : combined->columns) {
         auto found = categorical.find(column);
         if (found == categorical.end() && !isExternal(column)) {
            error(path,
                  "the column '" + column + "' of " + context + " is not declared by a categorical axis of a domain");
            return;
         }
         if (!columns.insert(column).second) {
            error(path, "the column '" + column + "' of " + context + " is listed twice");
            return;
         }
         columnCategories.push_back(found != categorical.end() ? found->second : nullptr);
      }
      std::set<std::vector<std::string>> keys;
      for (CombinedDataComponent const &component : combined->components) {
         for (std::size_t k = 0; k < component.key.size(); ++k) {
            auto const *categories = columnCategories[k];
            if (categories &&
                std::find(categories->begin(), categories->end(), component.key[k]) == categories->end()) {
               error(path, "the label '" + component.key[k] + "' in position " + std::to_string(k) + " of a key of " +
                              context + " is not one of the categories of '" + combined->columns[k] + "'");
            }
         }
         if (!keys.insert(component.key).second) {
            std::string key = "[";
            for (std::size_t k = 0; k < component.key.size(); ++k) {
               key += (k == 0 ? "\"" : ", \"") + component.key[k] + "\"";
            }
            error(path, "the key " + key + "] is used by two components of " + context);
         }
         std::string const &name = component.data.name;
         if (!isDefinedIn(name, "data") || combinedNames.count(name) || !usedComponents.insert(name).second) {
            error(path, "the component '" + name + "' of " + context +
                           " is not a data set of the document, or it is already a component of another combined "
                           "data set");
         }
      }
   }
}

void Validator::checkLikelihoods()
{
   for (std::size_t i = 0; i < _doc.likelihoods.size(); ++i) {
      auto const &likelihood = _doc.likelihoods[i];
      const std::string path = at("likelihoods", i, likelihood.name);
      for (auto const &term : likelihood.terms) {
         requireDistribution(term.distribution.name, path + ".distributions", "distribution");
         if (auto const *data = std::get_if<DataRef>(&term.data)) {
            if (!isDefinedIn(data->name, "data")) {
               error(path + ".data", "dataset '" + data->name + "' cannot be found!");
            }
         }
      }
      for (auto const &aux : likelihood.auxDistributions) {
         if (!isDefinedIn(aux.name, "distributions") && !isExternal(aux.name)) {
            strictError(path + ".aux_distributions", "the distribution '" + aux.name + "' does not exist");
         }
      }
   }
}

void Validator::checkAnalyses()
{
   for (std::size_t i = 0; i < _doc.analyses.size(); ++i) {
      auto const &analysis = _doc.analyses[i];
      const std::string path = at("analyses", i, analysis.name);
      if (!isDefinedIn(analysis.likelihood.name, "likelihoods")) {
         error(path + ".likelihood", "likelihood node not found!");
      }
      for (auto const &poi : analysis.parametersOfInterest) {
         if (char const *section = sectionOf(poi.name)) {
            error(path, "the parameter of interest '" + poi.name + "' of analysis '" + analysis.name +
                           "' is defined in \"" + section + "\", but it must be a parameter");
         } else if (!_parameters.count(poi.name) && !isExternal(poi.name)) {
            error(path, "the object '" + poi.name + "' in \"parameters_of_interest\" of analysis '" + analysis.name +
                           "' does not exist");
         }
      }
      for (auto const &domain : analysis.domains) {
         if (!isDefinedIn(domain.name, "domains")) {
            strictError(path + ".domains", "the domain '" + domain.name + "' does not exist");
         }
      }
      if (analysis.init && !isDefinedIn(analysis.init->name, "parameter_points")) {
         warning(path, "the parameter point '" + analysis.init->name + "' used as \"init\" of analysis '" +
                          analysis.name + "' does not exist.");
      }
      if (analysis.prior && !isDefinedIn(analysis.prior->name, "distributions") && !isExternal(analysis.prior->name)) {
         strictError(path + ".prior", "the distribution '" + analysis.prior->name + "' does not exist");
      }
   }
}

std::size_t Validator::checkAxis(BinnedAxis const &axis, std::string const &path)
{
   if (auto const *regular = std::get_if<RegularBinning>(&axis.binning)) {
      if (regular->nbins <= 0) {
         error(path, "the axis '" + axis.name + "' needs a positive number of bins");
         return 0;
      }
      if (!(axis.min < axis.max)) {
         error(path, "the axis '" + axis.name + "' needs \"min\" smaller than \"max\"");
      }
      return regular->nbins;
   }
   auto const &edges = std::get<EdgesBinning>(axis.binning).edges;
   return edges.empty() ? 0 : edges.size() - 1;
}

std::size_t Validator::checkAxis(CategoricalAxis const &axis, std::string const &path)
{
   if (axis.categories.empty()) {
      error(path, "the axis '" + axis.name + "' needs at least one category");
   }
   return axis.categories.size();
}

template <class Axes>
std::optional<std::size_t> Validator::numBins(Axes const &axes, std::string const &path)
{
   std::size_t nBins = 1;
   bool valid = true;
   for (std::size_t i = 0; i < axes.size(); ++i) {
      const std::size_t n = checkAxis(axes[i], path + ".axes[" + std::to_string(i) + "]");
      if (n == 0) {
         valid = false; // reported already
      } else if (nBins > std::numeric_limits<std::size_t>::max() / n) {
         error(path + ".axes", "the axes have more bins than can be stored");
         return std::nullopt;
      }
      nBins *= n;
   }
   return valid ? std::optional<std::size_t>{nBins} : std::nullopt;
}

void Validator::checkData()
{
   for (std::size_t i = 0; i < _doc.data.size(); ++i) {
      auto const &data = _doc.data[i];
      auto const *binned = std::get_if<BinnedData>(&data.body);
      if (!binned) {
         continue;
      }
      const std::string path = at("data", i, data.name);
      std::optional<std::size_t> nBinsOrNone = numBins(binned->axes, path);
      if (!nBinsOrNone) {
         continue;
      }
      const std::size_t nBins = *nBinsOrNone;
      if (binned->contents.size() != nBins) {
         error(path, "inconsistent bin numbers: contents=" + std::to_string(binned->contents.size()) +
                        ", bins=" + std::to_string(nBins));
      }
      if (binned->uncertainty) {
         auto const &sigma = binned->uncertainty->sigma;
         if (sigma.size() != binned->contents.size()) {
            error(path, "inconsistent number of errors and contents in '" + data.name +
                           "': errors=" + std::to_string(sigma.size()) +
                           ", contents=" + std::to_string(binned->contents.size()));
         }
         auto const &correlation = binned->uncertainty->correlation;
         if (!correlation.empty()) {
            bool square = correlation.size() == binned->contents.size();
            for (auto const &row : correlation) {
               square = square && row.size() == binned->contents.size();
            }
            if (!square) {
               error(path + ".uncertainty.correlation", "must be a square matrix of the size of \"contents\"");
            }
         }
      }
   }
}

void Validator::checkHistFactory(HistFactoryDist const &dist, std::string const &name, std::string const &path)
{
   if (dist.axes.empty()) {
      error(path + ".axes", "a HistFactory distribution needs at least one axis");
      return;
   }
   std::optional<std::size_t> nBinsOrNone = numBins(dist.axes, path);
   if (!nBinsOrNone) {
      return;
   }
   const std::size_t nBins = *nBinsOrNone;

   std::string channelName = name;
   if (channelName.rfind("model_", 0) == 0) {
      channelName.erase(0, 6);
   }

   auto checkSize = [&](std::string const &where, std::size_t size, char const *what) {
      if (size != nBins) {
         error(where, std::string(what) + " has " + std::to_string(size) + " entries, but the channel has " +
                         std::to_string(nBins) + " bins");
      }
   };

   // Two modifiers with the same parameter must have the same constraint.
   std::map<std::string, std::pair<Constraint, std::string>> constraintOfParameter;
   auto checkCorrelated = [&](std::string const &where, std::string const &parameter, Constraint const &constraint) {
      auto inserted = constraintOfParameter.emplace(parameter, std::make_pair(constraint, where));
      if (inserted.second) {
         return;
      }
      if (!(inserted.first->second.first == constraint)) {
         warning(where, "the parameter '" + parameter + "' has a different constraint than in " +
                           inserted.first->second.second +
                           ", but modifiers with the same parameter must share the constraint");
      }
   };

   auto checkConstraint = [&](std::string const &where, Constraint const &constraint) {
      if (auto const *pdf = std::get_if<ConstraintPdf>(&constraint)) {
         requireDistribution(pdf->pdf.name, where, "constraint");
      }
   };

   // The constraint distributions of the gamma parameters of shape modifiers.
   auto checkGammaConstraint = [&](std::string const &where, GammaConstraint const &constraint) {
      auto const *pdfs = std::get_if<GammaConstraintPdfs>(&constraint);
      if (!pdfs) {
         return;
      }
      checkSize(where + ".constraints", pdfs->pdfs.size(), "\"constraints\"");
      for (auto const &pdf : pdfs->pdfs) {
         if (pdf) {
            requireDistribution(pdf->name, where + ".constraints", "constraint");
         }
      }
   };

   auto checkParameter = [&](std::string const &where, std::string const &parameter, std::string const &modifier) {
      if (char const *section = sectionOf(parameter)) {
         error(where, "the parameter '" + parameter + "' of the modifier '" + modifier + "' is defined in \"" +
                         section + "\", but it must be a parameter");
      }
   };

   checkGammaConstraint(path + ".staterror", dist.statError.constraint);

   std::set<std::string> sampleNames;
   for (std::size_t iSample = 0; iSample < dist.samples.size(); ++iSample) {
      auto const &sample = dist.samples[iSample];
      const std::string samplePath = path + ".samples[" + std::to_string(iSample) + "](" + sample.name + ")";
      if (!sampleNames.insert(sample.name).second) {
         error(samplePath, "the name of the sample is not unique in the channel");
      }
      checkSize(samplePath + ".data.contents", sample.contents.size(), "\"contents\"");
      // These modifiers create objects that are named after them.
      std::set<std::string> objectNames;
      auto checkUniqueName = [&](std::string const &where, char const *type, std::string const &modifierName) {
         if (!objectNames.insert(std::string(type) + " " + modifierName).second) {
            error(where,
                  std::string("there is another ") + type + " modifier named '" + modifierName + "' in the sample");
         }
      };
      if (sample.errors && sample.errors->size() != sample.contents.size()) {
         error(samplePath + ".data", "inconsistent number of errors and contents in '" + sample.name +
                                        "': errors=" + std::to_string(sample.errors->size()) +
                                        ", contents=" + std::to_string(sample.contents.size()));
      }
      for (std::size_t iModifier = 0; iModifier < sample.modifiers.size(); ++iModifier) {
         std::string modifierPath = samplePath + ".modifiers[" + std::to_string(iModifier) + "]";
         auto checkInterpolation = [&](std::optional<Interpolation> const &interpolation, char const *type,
                                       std::string const &modifierName) {
            if (!interpolation && !dist.defaultInterpolation) {
               error(modifierPath, std::string("interpolation for ") + type + " modifier '" + modifierName +
                                      "' in sample '" + sample.name + "' of channel '" + channelName +
                                      "' is not given, and the channel has no \"default_interpolation\"");
            }
         };
         std::visit(overloaded{
                       [&](NormFactor const &m) {
                          modifierPath += "(" + m.name + ")";
                          checkParameter(modifierPath, m.parameter.name, m.name);
                          checkConstraint(modifierPath, m.constraint);
                          checkCorrelated(modifierPath, m.parameter.name, m.constraint);
                       },
                       [&](NormSys const &m) {
                          modifierPath += "(" + m.name + ")";
                          checkParameter(modifierPath, m.parameter.name, m.name);
                          checkConstraint(modifierPath, m.constraint);
                          checkCorrelated(modifierPath, m.parameter.name, m.constraint);
                          checkInterpolation(m.interpolation, "normsys", m.name);
                       },
                       [&](HistoSys const &m) {
                          modifierPath += "(" + m.name + ")";
                          checkUniqueName(modifierPath, "histosys", m.name);
                          checkParameter(modifierPath, m.parameter.name, m.name);
                          checkConstraint(modifierPath, m.constraint);
                          checkCorrelated(modifierPath, m.parameter.name, m.constraint);
                          checkInterpolation(m.interpolation, "histosys", m.name);
                          checkSize(modifierPath + ".data.hi.contents", m.hi.contents.size(), "\"contents\"");
                          checkSize(modifierPath + ".data.lo.contents", m.lo.contents.size(), "\"contents\"");
                       },
                       [&](ShapeSys const &m) {
                          modifierPath += "(" + m.name + ")";
                          checkUniqueName(modifierPath, "shapesys", m.name);
                          if (!m.vals.empty()) {
                             checkSize(modifierPath + ".data.vals", m.vals.size(), "\"vals\"");
                          }
                          if (!m.parameters.empty()) {
                             checkSize(modifierPath + ".parameters", m.parameters.size(), "\"parameters\"");
                          }
                          for (auto const &p : m.parameters) {
                             checkParameter(modifierPath, p.name, m.name);
                          }
                          checkGammaConstraint(modifierPath, m.constraint);
                       },
                       [&](StatError const &m) {
                          modifierPath += "(" + m.name + ")";
                          if (!m.parameters.empty()) {
                             checkSize(modifierPath + ".parameters", m.parameters.size(), "\"parameters\"");
                          }
                          checkGammaConstraint(modifierPath, m.constraint);
                       },
                       [&](CustomModifier const &m) {
                          modifierPath += "(" + m.name + ")";
                          if (!isDefinedIn(m.name, "functions") && !isDefinedIn(m.name, "distributions") &&
                              !isExternal(m.name)) {
                             error(modifierPath, "unable to find custom modifier '" + m.name + "'");
                          }
                       },
                    },
                    sample.modifiers[iModifier]);
      }
   }
}

} // namespace

void checkDocument(HS3Document const &doc, Diagnostics &diagnostics, ValidateOptions const &options)
{
   Validator{doc, diagnostics, options}.run();
}

std::optional<ValidatedHS3Document> validate(HS3Document doc, Diagnostics &diagnostics, ValidateOptions const &options)
{
   checkDocument(doc, diagnostics, options);
   if (hasErrors(diagnostics)) {
      return std::nullopt;
   }
   return ValidatedHS3Document{std::move(doc)};
}

void validateHistFactoryDist(HistFactoryDist const &dist, std::string const &name, Diagnostics &diagnostics)
{
   // Without a document, all names are resolved later in the workspace.
   HS3Document empty;
   ValidateOptions options;
   options.isExternal = [](std::string const &) { return true; };
   Validator{empty, diagnostics, options}.checkHistFactory(dist, name, "");
}

} // namespace HS3
} // namespace RooFit
