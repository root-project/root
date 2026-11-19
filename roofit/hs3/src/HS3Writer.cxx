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

// Writes the typed representation of HS3 documents. The keys are the same as in HS3Parser.cxx, which the round-trip
// tests in testHS3Document.cxx check.

#include "HS3Document.h"

#include <RooFit/Detail/JSONInterface.h>

#include <algorithm>
#include <cmath>
#include <limits>

using RooFit::Detail::JSONNode;

namespace RooFit {
namespace HS3 {
namespace {

// The counterparts of the readValue() functions of the parser, for the values that are written in one way only.
template <class T>
void writeValue(JSONNode &node, T const &value)
{
   node << value;
}

template <class Tag>
void writeValue(JSONNode &node, Ref<Tag> const &ref)
{
   node << ref.name;
}

void writeValue(JSONNode &node, NameOrNumber const &value)
{
   std::visit([&](auto const &v) { node << v; }, value);
}

template <class T>
void writeValue(JSONNode &node, std::vector<T> const &values)
{
   node.set_seq();
   for (auto const &value : values) {
      writeValue(node.append_child(), value);
   }
}

// A component that is left out if it has no value.
template <class T>
void writeOptional(JSONNode &node, const char *key, std::optional<T> const &value)
{
   if (value) {
      writeValue(node[key], *value);
   }
}

// Writes a number, as an integer if it has no fractional part.
void writeNumber(JSONNode &node, double value)
{
   if (static_cast<int>(value) == value) {
      node << static_cast<int>(value);
   } else {
      node << value;
   }
}

JSONNode &appendNamed(JSONNode &array, std::string const &name)
{
   JSONNode &child = array.set_seq().append_child().set_map();
   child["name"] << name;
   return child;
}

void writeBinning(JSONNode &node, Binning const &binning)
{
   if (auto const *edges = std::get_if<EdgesBinning>(&binning)) {
      node["edges"].fill_seq(edges->edges);
   } else {
      node["nbins"] << std::get<RegularBinning>(binning).nbins;
   }
}

// An infinite or missing bound of a continuous data axis is omitted, which means that the range is unbounded. JSON null
// is not defined for bounds.
void writeAxis(JSONNode &node, ContinuousAxis const &axis)
{
   node.set_map();
   node["name"] << axis.name;
   if (axis.min && !std::isinf(*axis.min)) {
      node["min"] << *axis.min;
   }
   if (axis.max && !std::isinf(*axis.max)) {
      node["max"] << *axis.max;
   }
}

void writeAxis(JSONNode &node, CategoricalAxis const &axis)
{
   node.set_map();
   node["name"] << axis.name;
   node["categories"].fill_seq(axis.categories);
}

// An infinite or missing bound of a domain axis is omitted, which means that the range is unbounded.
void writeAxis(JSONNode &node, DomainAxis const &axis)
{
   node.set_map();
   node["name"] << axis.name;
   if (axis.fixedValue) {
      node["value"] << *axis.fixedValue;
      return;
   }
   // HS3 0.3 forbids "min" and "max" alongside "edges", which give the range. They are only written if they differ from
   // the edges, which would otherwise be lost.
   auto const *edges = axis.binning ? std::get_if<EdgesBinning>(&*axis.binning) : nullptr;
   const bool rangeFromEdges = edges && !edges->edges.empty() && axis.min && axis.max &&
                               *axis.min == edges->edges.front() && *axis.max == edges->edges.back();
   if (!rangeFromEdges && axis.min && !std::isinf(*axis.min)) {
      node["min"] << *axis.min;
   }
   if (!rangeFromEdges && axis.max && !std::isinf(*axis.max)) {
      node["max"] << *axis.max;
   }
   if (axis.binning) {
      writeBinning(node, *axis.binning);
   }
}

template <class... Axes>
void writeAxes(JSONNode &node, std::vector<std::variant<Axes...>> const &axes)
{
   node.set_seq();
   for (auto const &axis : axes) {
      std::visit([&](auto const &a) { writeAxis(node.append_child(), a); }, axis);
   }
}

template <class Axis>
void writeAxes(JSONNode &node, std::vector<Axis> const &axes)
{
   node.set_seq();
   for (auto const &axis : axes) {
      writeAxis(node.append_child(), axis);
   }
}

// The parameter of a modifier with a single parameter, and the name of its constraint distribution. A constraint type
// is written in the form of older documents.
void writeParameter(JSONNode &mod, ParameterRef const &parameter, Constraint const &constraint)
{
   mod["parameters"].set_seq().append_child() << parameter.name;
   std::visit(
      overloaded{[](Unconstrained) {}, [&](ConstraintType type) { mod["constraint"] << toString(type); },
                 [&](ConstraintPdf const &pdf) { mod["constraints"].set_seq().append_child() << pdf.pdf.name; }},
      constraint);
}

void writeGammaConstraint(JSONNode &node, GammaConstraint const &constraint, const char *typeKey)
{
   std::visit(overloaded{[](Unconstrained) {}, [&](ConstraintType type) { node[typeKey] << toString(type); },
                         [&](ConstantGammas) { node[typeKey] << "Const"; },
                         [&](GammaConstraintPdfs const &pdfs) {
                            JSONNode &names = node["constraints"].set_seq();
                            for (auto const &pdf : pdfs.pdfs) {
                               if (pdf) {
                                  names.append_child() << pdf->name;
                               } else {
                                  names.append_child().set_null();
                               }
                            }
                         }},
              constraint);
}

void writeModifier(JSONNode &mod, Modifier const &modifier)
{
   mod.set_map();
   mod["name"] << nameOf(modifier);
   mod["type"] << typeOf(modifier);
   std::visit(overloaded{
                 [&](NormFactor const &m) { writeParameter(mod, m.parameter, m.constraint); },
                 [&](NormSys const &m) {
                    writeParameter(mod, m.parameter, m.constraint);
                    if (m.interpolation) {
                       writeInterpolation(mod["interpolation"], *m.interpolation);
                    }
                    JSONNode &data = mod["data"].set_map();
                    data["lo"] << m.lo;
                    data["hi"] << m.hi;
                 },
                 [&](HistoSys const &m) {
                    writeParameter(mod, m.parameter, m.constraint);
                    if (m.interpolation) {
                       writeInterpolation(mod["interpolation"], *m.interpolation);
                    }
                    JSONNode &data = mod["data"].set_map();
                    writeContents(data["lo"].set_map()["contents"], m.lo.contents);
                    writeContents(data["hi"].set_map()["contents"], m.hi.contents);
                 },
                 [&](ShapeSys const &m) {
                    writeValue(mod["parameters"], m.parameters);
                    writeGammaConstraint(mod, m.constraint, "constraint");
                    // The values are only used by the constraint types of older documents.
                    if (!m.vals.empty()) {
                       mod["data"].set_map()["vals"].fill_seq(m.vals);
                    }
                 },
                 [&](StatError const &m) {
                    writeValue(mod["parameters"], m.parameters);
                    writeGammaConstraint(mod, m.constraint, "constraint");
                 },
                 [](CustomModifier const &) {},
              },
              modifier);
}

// The columns and the components of a combined data set or distribution, which name what they select in `refKey`.
template <class Ref_t>
void writeCombined(JSONNode &node, Combined<Ref_t> const &combined, const char *refKey)
{
   node["columns"].fill_seq(combined.columns);
   JSONNode &components = node["components"].set_seq();
   for (auto const &component : combined.components) {
      JSONNode &out = components.append_child().set_map();
      out["key"].fill_seq(component.key);
      out[refKey] << component.ref.name;
   }
}

void writeParameterPoint(JSONNode &points, ParameterPoint const &point)
{
   JSONNode &parameters = appendNamed(points, point.name)["parameters"].set_seq();
   for (auto const &p : point.parameters) {
      JSONNode &node = appendNamed(parameters, p.name);
      if (p.value) {
         std::visit([&](auto const &value) { node["value"] << value; }, *p.value);
      }
      if (p.isConst) {
         node["const"] << true;
      }
   }
}

void writeData(JSONNode &dataArray, Data const &data)
{
   JSONNode &node = appendNamed(dataArray, data.name);
   std::visit(overloaded{
                 [&](BinnedData const &d) {
                    node["type"] << "binned";
                    writeBinnedData(node, d);
                 },
                 [&](UnbinnedData const &d) {
                    node["type"] << "unbinned";
                    writeAxes(node["axes"], d.axes);
                    JSONNode &entries = node["entries"].set_seq();
                    const std::size_t nAxes = d.axes.size();
                    for (std::size_t i = 0; i < d.numEntries(); ++i) {
                       JSONNode &row = entries.append_child().set_seq();
                       for (std::size_t j = 0; j < nAxes; ++j) {
                          const double value = d.entries[i * nAxes + j];
                          if (auto const *category = std::get_if<CategoricalAxis>(&d.axes[j])) {
                             row.append_child() << category->categories[static_cast<std::size_t>(value)];
                          } else {
                             row.append_child() << value;
                          }
                       }
                    }
                    writeOptional(node, "weights", d.weights);
                 },
                 [&](CombinedData const &d) {
                    node["type"] << "combined";
                    writeCombined(node, d, "dataset");
                 },
                 [&](PointData const &d) {
                    node["type"] << "point";
                    node["value"] << d.value;
                    writeOptional(node, "uncertainty", d.uncertainty);
                    if (!d.axes.empty()) {
                       writeAxes(node["axes"], d.axes);
                    }
                 },
              },
              data.body);
}

void writeLikelihood(JSONNode &likelihoods, Likelihood const &likelihood)
{
   JSONNode &node = appendNamed(likelihoods, likelihood.name);
   JSONNode &distributions = node["distributions"].set_seq();
   if (likelihood.pureModel) {
      for (auto const &term : likelihood.terms) {
         distributions.append_child() << term.distribution.name;
      }
      node["data"].set_null();
   } else {
      JSONNode &data = node["data"].set_seq();
      for (auto const &term : likelihood.terms) {
         distributions.append_child() << term.distribution.name;
         std::visit(overloaded{[&](DataRef const &ref) { data.append_child() << ref.name; },
                               [&](double value) { writeNumber(data.append_child(), value); },
                               [&](std::monostate) { data.append_child().set_null(); }},
                    term.data);
      }
   }
   if (!likelihood.auxDistributions.empty()) {
      writeValue(node["aux_distributions"], likelihood.auxDistributions);
   }
}

void writeAnalysis(JSONNode &analyses, Analysis const &analysis)
{
   JSONNode &node = appendNamed(analyses, analysis.name);
   writeValue(node["domains"], analysis.domains);
   node["likelihood"] << analysis.likelihood.name;
   if (!analysis.parametersOfInterest.empty()) {
      writeValue(node["parameters_of_interest"], analysis.parametersOfInterest);
   }
   writeOptional(node, "init", analysis.init);
   writeOptional(node, "prior", analysis.prior);
}

void writeValue(JSONNode &node, Attributes const &attributes)
{
   node.set_map();
   if (!attributes.dict.empty()) {
      JSONNode &dict = node["dict"].set_map();
      for (auto const &entry : attributes.dict) {
         dict[entry.first] << entry.second;
      }
   }
   if (!attributes.tags.empty()) {
      node["tags"].fill_seq(attributes.tags);
   }
}

void writeValue(JSONNode &node, ModelConfigInfo const &info)
{
   node.set_map();
   writeOptional(node, "mcName", info.mcName);
   writeOptional(node, "pdfName", info.pdfName);
   writeOptional(node, "observables", info.observables);
}

// A component of "misc.ROOT_internal" with one value per name, which is left out if there are no values.
template <class T>
void writeInternal(JSONNode &root, const char *key, std::map<std::string, T> const &values)
{
   if (values.empty()) {
      return;
   }
   JSONNode &node = root.get("misc", "ROOT_internal")[key].set_map();
   for (auto const &item : values) {
      writeValue(node[item.first], item.second);
   }
}

void writeRooFitInternal(JSONNode &root, RooFitInternal const &internal)
{
   writeInternal(root, "attributes", internal.attributes);
   writeInternal(root, "ModelConfigs", internal.modelConfigs);
   writeInternal(root, "histfactory_constraints", internal.channelConstraints);
   writeInternal(root, "auxiliary_channels", internal.auxiliaryChannels);
   writeInternal(root, "bookkeeping_columns", internal.bookkeepingColumns);
   if (!internal.stepWidths.empty()) {
      JSONNode &stepWidths = root.get("misc", "minimization")["parameter_stepwidths"].set_seq();
      for (auto const &stepWidth : internal.stepWidths) {
         appendNamed(stepWidths, stepWidth.name)["step_width"] << stepWidth.width;
      }
   }
}

} // namespace

void writeContents(JSONNode &node, std::vector<double> const &values)
{
   node.set_seq();
   for (double value : values) {
      writeNumber(node.append_child(), value);
   }
}

void writeInterpolation(JSONNode &node, Interpolation const &interpolation)
{
   node.set_map();
   node["type"] << toString(interpolation.type);
   node["in"] << toString(interpolation.in);
   if (interpolation.out) {
      node["out"] << toString(*interpolation.out);
   } else {
      node["out"].set_null();
   }
}

namespace {

template <class Function_t>
void writeInterpolated(JSONNode &node, Function_t const &function, char const *type)
{
   node["type"] << type;
   auto const &interpolations = function.interpolations;
   const bool allEqual = std::all_of(interpolations.begin(), interpolations.end(),
                                     [&](Interpolation const &i) { return i == interpolations.front(); });
   JSONNode &interpolationsNode = node["interpolations"].set_seq();
   for (std::size_t i = 0; i < (allEqual ? std::min<std::size_t>(interpolations.size(), 1) : interpolations.size());
        ++i) {
      writeInterpolation(interpolationsNode.append_child(), interpolations[i]);
   }
   writeValue(node["vars"], function.parameters);
   node["nom"] << function.nominal;
   writeValue(node["high"], function.high);
   writeValue(node["low"], function.low);
}

} // namespace

void writeAxis(JSONNode &node, BinnedAxis const &axis)
{
   node.set_map();
   node["name"] << axis.name;
   // The range of an irregular binning is given by its "edges": HS3 0.3 forbids "min" and "max" alongside them.
   if (!std::holds_alternative<EdgesBinning>(axis.binning)) {
      node["min"] << axis.min;
      node["max"] << axis.max;
   }
   writeBinning(node, axis.binning);
}

void writeBinnedData(JSONNode &node, BinnedData const &data)
{
   writeAxes(node["axes"], data.axes);
   writeContents(node["contents"], data.contents);
   if (data.uncertainty) {
      JSONNode &uncertainty = node["uncertainty"].set_map();
      uncertainty["type"] << "gaussian_uncertainty";
      uncertainty["sigma"].fill_seq(data.uncertainty->sigma);
   }
}

void writeInterpolatedFunction(JSONNode &node, InterpolatedFunction const &function)
{
   writeInterpolated(node, function, "interpolation");
   node["positiveDefinite"] << function.positiveDefinite;
}

void writeInterpolatedNumber(JSONNode &node, InterpolatedNumber const &function)
{
   writeInterpolated(node, function, "interpolation0d");
}

void writeHistFactoryDist(JSONNode &node, HistFactoryDist const &dist)
{
   node.set_map();
   node["type"] << "histfactory_dist";
   if (!std::holds_alternative<Unconstrained>(dist.statError.constraint)) {
      writeGammaConstraint(node["staterror"].set_map(), dist.statError.constraint, "constraint_type");
   }
   if (dist.defaultInterpolation) {
      writeInterpolation(node["default_interpolation"], *dist.defaultInterpolation);
   }
   writeAxes(node["axes"], dist.axes);
   JSONNode &samples = node["samples"].set_seq();
   for (auto const &sample : dist.samples) {
      JSONNode &s = appendNamed(samples, sample.name);
      JSONNode &modifiers = s["modifiers"].set_seq();
      for (auto const &modifier : sample.modifiers) {
         writeModifier(modifiers.append_child(), modifier);
      }
      JSONNode &data = s["data"].set_map();
      writeContents(data["contents"], sample.contents);
      if (sample.errors) {
         writeContents(data["errors"], *sample.errors);
      }
   }
}

void writeCombinedDist(JSONNode &node, CombinedDist const &dist)
{
   node["type"] << "combined_dist";
   writeCombined(node, dist, "dist");
}

void writeDocument(JSONNode &root, HS3Document const &doc)
{
   root.set_map();
   JSONNode &metadata = root["metadata"].set_map();
   metadata["hs3_version"] << doc.metadata.hs3Version;
   for (auto const &package : doc.metadata.packages) {
      appendNamed(metadata["packages"], package.name)["version"] << package.version;
   }

   for (auto const &point : doc.parameterPoints) {
      writeParameterPoint(root["parameter_points"], point);
   }
   for (auto const &domain : doc.domains) {
      JSONNode &node = appendNamed(root["domains"], domain.name);
      node["type"] << "product_domain";
      writeAxes(node["axes"], domain.axes);
   }
   for (auto const &data : doc.data) {
      writeData(root["data"], data);
   }
   for (auto const &likelihood : doc.likelihoods) {
      writeLikelihood(root["likelihoods"], likelihood);
   }
   for (auto const &analysis : doc.analyses) {
      writeAnalysis(root["analyses"], analysis);
   }
   writeRooFitInternal(root, doc.rooFit);
}

} // namespace HS3
} // namespace RooFit
