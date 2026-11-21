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

// Everything that older versions of RooFit wrote in a way that is not part of the HS3 standard is translated here into
// the form that the parser understands. The rewrites go by the shape of the JSON: the same version number was written
// by ROOT releases with different conventions. Only what a missing component means depends on the version.

#include "HS3Schema.h"

#include <functional>
#include <set>
#include <string>

using RooFit::Detail::JSONNode;

namespace RooFit {
namespace HS3 {
namespace {

// The version that the standard requires the interpolation of modifiers from on.
constexpr Version explicitInterpolationVersion{0, 3, 0};
// Before this version, normsys and histosys modifiers without constraint had an implicit Gaussian constraint.
constexpr Version explicitConstraintVersion{0, 2, 0};
// From this version on, the gamma parameters of modifiers are free unless "constraints" names their constraint terms,
// "shapesys" is deprecated, and a sample without "errors" has no statistical uncertainty.
constexpr Version namedConstraintVersion{0, 3, 0};

using IsDistribution = std::function<bool(std::string const &)>;

bool isScalar(JSONNode const &node)
{
   return !node.is_container() && !node.is_null();
}

// Calls fn(element, cursor, index) for each struct in the array `key` of `parent`, if there is one. What is skipped is
// reported by the parser.
template <class Fn>
void forEachStruct(JSONNode &parent, const char *key, Cursor const &parentCursor, Fn &&fn)
{
   if (!parent.is_map() || !parent.has_child(key) || !parent[key].is_seq()) {
      return;
   }
   Cursor array{parent[key], parentCursor, key};
   std::size_t index = 0;
   for (JSONNode &element : parent[key].children()) {
      if (element.is_map()) {
         fn(element, Cursor{element, array, index}, index);
      }
      ++index;
   }
}

bool isConstraintType(std::string const &value)
{
   return value == "Gauss" || value == "Poisson" || value == "LogNormal";
}

// Older versions of RooFit spelled "LogNormal" differently.
std::string canonicalConstraintType(std::string const &value)
{
   return value == "Lognormal" ? "LogNormal" : value;
}

// The "constraints" of the standard take precedence over the keys that older documents used.
bool eraseLegacyConstraints(JSONNode &mod)
{
   if (!mod.has_child("constraints")) {
      return false;
   }
   mod.erase("constraint");
   mod.erase("constraint_name");
   mod.erase("constraint_type");
   return true;
}

// Write a legacy constraint of a normsys, histosys or normfactor modifier as "constraint" if it is a type, or
// "constraint_name" if it refers to a distribution.
void upgradeConstraint(JSONNode &mod, IsDistribution const &isDistribution)
{
   if (eraseLegacyConstraints(mod)) {
      return;
   }
   if (mod.has_child("constraint_name")) {
      mod.erase("constraint");
      mod.erase("constraint_type");
      return;
   }
   if (mod.has_child("constraint")) {
      mod.erase("constraint_type");
      JSONNode &constraint = mod["constraint"];
      if (!isScalar(constraint)) {
         return; // reported by the parser
      }
      const std::string value = constraint.val();
      // In older documents, the "constraint" can be the name of a distribution, which has priority.
      if (isDistribution(value)) {
         mod["constraint_name"] << value;
         mod.erase("constraint");
      } else {
         constraint << canonicalConstraintType(value);
      }
      return;
   }
   if (mod.has_child("constraint_type")) {
      JSONNode &constraintType = mod["constraint_type"];
      if (isScalar(constraintType)) {
         mod["constraint"] << canonicalConstraintType(constraintType.val());
      }
      mod.erase("constraint_type");
   }
}

// The constraint of the gamma parameters of a shapesys modifier. In contrast to the other modifiers, it is not possible
// to name a distribution in this way, and "constraint_type" has priority.
void upgradeGammaConstraint(JSONNode &mod)
{
   if (eraseLegacyConstraints(mod)) {
      return;
   }
   if (mod.has_child("constraint_type")) {
      JSONNode &constraintType = mod["constraint_type"];
      if (isScalar(constraintType)) {
         mod["constraint"] << canonicalConstraintType(constraintType.val());
      }
      mod.erase("constraint_type");
   } else if (mod.has_child("constraint") && isScalar(mod["constraint"])) {
      mod["constraint"] << canonicalConstraintType(mod["constraint"].val());
   }
}

// An integer code that was used before the structured interpolations.
void upgradeInterpolationCode(JSONNode &mod, InterpolationClass interpolationClass, Cursor const &c)
{
   if (!mod.has_child("interpolation") || mod["interpolation"].is_map()) {
      return;
   }
   JSONNode &node = mod["interpolation"];
   if (auto interpolation = readInterpolationCode(Cursor{node, c, "interpolation"}, interpolationClass)) {
      node.set_null();
      writeInterpolation(node, *interpolation);
   }
}

struct ChannelState {
   std::string channelName;
   bool warnedInterpolation = false;
   bool warnedUnconstrained = false;
   bool warnedShapeSys = false;
};

// Whether "constraints" leaves the single parameter of a modifier unconstrained.
bool isExplicitlyUnconstrained(JSONNode const &constraints)
{
   return constraints.is_seq() && constraints.num_children() == 1 && constraints.child(0).is_null();
}

void upgradeModifier(JSONNode &mod, std::size_t index, std::string const &sampleName, Version const &version,
                     bool hasDefaultInterpolation, IsDistribution const &isDistribution, ChannelState &state,
                     Cursor const &c)
{
   if (!mod.has_child("type") || !isScalar(mod["type"])) {
      return; // reported by the parser
   }
   const std::string type = mod["type"].val();
   const bool hasSingleParameter = type == "normfactor" || type == "normsys" || type == "histosys";

   // The "parameters" of the standard take precedence over the "parameter" of older documents.
   std::string parameter;
   if (hasSingleParameter && mod.has_child("parameters")) {
      mod.erase("parameter");
      JSONNode const &parameters = mod["parameters"];
      if (parameters.is_seq() && parameters.num_children() == 1 && isScalar(parameters.child(0))) {
         parameter = parameters.child(0).val();
      }
   } else if (mod.has_child("parameter") && isScalar(mod["parameter"])) {
      parameter = mod["parameter"].val();
   }

   // The name of a modifier is not part of the standard. RooFit names objects after it, so a generated name must be
   // unique, unlike the name of a parameter, which is shared on purpose.
   if (!mod.has_child("name")) {
      mod["name"] << (!parameter.empty() ? parameter
                                         : state.channelName + "_" + sampleName + "_syst_" + std::to_string(index));
      // Older documents gave the function of a custom modifier in "parameter". It is the name now, so it is not left
      // for the parser to report as ignored.
      if (type == "custom" && !parameter.empty()) {
         mod.erase("parameter");
      }
   }
   const std::string name = isScalar(mod["name"]) ? mod["name"].val() : "";

   if (hasSingleParameter && !mod.has_child("parameter") && !mod.has_child("parameters")) {
      parameter = type == "normfactor" ? name : "alpha_" + name;
      mod["parameter"] << parameter;
   }

   if (type == "normfactor") {
      upgradeConstraint(mod, isDistribution);
   } else if (type == "normsys" || type == "histosys") {
      upgradeConstraint(mod, isDistribution);
      const bool constrained = mod.has_child("constraint") || mod.has_child("constraint_name") ||
                               (mod.has_child("constraints") && !isExplicitlyUnconstrained(mod["constraints"]));
      if (!constrained) {
         if (version < explicitConstraintVersion) {
            mod.erase("constraints");
            mod["constraint"] << "Gauss";
         } else if (!state.warnedUnconstrained) {
            c.warning("modifier '" + name + "' of sample '" + sampleName + "' has no constraint, so its parameter '" +
                      parameter +
                      "' is unconstrained. Older RooFit versions implied a unit Gaussian. Other modifiers of this "
                      "channel are not reported.");
            state.warnedUnconstrained = true;
         }
      }
      // Before structured interpolations were introduced, both modifier classes used the integer code 4 as their
      // implicit default. The meaning of the code is class-dependent. HS3 forbids implementation-specific defaults.
      if (!mod.has_child("interpolation") && !hasDefaultInterpolation && version < explicitInterpolationVersion) {
         if (!state.warnedInterpolation) {
            c.warning("interpolation for " + type + " modifier '" + name + "' in sample '" + sampleName +
                      "' of channel '" + state.channelName +
                      "' is not given, and the channel has no \"default_interpolation\". Using the implicit default "
                      "of older RooFit versions (code 4), which is deprecated.");
            state.warnedInterpolation = true;
         }
         mod["interpolation"] << 4;
      }
      const auto interpolationClass = type == "normsys" ? InterpolationClass::Flexible : InterpolationClass::Piecewise;
      upgradeInterpolationCode(mod, interpolationClass, c);
   } else if (type == "shapesys" || type == "shapefactor") {
      if (type == "shapesys" && !(version < namedConstraintVersion) && !state.warnedShapeSys) {
         c.warning("the modifier type \"shapesys\" is deprecated, please use \"shapefactor\" instead.");
         state.warnedShapeSys = true;
      }
      upgradeGammaConstraint(mod);
      if (!mod.has_child("constraint") && !mod.has_child("constraints")) {
         if (version < namedConstraintVersion) {
            // Without any information about the constraint, older versions created Gaussian constraints.
            mod["constraint"] << "Gauss";
         } else if (mod.has_child("data") && mod["data"].is_map() && mod["data"].has_child("vals")) {
            if (mod["data"]["vals"].is_seq() && mod["data"]["vals"].num_children() > 0) {
               c.warning("the \"data.vals\" of modifier '" + name + "' of sample '" + sampleName +
                         "' are ignored, because it names no constraints, so its parameters are free");
            }
            mod["data"].erase("vals");
         }
      }
   } else if (type == "staterror") {
      if (eraseLegacyConstraints(mod)) {
         return;
      }
      // ROOT 6.38 and 6.40.00 wrote "constraint_type".
      if (mod.has_child("constraint_type")) {
         if (!mod.has_child("constraint") && isScalar(mod["constraint_type"])) {
            mod["constraint"] << mod["constraint_type"].val();
         }
         mod.erase("constraint_type");
      }
   }
}

void upgradeHistFactory(JSONNode &dist, Version const &version, IsDistribution const &isDistribution, Cursor const &c)
{
   ChannelState state;
   if (dist.has_child("name") && isScalar(dist["name"])) {
      state.channelName = dist["name"].val();
      if (state.channelName.rfind("model_", 0) == 0) {
         state.channelName.erase(0, 6);
      }
   }

   // Before HS3 0.3, the gamma parameters of the staterror modifiers had Poisson constraints unless the document said
   // otherwise, and samples without "errors" had Poisson uncertainties. From HS3 0.3 on, RooFit only creates
   // constraints if its channel-level settings give their type.
   const bool legacyStatError = version < namedConstraintVersion;
   if (legacyStatError && !dist.has_child("staterror")) {
      dist["staterror"].set_map();
   }
   if (dist.has_child("staterror") && dist["staterror"].is_map()) {
      JSONNode &config = dist["staterror"];
      upgradeGammaConstraint(config);
      if (legacyStatError) {
         if (!config.has_child("constraint") && !config.has_child("constraints")) {
            config["constraint"] << "Poisson";
         }
         config["implied_poisson_errors"] << true;
      }
   }

   const bool hasDefaultInterpolation = dist.has_child("default_interpolation");
   forEachStruct(dist, "samples", c, [&](JSONNode &sample, Cursor const &sampleCursor, std::size_t) {
      const std::string sampleName = sample.has_child("name") && isScalar(sample["name"]) ? sample["name"].val() : "";
      forEachStruct(sample, "modifiers", sampleCursor, [&](JSONNode &mod, Cursor const &modCursor, std::size_t index) {
         upgradeModifier(mod, index, sampleName, version, hasDefaultInterpolation, isDistribution, state, modCursor);
      });
   });
}

// Older versions of RooFit wrote the binning of the observables also for unbinned data, which the standard forbids.
// The edges were written without "min" and "max".
void upgradeUnbinnedAxis(JSONNode &axis)
{
   if (axis.has_child("edges") && axis["edges"].is_seq() && axis["edges"].num_children() > 0) {
      JSONNode &edges = axis["edges"];
      JSONNode const &first = edges.child(0);
      JSONNode const &last = edges.child(edges.num_children() - 1);
      if (!axis.has_child("min") && first.is_number()) {
         axis["min"] << first.val_double();
      }
      if (!axis.has_child("max") && last.is_number()) {
         axis["max"] << last.val_double();
      }
   }
   axis.erase("edges");
   axis.erase("nbins");
}

void upgradeData(JSONNode &root, Version const &version, Cursor const &rootCursor)
{
   if (!(version < explicitInterpolationVersion)) {
      return;
   }
   forEachStruct(root, "data", rootCursor, [](JSONNode &data, Cursor const &dataCursor, std::size_t) {
      if (data.has_child("type") && isScalar(data["type"]) && data["type"].val() == "unbinned") {
         forEachStruct(data, "axes", dataCursor,
                       [](JSONNode &axis, Cursor const &, std::size_t) { upgradeUnbinnedAxis(axis); });
      }
   });
}

// HS3 0.3 forbids "min" and "max" alongside "edges" (HS3 #126), which documents before 0.3 had. Where they agree with
// the edges, which the parser checks, they are only redundant, so they give a warning.
void warnBoundsWithEdges(JSONNode &root, Version const &version, Cursor const &rootCursor)
{
   if (version < Version{0, 3, 0}) {
      return;
   }
   for (const char *section : {"domains", "data", "distributions"}) {
      forEachStruct(root, section, rootCursor, [](JSONNode &object, Cursor const &objectCursor, std::size_t) {
         forEachStruct(object, "axes", objectCursor, [](JSONNode &axis, Cursor const &axisCursor, std::size_t) {
            if (axis.has_child("edges") && (axis.has_child("min") || axis.has_child("max"))) {
               axisCursor.warning("\"min\" and \"max\" must not be given alongside \"edges\" from HS3 0.3 on, the "
                                  "edges give the range");
            }
         });
      });
   }
}

Version documentVersion(JSONNode const &root)
{
   Version version;
   if (JSONNode const *versionNode = root.find("metadata", "hs3_version")) {
      if (isScalar(*versionNode)) {
         parseVersion(versionNode->val(), version);
      }
   }
   return version;
}

} // namespace

void upgradeDocument(JSONNode &root, Diagnostics &diagnostics, ParseOptions const &options)
{
   if (!root.is_map()) {
      return;
   }
   const Version version = documentVersion(root);
   Cursor rootCursor{root, diagnostics};
   warnBoundsWithEdges(root, version, rootCursor);
   upgradeData(root, version, rootCursor);
   if (!options.typedHistFactory) {
      return;
   }

   std::set<std::string> distributionNames;
   forEachStruct(root, "distributions", rootCursor, [&](JSONNode &dist, Cursor const &, std::size_t) {
      if (dist.has_child("name") && isScalar(dist["name"])) {
         distributionNames.insert(dist["name"].val());
      }
   });
   // The version from which on "constraint" is only a type of the standard.
   const bool constraintCanBeName = version < explicitInterpolationVersion;
   IsDistribution isDistribution = [&](std::string const &name) {
      return constraintCanBeName && distributionNames.count(name) > 0;
   };

   forEachStruct(root, "distributions", rootCursor, [&](JSONNode &dist, Cursor const &distCursor, std::size_t) {
      if (dist.has_child("type") && isScalar(dist["type"]) && dist["type"].val() == "histfactory_dist") {
         upgradeHistFactory(dist, version, isDistribution, distCursor);
      }
   });
}

void upgradeHistFactoryDist(JSONNode &distribution, Version const &version, Diagnostics &diagnostics)
{
   IsDistribution isDistribution = [](std::string const &name) {
      return !isConstraintType(name) && name != "Lognormal" && name != "Const";
   };
   upgradeHistFactory(distribution, version, isDistribution, Cursor{distribution, diagnostics});
}

} // namespace HS3
} // namespace RooFit
