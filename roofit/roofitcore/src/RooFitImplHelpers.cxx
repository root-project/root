/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Author:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitImplHelpers.h>

#include <RooAbsCategory.h>
#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAbsRealLValue.h>
#include <RooArgList.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooProdPdf.h>
#include <RooRealSumPdf.h>
#include <RooSimultaneous.h>

#include <ROOT/StringUtils.hxx>
#include <TClass.h>

#include <unordered_map>

namespace RooHelpers {

namespace {
std::pair<double, double> getBinningInterval(RooAbsBinning const &binning)
{
   if (!binning.isParameterized()) {
      return {binning.lowBound(), binning.highBound()};
   } else {
      return {binning.lowBoundFunc()->getVal(), binning.highBoundFunc()->getVal()};
   }
}
} // namespace

/// Get the lower and upper bound of parameter range if arg can be casted to RooAbsRealLValue.
/// If no range with rangeName is defined for the argument, this will check if a binning of the
/// same name exists and return the interval covered by the binning.
/// Returns `{-infinity, infinity}` if argument can't be casted to RooAbsRealLValue* or if no
/// range or binning with the requested name exists.
/// \param[in] arg RooAbsArg for which to get the range.
/// \param[in] rangeName The name of the range.
std::pair<double, double> getRangeOrBinningInterval(RooAbsArg const *arg, const char *rangeName)
{
   auto rlv = dynamic_cast<RooAbsRealLValue const *>(arg);
   if (rlv) {
      if (rangeName && rlv->hasRange(rangeName)) {
         return {rlv->getMin(rangeName), rlv->getMax(rangeName)};
      } else if (auto binning = rlv->getBinningPtr(rangeName)) {
         return getBinningInterval(*binning);
      }
   }
   return {-std::numeric_limits<double>::infinity(), +std::numeric_limits<double>::infinity()};
}

/// Check if there is any overlap when a list of ranges is applied to a set of observables.
/// \param[in] observables The observables to check for overlap
/// \param[in] rangeNames The names of the ranges.
bool checkIfRangesOverlap(RooArgSet const &observables, std::vector<std::string> const &rangeNames)
{
   // cache the range limits in a flat vector
   std::vector<std::pair<double, double>> limits;
   limits.reserve(rangeNames.size() * observables.size());

   for (auto const &range : rangeNames) {
      for (auto const &obs : observables) {
         if (dynamic_cast<RooAbsCategory const *>(obs)) {
            // Nothing to be done for category observables
         } else if (auto *rlv = dynamic_cast<RooAbsRealLValue const *>(obs)) {
            limits.emplace_back(rlv->getMin(range.c_str()), rlv->getMax(range.c_str()));
         } else {
            throw std::logic_error(
               "Classes that represent observables are expected to inherit from RooAbsRealLValue or RooAbsCategory!");
         }
      }
   }

   auto nRanges = rangeNames.size();
   auto nObs = limits.size() / nRanges; // number of observables that are not categories

   // loop over pairs of ranges
   for (size_t ir1 = 0; ir1 < nRanges; ++ir1) {
      for (size_t ir2 = ir1 + 1; ir2 < nRanges; ++ir2) {

         // Loop over observables. If all observables have overlapping limits for
         // these ranges, the hypercubes defining the range are overlapping and we
         // can return `true`.
         size_t overlaps = 0;
         for (size_t io1 = 0; io1 < nObs; ++io1) {
            auto r1 = limits[ir1 * nObs + io1];
            auto r2 = limits[ir2 * nObs + io1];
            overlaps +=
               (r1.second > r2.first && r1.first < r2.second) || (r2.second > r1.first && r2.first < r1.second);
         }
         if (overlaps == nObs)
            return true;
      }
   }

   return false;
}

/// Create a string with all sorted names of RooArgSet elements separated by delimiters.
/// \param[in] argSet The input RooArgSet.
std::string getColonSeparatedNameString(RooArgSet const &argSet, char delim)
{

   RooArgList tmp(argSet);
   tmp.sort();

   std::string content;
   for (auto const &arg : tmp) {
      content += arg->GetName();
      content += delim;
   }
   if (!content.empty()) {
      content.pop_back();
   }
   return content;
}

/// Construct a RooArgSet of objects in a RooArgSet whose names match to those
/// in the names string.
/// \param[in] argSet The input RooArgSet.
/// \param[in] names The names of the objects to select in a colon-separated string.
RooArgSet selectFromArgSet(RooArgSet const &argSet, std::string const &names)
{
   RooArgSet output;
   for (auto const &name : ROOT::Split(names, ":")) {
      if (auto arg = argSet.find(name.c_str()))
         output.add(*arg);
   }
   return output;
}

std::string getRangeNameForSimComponent(std::string const &rangeName, bool splitRange, std::string const &catName)
{
   if (splitRange && !rangeName.empty()) {
      std::string out;
      auto tokens = ROOT::Split(rangeName, ",");
      for (std::string const &token : tokens) {
         out += token + "_" + catName + ",";
      }
      out.pop_back(); // to remove the last comma
      return out;
   }

   return rangeName;
}

BinnedLOutput getBinnedL(RooAbsPdf const &pdf)
{
   if (pdf.getAttribute("BinnedLikelihood") && pdf.IsA()->InheritsFrom(RooRealSumPdf::Class())) {
      // Simplest case: top-level of component is a RooRealSumPdf
      return {const_cast<RooAbsPdf *>(&pdf), true};
   } else if (pdf.IsA()->InheritsFrom(RooProdPdf::Class())) {
      // Default case: top-level pdf is a product of RooRealSumPdf and other pdfs
      for (RooAbsArg *component : static_cast<RooProdPdf const &>(pdf).pdfList()) {
         if (component->getAttribute("BinnedLikelihood") && component->IsA()->InheritsFrom(RooRealSumPdf::Class())) {
            return {static_cast<RooAbsPdf *>(component), true};
         }
         if (component->getAttribute("MAIN_MEASUREMENT")) {
            // not really a binned pdf, but this prevents a (potentially) long list of subsidiary measurements to be
            // passed to the slave calculator
            return {static_cast<RooAbsPdf *>(component), false};
         }
      }
   }
   return {nullptr, false};
}

/// Get the topologically-sorted list of all nodes in the computation graph.
void getSortedComputationGraph(RooAbsArg const &func, RooArgSet &out)
{
   // Get the set of nodes in the computation graph. Do the detour via
   // RooArgList to avoid deduplication done after adding each element.
   RooArgList serverList;
   func.treeNodeServerList(&serverList, nullptr, true, true, false, true);
   // If we fill the servers in reverse order, they are approximately in
   // topological order so we save a bit of work in sortTopologically().
   out.add(serverList.rbegin(), serverList.rend(), /*silent=*/true);
   // Sort nodes topologically: the servers of any node will be before that
   // node in the collection.
   out.sortTopologically();
}

namespace Detail {

namespace {

using ToCloneList = std::vector<RooAbsArg const *>;
using ToCloneMap = std::unordered_map<TNamed const *, RooAbsArg const *>;

// Add clones of servers of given argument to end of list
void addServerClonesToList(const RooAbsArg &var, ToCloneList &outlist, ToCloneMap &outmap, bool deepCopy,
                           RooArgSet const *observables)
{
   if (outmap.find(var.namePtr()) != outmap.end()) {
      return;
   }

   if (observables && var.isFundamental() && !observables->find(var)) {
      return;
   }

   outmap[var.namePtr()] = &var;
   outlist.push_back(&var);

   if (deepCopy) {
      for (const auto server : var.servers()) {
         addServerClonesToList(*server, outlist, outmap, deepCopy, observables);
      }
   }
}

} // namespace

/// Implementation of RooAbsCollection::snapshot() with some extra parameters.
/// to be used in other RooHelpers functions.
/// param[in] input The input collection.
/// param[in] output The output collection.
/// param[in] deepCopy If the whole computation graph should be cloned recursively.
/// param[in] observables If this is not a nullptr, only the fundamental
///                       variables that are in observables are deep cloned.
bool snapshotImpl(RooAbsCollection const &input, RooAbsCollection &output, bool deepCopy, RooArgSet const *observables)
{
   // Figure out what needs to be cloned
   ToCloneList toCloneList;
   ToCloneMap toCloneMap;
   for (RooAbsArg *orig : input) {
      addServerClonesToList(*orig, toCloneList, toCloneMap, deepCopy, observables);
   }

   // Actually do the cloning
   output.reserve(toCloneList.size());
   for (RooAbsArg const *arg : toCloneList) {
      std::unique_ptr<RooAbsArg> serverClone{static_cast<RooAbsArg *>(arg->Clone())};
      serverClone->setAttribute("SnapShot_ExtRefClone");
      output.addOwned(std::move(serverClone));
   }

   // Redirect all server connections to internal list members
   for (RooAbsArg *var : output) {
      var->redirectServers(output, deepCopy && !observables);
   }

   return false;
}

RooAbsArg *cloneTreeWithSameParametersImpl(RooAbsArg const &arg, RooArgSet const *observables)
{
   // Clone tree using snapshot
   RooArgSet clonedNodes;
   snapshotImpl(RooArgSet(arg), clonedNodes, true, observables);

   // Find the head node in the cloneSet
   RooAbsArg *head = clonedNodes.find(arg);
   assert(head);

   // We better to release the ownership before removing the "head". Otherwise,
   // "head" might also be deleted as the clonedNodes collection owns it.
   // (Actually this does not happen because even an owning collection doesn't
   // delete the element when removed by pointer lookup, but it's better not to
   // rely on this unexpected fact).
   clonedNodes.releaseOwnership();

   // Remove the head node from the cloneSet
   // To release it from the set ownership
   clonedNodes.remove(*head);

   // Add the set as owned component of the head
   head->addOwnedComponents(std::move(clonedNodes));

   return head;
}

} // namespace Detail

} // namespace RooHelpers

namespace RooFit {
namespace Detail {

/// Transform a string into a valid C++ variable name by replacing forbidden
/// characters with underscores.
/// @param in The input string.
/// @return A new string valid variable name.
std::string makeValidVarName(std::string const &in)
{
   std::string out;
   if (std::isdigit(in[0])) {
      out += '_';
   }
   out += in;
   std::transform(out.begin(), out.end(), out.begin(), [](char c) { return std::isalnum(c) ? c : '_'; });
   return out;
}

/// Replace all occurrences of `what` with `with` inside of `inOut`.
void replaceAll(std::string &inOut, std::string_view what, std::string_view with)
{
   for (std::string::size_type pos{}; inOut.npos != (pos = inOut.find(what.data(), pos, what.length()));
        pos += with.length()) {
      inOut.replace(pos, what.length(), with.data(), with.length());
   }
}

std::string makeSliceCutString(RooArgSet const &sliceDataSet)
{
   std::stringstream cutString;
   bool first = true;
   for (RooAbsArg *sliceVar : sliceDataSet) {
      if (!first) {
         cutString << "&&";
      } else {
         first = false;
      }

      if (auto *real = dynamic_cast<RooAbsRealLValue *>(sliceVar)) {
         cutString << real->GetName() << "==" << real->getVal();
      } else if (auto *cat = dynamic_cast<RooAbsCategoryLValue *>(sliceVar)) {
         cutString << cat->GetName() << "==" << cat->getCurrentIndex();
      }
   }
   return cutString.str();
}

} // namespace Detail
} // namespace RooFit

/// \endcond
