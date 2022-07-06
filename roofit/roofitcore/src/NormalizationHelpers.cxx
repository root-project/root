/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "NormalizationHelpers.h"

#include <TClass.h>
#include <RooAbsCachedPdf.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooAddition.h>
#include <RooConstraintSum.h>
#include <RooProdPdf.h>

#include "RooNormalizedPdf.h"

#include <TClass.h>

namespace {

using RooFit::Detail::DataKey;
using ServerLists = std::map<DataKey, std::vector<DataKey>>;

class GraphChecker {
public:
   GraphChecker(RooAbsArg const &topNode)
   {

      // To track the RooProdPdfs to figure out which ones are responsible for constraints.
      std::vector<RooAbsArg *> prodPdfs;

      // Get the list of servers for each node by data key.
      {
         RooArgList nodes;
         topNode.treeNodeServerList(&nodes, nullptr, true, true, false, true);
         RooArgSet nodesSet{nodes};
         for (RooAbsArg *node : nodesSet) {
            if (dynamic_cast<RooProdPdf *>(node)) {
               prodPdfs.push_back(node);
            }
            _serverLists[node];
            bool isConstraintSum = dynamic_cast<RooConstraintSum const *>(node);
            for (RooAbsArg *server : node->servers()) {
               _serverLists[node].push_back(server);
               if (isConstraintSum)
                  _constraints.insert(server);
            }
         }
      }
      for (auto &item : _serverLists) {
         auto &l = item.second;
         std::sort(l.begin(), l.end());
         l.erase(std::unique(l.begin(), l.end()), l.end());
      }

      // Loop over the RooProdPdfs to figure out which ones are responsible for constraints.
      for (auto *prodPdf : static_range_cast<RooProdPdf *>(prodPdfs)) {
         std::size_t actualPdfIdx = 0;
         std::size_t nNonConstraint = 0;
         for (std::size_t i = 0; i < prodPdf->pdfList().size(); ++i) {
            RooAbsArg &pdf = prodPdf->pdfList()[i];

            // Heuristic for HistFactory models to find also the constraints
            // that were not extracted for the RooConstraint sum, e.g. because
            // they were constant. TODO: fix RooProdPdf such that is also
            // extracts constraints for which the parameters is set constant.
            bool isProbablyConstraint = std::string(pdf.GetName()).find("onstrain") != std::string::npos;

            if (_constraints.find(&pdf) == _constraints.end() && !isProbablyConstraint) {
               actualPdfIdx = i;
               ++nNonConstraint;
            }
         }
         if (nNonConstraint != prodPdf->pdfList().size()) {
            if (nNonConstraint != 1) {
               throw std::runtime_error("A RooProdPdf that multiplies a pdf with constraints should contain only one "
                                        "pdf that is not a constraint!");
            }
            _prodPdfsWithConstraints[prodPdf] = actualPdfIdx;
         }
      }
   }

   bool dependsOn(DataKey arg, DataKey testArg)
   {

      std::pair<DataKey, DataKey> p{arg, testArg};

      auto found = _results.find(p);
      if (found != _results.end())
         return found->second;

      if (arg == testArg)
         return true;

      auto const &serverList = _serverLists.at(arg);

      // Next test direct dependence
      auto foundServer = std::find(serverList.begin(), serverList.end(), testArg);
      if (foundServer != serverList.end()) {
         _results.emplace(p, true);
         return true;
      }

      // If not, recurse
      for (auto const &server : serverList) {
         bool t = dependsOn(server, testArg);
         _results.emplace(std::pair<DataKey, DataKey>{server, testArg}, t);
         if (t) {
            return true;
         }
      }

      _results.emplace(p, false);
      return false;
   }

   bool isConstraint(DataKey key) const
   {
      auto found = _constraints.find(key);
      return found != _constraints.end();
   }

   std::unordered_map<RooAbsArg *, std::size_t> const &prodPdfsWithConstraints() const
   {
      return _prodPdfsWithConstraints;
   }

private:
   std::unordered_set<DataKey> _constraints;
   std::unordered_map<RooAbsArg *, std::size_t> _prodPdfsWithConstraints;
   ServerLists _serverLists;
   std::map<std::pair<DataKey, DataKey>, bool> _results;
};

void treeNodeServerListAndNormSets(const RooAbsArg &arg, RooAbsCollection &list, RooArgSet const &normSet,
                                   std::unordered_map<DataKey, RooArgSet *> &normSets, GraphChecker const &checker)
{
   if (normSets.find(&arg) != normSets.end())
      return;

   list.add(arg, true);

   // normalization sets only need to be added for pdfs
   if (dynamic_cast<RooAbsPdf const *>(&arg)) {
      normSets.insert({&arg, new RooArgSet{normSet}});
   }

   // Recurse if current node is derived
   if (arg.isDerived() && !arg.isFundamental()) {
      for (const auto server : arg.servers()) {

         if (!server->isValueServer(arg)) {
            continue;
         }

         // If this is a server that is also serving a RooConstraintSum, it
         // should be skipped because it is not evaluated by this client (e.g.
         // a RooProdPdf). It was only part of the servers to be extracted for
         // the constraint sum.
         if (!dynamic_cast<RooConstraintSum const *>(&arg) && checker.isConstraint(server)) {
            continue;
         }

         auto differentSet = arg.fillNormSetForServer(normSet, *server);
         if (differentSet)
            differentSet->sort();

         auto &serverNormSet = differentSet ? *differentSet : normSet;

         // Make sure that the server is not already part of the computation
         // graph with a different normalization set.
         auto found = normSets.find(server);
         if (found != normSets.end()) {
            if (found->second->size() != serverNormSet.size() || !serverNormSet.hasSameLayout(*found->second)) {
               std::stringstream ss;
               ss << server->ClassName() << "::" << server->GetName()
                  << " is requested to be evaluated with two different normalization sets in the same model!";
               ss << " This is not supported yet. The conflicting norm sets are:\n    RooArgSet";
               serverNormSet.printValue(ss);
               ss << " requested by " << arg.ClassName() << "::" << arg.GetName() << "\n    RooArgSet";
               found->second->printValue(ss);
               ss << " first requested by other client";
               auto errMsg = ss.str();
               oocoutE(server, Minimization) << errMsg << std::endl;
               throw std::runtime_error(errMsg);
            }
            continue;
         }

         treeNodeServerListAndNormSets(*server, list, serverNormSet, normSets, checker);
      }
   }
}

std::vector<std::unique_ptr<RooAbsArg>> unfoldIntegrals(RooAbsArg const &topNode, RooArgSet const &normSet,
                                                        std::unordered_map<DataKey, RooArgSet *> &normSets,
                                                        RooArgSet &replacedArgs)
{
   std::vector<std::unique_ptr<RooAbsArg>> newNodes;

   // No normalization set: we don't need to create any integrals
   if (normSet.empty())
      return newNodes;

   GraphChecker checker{topNode};

   RooArgSet nodes;
   // The norm sets are sorted to compare them for equality more easliy
   RooArgSet normSetSorted{normSet};
   normSetSorted.sort();
   treeNodeServerListAndNormSets(topNode, nodes, normSetSorted, normSets, checker);

   // Clean normsets of the variables that the arg does not depend on
   // std::unordered_map<std::pair<RooAbsArg const*,RooAbsArg const*>,bool> dependsResults;
   for (auto &item : normSets) {
      if (!item.second || item.second->empty())
         continue;
      auto actualNormSet = new RooArgSet{};
      for (auto *narg : *item.second) {
         if (checker.dependsOn(item.first, narg))
            actualNormSet->add(*narg);
      }
      delete item.second;
      item.second = actualNormSet;
   }

   // Function to `oldArg` with `newArg` in the computation graph.
   auto replaceArg = [&](RooAbsArg &newArg, RooAbsArg const &oldArg) {
      const std::string attrib = std::string("ORIGNAME:") + oldArg.GetName();

      newArg.setAttribute(attrib.c_str());
      newArg.setStringAttribute("_replaced_arg", oldArg.GetName());

      RooArgList newServerList{newArg};

      RooArgList originalClients;
      for (auto *client : oldArg.clients()) {
         originalClients.add(*client);
      }
      for (auto *client : originalClients) {
         if (!nodes.containsInstance(*client))
            continue;
         if (dynamic_cast<RooAbsCachedPdf *>(client))
            continue;
         client->redirectServers(newServerList, false, true);
      }
      replacedArgs.add(oldArg);

      newArg.setAttribute(attrib.c_str(), false);
   };

   // Replaces the RooProdPdfs that were used to wrap constraints with the actual pdf.
   for (RooAbsArg *node : nodes) {
      if (auto prodPdf = dynamic_cast<RooProdPdf *>(node)) {
         auto found = checker.prodPdfsWithConstraints().find(prodPdf);
         if (found != checker.prodPdfsWithConstraints().end()) {
            replaceArg(prodPdf->pdfList()[found->second], *prodPdf);
         }
      }
   }

   // Replace all pdfs that need to be normalized with a pdf wrapper that
   // applies the right normalization.
   for (RooAbsArg *node : nodes) {
      if (auto pdf = dynamic_cast<RooAbsPdf *>(node)) {
         RooArgSet const &currNormSet = *normSets.at(pdf);

         if (currNormSet.empty())
            continue;

         // The call to getVal() sets up cached states for this normalization
         // set, which is important in case this pdf is also used by clients
         // using the getVal() interface (without this, test 28 in stressRooFit
         // is failing for example).
         pdf->getVal(currNormSet);

         if (pdf->selfNormalized() && !dynamic_cast<RooAbsCachedPdf *>(pdf))
            continue;

         auto normalizedPdf = std::make_unique<RooNormalizedPdf>(*pdf, currNormSet);

         replaceArg(*normalizedPdf, *pdf);

         newNodes.emplace_back(std::move(normalizedPdf));
      }
   }

   return newNodes;
}

void foldIntegrals(RooAbsArg const &topNode, RooArgSet &replacedArgs)
{
   RooArgSet nodes;
   topNode.treeNodeServerList(&nodes);

   for (RooAbsArg *normalizedPdf : nodes) {

      if (auto const &replacedArgName = normalizedPdf->getStringAttribute("_replaced_arg")) {

         auto pdf = &replacedArgs[replacedArgName];

         pdf->setAttribute((std::string("ORIGNAME:") + normalizedPdf->GetName()).c_str());

         RooArgList newServerList{*pdf};
         for (auto *client : normalizedPdf->clients()) {
            client->redirectServers(newServerList, false, true);
         }

         pdf->setAttribute((std::string("ORIGNAME:") + normalizedPdf->GetName()).c_str(), false);

         normalizedPdf->setStringAttribute("_replaced_arg", nullptr);
      }
   }
}

} // namespace

/// \class NormalizationIntegralUnfolder
/// \ingroup Roofitcore
///
/// A NormalizationIntegralUnfolder takes the top node of a computation graph
/// and a normalization set for its constructor. The normalization integrals
/// for the PDFs in that graph will be created, and placed into the computation
/// graph itself, rewiring the existing RooAbsArgs. When the unfolder goes out
/// of scope, all changes to the computation graph will be reverted.
///
/// It also performs some other optimizations of the computation graph that are
/// reverted when the object goes out of scope:
///
///   1. Replacing RooProdPdfs that were used to bring constraints into the
///      likelihood with the actual pdf that is not a constraint.
///
/// Note that for evaluation, the original topNode should not be used anymore,
/// because if it is a pdf there is now a new normalized pdf wrapping it,
/// serving as the new top node. This normalized top node can be retreived by
/// NormalizationIntegralUnfolder::arg().

RooFit::NormalizationIntegralUnfolder::NormalizationIntegralUnfolder(RooAbsArg const &topNode, RooArgSet const &normSet)
   : _topNodeWrapper{std::make_unique<RooAddition>("_dummy", "_dummy", RooArgList{topNode})}, _normSetWasEmpty{
                                                                                                 normSet.empty()}
{
   auto ownedArgs = unfoldIntegrals(*_topNodeWrapper, normSet, _normSets, _replacedArgs);
   for (std::unique_ptr<RooAbsArg> &arg : ownedArgs) {
      _topNodeWrapper->addOwnedComponents(std::move(arg));
   }
   _arg = &static_cast<RooAddition &>(*_topNodeWrapper).list()[0];
}

RooFit::NormalizationIntegralUnfolder::~NormalizationIntegralUnfolder()
{
   // If there was no normalization set to compile the computation graph for,
   // we also don't need to fold the integrals back in.
   if (_normSetWasEmpty)
      return;

   foldIntegrals(*_topNodeWrapper, _replacedArgs);

   for (auto &item : _normSets) {
      delete item.second;
   }
}
