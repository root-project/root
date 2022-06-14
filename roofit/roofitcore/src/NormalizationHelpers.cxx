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

#include "RooNormalizedPdf.h"

#include <TClass.h>

namespace {

using RooFit::Detail::DataKey;
using ServerLists = std::map<DataKey, std::vector<DataKey>>;

class GraphChecker {
public:
   GraphChecker(RooAbsArg const &topNode)
   {

      // Get the list of servers for each node by data key.
      {
         RooArgList nodes;
         topNode.treeNodeServerList(&nodes, nullptr, true, true, false, true);
         RooArgSet nodesSet{nodes};
         for (RooAbsArg *node : nodesSet) {
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
      for (auto const& server : serverList) {
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

private:
   std::unordered_set<DataKey> _constraints;
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

std::vector<std::unique_ptr<RooAbsArg>>
unfoldIntegrals(RooAbsArg const &topNode, RooArgSet const &normSet, std::unordered_map<DataKey, RooArgSet *> &normSets)
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
         if (pdf->getAttribute("_integral_unfolded"))
            continue;

         pdf->setAttribute("_integral_unfolded", true);

         RooArgList originalClients;
         for (auto *client : pdf->clients()) {
            originalClients.add(*client);
         }

         auto normalizedPdf = std::make_unique<RooNormalizedPdf>(*pdf, currNormSet);

         normalizedPdf->setAttribute((std::string("ORIGNAME:") + pdf->GetName()).c_str());
         normalizedPdf->setStringAttribute("_normalized_pdf", pdf->GetName());

         RooArgList newServerList{*normalizedPdf};
         for (auto *client : originalClients) {
            if (!nodes.containsInstance(*client))
               continue;
            if (dynamic_cast<RooAbsCachedPdf *>(client))
               continue;
            client->redirectServers(newServerList, false, true);
         }

         normalizedPdf->setAttribute((std::string("ORIGNAME:") + pdf->GetName()).c_str(), false);

         newNodes.emplace_back(std::move(normalizedPdf));
      }
   }

   return newNodes;
}

void foldIntegrals(RooAbsArg const &topNode)
{
   RooArgSet nodes;
   topNode.treeNodeServerList(&nodes);

   for (RooAbsArg *normalizedPdf : nodes) {

      if (normalizedPdf->getStringAttribute("_normalized_pdf")) {

         auto pdf = &nodes[normalizedPdf->getStringAttribute("_normalized_pdf")];

         pdf->setAttribute((std::string("ORIGNAME:") + normalizedPdf->GetName()).c_str());
         pdf->setAttribute("_integral_unfolded", false);

         RooArgList newServerList{*pdf};
         for (auto *client : normalizedPdf->clients()) {
            if (!nodes.containsInstance(*client))
               continue;
            client->redirectServers(newServerList, false, true);
         }
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
/// Note that for evaluation, the original topNode should not be used anymore,
/// because if it is a pdf there is now a new normalized pdf wrapping it,
/// serving as the new top node. This normalized top node can be retreived by
/// NormalizationIntegralUnfolder::arg().

RooFit::NormalizationIntegralUnfolder::NormalizationIntegralUnfolder(RooAbsArg const &topNode, RooArgSet const &normSet)
   : _topNodeWrapper{std::make_unique<RooAddition>("_dummy", "_dummy", RooArgList{topNode})}
{
   auto ownedArgs = unfoldIntegrals(*_topNodeWrapper, normSet, _normSets);
   for (std::unique_ptr<RooAbsArg> &arg : ownedArgs) {
      _topNodeWrapper->addOwnedComponents(std::move(arg));
   }
   _arg = &static_cast<RooAddition &>(*_topNodeWrapper).list()[0];
}

RooFit::NormalizationIntegralUnfolder::~NormalizationIntegralUnfolder()
{
   foldIntegrals(*_topNodeWrapper);

   for (auto &item : _normSets) {
      delete item.second;
   }
}
