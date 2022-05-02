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

#include <RooAbsCachedPdf.h>
#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooAddition.h>

#include "RooNormalizedPdf.h"

namespace {

void treeNodeServerListAndNormSets(const RooAbsArg &arg, RooAbsCollection &list, RooArgSet const &normSet,
                                   std::unordered_map<RooAbsArg const *, RooArgSet *> &normSets)
{
   list.add(arg, true);
   normSets.insert({&arg, new RooArgSet{normSet}});

   // Recurse if current node is derived
   if (arg.isDerived() && !arg.isFundamental()) {
      for (const auto server : arg.servers()) {
         RooArgSet serverNormSet;
         arg.fillNormSetForServer(normSet, *server, serverNormSet);
         treeNodeServerListAndNormSets(*server, list, serverNormSet, normSets);
      }
   }
}

std::vector<std::unique_ptr<RooAbsArg>> unfoldIntegrals(RooAbsArg const &topNode, RooArgSet const &normSet,
                                                        std::unordered_map<RooAbsArg const *, RooArgSet *> &normSets)
{
   std::vector<std::unique_ptr<RooAbsArg>> newNodes;

   // No normalization set: we don't need to create any integrals
   if (normSet.empty())
      return newNodes;

   RooArgSet nodes;
   treeNodeServerListAndNormSets(topNode, nodes, normSet, normSets);

   for (RooAbsArg *node : nodes) {

      if (auto pdf = dynamic_cast<RooAbsPdf *>(node)) {

         RooArgSet const &currNormSet = *normSets.at(pdf);

         if (currNormSet.empty())
            continue;

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
}

RooFit::NormalizationIntegralUnfolder::~NormalizationIntegralUnfolder()
{
   foldIntegrals(*_topNodeWrapper);

   for (auto &item : _normSets) {
      delete item.second;
   }
}

/// Returns the top node of the modified computation graph.
RooAbsArg const &RooFit::NormalizationIntegralUnfolder::arg() const
{
   return static_cast<RooAddition &>(*_topNodeWrapper).list()[0];
}
