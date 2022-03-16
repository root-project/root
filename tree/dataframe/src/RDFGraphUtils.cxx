// Author: Enrico Guiraud, Danilo Piparo, CERN, Massimo Tumolo Politecnico di Torino 08/2018

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RColumnRegister.hxx"
#include "ROOT/RDF/GraphUtils.hxx"

#include <algorithm> // std::find

namespace ROOT {
namespace Internal {
namespace RDF {

std::shared_ptr<GraphDrawing::GraphNode>
GraphDrawing::CreateDefineNode(const std::string &columnName, const ROOT::Detail::RDF::RDefineBase *columnPtr,
                               std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap)
{
   // If there is already a node for this define (recognized by the custom column it is defining) return it. If there is
   // not, return a new one.
   auto duplicateDefineIt = visitedMap.find((void *)columnPtr);
   if (duplicateDefineIt != visitedMap.end())
      return duplicateDefineIt->second;

   auto node = std::make_shared<GraphNode>("Define<BR/>" + columnName, visitedMap.size(), ENodeType::kDefine);
   visitedMap[(void *)columnPtr] = node;
   return node;
}

std::shared_ptr<GraphDrawing::GraphNode>
GraphDrawing::CreateFilterNode(const ROOT::Detail::RDF::RFilterBase *filterPtr,
                               std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap)
{
   // If there is already a node for this filter return it. If there is not, return a new one.
   auto duplicateFilterIt = visitedMap.find((void *)filterPtr);
   if (duplicateFilterIt != visitedMap.end()) {
      duplicateFilterIt->second->SetIsNew(false);
      return duplicateFilterIt->second;
   }

   auto node = std::make_shared<GraphNode>((filterPtr->HasName() ? filterPtr->GetName() : "Filter"), visitedMap.size(),
                                           ENodeType::kFilter);
   visitedMap[(void *)filterPtr] = node;
   return node;
}

std::shared_ptr<GraphDrawing::GraphNode>
GraphDrawing::CreateRangeNode(const ROOT::Detail::RDF::RRangeBase *rangePtr,
                              std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap)
{
   // If there is already a node for this range return it. If there is not, return a new one.
   auto duplicateRangeIt = visitedMap.find((void *)rangePtr);
   if (duplicateRangeIt != visitedMap.end()) {
      duplicateRangeIt->second->SetIsNew(false);
      return duplicateRangeIt->second;
   }

   auto node = std::make_shared<GraphNode>("Range", visitedMap.size(), ENodeType::kRange);
   visitedMap[(void *)rangePtr] = node;
   return node;
}

std::shared_ptr<GraphDrawing::GraphNode>
GraphDrawing::AddDefinesToGraph(std::shared_ptr<GraphNode> node, const RColumnRegister &colRegister,
                                const std::vector<std::string> &prevNodeDefines,
                                std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap)
{
   auto upmostNode = node;
   const auto &defineNames = colRegister.GetNames();
   const auto &defineMap = colRegister.GetColumns();
   for (auto i = int(defineNames.size()) - 1; i >= 0; --i) { // walk backwards through the names of defined columns
      const auto colName = defineNames[i];
      const bool isAlias = defineMap.find(colName) == defineMap.end();
      if (isAlias || IsInternalColumn(colName))
         continue; // aliases appear in the list of defineNames but we don't support them yet
      const bool isANewDefine =
         std::find(prevNodeDefines.begin(), prevNodeDefines.end(), colName) == prevNodeDefines.end();
      if (!isANewDefine)
         break; // we walked back through all new defines, the rest is stuff that was already in the graph

      // create a node for this new Define
      auto defineNode = RDFGraphDrawing::CreateDefineNode(colName, defineMap.at(colName).get(), visitedMap);
      upmostNode->SetPrevNode(defineNode);
      upmostNode = defineNode;
   }

   return upmostNode;
}

namespace GraphDrawing {

std::string GraphCreatorHelper::FromGraphLeafToDot(const GraphNode &start)
{
   // Only the mapping between node id and node label (i.e. name)
   std::stringstream dotStringLabels;
   // Representation of the relationships between nodes
   std::stringstream dotStringGraph;

   // Explore the graph bottom-up and store its dot representation.
   const GraphNode *leaf = &start;
   while (leaf) {
      dotStringLabels << "\t" << leaf->GetID() << " [label=<" << leaf->GetName() << ">, style=\"filled\", fillcolor=\""
                      << leaf->GetColor() << "\", shape=\"" << leaf->GetShape() << "\"];\n";
      if (leaf->GetPrevNode()) {
         dotStringGraph << "\t" << leaf->GetPrevNode()->GetID() << " -> " << leaf->GetID() << ";\n";
      }
      leaf = leaf->GetPrevNode();
   }

   return "digraph {\n" + dotStringLabels.str() + dotStringGraph.str() + "}";
}

std::string GraphCreatorHelper::FromGraphActionsToDot(std::vector<std::shared_ptr<GraphNode>> leaves)
{
   // Only the mapping between node id and node label (i.e. name)
   std::stringstream dotStringLabels;
   // Representation of the relationships between nodes
   std::stringstream dotStringGraph;

   for (auto leafShPtr : leaves) {
      GraphNode *leaf = leafShPtr.get();
      while (leaf && !leaf->IsExplored()) {
         dotStringLabels << "\t" << leaf->GetID() << " [label=<" << leaf->GetName()
                         << ">, style=\"filled\", fillcolor=\"" << leaf->GetColor() << "\", shape=\""
                         << leaf->GetShape() << "\"];\n";
         if (leaf->GetPrevNode()) {
            dotStringGraph << "\t" << leaf->GetPrevNode()->GetID() << " -> " << leaf->GetID() << ";\n";
         }
         // Multiple branches may share the same nodes. It is wrong to explore them more than once.
         leaf->SetExplored();
         leaf = leaf->GetPrevNode();
      }
   }
   return "digraph {\n" + dotStringLabels.str() + dotStringGraph.str() + "}";
}

std::string GraphCreatorHelper::RepresentGraph(ROOT::RDataFrame &rDataFrame)
{
   auto loopManager = rDataFrame.GetLoopManager();
   // Jitting is triggered because nodes must not be empty at the time of the calling in order to draw the graph.
   loopManager->Jit();

   return RepresentGraph(loopManager);
}

std::string GraphCreatorHelper::RepresentGraph(RLoopManager *loopManager)
{
   const auto actions = loopManager->GetAllActions();
   const auto edges = loopManager->GetGraphEdges();

   std::vector<std::shared_ptr<GraphNode>> nodes;
   nodes.reserve(actions.size() + edges.size());

   for (auto *action : actions)
      nodes.emplace_back(action->GetGraph(fVisitedMap));
   for (auto *edge : edges)
      nodes.emplace_back(edge->GetGraph(fVisitedMap));

   return FromGraphActionsToDot(std::move(nodes));
}

} // namespace GraphDrawing
} // namespace RDF
} // namespace Internal
} // namespace ROOT
