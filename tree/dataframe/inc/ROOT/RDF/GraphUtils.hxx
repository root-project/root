// Author: Enrico Guiraud, Danilo Piparo, CERN, Massimo Tumolo Politecnico di Torino 08/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_GRAPHUTILS
#define ROOT_GRAPHUTILS

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDF/RInterface.hxx>
#include <ROOT/RDF/GraphNode.hxx>

namespace ROOT {
namespace Detail {
namespace RDF {
class RDefineBase;
class RFilterBase;
class RRangeBase;
} // namespace RDF
} // namespace Detail

namespace Internal {
namespace RDF {
class RColumnRegister;
namespace GraphDrawing {

std::shared_ptr<GraphNode> CreateDefineNode(const std::string &columnName,
                                            const ROOT::Detail::RDF::RDefineBase *columnPtr,
                                            std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);

std::shared_ptr<GraphNode> CreateFilterNode(const ROOT::Detail::RDF::RFilterBase *filterPtr,
                                            std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);

std::shared_ptr<GraphNode> CreateRangeNode(const ROOT::Detail::RDF::RRangeBase *rangePtr,
                                           std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);

// clang-format off
/**
\class ROOT::Internal::RDF::GraphCreatorHelper
\ingroup dataframe
\brief Helper class that provides the operation graph nodes.

 This class is the single point from which graph nodes can be retrieved. Every time an object is created,
 it clears the static members and starts again.
 By asking this class to create a node, it will return an existing node if already created, otherwise a new one.
*/
// clang-format on
class GraphCreatorHelper {
private:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Map to keep track of visited nodes when constructing the computation graph (SaveGraph)
   std::unordered_map<void *, std::shared_ptr<GraphNode>> fVisitedMap;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting from any leaf (Action, Filter, Range) it draws the dot representation of the branch.
   std::string FromGraphLeafToDot(std::shared_ptr<GraphNode> leaf);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting by an array of leaves, it draws the entire graph.
   std::string FromGraphActionsToDot(std::vector<std::shared_ptr<GraphNode>> leaves);

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting from the root node, prints the entire graph.
   std::string RepresentGraph(ROOT::RDataFrame &rDataFrame);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting from the root node, prints the entire graph.
   std::string RepresentGraph(RLoopManager *rLoopManager);

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting from a Filter or Range, prints the branch it belongs to
   template <typename Proxied, typename DataSource>
   std::string RepresentGraph(RInterface<Proxied, DataSource> &rInterface)
   {
      auto loopManager = rInterface.GetLoopManager();
      loopManager->Jit();

      return FromGraphLeafToDot(rInterface.GetProxiedPtr()->GetGraph(fVisitedMap));
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Starting from an action, prints the branch it belongs to
   template <typename T>
   std::string RepresentGraph(const RResultPtr<T> &resultPtr)
   {
      auto loopManager = resultPtr.fLoopManager;

      loopManager->Jit();

      auto actionPtr = resultPtr.fActionPtr;
      return FromGraphLeafToDot(actionPtr->GetGraph(fVisitedMap));
   }
};

} // namespace GraphDrawing
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
