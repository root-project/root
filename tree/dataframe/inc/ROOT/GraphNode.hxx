// Author: Enrico Guiraud, Danilo Piparo, CERN, Massimo Tumolo Politecnico di Torino 08/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_GRAPHNODE
#define ROOT_RDF_GRAPHNODE

#include <string>
#include <memory>
#include <vector>
#include "TString.h"

#include <iostream>

namespace ROOT {
namespace Internal {
namespace RDF {
namespace GraphDrawing {

class GraphCreatorHelper;

// clang-format off
/**
\class ROOT::Internal::RDF::GraphNode
\ingroup dataframe
\brief Class used to create the operation graph to be printed in the dot representation

 This represent a single node of the overall graph. Each node maps the real RNode keeping just
 the name and the columns defined up to that point.
*/
// clang-format on
class GraphNode {
   friend class GraphCreatorHelper;

private:
   unsigned int fCounter; ///< Nodes may share the same name (e.g. Filter). To manage this situation in dot, each node
   ///< is represented by an unique id.
   std::string fName, fColor, fShape;
   std::vector<std::string>
      fDefinedColumns; ///< Columns defined up to this node. By checking the defined columns between two consecutive
                       ///< nodes, it is possible to know if there was some Define in between.
   std::shared_ptr<GraphNode> fPrevNode;

   bool fIsExplored = false; ///< When the graph is reconstructed, the first time this node has been explored this flag
   ///< is set and it won't be explored anymore
   bool fIsNew = true; ///< A just created node. This means that in no other exploration the node was already created
   ///< (this is needed because branches may share some common node).

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Returns a static variable to allow each node to retrieve its counter
   static unsigned int &GetStaticGlobalCounter()
   {
      static unsigned int sGlobalCounter = 1;
      return sGlobalCounter;
   }

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node with a name and a counter
   GraphNode(const std::string_view &name) : fName(name) { fCounter = GetStaticGlobalCounter()++; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Resets the counter.
   /// This is not strictly needed but guarantees that two consecutive request to the graph return the same result.
   static void ClearCounter() { GraphNode::GetStaticGlobalCounter() = 1; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Appends a node on the head of the current node
   void SetPrevNode(const std::shared_ptr<GraphNode> &node) { fPrevNode = node; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Adds the column defined up to the node
   void AddDefinedColumns(const std::vector<std::string> &columns) { fDefinedColumns = columns; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gets the column defined up to the node
   std::vector<std::string> GetDefinedColumns() { return fDefinedColumns; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Manually sets the counter to a node.
   /// It is used by the root node to set its counter to zero.
   void SetCounter(unsigned int counter) { fCounter = counter; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Allows to stop the graph traversal when an explored node is encountered
   void SetIsExplored(bool isExplored) { fIsExplored = isExplored; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief The node is considered just created
   void SetIsNew(bool isNew) { fIsNew = isNew; }

   bool GetIsNew() { return fIsNew; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetRoot()
   {
      fColor = "#e8f8fc";
      fShape = "oval";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetFilter()
   {
      fColor = "#c4cfd4";
      fShape = "diamond";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetDefine()
   {
      fColor = "#60aef3";
      fShape = "oval";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetRange()
   {
      fColor = "#6F4D8F";
      fShape = "diamond";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetAction(bool hasRun)
   {
      if (hasRun) {
         fColor = "#baf1e5";
      } else {
         fColor = "#9cbbe5";
      }
      fShape = "box";
   }
};

} // namespace GraphDrawing
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
