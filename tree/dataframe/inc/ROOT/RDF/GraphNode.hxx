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
#include "ROOT/RStringView.hxx"

#include <iostream>

namespace ROOT {
namespace Internal {
namespace RDF {
namespace GraphDrawing {

enum class ENodeType {
   kAction,
   kDefine,
   kFilter,
   kRange,
   kRoot,
   kUsedAction,
};

class GraphCreatorHelper;

// clang-format off
/**
\class ROOT::Internal::RDF::GraphDrawing::GraphNode
\ingroup dataframe
\brief Class used to create the operation graph to be printed in the dot representation

 This represent a single node of the overall graph. Each node maps the real RNode keeping just
 the name and the columns defined up to that point.
*/
// clang-format on
class GraphNode {
   /// Nodes may share the same name (e.g. Filter). To manage this situation in dot, each node
   /// is represented by an unique id.
   unsigned int fID;

   std::string fName, fColor, fShape;

   /// Columns defined up to this node. By checking the defined columns between two consecutive
   /// nodes, it is possible to know if there was some Define in between.
   std::vector<std::string> fDefinedColumns;

   std::shared_ptr<GraphNode> fPrevNode;

   /// When the graph is reconstructed, the first time this node has been explored this flag
   /// is set and it won't be explored anymore.
   bool fIsExplored = false;

   /// A just created node. This means that in no other exploration the node was already created
   /// (this is needed because branches may share some common node).
   bool fIsNew = true;

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetRoot()
   {
      fColor = "#f4b400";
      fShape = "ellipse";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetFilter()
   {
      fColor = "#0f9d58";
      fShape = "hexagon";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetDefine()
   {
      fColor = "#4285f4";
      fShape = "ellipse";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetRange()
   {
      fColor = "#9574b4";
      fShape = "diamond";
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gives a different shape based on the node type
   void SetAction(bool hasRun)
   {
      if (hasRun) {
         fName += "\\n(already run)";
         fColor = "#e6e5e6";
      } else {
         fColor = "#e47c7e";
      }
      fShape = "box";
   }

public:
   ////////////////////////////////////////////////////////////////////////////
   /// \brief Creates a node with a name
   GraphNode(std::string_view name, unsigned int id, ENodeType t) : fID(id), fName(name)
   {
      switch (t) {
      case ENodeType::kAction: SetAction(/*hasRun=*/false); break;
      case ENodeType::kDefine: SetDefine(); break;
      case ENodeType::kFilter: SetFilter(); break;
      case ENodeType::kRange: SetRange(); break;
      case ENodeType::kRoot: SetRoot(); break;
      case ENodeType::kUsedAction: SetAction(/*hasRun=*/true); break;
      };
   }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Appends a node on the head of the current node
   void SetPrevNode(const std::shared_ptr<GraphNode> &node) { fPrevNode = node; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Adds the column defined up to the node
   void AddDefinedColumns(const std::vector<std::string> &columns) { fDefinedColumns = columns; }

   std::string GetColor() const { return fColor; }
   unsigned int GetID() const { return fID; }
   std::string GetName() const { return fName; }
   std::string GetShape() const { return fShape; }
   GraphNode *GetPrevNode() const { return fPrevNode.get(); }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Gets the column defined up to the node
   const std::vector<std::string> &GetDefinedColumns() const { return fDefinedColumns; }

   bool IsExplored() const { return fIsExplored; }
   bool IsNew() const { return fIsNew; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Allows to stop the graph traversal when an explored node is encountered
   void SetExplored() { fIsExplored = true; }

   ////////////////////////////////////////////////////////////////////////////
   /// \brief Mark this node as "not newly created"
   void SetNotNew() { fIsNew = false; }
};

} // namespace GraphDrawing
} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
