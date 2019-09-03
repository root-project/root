#ifndef __BRANCHEDTREE_HXX_
#define __BRANCHEDTREE_HXX_

#include <string>
#include <vector>
#include <memory>
#include "json.hpp"

using json = nlohmann::json;

/// Branched Tree classes and helpers
namespace BranchedTree {

/**
 * \class Node
 * Stores data and functions for a single node for the pointer representation of the forest
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class Node {
public:                        // TODO: switch to private
   bool is_leaf_node = 0;      ///< defines if the nodes has a leaf
   T    split_threshold;       ///< theshold for the cut
   int  split_variable;        ///< variable on which the cut is executed
   T    leaf_true, leaf_false; ///< leaf value for this node if it exists

   std::unique_ptr<Node<T>> child_true  = nullptr; ///< pointer to the next node if not a leaf
   std::unique_ptr<Node<T>> child_false = nullptr; ///< pointer to the next node if not a leaf

public:
   /// Perform inference on sigle node
   inline T inference(const T *event);
   // void set_split_variable(int split_variable) { this->split_variable = split_variable; }
   // void set_split_theshold(float split_threshold) { this->split_threshold = split_threshold; }
   // void set_is_leaf_node(bool is_leaf_node) { this->is_leaf_node = is_leaf_node; }
};

/**
 * \class Tree
 * Branched representation of a Tree, using unique_pointers
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class Tree {
public:
   std::unique_ptr<Node<T>> nodes; ///< pointer to the first node of the tree
   /// Perform inference on the whole tree
   inline T inference(const T *event) { return nodes->inference(event); }
};

////////////////////////////////////////////////////////////////////////////////
///// Functions definitions /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \param[in] event pointer to data containing the event
/// \param[out] Tree score, result of the inference
template <typename T>
inline T Node<T>::inference(const T *event)
{
   if (event[split_variable] <= this->split_threshold) {
      if (child_true)
         child_true->inference(event);
      else
         return this->leaf_true;
   } else {
      if (child_false)
         child_false->inference(event);
      else
         return this->leaf_false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reads members of a node of a tree that is in a json format
///
/// \tparam T type for the inference
/// \param[in] jTree json containing the tree
/// \param[in] tmp_node the node which members are gonna be filled by this function
template <typename T>
void write_node_members(json const &jTree, std::unique_ptr<Node<T>> &tmp_node)
{
   tmp_node->split_threshold = jTree.at("split_condition");
   std::string tmp_str       = jTree.at("split").get<std::string>();
   tmp_str.erase(tmp_str.begin(), tmp_str.begin() + 1); // remove initial "f"
   tmp_node->split_variable = std::stoi(tmp_str);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a json tree into a Branched pointer representation of the same tree node by node
///
/// \tparam T type for the inference
/// \param[in] jTree json containing the tree
/// \param[out] Node wrote by the function
template <typename T>
std::unique_ptr<Node<T>> _read_nodes_from_json(json const &jTree)
{
   std::unique_ptr<Node<T>> tmp_node(new Node<T>);
   write_node_members<T>(jTree, tmp_node);
   if (jTree.at("children").at(0).find("leaf") != jTree.at("children").at(0).end()) {
      tmp_node->leaf_true    = jTree.at("children").at(0).at("leaf");
      tmp_node->is_leaf_node = 1;
   } else {
      tmp_node->child_true = _read_nodes_from_json<T>(jTree.at("children").at(0));
   }
   if (jTree.at("children").at(1).find("leaf") != jTree.at("children").at(1).end()) {
      tmp_node->leaf_false   = jTree.at("children").at(1).at("leaf");
      tmp_node->is_leaf_node = 1;
   } else {
      tmp_node->child_false = _read_nodes_from_json<T>(jTree.at("children").at(1));
   }
   return std::move(tmp_node);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a json tree into a Branched pointer representation of the same tree
///
/// \tparam T type for the inference
/// \param[in] jTree json containing the tree
/// \param[in] tree pointer representation of the tree that is gonna be written
template <typename T>
void read_nodes_from_tree(json const &jTree, Tree<T> &tree)
{
   tree.nodes = _read_nodes_from_json<T>(jTree);
}

////////////////////////////////////////////////////////////////////////////////
/// Define a comparison between two trees
///
/// \tparam T type for the inference
/// \param[in] a first compared tree
/// \param[in] b second compared tree
/// \param[out] True if a<b, false if a>=b
///
/// The two trees are compared based on first cut-variable then cut-threshold of the first node of the tree
template <typename T>
bool cmp(const Tree<T> &a, const Tree<T> &b)
{
   if (a.nodes->split_variable == b.nodes->split_variable) {
      return a.nodes->split_threshold < b.nodes->split_threshold;
   } else {
      return a.nodes->split_variable < b.nodes->split_variable;
   }
}

} // namespace BranchedTree

#endif
// end
