#ifndef __BRANCHLESS_TREE_HXX_
#define __BRANCHLESS_TREE_HXX_

#include "json.hpp"

#include <string>
#include <vector>

using json = nlohmann::json;

/// Branchless Tree classes and helpers
namespace BranchlessTree {

/**
 * \class Tree
 * Branchless representation of a Tree, using topological ordering
 *
 * \tparam T type for the prediction. Usually floating point type (float, double, long double)
 */
template <typename T>
class Tree {
public:
   // size_t           array_length; ///< array_lengths
   int              tree_depth; ///< depth of the tree
   std::vector<T>   thresholds; ///< cut thresholds or scores if corresponding node is a leaf
   std::vector<int> features;   ///< cut variables / features
   void             set_tree_depth(const int &tree_depth) { this->tree_depth = tree_depth; }
   /// Perform inference on sigle node
   inline T inference(const T *event);
};

////////////////////////////////////////////////////////////////////////////////
///// functions definitions /////
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// \param[in] event pointer to data containing the event
/// \param[out] Tree score, result of the inference
template <typename T>
inline T Tree<T>::inference(const T *event)
{
   int index = 0; // should we switch to size_t ?
   for (int iLevel = 0; iLevel < this->tree_depth; ++iLevel) {
      index = 2 * index + 1 + (event[this->features[index]] > this->thresholds[index]);
   }
   return this->thresholds[index];
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
   if (a.features[0] == b.features[0]) {
      return a.thresholds[0] < b.thresholds[0];
   } else {
      return a.features[0] < b.features[0];
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get the maximal array_length, given a depth
///
/// \param[in] depth of the tree
/// \param[out] length of the topological sorted tree
inline size_t get_array_length(const size_t depth)
{
   return std::pow(2, depth + 1) - 1; // (2^0+2^1+2^2+...)
}

////////// reading functions //////////
/* // obsolete reading functions
// ----- Reading functions -----
template <typename T>
std::pair<int, T> get_node_members(json &jTree)
{
   T           threshold = jTree["split_condition"];
   std::string tmp_str   = jTree["split"].get<std::string>();
   tmp_str.erase(tmp_str.begin(), tmp_str.begin() + 1); // remove initial "f"
   int variable = std::stoi(tmp_str);
   return std::make_pair(variable, threshold);
}

template <typename T>
void _read_empty_nodes(std::vector<int> &tree_features, std::vector<T> &tree_thresholds, const int index,
                       const T &value, bool is_repetition = false)
{
   int true_index  = 2 * index + 1;
   int false_index = 2 * index + 2;
   // std::cout << "A: " << false_index;
   if (false_index < tree_features.size()) {
      // if (is_repetition == false) std::cout << "A: " << index << std::endl;
      // tree_features.at(true_index) = -1;
      tree_features.at(true_index)   = 0;
      tree_thresholds.at(true_index) = value;
      // tree_features.at(false_index) = -1;
      tree_features.at(false_index)   = 0; // for tree completion
      tree_thresholds.at(false_index) = value;

      _read_empty_nodes<T>(tree_features, tree_thresholds, true_index, value, true);
      _read_empty_nodes<T>(tree_features, tree_thresholds, false_index, value, true);
   }
}

/// read a model into the class
/// Need a nlohmann::json object from an xgboost saved format
template <typename T>
void _read_nodes(json &jTree, std::vector<int> &tree_features, std::vector<T> &tree_thresholds, int index = 0)
{
   int true_index  = 2 * index + 1;
   int false_index = 2 * index + 2;

   if (jTree.find("children") != jTree.end()) {
      std::pair<int, T> features_thresholds = get_node_members<T>(jTree);
      tree_features.at(index)               = features_thresholds.first;
      tree_thresholds.at(index)             = features_thresholds.second;
      _read_nodes<T>(jTree.at("children").at(0), tree_features, tree_thresholds, true_index);
      _read_nodes<T>(jTree.at("children").at(1), tree_features, tree_thresholds, false_index);
   } else {
      if (jTree.find("leaf") != jTree.end()) {
         // tree_features.at(index) = -1;
         tree_features.at(index)   = 0; // tree completion
         tree_thresholds.at(index) = jTree["leaf"];
         _read_empty_nodes<T>(tree_features, tree_thresholds, index, jTree["leaf"]);
      } else {
         std::cout << "Error! Unexpected node key\n";
      }
   }
}

/// Get the depth of a tree
int _get_tree_max_depth(json &jTree, int depth = 0)
{
   int depth_tmp = 0;
   if (jTree.find("children") != jTree.end()) {
      depth_tmp = jTree["depth"].get<int>() + 1;
      depth     = (depth < depth_tmp) ? depth_tmp : depth;
      depth_tmp = _get_tree_max_depth(jTree["children"][0], depth);
      depth     = (depth < depth_tmp) ? depth_tmp : depth;
      depth_tmp = _get_tree_max_depth(jTree["children"][1], depth);
      depth     = (depth < depth_tmp) ? depth_tmp : depth;
   }
   return depth;
} // end function

/// Sets up a Tree from a json object
template <typename T>
void read_nodes_from_tree(json &jTree, Tree<T> &tree)
{
   int    depth        = _get_tree_max_depth(jTree);
   size_t array_length = std::pow(2, depth + 1) - 1; // (2^0+2^1+2^2+...)
   tree.set_array_length(array_length);
   tree.set_tree_depth(depth);
   std::vector<T>   thresholds(array_length);
   std::vector<int> features(array_length);
   _read_nodes<T>(jTree, features, thresholds);
   tree.thresholds.swap(thresholds);
   tree.features.swap(features);
   // std::cout << "**  " << depth << " || " << array_length << "  **\n";

   // for (auto value : tree.features) std::cout << value << std::endl;
   // std::cout << "------- \n";
}
// */ // end of obsolete reading functions

} // namespace BranchlessTree
#endif
