#ifndef __FOREST_HELPERS_HXX_
#define __FOREST_HELPERS_HXX_

#include "BranchedTree.hxx"
#include "BranchlessTree.hxx"
#include "TreeHelpers.hxx"
#include "CodeGeneratorsJIT.hxx"

#include <memory>

////////////////////////////////////////////////////////////////////////////////
/// Classify scalar score into binary prediction
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] value score to be transformed in probability
/// \param[out] classified input
template <typename T>
inline int classify_binary(T value)
{
   return (value > 0.5);
}

////////////////////////////////////////////////////////////////////////////////
/// Classify vector score into binary prediction
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] value score to be transformed in probability
/// \param[in] pointer to data for the classified input
template <typename T>
void _predict(T *scores, const int num_predictions, std::vector<bool> &predictions)
{
   for (int i = 0; i < num_predictions; i++) {
      predictions[i] = classify_binary(scores[i]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get max depth of the tree
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] node where we are currently in the tree
/// \param[in] depth: current depth
/// \param[in] final_depth: updatable depth
/// \param[out] the maximal depth of the tree
template <typename T>
int get_max_depth(const std::unique_ptr<BranchedTree::Node<T>> &node, const int depth = 1, int final_depth = 0)
{
   if (depth > final_depth) final_depth = depth;

   if (node->child_true) {
      final_depth = get_max_depth(node->child_true, depth + 1, final_depth);
   }
   if (node->child_false) {
      final_depth = get_max_depth(node->child_false, depth + 1, final_depth);
   }
   return final_depth;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill branch of tree until maximal depth (very usefull for sparse trees)
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] index where we are currently in the tree
/// \param[in] max_index: current depth
/// \param[in] thresholds: topological representation of the thresholds
/// \param[in] features: topological representation of the features
/// \param[in] threshold_value: value used for the filling
/// \param[in] feature_value: value used for the filling
template <typename T>
void fill_tree(const int &index, const int &max_index, std::vector<T> &thresholds, std::vector<int> &features,
               const T &threshold_value, const int &feature_value)
{
   if (index < max_index / 2) features.at(index) = feature_value;
   if (index < max_index) {
      thresholds.at(index) = threshold_value;
      fill_tree<T>(index * 2 + 1, max_index, thresholds, features, threshold_value, feature_value); // fill true child
      fill_tree<T>(index * 2 + 2, max_index, thresholds, features, threshold_value, feature_value); // fill false child
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Fill array with topological ordering of the tree
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] node where we are currently in the tree
/// \param[in] thresholds: topological representation of the thresholds
/// \param[in] features: topological representation of the features
/// \param[in] index defining our position in the topological ordering
template <typename T>
void recurse_through_tree(const BranchedTree::Node<T> &node, std::vector<T> &thresholds, std::vector<int> &features,
                          int index = 0)
{
   thresholds.at(index) = node.split_threshold;
   features.at(index)   = node.split_variable;

   int index_true  = index * 2 + 1;
   int index_false = index * 2 + 2;
   if (node.child_true) {
      recurse_through_tree<T>(*node.child_true, thresholds, features, index_true);
   } else {
      fill_tree<T>(index_true, thresholds.size(), thresholds, features, node.leaf_true, 0);
   }
   if (node.child_false) {
      recurse_through_tree<T>(*node.child_false, thresholds, features, index_false);
   } else {
      fill_tree<T>(index_false, thresholds.size(), thresholds, features, node.leaf_false, 0);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] tree_unique: initial representation of the tree (Branched)
/// \param[in] tree: final representation of the tree (branchless)
template <typename T>
void convert_branchedTree_2_branchlessTree(const BranchedTree::Tree<T> &tree_unique, BranchlessTree::Tree<T> &tree)
{
   int    depth        = get_max_depth(tree_unique.nodes);
   size_t array_length = std::pow(2, depth + 1) - 1; // (2^0+2^1+2^2+...)
   tree.set_tree_depth(depth);
   std::vector<T>   thresholds(array_length);
   std::vector<int> features(array_length / 2);
   recurse_through_tree<T>(*tree_unique.nodes, thresholds, features);
   tree.thresholds.swap(thresholds);
   tree.features.swap(features);
}

////////////////////////////////////////////////////////////////////////////////
/// \tparam T type, usually floating point type (float, double, long double)
/// \param[in] trees_unique: branched representation of the trees
/// \param[out] branchless representation of the trees
template <typename T>
std::vector<BranchlessTree::Tree<T>> Branched2BranchlessTrees(const std::vector<BranchedTree::Tree<T>> &trees_unique)
{
   std::vector<BranchlessTree::Tree<T>> trees;
   trees.resize(trees_unique.size());
   for (int i = 0; i < trees.size(); i++) {
      convert_branchedTree_2_branchlessTree(trees_unique[i], trees[i]);
   }
   return std::move(trees);
}

////////////////////////////////////////////////////////////////////////////////
/// \tparam T type, usually floating point type (float, double, long double)
/// \tparam trees_kind type of trees to consider
template <typename T, typename trees_kind>
void write_generated_code_to_file(const std::vector<trees_kind> &trees, const std::string &s_obj_func,
                                  std::string &filename)
{
   std::filebuf fb;
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest<T>(os, trees, s_obj_func);
   fb.close();
}

////////////////////////////////////////////////////////////////////////////////
/// \param[out] string with the namespace to be used for code generation
std::string generate_namespace_name()
{
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(std::rand()) + std::to_string(my_time);
   return s_namespace_name;
}

////////////////////////////////////////////////////////////////////////////////
/// Generate and JIT a forest
///
/// \tparam T type, usually floating point type (float, double, long double)
/// \tparam trees_kind type of trees to consider
/// \param[in] trees: in memory representation of the trees
/// \param[in] s_obj_func name of objective functions
/// \param[out] jitted function
template <typename T, typename trees_kind>
std::function<T(const T *)> JitTrees(const std::vector<trees_kind> &trees, const std::string &s_obj_func)
{
   std::string       s_namespace_name = generate_namespace_name();
   std::stringstream ss;
   generate_code_forest<T>(ss, trees, s_namespace_name, s_obj_func);
   std::string s_trees = ss.str();
   return jit_forest<T>(s_trees, s_namespace_name);
}

#endif
