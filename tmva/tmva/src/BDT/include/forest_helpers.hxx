#ifndef __FOREST_HELPERS_HXX_
#define __FOREST_HELPERS_HXX_

#include "unique_bdt.hxx"
#include "array_bdt.hxx"
#include "bdt_helpers.hxx"
#include "jit_functions.hxx"
#include "jit_code_generators.hxx"

#include <memory>

///// To convert brached to branchless /////
template <typename T>
int get_max_depth(const std::unique_ptr<unique_bdt::Node<T>> &node, int index = 1, int final_index = 0)
{
   if (index > final_index) final_index = index;

   if (node->child_true) {
      final_index = get_max_depth(node->child_true, index + 1, final_index);
   }
   if (node->child_false) {
      final_index = get_max_depth(node->child_false, index + 1, final_index);
   }
   return final_index;
}

template <typename T>
void fill_tree(const int &index, const int &max_index, std::vector<T> &thresholds, std::vector<int> &features,
               const T &threshold_value, const int &feature_value)
{
   if (index < max_index) {
      thresholds.at(index) = threshold_value;
      features.at(index)   = feature_value;
      fill_tree<T>(index * 2 + 1, max_index, thresholds, features, threshold_value, feature_value); // fill true child
      fill_tree<T>(index * 2 + 2, max_index, thresholds, features, threshold_value, feature_value); // fill false child
   }
}

template <typename T>
void recurse_through_tree(const unique_bdt::Node<T> &node, std::vector<T> &thresholds, std::vector<int> &features,
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

template <typename T>
void convert_uniquePtrTree_2_arrayTree(const unique_bdt::Tree<T> &tree_unique, array_bdt::Tree<T> &tree)
{
   // array_bdt::Tree<T> tree;

   int    depth        = get_max_depth(tree_unique.nodes);
   size_t array_length = std::pow(2, depth + 1) - 1; // (2^0+2^1+2^2+...)
   tree.set_array_length(array_length);
   tree.set_tree_depth(depth);
   std::vector<T>   thresholds(array_length);
   std::vector<int> features(array_length);
   recurse_through_tree<T>(*tree_unique.nodes, thresholds, features);
   tree.thresholds.swap(thresholds);
   tree.features.swap(features);
}

template <typename T>
std::vector<array_bdt::Tree<T>> Branched2BranchlessTrees(const std::vector<unique_bdt::Tree<T>> &trees_unique)
{
   std::vector<array_bdt::Tree<T>> trees;
   trees.resize(trees_unique.size());
   for (int i = 0; i < trees.size(); i++) {
      convert_uniquePtrTree_2_arrayTree(trees_unique[i], trees[i]);
   }
   return std::move(trees);
}

///// For JIT /////
template <typename T, typename trees_kind>
void write_generated_code_to_file(const std::vector<trees_kind> &trees, std::string &filename)
{
   std::filebuf fb;
   // std::string  filename;
   // filename = generated_files_path + "generated_forest.h";
   fb.open(filename, std::ios::out);
   std::ostream os(&fb);
   generate_code_forest<T>(os, trees, trees.size());
   fb.close();
}

// template <typename T, typename trees_kind>
std::string generate_namespace_name()
{
   time_t      my_time          = time(0);
   std::string s_namespace_name = std::to_string(std::rand()) + std::to_string(my_time);
   return s_namespace_name;
}

template <typename T, typename trees_kind>
std::function<bool(const T *)> JitTrees(const std::vector<trees_kind> &trees)
{
   // JIT
   std::string       s_namespace_name = generate_namespace_name();
   std::stringstream ss;
   generate_code_forest(ss, trees, trees.size(), s_namespace_name);
   std::string s_trees = ss.str();
   return jit_forest<T>(s_trees, s_namespace_name);
}

#endif
