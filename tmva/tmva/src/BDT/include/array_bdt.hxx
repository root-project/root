#ifndef __ARRAY_BDT_H_
#define __ARRAY_BDT_H_

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <streambuf>
#include <map>
#include <vector>
#include <array>
#include <utility>
using json = nlohmann::json;

namespace array_bdt {

template <typename T>
class Tree {
private:
public:
   size_t           array_length;
   int              tree_depth;
   void             set_array_length(const size_t &array_length) { this->array_length = array_length; }
   void             set_tree_depth(const int &tree_depth) { this->tree_depth = tree_depth; }
   inline T         inference(const std::vector<T> &);
   inline T         inference_conditional(const std::vector<T> &);
   std::vector<T>   thresholds; // scores if it is a leaf
   std::vector<int> features;
};

template <typename T>
inline T Tree<T>::inference_conditional(const std::vector<T> &event)
{
   size_t index = 0;
   while (index < this->array_length) {
      if (this->features[index] == -1) {
         return this->thresholds[index];
      }
      index = 2 * index + 1 + (int)(event[this->features[index]] > this->thresholds[index]);
   } // end while
}

template <typename T>
inline T Tree<T>::inference(const std::vector<T> &event)
{
   size_t index = 0;
   for (unsigned int iLevel = 0; iLevel < this->tree_depth; ++iLevel) {
      index = 2 * index + 1 + (int)(event[this->features[index]] > this->thresholds[index]);
   }
   return this->thresholds[index];
}

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

   // for (auto value : tree.features) std::cout << value << std::endl;
   // std::cout << "------- \n";
}

// ordering function for vectors of trees
template <typename T>
bool cmp(const Tree<T> &a, const Tree<T> &b)
{
   if (a.features[0] == b.features[0]) {
      return a.thresholds[0] < b.thresholds[0];
   } else {
      return a.features[0] < b.features[0];
   }
}

} // namespace array_bdt
#endif
