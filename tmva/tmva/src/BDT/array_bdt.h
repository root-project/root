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
// Create one for thresholds, one for features number
/*
template <class T>
T * tree_creator(int depth){
  int node_number = std::pow(2, depth);
  T tree[node_number] = {0};
  return tree;
}
*/
/*
int get_node_index(int depth, int previous_row_index, int child_number){
  if (depth==0){return 0;}
  int row_index = (previous_row_index-1)*2+child_number;
  return std::pow(2, depth-1) + row_index;
}
*/
/*
float recursive_inference(std::vector event, int index=0){
  index = index*2
  return inference(event, index);
}
*/

class Tree {
private:
   size_t array_length;

public:
   void               set_array_length(size_t array_length) { this->array_length = array_length; }
   float              inference(const std::vector<float> &);
   std::vector<float> thresholds;
   std::vector<int>   features; // -1 if it is not a node
};

// float inference(std::vector<float> event, float thesholds[],
//              int features[], size_t array_length){
float Tree::inference(const std::vector<float> &event)
{
   size_t index  = 0;
   float  result = 0;
   while (index < this->array_length) {
      if (this->features[index] == -1) {
         result = this->thresholds[index];
         break;
      }
      if (event[this->features[index]] <= this->thresholds[index]) {
         index = index * 2 + 1;
      } else {
         index = index * 2 + 2;
      }
      // index += (event[this->features[index]] <= this->thresholds[index])? 1 : 2;
   }
   return result;
}

// ----- Reading functions -----
std::pair<int, float> get_node_members(json &jTree)
{
   float       threshold = jTree["split_condition"];
   std::string tmp_str   = jTree["split"].get<std::string>();
   tmp_str.erase(tmp_str.begin(), tmp_str.begin() + 1); // remove initial "f"
   int variable = std::stoi(tmp_str);
   return std::make_pair(variable, threshold);
}

/// Need a nlohmann::json object from an xgboost saved format
void _read_nodes(json &jTree, std::vector<int> &tree_features, std::vector<float> &tree_thresholds, int index = 0)
{

   int true_index  = 2 * index + 1;
   int false_index = 2 * index + 2;

   if (jTree.find("children") != jTree.end()) {
      std::pair<int, float> features_thresholds = get_node_members(jTree);
      tree_features.at(index)                   = features_thresholds.first;
      tree_thresholds.at(index)                 = features_thresholds.second;
      _read_nodes(jTree.at("children").at(0), tree_features, tree_thresholds, true_index);
      _read_nodes(jTree.at("children").at(1), tree_features, tree_thresholds, false_index);
   } else {
      if (jTree.find("leaf") != jTree.end()) {
         tree_features.at(index)   = -1;
         tree_thresholds.at(index) = jTree["leaf"];
      } else {
         std::cout << "Error! Unexpected node key\n";
      }
   }
}

/*
void _read_nodes_old(json &jTree, std::vector<int> &tree_features, std::vector<float> &tree_thresholds, int index = 0)
{
   // bool is_leaf_node = ((jTree["children"][0].find("leaf") != jTree["children"][0].end()) &&
   //                    (jTree["children"][0].find("nodeid") != jTree["children"][0].end()));

   std::pair<int, float> features_thresholds = get_node_members(jTree);

   tree_features.at(index)   = features_thresholds.first;
   tree_thresholds.at(index) = features_thresholds.second;

   int true_index  = 2 * index + 1;
   int false_index = 2 * index + 2;

   if (jTree.at("children")[0].find("leaf") != jTree.at("children")[0].end()) {
      tree_features.at(true_index)   = -1;
      tree_thresholds.at(true_index) = jTree["children"][0]["leaf"];
   } else {
      _read_nodes(jTree["children"][0], tree_features, tree_thresholds, true_index);
   }
   if (jTree["children"][1].find("leaf") != jTree["children"][1].end()) {
      tree_features.at(false_index)   = -1;
      tree_thresholds.at(false_index) = jTree["children"][1]["leaf"];
   } else {
      _read_nodes(jTree["children"][1], tree_features, tree_thresholds, false_index);
   }
}
*/

/// Get the depth of a tree
int _get_tree_max_depth(json &jTree, int depth = 0)
{
   int depth_tmp = 0;
   // bool is_leaf_node = ((jTree["children"][0].find("leaf") != jTree["children"][0].end()) &&
   //                      (jTree["children"][0].find("nodeid") != jTree["children"][0].end()));
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

void read_nodes_from_tree(json &jTree, Tree &tree)
{

   int    depth        = _get_tree_max_depth(jTree);
   size_t array_length = std::pow(2, depth + 1) - 1; // (2^0+2^1+2^2+...)
   tree.set_array_length(array_length);
   std::vector<float> Ts(array_length);
   std::vector<int>   Feats(array_length);
   _read_nodes(jTree, Feats, Ts);
   tree.thresholds.swap(Ts);
   tree.features.swap(Feats);
}
} // namespace array_bdt
#endif
